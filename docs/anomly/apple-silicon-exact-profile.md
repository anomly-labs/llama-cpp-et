<!-- Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. -->
# Apple Silicon (CPU build): the exact profile is bit-identical to x86

Measured 2026-09-10 on a MacBook Pro with an M4 Pro (arm64, NEON path of the exact b-posit8
kernel), macOS 26.6, Apple clang, against an x86-64 host (AVX-512 path), both at `et-bposit8`
@ edcefa116.

Build on macOS (the exact profile forces Metal, Accelerate, BLAS and llamafile off; stating it
explicitly is harmless):

    cmake -B build-exact -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF \
          -DGGML_LLAMAFILE=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-exact --target llama-cli -j

Gate (same command on both hosts; see `deterministic-graph.md`):

    INVAR_LOGITS_OUT=dump.jsonl INVAR_LOGITS_LAYERS=1 INVAR_LOGITS_MATMULS=1 \
    ./build-exact/bin/llama-cli -m SmolLM2-135M-Instruct-bposit8.gguf \
      -p "Write a Python function that returns the n-th Fibonacci number, with a docstring." \
      -n 8 --temp 0 --seed 1 -st --simple-io -t 4 -fa off

Result: the 5,140-line dumps (token ids, embeddings, every layer-1 matmul output, logits) are
**byte-identical** across the two machines (`cmp` clean; sha256 of both dumps
`b5303ed6bb0e9e82…`), and the generated text is identical.

What this does and does not say:

- It says the exact profile removes reduction order as an input on the CPU path across ISAs
  (AVX-512 vs NEON), the same result already shown for x86 vs an RTX 5090 and an Ultra96
  Cortex-A53.
- It does not give Apple GPU acceleration: there is **no Metal exact profile**; this is the CPU
  build, and on the M4 Pro at 4 threads it ran at 37.1 prompt / 11.7 generation tokens per second
  for this model, against 67.0 / 25.2 on the x86 host. A Metal port of the exact dots (the
  `ggml-det.h` header instantiated with Metal's IEEE operations, as `det.cuh` does for CUDA) is
  the open item.

## Metal exact profile (2026-09-10)

There is now a Metal path that stays inside the exact profile and runs the **whole graph on the
GPU**. Build with

    cmake -B build-metal -DGGML_METAL=ON -DANOMLY_METAL_EXACT=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-metal --target llama-cli -j

What it does. Metal has no `double` and Apple GPUs flush single-precision subnormals to zero
(measured on the M4 Pro even with `MTLMathModeSafe` and `#pragma METAL fp contract(off)`), so the
exact-profile kernels use no floating-point hardware at all for anything that touches a result:

- **b-posit8 W8A8 matmuls** (`kernel_quantize_bposit8_f32_sg`, `kernel_mul_mv_bposit8_exact`):
  the reference activation quantiser (exact block scale from the exact sum of squares, nearest-code
  encode on a software IEEE binary64) and the 256-bit exact accumulation with the single readout,
  on 64-bit integers. The quantised activation is reused across the adjacent Q/K/V (and gate/up)
  projections.
- **f16 attention matmuls** (`-fa off`; `kernel_cvt_f32_f16_det`, `kernel_mul_mv_f16_exact`):
  f32→f16 in integers (IEEE round-to-nearest-even, the same result as the CPU conversion), then
  the exact f16 dot through the same quire and readout as the CPU / CUDA kernels.
- **RMSNorm, softmax (f16/f32 mask, sinks), RoPE (norm/neox, no YaRN), SwiGLU, SiLU, GELU**: the
  reference's routines on the software binary64 (`ggml/src/ggml-det/det_soft_ops.h`, generated),
  and every float multiply/add/divide/sqrt/max in those kernels on a software binary32 layer
  (`dso_f*_bits`) so subnormals survive. `get_rows` of quantised tensors stays on the CPU.
- The shader library is compiled with fast-math off. `ANOMLY_METAL_EXACT` is the only way Metal
  is allowed on in this fork; ops without an exact kernel are declined by `supports_op`.

Gates, all on the M4 Pro:

- `tests/anomly/metal_ops_gate.mm`: the software float ops **inside a Metal kernel** against the
  same header on the host — mul/add/sub/div/sqrt/max, expf, tanhf, silu, f16→f32, and the
  integer quire readout against the reference Horner: 2 × 10⁶ + 5 × 10⁵ cases, 0 mismatches.
- `tests/anomly/test-metal-bposit8.cpp ops 200`: eleven ops on the Metal backend against the CPU
  backend, random shapes, a subnormal-heavy input stream: rms_norm (+mul), soft_max (f16 mask,
  plain), rope (norm, neox), swiglu, silu, gelu, mul_mat f16 (contiguous, permuted): **all 0
  mismatches** (636,384 elements per elementwise op); the b-posit8 matmul gate 0 / 8,288.
- Whole-graph dumps (`INVAR_LOGITS_OUT`, layer-1 matmuls, logits, tokens) with `-ngl 99` against
  the x86 dumps, compared tensor-by-tensor and as `LC_ALL=C sort | sha256sum` (the Metal graph
  optimizer schedules Qcur_rope/Kcur_rope after Vcur, so the raw line order differs; nothing
  else does):

  | model | lines | sorted sha256 (x86 == M4 GPU) | text |
  |---|---|---|---|
  | SmolLM2-135M-Instruct b-posit8 | 5,140 | 68edecdc52a5e19e… | identical |
  | Qwen2.5-0.5B-Instruct b-posit8 | 4,840 | ea35e8c5422b36cc… | identical |
  | Llama-3.2-1B-Instruct b-posit8 | 2,760 | e5e923a1cff55347… | identical |
  | Mistral-7B-Instruct-v0.3 b-posit8 | 5,480 | b244655a42e81053… | identical |

Two things the port found that matter beyond this fork. (1) A heavy thread-0-only block followed
by a threadgroup barrier and a broadcast miscompiled on the M4 Pro (threads 1..31 never executed
the kernel tail); the softmax kernel now keeps its tail uniform. (2) The subnormal flush above: a
"safe" math mode is not IEEE on Apple GPUs, so any bit-exactness claim for a Metal kernel that
uses float hardware needs a subnormal test.

Speed (`llama-bench -p 256 -n 64 -fa 0 -t 8`, M4 Pro 14-core, tokens/s, GPU = `-ngl 99`, CPU = the
same build with `-ngl 0`; the per-lane limbs of the exact matvec in threadgroup memory, 966ddd4bd):

| model | GPU pp256 | GPU tg64 | CPU pp256 | CPU tg64 | GPU/CPU decode |
|---|---|---|---|---|---|
| SmolLM2-135M | 219.1 | 52.9 | 108.9 | 55.2 | 0.96× |
| Qwen2.5-0.5B | 110.5 | 34.4 | 59.9 | 18.8 | 1.8× |
| Llama-3.2-1B | 54.6 | 25.3 | 27.7 | 8.8 | 2.9× |
| Mistral-7B (`-p 64 -n 32`) | 8.5 | 5.9 | 4.0 | 1.7 | 3.5× |

Prompt processing is compute-bound in the exact matvec (no floating-point unit is used; the
integer work per multiply-accumulate is what limits it) and runs at about 2× the 8-thread CPU;
single-token generation on the GPU scales from parity on the 135M model (about 750 dependent
dispatches per token) to 3.5× the CPU on the 7B. The 7B runs in the 24 GB of unified memory at
5.9 tokens/s with every activation, norm, softmax and attention product bit-identical to the x86
build. Nothing uses the simdgroup matrix units (they are float) and the kernels are one simdgroup
per output; there is headroom.

For scale, the ordinary Metal path of the same source tree (`-DANOMLY_ALLOW_INEXACT_BACKENDS=ON`,
Q8_0 weights, float accumulation, simdgroup matrices; not bit-reproducible across machines) on the
same M4 Pro: SmolLM2-135M pp256 15,936 / tg64 381; Mistral-7B pp64 423 / tg32 31.2. The exact
profile is currently 5.3× slower than that in 7B decode and about 50× slower in prefill. That is
the price of exact, order-independent arithmetic with no floating-point unit in the loop, today;
it is not a target this fork is trying to match.
