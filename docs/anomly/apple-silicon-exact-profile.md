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

## Metal exact profile (2026-09-10, later the same day)

There is now a Metal path that stays inside the exact profile. Build with

    cmake -B build-metal -DGGML_METAL=ON -DANOMLY_METAL_EXACT=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-metal --target llama-cli -j

What it does: the b-posit8 W8A8 matmuls run on the GPU through `ggml-metal-bposit8.h` — the
reference activation quantiser (exact block scale, nearest-code encode) and the 256-bit exact
accumulation with the single readout, written on 64-bit integers plus a software IEEE binary64
(`ggml/src/ggml-det/det_soft.h`), because Metal has no `double`. Every other op the exact profile
alters (RMSNorm, softmax, RoPE, SiLU/SwiGLU, the f16 attention matmuls, get_rows of quantised
tensors) is declined by the backend and runs on the exact CPU path; the shader library is compiled
with fast-math off. `ANOMLY_METAL_EXACT` is the only way Metal is allowed on in this fork.

Gates, all on the M4 Pro:

- `tests/anomly/metal-bp8-unit.mm`: the Metal arithmetic against the same header compiled on
  the host — f32→binary64 (65,536), block scale (2,048 blocks), nearest encode (65,536), whole
  block quantiser (2,048), 256-bit readout (2,048): 0 mismatches.
- `tests/anomly/test-metal-bposit8.cpp`: MUL_MAT(bposit8, f32) on the Metal backend against the
  CPU backend, random shapes K ≤ 768, N ≤ 200, M ≤ 9: **0 / 161,364 mismatches** over 300 trials.
- The whole-graph gate above with `-ngl 99`: the 5,140-line dump is **byte-identical** to the x86
  dump (sha256 `b5303ed6bb0e9e82…`), same text.

The host-side software double is itself gated against hardware double on 2 × 10⁸ cases and the
integer/soft-double b-posit8 path against `libggml-cpu` on 3 × 10⁵ rows (both on Linux; sources in
the Anomly `space-time` repo, `research/metal-exact/`).

Speed, SmolLM2-135M (`llama-bench -p 256 -n 64 -t 8`): Metal build pp256 **90.6** t/s, tg64
**16.7** t/s; CPU exact build pp256 74.6, tg64 54.7. Prompt processing gains from the GPU matmuls;
single-token generation is 3.3× slower than the CPU build because every declined op is a
CPU↔GPU hand-off. Moving RMSNorm, softmax, RoPE and the
activations onto the GPU with the same software-double discipline is the next step; the f16
attention matmuls would follow the same accumulate-and-read-out pattern.
