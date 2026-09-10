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
