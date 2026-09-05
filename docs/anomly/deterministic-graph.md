<!-- Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. -->
# ggml-det: whole-graph deterministic inference (CPU == CUDA, bit for bit)

`ggml/src/ggml-det/ggml-det.h` is one header included by the CPU backend (`ggml-det/det.c`,
compiled with `-ffp-contract=off`) and by the CUDA backend (`ggml-cuda/det.cuh`, `*_rn`
intrinsics, library built with `-fmad=false` and no fast-math). Everything in it is a fixed
sequence of IEEE round-to-nearest operations, no libm:

| Op | CPU site | CUDA site | Method |
|---|---|---|---|
| RMSNorm | ops.cpp `rms_norm_f32` | norm.cu `rms_norm_f32` | exact 640-bit sum of squares → correctly rounded double → `1/sqrtf(mean+eps)` |
| f16 matmul (KQ, KQV) | vec.cpp `ggml_vec_dot_f16` → det.c `ggml_det_dot_f16` (65,536-entry decode tables, 16-lane chunks two per step, 59 bins, zero guard — OpenEvolve round 1, 1.66x) | bposit8.cu `mul_mat_f16_exact` | src1 rounded to f16, exact products in the 256-bit quire, shared readout |
| softmax | ops.cpp `soft_max_f32` | softmax.cu `soft_max_f32` | `det_expf`, exact integer sum, `y * (1/(float)sum)` |
| SiLU / SwiGLU | vec.cpp | unary.cuh `silu_single` | `x * (1/(1+det_expf(-x)))` |
| RoPE (no YaRN/ff) | ops.cpp `rope_cache_init` | rope.cu `rope_norm/neox` | freq = exp2(-2i/d · log2(base)), double Cody-Waite reduction, Taylor sin/cos |
| b-posit8 matmul | quants.c | bposit8.cu | exact quire (both since 2026-09-05) |

Flash attention is reported unsupported by the CUDA backend and must be off on the CPU
(`-fa off`). `GGML_LLAMAFILE=OFF` so the CPU uses `ggml_vec_dot_f16`.

Gate: `INVAR_LOGITS_OUT=... INVAR_LOGITS_LAYERS=1 INVAR_LOGITS_MATMULS=1 llama-cli ... -fa off`
on the CPU (`CUDA_VISIBLE_DEVICES=`) and on the GPU (`-ngl 99`), then
`tests/csc/dump_diff.py a.jsonl b.jsonl` — expected: byte-identical files. Measured on
SmolLM2-135M b-posit8: 33,792 dump lines identical across the RTX 5090 and the CPU at
1/4/16 threads, identical text; every matmul unit re-executed by `invar-spotcheck`.

Speed work on the exact f16 dot: OpenEvolve workspace `space-time/openevolve/workspaces/
f16_exact_dot_speed` (gate: bit-exact vs `det_dot_f16` on 420 rows per seed incl. inf/nan/
subnormal/cancellation and under permutation; score = median speedup at n = 64/256/2048).
