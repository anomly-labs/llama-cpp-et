<!-- Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. -->
# b-posit8 exact-quire vec_dot — provenance harness

Proves the `ggml_vec_dot_bposit8_bposit8` kernel (in `ggml/src/ggml-cpu/quants.c`)
computes the W8A8 dot in a bit-exact 256-bit / 96-frac Kulisch quire.

- `gen_bp8_dot_golden.py` — regenerates `bp8_dot_golden.h` from open-bposit's
  rational (`fractions.Fraction`) reference: an integer (M,E) LUT for all 256
  codes + single-block, multi-block-streaming, and cancellation golden cases.
- `bp8_dot_verify.c` — standalone C mirror of the kernel; checks it bit-for-bit
  against the golden data.  Build: `cc -O2 bp8_dot_verify.c -lm && ./a.out`.
  Result: single-block 64/64 bit-exact, streaming 3/3 exact, cancellation
  gives exact 0 where fp32 drifts.
- `../test-bposit8-quire.c` — links the built ggml libs and exercises the
  REGISTERED type-traits dispatch (from_float + vec_dot), cross-checked vs the
  dequantized double dot (8/8 trials, rel 0).
