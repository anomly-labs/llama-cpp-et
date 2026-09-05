// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// b-posit8 code decode shared by the CUDA dequantize / get_rows / exact matmul paths.
// Mirrors ggml_bp8_code_to_ME in ggml-cpu/quants.c bit-for-bit: value = M * 2^E, M = 0 for zero/NaR.
#pragma once

#include <cstdint>

#define BP8_ES    2
#define BP8_QFRAC 96
#define BP8_ZERO  0x00
#define BP8_NAR   0x80

__host__ __device__ static inline void bp8_code_to_ME(uint8_t p, int * M, int * E) {
    if (p == BP8_ZERO || p == BP8_NAR) { *M = 0; *E = 0; return; }
    const int s = (p >> 7) & 1;
    int rest = p & 0x7F;
    if (s) rest = ((~rest) + 1) & 0x7F;
    const int leading = (rest >> 6) & 1;
    int rs = 0;
    while (rs < 7 && ((rest >> (6 - rs)) & 1) == leading) rs++;
    int k_reg, e = 0, fb = 0, fw = 0;
    if (rs == 7) {
        k_reg = leading ? 6 : -7;
    } else {
        k_reg = leading ? (rs - 1) : -rs;
        int rem = 7 - (rs + 1);
        const int r2 = rest & ((1 << rem) - 1);
        const int ew = BP8_ES < rem ? BP8_ES : rem;
        if (ew > 0) { e = (r2 >> (rem - ew)) & ((1 << ew) - 1); e <<= (BP8_ES - ew); }
        rem -= ew; fw = rem; fb = fw > 0 ? (r2 & ((1 << fw) - 1)) : 0;
    }
    const int m = (1 << fw) + fb;
    *M = s ? -m : m;
    *E = 4 * k_reg + e - fw;
}

// exact double value of a code (same as g_bp8_val[c] on the CPU: m * 2^E, exact)
__host__ __device__ static inline double bp8_code_to_double(uint8_t p) {
    int M, E;
    bp8_code_to_ME(p, &M, &E);
    return ldexp((double) M, E);
}
