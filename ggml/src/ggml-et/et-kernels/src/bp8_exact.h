/* Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
 *
 * bp8_exact.h: integer-only b-posit8 W8A8 exact path for bare-metal RISC-V (rv64imf, -nostdlib):
 * no double, no libm, no libgcc calls. Bit-identical by construction to llama-cpp-et's CPU path
 * (ggml-quants.c / ggml-cpu/quants.c), whose read-out and encoder use IEEE doubles: those double
 * operations are emulated here exactly (round-to-nearest-even at every point the reference rounds).
 *
 *   bp8_scale_exp_exact_i  : per-32-block power-of-two scale from the exact sum of squares
 *   bp8_encode_nearest_i   : nearest lattice code (ties to the lower code), exact comparisons
 *   bp8_q256_add_shifted   : 256-bit two's-complement accumulate of P * 2^shift (floor below radix)
 *   bp8_readout_f32_bits   : the reference read-out (limb fold in double, ldexp -96, to float)
 */
#ifndef BP8_EXACT_H
#define BP8_EXACT_H
#include <stdint.h>
#include "bp8_lattice.inc"

#define BP8_QK      32
#define BP8_QFRAC   96
#define BP8_ZERO    0x00
#define BP8_NAR     0x80
#define BP8_SHIFT_MAX 64    /* bins window (512 B); shifts outside it take the exact direct path */

typedef struct { int8_t scale_exp; uint8_t qs[BP8_QK]; } bp8_block_t;   /* == block_bposit8 */

static inline uint32_t bp8_f32_bits(float f) { union { float f; uint32_t u; } u; u.f = f; return u.u; }
static inline float bp8_bits_f32(uint32_t b) { union { float f; uint32_t u; } u; u.u = b; return u.f; }

/* ---- 256-bit accumulator (8 x 32-bit limbs, little-endian, two's complement) ---- */
static inline void bp8_q256_add_shifted(uint32_t q[8], int64_t P, int shift) {
    if (P == 0) return;
    uint32_t t[8];
    uint32_t sx = (P < 0) ? 0xFFFFFFFFu : 0u;
    uint64_t up = (uint64_t) P;
    t[0] = (uint32_t) up; t[1] = (uint32_t) (up >> 32);
    for (int i = 2; i < 8; i++) t[i] = sx;
    if (shift > 0) {
        int words = shift >> 5, bits = shift & 31;
        if (bits) { uint32_t prev = 0; for (int i = 0; i < 8; i++) { uint32_t cur = t[i]; t[i] = (cur << bits) | prev; prev = (uint32_t) ((uint64_t) cur >> (32 - bits)); } }
        if (words) for (int i = 7; i >= 0; i--) t[i] = (i - words >= 0) ? t[i - words] : 0u;
    } else if (shift < 0) {
        int sh = -shift, words = sh >> 5, bits = sh & 31;
        if (words) for (int i = 0; i < 8; i++) t[i] = (i + words < 8) ? t[i + words] : sx;
        if (bits) { uint32_t next = sx; for (int i = 7; i >= 0; i--) { uint32_t cur = t[i]; t[i] = (cur >> bits) | (next << (32 - bits)); next = cur; } }
    }
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) { uint64_t v = (uint64_t) q[i] + t[i] + carry; q[i] = (uint32_t) v; carry = v >> 32; }
}

static inline int bp8_bitlen64(uint64_t x) { int n = 0; while (x) { n++; x >>= 1; } return n; }

/* Exact emulation of: v = 0.0; for i = 7..0: v = v*2^32 + (double) m[i];  (IEEE double, RNE)
 * State: value = mant * 2^exp with mant < 2^53 (a double), exp >= 0. */
static inline void bp8_dbl_fold_step(uint64_t * mant, int * exp, uint32_t limb) {
    if (*mant == 0) { *mant = limb; *exp = 0; return; }           /* limb < 2^32 is exact */
    int bl = bp8_bitlen64(*mant);
    int C = *exp + 32 + bl - 53;                                   /* cut position (LSB of the kept 53 bits) */
    if (C <= 0) {                                                  /* everything fits exactly */
        *mant = (*mant << 32) | limb; *exp = 0;                     /* bl + 32 <= 53 */
        return;
    }
    uint64_t nm; int round, sticky;
    if (C <= 32) {
        nm = (*mant << (*exp + 32 - C)) | ((uint64_t) limb >> C);
        round  = (C >= 1) ? (int) ((limb >> (C - 1)) & 1u) : 0;
        sticky = (C >= 2) ? ((limb & ((1u << (C - 1)) - 1u)) != 0) : 0;
    } else {                                                       /* limb wholly below the cut */
        nm = *mant << (*exp + 32 - C);                             /* = mant << (53 - bl), exact */
        round = (C - 1 <= 31) ? (int) ((limb >> (C - 1)) & 1u) : 0; /* gap bits are zero */
        /* sticky: any limb bit below the round bit */
        sticky = (C - 1 <= 31) ? ((limb & ((1u << (C - 1)) - 1u)) != 0) : (limb != 0);
    }
    if (round && (sticky || (nm & 1))) nm += 1;
    if (nm >> 53) { nm >>= 1; C += 1; }                            /* carry out: 2^53 -> renormalise */
    *mant = nm; *exp = C;
}

/* (float)(ldexp(v, -96)): v = mant*2^exp (double), RNE to float32 with subnormal handling. */
static inline uint32_t bp8_dbl_to_f32_bits(uint64_t mant, int exp, int neg) {
    if (mant == 0) return neg ? 0x80000000u : 0u;
    int bl = bp8_bitlen64(mant);
    int e2 = bl - 1 + exp - BP8_QFRAC;                            /* unbiased exponent of the top bit */
    uint32_t sign = neg ? 0x80000000u : 0u;
    if (e2 > 127) return sign | 0x7F800000u;                       /* overflow -> inf (never in practice) */
    if (e2 >= -126) {
        int cut = bl - 24;                                         /* keep 24 bits */
        uint64_t m24; int round = 0, sticky = 0;
        if (cut <= 0) m24 = mant << (-cut);
        else { m24 = mant >> cut; round = (int) ((mant >> (cut - 1)) & 1u); sticky = (cut >= 2) ? ((mant & ((1ull << (cut - 1)) - 1ull)) != 0) : 0; }
        if (round && (sticky || (m24 & 1))) m24 += 1;
        if (m24 >> 24) { m24 >>= 1; e2 += 1; if (e2 > 127) return sign | 0x7F800000u; }
        return sign | ((uint32_t) (e2 + 127) << 23) | ((uint32_t) m24 & 0x7FFFFFu);
    }
    /* subnormal: quantum 2^-149; value = mant * 2^(exp-96); shift = exp - 96 + 149 */
    int sh = exp - BP8_QFRAC + 149;
    uint64_t q; int round = 0, sticky = 0;
    if (sh >= 0) q = mant << sh;                                   /* exact; q < 2^23 since e2 < -126 */
    else { int k = -sh; if (k > 63) { q = 0; round = 0; sticky = 1; } else { q = mant >> k; round = (int) ((mant >> (k - 1)) & 1u); sticky = (k >= 2) ? ((mant & ((1ull << (k - 1)) - 1ull)) != 0) : 0; } }
    if (round && (sticky || (q & 1))) q += 1;                     /* may become 2^23 = smallest normal: bits work out */
    return sign | (uint32_t) q;
}

static inline uint32_t bp8_readout_f32_bits(const uint32_t q[8]) {
    uint32_t m[8]; for (int i = 0; i < 8; i++) m[i] = q[i];
    int neg = (m[7] >> 31) & 1;
    if (neg) { uint64_t c = 1; for (int i = 0; i < 8; i++) { uint64_t v = (uint64_t) (~m[i]) + c; m[i] = (uint32_t) v; c = v >> 32; } }
    uint64_t mant = 0; int exp = 0;
    for (int i = 7; i >= 0; i--) bp8_dbl_fold_step(&mant, &exp, m[i]);
    return bp8_dbl_to_f32_bits(mant, exp, neg);
}

/* ---- exact block scale: se = round_half_even(log2(sqrt(S/32))), S = exact sum of f32 squares ---- */
#define BP8_SS_LIMBS 20
#define BP8_SS_RADIX 352
static inline int bp8_scale_exp_exact_i(const float * x, int * any_nonzero) {
    uint32_t acc[BP8_SS_LIMBS]; for (int i = 0; i < BP8_SS_LIMBS; i++) acc[i] = 0;
    int any = 0;
    for (int j = 0; j < BP8_QK; j++) {
        uint32_t b = bp8_f32_bits(x[j]);
        uint32_t ex = (b >> 23) & 0xFFu, fr = b & 0x7FFFFFu;
        if (ex == 0xFFu) { *any_nonzero = 1; return 0; }          /* non-finite -> se 0 */
        if (ex == 0 && fr == 0) continue;
        any = 1;
        uint64_t m; int e;                                         /* |x| = m * 2^e exactly */
        if (ex == 0) { m = fr; e = -149; } else { m = fr | 0x800000u; e = (int) ex - 150; }
        /* square = m*m * 2^(2e); normalise so the top bit index matches the double-based reference:
           the reference places the 53-bit double mantissa mi at pos = e2 - 52 + RADIX where p = mi*2^(e2-52).
           Here p = (m*m) * 2^(2e): same number, so place m*m at pos = 2e + RADIX. */
        uint64_t sq = m * m;                                       /* <= 48 bits */
        int pos = 2 * e + BP8_SS_RADIX;                            /* >= 2*(-149)+352 = 54 > 0 */
        int w = pos >> 5, bb = pos & 31;
        uint64_t lo = bb ? (sq << bb) : sq;
        uint64_t hi = bb ? (sq >> (64 - bb)) : 0ull;
        uint32_t parts[3] = { (uint32_t) lo, (uint32_t) (lo >> 32), (uint32_t) hi };
        uint64_t c = 0;
        for (int i = 0; i < 3; i++) { uint64_t t = (uint64_t) acc[w + i] + parts[i] + c; acc[w + i] = (uint32_t) t; c = t >> 32; }
        for (int i = w + 3; c && i < BP8_SS_LIMBS; i++) { uint64_t t = (uint64_t) acc[i] + c; acc[i] = (uint32_t) t; c = t >> 32; }
    }
    *any_nonzero = any;
    if (!any) return 0;
    int top = -1, pop = 0;
    for (int i = BP8_SS_LIMBS - 1; i >= 0; i--) {
        if (acc[i]) {
            if (top < 0) { int t = 31; while (!((acc[i] >> t) & 1u)) t--; top = 32 * i + t; }
            uint32_t mm = acc[i]; while (mm) { pop += (int) (mm & 1u); mm >>= 1; }
        }
    }
    int E = top - BP8_SS_RADIX - 5;
    int se;
    if ((E & 1) == 0) se = E / 2;                                  /* E even (C division truncates toward 0; E even -> exact) */
    else { int n = (E - 1) / 2; if ((E - 1) < 0 && ((E - 1) % 2) != 0) n = (E - 1 - 1) / 2; /* floor for negatives */
           se = (pop == 1) ? (((n & 1) == 0) ? n : n + 1) : n + 1; }
    if (se > 127) se = 127;
    if (se < -128) se = -128;
    return se;
}

/* exact dyadic compare: sign(a*2^p - b*2^q) for |a|,|b| < 2^40 */
static inline int bp8_cmp_dyadic(int64_t a, int p, int64_t b, int q) {
    if (a == 0 && b == 0) return 0;
    if (a == 0) return b > 0 ? -1 : 1;
    if (b == 0) return a > 0 ? 1 : -1;
    if ((a > 0) != (b > 0)) return a > 0 ? 1 : -1;
    /* same sign: compare magnitudes */
    uint64_t ua = a > 0 ? (uint64_t) a : (uint64_t) (-a), ub = b > 0 ? (uint64_t) b : (uint64_t) (-b);
    int r;
    if (p >= q) { int d = p - q; if (d >= 24) r = 1; else { uint64_t as = ua << d; r = (as > ub) - (as < ub); } }
    else        { int d = q - p; if (d >= 24) r = -1; else { uint64_t bs = ub << d; r = (ua > bs) - (ua < bs); } }
    return a > 0 ? r : -r;
}

/* nearest lattice code to x * 2^-se (exact), ties to the lower code number; 0 for x == 0. */
static inline uint8_t bp8_encode_nearest_i(float x, int se) {
    uint32_t b = bp8_f32_bits(x);
    uint32_t ex = (b >> 23) & 0xFFu, fr = b & 0x7FFFFFu;
    if (ex == 0 && fr == 0) return BP8_ZERO;
    int64_t m; int e;
    if (ex == 0) { m = fr; e = -149; } else { m = (int64_t) (fr | 0x800000u); e = (int) ex - 150; }
    if (b >> 31) m = -m;
    e -= se;                                                       /* x' = m * 2^e */
    /* binary search the sorted lattice for the first value >= x' */
    int lo = 0, hi = BP8_LAT_N;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (bp8_cmp_dyadic(bp8_lat_M[mid], bp8_lat_E[mid], m, e) < 0) lo = mid + 1; else hi = mid; }
    /* candidates: lo-1 and lo; pick the nearer by exact distance; tie -> lower code */
    int best = -1; int64_t bd = 0; int bde = 0;
    for (int k = (lo > 0 ? lo - 1 : lo); k <= (lo < BP8_LAT_N ? lo : BP8_LAT_N - 1); k++) {
        if (k < 0 || k >= BP8_LAT_N) continue;
        /* distance |v_k - x'|: both dyadic; compute at the lower exponent */
        int64_t vm = bp8_lat_M[k]; int ve = bp8_lat_E[k];
        int64_t dm; int de;
        if (vm == 0) { dm = (m < 0 ? -m : m); de = e; }                    /* |0 - x'| = |x'| */
        else if (ve >= e) { int d = ve - e;
            if (d > 40) { dm = (vm < 0 ? -vm : vm); de = ve; }            /* |x'| < 2^-40 |v|: ordering unaffected */
            else { dm = (vm << d) - m; de = e; } }
        else { int d = e - ve;
            if (d > 32) { dm = (m < 0 ? -m : m); de = e; }                /* |v| < 2^-32 |x'| */
            else { dm = vm - (m << d); de = ve; } }
        if (dm < 0) dm = -dm;
        if (best < 0) { best = k; bd = dm; bde = de; continue; }
        int c = bp8_cmp_dyadic(dm, de, bd, bde);
        if (c < 0 || (c == 0 && bp8_lat_code[k] < bp8_lat_code[best])) { best = k; bd = dm; bde = de; }
    }
    return bp8_lat_code[best];
}

static inline void bp8_quantize_row_i(const float * x, bp8_block_t * y, int k) {
    for (int i = 0; i < k / BP8_QK; i++) {
        int nz = 0;
        int se = bp8_scale_exp_exact_i(x + i * BP8_QK, &nz);
        if (!nz) { y[i].scale_exp = 0; for (int j = 0; j < BP8_QK; j++) y[i].qs[j] = BP8_ZERO; continue; }
        y[i].scale_exp = (int8_t) se;
        for (int j = 0; j < BP8_QK; j++) y[i].qs[j] = bp8_encode_nearest_i(x[i * BP8_QK + j], se);
    }
}

/* exact W8A8 dot of nb blocks -> float32 bits (binned accumulation, same as the CPU kernel) */
static inline uint32_t bp8_dot_f32_bits(const bp8_block_t * xb, const bp8_block_t * yb, int nb) {
    uint32_t quire[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    int64_t bins[BP8_SHIFT_MAX]; for (int i = 0; i < BP8_SHIFT_MAX; i++) bins[i] = 0;
    for (int ib = 0; ib < nb; ib++) {
        int se = (int) xb[ib].scale_exp + (int) yb[ib].scale_exp + BP8_QFRAC;
        for (int j = 0; j < BP8_QK; j++) {
            int64_t P = (int64_t) bp8_lut_M[xb[ib].qs[j]] * (int64_t) bp8_lut_M[yb[ib].qs[j]];
            if (P == 0) continue;
            int sh = (int) bp8_lut_E[xb[ib].qs[j]] + (int) bp8_lut_E[yb[ib].qs[j]] + se;
            if (sh >= 0 && sh < BP8_SHIFT_MAX) bins[sh] += P; else bp8_q256_add_shifted(quire, P, sh);
        }
    }
    for (int i = 0; i < BP8_SHIFT_MAX; i++) if (bins[i] != 0) bp8_q256_add_shifted(quire, bins[i], i);
    return bp8_readout_f32_bits(quire);
}
#endif
