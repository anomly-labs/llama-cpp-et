// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// bp8_exact_soft.h — the b-posit8 W8A8 exact path (block scale, nearest encode, 256-bit quire
// accumulate, single readout) written on integers and det_soft.h's software double, so it
// compiles unchanged as C and as Metal Shading Language. Contract: bit-identical to
// ggml-cpu/quants.c and ggml-cuda/bposit8.cu. Gate: test_bp8_soft.c against libggml-cpu.
#ifndef BP8_EXACT_SOFT_H
#define BP8_EXACT_SOFT_H
#include "det_soft.h"

#define BP8_ES    2
#define BP8_QFRAC 96
#define BP8_ZERO  0x00
#define BP8_NAR   0x80
#define BP8_QK    32
#define BP8_SS_LIMBS 20
#define BP8_SS_RADIX 352

#ifdef __METAL_VERSION__
#define BP8_PTR(T) thread T *
#define BP8_CPTR(T) const thread T *
#define BP8_TAB(T) constant T *
#define BP8_POPC32(x) ((int)metal::popcount((metal::uint)(x)))
#define BP8_CLZ32(x)  ((int)metal::clz((metal::uint)(x)))
#else
#define BP8_PTR(T) T *
#define BP8_CPTR(T) const T *
#define BP8_TAB(T) const T *
#define BP8_POPC32(x) __builtin_popcount((unsigned)(x))
#define BP8_CLZ32(x)  __builtin_clz((unsigned)(x))
#endif

// code -> M * 2^E (M = 0 for zero and NaR); mirrors ggml_bp8_code_to_ME
DS_FN void bp8_code_to_ME(ds_u32 p, BP8_PTR(ds_i32) M, BP8_PTR(ds_i32) E) {
    if (p == BP8_ZERO || p == BP8_NAR) { *M = 0; *E = 0; return; }
    const ds_i32 s = (p >> 7) & 1;
    ds_i32 rest = p & 0x7F;
    if (s) rest = ((~rest) + 1) & 0x7F;
    const ds_i32 leading = (rest >> 6) & 1;
    ds_i32 rs = 0;
    while (rs < 7 && ((rest >> (6 - rs)) & 1) == leading) rs++;
    ds_i32 k_reg, e = 0, fb = 0, fw = 0;
    if (rs == 7) {
        k_reg = leading ? 6 : -7;
    } else {
        k_reg = leading ? (rs - 1) : -rs;
        ds_i32 rem = 7 - (rs + 1);
        const ds_i32 r2 = rest & ((1 << rem) - 1);
        const ds_i32 ew = BP8_ES < rem ? BP8_ES : rem;
        if (ew > 0) { e = (r2 >> (rem - ew)) & ((1 << ew) - 1); e <<= (BP8_ES - ew); }
        rem -= ew; fw = rem; fb = fw > 0 ? (r2 & ((1 << fw) - 1)) : 0;
    }
    const ds_i32 m = (1 << fw) + fb;
    *M = s ? -m : m;
    *E = 4 * k_reg + e - fw;
}

// exact double bits of a code's value (M * 2^E)
DS_FN ds_u64 bp8_code_to_dbits(ds_u32 p) {
    ds_i32 M, E; bp8_code_to_ME(p, &M, &E);
    if (M == 0) return 0;
    return ds_ldexp(ds_from_i64_exact((ds_i64) M), E);
}

// exact block scale from float32 bit patterns; mirrors ggml_bp8_scale_exp_exact / CUDA
DS_FN ds_i32 bp8_scale_exp_exact_bits(BP8_CPTR(ds_u32) xb) {
    ds_u32 acc[BP8_SS_LIMBS];
    for (int i = 0; i < BP8_SS_LIMBS; i++) acc[i] = 0;
    int any = 0;
    for (int j = 0; j < BP8_QK; j++) {
        const ds_u32 f = xb[j];
        if ((f & 0x7FFFFFFFu) == 0) continue;
        const ds_i32 e = (f >> 23) & 0xFF; const ds_u64 mf = f & 0x7FFFFF;
        if (e == 0xFF) return 0;                                  // non-finite -> 0
        any = 1;
        ds_u64 mant; ds_i32 e2;
        if (e == 0) { mant = mf; e2 = -149; } else { mant = mf | 0x800000ull; e2 = e - 127 - 23; }
        const ds_u64 P = mant * mant;                             // <= 48 bits, exact
        const ds_i32 pos = 2 * e2 + BP8_SS_RADIX;                 // value = P * 2^(pos - RADIX)
        const ds_i32 w = pos >> 5, b = pos & 31;
        const ds_u64 lo = b ? (P << b) : P;
        const ds_u64 hi = b ? (P >> (64 - b)) : 0ull;
        ds_u32 parts[3]; parts[0] = (ds_u32) lo; parts[1] = (ds_u32) (lo >> 32); parts[2] = (ds_u32) hi;
        ds_u64 c = 0;
        for (int i = 0; i < 3; i++) { const ds_u64 t = (ds_u64) acc[w + i] + parts[i] + c; acc[w + i] = (ds_u32) t; c = t >> 32; }
        for (int i = w + 3; c && i < BP8_SS_LIMBS; i++) { const ds_u64 t = (ds_u64) acc[i] + c; acc[i] = (ds_u32) t; c = t >> 32; }
    }
    if (!any) return 0;
    ds_i32 top = -1, pop = 0;
    for (int i = BP8_SS_LIMBS - 1; i >= 0; i--) {
        if (acc[i]) { if (top < 0) top = 32 * i + 31 - BP8_CLZ32(acc[i]); pop += BP8_POPC32(acc[i]); }
    }
    const ds_i32 E = top - BP8_SS_RADIX - 5;
    ds_i32 se;
    if ((E & 1) == 0) se = E / 2;
    else { const ds_i32 n = (E - 1) / 2; se = (pop == 1) ? (((n & 1) == 0) ? n : n + 1) : n + 1; }
    return se > 127 ? 127 : (se < -128 ? -128 : se);
}

// nearest finite code to x (double bits); sv/sc = sorted value table (255). Mirrors the reference.
DS_FN ds_u32 bp8_encode_nearest_soft(ds_u64 x, BP8_TAB(ds_u64) sv, BP8_TAB(ds_u32) sc) {
    if ((x & 0x7FFFFFFFFFFFFFFFull) == 0) return BP8_ZERO;
    ds_i32 lo = 0, hi = 255;
    while (lo < hi) { const ds_i32 mid = (lo + hi) >> 1; if (ds_lt(sv[mid], x)) lo = mid + 1; else hi = mid; }
    ds_i32 best_k = -1; ds_u64 bestd = 0x7FF0000000000000ull;    // +inf
    const ds_i32 start = lo > 0 ? lo - 1 : lo, end = lo < 255 ? lo : lo - 1;
    for (ds_i32 k = start; k <= end; k++) {
        const ds_u64 d = ds_fabs(ds_sub(sv[k], x));
        if (ds_lt(d, bestd) || (ds_eq(d, bestd) && (best_k < 0 || sc[k] < sc[best_k]))) { bestd = d; best_k = k; }
    }
    if (best_k < 0 || !ds_lt(bestd, ds_fabs(x))) return BP8_ZERO;
    return sc[best_k];
}

// quantise one 32-float block (bit patterns in) -> scale_exp and 32 codes
DS_FN void bp8_quantize_block_soft(BP8_CPTR(ds_u32) xb, BP8_PTR(ds_i32) se_out, BP8_PTR(ds_u32) qs,
                                   BP8_TAB(ds_u64) sv, BP8_TAB(ds_u32) sc) {
    const ds_i32 se = bp8_scale_exp_exact_bits(xb);
    *se_out = se;
    for (int j = 0; j < BP8_QK; j++) {
        const ds_u64 x = ds_ldexp(ds_from_f32bits(xb[j]), -se);   // exact
        qs[j] = bp8_encode_nearest_soft(x, sv, sc);
    }
}

// lazily-carried limb accumulate (8 x int64 limbs), mirrors exact_lane_add
DS_FN void bp8_lane_add(BP8_PTR(ds_i64) a, ds_i64 P, ds_i32 sh) {
    if (sh >= 0) {
        const ds_i32 w = sh >> 5, bits = sh & 31;
        if (w < 8) {
            const ds_i64 V = P << bits;
            a[w] += (ds_i64) (ds_u32) V;
            if (w + 1 < 8) a[w + 1] += (V >> 32);
        }
    } else {
        const ds_i32 rs = -sh;
        const ds_i64 V = rs >= 63 ? (P < 0 ? -1 : 0) : (P >> rs);
        a[0] += V;
    }
}

// carry-normalise 8 int64 limbs into a 256-bit two's-complement word q[8]
DS_FN void bp8_limbs_to_q256(BP8_CPTR(ds_i64) a, BP8_PTR(ds_u32) q) {
    ds_i64 carry = 0;
    for (int w = 0; w < 8; w++) { const ds_i64 v = a[w] + carry; q[w] = (ds_u32) v; carry = v >> 32; }
}

// det_q256_to_double then round to float, on the software double
DS_FN ds_u32 bp8_q256_to_f32bits(BP8_CPTR(ds_u32) qin) {
    ds_u32 m[8]; for (int i = 0; i < 8; i++) m[i] = qin[i];
    const int neg = (m[7] >> 31) & 1;
    if (neg) { ds_u64 c = 1; for (int i = 0; i < 8; i++) { const ds_u64 v = (ds_u64) (~m[i]) + c; m[i] = (ds_u32) v; c = v >> 32; } }
    ds_u64 v = 0;
    for (int i = 7; i >= 0; i--) v = ds_add(ds_ldexp(v, 32), ds_from_u32(m[i]));
    v = ds_ldexp(v, -BP8_QFRAC);
    if (neg) v = ds_neg(v);
    return ds_to_f32bits(v);
}

// exact dot of two quantised rows (nblk blocks each) -> float bits; single-lane form
DS_FN ds_u32 bp8_dot_soft(BP8_CPTR(ds_i32) sex, BP8_CPTR(ds_u32) qx, BP8_CPTR(ds_i32) sey, BP8_CPTR(ds_u32) qy,
                          int nblk, BP8_TAB(ds_i32) tM, BP8_TAB(ds_i32) tE) {
    ds_i64 a[8]; for (int w = 0; w < 8; w++) a[w] = 0;
    for (int ib = 0; ib < nblk; ib++) {
        const ds_i32 se = sex[ib] + sey[ib] + BP8_QFRAC;
        for (int j = 0; j < BP8_QK; j++) {
            const ds_u32 cx = qx[ib * BP8_QK + j], cy = qy[ib * BP8_QK + j];
            const ds_i32 P = tM[cx] * tM[cy];
            if (P == 0) continue;
            bp8_lane_add(a, (ds_i64) P, tE[cx] + tE[cy] + se);
        }
    }
    ds_u32 q[8]; bp8_limbs_to_q256(a, q);
    return bp8_q256_to_f32bits(q);
}
#endif
