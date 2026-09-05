// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
//
// ggml-det: deterministic elementwise mathematics for the exact profile — bit-identical on
// every IEEE-754 platform (x86 CPU, CUDA, ARM, RISC-V). Every operation is an explicit IEEE
// round-to-nearest-even double or float operation in a fixed order; no libm, no FMA
// contraction (the host TU is compiled with -ffp-contract=off, the CUDA TU with -fmad=false
// and the *_rn intrinsics), no summation-order dependence (exact big-integer sums).
//
// Used by both backends through macros: include after defining DET_FN / DET_D* / DET_F* for
// the target, or take the defaults (plain C operators) on a contraction-free host TU.
#pragma once

#include <stdint.h>
#include <string.h>

#ifndef DET_FN
#define DET_FN static inline
#endif
#ifndef DET_DMUL
#define DET_DMUL(a, b) ((a) * (b))
#define DET_DADD(a, b) ((a) + (b))
#define DET_DSUB(a, b) ((a) - (b))
#define DET_DDIV(a, b) ((a) / (b))
#define DET_D2F(a)     ((float) (a))
#define DET_FMUL(a, b) ((a) * (b))
#define DET_FADD(a, b) ((a) + (b))
#define DET_FSUB(a, b) ((a) - (b))
#define DET_FDIV(a, b) ((a) / (b))
#define DET_FSQRT(a)   det_host_sqrtf(a)
#endif

// ---------------------------------------------------------------------------------------
// bit helpers (no libm)
// ---------------------------------------------------------------------------------------
DET_FN uint64_t det_d2bits(double x) { uint64_t b; memcpy(&b, &x, 8); return b; }
DET_FN double   det_bits2d(uint64_t b) { double x; memcpy(&x, &b, 8); return x; }
DET_FN uint32_t det_f2bits(float x) { uint32_t b; memcpy(&b, &x, 4); return b; }
DET_FN float    det_bits2f(uint32_t b) { float x; memcpy(&x, &b, 4); return x; }

// x * 2^k, exact while the result is normal; a subnormal result is rounded once (the
// first factor is exact, the second multiplication does the single rounding).
DET_FN double det_ldexp(double x, int k) {
    if (k > 1023) {
        x = DET_DMUL(x, det_bits2d((uint64_t) (1023 + 1023) << 52));
        k -= 1023;
        if (k > 1023) { x = DET_DMUL(x, det_bits2d((uint64_t) (1023 + 1023) << 52)); k -= 1023; if (k > 1023) k = 1023; }
    } else if (k < -1022) {
        x = DET_DMUL(x, det_bits2d((uint64_t) (1023 - 1022) << 52));
        k += 1022;
        if (k < -1022) { x = DET_DMUL(x, det_bits2d((uint64_t) (1023 - 1022) << 52)); k += 1022; if (k < -1022) k = -1022; }
    }
    return DET_DMUL(x, det_bits2d((uint64_t) (k + 1023) << 52));
}

// floor(x) for |x| < 2^62 without libm (truncation toward zero, then correction)
DET_FN double det_floor(double x) {
    const int64_t t = (int64_t) x;                 // truncates toward zero
    double r = (double) t;
    if (r > x) r = (double) (t - 1);
    return r;
}

// ---------------------------------------------------------------------------------------
// exact sum of squares of float32 values: 640-bit unsigned integer, radix point at bit 352
// (float32 squares span 2^-298 .. 2^254 and are exact in double: 24x24 bits fit in 53).
// The accumulator is a plain integer, so partial sums combine exactly in any order.
// ---------------------------------------------------------------------------------------
#define DET_BIG_LIMBS 20
#define DET_BIG_RADIX 352

DET_FN void det_big_zero(uint32_t acc[DET_BIG_LIMBS]) {
    for (int i = 0; i < DET_BIG_LIMBS; i++) acc[i] = 0;
}

// acc += v*v (exact); returns 0 for a non-finite v (caller decides), 1 otherwise
DET_FN int det_big_add_sq(uint32_t acc[DET_BIG_LIMBS], float v) {
    const double vd = (double) v;
    if (vd == 0.0) return 1;
    const uint32_t fb = det_f2bits(v);
    if (((fb >> 23) & 0xFF) == 0xFF) return 0;
    const double p = DET_DMUL(vd, vd);             // exact
    const uint64_t bits = det_d2bits(p);
    const uint64_t mi = (bits & 0xFFFFFFFFFFFFFull) | (1ull << 52);
    const int e2 = (int) ((bits >> 52) & 0x7FF) - 1023;
    const int pos = e2 - 52 + DET_BIG_RADIX;
    const int w = pos >> 5, b = pos & 31;
    const uint64_t lo = b ? (mi << b) : mi;
    const uint64_t hi = b ? (mi >> (64 - b)) : 0ull;
    const uint32_t parts[3] = { (uint32_t) lo, (uint32_t) (lo >> 32), (uint32_t) hi };
    uint64_t c = 0;
    for (int i = 0; i < 3; i++) { const uint64_t t = (uint64_t) acc[w + i] + parts[i] + c; acc[w + i] = (uint32_t) t; c = t >> 32; }
    for (int i = w + 3; c && i < DET_BIG_LIMBS; i++) { const uint64_t t = (uint64_t) acc[i] + c; acc[i] = (uint32_t) t; c = t >> 32; }
    return 1;
}

// bit index of the top set bit, or -1
DET_FN int det_big_top(const uint32_t acc[DET_BIG_LIMBS]) {
    for (int i = DET_BIG_LIMBS - 1; i >= 0; i--) {
        if (acc[i]) { int t = 31; while (!((acc[i] >> t) & 1u)) t--; return 32 * i + t; }
    }
    return -1;
}

DET_FN int det_big_popcount_is_one(const uint32_t acc[DET_BIG_LIMBS]) {
    int pop = 0;
    for (int i = 0; i < DET_BIG_LIMBS; i++) { uint32_t m = acc[i]; while (m) { pop += (int) (m & 1u); m >>= 1; if (pop > 1) return 0; } }
    return pop == 1;
}

// bit j of the accumulator (0 outside the range)
DET_FN uint32_t det_big_bit(const uint32_t acc[DET_BIG_LIMBS], int j) {
    if (j < 0 || j >= 32 * DET_BIG_LIMBS) return 0;
    return (acc[j >> 5] >> (j & 31)) & 1u;
}

// correctly rounded (nearest-even) double value of the accumulator (as S * 2^-RADIX)
DET_FN double det_big_to_double(const uint32_t acc[DET_BIG_LIMBS]) {
    const int t = det_big_top(acc);
    if (t < 0) return 0.0;
    uint64_t mant = 0;
    for (int j = t; j > t - 53; j--) mant = (mant << 1) | det_big_bit(acc, j);
    const uint32_t guard = det_big_bit(acc, t - 53);
    int sticky = 0;
    for (int j = t - 54; j >= 0 && !sticky; j--) sticky |= det_big_bit(acc, j);
    int e = t - 52;
    if (guard && (sticky || (mant & 1u))) { mant++; if (mant == (1ull << 53)) { mant >>= 1; e++; } }
    return det_ldexp((double) mant, e - DET_BIG_RADIX);
}

// exact sum of squares of n floats, correctly rounded to double (any non-finite -> +inf)
DET_FN double det_sumsq_f32(const float * x, int n) {
    uint32_t acc[DET_BIG_LIMBS];
    det_big_zero(acc);
    for (int i = 0; i < n; i++) if (!det_big_add_sq(acc, x[i])) return det_bits2d(0x7FF0000000000000ull);
    return det_big_to_double(acc);
}

// RMSNorm scale from the exact sum: mean = (float)(S/n) in double, scale = 1/sqrtf(mean+eps)
DET_FN float det_rms_scale(double sumsq, int n, float eps) {
    const float mean = DET_D2F(DET_DDIV(sumsq, (double) n));
    return DET_FDIV(1.0f, DET_FSQRT(DET_FADD(mean, eps)));
}

// ---------------------------------------------------------------------------------------
// exp / exp2 / log2 / sin / cos in double, Taylor series after Cody-Waite reduction.
// Accuracy ~1e-16 relative; deterministic by construction.
// ---------------------------------------------------------------------------------------
#define DET_INV_LN2 1.44269504088896338700e+00
#define DET_LN2_HI  6.93147180369123816490e-01
#define DET_LN2_LO  1.90821492927058770002e-10
#define DET_LN2     6.93147180559945286227e-01

// e^r for |r| <= 0.36: Taylor to r^13 (error < 5e-18)
DET_FN double det_exp_small(double r) {
    static const double c[] = {
        1.0, 1.0, 5.00000000000000000000e-01, 1.66666666666666657415e-01, 4.16666666666666643537e-02,
        8.33333333333333321769e-03, 1.38888888888888894189e-03, 1.98412698412698412526e-04,
        2.48015873015873015658e-05, 2.75573192239858925110e-06, 2.75573192239858883130e-07,
        2.50521083854417202234e-08, 2.08767569878681002718e-09, 1.60590438368216133364e-10 };
    double p = c[13];
    for (int i = 12; i >= 0; i--) p = DET_DADD(DET_DMUL(p, r), c[i]);
    return p;
}

DET_FN double det_exp_d(double x) {
    const double k = det_floor(DET_DADD(DET_DMUL(x, DET_INV_LN2), 0.5));
    const double r = DET_DSUB(DET_DSUB(x, DET_DMUL(k, DET_LN2_HI)), DET_DMUL(k, DET_LN2_LO));
    return det_ldexp(det_exp_small(r), (int) k);
}

DET_FN float det_expf(float x) {
    const uint32_t b = det_f2bits(x);
    if (((b >> 23) & 0xFF) == 0xFF) return (b & 0x7FFFFF) ? x : ((b >> 31) ? 0.0f : x);   // nan / -inf / +inf
    if (x > 88.75f) return det_bits2f(0x7F800000u);
    if (x < -104.0f) return 0.0f;
    return DET_D2F(det_exp_d((double) x));
}

DET_FN double det_exp2_d(double y) {
    const double k = det_floor(DET_DADD(y, 0.5));
    const double r = DET_DSUB(y, k);
    return det_ldexp(det_exp_small(DET_DMUL(r, DET_LN2)), (int) k);
}

// log2(x) for finite x > 0 (normal): atanh series in s = (m-1)/(m+1), m in [sqrt(1/2), sqrt(2))
DET_FN double det_log2_d(double x) {
    const uint64_t bits = det_d2bits(x);
    int e = (int) ((bits >> 52) & 0x7FF) - 1023;
    double m = det_bits2d((bits & 0xFFFFFFFFFFFFFull) | (1023ull << 52));   // [1, 2)
    if (m > 1.41421356237309514547e+00) { m = DET_DMUL(m, 0.5); e++; }
    const double s  = DET_DDIV(DET_DSUB(m, 1.0), DET_DADD(m, 1.0));
    const double s2 = DET_DMUL(s, s);
    double p = 0.0;
    for (int k = 29; k >= 1; k -= 2) p = DET_DADD(DET_DMUL(p, s2), DET_DDIV(1.0, (double) k));
    const double ln_m = DET_DMUL(2.0, DET_DMUL(s, p));
    return DET_DADD((double) e, DET_DMUL(ln_m, DET_INV_LN2));
}

#define DET_TWO_OVER_PI 6.36619772367581382433e-01
#define DET_PIO2_1 1.57079632673412561417e+00
#define DET_PIO2_2 6.07710050650619224932e-11
#define DET_PIO2_3 2.02226624879595063154e-21

// sin / cos of a double after reduction to |r| <= pi/4: Taylor to r^15 / r^16 (error < 1e-18)
DET_FN void det_sincos_d(double x, double * s, double * c) {
    const double n  = det_floor(DET_DADD(DET_DMUL(x, DET_TWO_OVER_PI), 0.5));
    const double r  = DET_DSUB(DET_DSUB(DET_DSUB(x, DET_DMUL(n, DET_PIO2_1)), DET_DMUL(n, DET_PIO2_2)), DET_DMUL(n, DET_PIO2_3));
    const double r2 = DET_DMUL(r, r);
    static const double sc[] = { 1.0, -1.66666666666666657415e-01, 8.33333333333333321769e-03, -1.98412698412698412526e-04,
                                 2.75573192239858925110e-06, -2.50521083854417202234e-08, 1.60590438368216133364e-10,
                                 -7.64716373181981647590e-13 };
    static const double cc[] = { 1.0, -5.00000000000000000000e-01, 4.16666666666666643537e-02, -1.38888888888888894189e-03,
                                 2.48015873015873015658e-05, -2.75573192239858883130e-07, 2.08767569878681002718e-09,
                                 -1.14707455977297245139e-11, 4.77947733238738529744e-14 };
    double ps = sc[7];
    for (int i = 6; i >= 0; i--) ps = DET_DADD(DET_DMUL(ps, r2), sc[i]);
    ps = DET_DMUL(ps, r);
    double pc = cc[8];
    for (int i = 7; i >= 0; i--) pc = DET_DADD(DET_DMUL(pc, r2), cc[i]);
    const int64_t q = ((int64_t) n) & 3;
    switch (q) {
        case 0: *s = ps;  *c = pc;  break;
        case 1: *s = pc;  *c = -ps; break;
        case 2: *s = -ps; *c = -pc; break;
        default: *s = -pc; *c = ps; break;
    }
}

DET_FN void det_sincosf(float theta, float * s, float * c) {
    double sd, cd;
    det_sincos_d((double) theta, &sd, &cd);
    *s = DET_D2F(sd);
    *c = DET_D2F(cd);
}

// RoPE: frequency of dimension pair i (0-based) = freq_base^(-2i/n_dims), in double
DET_FN double det_rope_freq(int i, int n_dims, float freq_base) {
    const double L = DET_DMUL(det_log2_d((double) freq_base), DET_DDIV(DET_DMUL(-2.0, (double) i), (double) n_dims));
    return det_exp2_d(L);
}

// RoPE cos/sin for (pos, pair i), freq_scale applied in double, one rounding to float
DET_FN void det_rope_sincos(float pos, int i, int n_dims, float freq_base, float freq_scale, float * s, float * c) {
    const double theta = DET_DMUL(DET_DMUL((double) pos, det_rope_freq(i, n_dims, freq_base)), (double) freq_scale);
    double sd, cd;
    det_sincos_d(theta, &sd, &cd);
    *s = DET_D2F(sd);
    *c = DET_D2F(cd);
}

// RoPE with a per-pair frequency factor (Llama 3 rope_freqs): theta = ((pos * freq_i) / ff) * freq_scale
DET_FN void det_rope_sincos_ff(float pos, int i, int n_dims, float freq_base, float freq_scale, float ff, float * s, float * c) {
    const double theta = DET_DMUL(DET_DDIV(DET_DMUL((double) pos, det_rope_freq(i, n_dims, freq_base)), (double) ff), (double) freq_scale);
    double sd, cd;
    det_sincos_d(theta, &sd, &cd);
    *s = DET_D2F(sd);
    *c = DET_D2F(cd);
}

// tanh in double via exp: (e^{2y} - 1) / (e^{2y} + 1); saturates beyond |y| > 20
DET_FN double det_tanh_d(double y) {
    if (y > 20.0) return 1.0;
    if (y < -20.0) return -1.0;
    const double u = det_exp_d(DET_DMUL(2.0, y));
    return DET_DDIV(DET_DSUB(u, 1.0), DET_DADD(u, 1.0));
}
DET_FN float det_tanhf(float x) {
    const uint32_t b = det_f2bits(x);
    if (((b >> 23) & 0xFF) == 0xFF && (b & 0x7FFFFF)) return x;    // nan
    return DET_D2F(det_tanh_d((double) x));
}

// GELU (tanh form) with ggml's constants and operation order:
// 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)))
#define DET_GELU_COEF_A    0.044715f
#define DET_SQRT_2_OVER_PI 0.79788456080286535587989211986876f
DET_FN float det_geluf(float x) {
    const float inner = DET_FMUL(DET_SQRT_2_OVER_PI, DET_FMUL(x, DET_FADD(1.0f, DET_FMUL(DET_GELU_COEF_A, DET_FMUL(x, x)))));
    return DET_FMUL(DET_FMUL(0.5f, x), DET_FADD(1.0f, det_tanhf(inner)));
}

// SiLU and sigmoid with a fixed formula: x * (1 / (1 + exp(-x)))
DET_FN float det_sigmoidf(float x) { return DET_FDIV(1.0f, DET_FADD(1.0f, det_expf(-x))); }
DET_FN float det_siluf(float x)    { return DET_FMUL(x, det_sigmoidf(x)); }

// ---------------------------------------------------------------------------------------
// 256-bit two's-complement quire (8x32 limbs, radix point at bit 96) — the readout shared by
// every exact matmul (bposit8 CPU/CUDA, f16 CPU/CUDA): one limb-to-double loop, one rounding.
// ---------------------------------------------------------------------------------------
#define DET_QFRAC 96

DET_FN void det_q256_add_shifted(uint32_t q[8], int64_t P, int shift) {
    if (P == 0) return;
    uint32_t t[8];
    const uint32_t sx = (P < 0) ? 0xFFFFFFFFu : 0u;
    const uint64_t up = (uint64_t) P;
    t[0] = (uint32_t) up; t[1] = (uint32_t) (up >> 32);
    for (int i = 2; i < 8; i++) t[i] = sx;
    if (shift > 0) {
        const int words = shift >> 5, bits = shift & 31;
        if (bits) { uint32_t prev = 0; for (int i = 0; i < 8; i++) { const uint32_t cur = t[i]; t[i] = (cur << bits) | prev; prev = (uint32_t) ((uint64_t) cur >> (32 - bits)); } }
        if (words) for (int i = 7; i >= 0; i--) t[i] = (i - words >= 0) ? t[i - words] : 0u;
    } else if (shift < 0) {
        const int sh = -shift, words = sh >> 5, bits = sh & 31;
        if (words) for (int i = 0; i < 8; i++) t[i] = (i + words < 8) ? t[i + words] : sx;
        if (bits) { uint32_t next = sx; for (int i = 7; i >= 0; i--) { const uint32_t cur = t[i]; t[i] = (cur >> bits) | (next << (32 - bits)); next = cur; } }
    }
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) { const uint64_t v = (uint64_t) q[i] + t[i] + carry; q[i] = (uint32_t) v; carry = v >> 32; }
}

DET_FN double det_q256_to_double(const uint32_t q[8]) {
    uint32_t m[8];
    for (int i = 0; i < 8; i++) m[i] = q[i];
    const int neg = (m[7] >> 31) & 1;
    if (neg) { uint64_t c = 1; for (int i = 0; i < 8; i++) { const uint64_t v = (uint64_t) (~m[i]) + c; m[i] = (uint32_t) v; c = v >> 32; } }
    double v = 0.0;
    for (int i = 7; i >= 0; i--) v = DET_DADD(DET_DMUL(v, 4294967296.0), (double) m[i]);
    v = det_ldexp(v, -DET_QFRAC);
    return neg ? -v : v;
}

// IEEE binary16 -> integer form value = M * 2^E (M = 0 for inf/nan: both sides agree)
DET_FN void det_f16_to_ME(uint16_t h, int * M, int * E) {
    const int s = (h >> 15) & 1, e = (h >> 10) & 0x1F, f = h & 0x3FF;
    int m;
    if (e == 0)       { m = f;        *E = -24; }
    else if (e == 31) { m = 0;        *E = 0;   }
    else              { m = 1024 + f; *E = e - 25; }
    *M = s ? -m : m;
}

// exact dot of two f16 rows (any order, one rounding) — the CPU form (shift bins)
DET_FN float det_dot_f16(const uint16_t * x, const uint16_t * y, int n) {
    int64_t bins[64];                       // shift = Ex + Ey + 96 in [48, 106]
    for (int i = 0; i < 64; i++) bins[i] = 0;
    for (int k = 0; k < n; k++) {
        int Mx, Ex, My, Ey;
        det_f16_to_ME(x[k], &Mx, &Ex);
        det_f16_to_ME(y[k], &My, &Ey);
        const int64_t P = (int64_t) Mx * (int64_t) My;
        if (P != 0) bins[Ex + Ey + DET_QFRAC - 48] += P;
    }
    uint32_t q[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < 64; i++) if (bins[i] != 0) det_q256_add_shifted(q, bins[i], i + 48);
    return DET_D2F(det_q256_to_double(q));
}

// acc += v for a finite non-negative float (softmax numerators); exact
DET_FN void det_big_add_f32_nonneg(uint32_t acc[DET_BIG_LIMBS], float v) {
    const uint32_t fb = det_f2bits(v);
    const int e = (fb >> 23) & 0xFF, f = fb & 0x7FFFFF;
    if ((fb & 0x7FFFFFFF) == 0 || e == 0xFF) return;
    uint64_t mi; int e2;
    if (e == 0) { mi = (uint64_t) f; e2 = -126 - 23; } else { mi = (uint64_t) (f | 0x800000); e2 = e - 127 - 23; }
    const int pos = e2 + DET_BIG_RADIX;   // value = mi * 2^e2
    const int w = pos >> 5, b = pos & 31;
    const uint64_t lo = b ? (mi << b) : mi;
    const uint64_t hi = b ? (mi >> (64 - b)) : 0ull;
    const uint32_t parts[3] = { (uint32_t) lo, (uint32_t) (lo >> 32), (uint32_t) hi };
    uint64_t c = 0;
    for (int i = 0; i < 3; i++) { const uint64_t t = (uint64_t) acc[w + i] + parts[i] + c; acc[w + i] = (uint32_t) t; c = t >> 32; }
    for (int i = w + 3; c && i < DET_BIG_LIMBS; i++) { const uint64_t t = (uint64_t) acc[i] + c; acc[i] = (uint32_t) t; c = t >> 32; }
}

// softmax numerators y[i] = exp(x[i] - max) (deterministic exp) and their EXACT sum, as a
// correctly rounded double; sink (already exp'd, or 0) is added to the sum too
DET_FN double det_soft_max_f32(int n, float * y, const float * x, float max, float sink_exp) {
    uint32_t acc[DET_BIG_LIMBS];
    det_big_zero(acc);
    for (int i = 0; i < n; i++) {
        const float v = det_expf(DET_FSUB(x[i], max));
        y[i] = v;
        det_big_add_f32_nonneg(acc, v);
    }
    if (sink_exp != 0.0f) det_big_add_f32_nonneg(acc, sink_exp);
    return det_big_to_double(acc);
}

// the normalisation both backends apply: inv = 1/(float)sum, y *= inv
DET_FN float det_soft_max_inv(double sum) { return DET_FDIV(1.0f, DET_D2F(sum)); }
