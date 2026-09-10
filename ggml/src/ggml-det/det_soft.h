// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// det_soft.h — software IEEE-754 binary64 add/sub (round-to-nearest-even) and binary64 ->
// binary32 conversion on 64-bit integers, for targets without double (Metal). Written so the
// same text compiles as C and as Metal Shading Language: only fixed-width integers, no libm.
// Gate: research/metal-exact/test_det_soft.c compares every routine against hardware double.
#ifndef DET_SOFT_H
#define DET_SOFT_H

#ifdef __METAL_VERSION__
#define DS_FN static inline
#define DS_PTR(T) thread T *
#define DS_CLZ64(x) ((int)metal::clz((metal::ulong)(x)))
typedef ulong ds_u64; typedef long ds_i64; typedef uint ds_u32; typedef int ds_i32;   // after `using namespace metal;`
#else
#include <stdint.h>
#define DS_FN static inline
#define DS_PTR(T) T *
#define DS_CLZ64(x) __builtin_clzll((unsigned long long)(x))
typedef uint64_t ds_u64; typedef int64_t ds_i64; typedef uint32_t ds_u32; typedef int32_t ds_i32;
#endif

// ---- binary64 layout helpers (bits form) ----
DS_FN ds_u64 ds_sign(ds_u64 a) { return a >> 63; }
DS_FN ds_i32 ds_exp(ds_u64 a)  { return (ds_i32)((a >> 52) & 0x7FF); }
DS_FN ds_u64 ds_frac(ds_u64 a) { return a & 0xFFFFFFFFFFFFFull; }
DS_FN ds_u64 ds_pack(ds_u64 s, ds_i32 e, ds_u64 f) { return (s << 63) | ((ds_u64)e << 52) | f; }

// value = m * 2^(e - 52) with m the full significand (hidden bit set for normals); zero -> m=0
DS_FN void ds_unpack(ds_u64 a, DS_PTR(ds_u64) m, DS_PTR(ds_i32) e) {
    ds_i32 ex = ds_exp(a); ds_u64 f = ds_frac(a);
    if (ex == 0) { *m = f; *e = 1; }                 // subnormal or zero: exponent 1, no hidden bit
    else         { *m = f | (1ull << 52); *e = ex; }
}

// round a positive value given as (mant, e) where value = mant * 2^(e-52-EXTRA) with EXTRA
// guard bits below the 53-bit target, sticky already folded into bit 0 of mant.
// Returns packed bits (sign applied by caller). mant may exceed 53+EXTRA bits by one (carry).
#define DS_EXTRA 10
DS_FN ds_u64 ds_round_pack(ds_u64 s, ds_i64 e, ds_u64 mant) {
    if (mant == 0) return s << 63;
    // normalise so the leading bit is at position 52 + DS_EXTRA
    ds_i32 lz = DS_CLZ64(mant);
    ds_i32 want = 63 - (52 + DS_EXTRA);              // leading zeros wanted
    if (lz > want) { mant <<= (lz - want); e -= (lz - want); }
    else if (lz < want) {                            // shift right with sticky
        ds_i32 sh = want - lz;
        ds_u64 lost = mant & ((1ull << sh) - 1);
        mant = (mant >> sh) | (lost ? 1ull : 0ull);
        e += sh;
    }
    // subnormal handling: if e < 1, shift right further (with sticky) so exponent field is 0
    if (e < 1) {
        ds_i64 sh = 1 - e;
        if (sh > 63) { mant = (mant ? 1ull : 0ull); }
        else { ds_u64 lost = mant & ((1ull << sh) - 1); mant = (mant >> sh) | (lost ? 1ull : 0ull); }
        e = 1;
    }
    // round to nearest even on the DS_EXTRA guard bits
    ds_u64 guard = mant & ((1ull << DS_EXTRA) - 1);
    ds_u64 hmid  = 1ull << (DS_EXTRA - 1);
    ds_u64 q = mant >> DS_EXTRA;
    if (guard > hmid || (guard == hmid && (q & 1ull))) q += 1;
    if (q >> 53) { q >>= 1; e += 1; }                 // rounding carry
    if (e >= 0x7FF) return ds_pack(s, 0x7FF, 0);      // overflow -> inf
    if ((q >> 52) == 0) return ds_pack(s, 0, q);      // stayed subnormal (or zero)
    return ds_pack(s, (ds_i32)e, q & 0xFFFFFFFFFFFFFull);
}

// a + b, both finite, IEEE RNE. (No NaN/inf inputs in our uses; zeros handled.)
DS_FN ds_u64 ds_add(ds_u64 a, ds_u64 b) {
    ds_u64 ma, mb; ds_i32 ea, eb;
    ds_unpack(a, &ma, &ea); ds_unpack(b, &mb, &eb);
    ds_u64 sa = ds_sign(a), sb = ds_sign(b);
    if (ma == 0 && mb == 0) return (sa & sb) << 63;   // -0 + -0 = -0, else +0
    if (ma == 0) return b;
    if (mb == 0) return a;
    // make a the larger magnitude (by exponent then mantissa)
    if (ea < eb || (ea == eb && ma < mb)) { ds_u64 t = ma; ma = mb; mb = t; ds_i32 te = ea; ea = eb; eb = te; ds_u64 ts = sa; sa = sb; sb = ts; }
    ma <<= DS_EXTRA; mb <<= DS_EXTRA;                 // 63-bit working width
    ds_i32 d = ea - eb;
    if (d > 0) {
        if (d >= 64) mb = (mb ? 1ull : 0ull);
        else { ds_u64 lost = mb & ((1ull << d) - 1); mb = (mb >> d) | (lost ? 1ull : 0ull); }
    }
    ds_u64 mant;
    if (sa == sb) mant = ma + mb; else mant = ma - mb; // ma >= mb by construction
    if (mant == 0) return 0;                          // exact cancellation -> +0 (RNE)
    return ds_round_pack(sa, ea, mant);
}
DS_FN ds_u64 ds_neg(ds_u64 a) { return a ^ (1ull << 63); }
DS_FN ds_u64 ds_sub(ds_u64 a, ds_u64 b) { return ds_add(a, ds_neg(b)); }
DS_FN ds_u64 ds_fabs(ds_u64 a) { return a & 0x7FFFFFFFFFFFFFFFull; }
// ordering of finite doubles (no NaN): a < b
DS_FN ds_i64 ds_key(ds_u64 a) { return (a >> 63) ? -(ds_i64)(a & 0x7FFFFFFFFFFFFFFFull) : (ds_i64)a; }
DS_FN int ds_lt(ds_u64 a, ds_u64 b) { return ds_key(a) < ds_key(b); }
DS_FN int ds_eq(ds_u64 a, ds_u64 b) { return ds_key(a) == ds_key(b); }
// exact scaling by 2^k (no overflow in our uses; underflow to subnormal handled by repack)
DS_FN ds_u64 ds_ldexp(ds_u64 a, ds_i32 k) {
    ds_u64 m; ds_i32 e; ds_unpack(a, &m, &e);
    if (m == 0) return a;
    return ds_round_pack(ds_sign(a), (ds_i64)e + k, m << DS_EXTRA);
}
// exact conversions
DS_FN ds_u64 ds_from_u32(ds_u32 v) { return v ? ds_round_pack(0, 1075, ((ds_u64)v) << DS_EXTRA) : 0; }   // e biased: value = m*2^(e-1075)
DS_FN ds_u64 ds_from_i64_exact(ds_i64 v) {        // |v| < 2^53 assumed exact
    ds_u64 s = v < 0; ds_u64 m = s ? (ds_u64)(-v) : (ds_u64)v;
    return m ? ds_round_pack(s, 1075, m << DS_EXTRA) : 0;
}
// float32 bits -> exact binary64 bits (finite inputs)
DS_FN ds_u64 ds_from_f32bits(ds_u32 f) {
    ds_u64 s = f >> 31; ds_i32 e = (f >> 23) & 0xFF; ds_u64 m = f & 0x7FFFFF;
    if (e == 0) { if (m == 0) return s << 63; return ds_round_pack(s, 1 - 127 + 1023 - 23 + 52, m << DS_EXTRA); }
    return ds_round_pack(s, (ds_i64)e - 127 + 1023, (m | 0x800000ull) << (29 + DS_EXTRA));
}
// binary64 -> binary32 bits, RNE, with subnormal floats and overflow to inf
DS_FN ds_u32 ds_to_f32bits(ds_u64 a) {
    ds_u64 s = ds_sign(a); ds_u64 m; ds_i32 e; ds_unpack(a, &m, &e);
    if (m == 0) return (ds_u32)(s << 31);
    ds_i32 lz = DS_CLZ64(m);                          // normalise leading bit to 52 for subnormal doubles
    if (lz > 11) { m <<= (lz - 11); e -= (lz - 11); }
    ds_i64 fe = (ds_i64)e - 1023 + 127;               // float exponent field candidate
    // keep 24 bits + guard: shift m (53 bits) right by 29 with sticky -> q has 24 bits + we need RNE
    ds_i32 sh = 29;
    if (fe < 1) { sh += (ds_i32)(1 - fe); fe = 1; }   // float subnormal: denormalise
    ds_u64 q, lost, hmid;
    if (sh > 64) return (ds_u32)(s << 31);                       // below hmid the smallest subnormal
    if (sh == 64) { q = 0; lost = m; hmid = 1ull << 63; }
    else { q = m >> sh; lost = m & ((1ull << sh) - 1); hmid = 1ull << (sh - 1); }
    if (lost > hmid || (lost == hmid && (q & 1ull))) q += 1;
    if (q >> 24) { q >>= 1; fe += 1; }
    if (fe >= 0xFF) return (ds_u32)((s << 31) | 0x7F800000u);
    if ((q >> 23) == 0) return (ds_u32)((s << 31) | (ds_u32)q);   // subnormal float (or zero)
    return (ds_u32)((s << 31) | ((ds_u32)fe << 23) | ((ds_u32)q & 0x7FFFFFu));
}

// ---- multiply, divide, floor, integer conversions (for the norm / softmax / rope ports) ----
#ifdef __METAL_VERSION__
#define DS_MULHI64(a, b) ((ds_u64)metal::mulhi((ulong)(a), (ulong)(b)))
#else
#define DS_MULHI64(a, b) ((ds_u64)(((unsigned __int128)(a) * (unsigned __int128)(b)) >> 64))
#endif
// a * b, finite inputs, IEEE RNE
DS_FN ds_u64 ds_mul(ds_u64 a, ds_u64 b) {
    ds_u64 ma, mb; ds_i32 ea, eb;
    ds_unpack(a, &ma, &ea); ds_unpack(b, &mb, &eb);
    const ds_u64 s = ds_sign(a) ^ ds_sign(b);
    if (ma == 0 || mb == 0) return s << 63;
    ds_i32 la = DS_CLZ64(ma) - 11, lb = DS_CLZ64(mb) - 11;     // subnormal significands -> 53 bits
    if (la > 0) { ma <<= la; ea -= la; }
    if (lb > 0) { mb <<= lb; eb -= lb; }
    const ds_u64 lo = ma * mb, hi = DS_MULHI64(ma, mb);       // 106-bit product, leading bit 104 or 105
    const ds_i32 top = 127 - DS_CLZ64(hi);
    const ds_i32 sh = top - 62;                                 // 42 or 43: leading bit -> 62
    ds_u64 mant = (hi << (64 - sh)) | (lo >> sh);
    const ds_u64 lost = lo & ((1ull << sh) - 1);
    mant |= (lost ? 1ull : 0ull);
    return ds_round_pack(s, (ds_i64) ea + eb + sh - 1065, mant);
}
// a / b, finite, b != 0, IEEE RNE (restoring division, 62 quotient bits + sticky)
DS_FN ds_u64 ds_div(ds_u64 a, ds_u64 b) {
    ds_u64 ma, mb; ds_i32 ea, eb;
    ds_unpack(a, &ma, &ea); ds_unpack(b, &mb, &eb);
    const ds_u64 s = ds_sign(a) ^ ds_sign(b);
    if (ma == 0) return s << 63;
    ds_i32 la = DS_CLZ64(ma) - 11, lb = DS_CLZ64(mb) - 11;
    if (la > 0) { ma <<= la; ea -= la; }
    if (lb > 0) { mb <<= lb; eb -= lb; }
    ds_u64 rem, q;                                              // integer part first so rem < mb
    if (ma >= mb) { rem = ma - mb; q = 1; } else { rem = ma; q = 0; }
    for (int i = 0; i < 62; i++) { rem <<= 1; q <<= 1; if (rem >= mb) { rem -= mb; q |= 1ull; } }
    q |= (rem ? 1ull : 0ull);                                   // sticky; q = floor(ma*2^62/mb) < 2^63
    return ds_round_pack(s, (ds_i64) ea - eb + 1023, q);
}
// any int64 -> double (RNE)
DS_FN ds_u64 ds_from_i64(ds_i64 v) {
    if (v == 0) return 0;
    const ds_u64 s = v < 0; const ds_u64 m = s ? (ds_u64) (-(v + 1)) + 1ull : (ds_u64) v;
    return ds_round_pack(s, 1085, m);                           // value = m * 2^(e - 1085)
}
// truncate toward zero to int64 (|x| < 2^63)
DS_FN ds_i64 ds_to_i64_trunc(ds_u64 a) {
    ds_u64 m; ds_i32 e; ds_unpack(a, &m, &e);
    if (m == 0) return 0;
    const ds_i32 k = e - 1075;                                  // value = m * 2^k
    ds_u64 mag;
    if (k >= 0) mag = (k >= 64) ? 0 : (m << k);
    else mag = (k <= -64) ? 0 : (m >> (-k));
    return ds_sign(a) ? -(ds_i64) mag : (ds_i64) mag;
}
// floor for |x| < 2^62, the reference's way: truncate toward zero, then correct
DS_FN ds_u64 ds_floor(ds_u64 a) {
    const ds_i64 t = ds_to_i64_trunc(a);
    ds_u64 r = ds_from_i64(t);
    if (ds_lt(a, r)) r = ds_from_i64(t - 1);
    return r;
}
#endif
