/* Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. */
/* Demonstrates WHY the b-posit8 exact-quire dot is reproducible across
 * hardware: it is invariant to reduction ORDER. The 256-bit Kulisch quire
 * accumulates every product with no intermediate rounding, so summing the
 * same products in any order yields bit-identical limbs. A float32 accumulator
 * does not: reordering the same products changes the result. Cross-hardware
 * bit-identity (host == ET-SoC1 board, max_abs = 0) rests on exactly this
 * property -- different SIMD widths / lane counts reduce in different orders.
 *
 * Self-contained (no golden header): generates its own deterministic vectors,
 * decodes with the same (M,E) logic as the shipped kernel, and checks
 * forward / reversed / shuffled orders. Build: cc -O2 bp8_reproducibility.c -lm */
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* ---- same (M,E) decode as ggml_vec_dot_bposit8_bposit8 -------------------- */
static void bp8_ME(uint8_t p, int64_t *M, int *E) {
    if (p == 0x00 || p == 0x80) { *M = 0; *E = 0; return; }
    int s = (p >> 7) & 1, rest = p & 0x7F;
    if (s) rest = ((~rest) + 1) & 0x7F;
    int leading = (rest >> 6) & 1, rs = 0;
    while (rs < 7 && ((rest >> (6 - rs)) & 1) == leading) rs++;
    int k_reg, e = 0, fb = 0, fw = 0;
    if (rs == 7) k_reg = leading ? 6 : -7;
    else {
        k_reg = leading ? (rs - 1) : -rs;
        int rem = 7 - (rs + 1), r2 = rest & ((1 << rem) - 1), ew = 2 < rem ? 2 : rem;
        if (ew > 0) { e = (r2 >> (rem - ew)) & ((1 << ew) - 1); e <<= (2 - ew); }
        rem -= ew; fw = rem; fb = fw > 0 ? (r2 & ((1 << fw) - 1)) : 0;
    }
    int64_t m = (1 << fw) + fb;
    *M = s ? -m : m; *E = 4 * k_reg + e - fw;
}
static double bp8_val(uint8_t p) { int64_t M; int E; bp8_ME(p, &M, &E); return ldexp((double)M, E); }

/* ---- 256-bit quire (8x32, radix at 96) ------------------------------------ */
#define QF 96
static void q_add(uint32_t q[8], int64_t P, int shift) {
    if (P == 0) return;
    uint32_t t[8], sx = (P < 0) ? 0xFFFFFFFFu : 0u; uint64_t up = (uint64_t)P;
    t[0] = (uint32_t)up; t[1] = (uint32_t)(up >> 32); for (int i = 2; i < 8; i++) t[i] = sx;
    if (shift > 0) {
        int w = shift >> 5, b = shift & 31;
        if (b) { uint32_t pv = 0; for (int i = 0; i < 8; i++) { uint32_t c = t[i]; t[i] = (c << b) | pv; pv = (uint32_t)((uint64_t)c >> (32 - b)); } }
        if (w) for (int i = 7; i >= 0; i--) t[i] = (i - w >= 0) ? t[i - w] : 0u;
    } else if (shift < 0) {
        int s = -shift, w = s >> 5, b = s & 31;
        if (w) for (int i = 0; i < 8; i++) t[i] = (i + w < 8) ? t[i + w] : sx;
        if (b) { uint32_t nx = sx; for (int i = 7; i >= 0; i--) { uint32_t c = t[i]; t[i] = (c >> b) | (nx << (32 - b)); nx = c; } }
    }
    uint64_t cy = 0; for (int i = 0; i < 8; i++) { uint64_t v = (uint64_t)q[i] + t[i] + cy; q[i] = (uint32_t)v; cy = v >> 32; }
}
/* quire dot over an index permutation `ord` */
static void quire_dot(const uint8_t *x, const uint8_t *y, const int *ord, int n, uint32_t q[8]) {
    for (int i = 0; i < 8; i++) q[i] = 0;
    for (int t = 0; t < n; t++) {
        int j = ord[t];
        int64_t Mx, My; int Ex, Ey;
        bp8_ME(x[j], &Mx, &Ex); if (!Mx) continue;
        bp8_ME(y[j], &My, &Ey); if (!My) continue;
        q_add(q, Mx * My, Ex + Ey + QF);
    }
}
static float float_dot(const uint8_t *x, const uint8_t *y, const int *ord, int n) {
    float s = 0.0f;
    for (int t = 0; t < n; t++) { int j = ord[t]; s += (float)bp8_val(x[j]) * (float)bp8_val(y[j]); }
    return s;
}

static uint64_t st = 0x5eed1234abcdULL;
static uint32_t rnd(void) { st = st * 6364136223846793005ULL + 1442695040888963407ULL; return st >> 33; }

int main(void) {
    const int N = 512, TRIALS = 200;
    int quire_orderdep = 0, float_orderdep = 0;
    for (int tr = 0; tr < TRIALS; tr++) {
        uint8_t x[512], y[512]; int fwd[512], rev[512], shuf[512];
        for (int i = 0; i < N; i++) {
            uint8_t a = rnd() & 0xFF, b = rnd() & 0xFF;
            x[i] = (a == 0x80) ? 0 : a; y[i] = (b == 0x80) ? 0 : b;
            fwd[i] = i; rev[i] = N - 1 - i; shuf[i] = i;
        }
        for (int i = N - 1; i > 0; i--) { int k = rnd() % (i + 1); int t = shuf[i]; shuf[i] = shuf[k]; shuf[k] = t; }

        uint32_t qf[8], qr[8], qs[8];
        quire_dot(x, y, fwd, N, qf); quire_dot(x, y, rev, N, qr); quire_dot(x, y, shuf, N, qs);
        int qsame = 1; for (int i = 0; i < 8; i++) if (qf[i] != qr[i] || qf[i] != qs[i]) qsame = 0;
        if (!qsame) quire_orderdep++;

        float ff = float_dot(x, y, fwd, N), fr = float_dot(x, y, rev, N), fs = float_dot(x, y, shuf, N);
        if (ff != fr || ff != fs) float_orderdep++;
    }
    printf("order-independence over %d trials of length %d:\n", TRIALS, N);
    printf("  exact-quire dot : %d/%d trials order-DEPENDENT  -> %s\n",
           quire_orderdep, TRIALS, quire_orderdep ? "FAIL" : "bit-identical in every order");
    printf("  float32 dot     : %d/%d trials order-DEPENDENT  (reordering changes the result)\n",
           float_orderdep, TRIALS);
    printf("== %s ==\n", quire_orderdep ? "REPRODUCIBILITY CLAIM VIOLATED" :
           "quire is order-invariant => same result on any hardware reduction order");
    return quire_orderdep ? 1 : 0;
}
