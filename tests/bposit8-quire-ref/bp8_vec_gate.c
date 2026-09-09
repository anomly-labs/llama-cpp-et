// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Gate for the vectorised exact-quire vec_dot: the AVX2 path (default dispatch) must equal the
// scalar reference bit-for-bit on random rows (uniform codes incl. zero/NaR), extreme block
// scales, blocks engineered to exceed the anchored int64 range (fallback), sub-radix blocks,
// and under block permutation (order independence). Build:
//   cc -O2 -mavx2 -Iggml/include -Iggml/src -Iggml/src/ggml-cpu tests/bposit8-quire-ref/bp8_vec_gate.c \
//      -Lbuild-bp8/bin -lggml-cpu -lggml-base -lm -Wl,-rpath,build-bp8/bin -o /tmp/bp8_vec_gate && /tmp/bp8_vec_gate
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t r32(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t) rng; }
static int rint_(int lo, int hi) { return lo + (int)(r32() % (uint32_t)(hi - lo + 1)); }

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// codes with extreme exponents (from the (M,E) lattice): E=+27 -> 0x7E/0x7F region, E=-32 -> 0x01
static void fill_block(block_bposit8 * b, int mode, int lo_scale, int hi_scale) {
    b->scale_exp = (int8_t) rint_(lo_scale, hi_scale);
    for (int j = 0; j < QK_BPOSIT8; j++) {
        uint8_t c = (uint8_t) r32();
        if (mode == 1) c = (j & 1) ? 0x7F : 0x01;            // widest E spread in one block -> fallback
        if (mode == 2) c = (r32() & 7) ? c : 0x80;           // many NaR/zero
        b->qs[j] = c;
    }
}

int main(void) {
    const int ns[] = { 32, 64, 576, 1536, 2048 };
    long checked = 0, bad = 0, fallback_rows = 0;
    char why[256] = "";
    for (int trial = 0; trial < 6000; trial++) {
        int n = ns[trial % 5]; int nb = n / QK_BPOSIT8;
        block_bposit8 * x = calloc(nb, sizeof *x), * y = calloc(nb, sizeof *y);
        int kind = trial % 6;
        for (int ib = 0; ib < nb; ib++) {
            switch (kind) {
                case 0: fill_block(&x[ib], 0, -20, 10);  fill_block(&y[ib], 0, -20, 10);  break;  // typical
                case 1: fill_block(&x[ib], 0, -128, 127); fill_block(&y[ib], 0, -128, 127); break; // extreme scales
                case 2: fill_block(&x[ib], 1, -20, 10);  fill_block(&y[ib], 0, -20, 10);  fallback_rows++; break; // wide E spread
                case 3: fill_block(&x[ib], 0, -128, -60); fill_block(&y[ib], 0, -128, -40); break; // sub-radix (se < 0)
                case 4: fill_block(&x[ib], 2, -20, 10);  fill_block(&y[ib], 2, -20, 10);  break;  // sparse
                default: fill_block(&x[ib], 0, 40, 127);  fill_block(&y[ib], 0, 40, 127);  break;  // huge shifts (>= 256 wrap)
            }
        }
        float sv = 0, sc = 0;
        ggml_vec_dot_bposit8_bposit8(n, &sv, 0, x, 0, y, 0, 1);
        ggml_vec_dot_bposit8_bposit8_scalar(n, &sc, 0, x, 0, y, 0, 1);
        checked++;
        if (f2u(sv) != f2u(sc)) { bad++; if (!why[0]) snprintf(why, sizeof why, "trial %d kind %d n %d: vec %08x scalar %08x", trial, kind, n, f2u(sv), f2u(sc)); }
        // block permutation: exact accumulation is order independent, both kernels must agree with themselves
        if (nb > 1) {
            for (int ib = nb - 1; ib > 0; ib--) { int j = (int)(r32() % (uint32_t)(ib + 1)); block_bposit8 t = x[ib]; x[ib] = x[j]; x[j] = t; t = y[ib]; y[ib] = y[j]; y[j] = t; }
            float sp = 0; ggml_vec_dot_bposit8_bposit8(n, &sp, 0, x, 0, y, 0, 1);
            checked++;
            if (f2u(sp) != f2u(sv)) { bad++; if (!why[0]) snprintf(why, sizeof why, "trial %d: permuted %08x vs %08x", trial, f2u(sp), f2u(sv)); }
        }
        // 2x2 tile: rows (x, x') x columns (y, y') must equal four single dots, and the scalar tile
        if (nb >= 1 && trial % 2 == 0) {
            block_bposit8 * x2 = calloc(nb, sizeof *x2), * y2 = calloc(nb, sizeof *y2);
            for (int ib = 0; ib < nb; ib++) { fill_block(&x2[ib], 0, -20, 10); fill_block(&y2[ib], 0, -20, 10); }
            block_bposit8 * xx = calloc(2 * nb, sizeof *xx), * yy = calloc(2 * nb, sizeof *yy);
            memcpy(xx, x, nb * sizeof *x); memcpy(xx + nb, x2, nb * sizeof *x); memcpy(yy, y, nb * sizeof *y); memcpy(yy + nb, y2, nb * sizeof *y);
            float tile[2 * 16] = { 0 }, ref[4], st[2 * 16] = { 0 };
            ggml_vec_dot_bposit8_bposit8(n, tile, 16, xx, nb * sizeof *xx, yy, nb * sizeof *yy, 2);
            ggml_vec_dot_bposit8_bposit8_scalar(n, &ref[0], 0, x, 0, y, 0, 1);  ggml_vec_dot_bposit8_bposit8_scalar(n, &ref[1], 0, x2, 0, y, 0, 1);
            ggml_vec_dot_bposit8_bposit8_scalar(n, &ref[2], 0, x, 0, y2, 0, 1); ggml_vec_dot_bposit8_bposit8_scalar(n, &ref[3], 0, x2, 0, y2, 0, 1);
            checked += 4;
            if (f2u(tile[0]) != f2u(ref[0]) || f2u(tile[1]) != f2u(ref[1]) || f2u(tile[16]) != f2u(ref[2]) || f2u(tile[17]) != f2u(ref[3])) {
                bad++; if (!why[0]) snprintf(why, sizeof why, "trial %d: 2x2 tile differs from single dots", trial);
            }
            free(x2); free(y2); free(xx); free(yy); (void) st;
        }
        free(x); free(y);
    }
    // throughput on REALISTIC rows: codes nearest to Gaussian samples (what quantize_row makes of
    // activations and weights), so the in-block exponent spread is the real one. Uniform-random
    // codes span E in [-32, 27] and put nearly every block on the scalar fallback.
    double val[256]; int M_[256], E_[256];
    for (int c = 0; c < 256; c++) {   // decode as the kernel does (mirror of ggml_bp8_code_to_ME)
        if (c == 0 || c == 0x80) { val[c] = 0; M_[c] = 0; E_[c] = 0; continue; }
        int sg = (c >> 7) & 1, rest = c & 0x7F; if (sg) rest = ((~rest) + 1) & 0x7F;
        int leading = (rest >> 6) & 1, rs = 0; while (rs < 7 && ((rest >> (6 - rs)) & 1) == leading) rs++;
        int k, e = 0, fb = 0, fw = 0;
        if (rs == 7) k = leading ? 6 : -7; else { k = leading ? rs - 1 : -rs; int rem = 7 - (rs + 1), r2 = rest & ((1 << rem) - 1), ew = rem < 2 ? rem : 2; if (ew > 0) { e = (r2 >> (rem - ew)) & ((1 << ew) - 1); e <<= (2 - ew); } rem -= ew; fw = rem; fb = fw > 0 ? (r2 & ((1 << fw) - 1)) : 0; }
        int m = (1 << fw) + fb; M_[c] = sg ? -m : m; E_[c] = 4 * k + e - fw; val[c] = M_[c] * pow(2.0, E_[c]);
    }
    int n = 2048, nb = 64; block_bposit8 * x = calloc(nb, sizeof *x), * y = calloc(nb, sizeof *y);
    long fb_blocks = 0;
    for (int ib = 0; ib < nb; ib++) {
        x[ib].scale_exp = 0; y[ib].scale_exp = 0;
        int smin = 1 << 20, smax = -(1 << 20);
        for (int j = 0; j < QK_BPOSIT8; j++) {
            for (int t = 0; t < 2; t++) {
                double u1 = (r32() + 1.0) / 4294967297.0, u2 = r32() / 4294967296.0;
                double g = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);   // N(0,1)
                int best = 0; double bd = 1e300;
                for (int c = 0; c < 256; c++) { if (c == 0x80) continue; double d = fabs(val[c] - g); if (d < bd) { bd = d; best = c; } }
                (t ? y : x)[ib].qs[j] = (uint8_t) best;
            }
            if (M_[x[ib].qs[j]] && M_[y[ib].qs[j]]) { int sh = E_[x[ib].qs[j]] + E_[y[ib].qs[j]]; if (sh < smin) smin = sh; if (sh > smax) smax = sh; }
        }
        if (smax - smin > 48) fb_blocks++;
    }
    printf("realistic rows: in-block shift range > 48 (scalar fallback) in %ld of %d blocks\n", fb_blocks, nb);
    float acc = 0, s; struct timespec t0, t1; const int reps = 20000;
    clock_gettime(CLOCK_MONOTONIC, &t0); for (int r = 0; r < reps; r++) { ggml_vec_dot_bposit8_bposit8(n, &s, 0, x, 0, y, 0, 1); acc += s; }
    clock_gettime(CLOCK_MONOTONIC, &t1); double tv = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    clock_gettime(CLOCK_MONOTONIC, &t0); for (int r = 0; r < reps; r++) { ggml_vec_dot_bposit8_bposit8_scalar(n, &s, 0, x, 0, y, 0, 1); acc += s; }
    clock_gettime(CLOCK_MONOTONIC, &t1); double ts = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("gate: %ld checks, %ld mismatches (%ld rows built to force the fallback)%s%s\n", checked, bad, fallback_rows, bad ? " -- FIRST: " : "", why);
    printf("throughput n=2048: vector %.1f ns/elem, scalar %.1f ns/elem, %.1fx (acc %g)\n", tv * 1e9 / (reps * n), ts * 1e9 / (reps * n), ts / tv, acc);
    return bad ? 1 : 0;
}
