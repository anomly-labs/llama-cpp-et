// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Host translation unit for ggml-det: compiled with -ffp-contract=off (see CMakeLists) so
// the plain-operator macros are exactly the IEEE operations the CUDA *_rn intrinsics perform.
#include <math.h>
static inline float det_host_sqrtf(float a) { return sqrtf(a); }   // IEEE correctly rounded
#include "ggml-det.h"
#include "ggml-det-api.h"

double ggml_det_sumsq_f32(const float * x, int n)                 { return det_sumsq_f32(x, n); }
float  ggml_det_rms_scale(double sumsq, int n, float eps)          { return det_rms_scale(sumsq, n, eps); }
float  ggml_det_expf(float x)                                      { return det_expf(x); }
float  ggml_det_siluf(float x)                                     { return det_siluf(x); }
float  ggml_det_sigmoidf(float x)                                  { return det_sigmoidf(x); }
void   ggml_det_sincosf(float t, float * s, float * c)             { det_sincosf(t, s, c); }
void   ggml_det_rope_sincos(float pos, int i, int n_dims, float fb, float fs, float * s, float * c) { det_rope_sincos(pos, i, n_dims, fb, fs, s, c); }
double ggml_det_log2_d(double x)                                   { return det_log2_d(x); }
double ggml_det_exp2_d(double y)                                   { return det_exp2_d(y); }
double ggml_det_exp_d(double x)                                    { return det_exp_d(x); }
// CPU fast path of det_dot_f16: 65,536-entry decode tables (M as int16, E as int8), unrolled
// lanes, the same bins / placement / readout — bit-identical (OpenEvolve workspace
// f16_exact_dot_speed gate: random rows incl. inf/nan/subnormal, cancellation, permutation).
static int16_t g_f16_M[65536];
static int8_t  g_f16_E[65536];
static volatile int g_f16_ready = 0;
static void det_f16_tables_init(void) {
    if (g_f16_ready) return;
    for (int h = 0; h < 65536; h++) {
        int M, E;
        det_f16_to_ME((uint16_t) h, &M, &E);
        g_f16_M[h] = (int16_t) M;
        g_f16_E[h] = (int8_t) E;
    }
    g_f16_ready = 1;
}
float ggml_det_dot_f16(const uint16_t * x, const uint16_t * y, int n) {
    // OpenEvolve f16_exact_dot_speed round 1 (2026-09-05): 16-lane chunks, two lanes per step,
    // 59 bins, zero-product guard — 1.66x over the seed under the bit-exact gate.
    det_f16_tables_init();
    int64_t bins[59] = { 0 };                              // shifts 48..106
    const int full = n & ~15;
    for (int k = 0; k < full; k += 16) {
        for (int i = 0; i < 16; i += 2) {
            const uint16_t hx1 = x[k + i], hy1 = y[k + i], hx2 = x[k + i + 1], hy2 = y[k + i + 1];
            const int64_t P1 = (int64_t) g_f16_M[hx1] * (int64_t) g_f16_M[hy1];
            if (P1 != 0) bins[g_f16_E[hx1] + g_f16_E[hy1] + (DET_QFRAC - 48)] += P1;
            const int64_t P2 = (int64_t) g_f16_M[hx2] * (int64_t) g_f16_M[hy2];
            if (P2 != 0) bins[g_f16_E[hx2] + g_f16_E[hy2] + (DET_QFRAC - 48)] += P2;
        }
    }
    for (int k = full; k < n; k++) {
        const int64_t P = (int64_t) g_f16_M[x[k]] * (int64_t) g_f16_M[y[k]];
        if (P != 0) bins[g_f16_E[x[k]] + g_f16_E[y[k]] + (DET_QFRAC - 48)] += P;
    }
    uint32_t q[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < 59; i++) if (bins[i] != 0) det_q256_add_shifted(q, bins[i], i + 48);
    return DET_D2F(det_q256_to_double(q));
}
double ggml_det_soft_max_f32(int n, float * y, const float * x, float max, float sink_exp) { return det_soft_max_f32(n, y, x, max, sink_exp); }
float  ggml_det_soft_max_inv(double sum)                           { return det_soft_max_inv(sum); }
void   ggml_det_rope_sincos_ff(float pos, int i, int n_dims, float fb, float fs, float ff, float * s, float * c) { det_rope_sincos_ff(pos, i, n_dims, fb, fs, ff, s, c); }
float  ggml_det_geluf(float x)                                     { return det_geluf(x); }
float  ggml_det_tanhf(float x)                                     { return det_tanhf(x); }
