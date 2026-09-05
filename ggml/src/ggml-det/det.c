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
float  ggml_det_dot_f16(const uint16_t * x, const uint16_t * y, int n)    { return det_dot_f16(x, y, n); }
double ggml_det_soft_max_f32(int n, float * y, const float * x, float max, float sink_exp) { return det_soft_max_f32(n, y, x, max, sink_exp); }
float  ggml_det_soft_max_inv(double sum)                           { return det_soft_max_inv(sum); }
