// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Non-inline host entry points for ggml-det (implemented in det.c, contraction-free).
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
double ggml_det_sumsq_f32(const float * x, int n);
float  ggml_det_rms_scale(double sumsq, int n, float eps);
float  ggml_det_expf(float x);
float  ggml_det_siluf(float x);
float  ggml_det_sigmoidf(float x);
void   ggml_det_sincosf(float t, float * s, float * c);
void   ggml_det_rope_sincos(float pos, int i, int n_dims, float freq_base, float freq_scale, float * s, float * c);
double ggml_det_log2_d(double x);
double ggml_det_exp2_d(double y);
double ggml_det_exp_d(double x);
float  ggml_det_dot_f16(const uint16_t * x, const uint16_t * y, int n);
double ggml_det_soft_max_f32(int n, float * y, const float * x, float max, float sink_exp);
float  ggml_det_soft_max_inv(double sum);
#ifdef __cplusplus
}
#endif
