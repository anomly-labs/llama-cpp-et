// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
#pragma once

#include "common.cuh"

// exact-quire b-posit8 W8A8 matmul (src0 bposit8 weights, src1 f32 activations quantised on
// device with the reference quantiser, dst f32). Bit-identical to ggml_vec_dot_bposit8_bposit8.
void ggml_cuda_mul_mat_bposit8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
