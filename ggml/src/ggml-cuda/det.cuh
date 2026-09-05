// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// CUDA instantiation of ggml-det: every operation is the explicit round-to-nearest
// intrinsic, so the device code performs exactly the host's IEEE operations.
#pragma once
#define DET_FN         static __device__ __forceinline__
#define DET_DMUL(a, b) __dmul_rn((a), (b))
#define DET_DADD(a, b) __dadd_rn((a), (b))
#define DET_DSUB(a, b) __dsub_rn((a), (b))
#define DET_DDIV(a, b) __ddiv_rn((a), (b))
#define DET_D2F(a)     __double2float_rn(a)
#define DET_FMUL(a, b) __fmul_rn((a), (b))
#define DET_FADD(a, b) __fadd_rn((a), (b))
#define DET_FSUB(a, b) __fsub_rn((a), (b))
#define DET_FDIV(a, b) __fdiv_rn((a), (b))
#define DET_FSQRT(a)   __fsqrt_rn(a)
#include "../ggml-det/ggml-det.h"
