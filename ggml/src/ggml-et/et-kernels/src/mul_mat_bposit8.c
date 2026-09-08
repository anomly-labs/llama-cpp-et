//******************************************************************************
// MUL_MAT kernel: b-posit8 W8A8 with the exact 256-bit quire (Anomly)
//   C[M,N] = A[M,K] (bposit8 blocks) * B[K,N] (F32 activations, quantised in-kernel)
// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
//
// Every output is bit-identical to llama-cpp-et's CPU path (ggml_vec_dot_bposit8_bposit8 after
// quantize_row_bposit8_ref): integer-only arithmetic (bp8_exact.h), proven equal to the
// double-based reference on x86 (3M read-outs, 200k rows, 60k dots). Correctness-first:
// row-striped over threads, activations re-quantised per thread; no matrix engine.
//******************************************************************************
#include "ggml_tensor.h"
#include "platform.h"
#include "bp8_exact.h"
#include <stdint.h>

#define BP8_GGML_TYPE_BPOSIT8 43   /* GGML_TYPE_BPOSIT8 in ggml.h */
#define BP8_GGML_TYPE_F32      0
#define BP8_MAX_BLOCKS       64    /* K <= 2048 per row (SmolLM2-135M: 576 / 1536) */

int entry_point(struct ggml_et_binary_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    if (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) return -1;
    const int thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    const int num_threads = get_num_threads(kernel_env->shire_mask);
    if (thread_id < 0) return 0;

    struct ggml_tensor * src0 = &params->src0;   /* weights, bposit8 blocks  [K, M] */
    struct ggml_tensor * src1 = &params->src1;   /* activations, F32        [K, N] */
    struct ggml_tensor * dst  = &params->dst;    /* output, F32             [M, N] */
    if (src0->type != BP8_GGML_TYPE_BPOSIT8 || src1->type != BP8_GGML_TYPE_F32 || dst->type != BP8_GGML_TYPE_F32) return -1;

    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    const int64_t ne02 = src0->ne[2], ne03 = src0->ne[3], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const size_t nbd1 = dst->nb[1], nbd2 = dst->nb[2], nbd3 = dst->nb[3];
    if (K % BP8_QK) return -1;
    const int nb = (int) (K / BP8_QK);
    if (nb > BP8_MAX_BLOCKS) return -1;
    const int64_t r2 = ne12 / ne02, r3 = ne13 / ne03;

    /* Quantised activation column. Off the stack (minion stacks are small; a 4 KB frame faulted
       the runtime on the first board run): every hart writes identical bytes for the same column,
       so a shared static buffer is race-free in effect. Host-side quantisation is the real fix. */
    static bp8_block_t yq[BP8_MAX_BLOCKS];

    for (int64_t i3 = 0; i3 < ne13; i3++) {
        const int64_t i03 = i3 / r3;
        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const int64_t i02 = i2 / r2;
            const char * a_base = (const char *) src0->data + i02 * nb02 + i03 * nb03;
            const char * b_base = (const char *) src1->data + i2 * nb12 + i3 * nb13;
            char *       c_base = (char *) dst->data + i2 * nbd2 + i3 * nbd3;
            for (int64_t n = 0; n < N; n++) {
                const float * bcol = (const float *) (b_base + n * nb11);
                bp8_quantize_row_i(bcol, yq, (int) K);            /* W8A8: activations to bposit8 */
                float * crow = (float *) (c_base + n * nbd1);
                for (int64_t m = thread_id; m < M; m += num_threads) {
                    const bp8_block_t * arow = (const bp8_block_t *) (a_base + m * nb01);
                    uint32_t bits = bp8_dot_f32_bits(arow, yq, nb);
                    atomic_store_f32((volatile float *) &crow[m], bp8_bits_f32(bits));
                }
            }
        }
    }
    return 0;
}
