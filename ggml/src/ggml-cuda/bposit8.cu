// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
//
// b-posit8 W8A8 exact-quire matmul for CUDA. Contract: every output float is bit-identical to
// the CPU kernel (ggml-cpu/quants.c ggml_vec_dot_bposit8_bposit8) and to the independent
// verifiers (invar.spotcheck, go/crverify): the activation quantiser is the reference one
// (exact integer block-scale rule + nearest-code bisect with the exact code-0 rule) and the
// dot product is a 256-bit two's-complement Kulisch sum read out once through the same
// limb-to-double loop. Nothing here depends on libm, FMA contraction, or summation order.
//
// Accumulation: each lane keeps 8 lazily-carried int64 limbs in shared memory. A product
// P*2^sh (|P| < 2^10) is split into its low 32 bits and its high part and added to limb
// sh>>5 and sh>>5 + 1; the lane limbs are warp-reduced as int64 and carry-normalised once.
// Modulo 2^256 this is exactly the CPU quire. Terms below the radix point (sh < 0) are
// floor-truncated per term, as on the CPU.

#include "bposit8.cuh"
#include "bposit8-common.cuh"
#include "det.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------------------
// quantiser tables (sorted value grid for the bisect encoder), uploaded once per device
// ---------------------------------------------------------------------------------------
__constant__ double  c_bp8_sv[255];
__constant__ uint8_t c_bp8_sc[255];

static int bp8_cmp_code_val(const void * a, const void * b) {
    const double x = bp8_code_to_double(*(const uint8_t *) a), y = bp8_code_to_double(*(const uint8_t *) b);
    return (x > y) - (x < y);
}

static void bp8_tables_upload(int device) {
    static bool uploaded[GGML_CUDA_MAX_DEVICES] = { false };
    if (uploaded[device]) return;
    uint8_t codes[255]; int n = 0;
    for (int c = 0; c < 256; c++) if (c != BP8_NAR) codes[n++] = (uint8_t) c;
    qsort(codes, n, 1, bp8_cmp_code_val);            // values are distinct: the order is total
    double sv[255]; uint8_t sc[255];
    for (int i = 0; i < n; i++) { sc[i] = codes[i]; sv[i] = bp8_code_to_double(codes[i]); }
    CUDA_CHECK(cudaMemcpyToSymbol(c_bp8_sv, sv, sizeof(sv)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bp8_sc, sc, sizeof(sc)));
    uploaded[device] = true;
}

// ---------------------------------------------------------------------------------------
// exact block scale: se = round_half_even(log2(sqrt(S/32))) with S = the EXACT sum of squares
// (640-bit integer, radix at bit 352). Mirrors ggml_bp8_scale_exp_exact (ggml-quants.c).
// ---------------------------------------------------------------------------------------
#define BP8_SS_LIMBS 20
#define BP8_SS_RADIX 352

static __device__ __forceinline__ int bp8_scale_exp_exact(const float * x) {
    uint32_t acc[BP8_SS_LIMBS];
    #pragma unroll
    for (int i = 0; i < BP8_SS_LIMBS; i++) acc[i] = 0;
    bool any = false;
    for (int j = 0; j < QK_BPOSIT8; j++) {
        const double v = (double) x[j];
        if (v == 0.0) continue;
        if (!isfinite(v)) return 0;
        any = true;
        const double p = __dmul_rn(v, v);                       // exact: 24x24 bits fit in 53
        const uint64_t bits = (uint64_t) __double_as_longlong(p);
        const uint64_t mi = (bits & 0xFFFFFFFFFFFFFull) | (1ull << 52);
        const int e2 = (int) ((bits >> 52) & 0x7FF) - 1023;     // p = mi * 2^(e2-52), p normal
        const int pos = e2 - 52 + BP8_SS_RADIX;
        const int w = pos >> 5, b = pos & 31;
        const uint64_t lo = b ? (mi << b) : mi;
        const uint64_t hi = b ? (mi >> (64 - b)) : 0ull;
        uint32_t parts[3] = { (uint32_t) lo, (uint32_t) (lo >> 32), (uint32_t) hi };
        uint64_t c = 0;
        #pragma unroll
        for (int i = 0; i < 3; i++) { const uint64_t t = (uint64_t) acc[w + i] + parts[i] + c; acc[w + i] = (uint32_t) t; c = t >> 32; }
        for (int i = w + 3; c && i < BP8_SS_LIMBS; i++) { const uint64_t t = (uint64_t) acc[i] + c; acc[i] = (uint32_t) t; c = t >> 32; }
    }
    if (!any) return 0;
    int top = -1, pop = 0;
    for (int i = BP8_SS_LIMBS - 1; i >= 0; i--) {
        if (acc[i]) { if (top < 0) top = 32 * i + 31 - __clz((int) acc[i]); pop += __popc((int) acc[i]); }
    }
    const int E = top - BP8_SS_RADIX - 5;                       // floor(log2(S/32))
    int se;
    if ((E & 1) == 0) {
        se = E / 2;
    } else {
        const int n = (E - 1) / 2;
        se = (pop == 1) ? (((n & 1) == 0) ? n : n + 1) : n + 1;
    }
    return se > 127 ? 127 : (se < -128 ? -128 : se);
}

// nearest finite code to x by value; ties to the lower code; code 0 unless a neighbour is
// STRICTLY closer than |x| (mirrors ggml_bp8_encode_nearest)
static __device__ __forceinline__ uint8_t bp8_encode_nearest(double x, const double * sv, const uint8_t * sc) {
    if (x == 0.0) return BP8_ZERO;
    int lo = 0, hi = 255;
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (sv[mid] < x) lo = mid + 1; else hi = mid; }
    int best_k = -1; double bestd = INFINITY;
    const int start = lo > 0 ? lo - 1 : lo, end = lo < 255 ? lo : lo - 1;
    for (int k = start; k <= end; k++) {
        const double d = fabs(__dsub_rn(sv[k], x));
        if (d < bestd || (d == bestd && (best_k < 0 || sc[k] < sc[best_k]))) { bestd = d; best_k = k; }
    }
    if (best_k < 0 || !(bestd < fabs(x))) return BP8_ZERO;
    return sc[best_k];
}

// one thread per 32-element block; blocks laid out contiguously as [i3][i2][i1][ib]
static __global__ void quantize_bposit8_kernel(const float * __restrict__ x, block_bposit8 * __restrict__ y,
        const int64_t nblk_row, const int64_t ne1, const int64_t ne2,
        const int64_t s1, const int64_t s2, const int64_t s3, const int64_t nblk_total) {
    __shared__ double  sv[255];
    __shared__ uint8_t sc[255];
    for (int i = threadIdx.x; i < 255; i += blockDim.x) { sv[i] = c_bp8_sv[i]; sc[i] = c_bp8_sc[i]; }
    __syncthreads();
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk_total) return;
    const int64_t ib = b % nblk_row;
    int64_t r = b / nblk_row;
    const int64_t i1 = r % ne1; r /= ne1;
    const int64_t i2 = r % ne2;
    const int64_t i3 = r / ne2;
    const float * xr = x + i3 * s3 + i2 * s2 + i1 * s1 + ib * QK_BPOSIT8;
    float v[QK_BPOSIT8];
    #pragma unroll
    for (int j = 0; j < QK_BPOSIT8; j++) v[j] = xr[j];
    block_bposit8 * yb = y + b;
    const int se = bp8_scale_exp_exact(v);
    yb->scale_exp = (int8_t) se;
    const double inv = ldexp(1.0, -se);                          // exact
    #pragma unroll
    for (int j = 0; j < QK_BPOSIT8; j++) yb->qs[j] = bp8_encode_nearest(__dmul_rn((double) v[j], inv), sv, sc);
}


#define BP8_WARPS_PER_BLOCK 4

// ---------------------------------------------------------------------------------------
// shared exact-accumulation machinery: 8 lazily-carried int64 limbs per lane (shared mem),
// exact int64 warp reduction, carry normalisation mod 2^256, the ggml-det readout loop.
// ---------------------------------------------------------------------------------------
static __device__ __forceinline__ void exact_lane_add(int64_t * a, int64_t P, int sh) {
    if (sh >= 0) {
        const int w = sh >> 5, bits = sh & 31;
        if (w < 8) {
            const int64_t V = P << bits;
            a[w] += (int64_t) (uint32_t) V;
            if (w + 1 < 8) a[w + 1] += (V >> 32);
        }
    } else {
        const int rs = -sh;
        const int64_t V = rs >= 63 ? (P < 0 ? -1 : 0) : (P >> rs);
        a[0] += V;
    }
}

// all 32 lanes must call; the result is valid in every lane
static __device__ __forceinline__ float exact_warp_readout(const int64_t * a) {
    uint32_t q[8];
    int64_t carry = 0;
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        int64_t v = a[w];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
        v += carry;
        q[w] = (uint32_t) v;
        carry = v >> 32;
    }
    return DET_D2F(det_q256_to_double(q));
}

// ---------------------------------------------------------------------------------------
// exact f16 x f16 matmul (attention KQ and KQV under the exact profile): src0 f16 rows,
// src1 f32 converted to f16 (round-to-nearest, as the CPU backend does), one warp per output
// ---------------------------------------------------------------------------------------
static __global__ void f32_to_f16_rows_kernel(const float * __restrict__ x, uint16_t * __restrict__ y,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t s1, const int64_t s2, const int64_t s3, const int64_t total) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t i0 = i % ne0; int64_t r = i / ne0;
    const int64_t i1 = r % ne1; r /= ne1;
    const int64_t i2 = r % ne2; const int64_t i3 = r / ne2;
    y[i] = __half_as_ushort(__float2half_rn(x[i3 * s3 + i2 * s2 + i1 * s1 + i0]));
}

static __global__ void __launch_bounds__(32 * BP8_WARPS_PER_BLOCK)
mul_mat_f16_exact_kernel(const uint16_t * __restrict__ x, const uint16_t * __restrict__ y, float * __restrict__ dst,
        const int n, const int64_t ne01, const int64_t ne12,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s11, const int64_t s12, const int64_t s13,
        const int64_t sd1, const int64_t sd2, const int64_t sd3,
        const int r2, const int r3) {
    __shared__ int64_t acc[32 * BP8_WARPS_PER_BLOCK][8];
    int64_t * a = acc[threadIdx.x];
    #pragma unroll
    for (int w = 0; w < 8; w++) a[w] = 0;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t row = (int64_t) blockIdx.x * BP8_WARPS_PER_BLOCK + warp;
    if (row >= ne01) return;
    const int64_t i1 = blockIdx.y;
    const int64_t i2 = blockIdx.z % ne12, i3 = blockIdx.z / ne12;
    const uint16_t * xr = x + (i3 / r3) * s03 + (i2 / r2) * s02 + row * s01;
    const uint16_t * yr = y + i3 * s13 + i2 * s12 + i1 * s11;
    for (int k = lane; k < n; k += 32) {
        int Mx, Ex, My, Ey;
        det_f16_to_ME(xr[k], &Mx, &Ex);
        det_f16_to_ME(yr[k], &My, &Ey);
        const int64_t P = (int64_t) Mx * (int64_t) My;
        if (P != 0) exact_lane_add(a, P, Ex + Ey + DET_QFRAC);
    }
    const float out = exact_warp_readout(a);
    if (lane != 0) return;
    dst[i3 * sd3 + i2 * sd2 + i1 * sd1 + row] = out;
}

void ggml_cuda_mul_mat_f16_exact(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne10 == ne00);
    GGML_ASSERT(nb00 == sizeof(uint16_t) && nb10 == sizeof(float) && nb0 == sizeof(float));
    GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);
    GGML_ASSERT(ne11 <= 65535 && ne12 * ne13 <= 65535);
    cudaStream_t stream = ctx.stream();
    const int64_t total = ne10 * ne11 * ne12 * ne13;
    ggml_cuda_pool_alloc<char> yh(ctx.pool(), (size_t) total * sizeof(uint16_t));
    uint16_t * yh_d = (uint16_t *) yh.get();
    {
        const int nthreads = 256;
        const dim3 grid((unsigned) ((total + nthreads - 1) / nthreads));
        f32_to_f16_rows_kernel<<<grid, nthreads, 0, stream>>>((const float *) src1->data, yh_d, ne10, ne11, ne12,
            nb11 / sizeof(float), nb12 / sizeof(float), nb13 / sizeof(float), total);
    }
    {
        const int64_t s01 = nb01 / sizeof(uint16_t), s02 = nb02 / sizeof(uint16_t), s03 = nb03 / sizeof(uint16_t);
        const int64_t s11 = ne10, s12 = ne10 * ne11, s13 = ne10 * ne11 * ne12;
        const int64_t sd1 = nb1 / sizeof(float), sd2 = nb2 / sizeof(float), sd3 = nb3 / sizeof(float);
        const dim3 grid((unsigned) ((ne01 + BP8_WARPS_PER_BLOCK - 1) / BP8_WARPS_PER_BLOCK), (unsigned) ne11, (unsigned) (ne12 * ne13));
        mul_mat_f16_exact_kernel<<<grid, 32 * BP8_WARPS_PER_BLOCK, 0, stream>>>(
            (const uint16_t *) src0->data, yh_d, (float *) dst->data, (int) ne00, ne01, ne12,
            s01, s02, s03, s11, s12, s13, sd1, sd2, sd3, (int) (ne12 / ne02), (int) (ne13 / ne03));
    }
}

// ---------------------------------------------------------------------------------------
// exact dot: one warp per output element
// ---------------------------------------------------------------------------------------

static __global__ void __launch_bounds__(32 * BP8_WARPS_PER_BLOCK)
mul_mat_bposit8_kernel(const block_bposit8 * __restrict__ x, const block_bposit8 * __restrict__ y, float * __restrict__ dst,
        const int nblk, const int64_t ne01, const int64_t ne12,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s11, const int64_t s12, const int64_t s13,
        const int64_t sd1, const int64_t sd2, const int64_t sd3,
        const int r2, const int r3) {
    __shared__ int8_t  sM[256];
    __shared__ int8_t  sE[256];
    __shared__ int64_t acc[32 * BP8_WARPS_PER_BLOCK][8];
    for (int c = threadIdx.x; c < 256; c += blockDim.x) { int M, E; bp8_code_to_ME((uint8_t) c, &M, &E); sM[c] = (int8_t) M; sE[c] = (int8_t) E; }
    int64_t * a = acc[threadIdx.x];
    #pragma unroll
    for (int w = 0; w < 8; w++) a[w] = 0;
    __syncthreads();

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t row = (int64_t) blockIdx.x * BP8_WARPS_PER_BLOCK + warp;
    if (row >= ne01) return;                                      // whole warp leaves together
    const int64_t i1 = blockIdx.y;
    const int64_t i2 = blockIdx.z % ne12, i3 = blockIdx.z / ne12;
    const block_bposit8 * xr = x + (i3 / r3) * s03 + (i2 / r2) * s02 + row * s01;
    const block_bposit8 * yr = y + i3 * s13 + i2 * s12 + i1 * s11;

    for (int ib = lane; ib < nblk; ib += 32) {
        const uint8_t * xq = xr[ib].qs;
        const uint8_t * yq = yr[ib].qs;
        const int se = (int) xr[ib].scale_exp + (int) yr[ib].scale_exp + BP8_QFRAC;
        #pragma unroll 8
        for (int j = 0; j < QK_BPOSIT8; j++) {
            const uint8_t cx = xq[j], cy = yq[j];
            const int P = (int) sM[cx] * (int) sM[cy];
            if (P == 0) continue;
            exact_lane_add(a, P, (int) sE[cx] + (int) sE[cy] + se);
        }
    }

    const float out = exact_warp_readout(a);
    if (lane != 0) return;
    dst[i3 * sd3 + i2 * sd2 + i1 * sd1 + row] = out;
}

void ggml_cuda_mul_mat_bposit8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_BPOSIT8);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_BPOSIT8 == 0);
    GGML_ASSERT(ne10 == ne00);
    GGML_ASSERT(nb00 == sizeof(block_bposit8));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);
    GGML_ASSERT(ne11 <= 65535 && ne12 * ne13 <= 65535);

    bp8_tables_upload(ctx.device);
    cudaStream_t stream = ctx.stream();

    const int64_t nblk       = ne00 / QK_BPOSIT8;
    const int64_t nblk_total = nblk * ne11 * ne12 * ne13;
    ggml_cuda_pool_alloc<char> yq(ctx.pool(), (size_t) nblk_total * sizeof(block_bposit8));
    block_bposit8 * yq_d = (block_bposit8 *) yq.get();
    {
        const int64_t s11 = nb11 / sizeof(float), s12 = nb12 / sizeof(float), s13 = nb13 / sizeof(float);
        const int nthreads = 128;
        const dim3 grid((unsigned) ((nblk_total + nthreads - 1) / nthreads));
        quantize_bposit8_kernel<<<grid, nthreads, 0, stream>>>((const float *) src1->data, yq_d, nblk, ne11, ne12, s11, s12, s13, nblk_total);
    }
    {
        const int64_t s01 = nb01 / sizeof(block_bposit8), s02 = nb02 / sizeof(block_bposit8), s03 = nb03 / sizeof(block_bposit8);
        const int64_t s11 = nblk, s12 = nblk * ne11, s13 = nblk * ne11 * ne12;
        const int64_t sd1 = nb1 / sizeof(float), sd2 = nb2 / sizeof(float), sd3 = nb3 / sizeof(float);
        const dim3 grid((unsigned) ((ne01 + BP8_WARPS_PER_BLOCK - 1) / BP8_WARPS_PER_BLOCK), (unsigned) ne11, (unsigned) (ne12 * ne13));
        mul_mat_bposit8_kernel<<<grid, 32 * BP8_WARPS_PER_BLOCK, 0, stream>>>(
            (const block_bposit8 *) src0->data, yq_d, (float *) dst->data,
            (int) nblk, ne01, ne12, s01, s02, s03, s11, s12, s13, sd1, sd2, sd3, (int) (ne12 / ne02), (int) (ne13 / ne03));
    }
}
