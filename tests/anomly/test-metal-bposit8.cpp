// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Exactness gate for the Metal b-posit8 kernel: MUL_MAT(bposit8 W [K x N], f32 X [K x M]) on the
// GPU backend vs the CPU backend, compared bit for bit. Build (macOS, from the repo root):
//   clang++ -std=c++17 -O2 -I ggml/include tests/anomly/test-metal-bposit8.cpp \
//     -L build-metal/bin -lggml -lggml-base -lggml-cpu -lggml-metal -Wl,-rpath,build-metal/bin -o build-metal/bin/test-metal-bposit8
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdint>

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t r64() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng * 0x2545F4914F6CDD1Dull; }
static float rf() { int k = r64() % 5; if (k == 4) { uint32_t b = (uint32_t) (r64() & 0x807FFFFFu) | ((r64() % 3 == 0) ? 0u : ((uint32_t) (1 + r64() % 40) << 23)); float f; memcpy(&f, &b, 4); return f; } if (k == 0) return ((int)(r64() % 20001) - 10000) / 1000.0f; if (k == 1) return ldexpf(1.0f, (int)(r64() % 40) - 20) * ((r64() & 1) ? -1 : 1); if (k == 2) return (r64() % 5 == 0) ? 0.0f : ((int)(r64() % 7) - 3) * 0.25f; return (float)((double)(int64_t)(r64() >> 40) / 8388608.0) * ((r64() & 1) ? -1 : 1); }

static std::vector<float> run(ggml_backend_t be, int K, int N, int M, const std::vector<uint8_t> & wq, const std::vector<float> & x) {
    ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_BPOSIT8, K, N);
    ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * y = ggml_mul_mat(ctx, w, xt);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(xt, x.data(), 0, x.size() * 4);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_graph_compute(be, gf);
    std::vector<float> out((size_t) N * M);
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * 4);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    return out;
}

// generic op gate: build a one-op graph on both backends and compare bit for bit
static long gate_op(ggml_backend_t cpu, ggml_backend_t gpu, const char * label, int variant, int trials) {
    long bad = 0, total = 0;
    for (int t = 0; t < trials; t++) {
        std::vector<float> outs[2];
        for (int be = 0; be < 2; be++) {
            ggml_init_params ip = { 64 * 1024 * 1024, nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            const int ne0 = 32 * (1 + (t % 24)), ne1 = 1 + (t % 7), ne2 = 1 + (t % 3);
            ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
            ggml_tensor * y = nullptr; ggml_tensor * w = nullptr; ggml_tensor * m = nullptr; ggml_tensor * pos = nullptr; ggml_tensor * g = nullptr; ggml_tensor * ins_extra = nullptr; ggml_tensor * ins_rows = nullptr;
            rng = 0xABCDEF1234567ull + t;                       // same data on both backends
            std::vector<float> xv((size_t) ne0 * ne1 * ne2); for (auto & v : xv) v = rf();
            if (variant == 0) { y = ggml_rms_norm(ctx, x, 1e-5f); }
            if (variant == 1) { w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0); y = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w); }
            if (variant == 2) { m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ne0, ne1); y = ggml_soft_max_ext(ctx, x, m, 0.125f, 0.0f); }
            if (variant == 3) { y = ggml_soft_max_ext(ctx, x, nullptr, 1.0f, 0.0f); }
            if (variant == 4 || variant == 5) { pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne2); y = ggml_rope_ext(ctx, x, pos, nullptr, ne0 / 2 * 2 > 64 ? 64 : ne0 / 2 * 2, variant == 5 ? GGML_ROPE_TYPE_NEOX : 0, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f); }
            if (variant == 6) { g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2); y = ggml_swiglu_split(ctx, x, g); }
            if (variant == 7) { y = ggml_silu(ctx, x); }
            if (variant == 8) { y = ggml_gelu(ctx, x); }
            if (variant == 11) { y = ggml_add(ctx, x, ggml_scale(ctx, x, 0.37f)); }          // hardware f32 add / scale on Metal
            if (variant == 12) { ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2); ins_extra = g; y = ggml_mul(ctx, x, g); }
            if (variant == 13) { y = ggml_cpy(ctx, x, ggml_new_tensor_3d(ctx, GGML_TYPE_F16, ne0, ne1, ne2)); }   // f32 -> f16 (KV-cache write path)
            if (variant == 14) { y = ggml_cpy(ctx, ggml_cpy(ctx, x, ggml_new_tensor_3d(ctx, GGML_TYPE_F16, ne0, ne1, ne2)), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2)); }   // f32 -> f16 -> f32
            if (variant == 15) {                                                                  // set_rows f32 -> f16 (the KV-cache write op)
                ggml_tensor * dst16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ne0, ne1 * ne2);
                ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, ne1 * ne2); ins_rows = rows;
                y = ggml_set_rows(ctx, dst16, ggml_reshape_2d(ctx, x, ne0, ne1 * ne2), rows);
            }
            if (variant == 16) { ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2); ins_extra = g; y = ggml_sub(ctx, x, g); }
            if (variant == 17) { ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2); ins_extra = g; y = ggml_div(ctx, x, g); }
            if (variant == 18) { ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2); ins_extra = g; y = ggml_add(ctx, ggml_add(ctx, ggml_add(ctx, x, g), g), g); }   // fused ADD chain
            if (variant == 19) { ggml_tensor * g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0); ins_extra = g; y = ggml_add(ctx, x, g); }              // row broadcast (bias add)
            if (variant == 20) { ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, ne1, ne2); ins_extra = g; y = ggml_mul(ctx, x, g); }     // ne0 broadcast (cb)
            if (variant == 21) { y = ggml_scale(ctx, x, -0.37f); }                                                                                     // scale, no bias
            if (variant == 22) {                                   // get_rows bposit8 (embedding lookup)
                const int K = 32 * (1 + (t % 24)), N = 4 + (t % 13), R = 1 + (t % 5);
                ggml_tensor * e = ggml_new_tensor_2d(ctx, GGML_TYPE_BPOSIT8, K, N);
                ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, R);
                y = ggml_get_rows(ctx, e, ids);
                ggml_backend_buffer_t bufe = ggml_backend_alloc_ctx_tensors(ctx, be == 0 ? cpu : gpu);
                std::vector<uint8_t> eb(ggml_nbytes(e)); for (size_t i = 0; i < eb.size(); i++) eb[i] = (i % 33 == 0) ? (uint8_t) (int8_t) ((int) (r64() % 256) - 128) : (uint8_t) (r64() % 256);
                ggml_backend_tensor_set(e, eb.data(), 0, eb.size());
                std::vector<int32_t> iv(R); for (int i = 0; i < R; i++) iv[i] = (int) (r64() % N); ggml_backend_tensor_set(ids, iv.data(), 0, R * 4);
                ggml_cgraph * gfe = ggml_new_graph(ctx); ggml_build_forward_expand(gfe, y);
                ggml_backend_graph_compute(be == 0 ? cpu : gpu, gfe);
                outs[be].resize(ggml_nelements(y)); ggml_backend_tensor_get(y, outs[be].data(), 0, outs[be].size() * 4);
                ggml_backend_buffer_free(bufe); ggml_free(ctx);
                continue;
            }
            if (variant == 9 || variant == 10) {
                // MUL_MAT(f16 W [K x N x H], f32 X [K x M x H*r]); variant 10 = permuted (non-contiguous) src1 rows + transposed-view src0
                const int K = 16 * (1 + (t % 12)) + (t % 3), N = 1 + (t % 9), H = 1 + (t % 2), M = ne1, r = 1 + (t % 2);
                ggml_tensor * w16 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, N, H);
                std::vector<uint16_t> wv((size_t) K * N * H); for (auto & v : wv) v = ggml_fp32_to_fp16(rf());
                ggml_tensor * xs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, H * r);
                ggml_tensor * xin = xs;
                if (variant == 10) { xin = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, H * r, M); xs = ggml_permute(ctx, xin, 0, 2, 1, 3); }
                y = ggml_mul_mat(ctx, w16, xs);
                ggml_backend_buffer_t bufm = ggml_backend_alloc_ctx_tensors(ctx, be == 0 ? cpu : gpu);
                ggml_backend_tensor_set(w16, wv.data(), 0, wv.size() * 2);
                std::vector<float> xv2((size_t) K * M * H * r); for (auto & v : xv2) v = rf();
                ggml_backend_tensor_set(xin, xv2.data(), 0, xv2.size() * 4);
                ggml_cgraph * gfm = ggml_new_graph(ctx); ggml_build_forward_expand(gfm, y);
                ggml_backend_graph_compute(be == 0 ? cpu : gpu, gfm);
                outs[be].resize(ggml_nelements(y)); ggml_backend_tensor_get(y, outs[be].data(), 0, outs[be].size() * 4);
                ggml_backend_buffer_free(bufm); ggml_free(ctx);
                continue;
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be == 0 ? cpu : gpu);
            ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
            if (w) { std::vector<float> wv(ne0); for (auto & v : wv) v = rf(); ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * 4); }
            if (g) { std::vector<float> gv(xv.size()); for (auto & v : gv) v = rf(); ggml_backend_tensor_set(g, gv.data(), 0, gv.size() * 4); }
            if (m) { std::vector<uint16_t> mv((size_t) ne0 * ne1); for (size_t i = 0; i < mv.size(); i++) { float f = (r64() % 5 == 0) ? -INFINITY : ((int)(r64() % 9) - 4) * 0.5f; mv[i] = ggml_fp32_to_fp16(f); } ggml_backend_tensor_set(m, mv.data(), 0, mv.size() * 2); }
            if (pos) { std::vector<int32_t> pv(ne2); for (int i = 0; i < ne2; i++) pv[i] = (int) (r64() % 4096); ggml_backend_tensor_set(pos, pv.data(), 0, pv.size() * 4); }
            if (ins_extra) { std::vector<float> ev(ggml_nelements(ins_extra)); for (auto & v : ev) v = rf(); ggml_backend_tensor_set(ins_extra, ev.data(), 0, ev.size() * 4); }
            if (ins_rows) { std::vector<int64_t> rv(ne1 * ne2); for (int i = 0; i < ne1 * ne2; i++) rv[i] = (ne1 * ne2 - 1 - i); ggml_backend_tensor_set(ins_rows, rv.data(), 0, rv.size() * 8); }
            ggml_cgraph * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, y);
            ggml_backend_graph_compute(be == 0 ? cpu : gpu, gf);
            const size_t nb_out = ggml_nbytes(y);
            outs[be].resize((nb_out + 3) / 4); ggml_backend_tensor_get(y, outs[be].data(), 0, nb_out);
            ggml_backend_buffer_free(buf); ggml_free(ctx);
        }
        long nb = 0; for (size_t i = 0; i < outs[0].size(); i++) { uint32_t ua, ub; memcpy(&ua, &outs[0][i], 4); memcpy(&ub, &outs[1][i], 4); if (ua != ub) nb++; }
        if (nb && bad == 0) {
            size_t fi = 0; for (size_t i = 0; i < outs[0].size(); i++) { uint32_t ua, ub; memcpy(&ua, &outs[0][i], 4); memcpy(&ub, &outs[1][i], 4); if (ua != ub) { fi = i; break; } }
            uint32_t ua, ub; memcpy(&ua, &outs[0][fi], 4); memcpy(&ub, &outs[1][fi], 4);
            printf("  %s trial %d: first mismatch idx=%zu cpu=%08x (%a) gpu=%08x (%a) (n=%zu, %ld differ)\n", label, t, fi, ua, outs[0][fi], ub, outs[1][fi], outs[0].size(), nb);
        }
        bad += nb; total += outs[0].size();
    }
    printf("OP GATE %-14s %ld/%ld mismatches -> %s\n", label, bad, total, bad ? "FAIL" : "PASS");
    return bad;
}

// per-op timing at SmolLM2-135M decode shapes (n_embd 576, ffn 1536, head 64, 9 heads / 3 kv heads, vocab 49152)
static double bench_op(ggml_backend_t be, const char * label, int variant, int reps, int copies) {
    ggml_init_params ip = { 256 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    std::vector<ggml_tensor *> ins;
    for (int c = 0; c < copies; c++) {
        ggml_tensor * y = nullptr;
        if (variant == 0 || variant == 1 || variant == 2 || variant == 9) {    // bposit8 matvec: 576->1536, 1536->576, 576->49152 (lm_head); 9 = prefill 576->1536 x 64 tokens
            const int K = variant == 1 ? 1536 : 576, N = variant == 0 || variant == 9 ? 1536 : (variant == 1 ? 576 : 49152);
            ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_BPOSIT8, K, N);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, variant == 9 ? 64 : 1);
            ins.push_back(w); ins.push_back(x); y = ggml_mul_mat(ctx, w, x);
        }
        if (variant == 3) {                                    // KQ: k [64, n_kv=128, 3] x q [64, 1, 9]
            ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, 128, 3);
            ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 1, 9);
            ins.push_back(k); ins.push_back(q); y = ggml_mul_mat(ctx, k, q);
        }
        if (variant == 4) {                                    // KQV: v [128, 64, 3] x kq [128, 1, 9]
            ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, 64, 3);
            ggml_tensor * kq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 1, 9);
            ins.push_back(v); ins.push_back(kq); y = ggml_mul_mat(ctx, v, kq);
        }
        if (variant == 5) { ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 1, 9); ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 128, 1); ins.push_back(x); ins.push_back(m); y = ggml_soft_max_ext(ctx, x, m, 0.125f, 0.0f); }
        if (variant == 6) { ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 576, 1); ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 576); ins.push_back(x); ins.push_back(w); y = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-5f), w); }
        if (variant == 7) { ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 9, 1); ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1); ins.push_back(x); ins.push_back(pos); y = ggml_rope_ext(ctx, x, pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 8192, 100000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f); }
        if (variant == 8) { ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1536, 1); ggml_tensor * g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1536, 1); ins.push_back(x); ins.push_back(g); y = ggml_swiglu_split(ctx, x, g); }
        ggml_build_forward_expand(gf, y);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    for (ggml_tensor * t : ins) {                               // deterministic fill; bposit8 rows get small codes + scale 0
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        if (t->type == GGML_TYPE_I32) { memset(bytes.data(), 0, bytes.size()); }
        else if (t->type == GGML_TYPE_BPOSIT8) { for (size_t i = 0; i < bytes.size(); i++) bytes[i] = (i % 33 == 0) ? 0 : (uint8_t) (0x30 + (r64() % 64)); }
        else if (t->type == GGML_TYPE_F16) { for (size_t i = 0; i + 1 < bytes.size(); i += 2) { uint16_t h = ggml_fp32_to_fp16(rf() * 0.01f); memcpy(&bytes[i], &h, 2); } }
        else { for (size_t i = 0; i + 3 < bytes.size(); i += 4) { float f = rf() * 0.01f; memcpy(&bytes[i], &f, 4); } }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    // warm-up: pipeline compile + let the GPU clock ramp (DVFS makes short bursts look slow)
    { const int64_t tw = ggml_time_us(); while (ggml_time_us() - tw < 400000) ggml_backend_graph_compute(be, gf); }
    const int64_t t0 = ggml_time_us();
    int done = 0;
    while (done < reps || ggml_time_us() - t0 < 300000) { ggml_backend_graph_compute(be, gf); done++; }
    const double us = (double) (ggml_time_us() - t0) / ((double) done * copies);
    printf("BENCH %-22s %-4s %9.1f us/op\n", label, ggml_backend_name(be), us);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    return us;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        ggml_backend_load_all();
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        ggml_backend_cpu_set_n_threads(cpu, 4);
        const char * names[10] = { "bp8 mv 576->1536", "bp8 mv 1536->576", "bp8 mv 576->49152", "f16 KQ 64x128 9h", "f16 KQV 128x64 9h", "softmax 128x9", "rms_norm+mul 576", "rope neox 64x9", "swiglu 1536", "bp8 mm 576->1536 x64" };
        const int only = argc > 2 ? atoi(argv[2]) : -1;
        for (int v = 0; v < 10; v++) { if (only >= 0 && v != only) continue; const int copies = (v == 2 || v == 9) ? 4 : 32; bench_op(cpu, names[v], v, 5, copies); bench_op(gpu, names[v], v, 5, copies); }
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "ops") == 0) {
        ggml_backend_load_all();
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        const int trials = argc > 2 ? atoi(argv[2]) : 24;
        long bad = 0;
        const char * names[23] = { "rms_norm", "rms_norm+mul", "soft_max f16m", "soft_max plain", "rope norm", "rope neox", "swiglu", "silu", "gelu", "mul_mat f16", "mul_mat f16 perm", "add+scale f32", "mul f32", "cpy f32->f16", "cpy f32->f16->f32", "set_rows f32->f16", "sub f32", "div f32", "add x3 fused", "add row bcast", "mul ne0 bcast", "scale f32", "get_rows bposit8" };
        const int only = argc > 3 ? atoi(argv[3]) : -1;
        for (int v = 0; v < 23; v++) { if (only >= 0 && v != only) continue; bad += gate_op(cpu, gpu, names[v], v, trials); }
        printf("ALL OP GATES: %s\n", bad ? "FAIL" : "PASS");
        return bad ? 1 : 0;
    }
    const int trials = argc > 1 ? atoi(argv[1]) : 20;
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!cpu || !gpu) { printf("backends: cpu=%p gpu=%p\n", (void*)cpu, (void*)gpu); return 2; }
    printf("gpu backend: %s\n", ggml_backend_name(gpu));
    long bad = 0, total = 0;
    for (int t = 0; t < trials; t++) {
        const int K = 32 * (1 + (int)(r64() % 24)), N = 1 + (int)(r64() % 200), M = 1 + (int)(r64() % 9);
        std::vector<float> wf((size_t) K * N), x((size_t) K * M);
        for (auto & v : wf) v = rf();
        for (auto & v : x) v = rf();
        std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_BPOSIT8, K) * N);
        ggml_quantize_chunk(GGML_TYPE_BPOSIT8, wf.data(), wq.data(), 0, N, K, nullptr);
        auto a = run(cpu, K, N, M, wq, x);
        auto b = run(gpu, K, N, M, wq, x);
        long nb = 0; for (size_t i = 0; i < a.size(); i++) { uint32_t ua, ub; memcpy(&ua, &a[i], 4); memcpy(&ub, &b[i], 4); if (ua != ub) nb++; }
        bad += nb; total += a.size();
        printf("trial %2d K=%4d N=%3d M=%d: %ld/%zu mismatches; cpu[0]=%a gpu[0]=%a cpu[last]=%a gpu[last]=%a\n", t, K, N, M, nb, a.size(), a[0], b[0], a.back(), b.back());
    }
    printf("METAL BPOSIT8 GATE: %ld/%ld mismatches -> %s\n", bad, total, bad ? "FAIL" : "PASS");
    ggml_backend_free(cpu); ggml_backend_free(gpu);
    return bad ? 1 : 0;
}
