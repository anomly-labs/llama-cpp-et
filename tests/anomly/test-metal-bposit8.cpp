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
static float rf() { int k = r64() % 4; if (k == 0) return ((int)(r64() % 20001) - 10000) / 1000.0f; if (k == 1) return ldexpf(1.0f, (int)(r64() % 40) - 20) * ((r64() & 1) ? -1 : 1); if (k == 2) return (r64() % 5 == 0) ? 0.0f : ((int)(r64() % 7) - 3) * 0.25f; return (float)((double)(int64_t)(r64() >> 40) / 8388608.0) * ((r64() & 1) ? -1 : 1); }

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

int main(int argc, char ** argv) {
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
