// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// metal_ops_gate.mm — runs det_soft_ops.h's software float ops INSIDE a Metal kernel and compares
// every result bit-for-bit with the same header compiled for the host. Isolates GPU-side
// miscompiles / semantic differences (shift widths, clz, mulhi) from kernel-level bugs.
//   clang++ -std=c++17 -O2 -ObjC++ -fobjc-arc -framework Metal -framework Foundation \
//       -I <dir with det_soft.h det_soft_ops.h> metal_ops_gate.mm -o metal_ops_gate && ./metal_ops_gate [n]
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "det_soft_ops.h"
#include "bp8_exact_soft.h"

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t r64() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng * 0x2545F4914F6CDD1Dull; }
static uint32_t rbits(int cls) {
    switch (cls % 6) {
        case 0: return (uint32_t) r64();                                            // any
        case 1: return (uint32_t) (r64() & 0x807FFFFFu);                            // subnormal / zero
        case 2: return (uint32_t) (r64() & 0x80FFFFFFu) | ((uint32_t) (60 + r64() % 12) << 23);   // near the subnormal edge
        case 3: { float f = ((int) (r64() % 200001) - 100000) / 1000.0f; uint32_t b; memcpy(&b, &f, 4); return b; }
        case 4: { float f = ((int) (r64() % 2001) - 1000) / 37.0f; uint32_t b; memcpy(&b, &f, 4); return b; }
        default: return (uint32_t) (r64() & 0x80FFFFFFu) | ((uint32_t) (100 + r64() % 55) << 23);  // ordinary magnitudes
    }
}
static std::string slurp(const char * p) { std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str(); }

#define NOPS 10
static const char * opnames[NOPS] = { "fmul", "fadd", "fsub", "fdiv", "fsqrt", "expf", "tanhf", "fmax", "f16->f32", "siluf" };
static uint32_t host_op(int op, uint32_t a, uint32_t b) {
    switch (op) {
        case 0: return dso_fmul_bits(a, b);
        case 1: return dso_fadd_bits(a, b);
        case 2: return dso_fsub_bits(a, b);
        case 3: return dso_fdiv_bits(a, b);
        case 4: return dso_fsqrt_bits(a);
        case 5: return dso_expf_bits(a);
        case 6: return dso_tanhf_bits(a);
        case 7: return dso_fmax_bits(a, b);
        case 8: return dso_f16_to_f32bits(a & 0xFFFFu);
        case 9: return DSO_F2B(dso_siluf(DSO_B2F(a)));
    }
    return 0;
}

int main(int argc, char ** argv) {
    const long n = argc > 1 ? atol(argv[1]) : 1000000;
    std::string src = "#include <metal_stdlib>\nusing namespace metal;\n";
    src += slurp("det_soft.h");
    src += slurp("det_soft_ops.h");
    src += slurp("bp8_exact_soft.h");
    src += R"(
kernel void gate(device const uint * a, device const uint * b, device uint * out, constant uint & n, uint tid [[thread_position_in_grid]]) {
    if (tid >= n) return;
    const uint x = a[tid], y = b[tid];
    out[0 * n + tid] = dso_fmul_bits(x, y);
    out[1 * n + tid] = dso_fadd_bits(x, y);
    out[2 * n + tid] = dso_fsub_bits(x, y);
    out[3 * n + tid] = dso_fdiv_bits(x, y);
    out[4 * n + tid] = dso_fsqrt_bits(x);
    out[5 * n + tid] = dso_expf_bits(x);
    out[6 * n + tid] = dso_tanhf_bits(x);
    out[7 * n + tid] = dso_fmax_bits(x, y);
    out[8 * n + tid] = dso_f16_to_f32bits(x & 0xFFFFu);
    out[9 * n + tid] = as_type<uint>(dso_siluf(as_type<float>(x)));
}
kernel void gate_q256(device const uint * q, device uint * out, constant uint & n, uint tid [[thread_position_in_grid]]) {
    if (tid >= n) return;
    uint qq[8]; for (int i = 0; i < 8; i++) qq[i] = q[tid * 8 + i];
    out[2 * tid + 0] = bp8_q256_to_f32bits(qq);
    out[2 * tid + 1] = bp8_q256_to_f32bits_fast(qq);
}
)";
    // the header's includes are inlined above
    size_t p; while ((p = src.find("#include \"det_soft.h\"")) != std::string::npos) src.replace(p, 21, "");
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        MTLCompileOptions * opts = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) { opts.mathMode = MTLMathModeSafe; } else { [opts setFastMathEnabled:NO]; }
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:opts error:&err];
        if (!lib) { printf("compile error: %s\n", [[err description] UTF8String]); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"gate"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { printf("pso error: %s\n", [[err description] UTF8String]); return 2; }
        std::vector<uint32_t> A(n), B(n);
        for (long i = 0; i < n; i++) { A[i] = rbits((int) (i % 6)); B[i] = rbits((int) ((i / 6) % 6)); }
        id<MTLBuffer> ba = [dev newBufferWithBytes:A.data() length:n * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bb = [dev newBufferWithBytes:B.data() length:n * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo = [dev newBufferWithLength:(size_t) NOPS * n * 4 options:MTLResourceStorageModeShared];
        uint32_t nn = (uint32_t) n;
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:ba offset:0 atIndex:0]; [enc setBuffer:bb offset:0 atIndex:1]; [enc setBuffer:bo offset:0 atIndex:2];
        [enc setBytes:&nn length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) { printf("command buffer status %ld\n", (long) cb.status); return 2; }
        const uint32_t * O = (const uint32_t *) bo.contents;
        long total_bad = 0;
        for (int op = 0; op < NOPS; op++) {
            long bad = 0;
            for (long i = 0; i < n; i++) {
                const uint32_t h = host_op(op, A[i], B[i]), g = O[(size_t) op * n + i];
                const bool both_nan = dso_f_is_nan(h) && dso_f_is_nan(g);
                if (h != g && !both_nan) { if (bad < 5) printf("  %-8s a=%08x b=%08x host=%08x gpu=%08x\n", opnames[op], A[i], B[i], h, g); bad++; }
            }
            printf("METAL SOFT-OP %-8s %ld/%ld mismatches -> %s\n", opnames[op], bad, n, bad ? "FAIL" : "PASS");
            total_bad += bad;
        }
        // q256 readout: the integer readout vs the Horner, both inside the kernel, and both vs the host
        {
            const long nq = n / 4;
            std::vector<uint32_t> Q((size_t) nq * 8);
            for (long i = 0; i < nq; i++) {
                uint32_t * q = &Q[(size_t) i * 8]; memset(q, 0, 32);
                const int cls = (int) (i % 4);
                if (cls == 0) { for (int w = 0; w < 8; w++) q[w] = (uint32_t) r64(); }
                else if (cls == 1) { int top = (int) (r64() % 8); for (int w = 0; w <= top; w++) q[w] = (uint32_t) r64(); q[top] >>= (r64() % 32); }
                else if (cls == 2) { for (int w = 0; w < 4; w++) q[w] = (uint32_t) r64(); if (r64() & 1) { uint64_t c = 1; for (int w = 0; w < 8; w++) { uint64_t v = (uint64_t) (~q[w]) + c; q[w] = (uint32_t) v; c = v >> 32; } } }
                else { q[1] = (uint32_t) (r64() & 0x1FFFFF); q[0] = (uint32_t) r64(); }
            }
            id<MTLFunction> fq = [lib newFunctionWithName:@"gate_q256"];
            id<MTLComputePipelineState> pq = [dev newComputePipelineStateWithFunction:fq error:&err];
            id<MTLBuffer> bq = [dev newBufferWithBytes:Q.data() length:Q.size() * 4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bq_out = [dev newBufferWithLength:(size_t) nq * 8 options:MTLResourceStorageModeShared];
            uint32_t nnq = (uint32_t) nq;
            id<MTLCommandBuffer> cb2 = [q commandBuffer];
            id<MTLComputeCommandEncoder> e2 = [cb2 computeCommandEncoder];
            [e2 setComputePipelineState:pq];
            [e2 setBuffer:bq offset:0 atIndex:0]; [e2 setBuffer:bq_out offset:0 atIndex:1]; [e2 setBytes:&nnq length:4 atIndex:2];
            [e2 dispatchThreads:MTLSizeMake(nq, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e2 endEncoding]; [cb2 commit]; [cb2 waitUntilCompleted];
            const uint32_t * R = (const uint32_t *) bq_out.contents;
            long bad = 0;
            for (long i = 0; i < nq; i++) {
                const uint32_t h = bp8_q256_to_f32bits(&Q[(size_t) i * 8]);
                if (R[2 * i] != h || R[2 * i + 1] != h) { if (bad < 5) printf("  q256 i=%ld host=%08x gpu_horner=%08x gpu_fast=%08x\n", i, h, R[2 * i], R[2 * i + 1]); bad++; }
            }
            printf("METAL Q256 READOUT %ld/%ld mismatches -> %s\n", bad, nq, bad ? "FAIL" : "PASS");
            total_bad += bad;
        }
        printf("METAL SOFT-OP GATE: %s\n", total_bad ? "FAIL" : "PASS");
        return total_bad ? 1 : 0;
    }
}
