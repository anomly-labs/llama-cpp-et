// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Unit test of the Metal b-posit8 arithmetic (ggml-metal-bposit8.h) outside ggml: the same header
// compiled as C++ on the host is the oracle (it is gated against libggml-cpu on Linux).
//   clang++ -std=c++17 -O2 -fobjc-arc -I research/metal-exact tests/anomly/metal-bp8-unit.mm -framework Metal -framework Foundation -o /tmp/mbp8 && /tmp/mbp8
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include "det_soft.h"
#include "bp8_exact_soft.h"

static uint64_t rng = 0x1234567887654321ull;
static uint64_t r64() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng * 0x2545F4914F6CDD1Dull; }
static float rf() { int k = r64() % 4; if (k == 0) return ((int)(r64() % 20001) - 10000) / 1000.0f; if (k == 1) return ldexpf(1.0f, (int)(r64() % 40) - 20) * ((r64() & 1) ? -1 : 1); if (k == 2) return (r64() % 5 == 0) ? 0.0f : ((int)(r64() % 7) - 3) * 0.25f; return (float)((double)(int64_t)(r64() >> 40) / 8388608.0) * ((r64() & 1) ? -1 : 1); }

int main(int argc, char ** argv) {
    const char * hdr_path = argc > 1 ? argv[1] : "ggml/src/ggml-metal/ggml-metal-bposit8.h";
    std::ifstream f(hdr_path); std::stringstream ss; ss << f.rdbuf(); std::string hdr = ss.str();
    if (hdr.empty()) { printf("cannot read %s\n", hdr_path); return 2; }
    std::string src =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "typedef struct { char scale_exp; uchar qs[32]; } block_bposit8;\n"
        "typedef struct { int64_t nblk_row, ne1, ne2, s1, s2, s3, nblk_total; } ggml_metal_kargs_bp8_quant;\n"
        "typedef struct { int32_t nblk; int64_t ne01, ne12, s01, s02, s03, s11, s12, s13, sd1, sd2, sd3; int32_t r2, r3; } ggml_metal_kargs_bp8_mv;\n"
        + hdr +
        "kernel void t_quant(device const uint * xb, device int * se, device uint * qs, uint tid [[thread_position_in_grid]]) {\n"
        "  uint x[32]; for (int j = 0; j < 32; j++) x[j] = xb[tid*32+j]; int s; uint q[32];\n"
        "  bp8_quantize_block_soft(x, &s, q, bp8_tab_sv, bp8_tab_sc); se[tid] = s; for (int j = 0; j < 32; j++) qs[tid*32+j] = q[j]; }\n"
        "kernel void t_readout(device const uint * q8, device uint * out, uint tid [[thread_position_in_grid]]) {\n"
        "  uint q[8]; for (int i = 0; i < 8; i++) q[i] = q8[tid*8+i]; out[tid] = bp8_q256_to_f32bits(q); }\n"
        "kernel void t_scale(device const uint * xb, device int * se, uint tid [[thread_position_in_grid]]) {\n"
        "  uint x[32]; for (int j = 0; j < 32; j++) x[j] = xb[tid*32+j]; se[tid] = bp8_scale_exp_exact_bits(x); }\n"
        "kernel void t_f2d(device const uint * fb, device ulong * out, uint tid [[thread_position_in_grid]]) { out[tid] = ds_from_f32bits(fb[tid]); }\n"
        "kernel void t_enc(device const ulong * xd, device uint * out, uint tid [[thread_position_in_grid]]) { out[tid] = bp8_encode_nearest_soft(xd[tid], bp8_tab_sv, bp8_tab_sc); }\n";
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError * err = nil;
    MTLCompileOptions * opts = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) opts.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:opts error:&err];
    if (!lib) { printf("compile error: %s\n", [[err description] UTF8String]); return 2; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    auto run = [&](const char * kname, std::vector<id<MTLBuffer>> bufs, int n) {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kname]];
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) { printf("pipeline error %s: %s\n", kname, [[err description] UTF8String]); exit(2); }
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        for (size_t i = 0; i < bufs.size(); i++) [enc setBuffer:bufs[i] offset:0 atIndex:i];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    };
    auto mkbuf = [&](size_t bytes, const void * data) { id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; if (data) memcpy(b.contents, data, bytes); return b; };
    // host tables
    ds_u64 sv[255]; ds_u32 sc[255]; { uint8_t codes[255]; int m = 0; for (int c = 0; c < 256; c++) if (c != BP8_NAR) codes[m++] = c;
      for (int i = 1; i < 255; i++) { uint8_t k = codes[i]; int j = i - 1; while (j >= 0 && ds_lt(bp8_code_to_dbits(k), bp8_code_to_dbits(codes[j]))) { codes[j+1] = codes[j]; j--; } codes[j+1] = k; }
      for (int i = 0; i < 255; i++) { sc[i] = codes[i]; sv[i] = bp8_code_to_dbits(codes[i]); } }
    const int NB = 2048;
    std::vector<uint32_t> xb(NB * 32); for (auto & v : xb) { float f = rf(); memcpy(&v, &f, 4); }
    // 1. f32 -> double bits
    { auto bi = mkbuf(xb.size() * 4, xb.data()); auto bo = mkbuf(xb.size() * 8, nullptr); run("t_f2d", {bi, bo}, (int) xb.size());
      long bad = 0; const ds_u64 * o = (const ds_u64 *) bo.contents; for (size_t i = 0; i < xb.size(); i++) if (o[i] != ds_from_f32bits(xb[i])) { if (bad < 3) printf("f2d mismatch %zu: f=%08x gpu=%016llx host=%016llx\n", i, xb[i], (unsigned long long) o[i], (unsigned long long) ds_from_f32bits(xb[i])); bad++; }
      printf("f2d: %ld mismatches / %zu\n", bad, xb.size()); }
    // 2. scale exponent
    { auto bi = mkbuf(xb.size() * 4, xb.data()); auto bo = mkbuf(NB * 4, nullptr); run("t_scale", {bi, bo}, NB);
      long bad = 0; const int * o = (const int *) bo.contents; for (int b = 0; b < NB; b++) { int h = bp8_scale_exp_exact_bits(&xb[b*32]); if (o[b] != h) { if (bad < 3) printf("scale mismatch blk %d: gpu %d host %d\n", b, o[b], h); bad++; } }
      printf("scale_exp: %ld mismatches / %d\n", bad, NB); }
    // 3. encoder on scaled values
    { std::vector<ds_u64> xd(xb.size()); for (size_t i = 0; i < xb.size(); i++) xd[i] = ds_ldexp(ds_from_f32bits(xb[i]), -bp8_scale_exp_exact_bits(&xb[(i/32)*32]));
      auto bi = mkbuf(xd.size() * 8, xd.data()); auto bo = mkbuf(xd.size() * 4, nullptr); run("t_enc", {bi, bo}, (int) xd.size());
      long bad = 0; const uint32_t * o = (const uint32_t *) bo.contents; for (size_t i = 0; i < xd.size(); i++) { uint32_t h = bp8_encode_nearest_soft(xd[i], sv, sc); if (o[i] != h) { if (bad < 3) printf("enc mismatch %zu: x=%016llx gpu %u host %u\n", i, (unsigned long long) xd[i], o[i], h); bad++; } }
      printf("encode: %ld mismatches / %zu\n", bad, xd.size()); }
    // 4. whole block quantiser
    { auto bi = mkbuf(xb.size() * 4, xb.data()); auto bse = mkbuf(NB * 4, nullptr); auto bq = mkbuf(xb.size() * 4, nullptr); run("t_quant", {bi, bse, bq}, NB);
      long bad = 0; const int * se = (const int *) bse.contents; const uint32_t * qs = (const uint32_t *) bq.contents;
      for (int b = 0; b < NB; b++) { int hs; ds_u32 hq[32]; bp8_quantize_block_soft(&xb[b*32], &hs, hq, sv, sc); if (se[b] != hs) { if (bad < 3) printf("quant se mismatch blk %d: gpu %d host %d\n", b, se[b], hs); bad++; } for (int j = 0; j < 32; j++) if (qs[b*32+j] != hq[j]) { if (bad < 3) printf("quant code mismatch blk %d j %d: gpu %u host %u\n", b, j, qs[b*32+j], hq[j]); bad++; } }
      printf("quantize block: %ld mismatches / %d blocks\n", bad, NB); }
    // 5. readout
    { std::vector<uint32_t> q8(NB * 8); for (auto & v : q8) v = (uint32_t) r64(); for (int b = 0; b < NB; b += 3) { for (int i = 3; i < 8; i++) q8[b*8+i] = (b % 2) ? 0xFFFFFFFFu : 0; }
      auto bi = mkbuf(q8.size() * 4, q8.data()); auto bo = mkbuf(NB * 4, nullptr); run("t_readout", {bi, bo}, NB);
      long bad = 0; const uint32_t * o = (const uint32_t *) bo.contents; for (int b = 0; b < NB; b++) { uint32_t h = bp8_q256_to_f32bits(&q8[b*8]); if (o[b] != h) { if (bad < 3) printf("readout mismatch %d: gpu %08x host %08x\n", b, o[b], h); bad++; } }
      printf("readout: %ld mismatches / %d\n", bad, NB); }
    return 0;
}
