/* Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. */
/* End-to-end dispatch test for the b-posit8 W8A8 exact-quire vec_dot:
 * exercises the REGISTERED type-traits path (from_float + vec_dot) as ggml's
 * mul_mat would, and cross-checks the result against the dequantized double
 * dot. The bit-exactness of the quire itself is proven separately against the
 * open-bposit rational reference; this guards the wiring/registration. */
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static unsigned long long st = 0x2026072207ULL;
static float frand(void) { /* deterministic LCG in [-2,2) */
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((float)((st >> 33) & 0xFFFFF) / (float)0x80000) * 2.0f - 2.0f;
}

int main(void) {
    const enum ggml_type T = GGML_TYPE_BPOSIT8;
    const struct ggml_type_traits     * tt  = ggml_get_type_traits(T);
    const struct ggml_type_traits_cpu * ttc = ggml_get_type_traits_cpu(T);

    if (!ttc || !ttc->vec_dot || !ttc->from_float || !tt || !tt->to_float) {
        printf("FAIL: BPOSIT8 type traits not registered (vec_dot=%p from_float=%p to_float=%p)\n",
               (void*)(ttc ? (void*)ttc->vec_dot : 0),
               (void*)(ttc ? (void*)ttc->from_float : 0),
               (void*)(tt ? (void*)tt->to_float : 0));
        return 1;
    }
    printf("registration OK: name=%s blck=%d size=%zu vec_dot_type=%d\n",
           tt->type_name, (int)ggml_blck_size(T), ggml_type_size(T), (int)ttc->vec_dot_type);

    const int qk = ggml_blck_size(T);
    int fails = 0;
    for (int trial = 0; trial < 8; trial++) {
        const int nb = 1 + trial;          /* 1..8 blocks -> streaming K */
        const int n  = nb * qk;
        float *x = malloc(n*sizeof(float)), *y = malloc(n*sizeof(float));
        for (int i = 0; i < n; i++) { x[i] = frand(); y[i] = frand(); }

        size_t bs = ggml_type_size(T);
        void *qx = malloc(nb*bs), *qy = malloc(nb*bs);
        ttc->from_float(x, qx, n);
        ttc->from_float(y, qy, n);

        float s = 0.0f;
        ttc->vec_dot(n, &s, 0, qx, 0, qy, 0, 1);

        /* reference: dequantize (exact per element) then double dot. For modest
         * n the quire readout and the double dot agree to float precision. */
        float *dx = malloc(n*sizeof(float)), *dy = malloc(n*sizeof(float));
        tt->to_float(qx, dx, n);
        tt->to_float(qy, dy, n);
        double ref = 0.0;
        for (int i = 0; i < n; i++) ref += (double)dx[i]*(double)dy[i];

        double denom = fabs(ref) > 1e-9 ? fabs(ref) : 1.0;
        double rel = fabs((double)s - ref) / denom;
        int ok = rel < 1e-5;
        printf("  nb=%d n=%4d  vec_dot=%- .8g  dequant-dot=%- .8g  rel=%.2e  %s\n",
               nb, n, (double)s, ref, rel, ok ? "ok" : "MISMATCH");
        if (!ok) fails++;
        free(x); free(y); free(qx); free(qy); free(dx); free(dy);
    }
    printf("== bposit8 registered vec_dot dispatch: %s ==\n", fails ? "FAIL" : "ALL OK");
    return fails ? 1 : 0;
}
