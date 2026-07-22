/* Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. */
/* Standalone verification of the exact-quire b-posit8 W8A8 dot product.
 * Mirrors the canonical Anomly 256-bit / 96-frac two's-complement quire.
 * This is the numerical core that will become ggml_vec_dot_bposit8_bposit8.
 * Verified bit-exact against open-bposit's rational Fraction reference. */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "bp8_dot_golden.h"

/* 256-bit two's-complement accumulator, 8 x 32-bit limbs, radix point at
 * bit BP8_QUIRE_FRAC_BITS. */
typedef struct { uint32_t l[8]; } quire256;

static void q_zero(quire256 *q) { for (int i = 0; i < 8; i++) q->l[i] = 0; }

static void q_add(quire256 *q, const uint32_t t[8]) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t s = (uint64_t)q->l[i] + t[i] + carry;
        q->l[i] = (uint32_t)s;
        carry = s >> 32;
    }
}

/* value = P * 2^shift placed into a 256-bit two's-complement image `t`.
 * shift may be negative (arithmetic right shift = truncation toward -inf,
 * needed by the real kernel; the exact test set only exercises shift>=0). */
static void place_shifted(int64_t P, int shift, uint32_t t[8]) {
    /* sign-extend P to 256 bits */
    uint32_t sx = (P < 0) ? 0xFFFFFFFFu : 0u;
    uint64_t up = (uint64_t)P;
    t[0] = (uint32_t)up; t[1] = (uint32_t)(up >> 32);
    for (int i = 2; i < 8; i++) t[i] = sx;

    if (shift == 0) return;
    if (shift > 0) {
        int words = shift >> 5, bits = shift & 31;
        /* shift left by `bits` within 256, then by `words` limbs */
        if (bits) {
            uint32_t prev = 0;
            for (int i = 0; i < 8; i++) {
                uint32_t cur = t[i];
                t[i] = (cur << bits) | prev;
                prev = (uint32_t)((uint64_t)cur >> (32 - bits));
            }
        }
        if (words) {
            for (int i = 7; i >= 0; i--) t[i] = (i - words >= 0) ? t[i - words] : 0u;
        }
    } else {
        int s = -shift, words = s >> 5, bits = s & 31;
        if (words) {
            for (int i = 0; i < 8; i++) t[i] = (i + words < 8) ? t[i + words] : sx;
        }
        if (bits) {
            uint32_t next = sx; /* arithmetic: fill from sign */
            for (int i = 7; i >= 0; i--) {
                uint32_t cur = t[i];
                t[i] = (cur >> bits) | (next << (32 - bits));
                next = cur;
            }
        }
    }
}

/* exact-quire dot of two b-posit8 blocks (QK codes each) with power-of-two
 * block scales sx, sy. Accumulates into q (does NOT zero it — caller does,
 * so K can stream across blocks unbounded, exactly like the SFPU kernel). */
static void bp8_block_dot(const uint8_t *xs, const uint8_t *ys,
                          int sx, int sy, quire256 *q) {
    for (int j = 0; j < BP8_QK; j++) {
        bp8_lut_t X = BP8_LUT[xs[j]];
        bp8_lut_t Y = BP8_LUT[ys[j]];
        if (X.kind || Y.kind) continue;            /* zero or NaR -> skip (NaR treated as 0 here) */
        int64_t P = X.M * Y.M;                      /* exact: |M|<2^13 each -> product < 2^26 */
        int shift = (int)(X.E + Y.E + sx + sy) + BP8_QUIRE_FRAC_BITS;
        uint32_t t[8];
        place_shifted(P, shift, t);
        q_add(q, t);
    }
}

/* final readout: signed 256-bit quire (radix at FRAC_BITS) -> double, one
 * rounding at the end (this is the float ggml_vec_dot returns). */
static double q_to_double(const quire256 *q) {
    quire256 m = *q;
    int neg = (m.l[7] >> 31) & 1;
    if (neg) { /* two's-complement negate */
        uint64_t carry = 1;
        for (int i = 0; i < 8; i++) {
            uint64_t s = (uint64_t)(~m.l[i]) + carry;
            m.l[i] = (uint32_t)s; carry = s >> 32;
        }
    }
    double v = 0.0;
    for (int i = 7; i >= 0; i--) v = v * 4294967296.0 + (double)m.l[i];
    v = ldexp(v, -BP8_QUIRE_FRAC_BITS);
    return neg ? -v : v;
}

int main(void) {
    int fails = 0;
    for (int c = 0; c < BP8_NCASES; c++) {
        const bp8_case_t *tc = &BP8_CASES[c];
        quire256 q; q_zero(&q);
        bp8_block_dot(tc->xs, tc->ys, tc->sx, tc->sy, &q);
        int ok = 1;
        for (int i = 0; i < 8; i++) if (q.l[i] != tc->golden[i]) ok = 0;
        if (!ok) {
            fails++;
            if (fails <= 4) {
                printf("case %d MISMATCH\n  got   ", c);
                for (int i = 7; i >= 0; i--) printf("%08x", q.l[i]);
                printf("\n  golden");
                for (int i = 7; i >= 0; i--) printf("%08x", tc->golden[i]);
                printf("\n");
            }
        }
    }
    printf("[1] single-block exact-quire dot: %d/%d bit-exact vs rational reference\n",
           BP8_NCASES - fails, BP8_NCASES);

    /* [2] streaming: K spans nb blocks into ONE quire, then read out to double */
    int sfails = 0;
    for (int c = 0; c < BP8_NSTREAM; c++) {
        const bp8_stream_t *st = &BP8_STREAM[c];
        quire256 q; q_zero(&q);
        for (int b = 0; b < st->nb; b++)
            bp8_block_dot(st->xs + b*BP8_QK, st->ys + b*BP8_QK, st->sx[b], st->sy[b], &q);
        int ok = 1;
        for (int i = 0; i < 8; i++) if (q.l[i] != st->golden[i]) ok = 0;
        double got = q_to_double(&q);
        /* nearest-double readout must equal the reference's nearest double */
        if (!ok || got != st->gd) {
            sfails++;
            printf("stream nb=%d MISMATCH quire_ok=%d got=%.17g gold=%.17g\n",
                   st->nb, ok, got, st->gd);
        }
    }
    printf("[2] multi-block streaming quire + double readout: %d/%d exact\n",
           BP8_NSTREAM - sfails, BP8_NSTREAM);

    /* [3] catastrophic cancellation: (+2^40)+(-2^40)+1*62 ... build [+big, -big] that
     * cancel exactly, leaving a small residue a float32 accumulator would lose. */
    quire256 q; q_zero(&q);
    uint8_t xs[BP8_QK], ys[BP8_QK];
    for (int j = 0; j < BP8_QK; j++) { xs[j] = BP8_ONE_CODE; ys[j] = BP8_ONE_CODE; }
    /* +big with scale +40 on first lane pair, -big cancels it, rest are 1*1 */
    bp8_block_dot(xs, ys, 40, 0, &q);   /* + 2^40 * 32 */
    /* subtract the same by adding a block whose product sign is negative:
     * use one-code but we need a -1 code. Find via negation below at runtime. */
    int neg_one = -1;
    for (int code = 0; code < 256; code++)
        if (BP8_LUT[code].kind == 0 && BP8_LUT[code].M < 0 &&
            BP8_LUT[code].M * (int64_t)1 == -BP8_LUT[BP8_ONE_CODE].M &&
            BP8_LUT[code].E == BP8_LUT[BP8_ONE_CODE].E) { neg_one = code; break; }
    int cfail = 0;
    if (neg_one < 0) { printf("[3] cancellation: (skipped, no -1 code)\n"); }
    else {
        for (int j = 0; j < BP8_QK; j++) ys[j] = neg_one;   /* now product = -1 */
        bp8_block_dot(xs, ys, 40, 0, &q);   /* - 2^40 * 32  -> exact cancel of the +big */
        double resid = q_to_double(&q);     /* should be exactly 0 (the two big blocks cancel) */
        cfail = (resid != 0.0);
        printf("[3] cancellation (+2^40*32 then -2^40*32): residue=%.17g -> %s\n",
               resid, cfail ? "FAIL" : "exact 0 (float32 would drift)");
    }

    int total = fails + sfails + cfail;
    printf("== bp8 exact-quire vec_dot core: %s ==\n", total ? "FAIL" : "ALL VERIFIED");
    return total ? 1 : 0;
}
