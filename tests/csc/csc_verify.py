#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
csc_verify.py — client-side spot-check of an exact-profile generation.

A serving box ran llama-cli with INVAR_LOGITS_OUT=<dump> on a b-posit8 GGUF. Each graph
evaluation appended the last row of `result_norm` (the final-norm hidden state the lm_head
consumes) and of `result_output` (the logits). This script, holding only the GGUF and the
dump, re-executes SAMPLED lm_head rows in pure Python integer arithmetic — an
implementation that shares no code with the C kernel — and demands the float32 logits
match BIT FOR BIT. That is the property the `llamacpp-bposit8-quire-v0` profile certifies:
order-independent exact accumulation, reproducible across implementations.

Pipeline per step (mirrors ggml exactly, by construction of the format):
  1. quantise the hidden row with the same rule as quantize_row_bposit8_ref: per block of
     32, scale_exp = lrint(log2(rms)) clamped to int8, codes = nearest b-posit8 value to
     x * 2^-scale_exp (ties -> lowest code), zero stays code 0x00;
  2. for each challenged row r of the (tied) lm_head weight, decode both operands to exact
     integers M*2^E and accumulate the products in a Python int at fixed point 2^-96
     (the 256-bit quire is a fixed-point integer; Python ints are unbounded so this is the
     same value with no truncation for shift >= 0, and per-term floor for shift < 0 as the
     kernel does);
  3. one rounding to double (round-half-even on the exact rational, as the kernel's
     q256_to_double does via the double conversion of a 256-bit integer), then to float32;
  4. compare to the dumped logits[r] bit pattern.

Sampled rows come from a challenge nonce (PRF = sha256(nonce || counter)), so a prover
cannot know which rows will be checked. Rows not sampled are not attested (CR §10).

Usage:
  python3 tests/csc/csc_verify.py --gguf model.gguf --dump logits.jsonl [--rows 512]
                                  [--nonce hex] [--tamper]   # --tamper flips one logit bit
Exit 0 = every sampled row of every step matched; 1 = any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gguf-py"))
import gguf  # noqa: E402
import numpy as np  # noqa: E402

QK = 32
ES = 2
QFRAC = 96
ZERO, NAR = 0x00, 0x80


# ---------------------------------------------------------------- b-posit8 codec

def code_to_ME(p: int) -> tuple[int, int]:
    if p in (ZERO, NAR):
        return 0, 0
    s = (p >> 7) & 1
    rest = p & 0x7F
    if s:
        rest = ((~rest) + 1) & 0x7F
    leading = (rest >> 6) & 1
    rs = 0
    while rs < 7 and ((rest >> (6 - rs)) & 1) == leading:
        rs += 1
    e = fb = fw = 0
    if rs == 7:
        k = 6 if leading else -7
    else:
        k = (rs - 1) if leading else -rs
        rem = 7 - (rs + 1)
        r2 = rest & ((1 << rem) - 1)
        ew = ES if ES < rem else rem
        if ew > 0:
            e = ((r2 >> (rem - ew)) & ((1 << ew) - 1)) << (ES - ew)
        rem -= ew
        fw = rem
        fb = r2 & ((1 << fw) - 1) if fw > 0 else 0
    m = (1 << fw) + fb
    return (-m if s else m), 4 * k + e - fw


LUT_M = [0] * 256
LUT_E = [0] * 256
VAL = [0.0] * 256
for _c in range(256):
    LUT_M[_c], LUT_E[_c] = code_to_ME(_c)
    VAL[_c] = LUT_M[_c] * math.ldexp(1.0, LUT_E[_c])     # exact in double


def encode_nearest(x: float) -> int:
    if x == 0.0:
        return ZERO
    best, bestd = ZERO, math.inf
    for c in range(256):
        if c == NAR:
            continue
        d = abs(VAL[c] - x)
        if d < bestd:
            bestd, best = d, c
    return best


def scale_exp_exact(blk) -> int:
    """ggml_bp8_scale_exp_exact: round_half_even(log2(sqrt(S/32))) from the EXACT sum of
    squares S (rational); tie = S a power of two; non-finite -> 0; all-zero -> 0."""
    from fractions import Fraction
    S = Fraction(0)
    any_nz = False
    for v in blk:
        if v == 0.0:
            continue
        if not math.isfinite(v):
            return 0
        any_nz = True
        f = Fraction(v)
        S += f * f
    if not any_nz:
        return 0
    N, D = S.numerator, S.denominator
    E = (N.bit_length() - 1) - (D.bit_length() - 1) - 5
    tie = (N & (N - 1)) == 0
    if E % 2 == 0:
        se = E // 2
    else:
        n = (E - 1) // 2
        se = (n if n % 2 == 0 else n + 1) if tie else n + 1
    return max(-128, min(127, se))


def quantize_row(x: np.ndarray) -> list[tuple[int, list[int]]]:
    """quantize_row_bposit8_ref in Python: [(scale_exp, codes[32]) ...]. Exact integer
    block-scale rule (scale_exp_exact), double math for the encode."""
    assert x.shape[0] % QK == 0
    out = []
    for i in range(x.shape[0] // QK):
        blk = x[i * QK:(i + 1) * QK].astype(np.float64)
        se = scale_exp_exact(blk.tolist())
        inv = math.ldexp(1.0, -se)
        codes = [encode_nearest(v * inv) for v in blk.tolist()]
        out.append((se, codes))
    return out


# ---------------------------------------------------------------- exact dot + readout

def exact_dot(xblocks, yblocks) -> float:
    """Exact quire dot in Python integers at fixed point 2^-QFRAC, then the kernel's
    single rounding: 256-bit two's-complement integer -> double (nearest-even) -> caller
    casts to float32. Sub-radix terms (shift < 0) are floored per term as the kernel does."""
    acc = 0
    for (sx, xq), (sy, yq) in zip(xblocks, yblocks):
        se = sx + sy + QFRAC
        for j in range(QK):
            P = LUT_M[xq[j]] * LUT_M[yq[j]]
            if P == 0:
                continue
            shift = LUT_E[xq[j]] + LUT_E[yq[j]] + se
            if shift >= 0:
                acc += P << shift
            else:
                acc += P >> (-shift)                  # Python >> floors (toward -inf) like the kernel
    # wrap to 256-bit two's complement, then the kernel's readout: magnitude via
    # 8 limbs folded into a double (v = v*2^32 + limb, one rounding per fold — the kernel
    # does exactly this, so mirror it rather than using an exact conversion)
    acc &= (1 << 256) - 1
    neg = (acc >> 255) & 1
    mag = ((~acc) + 1) & ((1 << 256) - 1) if neg else acc
    v = 0.0
    for i in range(7, -1, -1):
        limb = (mag >> (32 * i)) & 0xFFFFFFFF
        v = v * 4294967296.0 + float(limb)
    v = math.ldexp(v, -QFRAC)
    return -v if neg else v


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


# ---------------------------------------------------------------- driver

def sampled_rows(nonce: bytes, n_vocab: int, k: int) -> list[int]:
    rows, ctr, seen = [], 0, set()
    while len(rows) < k:
        h = hashlib.sha256(nonce + ctr.to_bytes(4, "big")).digest()
        ctr += 1
        r = int.from_bytes(h[:8], "big") % n_vocab
        if r not in seen:
            seen.add(r)
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--nonce", default=None, help="hex; default random (printed)")
    ap.add_argument("--tamper", action="store_true", help="flip one bit of one dumped logit")
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()

    r = gguf.GGUFReader(a.gguf)
    ft = int(r.fields["general.file_type"].parts[r.fields["general.file_type"].data[0]][0])
    if ft != 42:
        print(f"REJECT — GGUF file_type {ft} is not b-posit8 (42); exact re-execution not defined")
        return 1
    tensors = {t.name: t for t in r.tensors}
    w = tensors.get("output.weight") or tensors["token_embd.weight"]      # tied lm_head
    n_vocab, row_bytes = w.data.shape
    n_embd = (row_bytes // (1 + QK)) * QK
    W = np.asarray(w.data)                                                # uint8 [n_vocab, nb*33]

    def wrow(rr: int):
        raw = W[rr].tobytes()
        return [(struct.unpack("<b", raw[i * 33:i * 33 + 1])[0], list(raw[i * 33 + 1:i * 33 + 33]))
                for i in range(len(raw) // 33)]

    lines = [json.loads(x) for x in open(a.dump) if x.strip()]
    steps = []
    pending = None
    for ln in lines:
        if ln["tensor"] == "result_norm":
            pending = ln
        elif ln["tensor"] == "result_output" and pending is not None:
            steps.append((pending, ln))
            pending = None
    if a.max_steps:
        steps = steps[:a.max_steps]
    nonce = bytes.fromhex(a.nonce) if a.nonce else secrets.token_bytes(16)
    print(f"gguf={os.path.basename(a.gguf)} n_vocab={n_vocab} n_embd={n_embd} steps={len(steps)} "
          f"rows/step={a.rows} nonce={nonce.hex()}")
    bad = 0
    checked = 0
    for si, (hn, lo) in enumerate(steps):
        hidden = np.frombuffer(bytes.fromhex(hn["hex"]), dtype="<f4")
        logits = np.frombuffer(bytes.fromhex(lo["hex"]), dtype="<f4").copy()
        if hidden.shape[0] != n_embd or logits.shape[0] != n_vocab:
            print(f"step {si}: REJECT — shape mismatch hidden={hidden.shape} logits={logits.shape}")
            bad += 1
            continue
        rows = sampled_rows(nonce + si.to_bytes(4, "big"), n_vocab, a.rows)
        if a.tamper and si == 0:
            # negative control: flip the lowest bit of the FIRST SAMPLED logit (a 1-ulp change)
            bits = f32_bits(float(logits[rows[0]])) ^ 0x1
            logits[rows[0]] = struct.unpack("<f", struct.pack("<I", bits))[0]
            print(f"  (tamper: logits[{rows[0]}] flipped by 1 ulp)")
        xq = quantize_row(hidden)
        for rr in rows:
            got = f32_bits(float(np.float32(exact_dot(xq, wrow(rr)))))
            want = f32_bits(float(logits[rr]))
            checked += 1
            if got != want:
                bad += 1
                if bad <= 5:
                    print(f"step {si} row {rr}: MISMATCH re-executed {got:08x} vs served {want:08x}")
    print(f"{'ACCEPT' if bad == 0 else 'REJECT'} — {checked - bad}/{checked} sampled lm_head rows "
          f"re-executed bit-exactly across {len(steps)} steps")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
