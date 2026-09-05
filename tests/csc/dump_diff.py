#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""dump_diff: localise where two INVAR_LOGITS_OUT dumps (same model, same prompt, two
deployments, e.g. CPU vs CUDA) first diverge, per graph evaluation, in graph order.

Prints for each evaluation the first differing tensor line, how many floats differ and the
max ulp distance, then a per-tensor-name summary over the whole dump. Tensors are compared
by their position in the dump (the hook appends in graph order), so a line-count mismatch
is reported as such."""
from __future__ import annotations

import json
import struct
import sys
from collections import Counter, OrderedDict


def floats(hexstr: str) -> list[float]:
    b = bytes.fromhex(hexstr)
    return list(struct.unpack("<%df" % (len(b) // 4), b))


def bits(hexstr: str) -> list[int]:
    b = bytes.fromhex(hexstr)
    return list(struct.unpack("<%dI" % (len(b) // 4), b))


def ulp_dist(a: int, b: int) -> int:
    # ordered-int distance for float32 bit patterns
    def key(x):
        return x if x < 0x80000000 else 0x80000000 - x
    return abs(key(a) - key(b))


def main():
    if len(sys.argv) != 3:
        print("usage: dump_diff.py A.jsonl B.jsonl", file=sys.stderr)
        sys.exit(2)
    A = [json.loads(l) for l in open(sys.argv[1])]
    B = [json.loads(l) for l in open(sys.argv[2])]
    if len(A) != len(B):
        print(f"line count differs: {len(A)} vs {len(B)}")
    n = min(len(A), len(B))
    per_name = OrderedDict()
    ev = 0
    first_reported = False
    evals_first = []
    for i in range(n):
        a, b = A[i], B[i]
        name = a.get("tensor")
        if name != b.get("tensor"):
            print(f"line {i}: tensor name differs {name} vs {b.get('tensor')}; stopping")
            break
        # a new evaluation starts at the first tensor of layer 0 (attn_norm-0) or result_norm
        if name in ("attn_norm-0",) and i > 0:
            ev += 1
            first_reported = False
        if a.get("hex") == b.get("hex"):
            per_name.setdefault(name, [0, 0, 0])[0] += 1
            continue
        xa, xb = bits(a["hex"]), bits(b["hex"])
        nd = sum(1 for p, q in zip(xa, xb) if p != q)
        mu = max(ulp_dist(p, q) for p, q in zip(xa, xb))
        st = per_name.setdefault(name, [0, 0, 0])
        st[1] += 1
        st[2] = max(st[2], mu)
        if not first_reported:
            evals_first.append((ev, name, nd, len(xa), mu))
            first_reported = True
    print("first divergence per evaluation (eval, tensor, differing/total floats, max ulp):")
    for e in evals_first[:12]:
        print("  ", e)
    if len(evals_first) > 12:
        print(f"   ... {len(evals_first)} evaluations diverge in total")
    print("\nper tensor kind (name without layer index): identical lines / differing lines / max ulp")
    kinds = OrderedDict()
    for name, (same, diff, mu) in per_name.items():
        kind = name.rsplit("-", 1)[0] if "-" in name else name
        k = kinds.setdefault(kind, [0, 0, 0])
        k[0] += same; k[1] += diff; k[2] = max(k[2], mu)
    for kind, (same, diff, mu) in kinds.items():
        print(f"   {kind:14s} {same:6d} {diff:6d} {mu:>8d}")


if __name__ == "__main__":
    main()
