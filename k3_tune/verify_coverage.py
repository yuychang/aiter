#!/usr/bin/env python3
"""CPU-only: replay the M values that missed in serving logs against the tuned
tables (+ the shapes in the untuned CSVs) with aiter's exact/gl0/gl1 lookup, and
report how many would now hit.

    verify_coverage.py LOG [LOG ...]
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import gen_shapes as g
import pandas as pd

NOT_FOUND = re.compile(
    r"shape is M:(\d+), N:(\d+), K:(\d+)(.*?)not found tuned config in (\S+)"
)


def next_pow2(m):
    return 1 if m <= 1 else 1 << (m - 1).bit_length()


def padded(m, n, gl):
    """csrc/py_itfs_cu/gemm_common.cu::getPaddedM."""
    if gl == 0:
        if 0 < m < 16:
            return next_pow2(m)
        if m <= 256:
            return (m + 15) // 16 * 16
        if m <= 1024:
            return (m + 31) // 32 * 32
        if m <= 4096:
            return (m + 63) // 64 * 64
        return (m + 127) // 128 * 128
    return 8192 if (m > 8192 and n > 4096) else next_pow2(m)


def hits(table, m, n, k):
    for cand in (m, padded(m, n, 0), padded(m, n, 1)):
        if (cand, n, k) in table:
            return cand
    return None


def main(logs):
    a8 = g.merged_tuned(
        "a8w8_bpreshuffle_tuned_gemm.csv", "a8w8_bpreshuffle_tuned_gemm"
    )
    bf = g.merged_tuned("bf16_tuned_gemm.csv", "bf16_tuned_gemm")
    bf = bf[(bf.bias == False) & (bf.outdtype == g.BF16)]
    old = {
        "a8w8": set(zip(a8.M, a8.N, a8.K)),
        "bf16": set(zip(bf.M, bf.N, bf.K)),
    }
    new = {k: set(old[k]) for k in old}
    for _, r in pd.read_csv(g.A8_OUT).iterrows():
        new["a8w8"].add((r.M, r.N, r.K))
    for _, r in pd.read_csv(g.BF16_OUT).iterrows():
        new["bf16"].add((r.M, r.N, r.K))

    seen = defaultdict(set)
    for path in logs:
        for line in Path(path).read_text(errors="ignore").splitlines():
            for m in NOT_FOUND.finditer(line):
                M, N, K = map(int, m.group(1, 2, 3))
                kind = "a8w8" if "a8w8" in m.group(5) else "bf16"
                seen[(kind, N, K)].add(M)

    print(
        f"{'kind':5} {'N':>5} {'K':>5} {'distinct M':>10} {'hit before':>10} {'hit after':>10}  rows used after"
    )
    for (kind, n, k), ms in sorted(seen.items()):
        b = sum(hits(old[kind], m, n, k) is not None for m in ms)
        used = Counter(hits(new[kind], m, n, k) for m in ms)
        a = sum(v for key, v in used.items() if key is not None)
        rows = ",".join(str(x) for x in sorted(x for x in used if x is not None))
        print(f"{kind:5} {n:5d} {k:5d} {len(ms):10d} {b:10d} {a:10d}  {rows}")


if __name__ == "__main__":
    main(sys.argv[1:])
