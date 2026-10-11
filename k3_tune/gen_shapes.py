#!/usr/bin/env python3
"""Generate the Kimi-K3 prefill GEMM untuned shape CSVs.

CPU-only. Writes the untuned CSVs under aiter/configs/model_configs/ and drops
every (gfx950, cu_num=256, M, N, K, ...) row that an existing tuned table
already holds: aiter/jit/core.py::update_config_files treats a key present in
two tuned files as a duplicate, rewrites the source files and raises, so the
tuner output must never collide with a row that is already tuned.

M buckets follow the runtime lookup (aiter/ops/gemm_op_a8w8.py
get_GEMM_config_with_quant_type, aiter/tuned_gemm.py get_GEMM_A16W16_config):
exact M, then getPaddedM gl=0 (x16 to 256, x32 to 1024, x64 to 4096, x128
above), then gl=1 (next power of two; M > 8192 with N > 4096 -> 8192). A
power-of-two row therefore serves every M that has no finer row, which is why
the lists below are power-of-two buckets.
"""
import argparse
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CFG = os.path.join(ROOT, "aiter", "configs")
MC = os.path.join(CFG, "model_configs")

GFX, CU_NUM = "gfx950", 256

A8_OUT = os.path.join(MC, "a8w8_bpreshuffle_untuned_gemm_kimik3_prefill.csv")
BF16_OUT = os.path.join(MC, "kimik3_prefill_bf16_untuned_gemm.csv")

FP8 = "torch.float8_e4m3fn"
BF16 = "torch.bfloat16"

# (N, K) -> M buckets still missing from the tuned tables.
A8W8_SHAPES = {
    (6144, 7168): [512, 1024, 2048, 4096],  # 8192/16384/32768 already tuned
    (128, 7168): [512, 1024, 2048, 4096, 8192],
    # kv_b_proj of the cached prefix, up to 65536 rows: N <= 4096 pads to the
    # next power of two, so 32769..65536 all land on the 65536 row.
    (3072, 512): [65536],
    # Tuned table stops at M=256 and the M=257..4096 prefill chunks run the BF16
    # up projection, so only the M > 4096 chunks (-> 8192 bucket) need a row.
    (7168, 3584): [4096, 8192],
}

SMALL_M_16 = [1, 2, 4, 8] + list(range(16, 257, 16))
POW2_1_8192 = [1 << i for i in range(14)]
BF16_SHAPES = {
    (7168, 3584): [2048, 4096],  # latent up projection, M 1025..4096
    (768, 7168): SMALL_M_16,  # DSpark draft model, M 1..256 (every x16 bucket)
    (128, 7168): POW2_1_8192,
    (3072, 512): [2048, 4096, 16384, 32768, 65536],
}


def merged_tuned(default_name, pattern):
    files = [os.path.join(CFG, default_name)]
    files += sorted(
        p
        for p in glob.glob(os.path.join(MC, f"*{pattern}*.csv"))
        if "untuned" not in os.path.basename(p)
    )
    frames = []
    for f in files:
        if os.path.basename(f) in (
            os.path.basename(A8_OUT).replace("untuned", "tuned"),
            os.path.basename(BF16_OUT).replace("untuned", "tuned"),
        ):
            continue  # our own output from an earlier run
        df = pd.read_csv(f)
        if "gfx" in df.columns:
            df = df[(df.gfx == GFX) & (df.cu_num == CU_NUM)]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_a8w8():
    tuned = merged_tuned("a8w8_bpreshuffle_tuned_gemm.csv", "a8w8_bpreshuffle_tuned_gemm")
    have = set(zip(tuned.M, tuned.N, tuned.K, tuned.q_dtype_w))
    rows = []
    for (n, k), ms in A8W8_SHAPES.items():
        for m in ms:
            if (m, n, k, FP8) not in have:
                rows.append((m, n, k, FP8))
    return pd.DataFrame(rows, columns=["M", "N", "K", "q_dtype_w"])


def build_bf16():
    tuned = merged_tuned("bf16_tuned_gemm.csv", "bf16_tuned_gemm")
    have = set(
        zip(
            tuned.M, tuned.N, tuned.K, tuned.bias, tuned.dtype, tuned.outdtype,
            tuned.scaleAB, tuned.bpreshuffle,
        )
    )
    rows = []
    for (n, k), ms in BF16_SHAPES.items():
        for m in ms:
            key = (m, n, k, False, BF16, BF16, False, False)
            if key not in have:
                rows.append(key)
    return pd.DataFrame(
        rows,
        columns=["M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed CSVs differ from a fresh generation")
    args = ap.parse_args()
    outs = [(A8_OUT, build_a8w8()), (BF16_OUT, build_bf16())]
    stale = False
    for path, df in outs:
        text = df.to_csv(index=False)
        if args.check:
            cur = open(path).read() if os.path.exists(path) else ""
            if cur != text:
                print(f"STALE: {path}")
                stale = True
            continue
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {len(df):3d} shapes -> {os.path.relpath(path, ROOT)}")
    sys.exit(1 if stale else 0)


if __name__ == "__main__":
    main()
