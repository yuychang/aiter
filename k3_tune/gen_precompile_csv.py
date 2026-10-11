#!/usr/bin/env python3
"""List every FlyDSL candidate the tuners will try, as a CSV that
``python -m aiter.aot.flydsl.gemm --csv`` can compile on CPU.

FlyDSL compiles are M-independent (N/K and the tile config are the cache key),
so one job per (kernel, N, K) warms the cache for every M bucket. Without this
step the tuner JITs ~2000 kernels per (N, K) inside its 8 GPU workers.

  gen_precompile_csv.py a8w8 UNTUNED.csv OUT.csv     # CPU only
  gen_precompile_csv.py bf16 UNTUNED.csv OUT.csv     # reads device properties
"""
import sys

import pandas as pd

HEADER = ["gfx", "cu_num", "M", "N", "K", "bias", "q_dtype_w", "libtype", "kernelName"]


def a8w8(df, gfx, cu_num):
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import PIPELINES

    rows, seen = [], set()
    for r in df.itertuples():
        qname = {"torch.float8_e4m3fn": "fp8", "torch.int8": "int8"}[r.q_dtype_w]
        for pipe in PIPELINES:
            if qname not in pipe.q_dtypes_w:
                continue
            for i in sorted(pipe.kernels_list):
                ki = pipe.kernels_list[i]
                if not pipe.fits(ki, r.M, r.N, r.K) or (ki.name, r.N, r.K) in seen:
                    continue
                seen.add((ki.name, r.N, r.K))
                rows.append((gfx, cu_num, r.M, r.N, r.K, False, r.q_dtype_w, "flydsl", ki.name))
    return rows


def bf16(df, gfx, cu_num):
    import torch

    from aiter.ops.flydsl.gemm_a16w16_policy import get_flydsl_a16w16_configs
    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm_kernel_name

    rows, seen = [], set()
    for r in df.itertuples():
        out_dtype = eval(r.outdtype)
        fused_bias = bool(r.bias and out_dtype == torch.bfloat16)
        for cfg in get_flydsl_a16w16_configs(
            r.M, r.N, r.K, torch.bfloat16, out_dtype, fused_bias
        ):
            name = flydsl_hgemm_kernel_name(
                dtype=torch.bfloat16, out_dtype=out_dtype, config=cfg, has_bias=fused_bias
            )
            if (name, r.N, r.K) in seen:
                continue
            seen.add((name, r.N, r.K))
            rows.append((gfx, cu_num, r.M, r.N, r.K, fused_bias, r.dtype, "flydsl", name))
    return rows


def main():
    kind, untuned, out = sys.argv[1:4]
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    gfx, cu_num = get_gfx(), get_cu_num()
    df = pd.read_csv(untuned)
    rows = {"a8w8": a8w8, "bf16": bf16}[kind](df, gfx, cu_num)
    pd.DataFrame(rows, columns=HEADER).to_csv(out, index=False)
    nk = {(r[3], r[4]) for r in rows}
    print(f"{kind}: {len(rows)} FlyDSL kernels to precompile over {len(nk)} (N,K) pairs -> {out}")


if __name__ == "__main__":
    main()
