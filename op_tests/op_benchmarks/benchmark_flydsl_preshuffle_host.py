#!/usr/bin/env python3
"""Measure host overhead and dynamic-M reuse for Kimi-K3 FlyDSL GEMMs.

Run this script in a fresh process for each AITER tree. The caller is expected
to set PYTHONPATH, AITER_JIT_DIR, FLYDSL_RUNTIME_CACHE_DIR, and
HIP_VISIBLE_DEVICES before Python starts.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function

import aiter
from aiter import dtypes
from aiter.ops import gemm_op_a8w8 as gemm_mod
from aiter.ops.shuffle import shuffle_weight


SHAPES = ((1536, 2304), (7168, 1536), (1536, 7168), (7168, 6144), (512, 3072))
MAX_M = 8192
WARM_M = MAX_M
EQUALITY_M = 257


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p))]


def _selected_rows(csv_path: Path) -> dict[tuple[int, int], dict]:
    table = pd.read_csv(csv_path).drop_duplicates()
    selected = {}
    for k, n in SHAPES:
        rows = table[
            (table["K"] == k)
            & (table["N"] == n)
            & (table["libtype"] == "flydsl")
            & (table["M"] <= MAX_M)
        ]
        if rows.empty:
            raise RuntimeError(f"no FlyDSL tuned row for K={k}, N={n} in {csv_path}")
        exact = rows[rows["M"] == MAX_M]
        row = (exact if not exact.empty else rows[rows["M"] == rows["M"].max()]).iloc[0]
        selected[(k, n)] = row.to_dict()
    return selected


def _inputs(k: int, n: int):
    # Small integral values avoid random-generator differences between trees.
    x = torch.randint(-2, 3, (MAX_M, k), device="cpu", dtype=torch.int8)
    w = torch.randint(-2, 3, (n, k), device="cpu", dtype=torch.int8)
    xq = x.to(torch.float32).to(dtypes.fp8).cuda()
    wq = shuffle_weight(w.to(torch.float32).to(dtypes.fp8).cuda(), layout=(16, 16))
    xs = torch.ones(MAX_M, 1, dtype=torch.float32, device="cuda")
    ws = torch.ones(1, n, dtype=torch.float32, device="cuda")
    out = torch.empty(MAX_M, n, dtype=torch.bfloat16, device="cuda")
    return xq, wq, xs, ws, out


def _call(xq, wq, xs, ws, out, m: int):
    return gemm_mod.gemm_a8w8_bpreshuffle(
        xq[:m],
        wq,
        xs[:m],
        ws,
        dtype=torch.bfloat16,
        out=out[:m],
    )


def _profiler_self_us(xq, wq, xs, ws, out, m: int, calls: int) -> float:
    label = "benchmark::flydsl_gemm_a8w8_bpreshuffle"
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(calls):
            with record_function(label):
                _call(xq, wq, xs, ws, out, m)
    torch.cuda.synchronize()
    events = [
        event
        for event in prof.key_averages()
        if "gemm_a8w8_bpreshuffle" in event.key and event.key != label
    ]
    if not events:
        events = [event for event in prof.key_averages() if event.key == label]
    return sum(event.self_cpu_time_total for event in events) / calls


def _equality(
    reference_dir: Path | None,
    compare_dir: Path | None,
    key: str,
    output: torch.Tensor,
) -> dict | None:
    if reference_dir is not None:
        reference_dir.mkdir(parents=True, exist_ok=True)
        torch.save(output.cpu(), reference_dir / f"{key}.pt")
        return {"saved": True}
    if compare_dir is None:
        return None
    reference = torch.load(compare_dir / f"{key}.pt", map_location="cpu")
    actual = output.cpu()
    delta = (actual.float() - reference.float()).abs()
    return {
        "equal": bool(torch.equal(actual, reference)),
        "max_abs": float(delta.max()),
        "mismatch_count": int(torch.count_nonzero(actual != reference)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-aiter-root", type=Path)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--distinct-m", type=int, default=20)
    parser.add_argument("--steady-calls", type=int, default=100)
    parser.add_argument("--profiler-calls", type=int, default=50)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--compare-dir", type=Path)
    parser.add_argument("--gpu-free-sentinel", type=Path)
    args = parser.parse_args()

    imported = Path(aiter.__file__).resolve()
    if args.expect_aiter_root is not None:
        expected = args.expect_aiter_root.resolve()
        if expected not in imported.parents:
            raise RuntimeError(f"expected AITER under {expected}, imported {imported}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    ms = random.sample(range(1, MAX_M), args.distinct_m)
    rows = _selected_rows(args.csv)
    results = {
        "aiter": str(imported),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "random_m": ms,
        "shapes": [],
    }

    for k, n in SHAPES:
        if (
            args.gpu_free_sentinel is not None
            and not args.gpu_free_sentinel.exists()
        ):
            raise RuntimeError(
                f"GPU permission sentinel disappeared: {args.gpu_free_sentinel}"
            )
        config = rows[(k, n)]
        # Isolate dynamic-M launch behavior from the table's per-M tuning
        # choice: every M deliberately uses this one valid tuned executable.
        gemm_mod.get_GEMM_config_with_quant_type = lambda *a, _cfg=config, **kw: _cfg
        xq, wq, xs, ws, out = _inputs(k, n)

        _call(xq, wq, xs, ws, out, WARM_M)
        torch.cuda.synchronize()

        first_host_ms = []
        for m in ms:
            start = time.perf_counter_ns()
            _call(xq, wq, xs, ws, out, m)
            first_host_ms.append((time.perf_counter_ns() - start) / 1e6)
            torch.cuda.synchronize()

        steady_m = ms[0]
        _call(xq, wq, xs, ws, out, steady_m)
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(args.steady_calls):
            _call(xq, wq, xs, ws, out, steady_m)
        steady_us = (time.perf_counter_ns() - start) / args.steady_calls / 1e3
        torch.cuda.synchronize()

        equality_out = _call(xq, wq, xs, ws, out, EQUALITY_M).clone()
        torch.cuda.synchronize()
        key = f"k{k}-n{n}-m{EQUALITY_M}"
        equality = _equality(args.reference_dir, args.compare_dir, key, equality_out)
        profiler_us = _profiler_self_us(
            xq, wq, xs, ws, out, steady_m, args.profiler_calls
        )
        results["shapes"].append(
            {
                "K": k,
                "N": n,
                "kernelName": config["kernelName"],
                "warm_M": WARM_M,
                "steady_M": steady_m,
                "first_host_ms": first_host_ms,
                "first_mean_ms": statistics.fmean(first_host_ms),
                "first_median_ms": statistics.median(first_host_ms),
                "first_p95_ms": _percentile(first_host_ms, 0.95),
                "first_max_ms": max(first_host_ms),
                "steady_timer_us": steady_us,
                "profiler_self_cpu_us": profiler_us,
                "equality": equality,
            }
        )
        del xq, wq, xs, ws, out, equality_out
        gc.collect()
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
