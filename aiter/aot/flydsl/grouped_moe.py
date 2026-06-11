#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for gfx1250 grouped MoE GEMM kernels."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from contextlib import contextmanager

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    job_identity,
    override_env,
)
from aiter.jit.core import AITER_CONFIGS

DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_GROUPED_FMOE_FILE]
_WARP_TILE_N = 64
_TILE_K = 256
# The grouped MoE FlyDSL path is gfx1250-only; cross-compile defaults to it
# unless ARCH/GPU_ARCHS pin a specific target.
GROUPED_AOT_ARCH_DEFAULT = "gfx1250"


@contextmanager
def _fake_cuda_stream(torch):
    """Make ``torch.cuda.current_stream()`` return ``0`` during compilation.

    The runtime grouped-MoE kernel wrappers hardcode
    ``stream=torch.cuda.current_stream()``. Under ``COMPILE_ONLY`` the stream is
    never used for a real launch, so returning the ``0`` sentinel (same value
    moe.py passes) lets the whole AOT pass run GPU-free under ``FakeTensorMode``.
    """
    orig = torch.cuda.current_stream
    torch.cuda.current_stream = lambda *a, **k: 0
    try:
        yield
    finally:
        torch.cuda.current_stream = orig


def _preshuffled_scale_shape(
    rows: int, k_dim: int, warp_tile: int, tile_k: int = _TILE_K
) -> tuple[int, int]:
    """Mirror moe_grouped_gemm_mxscale_gfx1250._preshuffled_scale_shape.

    The grouped GEMM launchers validate an exact preshuffled E8M0 scale layout
    (see tests.kernels.test_gemm_mxscale_gfx1250.preshuffle_e8m0_scale), so the
    AOT dummy tensors must use the same shape, not the plain (rows, k//32) one.
    """
    k_scale = int(k_dim) // 32
    scale_k_per_tile = int(tile_k) // 32
    if k_scale % scale_k_per_tile != 0:
        raise ValueError(
            f"K scale columns must be divisible by tile_k/32, got {k_scale} and {scale_k_per_tile}"
        )
    wmma_rep = int(warp_tile) // 16
    if wmma_rep < 1:
        raise ValueError(f"warp_tile must be >= 16, got {warp_tile}")
    if int(rows) % wmma_rep != 0:
        raise ValueError(
            f"scale rows must be divisible by wmma_rep={wmma_rep}, got {rows}"
        )
    return int(rows) // wmma_rep, k_scale * wmma_rep


def _as_bool(value, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes")


def _as_int(value, default: int | None = None) -> int | None:
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def parse_csv(csv_path: str):
    jobs = []
    seen = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not row:
                continue
            # Skip blank/incomplete rows (e.g. a trailing whitespace line),
            # where required columns come back empty or None.
            if any(
                row.get(col) is None or str(row.get(col)).strip() == ""
                for col in ("model_dim", "inter_dim", "expert", "token")
            ):
                continue
            n_warp = int(row.get("n_warp") or 4)
            job = {
                "kernel_name": row.get("kernelName1", "grouped_gemm1"),
                "cu_num": int(row.get("cu_num") or 0),
                "model_dim": int(row["model_dim"]),
                "inter_dim": int(row["inter_dim"]),
                "experts": int(row["expert"]),
                "topk": int(row.get("topk") or 1),
                "max_m": int(row["token"]),
                "tile_m": int(row.get("tile_m") or 64),
                "tile_n": n_warp * _WARP_TILE_N,
                "tile_k": _TILE_K,
                "m_warp": int(row.get("m_warp") or 1),
                "n_warp": n_warp,
                "num_buffers": int(row.get("num_buffers") or 2),
                "split_k1": int(row.get("split_k1") or 1),
                "split_k2": int(row.get("split_k2") or 1),
                "out_dtype": "bf16" if row.get("dtype") == "torch.bfloat16" else "f16",
                "grouped_persistent_m": _as_bool(row.get("grouped_persistent_m"), True),
                "grouped_contiguous_m": _as_bool(row.get("grouped_contiguous_m"), False),
                "persistent_workers": _as_int(row.get("persistent_workers"), None),
                "stage1_weight_layout": row.get("stage1_weight_layout") or "gguu",
                "act": "swiglu" if "Swiglu" in row.get("act_type", "") else "silu",
                "data_format": (
                    "fp4" if "float4" in row.get("q_dtype_a", "") else "a8w4"
                ),
            }
            key = job_identity(job)
            if key in seen:
                continue
            seen.add(key)
            jobs.append(job)
    for job in jobs:
        if job.get("grouped_contiguous_m"):
            job["grouped_persistent_m"] = False
    return jobs


def _precompile_to_cache(**job):
    import torch

    from aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 import (
        _get_compiled_m_tile_prefix_map,
        compile_moe_grouped_gemm1_a8w4_masked,
        compile_moe_grouped_gemm1_mxfp4_masked,
        compile_moe_grouped_gemm2_a8w4_masked,
        compile_moe_grouped_gemm2_mxfp4_masked,
    )
    from aiter.ops.flydsl.moe_kernels import (
        build_route_maps,
        contiguous_psum,
        flydsl_moe_gather_reduce,
        flydsl_moe_preshuffle_scale,
        flydsl_moe_scatter_copy_token,
        flydsl_moe_scatter_preshuffle_scale,
    )

    dev = torch.device("cpu")
    dtype = torch.bfloat16 if job["out_dtype"] == "bf16" else torch.float16
    pack = 2 if job["data_format"] == "fp4" else 1
    compiler1 = (
        compile_moe_grouped_gemm1_mxfp4_masked
        if job["data_format"] == "fp4"
        else compile_moe_grouped_gemm1_a8w4_masked
    )
    compiler2 = (
        compile_moe_grouped_gemm2_mxfp4_masked
        if job["data_format"] == "fp4"
        else compile_moe_grouped_gemm2_a8w4_masked
    )
    warp_tile_m = job["tile_m"] // job["m_warp"]
    warp_tile_n = job["tile_n"] // job["n_warp"]
    experts = job["experts"]
    model_dim = job["model_dim"]
    inter_dim = job["inter_dim"]
    max_m = job["max_m"]
    tile_m = job["tile_m"]
    masked_m = torch.full((experts,), max_m, dtype=torch.int32, device=dev)

    def _common(*, grouped_persistent_m: bool, grouped_contiguous_m: bool):
        return dict(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            max_m=max_m,
            tile_m=tile_m,
            tile_n=job["tile_n"],
            tile_k=job["tile_k"],
            m_warp=job["m_warp"],
            n_warp=job["n_warp"],
            out_dtype=job["out_dtype"],
            num_buffers=job["num_buffers"],
            grouped_persistent_m=grouped_persistent_m,
            grouped_contiguous_m=grouped_contiguous_m,
            persistent_workers=job["persistent_workers"],
        )

    def _compile_variant(*, grouped_persistent_m: bool, grouped_contiguous_m: bool):
        common = _common(
            grouped_persistent_m=grouped_persistent_m,
            grouped_contiguous_m=grouped_contiguous_m,
        )
        if grouped_contiguous_m:
            # DeepGEMM-style contiguous-M scheduler flattens all experts into a
            # single (1, M, ...) group; M is rounded up to a tile_m multiple.
            rows = max(tile_m, ((max_m + tile_m - 1) // tile_m) * tile_m)
            groups = 1
        else:
            rows = max_m
            groups = experts

        # stage 1
        y1 = torch.empty((groups, rows, inter_dim), dtype=dtype)
        x1 = torch.empty((groups, rows, model_dim // pack), dtype=torch.uint8)
        w1 = torch.empty(
            (experts, 2 * inter_dim, model_dim // 2), dtype=torch.uint8
        )
        sx1 = torch.empty(
            (groups, *_preshuffled_scale_shape(rows, model_dim, warp_tile_m)),
            dtype=torch.uint8,
        )
        sw1 = torch.empty(
            (experts, *_preshuffled_scale_shape(2 * inter_dim, model_dim, warp_tile_n)),
            dtype=torch.uint8,
        )
        exe1 = compiler1(
            act=job["act"],
            stage1_weight_layout=job["stage1_weight_layout"],
            split_k=job["split_k1"],
            **common,
        )
        exe1(
            y1, x1, w1, sx1, sw1, masked_m,
            max_m, inter_dim, model_dim, experts,
            stream=0,
        )

        # stage 2
        y2 = torch.empty((groups, rows, model_dim), dtype=dtype)
        x2 = torch.empty((groups, rows, inter_dim // pack), dtype=torch.uint8)
        w2 = torch.empty(
            (experts, model_dim, inter_dim // 2), dtype=torch.uint8
        )
        sx2 = torch.empty(
            (groups, *_preshuffled_scale_shape(rows, inter_dim, warp_tile_m)),
            dtype=torch.uint8,
        )
        sw2 = torch.empty(
            (experts, *_preshuffled_scale_shape(model_dim, inter_dim, warp_tile_n)),
            dtype=torch.uint8,
        )
        exe2 = compiler2(split_k=job["split_k2"], **common)
        exe2(
            y2, x2, w2, sx2, sw2, masked_m,
            max_m, model_dim, inter_dim, experts,
            stream=0,
        )

    def _compile_aux_kernels():
        """Compile the route / scatter / psum / preshuffle / gather kernels that
        the runtime grouped-MoE path runs around the two GEMMs.

        These FlyDSL kernels are driven through the very same runtime wrappers the
        inference path uses (so the cache keys match exactly). They are scheduler
        independent (keyed by tensor widths / dims, not persistent-vs-contiguous),
        but the contiguous path is the one that exercises them at runtime, so
        without this pass they JIT-compile on the first large prefill batch.
        """
        topk = int(job.get("topk") or 1)
        wmma_rep_m = warp_tile_m // 16
        scale_k_per_tile = job["tile_k"] // 32
        # Pad max_m to a warp_tile_m multiple, mirroring the runtime so the
        # preshuffle kernels' (max_m % wmma_rep*16 == 0) invariant holds.
        pmax = max(
            warp_tile_m, ((max_m + warp_tile_m - 1) // warp_tile_m) * warp_tile_m
        )
        tok = pmax
        wp1 = model_dim // pack
        ws1 = model_dim // 32
        ws2 = inter_dim // 32

        # Routing: route maps (atomic scatter) + contiguous-M tile prefix sum.
        topk_ids = torch.zeros((tok, topk), dtype=torch.int32, device=dev)
        build_route_maps(topk_ids, experts, pmax)
        contiguous_psum(
            torch.full((experts,), pmax, dtype=torch.int32, device=dev),
            experts,
            tile_m,
        )

        # Pre-stage1 route gather: payload scatter-copy + fused scale preshuffle.
        rows_to_tokens = torch.full(
            (experts * pmax,), -1, dtype=torch.int32, device=dev
        )
        flydsl_moe_scatter_copy_token(
            torch.zeros((tok, wp1), dtype=torch.uint8, device=dev),
            None,
            rows_to_tokens,
            experts,
            pmax,
            grouped_a1=torch.zeros((experts, pmax, wp1), dtype=torch.uint8, device=dev),
        )
        flydsl_moe_scatter_preshuffle_scale(
            torch.zeros((tok, ws1), dtype=torch.uint8, device=dev),
            rows_to_tokens,
            experts,
            pmax,
            wmma_rep=wmma_rep_m,
            scale_k_per_tile=scale_k_per_tile,
        )

        # Pre-stage2 scale preshuffle (already grouped, gather-less variant).
        flydsl_moe_preshuffle_scale(
            torch.zeros((experts, pmax, ws2), dtype=torch.uint8, device=dev),
            experts,
            pmax,
            wmma_rep=wmma_rep_m,
            scale_k_per_tile=scale_k_per_tile,
        )

        # Output epilogue: fused gather-reduce back to token order (bf16/f16).
        flydsl_moe_gather_reduce(
            torch.zeros((experts, pmax, model_dim), dtype=dtype, device=dev),
            torch.zeros((tok, topk), dtype=torch.int32, device=dev),
            torch.zeros((tok, topk), dtype=dtype, device=dev),
            out=torch.zeros((tok, model_dim), dtype=dtype, device=dev),
        )

        # Persistent-M layout: combined masked_m -> (prefix, tile-map) kernel that
        # the runtime persistent path uses (distinct from the GEMM's own
        # internal m-tile-map kernel).
        max_m_tiles = (pmax + tile_m - 1) // tile_m
        _get_compiled_m_tile_prefix_map()(
            torch.full((experts,), pmax, dtype=torch.int32, device=dev),
            torch.empty((experts + 1,), dtype=torch.int32, device=dev),
            torch.empty(experts * max_m_tiles, dtype=torch.int32, device=dev),
            experts,
            pmax,
            tile_m,
            max_m_tiles,
            stream=0,
        )

    with compile_only_env(), _fake_cuda_stream(torch):
        # Persistent-M scheduler: used at decode / small-batch token counts.
        _compile_variant(grouped_persistent_m=True, grouped_contiguous_m=False)
        # DeepGEMM-style contiguous-M scheduler: the runtime auto-switches to
        # this once token_num > AITER_GROUPED_CONTIGUOUS_TOKEN_THRESHOLD (default
        # 512), e.g. on prefill batches. It has a distinct launcher signature
        # (extra m_tile_total / contiguous_m scalars), so it needs its own AOT
        # pass or it falls back to start-up JIT compilation.
        _compile_variant(grouped_persistent_m=False, grouped_contiguous_m=True)
        # Route / scatter / psum / preshuffle / gather kernels around the GEMMs.
        _compile_aux_kernels()


def compile_one_config(**job):
    """Compile one grouped-MoE config and persist to the FlyDSL cache.

    Mirrors aiter.aot.flydsl.moe.compile_one_config: drives compilation with
    ``FakeTensorMode`` + ``COMPILE_ONLY=1`` (no GPU allocation / execution) under
    an ARCH override so the cache can be cross-compiled for gfx1250. Returns a
    dict with timing / arch info; never raises (failures are reported + recorded).
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    aot_arch = os.environ.get("ARCH") or os.environ.get("GPU_ARCHS") or (
        GROUPED_AOT_ARCH_DEFAULT
    )
    shape_str = (
        f"{job.get('kernel_name', 'grouped')}  "
        f"model_dim={job['model_dim']} inter_dim={job['inter_dim']} "
        f"E={job['experts']} max_m={job['max_m']} fmt={job['data_format']}"
    )
    result = {
        "kernel_name": job.get("kernel_name"),
        "shape": shape_str,
        "compile_time": None,
        "compile_arch": aot_arch,
    }

    t0 = time.time()
    try:
        with override_env("ARCH", aot_arch), override_env(
            "FLYDSL_GPU_ARCH", aot_arch
        ), FakeTensorMode():
            _precompile_to_cache(**job)
        result["compile_time"] = time.time() - t0
        print(f"  [OK] compile  {result['compile_time']:6.1f}s  {shape_str}  arch={aot_arch}")
    except Exception as e:
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", default=DEFAULT_CSVS)
    args = parser.parse_args(argv)
    jobs = collect_aot_jobs(args.csv, parse_csv)
    print(f"Grouped MoE AOT: {len(jobs)} config(s)")
    t0 = time.time()
    results = [compile_one_config(**job) for job in jobs]
    ok = sum(1 for r in results if r["compile_time"] is not None)
    fail = len(results) - ok
    print(
        f"\nDone in {time.time() - t0:.1f}s: {ok} ok, {fail} failed "
        f"({len(jobs)} config(s))"
    )
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
