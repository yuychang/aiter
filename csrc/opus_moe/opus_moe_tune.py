# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stage2-only tuner for the experimental Opus MoE kernels.

This intentionally mirrors the useful parts of ``csrc/opus_gemm`` tuning:
kernel metadata lives in ``opus_moe_common.py``, the tuner sweeps candidate
kids for a shape, writes a profile CSV for all candidates, and writes a tuned
CSV containing one winning row per shape/config key.

It does not use the existing fused_moe ``tuned_fmoe.csv`` format. That file
selects full MoE stage1/stage2 kernel names. This tuner only selects the Opus
MoE stage2 ``kid`` for the current BF16 token-major route-output prototype.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiter.fused_moe import moe_sorting  # noqa: E402
from aiter.jit.utils.chip_info import get_gfx  # noqa: E402
from aiter.ops.opus.moe_stage2 import opus_moe_stage2_route_reduce_fwd  # noqa: E402
from opus_moe_common import (  # noqa: E402
    STAGE2_TUNE_COLUMNS,
    STAGE2_TUNE_KEY_COLUMNS,
    candidate_stage2_kids_for_shape,
    default_stage2_tuned_csv,
)


def _parse_shape(value: str) -> tuple[int, int, int, int, int]:
    parts = tuple(int(x) for x in value.split(","))
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("shape must be T,H,I,E,K")
    return parts


def _parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(x) for x in value.split(",") if x.strip()]


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iters)


def _reference_stage2(
    inter_states: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    token, _, _ = inter_states.shape
    experts, hidden, _ = w2.shape
    out = torch.zeros(token, hidden, dtype=torch.float32, device=inter_states.device)
    for expert in range(experts):
        mask = topk_ids == expert
        if not bool(mask.any()):
            continue
        token_ids = torch.nonzero(mask, as_tuple=False)[:, 0].to(torch.int64)
        partial = inter_states[mask].float() @ w2[expert].float().t()
        partial = partial * topk_weights[mask].view(-1, 1)
        out.index_add_(0, token_ids, partial)
    return out


def _load_shapes(args) -> list[tuple[int, int, int, int, int, int]]:
    shapes: list[tuple[int, int, int, int, int, int]] = []
    block_ms = _parse_int_list(args.block_ms)
    if args.shape:
        for shape in args.shape:
            for block_m in block_ms:
                shapes.append((*shape, block_m))

    if args.input:
        df = pd.read_csv(args.input)
        required = ["token", "model_dim", "inter_dim", "expert", "topk"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{args.input} missing required columns: {missing}")
        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            row_block_ms = (
                [int(row_dict["block_m"])]
                if "block_m" in row_dict and pd.notna(row_dict["block_m"])
                else block_ms
            )
            for block_m in row_block_ms:
                shapes.append(
                    (
                        int(row_dict["token"]),
                        int(row_dict["model_dim"]),
                        int(row_dict["inter_dim"]),
                        int(row_dict["expert"]),
                        int(row_dict["topk"]),
                        int(block_m),
                    )
                )

    if not shapes:
        raise ValueError("provide --shape T,H,I,E,K or --input CSV")

    return list(dict.fromkeys(shapes))


def _make_data(
    token: int,
    hidden: int,
    inter: int,
    experts: int,
    topk: int,
    block_m: int,
    seed: int,
):
    torch.manual_seed(seed)
    inter_states = torch.randn(
        token, topk, inter, dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    w2 = torch.randn(
        experts, hidden, inter, dtype=torch.bfloat16, device="cuda"
    ).contiguous()

    token_idx = torch.arange(token, device="cuda").view(-1, 1)
    slot_idx = torch.arange(topk, device="cuda").view(1, -1)
    topk_ids = ((token_idx * topk + slot_idx) % experts).to(torch.int32)
    topk_weights = torch.full(
        (token, topk), 1.0 / topk, dtype=torch.float32, device="cuda"
    )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        hidden,
        torch.bfloat16,
        block_size=block_m,
    )
    return (
        inter_states,
        w2,
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
    )


def _merge_csv(
    path: str, rows: list[dict], columns: list[str], overwrite: bool
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows, columns=columns)
    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        old_df = pd.read_csv(out_path)
        for col in columns:
            if col not in old_df.columns:
                old_df[col] = None
        new_df = pd.concat([old_df[columns], new_df], ignore_index=True)
    new_df = new_df.drop_duplicates(subset=STAGE2_TUNE_KEY_COLUMNS, keep="last")
    new_df.to_csv(out_path, index=False)


def _write_profile(path: str | None, rows: list[dict], overwrite: bool) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = STAGE2_TUNE_COLUMNS + ["error"]
    new_df = pd.DataFrame(rows, columns=columns)
    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        old_df = pd.read_csv(out_path)
        for col in columns:
            if col not in old_df.columns:
                old_df[col] = None
        new_df = pd.concat([old_df[columns], new_df], ignore_index=True)
    new_df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Opus MoE BF16 stage2 kids")
    parser.add_argument(
        "--shape",
        type=_parse_shape,
        action="append",
        help="Shape as T,H,I,E,K. Can be passed more than once.",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Optional untuned-shape CSV with token,model_dim,inter_dim,expert,topk[,block_m].",
    )
    parser.add_argument("-o", "--output", default=default_stage2_tuned_csv())
    parser.add_argument("-o2", "--profile-output", default="")
    parser.add_argument("--block-ms", default="128")
    parser.add_argument("--kids", default="1")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required")

    arch = get_gfx()
    cu_num = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    requested_kids = _parse_int_list(args.kids)
    shapes = _load_shapes(args)

    tuned_rows: list[dict] = []
    profile_rows: list[dict] = []

    for shape_idx, (token, hidden, inter, experts, topk, block_m) in enumerate(shapes):
        print(
            "\nStart tuning Opus MoE stage2 "
            f"shape=T{token},H{hidden},I{inter},E{experts},K{topk},block_m={block_m}",
            flush=True,
        )
        candidates = candidate_stage2_kids_for_shape(
            model_dim=hidden,
            inter_dim=inter,
            block_m=block_m,
            requested_kids=requested_kids,
        )
        if not candidates:
            print("  no valid candidates for this shape", flush=True)
            continue

        data = _make_data(
            token, hidden, inter, experts, topk, block_m, seed=args.seed + shape_idx
        )
        (
            inter_states,
            w2,
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
        ) = data
        ref = (
            None
            if args.no_check
            else _reference_stage2(inter_states, w2, topk_ids, topk_weights)
        )
        key = {
            "arch": arch,
            "cu_num": cu_num,
            "token": token,
            "model_dim": hidden,
            "inter_dim": inter,
            "expert": experts,
            "topk": topk,
            "block_m": block_m,
        }

        candidate_rows: list[dict] = []
        for inst in candidates:
            out = torch.empty(token, hidden, dtype=torch.bfloat16, device="cuda")
            route_out = torch.empty(
                (token * topk, hidden), dtype=torch.bfloat16, device="cuda"
            )
            row = {
                **key,
                "dtype": inst.dtype,
                "a2_layout": inst.a2_layout,
                "output_mode": inst.output_mode,
                "kid": inst.kid,
                "kernel_name": inst.name,
                "block_n": inst.block_n,
                "block_k": inst.block_k,
                "us": float("inf"),
                "max_abs": float("inf"),
                "mean_abs": float("inf"),
                "valid": 0,
                "error": "",
            }

            def run_candidate() -> None:
                opus_moe_stage2_route_reduce_fwd(
                    inter_states,
                    w2,
                    sorted_ids,
                    sorted_weights,
                    sorted_expert_ids,
                    num_valid_ids,
                    route_out=route_out,
                    out=out,
                    block_m=block_m,
                    kernel_id=inst.kid,
                )

            try:
                us = _bench(run_candidate, args.warmup, args.iters)
                run_candidate()
                torch.cuda.synchronize()
                row["us"] = round(us, 4)
                if ref is not None:
                    diff = (out.float() - ref).abs()
                    max_abs = float(diff.max().item())
                    mean_abs = float(diff.mean().item())
                    ref_abs = float(ref.abs().max().item())
                    ok = max_abs <= args.atol + args.rtol * ref_abs
                    row["max_abs"] = max_abs
                    row["mean_abs"] = mean_abs
                    row["valid"] = int(ok)
                    if not ok:
                        row["error"] = (
                            f"diff too large: max_abs={max_abs:.6e}, ref_abs={ref_abs:.6e}"
                        )
                else:
                    row["max_abs"] = 0.0
                    row["mean_abs"] = 0.0
                    row["valid"] = 1
            except Exception as exc:
                row["error"] = repr(exc)

            profile_rows.append(row)
            candidate_rows.append(row)
            print(
                f"  kid={row['kid']} {row['kernel_name']} "
                f"us={row['us']} valid={row['valid']} max_abs={row['max_abs']}",
                flush=True,
            )

        valid_rows = [row for row in candidate_rows if int(row["valid"]) == 1]
        if not valid_rows:
            print("  no valid candidate passed correctness", flush=True)
            continue
        best = min(valid_rows, key=lambda row: float(row["us"]))
        tuned_rows.append({col: best[col] for col in STAGE2_TUNE_COLUMNS})
        print(
            f"  winner kid={best['kid']} {best['kernel_name']} us={best['us']}",
            flush=True,
        )

    if tuned_rows:
        _merge_csv(args.output, tuned_rows, STAGE2_TUNE_COLUMNS, args.overwrite)
        print(f"\nWrote tuned rows: {args.output}", flush=True)
    else:
        print("\nNo tuned rows produced", flush=True)

    if profile_rows and args.profile_output:
        _write_profile(args.profile_output, profile_rows, args.overwrite)
        print(f"Wrote profile rows: {args.profile_output}", flush=True)


if __name__ == "__main__":
    main()
