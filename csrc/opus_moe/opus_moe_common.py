# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpusMoeStage2Instance:
    kid: int
    name: str
    trait: str
    block_m: int
    block_n: int
    block_k: int
    dtype: str = "bf16"
    a2_layout: str = "token_major"
    output_mode: str = "token_slot_route_output_reduce"
    launcher: str = "opus_moe_stage2_gemmstyle_launch_gfx950"


STAGE2_BF16_KERNELS = {
    1: OpusMoeStage2Instance(
        1,
        "bf16_gemmstyle256x256x64_token_slot_route_out_no_oob_nfast",
        "OpusMoeStage2Bf16GemmStyle256x256x64TokenSlotRouteOutNoOobNFast",
        block_m=256,
        block_n=256,
        block_k=64,
        output_mode="token_slot_route_output_reduce",
    ),
}

DEFAULT_STAGE2_KIDS = tuple(STAGE2_BF16_KERNELS)

STAGE2_KERNELS_BY_DTYPE = {
    "bf16": STAGE2_BF16_KERNELS,
}

STAGE2_TUNE_KEY_COLUMNS = [
    "arch",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "dtype",
    "a2_layout",
    "output_mode",
    "block_m",
]

STAGE2_TUNE_RESULT_COLUMNS = [
    "kid",
    "kernel_name",
    "block_n",
    "block_k",
    "us",
    "max_abs",
    "mean_abs",
    "valid",
]

STAGE2_TUNE_COLUMNS = STAGE2_TUNE_KEY_COLUMNS + STAGE2_TUNE_RESULT_COLUMNS


def default_stage2_tuned_csv() -> str:
    env_path = os.environ.get("OPUS_MOE_STAGE2_TUNED_CSV")
    if env_path:
        return env_path

    for data_dir in (
        Path("/shared/amdgpu/home/hyi_qle/yifehuan_temp/data"),
        Path("/app/yifehuan_temp/data"),
    ):
        if data_dir.exists():
            return str(data_dir / "opus_moe_stage2_tuned.csv")

    return "/tmp/opus_moe_stage2_tuned.csv"


def candidate_stage2_kids_for_shape(
    *,
    model_dim: int,
    inter_dim: int,
    block_m: int,
    dtype: str = "bf16",
    requested_kids: list[int] | tuple[int, ...] | None = None,
) -> list[OpusMoeStage2Instance]:
    kernel_table = STAGE2_KERNELS_BY_DTYPE.get(dtype, {})
    default_kids = tuple(kernel_table)
    kids = list(requested_kids or default_kids)
    instances: list[OpusMoeStage2Instance] = []

    for kid in kids:
        inst = kernel_table.get(int(kid))
        if inst is None:
            continue
        if block_m % inst.block_m != 0:
            continue
        if model_dim % inst.block_n != 0 or inter_dim % inst.block_k != 0:
            continue
        instances.append(inst)

    return instances
