# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ...jit.core import compile_ops

_OPUS_MOE_STAGE2_ROUTE_REDUCE_AUTO_BLOCK_N = -1


def _contiguous(tensor: Tensor) -> Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _optional_contiguous(tensor: Optional[Tensor]) -> Optional[Tensor]:
    return None if tensor is None else _contiguous(tensor)


def _gen_opus_moe_stage2_a8w4_decode_fake_tensors(
    inter_states: Tensor,
    w2: Tensor,
    a2_scale: Tensor,
    w2_scale: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Optional[Tensor],
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
) -> Tensor:
    return out


def _gen_opus_moe_stage2_reduce_fake_tensors(
    route_out: Tensor,
    out: Tensor,
    topk: int,
    block_n: int,
) -> Tensor:
    return out


@compile_ops(
    "module_moe_opus",
    fc_name="opus_moe_stage2_a8w4_decode_fwd",
    gen_fake=_gen_opus_moe_stage2_a8w4_decode_fake_tensors,
    develop=True,
)
def _opus_moe_stage2_a8w4_decode_fwd_raw(
    inter_states: Tensor,
    w2: Tensor,
    a2_scale: Tensor,
    w2_scale: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Optional[Tensor],
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    block_m: int,
    kernel_id: int,
    inter_dim_pad: int,
) -> Tensor: ...


@compile_ops(
    "module_moe_opus",
    fc_name="opus_moe_stage2_reduce_token_slot_route_output_fwd",
    gen_fake=_gen_opus_moe_stage2_reduce_fake_tensors,
    develop=True,
)
def _opus_moe_stage2_reduce_token_slot_route_output_fwd_raw(
    route_out: Tensor,
    out: Tensor,
    topk: int,
    block_n: int,
) -> Tensor: ...


def opus_moe_stage2_a8w4_decode_fwd(
    inter_states: Tensor,
    w2: Tensor,
    a2_scale: Tensor,
    w2_scale: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Optional[Tensor],
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    *,
    block_m: int,
    inter_dim_pad: int,
    out: Optional[Tensor] = None,
    kernel_id: int = -1,
    return_per_slot: bool = False,
) -> Tensor:
    if out is None:
        shape = (
            (inter_states.shape[0], inter_states.shape[1], w2.shape[1])
            if return_per_slot
            else (inter_states.shape[0], w2.shape[1])
        )
        alloc = torch.empty if return_per_slot else torch.zeros
        out = alloc(shape, dtype=torch.bfloat16, device=w2.device)

    kernel_out = out.view(-1, w2.shape[1]) if return_per_slot else out
    if return_per_slot and kernel_id == -1:
        kernel_id = 2040

    _opus_moe_stage2_a8w4_decode_fwd_raw(
        _contiguous(inter_states),
        _contiguous(w2),
        _contiguous(a2_scale),
        _contiguous(w2_scale),
        _contiguous(sorted_token_ids),
        _optional_contiguous(sorted_weights),
        _contiguous(sorted_expert_ids),
        _contiguous(num_valid_ids),
        kernel_out,
        int(block_m),
        int(kernel_id),
        int(inter_dim_pad),
    )
    return out


def opus_moe_stage2_reduce_token_slot_route_output_fwd(
    route_out: Tensor,
    out: Optional[Tensor] = None,
    *,
    topk: int | None = None,
    block_n: int | None = None,
) -> Tensor:
    if route_out.dim() != 3:
        raise ValueError(
            f"route_out must be [token, topk, hidden], got {tuple(route_out.shape)}"
        )
    if topk is None:
        topk = int(route_out.shape[1])
    if out is None:
        out = torch.empty(
            (route_out.shape[0], route_out.shape[2]),
            dtype=route_out.dtype,
            device=route_out.device,
        )
    if block_n is None:
        block_n = _OPUS_MOE_STAGE2_ROUTE_REDUCE_AUTO_BLOCK_N
    _opus_moe_stage2_reduce_token_slot_route_output_fwd_raw(
        _contiguous(route_out),
        _contiguous(out),
        int(topk),
        int(block_n),
    )
    return out


__all__ = [
    "opus_moe_stage2_a8w4_decode_fwd",
    "opus_moe_stage2_reduce_token_slot_route_output_fwd",
]
