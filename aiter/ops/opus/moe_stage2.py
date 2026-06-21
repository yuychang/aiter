# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ...jit.core import compile_ops
from ._arch import _detect_arch

_SUPPORTED_ARCHS = {"gfx950"}
_ARCH_HINT = (
    "opus_moe stage2 currently supports gfx950 only. Set GPU_ARCHS=gfx950 "
    "or run on a matching MI350/MI355 device to use this module."
)
_arch_ok, _detected_arch = _detect_arch(_SUPPORTED_ARCHS)
_OPUS_MOE_STAGE2_AUTO_KERNEL_ID = -1
_OPUS_MOE_STAGE2_BLOCK_M = 256
_OPUS_MOE_STAGE2_BLOCK_N = 256
_OPUS_MOE_STAGE2_BLOCK_K = 64


def _require_supported_arch() -> None:
    if not _arch_ok:
        raise RuntimeError(
            "opus_moe stage2 requires GPU arch in "
            f"{sorted(_SUPPORTED_ARCHS)}; detected {_detected_arch!r}. {_ARCH_HINT}"
        )


def _gen_opus_moe_stage2_route_reduce_fake_tensors(
    inter_states: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Optional[Tensor],
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    route_out: Tensor,
    out: Tensor,
    block_m: int = _OPUS_MOE_STAGE2_BLOCK_M,
    kernel_id: int = _OPUS_MOE_STAGE2_AUTO_KERNEL_ID,
) -> Tensor:
    return out


@compile_ops(
    "module_moe_opus",
    fc_name="opus_moe_stage2_route_reduce_fwd",
    gen_fake=_gen_opus_moe_stage2_route_reduce_fake_tensors,
    develop=True,
)
def _opus_moe_stage2_route_reduce_fwd_raw(
    inter_states: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Optional[Tensor],
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    route_out: Tensor,
    out: Tensor,
    block_m: int = _OPUS_MOE_STAGE2_BLOCK_M,
    kernel_id: int = _OPUS_MOE_STAGE2_AUTO_KERNEL_ID,
) -> Tensor: ...


def opus_moe_stage2_route_reduce_fwd(
    inter_states: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Optional[Tensor],
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    *,
    route_out: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    block_m: int = _OPUS_MOE_STAGE2_BLOCK_M,
    kernel_id: int = _OPUS_MOE_STAGE2_AUTO_KERNEL_ID,
) -> Tensor:
    _require_supported_arch()
    token_num, actual_topk, _ = inter_states.shape
    _, hidden, _ = w2.shape

    if out is None:
        out = torch.empty((token_num, hidden), dtype=torch.bfloat16, device=w2.device)
    if route_out is None:
        route_out = torch.empty(
            (token_num * actual_topk, hidden),
            dtype=torch.bfloat16,
            device=w2.device,
        )

    if not inter_states.is_contiguous():
        inter_states = inter_states.contiguous()
    if not w2.is_contiguous():
        w2 = w2.contiguous()
    if not sorted_token_ids.is_contiguous():
        sorted_token_ids = sorted_token_ids.contiguous()
    if sorted_weights is not None and not sorted_weights.is_contiguous():
        sorted_weights = sorted_weights.contiguous()
    if not sorted_expert_ids.is_contiguous():
        sorted_expert_ids = sorted_expert_ids.contiguous()
    if not num_valid_ids.is_contiguous():
        num_valid_ids = num_valid_ids.contiguous()
    if not route_out.is_contiguous():
        route_out = route_out.contiguous()

    _opus_moe_stage2_route_reduce_fwd_raw(
        inter_states,
        w2,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        route_out,
        out,
        int(block_m),
        int(kernel_id),
    )
    return out


def can_use_opus_moe_stage2_bf16(
    inter_states: Tensor,
    w2: Tensor,
    num_valid_ids: Tensor,
    *,
    token_num: int,
    topk: int,
    block_m: int,
    w2_scale: Optional[Tensor] = None,
    a2_scale: Optional[Tensor] = None,
    has_extra_stage2_args: bool = False,
    expert_mask: Optional[Tensor] = None,
) -> bool:
    """Current fast-path predicate for the no-OOB BF16 stage2 instance."""
    return (
        _arch_ok
        and inter_states.dtype == torch.bfloat16
        and w2.dtype == torch.bfloat16
        and w2_scale is None
        and a2_scale is None
        and not has_extra_stage2_args
        and expert_mask is None
        and block_m % _OPUS_MOE_STAGE2_BLOCK_M == 0
        and inter_states.shape[0] == token_num
        and inter_states.shape[1] == topk
        and inter_states.shape[2] == w2.shape[2]
        and inter_states.shape[2] % _OPUS_MOE_STAGE2_BLOCK_K == 0
        and w2.shape[1] % _OPUS_MOE_STAGE2_BLOCK_N == 0
        and int(num_valid_ids[0].item()) == token_num * topk
    )


def opus_moe_stage2_bf16(
    inter_states: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Optional[Tensor],
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    *,
    block_m: int = _OPUS_MOE_STAGE2_BLOCK_M,
    kernel_id: int = _OPUS_MOE_STAGE2_AUTO_KERNEL_ID,
) -> Tensor:
    """Convenience wrapper returning BF16."""
    return opus_moe_stage2_route_reduce_fwd(
        inter_states,
        w2,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        block_m=block_m,
        kernel_id=kernel_id,
    )


__all__ = [
    "opus_moe_stage2_bf16",
    "opus_moe_stage2_route_reduce_fwd",
]
