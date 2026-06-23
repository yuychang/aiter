# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from aiter import ActivationType, QuantType
from aiter.fused_moe import moe_sorting
from aiter.ops.opus.moe_stage2 import (
    opus_moe_stage2_bf16,
    opus_moe_stage2_route_reduce_fwd,
)


def _manual_sorted_metadata(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_num, topk = topk_ids.shape
    rows: list[int] = []
    weights: list[float] = []
    expert_ids: list[int] = []

    for expert in range(num_experts):
        expert_rows: list[tuple[int, float]] = []
        for token in range(token_num):
            for slot in range(topk):
                if int(topk_ids[token, slot].item()) == expert:
                    expert_rows.append(
                        (token | (slot << 24), float(topk_weights[token, slot].item()))
                    )
        pad = (-len(expert_rows)) % block_m
        expert_rows.extend([(token_num, 0.0)] * pad)
        for block_start in range(0, len(expert_rows), block_m):
            block = expert_rows[block_start : block_start + block_m]
            if not block:
                continue
            rows.extend(packed for packed, _ in block)
            weights.extend(weight for _, weight in block)
            expert_ids.append(expert)

    device = topk_ids.device
    return (
        torch.tensor(rows, dtype=torch.int32, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
        torch.tensor(expert_ids, dtype=torch.int32, device=device),
        torch.tensor([len(rows)], dtype=torch.int32, device=device),
    )


def _reference_stage2(
    inter_states: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    token_num, topk, _ = inter_states.shape
    _, hidden, _ = w2.shape
    out = torch.zeros(
        token_num, hidden, dtype=torch.float32, device=inter_states.device
    )
    for token in range(token_num):
        for slot in range(topk):
            expert = int(topk_ids[token, slot].item())
            out[token] += topk_weights[token, slot] * (
                inter_states[token, slot].float() @ w2[expert].float().t()
            )
    return out


def _balanced_case(seed: int = 0):
    torch.manual_seed(seed)
    token_num, topk, inter_dim, hidden, experts = 256, 4, 64, 256, 4
    block_m = 256
    inter_states = (
        0.1
        * torch.randn(token_num, topk, inter_dim, dtype=torch.bfloat16, device="cuda")
    ).contiguous()
    w2 = (
        0.1
        * torch.randn(experts, hidden, inter_dim, dtype=torch.bfloat16, device="cuda")
    ).contiguous()
    token_idx = torch.arange(token_num, device="cuda").view(-1, 1)
    slot_idx = torch.arange(topk, device="cuda").view(1, -1)
    topk_ids = ((token_idx * topk + slot_idx) % experts).to(torch.int64)
    topk_weights = torch.full(
        (token_num, topk), 1.0 / topk, dtype=torch.float32, device="cuda"
    )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = (
        _manual_sorted_metadata(topk_ids, topk_weights, experts, block_m)
    )
    assert int(num_valid_ids[0].item()) == token_num * topk
    return (
        inter_states,
        w2,
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        block_m,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
def test_opus_moe_stage2_auto_dispatch_matches_torch_reference():
    (
        inter_states,
        w2,
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        block_m,
    ) = _balanced_case(seed=8)

    actual = opus_moe_stage2_route_reduce_fwd(
        inter_states,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        block_m=block_m,
    )
    expected = _reference_stage2(inter_states, w2, topk_ids, topk_weights)

    assert actual.dtype == torch.bfloat16
    assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
def test_opus_moe_stage2_kid1_matches_torch_reference():
    (
        inter_states,
        w2,
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        block_m,
    ) = _balanced_case(seed=18)
    out = torch.empty(
        inter_states.shape[0], w2.shape[1], dtype=torch.bfloat16, device="cuda"
    )

    actual = opus_moe_stage2_route_reduce_fwd(
        inter_states,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        out=out,
        block_m=block_m,
        kernel_id=1,
    )
    expected = _reference_stage2(inter_states, w2, topk_ids, topk_weights)

    assert actual.dtype == torch.bfloat16
    assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
def test_opus_moe_stage2_bf16_wrapper_returns_bf16():
    (
        inter_states,
        w2,
        _topk_ids,
        _topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        block_m,
    ) = _balanced_case(seed=1)

    actual = opus_moe_stage2_bf16(
        inter_states,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        block_m=block_m,
    )

    assert actual.dtype == torch.bfloat16
    assert tuple(actual.shape) == (inter_states.shape[0], w2.shape[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
def test_opus_moe_stage2_matches_torch_with_aiter_sorting_metadata():
    torch.manual_seed(6)
    token_num, topk, inter_dim, hidden, experts = 256, 4, 64, 256, 4
    block_m = 256
    inter_states = (
        0.1
        * torch.randn(token_num, topk, inter_dim, dtype=torch.bfloat16, device="cuda")
    ).contiguous()
    w2 = (
        0.1
        * torch.randn(experts, hidden, inter_dim, dtype=torch.bfloat16, device="cuda")
    ).contiguous()
    token_idx = torch.arange(token_num, device="cuda").view(-1, 1)
    slot_idx = torch.arange(topk, device="cuda").view(1, -1)
    topk_ids = ((token_idx * topk + slot_idx) % experts).to(torch.int32)
    topk_weights = torch.full(
        (token_num, topk), 1.0 / topk, dtype=torch.float32, device="cuda"
    )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        hidden,
        torch.bfloat16,
        block_size=block_m,
    )
    assert int(num_valid_ids[0].item()) == token_num * topk

    actual = opus_moe_stage2_route_reduce_fwd(
        inter_states,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        block_m=block_m,
        kernel_id=1,
    )
    expected = _reference_stage2(
        inter_states, w2, topk_ids.to(torch.int64), topk_weights
    )

    assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
def test_opus_moe_stage2_hook_matches_default_fused_moe(monkeypatch):
    import aiter.fused_moe as fused_moe_mod

    torch.manual_seed(7)
    token_num, hidden, inter_dim, experts, topk = 256, 256, 64, 4, 4
    block_m = 256
    hidden_states = (
        0.1 * torch.randn(token_num, hidden, dtype=torch.bfloat16, device="cuda")
    ).contiguous()
    w1 = (
        0.1
        * torch.randn(
            experts, 2 * inter_dim, hidden, dtype=torch.bfloat16, device="cuda"
        )
    ).contiguous()
    w2 = (
        0.1
        * torch.randn(experts, hidden, inter_dim, dtype=torch.bfloat16, device="cuda")
    ).contiguous()
    token_idx = torch.arange(token_num, device="cuda").view(-1, 1)
    slot_idx = torch.arange(topk, device="cuda").view(1, -1)
    topk_ids = ((token_idx * topk + slot_idx) % experts).to(torch.int32)
    topk_weights = torch.full(
        (token_num, topk), 1.0 / topk, dtype=torch.float32, device="cuda"
    )

    monkeypatch.setattr(fused_moe_mod, "_USE_OPUS_MOE_STAGE2", False)
    ck_expected = fused_moe_mod.fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
        block_size_M=block_m,
    )
    torch.cuda.synchronize()
    torch_expected = fused_moe_mod.torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids.to(torch.int64),
        activation=ActivationType.Silu,
    )

    monkeypatch.setattr(fused_moe_mod, "_USE_OPUS_MOE_STAGE2", True)
    actual = fused_moe_mod.fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
        block_size_M=block_m,
    )
    torch.cuda.synchronize()

    assert_close(actual.float(), ck_expected.float(), atol=8e-2, rtol=8e-2)
    assert_close(actual.float(), torch_expected.float(), atol=8e-2, rtol=8e-2)
