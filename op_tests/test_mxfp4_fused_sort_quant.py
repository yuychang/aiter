# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for the fused route-sort + MXFP4 input quant kernel."""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import _fused_decode_sort_quant, moe_sorting
from aiter.ops.quant import (
    fused_dynamic_mxfp4_quant_moe_sort,
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant_hip,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm/CUDA device is required"
)

ROUTED_EXPERTS = 384
HIDDEN = 7168
BLOCK_M = 32
ROUTE_VARIANTS = (
    pytest.param(385, 9, id="fused_shared_e385_topk9"),
    pytest.param(384, 8, id="separate_shared_e384_topk8"),
)


def _make_inputs(tokens: int, experts: int, topk: int):
    torch.manual_seed(1000 + tokens)
    hidden = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    if experts == ROUTED_EXPERTS:
        topk_ids = torch.stack(
            [
                torch.randperm(experts, dtype=torch.int32, device="cuda")[:topk]
                for _ in range(tokens)
            ]
        )
        topk_weights = torch.rand(
            (tokens, topk), dtype=torch.float32, device="cuda"
        )
    else:
        routed_ids = torch.stack(
            [
                torch.randperm(ROUTED_EXPERTS, dtype=torch.int32, device="cuda")[
                    : topk - 1
                ]
                for _ in range(tokens)
            ]
        )
        topk_ids = torch.cat(
            (
                routed_ids,
                torch.full(
                    (tokens, 1), ROUTED_EXPERTS, dtype=torch.int32, device="cuda"
                ),
            ),
            dim=1,
        )
        topk_weights = torch.cat(
            (
                torch.rand((tokens, topk - 1), dtype=torch.float32, device="cuda"),
                torch.ones((tokens, 1), dtype=torch.float32, device="cuda"),
            ),
            dim=1,
        )
    return hidden, topk_ids, topk_weights


def _canonical_entries(sorted_ids, sorted_weights, sorted_experts, num_valid_ids):
    valid, tokens = [int(v) for v in num_valid_ids.cpu().tolist()]
    entries = []
    for block, expert in enumerate(sorted_experts[: valid // BLOCK_M].cpu().tolist()):
        for row in range(block * BLOCK_M, block * BLOCK_M + BLOCK_M):
            packed = int(sorted_ids[row].item())
            token = packed & 0x00FFFFFF
            if token != tokens:
                entries.append(
                    (
                        expert,
                        token,
                        (packed >> 24) & 0xFF,
                        float(sorted_weights[row].item()),
                    )
                )
    return sorted(entries)


def _run_fused(hidden, topk_ids, topk_weights, experts):
    tokens = hidden.shape[0]
    route_count = topk_ids.numel()
    active_experts = min(experts, route_count)
    max_sorted = (
        route_count + active_experts * (BLOCK_M - 1) + BLOCK_M - 1
    ) // BLOCK_M * BLOCK_M
    sorted_ids = torch.empty(max_sorted, dtype=torch.int32, device="cuda")
    sorted_weights = torch.empty(max_sorted, dtype=torch.float32, device="cuda")
    sorted_experts = torch.empty(
        (max_sorted + BLOCK_M - 1) // BLOCK_M, dtype=torch.int32, device="cuda"
    )
    num_valid_ids = torch.empty(2, dtype=torch.int32, device="cuda")
    moe_buf = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    activation_quant = torch.empty(
        (tokens, HIDDEN // 2), dtype=dtypes.fp4x2, device="cuda"
    )
    activation_scale_token = torch.empty(
        (tokens, HIDDEN // 32), dtype=dtypes.fp8_e8m0, device="cuda"
    )
    aiter.mxfp4_moe_sort_quant_fwd(
        hidden,
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_experts,
        num_valid_ids,
        moe_buf,
        activation_quant,
        activation_scale_token,
    )
    return (
        sorted_ids,
        sorted_weights,
        sorted_experts,
        num_valid_ids,
        moe_buf,
        activation_quant,
        activation_scale_token,
    )


@pytest.mark.parametrize("experts, topk", ROUTE_VARIANTS)
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 32, 64, 128])
def test_fused_sort_quant_matches_opus_and_generic_quant(tokens, experts, topk):
    hidden, topk_ids, topk_weights = _make_inputs(tokens, experts, topk)

    ref_ids, ref_weights, ref_experts, ref_valid, _ = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        HIDDEN,
        dtypes.bf16,
        BLOCK_M,
    )
    ref_quant, _ = fused_dynamic_mxfp4_quant_moe_sort(
        hidden,
        sorted_ids=ref_ids,
        num_valid_ids=ref_valid,
        token_num=tokens,
        topk=topk,
        block_size=BLOCK_M,
        sorted_weights=ref_weights,
    )
    compact_quant, compact_scale = per_1x32_f4_quant_hip(hidden, shuffle=False)
    fused = _run_fused(hidden, topk_ids, topk_weights, experts)
    torch.cuda.synchronize()

    (
        fused_ids,
        fused_weights,
        fused_experts,
        fused_valid,
        fused_moe_buf,
        fused_quant,
        fused_scale_token,
    ) = fused
    assert _canonical_entries(
        ref_ids, ref_weights, ref_experts, ref_valid
    ) == _canonical_entries(fused_ids, fused_weights, fused_experts, fused_valid)
    torch.testing.assert_close(fused_valid, ref_valid, rtol=0, atol=0)
    torch.testing.assert_close(fused_moe_buf, torch.zeros_like(fused_moe_buf))
    torch.testing.assert_close(
        fused_quant.view(torch.uint8), ref_quant.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        fused_quant.view(torch.uint8), compact_quant.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        fused_scale_token.view(torch.uint8),
        compact_scale.view(torch.uint8),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("tokens", [2, 4])
def test_fused_decode_sort_quant_returns_sorted_scale(tokens):
    hidden, topk_ids, topk_weights = _make_inputs(tokens, 384, 8)
    result = _fused_decode_sort_quant(
        hidden,
        topk_ids,
        topk_weights,
        model_dim=HIDDEN,
        global_experts=384,
        block_m=BLOCK_M,
        moebuf_dtype=torch.bfloat16,
        accumulate=True,
    )
    activation_scale = result[-1]
    assert activation_scale.shape[0] == result[0].shape[0]
    assert activation_scale.shape[1] == HIDDEN // 32
    compact_result = _fused_decode_sort_quant(
        hidden,
        topk_ids,
        topk_weights,
        model_dim=HIDDEN,
        global_experts=384,
        block_m=BLOCK_M,
        moebuf_dtype=torch.bfloat16,
        accumulate=True,
        compact_scale=True,
    )
    compact_scale = compact_result[-1]
    assert compact_scale.shape == (tokens, HIDDEN // 32)


@pytest.mark.parametrize("experts, topk", ROUTE_VARIANTS)
def test_fused_sort_quant_is_cuda_graph_replay_safe(experts, topk):
    hidden, topk_ids, topk_weights = _make_inputs(4, experts, topk)
    # Compile/warm up before capture. The graph itself only records the
    # fixed-address operator launch; all buffers are preallocated.
    _run_fused(hidden, topk_ids, topk_weights, experts)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused = _run_fused(hidden, topk_ids, topk_weights, experts)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()

    assert int(fused[3][1].item()) == 4
    torch.testing.assert_close(fused[4], torch.zeros_like(fused[4]))


def test_flydsl_compact_scale_consumer_matches_conventional_layout():
    if torch.cuda.get_device_name().find("MI355") < 0:
        pytest.skip("compact-scale FlyDSL consumer is currently gfx950-only")

    tokens, hidden_size, inter_dim, experts, topk = 8, 256, 128, 16, 4
    torch.manual_seed(20260715)
    hidden = (
        torch.randn(
            (tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        * 0.1
    )
    topk_ids = torch.stack(
        [
            torch.randperm(experts, dtype=torch.int32, device="cuda")[:topk]
            for _ in range(tokens)
        ]
    )
    topk_weights = torch.rand(
        (tokens, topk), dtype=torch.float32, device="cuda"
    )
    sorted_ids, _, sorted_experts, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        hidden_size,
        dtypes.bf16,
        BLOCK_M,
    )

    activation_quant, compact_scale = per_1x32_f4_quant_hip(
        hidden, shuffle=False
    )
    conventional_scale = mxfp4_moe_sort_fwd(
        compact_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=tokens,
        cols=hidden_size,
    )

    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    weight = (
        torch.randn(
            (experts, 2 * inter_dim, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.1
    )
    weight_quant, weight_scale = quantize(weight, quant_dtype=dtypes.fp4x2)
    weight_quant = weight_quant.view(
        experts, 2 * inter_dim, hidden_size // 2
    )
    weight_quant = shuffle_weight_a16w4(weight_quant, 16, True)
    weight_scale = shuffle_scale_a16w4(weight_scale, experts, True)

    kwargs = dict(
        a=activation_quant,
        w1=weight_quant,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_experts,
        num_valid_ids=num_valid_ids,
        out=None,
        topk=topk,
        tile_m=32,
        tile_n=32,
        tile_k=256,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype="bf16",
        act="silu",
        w1_scale=weight_scale,
        sorted_weights=None,
        use_async_copy=True,
        waves_per_eu=3,
        b_nt=2,
        gate_mode="separated",
        k_wave=1,
    )
    reference = flydsl_moe_stage1(
        a1_scale=conventional_scale,
        a_scale_compact=False,
        **kwargs,
    )
    compact = flydsl_moe_stage1(
        a1_scale=compact_scale,
        a_scale_compact=True,
        **kwargs,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(compact, reference, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = flydsl_moe_stage1(
            a1_scale=compact_scale,
            a_scale_compact=True,
            **kwargs,
        )
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, reference, rtol=0, atol=0)
