import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.ops.quant import (
    per_1x32_mx_quant_hip,
    sort_prequantized_mxfp8_for_moe,
)


@pytest.mark.parametrize("tokens", [1, 4, 32])
def test_prequantized_mxfp8_moe_sort(tokens):
    torch.manual_seed(17)
    hidden = 3584
    experts = 896
    topk = 16
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    scores = torch.randn(tokens, experts, device="cuda", dtype=torch.bfloat16)
    weights, ids = fused_topk(x, scores, topk, True)
    sorted_ids, _, _, num_valid_ids, _ = moe_sorting(
        ids, weights, experts, hidden, torch.bfloat16
    )

    expected_q, expected_s = aiter.fused_dynamic_mxfp8_quant_moe_sort(
        x, sorted_ids, num_valid_ids[0], tokens, topk, 64
    )
    x_q, x_s = per_1x32_mx_quant_hip(
        x,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    actual_q, actual_s = sort_prequantized_mxfp8_for_moe(
        x_q, x_s, sorted_ids, num_valid_ids[0], tokens
    )

    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    valid = int(num_valid_ids[0].item())
    mask = expected_s[:valid].view(torch.uint8) != 0
    assert torch.equal(
        actual_s[:valid].view(torch.uint8)[mask],
        expected_s[:valid].view(torch.uint8)[mask],
    )
