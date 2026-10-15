#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Fused gated RMSNorm + FP8 quantization HIP kernels.

Both kernels compute per-head RMSNorm(x, weight, eps) * gate(z) over
[num_tokens, num_heads, head_dim=128] and flatten to
[num_tokens, num_heads * head_dim]:
- gated_rmsnorm_fp8_group_quant: SiLU gate, one FP8 scale per (token, head).
- gated_rmsnorm_fp8_per_token_quant: SiLU or sigmoid gate, one FP8 scale per
  token.
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.gated_rmsnorm_fp8_group_quant import gated_rmsnorm_fp8_group_quant
from aiter.ops.gated_rmsnorm_fp8_per_token_quant import (
    gated_rmsnorm_fp8_per_token_quant,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]  # warp_size=64 kernels
HEAD_DIM = 128  # the only head_dim (and group size) the kernels support

# Per-token gate modes: gate name -> use_sigmoid.
GATES = {"silu": False, "sigmoid": True}


def run_torch(x, z, weight, eps, quant_dtype, per_token, gate="silu"):
    """Reference that matches the fused kernels' fp32 math. Not timed."""
    use_sigmoid = GATES[gate]
    fp8_max = torch.finfo(quant_dtype).max
    num_tokens = x.shape[0]

    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = x.float() * torch.rsqrt(variance + eps) * weight.float().view(1, 1, -1)
    z = z.float()
    gated = normed * (torch.sigmoid(z) if use_sigmoid else z * torch.sigmoid(z))

    # Per-token: one scale over the flattened row. Group: one scale per head.
    gated = gated.reshape(num_tokens, -1) if per_token else gated
    scales = (gated.abs().amax(dim=-1) / fp8_max).clamp_min(1e-10)
    out = (gated / scales.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(quant_dtype)
    return out.reshape(num_tokens, -1), scales


def make_inputs(num_tokens, num_heads, dtype):
    torch.manual_seed(42)
    x = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype)
    z = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype)
    weight = torch.randn(HEAD_DIM, dtype=dtype)
    return x, z, weight


def perf_metrics(x, num_scales, us):
    """Approximate roofline numbers for the elementwise op."""
    elems = x.numel()
    # ~10 flops per element: square-accumulate, normalize, gate, scale.
    flops = 10 * elems
    # read x and z, read weight, write fp8 out, write fp32 scales
    nbytes = 2 * elems * x.element_size() + HEAD_DIM * x.element_size()
    nbytes += elems + num_scales * 4
    return flops / us / 1e6, nbytes / us / 1e6


def check_dequant(name, ref, ref_scales, out, scales, groups):
    num_tokens = ref.shape[0]

    def dequant(q, s):
        return q.float().view(num_tokens, groups, -1) * s.view(num_tokens, -1, 1)

    err = checkAllclose(
        dequant(ref, ref_scales),
        dequant(out, scales),
        rtol=1e-2,
        atol=1e-2,
        msg=f"{name}: dequantized",
    )
    checkAllclose(
        ref_scales.float(), scales.float(), rtol=1e-3, atol=1e-3, msg=f"{name}: scale"
    )
    return err


@benchmark()
def test_gated_rmsnorm_fp8_group_quant(
    num_tokens, num_heads, dtype, transpose_scale, eps=1e-6
):
    x, z, weight = make_inputs(num_tokens, num_heads, dtype)
    ref, ref_scales = run_torch(x, z, weight, eps, dtypes.fp8, per_token=False)

    out = torch.empty(num_tokens, num_heads * HEAD_DIM, dtype=dtypes.fp8)
    # Allocated as [num_tokens, num_heads]; transpose_scale only changes how
    # the kernel lays the values out in it ([num_heads, num_tokens] order).
    scales = torch.empty(num_tokens, num_heads, dtype=dtypes.fp32)

    _, us = run_perftest(
        gated_rmsnorm_fp8_group_quant,
        out,
        scales,
        x,
        z,
        weight,
        eps,
        HEAD_DIM,
        transpose_scale,
    )
    if transpose_scale:
        scales = scales.view(num_heads, num_tokens).t()
    err = check_dequant("hip", ref, ref_scales, out, scales, groups=num_heads)

    tflops, tbps = perf_metrics(x, scales.numel(), us)
    return {
        "gfx": get_gfx(),
        "hip us": us,
        "hip TFLOPS": tflops,
        "hip TB/s": tbps,
        "hip err": err,
    }


@benchmark()
def test_gated_rmsnorm_fp8_per_token_quant(
    num_tokens, num_heads, dtype, gate, eps=1e-6
):
    use_sigmoid = GATES[gate]
    x, z, weight = make_inputs(num_tokens, num_heads, dtype)
    ref, ref_scales = run_torch(x, z, weight, eps, dtypes.fp8, True, gate)

    out = torch.empty(num_tokens, num_heads * HEAD_DIM, dtype=dtypes.fp8)
    scales = torch.empty(num_tokens, dtype=dtypes.fp32)

    _, us = run_perftest(
        gated_rmsnorm_fp8_per_token_quant,
        out,
        scales,
        x,
        z,
        weight,
        eps,
        use_sigmoid,
    )
    err = check_dequant("hip", ref, ref_scales, out, scales, groups=1)

    tflops, tbps = perf_metrics(x, scales.numel(), us)
    return {
        "gfx": get_gfx(),
        "hip us": us,
        "hip TFLOPS": tflops,
        "hip TB/s": tbps,
        "hip err": err,
    }


def summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "gated_rmsnorm_fp8 quant unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Fused gated RMSNorm + FP8 group/per-token quant",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        help="""Input dtype.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (128, 32),
            (256, 32),
            (512, 32),
            (1024, 32),
            (2048, 32),
            (4096, 32),
            (8192, 32),
            (1024, 16),
            (1024, 64),
            (2048, 16),
            (2048, 64),
            (1, 32),  # single decode token
            (3, 32),  # odd token count
            (1024, 1),  # single head
            (1024, 12),  # partial warp (groups_per_warp=8)
            (1024, 40),  # partial warp spanning more than one warp
            (2048, 128),  # max heads
        ],
        help="""(num_tokens, num_heads) pairs; head_dim is always 128.
    e.g.: -s 1024,32""",
    )
    parser.add_argument(
        "-t",
        "--transpose_scale",
        type=dtypes.str2bool,
        nargs="*",
        default=[False, True],
        help="""Group quant scale layouts.
    e.g.: -t 0 1""",
    )
    parser.add_argument(
        "-g",
        "--gate",
        type=str,
        nargs="*",
        choices=list(GATES),
        default=list(GATES),
        help="""Per-token quant gate modes.
    e.g.: -g silu sigmoid""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        summarize(
            "gated_rmsnorm_fp8_group_quant",
            [
                test_gated_rmsnorm_fp8_group_quant(m, h, dtype, t)
                for (m, h), t in itertools.product(args.shape, args.transpose_scale)
            ],
        )
        summarize(
            "gated_rmsnorm_fp8_per_token_quant",
            [
                test_gated_rmsnorm_fp8_per_token_quant(m, h, dtype, g)
                for (m, h), g in itertools.product(args.shape, args.gate)
            ],
        )


if __name__ == "__main__":
    main()
