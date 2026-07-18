# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf for the fused MoE finalize + shared-expert add kernel.

`moe_finalize_shared` is the ROCm analog of CUDA's `moe_finalize_fuse_shared`:
one kernel does the routed grouped weighted-combine, the routed scaling, and the
shared-expert add. See `aiter/ops/triton/moe/finalize_shared.py`.
"""

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.moe.finalize_shared import (
    moe_finalize_shared,
    moe_finalize_shared_ref,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]


def _build_inputs(m, k, n, e, dtype, drop_frac=0.0):
    """Build a routed permuted output + scatter index + shared output.

    Shapes mirror the deferred-finalize decode path:
      - routed_permuted [P, N] with P = m * k permuted expert-slot rows.
      - scatter_index   [M, K] mapping each token/expert to a permuted row; a
        `drop_frac` fraction of entries is set to -1 (padding / no-expert).
      - shared_output   [M, N].
    """
    p = m * k
    routed_permuted = torch.randn((p, n), dtype=dtype) * 0.1
    # Each token's k experts map to k distinct permuted rows; here row = m*k + j.
    scatter_index = torch.arange(p, dtype=torch.int32).reshape(m, k)
    if drop_frac > 0.0:
        drop = torch.rand((m, k)) < drop_frac
        scatter_index = torch.where(
            drop, torch.full_like(scatter_index, -1), scatter_index
        )
    shared_output = torch.randn((m, n), dtype=dtype) * 0.1
    return routed_permuted, scatter_index, shared_output


@benchmark()
def test_moe_finalize_shared(m, k, n, e, dtype, with_shared, with_weights):
    routed_permuted, scatter_index, shared_output = _build_inputs(
        m, k, n, e, dtype, drop_frac=0.0
    )
    shared = shared_output if with_shared else None
    # AITER no_combine returns unweighted slots; pass topk_weights (which for
    # Kimi already fold in routed_scaling, hence alpha=1 in the real call). Here
    # we exercise both the weighted and unweighted contracts with alpha=2.5.
    topk_weights = torch.rand((m, k), dtype=torch.float32) if with_weights else None
    routed_scaling_factor = 2.5

    ref = moe_finalize_shared_ref(
        routed_permuted, scatter_index, shared, routed_scaling_factor, topk_weights
    )

    candidates = {
        "triton": lambda: moe_finalize_shared(
            routed_permuted,
            scatter_index,
            shared,
            routed_scaling_factor,
            topk_weights,
        ),
    }

    # bytes: read k routed rows + (optional) shared, write out; plus index reads.
    read_rows = m * k + (m if with_shared else 0)
    nbytes = (read_rows + m) * n * routed_permuted.element_size()
    # flops: k adds per output element + scale (+ shared add) ~ (k + 1) * m * n
    flops = (k + (1 if with_shared else 0)) * m * n

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: moe_finalize_shared",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "moe_finalize_shared unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=["bf16"],
        help="data type, e.g. -d bf16",
    )
    parser.add_argument(
        "-m",
        "--tokens",
        type=int,
        nargs="*",
        # Kimi-K2.5 TP4 decode: M = concurrency (1 token/req). ISL8k sweep conc 4..64.
        default=[4, 8, 16, 32, 64],
        help="number of decode tokens (concurrency)",
    )
    parser.add_argument(
        "-k",
        "--topk",
        type=int,
        nargs="*",
        default=[8],  # Kimi-K2.5 routes to 8 experts
        help="experts per token",
    )
    parser.add_argument(
        "-n",
        "--hidden",
        type=int,
        nargs="*",
        default=[7168],  # Kimi-K2.5 hidden size
        help="hidden dim N",
    )
    parser.add_argument(
        "-e",
        "--experts",
        type=int,
        nargs="*",
        default=[384],  # Kimi-K2.5 routed experts (metadata column only)
        help="num routed experts",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for m, k, n, e, with_shared, with_weights in itertools.product(
            args.tokens, args.topk, args.hidden, args.experts, [True, False], [True, False]
        ):
            df.append(
                test_moe_finalize_shared(m, k, n, e, dtype, with_shared, with_weights)
            )
        df = pd.DataFrame(df)
        aiter.logger.info(
            "moe_finalize_shared summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
