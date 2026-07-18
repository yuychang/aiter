# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused MoE finalize + shared-expert add (ROCm analog of CUDA moe_finalize_fuse_shared).

The CUDA `flashinfer_trtllm` decode path runs the routed MoE with
`do_finalize=False`, computes the shared expert concurrently, then combines both
in a single `moe_finalize_fuse_shared` kernel:

    out[m] = routed_scaling_factor * sum_k routed_permuted[scatter_index[m, k]]
             + shared_output[m]

This module provides the same fused combine for the ROCm/AITER path as one
Triton kernel, so the routed weighted-combine, the routed scaling, and the
shared-expert add do not cost three separate passes over the [M, N] hidden state.

Contract (self-defined, unambiguous — see `moe_finalize_shared_ref`):
  - `routed_permuted`  : [P, N]      per-expert-slot routed outputs (sorted/permuted).
  - `scatter_index`    : [M, K] int  row in `routed_permuted` for token m's k-th
                          expert; a negative entry means "no expert" (skipped).
  - `shared_output`    : [M, N] | None  shared-expert output, added un-scaled.
  - `routed_scaling_factor` : float  scale applied to the routed sum only.

The routed sum is scaled, the shared output is added un-scaled — matching
SGLang's `shared.add_(routed, alpha=routed_scaling_factor)` semantics.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils.types import torch_to_triton_dtype


@triton.jit
def _moe_finalize_shared_kernel(
    routed_ptr,  # [P, N]
    scatter_ptr,  # [M, K] int32
    weight_ptr,  # [M, K] or dummy
    shared_ptr,  # [M, N] or dummy
    out_ptr,  # [M, N]
    stride_rp,
    stride_rn,
    stride_sm,
    stride_sk,
    stride_wm,
    stride_wk,
    stride_shm,
    stride_shn,
    stride_om,
    stride_on,
    N,
    K: tl.constexpr,
    alpha,
    HAS_WEIGHTS: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # One program per (token m, N-block). Gather the K routed slots for token m,
    # accumulate (optionally weighted) in fp32, scale, add the shared output, store.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in tl.static_range(K):
        idx = tl.load(scatter_ptr + pid_m * stride_sm + k * stride_sk)
        if idx >= 0:
            row = tl.load(
                routed_ptr + idx * stride_rp + offs_n * stride_rn,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            if HAS_WEIGHTS:
                w = tl.load(weight_ptr + pid_m * stride_wm + k * stride_wk).to(
                    tl.float32
                )
                row = row * w
            acc += row

    acc = acc * alpha

    if HAS_SHARED:
        sh = tl.load(
            shared_ptr + pid_m * stride_shm + offs_n * stride_shn,
            mask=mask_n,
            other=0.0,
        )
        acc += sh.to(tl.float32)

    tl.store(
        out_ptr + pid_m * stride_om + offs_n * stride_on,
        acc.to(OUT_DTYPE),
        mask=mask_n,
    )


def moe_finalize_shared(
    routed_permuted: torch.Tensor,
    scatter_index: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    routed_scaling_factor: float = 1.0,
    topk_weights: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    block_n: int = 1024,
) -> torch.Tensor:
    """Fused routed grouped-combine + routed scaling + shared-expert add.

    out[m] = routed_scaling_factor
             * sum_k (topk_weights[m, k] if given else 1) * routed_permuted[scatter_index[m, k]]
             + (shared_output[m] if shared_output is not None else 0)

    Pass ``topk_weights`` when the per-slot routed outputs are *unweighted* (the
    combine that would apply the routing weight was skipped, e.g. AITER
    ``no_combine`` output). Pass ``None`` when the slots are already weighted.
    Negative ``scatter_index`` entries are skipped (padding / no-expert slots).
    """
    assert routed_permuted.ndim == 2, "routed_permuted must be [P, N]"
    assert scatter_index.ndim == 2, "scatter_index must be [M, K]"
    M, K = scatter_index.shape
    N = routed_permuted.shape[1]

    if out_dtype is None:
        out_dtype = (
            shared_output.dtype if shared_output is not None else routed_permuted.dtype
        )
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype, device=routed_permuted.device)

    scatter_index = scatter_index.to(torch.int32)
    has_shared = shared_output is not None
    shared_arg = shared_output if has_shared else out  # dummy strides when unused
    has_weights = topk_weights is not None
    weight_arg = topk_weights if has_weights else scatter_index  # dummy when unused

    grid = (M, triton.cdiv(N, block_n))
    _moe_finalize_shared_kernel[grid](
        routed_permuted,
        scatter_index,
        weight_arg,
        shared_arg,
        out,
        routed_permuted.stride(0),
        routed_permuted.stride(1),
        scatter_index.stride(0),
        scatter_index.stride(1),
        weight_arg.stride(0),
        weight_arg.stride(1),
        shared_arg.stride(0),
        shared_arg.stride(1),
        out.stride(0),
        out.stride(1),
        N,
        K=K,
        alpha=float(routed_scaling_factor),
        HAS_WEIGHTS=has_weights,
        HAS_SHARED=has_shared,
        BLOCK_N=block_n,
        OUT_DTYPE=torch_to_triton_dtype[out_dtype],
    )
    return out


def moe_finalize_shared_ref(
    routed_permuted: torch.Tensor,
    scatter_index: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    routed_scaling_factor: float = 1.0,
    topk_weights: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Torch reference for `moe_finalize_shared` (fp32 math, cast back)."""
    M, K = scatter_index.shape
    N = routed_permuted.shape[1]
    if out_dtype is None:
        out_dtype = (
            shared_output.dtype if shared_output is not None else routed_permuted.dtype
        )
    idx = scatter_index.to(torch.long)
    valid = idx >= 0
    gathered = routed_permuted.to(torch.float32)[idx.clamp(min=0)]  # [M, K, N]
    if topk_weights is not None:
        gathered = gathered * topk_weights.to(torch.float32)[..., None]
    gathered = gathered * valid.to(torch.float32)[..., None]
    acc = gathered.sum(dim=1) * routed_scaling_factor  # [M, N]
    if shared_output is not None:
        acc = acc + shared_output.to(torch.float32)
    return acc.to(out_dtype)
