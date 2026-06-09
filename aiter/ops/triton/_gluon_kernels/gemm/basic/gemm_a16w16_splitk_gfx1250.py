# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experimental split-K gluon GEMM for gfx1250 (memory-bound, low-M decode).

At low M the standard kernel launches grid = (cdiv(M,BLOCK_M) * cdiv(N,BLOCK_N)),
so a small-N projection (e.g. the GPTOSS router, N=128) launches only 1-2 CTAs
over the whole K reduction and leaves the device almost idle -- nowhere near the
HBM-bandwidth floor that bounds a memory-bound GEMM. Split-K partitions the K
reduction across NUM_KSPLIT CTAs per output tile (grid is multiplied by the
effective split count), each accumulating a partial result into a
(KSPLIT, M, N) fp32 scratch buffer; a second pass reduces along the split axis.

This trades extra DRAM traffic for the partials (KSPLIT * M * N * 4 bytes, tiny
when M is small) and one extra kernel launch for far higher CTA occupancy, which
is the dominant lever when the kernel is bandwidth-bound and under-occupied.

This module is deliberately separate from gemm_a16w16_gfx1250.py (the committed
API) so it can be iterated on without touching production dispatch.
"""

import math
from typing import Dict, Optional

import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl

from aiter.ops.triton._gluon_kernels.gemm.basic.gemm_a16w16_gfx1250 import (
    create_shared_layouts,
    _get_config,
)
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.activation import _get_activation_from_str
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


@gluon.jit
def _gemm_a16w16_splitk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,  # (KSPLIT, M, N) fp32 partials
    M,
    N,
    K,  # padded to a multiple of BLOCK_K by the caller
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_K_TILES_SPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_mn = num_pid_m * num_pid_n

    # K split is the outer grid dimension; (m, n) tile is the inner index.
    pid_k = pid // num_pid_mn
    pid_mn = pid % num_pid_mn
    pid_m = pid_mn % num_pid_m
    pid_n = pid_mn // num_pid_m

    k_start = pid_k * SPLITK_BLOCK_SIZE
    # Remaining K from this split's start to the (padded) end. The TDM
    # descriptor uses this as its K-dim bound so any tile this split walks past
    # the matrix end zero-fills instead of reading OOB (the last split may reach
    # slightly beyond K when KSPLIT does not evenly divide the tile count).
    k_rem = K - k_start

    a_base = a_ptr + pid_m * BLOCK_M * stride_am + k_start * stride_ak
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn + k_start * stride_bk

    if PHYSICAL_MK:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_base,
            shape=(M - pid_m * BLOCK_M, k_rem),
            strides=(stride_am, stride_ak),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_LAYOUT_A,
        )
    else:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_base,
            shape=(k_rem, M - pid_m * BLOCK_M),
            strides=(stride_ak, stride_am),
            block_shape=(BLOCK_K, BLOCK_M),
            layout=SHARED_LAYOUT_A,
        )

    if PHYSICAL_KN:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_base,
            shape=(k_rem, N - pid_n * BLOCK_N),
            strides=(stride_bk, stride_bn),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_base,
            shape=(N - pid_n * BLOCK_N, k_rem),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_LAYOUT_B,
        )

    if PHYSICAL_MK:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
            layout=SHARED_LAYOUT_A,
        )
    else:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_M],
            layout=SHARED_LAYOUT_A,
        )

    if PHYSICAL_KN:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_N],
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
            layout=SHARED_LAYOUT_B,
        )

    load_idx = 0
    compute_idx = 0

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    # Fill the pipeline
    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )

        if PHYSICAL_MK:
            a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                a_desc, add_offsets=[0, BLOCK_K]
            )
        else:
            a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                a_desc, add_offsets=[BLOCK_K, 0]
            )

        if PHYSICAL_KN:
            b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                b_desc, add_offsets=[BLOCK_K, 0]
            )
        else:
            b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                b_desc, add_offsets=[0, BLOCK_K]
            )

        load_idx += 1

    # Main pipeline loop over this split's K tiles only.
    for _ in range(NUM_K_TILES_SPLIT - (NUM_BUFFERS - 1)):
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)

        if PHYSICAL_MK:
            a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                a_desc, add_offsets=[0, BLOCK_K]
            )
        else:
            a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                a_desc, add_offsets=[BLOCK_K, 0]
            )

        if PHYSICAL_KN:
            b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                b_desc, add_offsets=[BLOCK_K, 0]
            )
        else:
            b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                b_desc, add_offsets=[0, BLOCK_K]
            )

        load_idx += 1

        if PHYSICAL_MK:
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
            )
        else:
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
                OPERAND_LAYOUT_A,
            )

        if PHYSICAL_KN:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_B
            )
        else:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
                OPERAND_LAYOUT_B,
            )

        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    # Epilogue: drain remaining buffered tiles.
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)

        if PHYSICAL_MK:
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
            )
        else:
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
                OPERAND_LAYOUT_A,
            )

        if PHYSICAL_KN:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_B
            )
        else:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
                OPERAND_LAYOUT_B,
            )

        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    # Store this split's partial into c[pid_k, m_tile, n_tile] (fp32). Bias and
    # activation are deferred to the reduce pass.
    offs_cm = pid_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT)
    )
    offs_cn = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
    )

    offs_c = (
        pid_k * stride_ck
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
    )
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    gl.amd.gfx1250.buffer_store(
        accumulator.to(c_ptr.type.element_ty), c_ptr, offs_c, mask=mask_c
    )


def splitk_plan(K, num_ksplit, config):
    """Resolve the launch geometry for a split-K call.

    Returns (effective_ksplit, tiles_per_split, splitk_block_size, K_padded).
    Lets a caller pre-allocate the (effective_ksplit, M, N) fp32 partials buffer
    *outside* a HIP-graph capture region so capture has no allocation.
    """
    BLOCK_K = config["BLOCK_K"]
    K_padded = triton.cdiv(K, BLOCK_K) * BLOCK_K
    num_k_tiles = K_padded // BLOCK_K
    tiles_per_split = max(1, triton.cdiv(num_k_tiles, num_ksplit))
    effective_ksplit = triton.cdiv(num_k_tiles, tiles_per_split)
    return effective_ksplit, tiles_per_split, tiles_per_split * BLOCK_K, K_padded


def gemm_a16w16_splitk(
    x: torch.Tensor,
    w: torch.Tensor,
    num_ksplit: int,
    bias: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[Dict] = None,
    activation: Optional[str] = None,
    y_pp: Optional[torch.Tensor] = None,
):
    """Split-K BF16 GEMM y = x @ w^T (+ bias) on gfx1250.

    ``num_ksplit`` is the requested number of K partitions; the effective count
    is clamped so every split holds at least one BLOCK_K tile. Returns
    (y, effective_ksplit). ``y_pp`` may be a pre-allocated
    (effective_ksplit, M, N) fp32 scratch buffer (for HIP-graph capture).
    """
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert w.dtype in (torch.float16, torch.bfloat16)
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."

    M, K = x.shape
    N, _ = w.shape

    if config is None:
        config, _ = _get_config(M, N, K)

    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]
    NUM_BUFFERS = config.get("NUM_BUFFERS", 2)
    num_warps = config["num_warps"]

    effective_ksplit, tiles_per_split, splitk_block_size, K_padded = splitk_plan(
        K, num_ksplit, config
    )

    # Pad K to a multiple of BLOCK_K so TDM loads never read OOB.
    if K_padded != K:
        pad = K_padded - K
        x = torch.nn.functional.pad(x, (0, pad))
        w = torch.nn.functional.pad(w, (0, pad))
        K = K_padded

    # Pipeline depth cannot exceed the tiles this split actually walks.
    NUM_BUFFERS = max(1, min(NUM_BUFFERS, tiles_per_split))

    w = w.T

    if x.stride(1) == 1:
        physical_mk = True
    elif x.stride(0) == 1:
        physical_mk = False
    else:
        raise ValueError(f"x must be contiguous in one dim, got {x.stride()}")

    if w.stride(1) == 1:
        physical_kn = True
    elif w.stride(0) == 1:
        physical_kn = False
    else:
        raise ValueError(f"w must be contiguous in one dim, got {w.stride()}")

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    # The reduce kernel walks MAX_KSPLIT = next_pow2(effective_ksplit) rows
    # (tl.arange needs a power-of-2 length) and masks rows >= effective_ksplit.
    # Allocate the partials with MAX_KSPLIT rows so those masked reads stay
    # in-bounds; the split kernel only writes the first effective_ksplit rows.
    MAX_KSPLIT = triton.next_power_of_2(effective_ksplit)
    if y_pp is None:
        y_pp = torch.empty((MAX_KSPLIT, M, N), dtype=torch.float32, device=x.device)
    else:
        assert y_pp.shape == (MAX_KSPLIT, M, N), (
            f"y_pp must be {(MAX_KSPLIT, M, N)}, got {tuple(y_pp.shape)}"
        )

    warp_bases = [(0, 1)]
    for i in range(int(math.log2(num_warps // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)

    wmma_layout = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=warp_bases, instr_shape=[16, 16, 32]
    )
    operand_a = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=8)
    operand_b = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=8)

    shared_a, shared_b = create_shared_layouts(
        BLOCK_M, BLOCK_N, BLOCK_K, physical_mk, physical_kn
    )

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m * num_pid_n * effective_ksplit, 1)

    _gemm_a16w16_splitk_kernel[grid](
        x,
        w,
        y_pp,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y_pp.stride(0),
        y_pp.stride(1),
        y_pp.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=NUM_BUFFERS,
        NUM_K_TILES_SPLIT=tiles_per_split,
        SPLITK_BLOCK_SIZE=splitk_block_size,
        PHYSICAL_MK=physical_mk,
        PHYSICAL_KN=physical_kn,
        SHARED_LAYOUT_A=shared_a,
        SHARED_LAYOUT_B=shared_b,
        WMMA_LAYOUT=wmma_layout,
        OPERAND_LAYOUT_A=operand_a,
        OPERAND_LAYOUT_B=operand_b,
        num_warps=num_warps,
    )

    # Reduce the K-split partials into the final output (applies bias/activation).
    # MAX_KSPLIT (= next_pow2(effective_ksplit)) was computed above with the
    # partials allocation so the reduce's masked rows stay in-bounds.
    REDUCE_BLOCK_M = 32
    REDUCE_BLOCK_N = 32
    grid_reduce = (triton.cdiv(M, REDUCE_BLOCK_M), triton.cdiv(N, REDUCE_BLOCK_N))
    _gemm_splitk_reduce_kernel[grid_reduce](
        y_pp,
        y,
        bias,
        M,
        N,
        y_pp.stride(0),
        y_pp.stride(1),
        y_pp.stride(2),
        y.stride(0),
        y.stride(1),
        REDUCE_BLOCK_M,
        REDUCE_BLOCK_N,
        effective_ksplit,
        MAX_KSPLIT,
        ADD_BIAS=(bias is not None),
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        KERNEL_NAME="_gemm_a16w16_splitk_reduce_kernel",
    )

    return y, effective_ksplit
