# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Packed-query Gluon MLA decode for speculative-decoding verify (gfx950, fp8 KV).
#
# mla_gluon's bh16bn128 regime runs one program per (request, KV split, query
# position), so a request's KV cache is streamed once per verify token and 12
# heads fill a 16-row tile. Here one program owns BLOCK_M=64 rows packed as
# (query position, head) pairs, e.g. 4 positions x 12 heads, and streams each KV
# tile once for all of them:
#
#   qlen 8, 12 heads: 2 row blocks of 4 positions -> KV read 2x instead of 8x
#   qlen 4, 12 heads: 1 row block                 -> KV read 1x instead of 4x
#
# Row r of block m is query position m * Q_PER_BLOCK + r // NHEAD and head
# r % NHEAD. Rows past the last (position, head) pair are zero-Q padding and
# are never stored.
#
# Math is the bh16bn128 math: bf16 Q, fp8 KV converted to bf16 before each mfma,
# the KV scale folded into the QK temperature and applied once to acc, fp32
# accumulation. Stage-1 writes the same [B, QLEN, H, SPLIT, D] partials and
# [B, QLEN, H, SPLIT] natural-log lse as mla_gluon, so its stage-2
# _mla_softmax_reducev_kernel (same split partition, BLOCK_N=64) merges them.
#
# Causal verify mask: position q sees KV [0, seq_len - QLEN + q]. Only tiles
# that reach past min(split end, first causal limit of the block) are masked;
# every other tile runs the unmasked step.
#
# Pipeline (BLOCK_N=64; KV and the KV slot indices double-buffered in LDS, the
# indices one tile ahead of KV, all moved by async copies):
#   iter i: wait copies -> barrier -> async copy KV[i+1], indices[i+2]
#           -> QK mfma, softmax, PV mfma on KV[i]
#
# Warps tile M (4 x 16 rows) so the softmax is warp-local; every warp reads the
# whole KV tile from LDS, which is why BLOCK_N is 64 rather than 128 (a
# [512, 128] B operand per warp does not fit next to the [16, 512] fp32 acc).

import os

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.gluon.mla_gluon import _mla_softmax_reducev_kernel
from aiter.ops.triton.utils._triton import arch_info

PACKED_Q_ENV = "AITER_MLA_GLUON_PACKED_Q"
_BLOCK_M = 64
_BLOCK_N = 64
_DEFAULT_WG_BUDGET = 256


def mla_gluon_packed_q_enabled() -> bool:
    return os.environ.get(PACKED_Q_ENV, "0") == "1"


# fmt: off
@gluon.jit
def _packed_q_step(
    q_nope, q_pe, acc, e_max, e_sum,
    kv_buf, kpe_buf, tile_lo, row_limit,
    qk_scale,
    MASKED: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    mfma_layout: gl.constexpr,
    mfma_layout_a: gl.constexpr,
    mfma_layout_b: gl.constexpr,
    linear_v: gl.constexpr,
):
    dtype: gl.constexpr = q_nope.dtype
    k_c = gl.amd.cdna4.async_copy.load_shared_relaxed(kv_buf, mfma_layout_b)
    zeros = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)
    k_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(kpe_buf, mfma_layout_b)
    qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)
    # qk_scale carries log2(e): e_max / e_sum live in base 2 until the epilogue.
    qk *= qk_scale
    if MASKED:
        offs_n = tile_lo + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
        qk = gl.where(offs_n[None, :] < row_limit[:, None], qk, float("-inf"))
    n_e_max = gl.maximum(gl.max(qk, 1), e_max)
    m_shift = n_e_max
    if MASKED:
        # A row with no visible token in this split (only possible in a short
        # last split) keeps e_max == -inf; shift by 0 so re_scale and p are 0,
        # not NaN, and the split ends with e_sum == 0 -> lse == -inf.
        m_shift = gl.where(n_e_max == float("-inf"), 0.0, n_e_max)
    re_scale = gl.exp2(e_max - m_shift)
    p = gl.exp2(qk - m_shift[:, None])
    e_sum = e_sum * re_scale + gl.sum(p, 1)
    acc *= re_scale[:, None]
    p = gl.convert_layout(p.to(dtype), mfma_layout_a)
    v_c = gl.amd.cdna4.async_copy.load_shared_relaxed(kv_buf, linear_v)
    v_c = gl.permute(v_c.to(dtype), [1, 0])
    v_c = gl.convert_layout(v_c, mfma_layout_b)
    acc = gl.amd.cdna4.mfma(p, v_c, acc)
    return acc, n_e_max, e_sum


@gluon.jit
def _packed_q_issue_kv(
    bufs_kv, bufs_kpe, kv_slot, idx_buf,
    Kv_c_cache, K_pe_cache,
    tile_lo, split_kv_end,
    offs_tok_kv, offs_d_kv, offs_tok_pe, offs_d_pe,
    stride_kv_c_bs, stride_k_pe_bs,
    KV_PE_OFFSET: gl.constexpr,
    WITHIN_2GB: gl.constexpr,
):
    # KV slot index of each token of the tile, staged in LDS by _packed_q_issue_idx.
    loc_kv = gl.amd.cdna4.async_copy.load_shared_relaxed(idx_buf, offs_tok_kv.type.layout)
    loc_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(idx_buf, offs_tok_pe.type.layout)
    offs_k_c = loc_kv[None, :] * stride_kv_c_bs + offs_d_kv[:, None]
    mask_kv = (tile_lo + offs_tok_kv)[None, :] < split_kv_end
    offs_k_pe = loc_pe[None, :] * stride_k_pe_bs + offs_d_pe[:, None] + KV_PE_OFFSET
    mask_pe = (tile_lo + offs_tok_pe)[None, :] < split_kv_end
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv.index(kv_slot), Kv_c_cache, offs_k_c, mask=mask_kv)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kpe.index(kv_slot), K_pe_cache, offs_k_pe, mask=mask_pe)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv.index(kv_slot), Kv_c_cache + offs_k_c, mask=mask_kv, other=0.0)
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kpe.index(kv_slot), K_pe_cache + offs_k_pe, mask=mask_pe, other=0.0)


@gluon.jit
def _packed_q_issue_idx(idx_buf, Kv_indices, batch_page_start, tile_lo, split_kv_end, offs_tok):
    tok = tile_lo + offs_tok
    gl.amd.cdna4.async_copy.buffer_load_to_shared(idx_buf, Kv_indices, batch_page_start + tok, mask=tok < split_kv_end)


@gluon.jit
def _mla_gluon_packed_q(
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    Kv_indices,
    Seq_info,
    O,
    sm_scale,
    kv_scale,
    stride_q_nope_b,
    stride_q_nope_s,
    stride_q_nope_h,
    stride_q_pe_b,
    stride_q_pe_s,
    stride_q_pe_h,
    stride_kv_c_bs,
    stride_k_pe_bs,
    stride_indices_bs,
    stride_o_b,
    stride_o_s,
    stride_o_h,
    stride_o_split,
    Mid_lse,
    stride_mid_lse_b,
    stride_mid_lse_s,
    stride_mid_lse_h,
    stride_mid_lse_split,
    Final_lse,
    stride_final_lse_b,
    stride_final_lse_s,
    stride_final_lse_h,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    HEAD_DIM_CKV: gl.constexpr,
    HEAD_DIM_KPE: gl.constexpr,
    KV_PE_OFFSET: gl.constexpr,
    USE_2D_VIEW: gl.constexpr,
    WITHIN_2GB: gl.constexpr,
    NHEAD: gl.constexpr,
    QLEN: gl.constexpr,
    Q_PER_BLOCK: gl.constexpr,
    RETURN_LSE: gl.constexpr,
):
    gl.static_assert(BLOCK_M == 64)
    gl.static_assert(BLOCK_N == 64)
    gl.static_assert(Q_PER_BLOCK * NHEAD <= BLOCK_M)

    cur_batch = gl.program_id(0)
    split_kv_id = gl.program_id(1)
    m_block = gl.program_id(2)

    # USE_2D_VIEW=True: Kv_indices = block_table[batch, max_seqlen], Seq_info = cache_seqlens[batch]
    # USE_2D_VIEW=False: Kv_indices = kv_indices[total_kv],          Seq_info = kv_indptr[batch+1]
    if USE_2D_VIEW:
        batch_page_start = stride_indices_bs * cur_batch
        cur_batch_seq_len = gl.load(Seq_info + cur_batch)
    else:
        batch_page_start = gl.load(Seq_info + cur_batch)
        cur_batch_seq_len = gl.load(Seq_info + cur_batch + 1) - batch_page_start

    # Same partition as mla_gluon / _mla_softmax_reducev_kernel.
    kv_len_per_split = gl.maximum(BLOCK_N, cur_batch_seq_len // NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    if split_kv_id == NUM_KV_SPLITS - 1:
        split_kv_end = cur_batch_seq_len
    if split_kv_start >= split_kv_end:
        return
    num_iter = gl.cdiv(split_kv_end - split_kv_start, BLOCK_N)

    if not WITHIN_2GB:
        stride_kv_c_bs = stride_kv_c_bs.to(gl.int64)
        stride_k_pe_bs = stride_k_pe_bs.to(gl.int64)

    ######### layouts #########
    # M side: 4 warps tile the 64 rows (as mla_gluon's bh64 regime).
    blocked_q_nope: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    shared_q_nope: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
        cga_layout=[],
        shape=[64, 512]
    )
    blocked_q_pe: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (32, 0)),
        lane_bases=((0, 8), (0, 16), (0, 32), (4, 0), (8, 0), (16, 0)),
        warp_bases=((1, 0), (2, 0)),
        block_bases=[],
        shape=[64, 64],
    )
    shared_q_pe: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [4, 0], [8, 0], [16, 0], [1, 0], [2, 0], [32, 0]],
        cga_layout=[],
        shape=[64, 64]
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    # KV side: fp8 [512, 64] / [64, 64] tiles, 16 B per lane per async copy
    # (the first half of mla_gluon's bh16bn128 [*, 128] tiles).
    blocked_kv: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 4), (0, 32)),
        lane_bases=((16, 0), (32, 0), (64, 0), (128, 0), (256, 0), (0, 16)),
        warp_bases=((0, 1), (0, 2)),
        block_bases=[],
        shape=[512, 64],
    )
    shared_kv: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[1024, 32], [8192, 16]],
        offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0], [0, 16], [0, 1], [0, 2], [0, 8], [0, 4], [0, 32]],
        cga_layout=[],
        shape=[512, 64]
    )
    blocked_kpe: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1, 0), (2, 0), (4, 0), (8, 0)),
        lane_bases=((16, 0), (32, 0), (0, 4), (0, 8), (0, 16), (0, 32)),
        warp_bases=((0, 1), (0, 2)),
        block_bases=[],
        shape=[64, 64],
    )
    shared_kpe: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[2048, 16]],
        offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16], [0, 32], [0, 1], [0, 2]],
        cga_layout=[],
        shape=[64, 64]
    )
    # V read for the PV mfma; warps broadcast because each warp needs all of V.
    linear_v: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
        lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
        warp_bases=((0, 0), (0, 0)),
        block_bases=[],
        shape=[512, 64],
    )
    # KV slot indices: one int32 per lane, warps write the same copy.
    blocked_idx: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[],
        lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
        warp_bases=((0,), (0,)),
        block_bases=[],
        shape=[64],
    )
    shared_idx: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)
    dtype = Q_nope.type.element_ty
    kvtype = Kv_c_cache.type.element_ty
    LOG2E: gl.constexpr = 1.4426950408889634
    LN2: gl.constexpr = 0.6931471805599453

    ######### Q: packed (position, head) rows #########
    buf_q_nope = gl.allocate_shared_memory(dtype, shape=[BLOCK_M, HEAD_DIM_CKV], layout=shared_q_nope)
    buf_q_pe = gl.allocate_shared_memory(dtype, shape=[BLOCK_M, HEAD_DIM_KPE], layout=shared_q_pe)

    rows_q = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_q_nope))
    q_pos_q = m_block * Q_PER_BLOCK + rows_q // NHEAD
    row_ok_q = (rows_q < Q_PER_BLOCK * NHEAD) & (q_pos_q < QLEN)
    offs_d_ckv = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, blocked_q_nope))
    offs_q_nope = (cur_batch * stride_q_nope_b + q_pos_q * stride_q_nope_s + (rows_q % NHEAD) * stride_q_nope_h)[:, None] + offs_d_ckv[None, :]
    gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_nope, Q_nope, offs_q_nope, mask=row_ok_q[:, None])

    rows_qpe = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_q_pe))
    q_pos_qpe = m_block * Q_PER_BLOCK + rows_qpe // NHEAD
    row_ok_qpe = (rows_qpe < Q_PER_BLOCK * NHEAD) & (q_pos_qpe < QLEN)
    offs_d_kpe = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(0, blocked_q_pe))
    offs_q_pe = (cur_batch * stride_q_pe_b + q_pos_qpe * stride_q_pe_s + (rows_qpe % NHEAD) * stride_q_pe_h)[:, None] + offs_d_kpe[None, :]
    gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_pe, Q_pe, offs_q_pe, mask=row_ok_qpe[:, None])
    gl.amd.cdna4.async_copy.commit_group()

    ######### KV prologue #########
    # Indices run one tile ahead of KV: copy KV[0] needs indices[0] in LDS
    # first, and every iteration i copies KV[i+1] and indices[i+2].
    bufs_kv = gl.allocate_shared_memory(kvtype, shape=[2, HEAD_DIM_CKV, BLOCK_N], layout=shared_kv)
    bufs_kpe = gl.allocate_shared_memory(kvtype, shape=[2, HEAD_DIM_KPE, BLOCK_N], layout=shared_kpe)
    bufs_idx = gl.allocate_shared_memory(gl.int32, shape=[2, BLOCK_N], layout=shared_idx)
    offs_tok_kv = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kv))
    offs_d_kv = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv))
    offs_tok_pe = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe))
    offs_d_pe = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
    offs_tok_idx = gl.arange(0, BLOCK_N, layout=blocked_idx)

    _packed_q_issue_idx(bufs_idx.index(0), Kv_indices, batch_page_start, split_kv_start, split_kv_end, offs_tok_idx)
    _packed_q_issue_idx(bufs_idx.index(1), Kv_indices, batch_page_start, split_kv_start + BLOCK_N, split_kv_end, offs_tok_idx)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.barrier()
    _packed_q_issue_kv(bufs_kv, bufs_kpe, 0, bufs_idx.index(0), Kv_c_cache, K_pe_cache, split_kv_start, split_kv_end,
                       offs_tok_kv, offs_d_kv, offs_tok_pe, offs_d_pe, stride_kv_c_bs, stride_k_pe_bs,
                       KV_PE_OFFSET, WITHIN_2GB)
    gl.amd.cdna4.async_copy.commit_group()

    q_nope = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q_nope, mfma_layout_a)
    q_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q_pe, mfma_layout_a)

    e_max = gl.zeros([BLOCK_M], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout)) - float("inf")
    e_sum = gl.zeros([BLOCK_M], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout))
    acc = gl.zeros([BLOCK_M, HEAD_DIM_CKV], dtype=gl.float32, layout=mfma_layout)
    qk_scale = sm_scale * kv_scale * LOG2E

    # Per-row visible-KV bound in this split. Padding rows see the whole split
    # so they never hit the all-masked path; they are not stored.
    rows_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout))
    q_pos_m = m_block * Q_PER_BLOCK + rows_m // NHEAD
    row_ok_m = (rows_m < Q_PER_BLOCK * NHEAD) & (q_pos_m < QLEN)
    causal_end_m = cur_batch_seq_len - QLEN + 1 + q_pos_m
    row_limit = gl.where(row_ok_m, gl.minimum(causal_end_m, split_kv_end), split_kv_end)
    # Tiles ending at or below mask_from are fully visible to every row of
    # this block (the block's first position has the tightest causal bound).
    mask_from = gl.minimum(split_kv_end, cur_batch_seq_len - QLEN + 1 + m_block * Q_PER_BLOCK)
    num_unmasked = gl.minimum(gl.maximum(mask_from - split_kv_start, 0) // BLOCK_N, num_iter)

    for i in range(num_unmasked):
        buf = i % 2
        tile_lo = split_kv_start + i * BLOCK_N
        # KV[i] and indices[i+1] have landed for every warp; nobody still
        # reads the slots refilled below.
        gl.amd.cdna4.async_copy.wait_group(0)
        gl.barrier()
        _packed_q_issue_idx(bufs_idx.index(buf), Kv_indices, batch_page_start, tile_lo + 2 * BLOCK_N, split_kv_end,
                            offs_tok_idx)
        _packed_q_issue_kv(bufs_kv, bufs_kpe, 1 - buf, bufs_idx.index(1 - buf), Kv_c_cache, K_pe_cache,
                           tile_lo + BLOCK_N, split_kv_end, offs_tok_kv, offs_d_kv, offs_tok_pe, offs_d_pe,
                           stride_kv_c_bs, stride_k_pe_bs, KV_PE_OFFSET, WITHIN_2GB)
        gl.amd.cdna4.async_copy.commit_group()
        acc, e_max, e_sum = _packed_q_step(q_nope, q_pe, acc, e_max, e_sum, bufs_kv.index(buf), bufs_kpe.index(buf),
                                           tile_lo, row_limit, qk_scale, False, BLOCK_M, BLOCK_N,
                                           mfma_layout, mfma_layout_a, mfma_layout_b, linear_v)

    for i in range(num_unmasked, num_iter):
        buf = i % 2
        tile_lo = split_kv_start + i * BLOCK_N
        gl.amd.cdna4.async_copy.wait_group(0)
        gl.barrier()
        _packed_q_issue_idx(bufs_idx.index(buf), Kv_indices, batch_page_start, tile_lo + 2 * BLOCK_N, split_kv_end,
                            offs_tok_idx)
        _packed_q_issue_kv(bufs_kv, bufs_kpe, 1 - buf, bufs_idx.index(1 - buf), Kv_c_cache, K_pe_cache,
                           tile_lo + BLOCK_N, split_kv_end, offs_tok_kv, offs_d_kv, offs_tok_pe, offs_d_pe,
                           stride_kv_c_bs, stride_k_pe_bs, KV_PE_OFFSET, WITHIN_2GB)
        gl.amd.cdna4.async_copy.commit_group()
        acc, e_max, e_sum = _packed_q_step(q_nope, q_pe, acc, e_max, e_sum, bufs_kv.index(buf), bufs_kpe.index(buf),
                                           tile_lo, row_limit, qk_scale, True, BLOCK_M, BLOCK_N,
                                           mfma_layout, mfma_layout_a, mfma_layout_b, linear_v)
    gl.amd.cdna4.async_copy.wait_group(0)

    ######### epilogue #########
    acc *= kv_scale
    rcp = 1.0 / e_sum
    stored_value = (acc * rcp[:, None]).to(dtype)
    head_m = rows_m % NHEAD
    offs_d_o = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, mfma_layout))
    offs_o = (cur_batch * stride_o_b + q_pos_m * stride_o_s + head_m * stride_o_h + split_kv_id * stride_o_split)[:, None] + offs_d_o[None, :]
    gl.amd.cdna4.buffer_store(stored_value, ptr=O, offsets=offs_o, mask=row_ok_m[:, None])

    # Natural-log lse; e_sum == 0 (row with no visible token here) gives -inf,
    # which stage-2 drops.
    lse = (e_max + gl.log2(e_sum)) * LN2
    if RETURN_LSE and NUM_KV_SPLITS == 1:
        offs_final_lse = cur_batch * stride_final_lse_b + q_pos_m * stride_final_lse_s + head_m * stride_final_lse_h
        gl.amd.cdna4.buffer_store(lse, ptr=Final_lse, offsets=offs_final_lse, mask=row_ok_m)
    elif NUM_KV_SPLITS > 1:
        offs_mid_lse = cur_batch * stride_mid_lse_b + q_pos_m * stride_mid_lse_s + head_m * stride_mid_lse_h + split_kv_id * stride_mid_lse_split
        gl.amd.cdna4.buffer_store(lse, ptr=Mid_lse, offsets=offs_mid_lse, mask=row_ok_m)
# fmt: on


def packed_q_launch_params(batch_size, qlen, nhead, wg_budget=None):
    """(NUM_M_BLOCKS, Q_PER_BLOCK, NUM_KV_SPLITS); depends only on batch/qlen/nhead."""
    if wg_budget is None:
        wg_budget = int(
            os.environ.get("AITER_MLA_GLUON_PACKED_Q_WG", _DEFAULT_WG_BUDGET)
        )
    max_q_per_block = _BLOCK_M // nhead
    num_m_blocks = triton.cdiv(qlen, max_q_per_block)
    q_per_block = triton.cdiv(qlen, num_m_blocks)
    num_kv_splits = max(1, wg_budget // (batch_size * num_m_blocks))
    return num_m_blocks, q_per_block, num_kv_splits


def mla_gluon_packed_q_supported(q_nope, q_pe, kv_c, k_pe, has_pe=True, attn_sink=None):
    """Shapes the packed-query kernel handles; mla_gluon keeps everything else."""
    if q_nope.dim() != 4 or not has_pe or attn_sink is not None:
        return False
    _, qlen, nhead, head_dim_ckv = q_nope.shape
    k_pe = kv_c if k_pe is None else k_pe
    return (
        qlen >= 2
        and 1 <= nhead <= _BLOCK_M
        and head_dim_ckv == 512
        and q_pe.shape[-1] == 64
        and q_nope.dtype == torch.bfloat16
        and q_pe.dtype == torch.bfloat16
        and kv_c.dtype == torch.float8_e4m3fn
        and k_pe.dtype == torch.float8_e4m3fn
    )


def mla_gluon_packed_q(
    q_nope,  # [batch, qlen, nhead, kv_lora_rank] bf16 (strided views allowed)
    q_pe,  # [batch, qlen, nhead, qk_rope_head_dim] bf16
    kv_c,  # [N, kv_lora_rank + qk_rope_head_dim] (shared) or [N, kv_lora_rank] fp8
    o,  # [batch, qlen, nhead, kv_lora_rank]
    page_table,  # 2D: block_table [batch, max_seqlen] | 1D: kv_indices [total_kv]
    seq_info,  # 2D: cache_seqlens [batch]           | 1D: kv_indptr [batch+1]
    sm_scale,
    k_pe=None,
    kv_pe_offset=512,
    use_2d_view=True,
    kv_scale=1.0,
    return_lse=False,
    wg_budget=None,
):
    """Packed-query MLA verify decode; same contract as mla_gluon's MTP path.

    Query position q of each request attends KV [0, seq_len - qlen + q].
    Returns (o, lse) with lse [batch, qlen, nhead] fp32 when return_lse, else
    (o, None). Launch parameters depend only on (batch, qlen, nhead), so the
    call is CUDA-graph safe.
    """
    if k_pe is None:
        k_pe = kv_c
    assert arch_info.get_arch() == "gfx950", "mla_gluon_packed_q requires gfx950"
    assert mla_gluon_packed_q_supported(q_nope, q_pe, kv_c, k_pe)
    batch_size, qlen, nhead, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]

    num_m_blocks, q_per_block, num_kv_splits = packed_q_launch_params(
        batch_size, qlen, nhead, wg_budget
    )

    max_kv_bytes = kv_c.shape[0] * kv_c.stride(0) * kv_c.element_size()
    within_2gb = max_kv_bytes <= 0x80000000

    if num_kv_splits == 1:
        logits_buf = o.view(batch_size, qlen, nhead, 1, head_dim_ckv)
        mid_lse = None
        mid_lse_strides = (0, 0, 0, 0)
    else:
        logits_buf = torch.empty(
            (batch_size, qlen, nhead, num_kv_splits, head_dim_ckv),
            dtype=o.dtype,
            device=o.device,
        )
        mid_lse = torch.empty(
            (batch_size, qlen, nhead, num_kv_splits),
            dtype=torch.float32,
            device=o.device,
        )
        mid_lse_strides = mid_lse.stride()
    stride_o_b, stride_o_s, stride_o_h, stride_o_split, _ = logits_buf.stride()

    if return_lse:
        final_lse = torch.empty(
            (batch_size, qlen, nhead), dtype=torch.float32, device=o.device
        )
        final_lse_strides = final_lse.stride()
    else:
        final_lse = None
        final_lse_strides = (0, 0, 0)

    grid = (batch_size, num_kv_splits, num_m_blocks)
    _mla_gluon_packed_q[grid](
        q_nope,
        q_pe,
        kv_c,
        k_pe,
        page_table,
        seq_info,
        logits_buf,
        sm_scale,
        kv_scale,
        *q_nope.stride()[:3],
        *q_pe.stride()[:3],
        kv_c.stride(-2),
        k_pe.stride(-2),
        page_table.stride(0) if use_2d_view else 0,
        stride_o_b,
        stride_o_s,
        stride_o_h,
        stride_o_split,
        mid_lse,
        *mid_lse_strides,
        final_lse,
        *final_lse_strides,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        NUM_KV_SPLITS=num_kv_splits,
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
        KV_PE_OFFSET=kv_pe_offset,
        USE_2D_VIEW=use_2d_view,
        WITHIN_2GB=within_2gb,
        NHEAD=nhead,
        QLEN=qlen,
        Q_PER_BLOCK=q_per_block,
        RETURN_LSE=return_lse,
        num_warps=4,
    )

    if num_kv_splits == 1:
        return o, final_lse

    sl_b, sl_qs, sl_h, sl_split, _ = logits_buf.stride()
    so_b, so_qs, so_h = o.stride()[:3]
    sfl = final_lse_strides
    _mla_softmax_reducev_kernel[(batch_size, nhead, qlen)](
        logits_buf,
        mid_lse,
        o,
        final_lse,
        seq_info,
        sl_b,
        sl_qs,
        sl_h,
        sl_split,
        *mid_lse_strides,
        so_b,
        so_qs,
        so_h,
        *sfl,
        NUM_KV_SPLITS=num_kv_splits,
        HEAD_DIM_CKV=head_dim_ckv,
        HAS_FINAL_LSE=return_lse,
        USE_2D_VIEW=use_2d_view,
        BLOCK_S=min(64, triton.next_power_of_2(num_kv_splits)),
        BLOCK_N=_BLOCK_N,
        num_warps=8,
    )
    return o, final_lse
