"""Eligibility of row-strided activations for MXFP4 inline-sort gemm1."""

import torch

from aiter.fused_moe import _mxfp4_inline_sort_activation_supported


def test_contiguous_bf16_is_supported():
    hidden = torch.empty((8, 3584), dtype=torch.bfloat16)
    assert _mxfp4_inline_sort_activation_supported(hidden)


def test_unit_inner_stride_slice_is_supported():
    # K3 fused front: [M, 6016] split into a routed [M, 3584] view.
    parent = torch.empty((8, 6016), dtype=torch.bfloat16)
    routed = parent[:, 2432 : 2432 + 3584]
    assert routed.shape == (8, 3584)
    assert routed.stride() == (6016, 1)
    assert not routed.is_contiguous()
    assert routed.data_ptr() % 16 == 0
    assert _mxfp4_inline_sort_activation_supported(routed)


def test_non_unit_inner_stride_is_rejected():
    parent = torch.empty((8, 7168), dtype=torch.bfloat16)
    strided = parent[:, ::2]
    assert strided.shape == (8, 3584)
    assert strided.stride(-1) != 1
    assert not _mxfp4_inline_sort_activation_supported(strided)


def test_misaligned_row_pitch_is_rejected():
    parent = torch.empty((4, 3600), dtype=torch.bfloat16)
    # 3585 * 2 bytes is not a multiple of 16, so the //16 buffer address wraps.
    view = parent.as_strided((4, 3584), (3585, 1))
    assert view.stride(-1) == 1
    assert not _mxfp4_inline_sort_activation_supported(view)


def test_misaligned_base_pointer_is_rejected():
    parent = torch.empty((8, 6016), dtype=torch.bfloat16)
    routed = parent[:, 2433 : 2433 + 3584]
    assert routed.stride() == (6016, 1)
    assert routed.data_ptr() % 16 != 0
    assert not _mxfp4_inline_sort_activation_supported(routed)


def test_overlapping_row_pitch_is_rejected():
    parent = torch.empty((8, 3584), dtype=torch.bfloat16)
    view = parent.as_strided((8, 3584), (16, 1))
    assert view.stride(-1) == 1
    assert not _mxfp4_inline_sort_activation_supported(view)
