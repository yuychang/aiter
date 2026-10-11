# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
import torch

import aiter as ops
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_registered_input_view_and_cpp_staging_predicate():
    device = torch.device("cuda", torch.cuda.current_device())
    nbytes = 4096
    ptr = ops.allocate_data_buffer(nbytes)
    ca = CustomAllreduce.__new__(CustomAllreduce)
    ca.disabled = False
    ca.device = device
    ca._pool = {
        "input": SimpleNamespace(data_ptr=ptr, max_size=nbytes),
    }
    try:
        view = ca.get_registered_input_buffer((8, 16), torch.bfloat16)
        assert view.shape == (8, 16)
        assert view.dtype == torch.bfloat16
        assert view.is_contiguous()
        assert view.data_ptr() == ptr

        view.fill_(3)
        assert not ops.input_needs_staging(view, ptr)

        ordinary = torch.empty_like(view)
        assert ordinary.data_ptr() != ptr
        assert ops.input_needs_staging(ordinary, ptr)

        offset = view[:, 1:]
        assert offset.data_ptr() != ptr
        assert ops.input_needs_staging(offset, ptr)

        with pytest.raises(ValueError, match="pool holds"):
            ca.get_registered_input_buffer((nbytes,), torch.float32)
    finally:
        ca._registered_input_storage = None
        ops.free_meta_buffer(ptr)
