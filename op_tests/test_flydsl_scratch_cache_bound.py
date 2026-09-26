# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.fused_moe import (
    _FLYDSL_SCRATCH_POOL,
    _get_flydsl_stage1_out,
    _get_flydsl_stage2_reduce_target,
    flydsl_scratch_cache_bytes,
)


@pytest.fixture(autouse=True)
def clear_scratch_pool():
    _FLYDSL_SCRATCH_POOL.clear()
    yield
    _FLYDSL_SCRATCH_POOL.clear()


def _stage2(shape, device):
    return _get_flydsl_stage2_reduce_target(shape, torch.bfloat16, device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_unpinned_flood_stays_under_cap(monkeypatch):
    """Distinct eager shapes used to retain every buffer. A flood that would
    have been ~1 GiB must stay inside the shared cap, including allocator
    accounting, not only the pool's own counter."""
    device = torch.device("cuda:0")
    cap = 16 * 1024 * 1024
    monkeypatch.setenv("AITER_FLYDSL_STAGE2_SCRATCH_REUSE", "1")
    monkeypatch.setenv("AITER_FLYDSL_SCRATCH_CACHE_MAX_BYTES", str(cap))
    shape_bytes = 1024 * 1024 * 2  # bf16
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated(device)
    for rows in range(1, 501):
        _stage2((rows, 1024), device)
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated(device) - before
    assert flydsl_scratch_cache_bytes() <= cap
    assert flydsl_scratch_cache_bytes() >= cap - shape_bytes
    assert retained <= cap + shape_bytes
    # 500 * 2 MiB unbounded would be 1000 MiB.
    assert retained < 64 * 1024 * 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_stage1_and_stage2_share_the_cap(monkeypatch):
    device = torch.device("cuda:0")
    cap = 8 * 1024 * 1024
    monkeypatch.setenv("AITER_FLYDSL_STAGE1_SCRATCH_REUSE", "1")
    monkeypatch.setenv("AITER_FLYDSL_STAGE2_SCRATCH_REUSE", "1")
    monkeypatch.setenv("AITER_FLYDSL_SCRATCH_CACHE_MAX_BYTES", str(cap))
    for rows in range(1, 40):
        _get_flydsl_stage1_out((rows, 4096), device)
        _stage2((rows, 2048), device)
    assert flydsl_scratch_cache_bytes() <= cap
    assert _FLYDSL_SCRATCH_POOL.count("stage1") >= 1
    assert _FLYDSL_SCRATCH_POOL.count("stage2") >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_oversized_allocation_is_kept(monkeypatch):
    device = torch.device("cuda:0")
    cap = 4096
    monkeypatch.setenv("AITER_FLYDSL_STAGE2_SCRATCH_REUSE", "1")
    monkeypatch.setenv("AITER_FLYDSL_SCRATCH_CACHE_MAX_BYTES", str(cap))
    tensor = _stage2((1024, 1024), device)
    nbytes = tensor.numel() * tensor.element_size()
    assert nbytes > cap
    assert flydsl_scratch_cache_bytes() == nbytes
    again = _stage2((1024, 1024), device)
    assert again.data_ptr() == tensor.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_captured_pointer_survives_eviction_pressure(monkeypatch):
    device = torch.device("cuda:0")
    cap = 4 * 1024 * 1024
    monkeypatch.setenv("AITER_FLYDSL_STAGE2_SCRATCH_REUSE", "1")
    monkeypatch.setenv("AITER_FLYDSL_SCRATCH_CACHE_MAX_BYTES", str(cap))
    shape = (1024, 256)  # 512 KiB bf16
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            warmup = _stage2(shape, device)
            warmup.zero_()
        with torch.cuda.graph(graph):
            captured = _stage2(shape, device)
            captured.fill_(1)
    torch.cuda.current_stream(device).wait_stream(stream)
    captured_ptr = captured.data_ptr()
    assert _FLYDSL_SCRATCH_POOL.pinned_nbytes() >= captured.numel() * captured.element_size()

    for rows in range(1, 80):
        _stage2((rows, 1024), device)

    assert captured.data_ptr() == captured_ptr
    assert flydsl_scratch_cache_bytes() <= cap + captured.numel() * captured.element_size()
    captured.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert captured.flatten()[0].item() == 1
    assert captured.data_ptr() == captured_ptr
