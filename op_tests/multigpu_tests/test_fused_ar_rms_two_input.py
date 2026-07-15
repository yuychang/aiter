# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os
import statistics
from multiprocessing import Pool, set_start_method

import torch
import torch.distributed as dist
import torch.nn.functional as F

from aiter.dist.communication_op import (
    tensor_model_parallel_fused_allreduce_rmsnorm_two_input,
)
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

set_start_method("spawn", force=True)

TOKEN_COUNTS = (1, 2, 4, 8, 16, 32)
HIDDEN_SIZE = 7168
EPS = 1e-6


def _reference(routed, shared, residual, weight, group):
    local_sum = (routed.float() + shared.float()).to(torch.bfloat16)
    dist.all_reduce(local_sum, group=group)
    residual_out = (local_sum.float() + residual.float()).to(torch.bfloat16)
    norm_out = F.rms_norm(
        residual_out,
        (residual_out.shape[-1],),
        weight=weight,
        eps=EPS,
    )
    return norm_out, residual_out


def _run_rank(
    rank: int,
    world_size: int,
    init_method: str,
    replay_count: int,
    benchmark: bool,
    benchmark_iters: int,
):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=init_method,
    )
    ensure_model_parallel_initialized(world_size, 1)
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    rows = []
    failures = []
    for m in TOKEN_COUNTS:
        torch.manual_seed(20260713 + rank * 101 + m)
        routed = torch.randn(
            (m, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
        )
        shared = torch.randn_like(routed)
        residual = torch.randn_like(routed)
        weight = torch.randn(
            (HIDDEN_SIZE,), device=device, dtype=torch.bfloat16
        )
        norm_ref, residual_ref = _reference(
            routed.clone(), shared.clone(), residual.clone(), weight, group
        )

        dist.barrier(group=group)
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as capture:
            with torch.cuda.graph(graph, stream=capture.stream):
                norm_out, residual_out = (
                    tensor_model_parallel_fused_allreduce_rmsnorm_two_input(
                        routed,
                        shared,
                        residual,
                        weight,
                        EPS,
                    )
                )

        max_norm_diff = 0.0
        max_residual_diff = 0.0
        for _ in range(replay_count):
            dist.barrier(group=group)
            norm_out.zero_()
            residual_out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            max_norm_diff = max(
                max_norm_diff,
                (norm_out.float() - norm_ref.float()).abs().max().item(),
            )
            max_residual_diff = max(
                max_residual_diff,
                (residual_out.float() - residual_ref.float()).abs().max().item(),
            )

        dist.barrier(group=group)
        if max_residual_diff > 0.125:
            failures.append(
                f"rank={rank} M={m}: residual mismatch {max_residual_diff}"
            )
        if max_norm_diff > 0.125:
            failures.append(f"rank={rank} M={m}: RMSNorm mismatch {max_norm_diff}")
        row = {
            "m": m,
            "max_norm_diff": max_norm_diff,
            "max_residual_diff": max_residual_diff,
        }
        if benchmark:
            for _ in range(20):
                graph.replay()
            torch.cuda.synchronize()
            samples = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(benchmark_iters):
                dist.barrier(group=group)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000.0)
            row.update(
                {
                    "median_us": statistics.median(samples),
                    "min_us": min(samples),
                    "max_us": max(samples),
                }
            )
        rows.append(row)

    failed = torch.tensor(int(bool(failures)), device=device, dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
    dist.barrier(group=group)
    destroy_model_parallel()
    destroy_distributed_environment()
    if failed.item():
        print(f"rank={rank} rows={rows} failures={failures}", flush=True)
        raise AssertionError("; ".join(failures) or "another rank failed")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-iters", type=int, default=100)
    args = parser.parse_args()

    world_size = 4
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "49413")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=world_size) as pool:
        futures = [
            pool.apply_async(
                _run_rank,
                args=(
                    rank,
                    world_size,
                    init_method,
                    args.replays,
                    args.benchmark,
                    args.benchmark_iters,
                ),
            )
            for rank in range(world_size)
        ]
        results = [future.get() for future in futures]
    for rank, rows in enumerate(results):
        print(f"rank={rank}: {rows}")


if __name__ == "__main__":
    main()
