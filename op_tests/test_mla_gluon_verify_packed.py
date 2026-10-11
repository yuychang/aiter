# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness test and microbenchmark for Gluon MLA speculative-verify decode.

Compares mla_gluon's per-position bh16bn128 path (default) with the packed-query
kernel (AITER_MLA_GLUON_PACKED_Q=1) for the Kimi-K3 TP8 verify shape: 12 heads,
kv_lora_rank 512 + rope 64, bf16 Q, fp8_e4m3 KV, page size 128, qlen 4 / 8.

The call is the one SGLang's aiter backend makes for target-verify
(sglang/srt/layers/attention/aiter_mla_gluon.py::mla_gluon_decode): fused
Q [bs * qlen, 12, 576] split into strided q_nope / q_pe views, the MLA pool
viewed as [N, 576], token-level kv_indices / kv_indptr (kv_len = cached
prefix + qlen), use_2d_view=False and a float kv_scale. The reference is fp32
attention on the dequantized fp8 KV.

    # correctness only (pytest)
    pytest op_tests/test_mla_gluon_verify_packed.py -q
    # before/after table over the full grid
    python op_tests/test_mla_gluon_verify_packed.py
    python op_tests/test_mla_gluon_verify_packed.py --qlen 8 --batch 3 --ctx 110000
"""

import argparse
import os
import statistics
from contextlib import contextmanager

import pytest
import torch

NHEAD = 12
KV_LORA_RANK = 512
ROPE_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + ROPE_DIM
PAGE_SIZE = 128
SM_SCALE = (128 + 64) ** -0.5  # Kimi-K3 qk_nope_head_dim + qk_rope_head_dim
FP8 = torch.float8_e4m3fn
PACKED_ENV = "AITER_MLA_GLUON_PACKED_Q"


def _is_gfx950():
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(not _is_gfx950(), reason="Gluon MLA is gfx950-only")


@contextmanager
def packed_q(enabled):
    old = os.environ.get(PACKED_ENV)
    os.environ[PACKED_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(PACKED_ENV, None)
        else:
            os.environ[PACKED_ENV] = old


def make_case(
    kv_lens, qlen, nhead=NHEAD, kv_scale=0.5, seed=0, device="cuda", pool_pad_tokens=0
):
    """Paged fp8 MLA pool + token-level kv_indices for requests of kv_lens tokens.

    kv_lens include the qlen verify tokens (SGLang: seq_lens + num_draft_tokens).
    Pages are shuffled so consecutive pages of a request are not adjacent.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    batch = len(kv_lens)
    pages_per_req = [(n + PAGE_SIZE - 1) // PAGE_SIZE for n in kv_lens]
    num_pages = sum(pages_per_req) + 1 + pool_pad_tokens // PAGE_SIZE
    # page 0 stays unused, like SGLang's padding slot.
    perm = (torch.randperm(num_pages - 1, generator=gen) + 1).to(device)
    within = torch.arange(PAGE_SIZE, device=device, dtype=torch.int32)
    indices, start = [], 0
    for n, p in zip(kv_lens, pages_per_req):
        pages = perm[start : start + p].to(torch.int32)
        start += p
        indices.append((pages[:, None] * PAGE_SIZE + within[None, :]).flatten()[:n])
    kv_indices = torch.cat(indices)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(
        torch.tensor(kv_lens, dtype=torch.int32, device=device), 0
    )

    torch.manual_seed(seed)
    n_slots = num_pages * PAGE_SIZE
    k_buffer = torch.empty((n_slots, 1, QK_HEAD_DIM), dtype=FP8, device=device)
    chunk = 1 << 20
    for s in range(0, n_slots, chunk):
        e = min(s + chunk, n_slots)
        k_buffer[s:e] = (
            torch.randn((e - s, 1, QK_HEAD_DIM), device=device) / kv_scale
        ).to(FP8)
    q = torch.randn(
        (batch * qlen, nhead, QK_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    return dict(
        q=q,
        k_buffer=k_buffer,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        kv_lens=list(kv_lens),
        qlen=qlen,
        kv_scale=kv_scale,
    )


def sglang_verify_call(case, return_lse=False):
    """Mirror of sglang aiter_mla_gluon.mla_gluon_decode for qlen > 1."""
    from aiter.ops.triton.gluon.mla_gluon import mla_gluon

    q, qlen = case["q"], case["qlen"]
    num_head = q.shape[1]
    batch_size = q.shape[0] // qlen
    q_nope, q_pe = torch.split(q, [KV_LORA_RANK, ROPE_DIM], dim=-1)
    q_nope = q_nope.view(batch_size, qlen, num_head, KV_LORA_RANK)
    q_pe = q_pe.view(batch_size, qlen, num_head, ROPE_DIM)
    o = q.new_empty((batch_size, qlen, num_head, KV_LORA_RANK))
    _, lse = mla_gluon(
        q_nope,
        q_pe,
        case["k_buffer"].view(-1, QK_HEAD_DIM),
        o,
        case["kv_indices"],
        case["kv_indptr"],
        SM_SCALE,
        k_pe=None,
        kv_pe_offset=KV_LORA_RANK,
        use_2d_view=False,
        kv_scale=case["kv_scale"],
        min_kv_seq_len=max(case["kv_lens"]),
        return_lse=return_lse,
    )
    out = o.flatten(0, 1)
    return (out, lse) if return_lse else out


def reference(case):
    """fp32 causal-verify attention -> (out [bs*qlen, H, 512], lse [bs, qlen, H])."""
    q, qlen, kv_scale = case["q"], case["qlen"], case["kv_scale"]
    nhead = q.shape[1]
    indptr = case["kv_indptr"].tolist()
    outs, lses = [], []
    for b, n in enumerate(case["kv_lens"]):
        idx = case["kv_indices"][indptr[b] : indptr[b + 1]].long()
        kv = case["k_buffer"][idx, 0].float() * kv_scale  # [n, 576]
        qb = q[b * qlen : (b + 1) * qlen].float()  # [qlen, H, 576]
        s = torch.einsum("qhd,nd->qhn", qb, kv) * SM_SCALE
        pos = torch.arange(n, device=q.device)
        limit = n - qlen + torch.arange(qlen, device=q.device)  # inclusive
        s.masked_fill_(pos[None, None, :] > limit[:, None, None], float("-inf"))
        lses.append(torch.logsumexp(s, dim=-1))
        outs.append(torch.einsum("qhn,nd->qhd", torch.softmax(s, -1), kv[:, :512]))
        del kv, s
    out = torch.cat(outs).view(-1, nhead, KV_LORA_RANK)
    return out, torch.stack(lses)


def errors(out, ref):
    d = (out.float() - ref).abs()
    max_abs = d.max().item()
    return max_abs, max_abs / ref.abs().max().item()


def check(case, packed, atol=2e-2, rel=1e-2, return_lse=False):
    ref, ref_lse = reference(case)
    with packed_q(packed):
        res = sglang_verify_call(case, return_lse=return_lse)
    out, lse = res if return_lse else (res, None)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all(), "non-finite output"
    max_abs, max_rel = errors(out, ref)
    assert max_abs <= atol and max_rel <= rel, f"{max_abs=} {max_rel=}"
    if return_lse:
        lse_err = (lse - ref_lse).abs().max().item()
        assert lse_err < 1e-2, f"{lse_err=}"
    return max_abs, max_rel


# ---------------------------------------------------------------- pytest


@pytest.mark.parametrize("qlen", [4, 8])
@pytest.mark.parametrize("batch, ctx", [(1, 8192), (3, 33000), (4, 4321), (16, 2048)])
@pytest.mark.parametrize("packed", [False, True])
def test_verify_uniform(batch, ctx, qlen, packed):
    check(make_case([ctx] * batch, qlen), packed)


@pytest.mark.parametrize("qlen", [2, 3, 4, 5, 6, 7, 8])
def test_verify_ragged_short(qlen):
    # Lengths around the BLOCK_N=64 split floor and the qlen causal window:
    # tiny last splits leave some query rows with no visible token there.
    lens = [qlen, qlen + 1, 63, 64, 65, 127, 129, 200, 777, 1000, 4097, 12345]
    check(make_case(lens, qlen, seed=qlen), True)


@pytest.mark.parametrize("batch", [64, 128, 256])
def test_verify_large_batch(batch):
    # 256 // (batch * blocks) -> 2, 1, 1 splits: the 1-split fast path writes o.
    lens = [
        int(x)
        for x in torch.randint(
            100, 3000, (batch,), generator=torch.Generator().manual_seed(batch)
        )
    ]
    check(make_case(lens, 8, seed=batch), True)


@pytest.mark.parametrize("nhead", [8, 16, 1])
def test_verify_other_heads(nhead):
    check(make_case([5000, 70, 3000], 8, nhead=nhead, seed=nhead), True)


@pytest.mark.parametrize("batch", [1, 128])
def test_verify_return_lse(batch):
    lens = [3000 + 17 * i for i in range(batch)]
    check(make_case(lens, 8, seed=3), True, return_lse=True)


def test_verify_kv_scale_one():
    check(make_case([20000, 9000], 4, kv_scale=1.0, seed=5), True)


def test_verify_cuda_graph_replay():
    """Capture at one set of lengths, replay with others (same batch/qlen)."""
    qlen, batch = 8, 3
    cap = make_case([60000] * batch, qlen, seed=11)
    static = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in cap.items()}
    with packed_q(True):
        sglang_verify_call(static)  # compile outside capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = sglang_verify_call(static)
        for lens in ([60000, 100, 33333], [9, 59999, 1024]):
            case = make_case(lens, qlen, seed=sum(lens))
            n = case["kv_indices"].numel()
            assert case["k_buffer"].shape[0] <= static["k_buffer"].shape[0]
            static["k_buffer"][: case["k_buffer"].shape[0]].copy_(case["k_buffer"])
            static["kv_indices"][:n].copy_(case["kv_indices"])
            static["kv_indptr"].copy_(case["kv_indptr"])
            static["q"].copy_(case["q"])
            g.replay()
            torch.cuda.synchronize()
            ref, _ = reference(case)
            max_abs, max_rel = errors(out, ref)
            assert max_abs < 2e-2 and max_rel < 1e-2, f"{lens=} {max_abs=} {max_rel=}"


def test_verify_over_2gb():
    """>2 GB pool -> global_load_to_shared path; must match the buffer path."""
    free, _ = torch.cuda.mem_get_info()
    if free < 8 * 2**30:
        pytest.skip("needs ~8 GiB free")
    lens, qlen = [40000, 777], 8
    small = make_case(lens, qlen, seed=21)
    gb2_tokens = (0x80000000 // QK_HEAD_DIM) + 4 * PAGE_SIZE
    big = dict(small)
    big["k_buffer"] = torch.zeros(
        (gb2_tokens + small["k_buffer"].shape[0], 1, QK_HEAD_DIM),
        dtype=FP8,
        device="cuda",
    )
    big["k_buffer"][gb2_tokens:] = small["k_buffer"]
    big["kv_indices"] = small["kv_indices"] + gb2_tokens
    with packed_q(True):
        o_small = sglang_verify_call(small)
        o_big = sglang_verify_call(big)
    torch.cuda.synchronize()
    torch.testing.assert_close(o_big, o_small, atol=0, rtol=0)


# ---------------------------------------------------------------- benchmark


class _Flusher:
    """Evicts L2 and MALL (Infinity Cache) between timed calls."""

    def __init__(self, nbytes=1 << 30):
        self.buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")

    def __call__(self):
        self.buf.zero_()


def bench_cold(fn, flusher, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        flusher()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1e3)
    return statistics.median(times)


def bench_graph(fn, iters=50, reps=5):
    """Back-to-back replays of a captured call (KV may stay in MALL)."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1e3 / (iters * reps)


def run_grid(qlens, batches, ctxs, kv_scale, check_ref, iters):
    from aiter.ops.triton.gluon.mla_gluon_packed_q import packed_q_launch_params

    flusher = _Flusher()
    rows = []
    hdr = (
        f"{'qlen':>4} {'bs':>3} {'ctx':>7} {'splits':>9} | {'base_us':>9} {'packed_us':>9} "
        f"{'speedup':>7} | {'base_hot':>8} {'pack_hot':>8} {'spdup':>6} | "
        f"{'base_abs':>8} {'base_rel':>8} {'pack_abs':>8} {'pack_rel':>8}"
    )
    print(hdr, flush=True)
    for qlen in qlens:
        for bs in batches:
            for ctx in ctxs:
                case = make_case([ctx] * bs, qlen, kv_scale=kv_scale, seed=bs * 7 + ctx)
                res = {}
                for packed in (False, True):
                    with packed_q(packed):
                        fn = lambda: sglang_verify_call(case)  # noqa: E731
                        out = fn()
                        torch.cuda.synchronize()
                        res[packed] = dict(
                            out=out,
                            cold=bench_cold(fn, flusher, iters=iters),
                            hot=bench_graph(fn),
                        )
                if check_ref:
                    ref, _ = reference(case)
                    for packed in (False, True):
                        res[packed]["err"] = errors(res[packed]["out"], ref)
                    del ref
                else:
                    for packed in (False, True):
                        res[packed]["err"] = (float("nan"), float("nan"))
                base_splits = max(1, 256 // (bs * qlen))
                _, _, pk_splits = packed_q_launch_params(bs, qlen, NHEAD)
                b, p = res[False], res[True]
                row = dict(
                    qlen=qlen,
                    bs=bs,
                    ctx=ctx,
                    base_splits=base_splits,
                    packed_splits=pk_splits,
                    base_us=b["cold"],
                    packed_us=p["cold"],
                    base_hot=b["hot"],
                    packed_hot=p["hot"],
                    base_err=b["err"],
                    packed_err=p["err"],
                )
                rows.append(row)
                print(
                    f"{qlen:>4} {bs:>3} {ctx:>7} {base_splits:>4}/{pk_splits:<4} | {b['cold']:>9.1f} "
                    f"{p['cold']:>9.1f} {b['cold'] / p['cold']:>6.2f}x | {b['hot']:>8.1f} {p['hot']:>8.1f} "
                    f"{b['hot'] / p['hot']:>5.2f}x | {b['err'][0]:>8.2e} {b['err'][1]:>8.2e} "
                    f"{p['err'][0]:>8.2e} {p['err'][1]:>8.2e}",
                    flush=True,
                )
                del case, res
                torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--qlen", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 3, 4, 8, 16])
    ap.add_argument("--ctx", type=int, nargs="+", default=[8192, 32768, 110000, 200000])
    ap.add_argument("--kv-scale", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-ref", action="store_true", help="skip the fp32 reference")
    args = ap.parse_args()
    run_grid(
        args.qlen, args.batch, args.ctx, args.kv_scale, not args.no_ref, args.iters
    )


if __name__ == "__main__":
    main()
