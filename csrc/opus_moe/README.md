<!--
SPDX-License-Identifier: MIT
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
-->

# Opus MoE

This directory contains experimental Opus MoE stage2 kernels and their Python
bindings. The current code is gfx950-only and intentionally narrow: it is a
place to develop and measure Opus stage2 kernels before wiring individual cases
into the production fused MoE tuned configuration.

There are two active paths:

- BF16 route-reduce stage2 prototype.
- DSV4 A8W4 decode stage2 kernels for `topk=6`, `hidden=7168`,
  `logical_inter_dim=512`, `inter_dim_pad=128`, and `experts=384`.

## Kernel Surfaces

### BF16 Stage2

The BF16 path takes:

- `inter_states [token, topk, inter_dim]`, BF16.
- `w2 [expert, hidden, inter_dim]`, BF16.
- CK/Opus MoE sorting metadata: `sorted_token_ids`, `sorted_expert_ids`,
  `num_valid_ids`, and optional `sorted_weights`.
- `route_out [token * topk, hidden]`, BF16 scratch.
- `out [token, hidden]`, BF16 final output.

It writes token-slot route output first, then runs a separate token/topk reduce.

Supported BF16 kernel id:

| kid | name | contract |
|---:|---|---|
| `-1` | auto | Select current gfx950 BF16 stage2 kid. |
| `1` | `bf16_gemmstyle256x256x64_token_slot_route_out_no_oob_nfast` | `256 routes x 256 hidden x 64 K`, route-output then reduce, no padded/OOB route rows. |

The BF16 fused MoE hook is opt-in:

```bash
AITER_USE_OPUS_MOE_STAGE2=1
```

That hook only accepts BF16, `QuantType.No`, no EP, no bias, no stage2 scales,
`block_m` multiple of 256, and `num_valid_ids[0] == token * topk`.

### DSV4 A8W4 Decode Stage2

The A8W4 path takes:

- `inter_states [token, 6, 512]`, FP8.
- `w2 [384, 7168, 256]`, FP4x2 packed.
- `a2_scale [route, scale_cols]`, FP8 E8M0.
- `w2_scale [expert * hidden, scale_cols]`, FP8 E8M0.
- CK/Opus MoE sorting metadata and optional `sorted_weights`.
- `out [token, 7168]`, BF16 direct output, or per-slot route output for P4.

Direct-output kids atomically accumulate into `[token, hidden]`. The P4 route-out
kid writes `[token * topk, hidden]` and then uses the shared route-output reduce.

Supported A8W4 kernel ids:

| kid | name | block shape | output contract |
|---:|---|---|---|
| `-1` | auto | selected by host shape/block rules | direct atomic, except `return_per_slot=True` selects P4 route-out |
| `2010` | `dsv4_a8w4_decode_bm32_dynamic` | `BM32 x BN256` | direct atomic |
| `2011` | `dsv4_a8w4_decode_bm32_dynamic_paced` | `BM32 x BN256` | direct atomic with pow2 route-block pacing |
| `2020` | `dsv4_a8w4_decode_bm64_dynamic` | `BM64 x BN256` | direct atomic |
| `2030` | `dsv4_a8w4_decode_bm16_bn128_dynamic` | `BM16 x BN128`, `sort_block_m=32` | direct atomic |
| `2040` | `dsv4_a8w4_p4_route_out64x256x256_sbm128` | `BM64 x BN256`, `sort_block_m=128` | route output, then reduce |

In fused MoE tuned configs, A8W4 stage2 uses one public stage2 name:
`opus_moe_stage2_a8w4_decode`. Tuned CSV rows choose the concrete C++ kid with
explicit columns instead of encoding it in `kernelName2`:

- `stage2_kernel_id`: `-1` for auto, or one of the A8W4 kids above.
- `stage2_block_m`: the kernel tile M passed to Opus stage2. This can differ
  from CSV `block_m`, which controls MoE sorting. For example P4 sorts with
  `block_m=128` but runs the `BM64` route-out kernel.
- `stage2_route_out`: `1` when stage2 returns `[token * topk, hidden]` and
  needs route-output reduce, otherwise `0`.

BM32 uses the dynamic kid for all route buckets. Optional tuned CSV metadata
columns `route_bucket`, `expected_sorted_blocks`, `min_sorted_blocks`, and
`max_sorted_blocks` are carried to runtime and checked after sorting.

## File Layout

Host and shared code:

- `include/opus_moe.h`: C++ entry points exposed to pybind/JIT.
- `include/opus_moe_common.cuh`: shared kernel ids, constants, kargs, and
  metadata helpers.
- `include/opus_moe_arch.cuh`: runtime architecture probe wrapper.
- `include/opus_moe_host_impl.cuh`: host validation and launch selection.
- `opus_moe.cu`: pybind-facing translation unit.

gfx950 code:

- `include/gfx950/opus_moe_arch_gfx950.cuh`: gfx950 launch wrappers and BF16
  generated manifest dispatch.
- `include/gfx950/opus_moe_stage2_route_output_reduce_gfx950.cuh`: shared
  token/topk route-output reduction.
- `include/gfx950/opus_moe_stage2_utils_gfx950.cuh`: small gfx950 device
  helpers, including BF16 packing/conversion helpers.
- `include/gfx950/a16w16/`: BF16/A16W16 stage2 traits and pipeline.
- `include/gfx950/a8w4/opus_moe_traits_stage2_a8w4_decode_gfx950.cuh`:
  A8W4 decode shape traits.
- `include/gfx950/a8w4/opus_moe_pipeline_stage2_a8w4_decode_policy_gfx950.cuh`:
  A8W4 decode schedule policy and layout helpers.
- `include/gfx950/a8w4/opus_moe_pipeline_stage2_a8w4_decode_gfx950.cuh`:
  A8W4 decode prologue, mainloop, epilogue, and kernel entry.
- `include/gfx950/a8w4/opus_moe_stage2_a8w4_decode_dispatch_gfx950.cuh`:
  A8W4 kid-to-trait dispatch.

Python/JIT code:

- `aiter/ops/opus/moe_stage2.py`: BF16 Python wrapper.
- `aiter/ops/opus/moe_stage2_a8w4.py`: A8W4 Python wrapper and route-output
  reduce wrapper.
- `aiter/ops/opus/moe_stage2_a8w4_fused.py`: fused MoE CSV parsing and
  stage2 wrapper glue for A8W4.
- `gen_instances.py`: JIT-time BF16 manifest generator.
- `opus_moe_common.py`: BF16 tuner metadata and tuned CSV schema.
- `opus_moe_tune.py`: BF16 stage2-only tuner.

## Tuning and Dispatch

`gen_instances.py` emits `opus_moe_stage2_manifest.h` into the JIT build blob.
That generated header is consumed by `include/gfx950/opus_moe_arch_gfx950.cuh`
for the BF16 stage2 dispatch table.

The lightweight tuner currently covers the BF16 route-reduce prototype only:

```bash
python3 csrc/opus_moe/opus_moe_tune.py \
  --shape 2048,4096,1024,64,8 \
  --block-ms 256 \
  --kids 1 \
  -o /shared/amdgpu/home/hyi_qle/yifehuan_temp/data/opus_moe_stage2_tuned.csv \
  -o2 /shared/amdgpu/home/hyi_qle/yifehuan_temp/data/opus_moe_stage2_profile.csv \
  --warmup 5 --iters 20
```

The BF16 tuned CSV columns are:

```text
arch,cu_num,token,model_dim,inter_dim,expert,topk,dtype,
a2_layout,output_mode,block_m,kid,kernel_name,block_n,block_k,
us,max_abs,mean_abs,valid
```

A8W4 production selection should be done through the fused MoE tuned
configuration by adding `opus_...` stage2 kernel names only for measured cases
where Opus is correct and faster than the baseline. The A8W4 kernels are not
selected through `opus_moe_tune.py`.

## Validation

`op_tests/test_opus_moe_stage2.py` covers BF16 route-reduce correctness and the
opt-in fused MoE hook plumbing. A8W4 production selection should be validated
with model-level traces before adding or changing tuned CSV entries; local replay
manifests and captured routing dumps should stay outside the repository.

## Current Limits

- gfx950 only.
- BF16 path is a no-OOB route-output prototype.
- A8W4 path is DSV4 decode-specific: `topk=6`, `hidden=7168`,
  `logical_inter_dim=512`, `inter_dim_pad=128`, `experts=384`.
- A8W4 direct-output kernels do not support EP `expert_mask/topk_ids` or
  `bias2`.
- P4 currently uses route-output plus reduce to match the reduce/route-out
  contract. A direct-atomic P4 variant is a separate optimization target.
- Final fused MoE enablement should be case-gated through tuned CSV entries, not
  globally enabled for every compatible-looking shape.
