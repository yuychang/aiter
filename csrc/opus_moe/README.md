<!--
SPDX-License-Identifier: MIT
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
-->

# Opus MoE

This directory hosts experimental Opus-style MoE kernels.

The first target is a narrow BF16 stage2 path for AITER fused MoE:

- input activation: `inter_states [T, topk, I]`, BF16
- weight: `w2 [E, H, I]`, BF16
- metadata: Opus/CK MoE sorting outputs
- output: `route_out [T * topk, H]` plus BF16 reduce into `out [T, H]`

The module currently exposes these kernel ids:

- `kernel_id=-1` (default): auto dispatch to the current gfx950 BF16 stage2
  kernel.
- `kernel_id=1` (`bf16_gemmstyle256x256x64_token_slot_route_out_no_oob_nfast`):
  the OpusGEMM-style `256 routes x 256 hidden x 64 K` route-output fast path
  with N-fast grid ordering.

The supported feature matrix is intentionally narrow:

- dtype: BF16 A2 and BF16 W2 only.
- A2 layout: token-major `[token, topk, inter_dim]` only.
- output mode: token-slot route output plus separate BF16 reduce only.
- topk weighting: applied in the stage2 epilogue.
- architecture: gfx950 only.
- tile selection: fixed by `kernel_id=1`; `block_m` follows the MoE sorting
  block size and must be a multiple of 256.
- metadata: the fast path assumes no OOB/padded route rows. In practice,
  `num_valid_ids[0]` must equal `token_num * topk`.

This gives us a dedicated build/test surface for the current stage2 pipeline.

The C++ layout mirrors the `opus_gemm` dispatch stack:

- `include/opus_moe_arch.cuh`: shared runtime arch probe wrapper.
- `include/gfx950/a16w16/opus_moe_traits_stage2_gfx950.cuh`: A16W16 static tile traits.
- `include/gfx950/a16w16/opus_moe_pipeline_stage2_gemmstyle_gfx950.cuh`: A16W16 device pipeline.
- `include/gfx950/opus_moe_arch_gfx950.cuh`: per-arch id dispatch and launch wrappers.
- `opus_moe.cu`: host validation, arch router, and pybind-facing launcher.

The tuning scaffold mirrors the lightweight `opus_gemm` pattern:

- `opus_moe_common.py`: Python-side stage2 kid metadata and candidate filters.
- `opus_moe_tune.py`: stage2-only tuner that sweeps candidate kids, writes all
  candidate timings to an optional profile CSV, and writes one winner per
  shape/config key to a tuned CSV.
- `gen_instances.py`: JIT-time stage2 dispatch header generator. It emits the
  `opus_moe_stage2_manifest.h` kid-to-launcher table into the module build
  blob directory.

Example:

```bash
python3 csrc/opus_moe/opus_moe_tune.py \
  --shape 2048,4096,1024,64,8 \
  --block-ms 256 \
  --kids 1 \
  -o /shared/amdgpu/home/hyi_qle/yifehuan_temp/data/opus_moe_stage2_tuned.csv \
  -o2 /shared/amdgpu/home/hyi_qle/yifehuan_temp/data/opus_moe_stage2_profile.csv \
  --warmup 5 --iters 20
```

The tuned CSV columns are:

```text
arch,cu_num,token,model_dim,inter_dim,expert,topk,dtype,
a2_layout,output_mode,block_m,kid,kernel_name,block_n,block_k,
us,max_abs,mean_abs,valid
```

During `module_moe_opus` build, `gen_instances.py` emits the C++ kid manifest
consumed by `include/gfx950/opus_moe_arch_gfx950.cuh`.

There is also an opt-in fused MoE hook:

```bash
AITER_USE_OPUS_MOE_STAGE2=1
```

The hook is intentionally narrow: BF16, `QuantType.No`, two-stage path, no
EP/bias, no stage2 scales, `block_m` multiple of 256, and no padded route rows.
It writes route outputs to a BF16 temporary, then runs a separate BF16 reduce
into the fused MoE output, so it is still for end-to-end correctness and
benchmark plumbing only.
