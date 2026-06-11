# MOE GEMM Kernel Replacement Notes

Branch: `fused-moe-route-quant-scatter-gemmopt`
Date: 2026-06-11
Author: jli10004

## Goal

Replace the MOE GEMM kernel (`gemm_mxscale_gfx1250.py`) with the newer FlyDSL repo version (`FlyDSL/kernels/gemm_fp8fp4_gfx1250.py`) to pick up performance improvements (new tuning knobs, optimized TDM/LDS/WMMA codegen).

## Files Changed

### Replaced (from `../FlyDSL/kernels/`)

| File | Description |
|------|-------------|
| `aiter/ops/flydsl/kernels/gemm_mxscale_gfx1250.py` | Main GEMM kernel — replaced with `gemm_fp8fp4_gfx1250.py` content |
| `aiter/ops/flydsl/kernels/gemm_common_gfx1250.py` | TDM/LDS helpers — new version adds `WGP_BARRIER_ID`, `imm_byte_offset` for `lds_load_b128_raw`, `cache_policies`/`enabled` for `issue_tdm_loads` |
| `aiter/ops/flydsl/kernels/pipeline_utils.py` | Pipeline tail plan helpers — minor updates |

### Modified

| File | Description |
|------|-------------|
| `aiter/ops/flydsl/kernels/moe_grouped_gemm_mxscale_gfx1250.py` | Grouped wrapper — adapted `_compile_base_a8w4_gemm` and launch closures |

### Added

| File | Description |
|------|-------------|
| `run_moe_test.sh` | Correctness test runner (route+quant+scatter tests) |
| `run_moe_perf.sh` | Performance benchmark runner |

## What the New Kernel Brings

### New compile-time parameters (not in old kernel)

| Parameter | Description |
|-----------|-------------|
| `scale_mode` | `"mxscale"` (E8M0 block scale) or `"ptpc"` (per-token/per-channel fp32 scale) |
| `weight_nt` | GFX12+ TH_NT non-temporal cache hint for weight B TDM loads |
| `b_streaming` | B-streaming compute schedule |
| `fp8_schedule` | `"auto"`, `"quadrant"`, `"deep-pipeline"` for FP8 |
| `scale_load_path` | `"tdm"` (default), `"vgpr"` (bypass TDM/LDS for scales), `"vgpr_ab_split"` |
| `tdm_b_split` | Split B across two loader waves (ablation) |
| `tdm_load_only` | Selective TDM load (ablation) |
| `tdm_force_tensor` | Force all loader waves to load same tensor (ablation) |

### Old kernel parameters removed from new kernel

| Parameter | Description |
|-----------|-------------|
| `M` | Compile-time M (new kernel uses runtime M only) |
| `batch_count` | Batched expert dispatch |
| `grouped_masked_m` | Per-expert M masking |
| `grouped_persistent_m` | Persistent-M tile scheduling |
| `persistent_workers` | Number of persistent worker WGs |
| `stage1_act` | Fused silu/swiglu epilogue in GEMM |
| `stage1_weight_layout` | Gate/up weight interleaving |
| `epilogue_bias` | Fused bias add in epilogue |

### Structural differences

| Aspect | Old Kernel | New Kernel |
|--------|-----------|------------|
| Core function | `compile_mxscale_gemm()` | `compile_fp8fp4_gemm()` |
| Launch variants | 6 (plain, masked, masked_persistent, x bias) | 1 unified |
| Launch args | `(C, A, B, As, Bs, M, N, stream)` | `(C, A, B, As, Bs, M, N, lda, ldc, stream)` |
| Kernel args | 11 (incl bias, masked_m, prefix, map) | 9 (no bias/mask/prefix/map) |
| JIT cache | None | `@functools.lru_cache(maxsize=256)` |
| LDS layout | Single arena | Segmented (5x64KB for FP8 256x256x128) |
| TDM desc | `tdm_ops.make_tensor_descriptor_2d(...)` | `_make_tdm_desc(early_timeout=..., oob_outer_bound=...)` |
| Inst prefetch | Inline ASM `s_prefetch_inst` | `_s_prefetch_inst_burst()` helper |

## What Was Done to Adapt

### 1. Import path fixup
- `from kernels.xxx` → `from aiter.ops.flydsl.kernels.xxx`

### 2. flydsl compatibility shim
- `_make_tdm_desc()` filters unknown kwargs (`oob_outer_bound`, `early_timeout`) for flydsl 0.2.0 which doesn't support them yet
- `lda`/`ldc` runtime stride args → replaced with compile-time `K_packed_a`/`N` constants (flydsl 0.2.0's `make_tensor_descriptor_2d` doesn't accept runtime stride values)

### 3. Re-added batching/masking to new kernel
- Added `batch_count`, `grouped_masked_m`, `grouped_persistent_m`, `persistent_workers` params to `compile_fp8fp4_gemm()`
- Added `arg_masked_m`, `arg_m_tile_prefix`, `arg_m_tile_map` to kernel function signature
- Added `batch_idx` derivation from `gpu.block_id("x")`: `batch_idx = bx / m_tiles_per_batch`
- Added `batch_*_base` offsets to all TDM descriptors (A, B, A_scale, B_scale) `global_offset` and `tensor_shape`
- Added `batch_m_base` to all C output row addressing (epilogue stores, TDM store descriptor, atomic adds)
- When `grouped_masked_m=True`: load `valid_m` from `arg_masked_m[batch_idx]`, set `m_idx = valid_m` for per-expert OOB clipping
- Added `tile_has_work = blk_m < m_idx` guard around epilogue stores
- Added `launch_mxscale_gemm_masked` and `launch_mxscale_gemm_masked_persistent` launch variants
- Updated return logic to pick correct variant based on `grouped_masked_m`/`grouped_persistent_m`
- Updated `cache_tag` tuple with all new params

### 4. Updated grouped wrapper
- `_compile_base_a8w4_gemm()` restored to pass `batch_count=cfg.experts`, `grouped_masked_m=True`, etc.
- Stage1/stage2 launch closures restored to single batched `_run_compiled()` call (no per-expert loop)

## Current Status

### Correctness

| Config | Result |
|--------|--------|
| E=8, T=32 | PASS (rel_l2=0.0135) |
| E=8, T=256 | PASS (rel_l2=0.0135) |
| E=64, T=256 | PASS (rel_l2=0.0141) |
| E=128, T=256 | PASS (rel_l2=0.0152) |
| E=256, T=256 | PASS (rel_l2=0.0145) |
| E=256, T=1024 | PASS (rel_l2=0.0143) |
| E=256, T=2048 | **FAIL** (rel_l2=0.6525) |
| E=256, T=4096 | **FAIL** (rel_l2=0.8432) |

T>=2048 failure root cause: buffer resource descriptor `num_records` is 32-bit (4GB max). At E=256, T=2048, the stage1 tmp buffer is `256 * 2048 * 4096 * 2 = 4GB`, exactly at the overflow boundary. This is a pre-existing hardware limitation.

### Performance (E=256, T=4096, a8w4, silu)

| Metric | Old Kernel | New Kernel | Delta |
|--------|-----------|------------|-------|
| GEMM kernel (per iter) | ~3,000 us | 12,879 us | **4.3x slower** |
| finalize_act_silu | 0 (fused) | 2,124 us | **new overhead** |
| route+quant+scatter | 158 us | 114 us | OK |
| quant_preshuffle | 475 us | 476 us | OK |
| gather_reduce | 49 us | 49 us | OK |
| **End-to-end fused_moe** | **5,667 us** | **28,581 us** | **5.0x slower** |

### Performance gap root causes

1. **Different WMMA/TDM codegen**: The new kernel's inner loop body (TDM pipeline, LDS layout, WMMA schedule, scale handling) is entirely different from the old kernel — not just parameters but the fundamental code structure. This is not a simple parameter tuning gap.

2. **Lost fused silu epilogue**: Old kernel had `stage1_act="silu"` that fused gate*silu*up into the GEMM epilogue (zero extra kernel launch). New kernel doesn't have this — adds a separate `finalize_act` kernel at 2.1ms/iter.

3. **Tile masking overhead**: The `tile_has_work` guard wraps the epilogue but not the compute — workgroups for empty expert tiles still execute full TDM loads + WMMA, wasting compute.

## Lessons Learned

1. **The two kernels are fundamentally different implementations** (5544 lines of diff across 3300 lines). Replacing the whole file replaces the entire codegen, not just "swapping the kernel".

2. **The right approach is surgical**: keep the old kernel's framework (batch/mask/persistent/fused-silu) intact, and port specific optimizations from the new kernel (e.g., `scale_load_path="vgpr"`, `fp8_schedule="deep-pipeline"`, LDS segmented layout) one at a time.

3. **flydsl version matters**: The new kernel was developed against a newer flydsl (from `build-fly-coexec-samebank`) that supports runtime TDM strides, `oob_outer_bound`, `early_timeout`. The pip flydsl 0.2.0 doesn't support these, requiring compatibility shims.

## Recommended Next Steps

1. **Revert to old kernel**, then selectively port optimizations from the new kernel
2. **Prioritize**: identify which specific new-kernel feature gives the biggest perf win (likely `scale_load_path="vgpr"` or FP8 deep pipeline)
3. **Port incrementally**: add one feature at a time, benchmark after each
4. **Fix 4GB overflow**: for T>=2048 at E=256, need flat global stores or split-buffer addressing
