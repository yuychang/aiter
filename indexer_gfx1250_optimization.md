# Indexer (gfx1250 preshuffle) 优化分析

对照 MLA 的 gfx1250 gluon 实现,针对 indexer 的 `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`
(`aiter/ops/triton/gluon/pa_mqa_logits.py:436-564`,gfx1250 分支)提出的优化点。

## 背景

- indexer 与 MLA 的 q@k 数学第一步同构(fp8 矩阵乘 `Q@K`)。
- 区别:q@k 之后,**indexer 沿 head 轴 reduce 出 `logits[N]` 做 top-k 选 token**;
  **MLA 沿 N 轴 softmax 保留 head,再做 P@V**。
- indexer 在 decode 时是访存(TDM load)主导,每个 block 只有一次小 wmma。

indexer gfx1250 内循环结构(`pa_mqa_logits.py:502-563`):
```
async_load(blk j+1) -> async_wait(1) -> load+5D-unswizzle K -> wmma ->
buffer_load(k_scale) -> *k_scale -> relu -> *weight -> mask -> reduce(axis=0) -> buffer_store
```

---

## 高优先 / 高收益

### 1. buffer 深度 2 -> 可配环形(借鉴 MLA)
- 现状:indexer `NUM_BUFFERS=2` 硬编码双缓冲(`pa_mqa_logits.py:437`),
  `async_wait(1)` 只允许一个 TDM load 在飞。
- MLA:`NUM_STAGES` 可配 constexpr + `get_next_buffer_id` 的 `(id+1)%NUM_STAGES`
  (`mla.py:797-801`)+ `[NUM_STAGES]+block_shape` 分配(`mla.py:715`)。
- 做法:把 `NUM_BUFFERS` 提到 3,`async_wait(2)`,让 K 的 TDM 延迟被两轮计算掩盖。
  KVBlockSize=128 时每块 16KB,LDS 放得下 3~4 个 buffer。
- 收益:直接打 indexer 的访存瓶颈。改动局部,风险低。

### 2. K unswizzle 封装 + 复用固定 layout(借鉴 MLA)
- 现状:`pa_mqa_logits.py:526-541` 每个 block 现场做
  `reshape(5D).permute().reshape().permute().load()`,内联让编译器每轮重算 address。
- MLA:封装成 `lds_unshuffle_kv_lora(buffer_id).load(layout=K_DOT_LAYOUT)`
  (`mla.py:1008`),unshuffle layout 是 cfg constexpr,编译期定死。
- 做法:抽成 helper + 固定 `K_DOT_LAYOUT` constexpr,降低寄存器压力和指令数。

### 3. scale 并入 async 预取(MLA + gfx942/950 混合)
- 现状:`pa_mqa_logits.py:544-547` `k_scale_f` 用 `gl.amd.cdna3.buffer_load` 同步读 global,
  **卡在 wmma 之后**,成为串行点。
- 来源拆解:
  - "提前预取 scale" 的思路 gfx942/950 indexer 已有
    (`k_scale_f_next_0/next_1` 提前一轮 buffer_load,`pa_mqa_logits.py:801-806, 872-877`)。
  - "把 scale 走 TDM async_load 进 shared、和 K 一起 wait" 是 MLA 手法
    (`mla.py:1045-1050`,nvfp4 路径)。
- 做法:k_scale(每 block 一个 `[KVBlockSize]` 向量)和 K 一起 async 预取,
  消掉 wmma 后的同步 load 气泡。

---

## 低优先 / 低收益

### 4. reduce(axis=0) 的 lane 浪费(自行分析,需实测)
- `reduce(o, axis=0)` 沿 head 轴求和(`pa_mqa_logits.py:554`),`o` 是 `[heads, KVBlockSize]`,
  reduce 后只剩 `[KVBlockSize]`,大部分 lane 结果被丢弃。
- gfx1250 与 gfx942/950 indexer 都有这个 reduce,两边都没针对降维优化;MLA 不做此 reduce。
- 设想:能否用 wmma 直接出规约(把 relu 后的 `o` 与 `weight[heads,1]` 的乘加合进矩阵乘)?
  —— relu 在中间挡住了这条路。退一步:确认 reduce 的 combine 走 DPP/swizzle 而非 LDS round-trip。
- 推测性,最不确定,最需要验证。

### 5. relu 提前 + 缩放融合(自行分析)
- 现状:`pa_mqa_logits.py:548-550` `o*k_scale -> max(o,0) -> o*weight` 三趟独立 VALU。
- 数学洞察:k_scale = amax/240 恒正,故 `max(o*s,0) = max(o,0)*s`,relu 可提前到 ×k_scale 之前。
- 做法:relu 提前后,k_scale(per-N)与 weight(per-head)是外积,
  可合成 `max(o,0) * k_scale[None,:] * weight[:,None]` 一次 FMA 链。
- 零风险,但收益小。

### 6. iglp 调度提示(借鉴 gfx942/950)
- gfx942/950 indexer 密集使用 `_amd_iglp_sched_group_barrier(MFMA/BUFFER_LOAD, ...)`
  手工编排(`pa_mqa_logits.py:745-752` 等)。
- gfx1250 indexer 和 MLA gfx1250 都没用,靠 TDM 自然重叠。
- 不一定该照搬到 gfx1250(TDM 模型不同),需 profile 看 TDM 是否真把 wmma 喂满。

---

## 来源分类汇总

| # | 优化点 | 来源 | 优先级 |
|---|---|---|---|
| 1 | buffer 深度 2->可配环形 | 借鉴 MLA | 高 |
| 2 | K unswizzle 封装+复用 layout | 借鉴 MLA | 高 |
| 3 | scale 并入 async 预取 | MLA + gfx942/950 混合 | 高 |
| 4 | reduce(axis=0) lane 浪费 | 自行分析(需实测) | 低 |
| 5 | relu 提前 + 缩放融合 | 自行分析 | 低 |
| 6 | iglp 调度提示 | 借鉴 gfx942/950 | 低 |

## 建议

动手前先用 `rocprofiler-compute` 跑现状 indexer,确认是 TDM-bound / wmma-bound / VALU-bound,
再决定主攻 #1 还是 #4/#5,避免优化非瓶颈。
高收益组合:#1(buffer 深度->3)+ #3(scale 预取)一起做,#5 顺手化简。
