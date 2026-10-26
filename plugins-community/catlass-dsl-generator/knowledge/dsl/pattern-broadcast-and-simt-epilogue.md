---
type: CATLASS DSL Programming Concept
title: 广播加载与 SIMT 收尾
description: 用 DIST_BRC_B32 做行广播，用 simt 模式的 thread_block_dim 处理非对齐标量写。
tags: [catlass-dsl, vector, broadcast, simt, epilogue]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-10-23T12:00:00Z'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-10-23T12:00:00Z'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: 广播加载与 SIMT 线程索引
dsl_version: ascend-catlass-dsl 2.1.0
arch: [c310]
---

# 接口与概念

两类"不舒服"的访问各有专用手段：

1. **把 [N] 的统计量广播到 [N, M] 的每一行**：不必手工 interleave 展开，可用广播加载。
2. **输出行宽不是 32B 的整数倍时，逐行向量/MTE 写会报对齐错误**：
   例如 `[M, K+1]` 的 fp32 输出，K=128 时行宽 `129 × 4 = 516B`，`516 mod 32 = 4`，
   逐行跨步访问会触发
   `The address for VEC to access UB is not aligned`（errcode 340）或
   `The MTE non-atomic instruction address is not aligned`（errcode 80）。
   做法：**逐行结果用 SIMT 标量逐元素写**；行宽本就 32B 对齐的连续输出
   （如 `[M]` 的 local_max）才用 `tla.copy` 连续写。
   **UB 内的暂存缓冲同理**：`[rows, K+1]` 的 stride（516B）同样不允许向量访问，
   应把暂存建为 `[rows, K]`（K=128 → 512B = 16×32B，对齐），payload 的第 0 列与
   第 1..K 列分开处理。

# 用法

```python
# 广播
brc = tla.tile_view(stat_ub, tla.make_shape(1), tla.make_coord(i)).load(
    params=tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
)

# SIMT 收尾
with tla.vec.func(mode="simt", thread_block_dim=128):
    tid, _, _ = tla.arch.thread_idx()
    nthreads, _, _ = tla.arch.thread_block_dim()
    for r in tla.range(tid, ROWS, nthreads):
        gm_out[row0 + r, 0] = sum_ub[r]
```

# 约束

- SIMT 区域的栈预算有限（实测出现 `stack object size exceeded vf stack size`），
  区域内不要捕获大张量或做深调用；复杂逻辑放在区域外。
- UB 指针必须在 SIMT 区域**内**构造为 tensor，捕获外部裸指针会报错。
- 广播加载只适用于"同值铺满一行"的语义，逐元素不同仍需向量运算。
- SIMT region 要求被访问的 buffer 是**静态 shape**；对输出张量用
  `mark_layout_dynamic` 会报 `requires statically shaped buffers`。
- UB 标量访问的基址必须直接来自 `tla.allocate` 返回的指针；先对 arena 指针做加减
  再标量访问会报 `tla.vec.func cannot outline scalar access`。

# 收益参考

同一形状下把 `[M, K+1]` 的写回从"向量方式"改成 SIMT 标量逐元素写后，
该段写回耗时从约 **225 秒**降到约 **1 毫秒**（原先向量方式直接触发 errcode 340/80）。

接口行为以固定提交的实现为准。[^api]

[^api]: 固定提交的源码与文档中对应实现。

# 代码模式

见上文 `NormalLoadParams(load_dist=DIST_BRC_B32)` 与 `vec.func(mode="simt", thread_block_dim=...)` 片段。

# 失败表现

- SIMT 区域报 `stack object size ... exceeded vf stack size`；
- 捕获外部 UB 裸指针报 "Buffers must be captured as tensors"。

# 验证方法

小 shape 下与 host 侧逐元素对拍；把区域内的复杂逻辑移到区域外后再测。
