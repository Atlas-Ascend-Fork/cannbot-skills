---
type: CATLASS DSL Programming Concept
title: 按行独占的并行划分（每 block 一个 Cube 与两个 AIV）
description: 按 M 切块让每个 block 独占完整行，行统计免跨块归约；代价是小 M 时块数不足。
tags: [catlass-dsl, parallelism, grid, mixed-kernel, reduction]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-10-23T16:00:00+08:00'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-10-23T16:00:00+08:00'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: arch.block_idx / sub_block_idx 与 grid 语义
dsl_version: ascend-catlass-dsl 2.1.0
arch: [c310]
---

# 接口与概念

把 M 维切成 BM 行一个任务，`arch.block_idx()` 跨步遍历任务，每个 block 内用一个
Cube 配两个 AIV（`sub_block_idx()` 取 0/1），每个 AIV 占 BM/2 行。因为**每一行的全部
V 列都在同一个 block 内算完**，行统计（max / sum）不需要任何跨 block 归约。

# 用法

```python
for mi in tla.range(tla.arch.block_idx(), (m + bm - 1) // bm, tla.arch.block_num()):
    with tla.vector():
        sub = tla.arch.sub_block_idx()      # 0 或 1
        row0 = mi * bm + sub * (bm // 2)
    for vi in tla.range(nv):                # 该行段的全部 V 块
        ...
```

# 代码模式

见上文；行统计量在每个 block 内闭合，最后直接写 GM。

# 约束

- 块数 = `ceil(M / BM)`，必须 >= 物理核数才能吃满设备；小 M 时会空转。
- BM 要按 2 的倍数切（两个 AIV 各半），且与 SPLIT_M 的行对半语义一致。
- 若改为按 V 切分，行统计必须跨 block 归约（多一次 GM 往返 + 同步），代价明显更高。

# 失败表现

- 小 M（如 M=2048、BM=128 只有 16 个 block）时性能显著低于大 M，甚至不如 torch。
- BM 取奇数或与 SPLIT_M 不匹配：两个 AIV 的行错位。

# 验证方法

打印 `block_num` 与实际任务数，确认任务数 >= 核数；小 M 与大 M 各测一次，
记录块数利用率；若小 M 明显劣化，在设计文档中如实标注适用域。

[^api]: 固定提交的 block_idx / sub_block_idx 实现。
