---
type: CATLASS DSL Programming Concept
title: 统计量打包在同一 UB 张量的列偏移里
description: 把行 max、临时值、行 sum 存在 [rows, 64] 张量的第 0/16/32 列，配合广播加载减少 buffer 数量。
tags: [catlass-dsl, ub, layout, broadcast, statistics]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-10-23T15:40:00+08:00'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-10-23T15:40:00+08:00'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: NormalLoadParams 与 DIST_BRC_B32 广播加载
dsl_version: ascend-catlass-dsl 2.1.0
arch: [c310]
---

# 接口与概念

在线 softmax 每行要维护多个标量（running max、临时块 max、running sum）。为每个量
单独 allocate 一个 [rows] 缓冲会占用多份 UB 和小粒度访问；把它们放进同一个
[rows, 64] 张量的不同列，用列偏移区分，即可共享一个 buffer 与广播加载路径。

# 用法

```python
stats = tla.make_tensor(sp, tla.make_layout(
    tla.make_shape(half, 64), tla.make_stride(64, 1)))
brc = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)

old_max = tla.tile_view(stats, tla.make_shape(1, 1), tla.make_coord(rr, 0)).load(params=brc)
old_sum = tla.tile_view(stats, tla.make_shape(1, 1), tla.make_coord(rr, 32)).load(params=brc)
tla.tile_view(stats, tla.make_shape(1, 1), tla.make_coord(rr, 16)).store(
    block_max, params=tla.params.UnalignStoreParams(), mask=one)
```

列约定：0 = 行 max，16 = 临时标量，32 = 行 sum。

# 代码模式

见上文片段；写用 UnalignStoreParams + 单 lane mask，读用 DIST_BRC_B32 广播。

# 约束

- 行宽（如 64）必须能容纳所有列位并保持 32B 对齐，否则标量 store 会跨行。
- 各列语义要写在代码注释里，避免后续改动错位。
- 广播加载只适用于"同值铺满一行"；逐 lane 不同的量不能塞进这种布局。

# 失败表现

- 列偏移写错：max 与 sum 互相覆盖，结果表现为 max 异常或 sum 为 0。
- 行宽不是 2 的幂时，向量访问跨行导致相邻行被污染。

# 验证方法

小 shape 下打印每行 (max, sum) 与 host 侧对拍；检查 store 的 mask 是否只放行 1 lane。

[^api]: 固定提交的 NormalLoadParams / UnalignStoreParams 实现。
