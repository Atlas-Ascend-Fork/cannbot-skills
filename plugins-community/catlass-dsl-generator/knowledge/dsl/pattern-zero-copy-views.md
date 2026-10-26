---
type: CATLASS DSL Programming Concept
title: 零拷贝视图：用 layout 表达转置与切片
description: 通过 ColumnMajor/ptr 算术构造 GM 视图，避免 host 侧预转置与额外搬运。
tags: [catlass-dsl, layout, transpose, column-major, zero-copy]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-10-23T12:00:00Z'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-10-23T12:00:00Z'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: layout 直接表达 stride 与 layoutTag
dsl_version: ascend-catlass-dsl 2.1.0
arch: [c310]
---

# 接口与概念

矩阵运算常需要 `B^T`。在 host 侧预转置会多一次全量读写（大权重可达 GB 级），
而 MMA D 的 B 操作数只需要"逻辑上转置"，可以直接用 GM 上的原始布局构造视图。

# 用法

对 `[V, H]` 行主序权重，取元素 `(k, n) = W[v0+n, h0+k]` 的 `[K, N]` 子块：

```python
gm_b_tile = tla.make_tensor(
    gm_weight.ptr + v0 * H + h0,
    tla.make_layout(
        tla.make_shape(K, N),
        tla.make_stride(1, H),          # 第 0 维步长 1，第 1 维步长 H → 逻辑转置
        layoutTag=tla.arch.ColumnMajor,
    ),
)
```

再 `make_tensor_like(l1_ptr, gm_b_tile, tla.arch.nZ)` 送入 L1。

# 约束

- 视图必须保证实际访问的元素全在原始张量范围内（`v0+N<=V`、`h0+K<=H`）。
- 跨行 stride 非线性对齐时，搬运效率下降，仍优于全量转置。
- 布局语义（nZ/zN、RowMajor/ColumnMajor）要与 MMAD 操作数契约匹配。

接口行为以固定提交的实现为准。[^api]

[^api]: 固定提交的源码与文档中对应实现。

# 代码模式

见上文 `make_tensor(ptr, make_layout(shape, stride, layoutTag=ColumnMajor))` 片段。

# 失败表现

- 数值整体转置错位；
- 访问越界（视图范围超出原始张量）。

# 验证方法

用 host 侧显式转置的结果对拍；先用 1 个 tile 验证视图正确，再接入 MMAD。
