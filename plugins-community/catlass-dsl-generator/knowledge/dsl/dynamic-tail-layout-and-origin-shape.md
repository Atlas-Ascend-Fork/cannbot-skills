---
type: CATLASS DSL Programming Concept
title: 动态尾块 Layout 与 Origin Shape
description: 区分固定 storage tile、逻辑有效区域和真实地址 stride，构造可用于 GM、UB、L1 的动态尾块 tensor view。
tags: [catlass-dsl, dsl, tail, layout, origin-shape, stride, leaf-count, dynamic-layout]
status: stable
generated: {by: human:maintainer, at: '2026-09-20T00:00:00-07:00'}
verified:
  - {by: human:maintainer, at: '2026-09-20T00:00:00-07:00'}
sources:
  - id: layout
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/docs/en/core_concepts/tensor_layout.md
    title: CATLASS Tensor and Layout concepts
  - id: core
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: CATLASS tensor and layout API implementation
operator_families: [elementwise, matmul, sparse-attention]
arch: [c310]
---

# 接口与概念

layout 的 shape 与 stride 必须有相同叶子数；coordinate 也必须与该结构兼容。storage shape
描述实际 tile，`origin_shape` 描述逻辑有效区域，二者不能互相替代。[^layout][^core]

# 用法

矩阵计算通常保持固定 storage tile，例如 `128x128`，尾部通过 `origin_shape`、mask 或零填充
限制有效元素。GM packed/TND 源可使用跨 head stride，grouped workspace 使用连续 row stride；
两者不能共享一个“看起来同形状”的 layout。

# 代码模式

```python
full = tla.make_layout(
    tla.make_shape(128, 128),
    tla.make_stride(128, 1),
    origin_shape=tla.make_shape(valid_rows, 128),
)
```

对于 TND `[T,H,D]` 的某个 head，连续 token 行的 stride 是 `H*D`；对 grouped workspace
`[B*H,padded_seq,D]`，连续行 stride 是 `D`。构造 source/destination view 时分别表达。

# 约束

- shape/stride/coordinate 的嵌套结构和叶子数必须一致。
- `origin_shape` 不会改变 pointer、stride、DMA 对齐或底层 allocation。
- L1/L0 计算 layout 不应仅因有效行较少而改变物理 tile 结构。
- 动态有效行不能导致负 extent；先 clamp 到 `[0,tile]`。
- 短尾 copy 是否支持动态 extent必须按具体地址空间和 copy 参数验证。

# 失败表现

- shape and stride leaf count mismatch：layout 结构不一致。
- layout/coord/ptr fields must be non-null：动态路径产生不完整 descriptor。
- 尾块设备异常：物理搬运仍使用完整 tile，但 allocation 或源地址不足。
- 前若干行正确、边界后错误：stride、half split 或 storage/origin 语义混用。

# 验证方法

静态打印 logical/storage shape、stride 和起始 pointer；覆盖 1/15/16/17/63/64/65/127/128
行。先验证 GM round-trip，再验证 UB/L1 consumer，最后验证计算输出。

[^layout]: 固定提交 layout 文档对 shape、stride、coordinate、logical shape 的定义。
[^core]: 固定提交 core API 对 tensor/layout 构造和校验的实现。
