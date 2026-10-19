---
type: CATLASS DSL Optimization Guide
title: AIV 向量布局与 MTE
description: 面向纯 AIV 算子及 mixed kernel Vector Stage 的布局、连续访问和任务粒度优化候选。
tags: [catlass-dsl, optimization, aiv, mte, vector, layout, task-coarsening, grouped-dma]
status: stable
generated: {by: human:caijianlong, at: '2026-08-06T00:00:00+08:00'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-06T00:00:00+08:00'}
  - {by: process:catlass-dsl-kda-architecture-review, at: '2026-10-11T00:00:00+08:00'}
sources:
  - id: dsl
    resource: https://gitcode.com/cann/catlass/blob/6ccf88e89723b65461e9921047c7970a71b67b42/python/tla_dsl/catlass/core_api.py
    title: CATLASS DSL vector examples and APIs
operator_families: [elementwise, scan, reduction, mixed, attention, linear-attention, gated-delta-rule, kda]
arch: [c310]
---

# 接口与概念

适用于 scan、cumsum、elementwise、轻量归约、数据重排，以及 mixed kernel 中可独立分析的
Vector Stage。先从 profile 区分 `aiv_mte2_time`（GM 到 UB）、`aiv_mte3_time`（UB 到 GM）、
`aiv_vec_time` 和 `aiv_scalar_time`；`aicore_time=0` 或 `cube_util=0` 说明不应把 Cube
当作默认优化对象。[^dsl]

MTE2 高时优先检查输入布局和 load 粒度；MTE3 高时检查输出连续性、store 宽度和 task
切分；vector/scalar 高时检查 unpack、shuffle、分支、循环和地址计算。不要仅因总耗时高
就默认怀疑 Cube。

# 用法

生成和重构 Vector Stage 时使用 SIMD 风格，不用 SIMT 逐线程循环模拟规则化 scan、归约、
逐元素或布局运算；以 `tla.vec.func(mode="simd")`、连续 tile、向量寄存器及 lane mask 表达。
从其他实现借鉴时应转换任务和访存形态，不直接照搬 thread/program 的标量执行风格。[^dsl]

mixed kernel 中可表达的矩阵运算应利用 Cube，并在依赖、容量和 ready/free 证明成立后，让
Cube 当前任务与 Vector 相邻任务的准备或后处理重叠。纯 Vector 计算无需人为增加无效矩阵
运算；实际收益以相同配置的总时延与 MTE/vector/scalar 指标验证。

每轮只提出一个可证伪轴，例如 `aiv-mte-compact-input-layout`、
`aiv-mte-wider-contiguous-store`、`aiv-task-coarsening-contiguous-dim`、
`aiv-vector-unpack-cost-reduction` 或 `aiv-scalar-address-simplification`。

从 Triton 或其它实现借鉴时，只提取数据访问形态：grid 对 batch/sequence/head/feature/group
的映射、每个 program/tile 覆盖的连续元素、mask 是否只处理边界、store 的连续程度，以及
scan/reduce/elementwise 的计算轴是否匹配连续内存轴。将这些信息转换为 CATLASS DSL 的
vector 长度、GM stride、task 映射和 UB 布局，不机械复刻 API。

# 候选策略

- GM 输入不要保存 padding、插零 lane 或对齐空洞；在 host/前序阶段压紧连续有效元素，再
  在 AIV 中用 cast、lane 拆分、interleave 或 reshape 恢复计算形态。
- 将多个连续真实小向量合并为更宽的 vector load，确保 GM 读取的都是有效元素。
- 合并 GM 上连续的 head、feature、channel、小 batch 或 group 维度，以减少 kernel task
  数和小块 store。sequence、chunk 或 tile 步长较大时，可在一个 task 中合并少量连续逻辑
  单元；报告时必须区分逻辑工作单元与实际 kernel task。
- mixed kernel 有两个 AIV 且局部任务互相独立时，把“两个 AIV 各自拥有连续 task group”与
  AIV0-only baseline 同时纳入候选；分别预算两份 UB/L1 slot 和 AIC 消费顺序，不能只扩大
  单个 AIV 的线程数来替代 owner 级并行。
- 借鉴 Triton 时提取 grid 映射、连续 load/store 宽度、边界 mask 和计算轴，而非照搬 API。
- group/vector 粒度变大后必须比较 MTE、vector、scalar 和 UB 占用。MTE 下降而 vector 或
  scalar 上升导致总时延回退时，应拒绝该候选。

连续维合并的收益应体现在 task 数下降、row store 变宽、`aiv_mte3_time` 小块写回/地址
开销下降，且 device 内 task 循环更接近真实连续访问。只看逻辑二维或三维划分容易误判
实际并行度和搬运粒度。

双 AIV 与 group DMA 在 mixed kernel 中通常是耦合候选。实现前必须列出
`(logical_task, group_member, AIV) -> [output_begin, output_end)` owner 表，以及每个 owner 的
私有 UB/L1 slot、GM stride、copy mode 和 ready/free 代际。有效输出区间必须形成无重叠、无遗漏
的覆盖。若 group DMA 的收益依赖双 owner 并行，不能用 AIV0-only 下单独增加 group 循环所得的
性能结果否定完整候选；应把必要的 owner、slot 和 DMA 改动作为一个可证伪架构包共同验证。[^dsl]

# 代码模式

```text
axis_id: aiv-mte-wider-contiguous-store
hypothesis: 合并连续输出维度可降低 MTE3 和地址计算。
validation: 完整正确性后比较总时延、MTE3、vector 与 scalar 子项。
```

# 约束

任何改变输入元素数、task 映射或 buffer 大小的方案都必须更新 compile cache key。覆盖
`group=1`、非整除维度、dtype、功能开关、layout、小 shape 与性能目标 shape。先完整
正确性，再在同一配置比较总时延和 MTE/vector/scalar 子项。[^dsl]

增大 group/vector 粒度可能降低 MTE2/MTE3，但也会增加 AIV unpack/interleave/shuffle、
临时 vector 数、寄存器压力、分支与地址计算，或占用更多 UB 并降低并发。group 候选必须
逐个 profile 证伪，不得把局部子项下降当作接受依据。

# 失败表现

- 为满足 vector 宽度而保存 padding 或空洞，导致 MTE2 搬运无效 lane。
- group 增大后 unpack、shuffle 或地址计算抵消 MTE 收益。
- 仅性能目标 shape 通过，非整除边界或 `group=1` 路径错误。
- group member 地址仍使用组首元素的 row/column 基址，导致重复写首 tile 或遗漏后续 tile。
- 两个 AIV 逻辑上分工但物理 UB/L1 slot 重叠，出现跨次不稳定、旧值或 NaN。
- 忘记把 pack、layout、task 映射或 buffer 大小加入 cache key，复用了旧 kernel。

# 验证方法

单个大 case 的成功不构成通用结论。运行小 shape、非整齐 shape 和性能目标 shape 的完整
正确性，再在同一配置比较总时延和 MTE/vector/scalar 子项。引入新 layout、pack 或 task
映射时，同步更新 meta tensor 大小、host pack/unpack、golden 对比和 benchmark 中的任务
统计口径。[^dsl]

[^dsl]: 固定提交 CATLASS DSL 的向量 API 与端到端示例。
