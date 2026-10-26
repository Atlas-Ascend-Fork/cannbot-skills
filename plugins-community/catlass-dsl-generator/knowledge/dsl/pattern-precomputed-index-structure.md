---
type: CATLASS DSL Programming Concept
title: 预计算索引结构（bucket + 链表）替代 gather
description: 逐 chunk 寄存器 gather 的成本随 V 线性增长（O(M·K·V/64)）；预计算索引把它降到 O(M·K + M·V/BN)，大 V 时快两个数量级。
tags: [catlass-dsl, gather, index, scatter, simt, fallback]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-10-23T15:20:00+08:00'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-10-23T15:20:00+08:00'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: gather 与 SIMT 标量访问的契约
dsl_version: ascend-catlass-dsl 2.1.0
arch: [c310]
---

# 接口与概念

> **选型优先级**：本条描述的「预计算索引 + SIMT 标量读」是已知性能最优的做法，优先采用；
> 逐 lane 标量读次之；`tla.gather` 是最后选项（2.1.0 下行为不稳，见下方实测契约）。


**为什么需要预计算索引**：`tla.gather(x, idx)` 的 `x` 会被转成 VectorSSA 类型校验，
容量上限为单寄存器宽（f32 ≤64 lane、bf16 ≤128 lane），无法从大 UB 缓冲按 lane 任意取数。
把大表拆成若干个 64-lane 窗口逐次 gather 虽然可行，但每行每 V tile 要做
`K × (BV/64)` 次，总量 `O(M·K·V/64)`——**含 V 因子**，大词表下不可行。
预计算索引把它降到 `O(M·K + M·V/BN)`，且消费阶段与 V 无关。


当目标架构没有 scatter、gather 的表宽又受单寄存器限制时，可以用**预计算索引**把
"运行时按索引取数"改写成"运行时按链表顺序写数"：先用一遍 SIMT 把每行的命中位置
组织成桶与链表，再在消费阶段沿链表顺序取值并直接写入目标位置。

# 用法

两遍结构：

1. **建索引（每 M 块一次）**：清零 `heads[M, nv]`，其中 `nv` 是 V 方向块数；
   对每行每个 k，把 id 映射到桶 `b = id / bn`，把"桶内计数 + 链头"打包进
   `heads[row, b]`，用 `links[row, k]` 指向同桶内上一个 k。
2. **消费（每个 V 块一次）**：每行读 `heads[row, vi]`，沿 `links` 走链，
   把 `x[row, tok]` 写入 `payload[row, k]`。

因为每个 (row, k) 只属于一个 V 块，写入不需要原子操作；每行由固定线程负责，
天然满足单写者原则。

# 若坚持用 gather：2.1.0 的实测契约

在 ascend-catlass-dsl **2.1.0** 上用最小探针实测（f32、单 block、64 元素索引）：

| 用例 | 结果 |
|---|---|
| 表宽 64，索引由 `tla.arange(...)` 生成 | ✅ 结果正确 |
| 表宽 64，索引从 UB `tensor.load()` 直接得到 | ❌ **静默返回全 0**（不报错） |
| 表宽 64，索引 = 载入后经任一向量运算（如 `add(iv, 0)`） | ✅ 结果正确 |
| 表宽 128 | ❌ 明确报错 `VectorSSA valid_lanes must be <= 64 for f32, got 128` |
| 表宽 256 | ❌ 同上，`got 256` |

结论：

- **表宽上限 = 单寄存器宽度**（f32 64 lane），超限会在 lowering 阶段明确报错；
- **索引必须来自向量运算**：直接把刚从 UB 载入的向量当索引会静默得到全 0，
  加一次任意向量运算（哪怕 `+0`）即恢复正常——这是 2.1.0 的一个坑，不报错；
- 因此跨多窗口取值时，按 64 列分块、每块各 gather 一次仍是唯一可行形态，但**每块的索引都要经向量运算得到**（通常本来就有 `sub`/`clamp`，正好满足）。

# 代码模式

```python
# 建索引
code = heads[row * nv + bucket]
links[row * kp + ki] = (code % base).to(tla.Int32)
heads[row * nv + bucket] = (code // base + one) * base + ki.to(dtype) + one

# 消费
selected = (code % base).to(tla.Int32) - 1
for _ in tla.range((code // base).to(tla.Int32)):
    tok = ids[row * kp + selected] - start
    payload[row * (n_k + 1) + selected + 1] = x[gr, tok - vi * bn] * inv_t
    selected = links[row * kp + selected] - 1
```

# 约束

**成本模型（选型时按此估算，不要凭直觉）**：

| 方案 | 每行每 V tile 的操作 | 全算子总量（按 M、V、K 估算） |
|---|---|---|
| 逐 chunk 寄存器 gather | `K × (BV/64)` 次 gather + 掩码累加 | `M×K×V/64` |
| 预计算索引（本条目） | 建索引均摊 + 沿链 **K 次** | `M×K + M×V/BN` |

gather 路径的成本里含 `V/64` 因子，V 越大越不可行；索引路径的消费阶段每行只走 K 次
（每个 id 恰好访问一次），**与 V 无关**。虽然索引路径全是标量 GM 访问（单次比 SIMD 慢），
但两个数量级的操作数差距会让它整体胜出。

其他约束：

- 索引结构大小随 `nv = ceil(V/BN)` 增长，BN 取小会让建索引清零的开销上升；
- 每行由固定线程负责，天然单写者，不需要原子；
- 只有"没有可行寄存器 gather"或"V 很大"时才值得用；V 很小时直接 gather 更简单。

# 失败表现

- 编码错误：链断导致漏选、链成环导致重复写；
- `heads` 的打包基数取错（必须与 K 对齐），表现为选中位置整体错位；
- BN 取过小、V 极大时，建索引的清零开销反超收益。

# 验证方法

用最小 V 与小 K 对拍，确认每个 (row, k) 恰好命中一次；统计标量写次数；
同 shape 下与寄存器 gather 版本对比耗时——V 越大本方案优势越明显。

接口行为以固定提交的实现为准。[^api]

[^api]: 固定提交的源码与文档中对应实现。

# 反例（实测）

同一算子在 V 远大于 K 的形状下：

- 按 V 块逐块扫描全部 K（算术掩码选择）：耗时随 `nv` 线性增长
- 预计算索引（每个 id 只访问一次）：耗时与 V 基本无关

两者数值都正确，实测前者耗时约为后者的 `nv` 分之一到数倍之间（取决于实现细节）。
实现前若打算"每块扫一遍"，先确认 `nv` 的量级。
