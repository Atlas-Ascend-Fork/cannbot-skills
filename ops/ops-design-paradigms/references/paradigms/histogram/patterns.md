<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 范式入口（SIMT：混合 VF）

> 本文件是 Histogram 范式的**统一入口**：场景判定 + 语言路由 + 通用层索引 + 消费规则。
>
> 本范式抽象的是 Histogram 类算子在 **SIMT** 模型下的实现方式，唯一形态为**混合 SIMT VF**：
>
> - **混合 SIMT VF**：regbase 框架（`op_host` TilingFunc + `__global__ __aicore__` 入口）+ SIMT VF 内核
>   （`__simt_vf__ __aicore__` 函数 + `asc_vf_call` 启动）。SIMT 内核以 每核分块跨步并行计数，按边界条件
>   `bins/inputsize < GM_ATOMIC_ADD_FACTOR` 选用两种模板：**局部直方图模板**（`__ubuf__` 每核分批 bins + `asc_atomic_add` 片上原子计数 +
>   `SetAtomicAdd`/`DataCopyPad`/`SetAtomicNone` DMA 原子归并）或**无局部直方图模板**（`asc_atomic_add` 直接 GM）。
>   另有第三模板**确定性模板**：需确定（可复现）浮点累加时（Ascend C 由 tiling 侧 `` `GetDeterministic()` ``==1 判定）按
>   bins 分核后核内线程按 stride 遍历本核所有 bins、每线程串行遍历全部 input、每 bin 仅单线程累加，无原子并发写，累加顺序确定（见 [generic/accumulation.md](generic/accumulation.md) §4）。
>
> 混合 VF 即 regbase 框架 + SIMT 内核，属本范式覆盖；纯 SIMD regbase（`__aicore__` + `TBuf/TQue`，无 SIMT 内核）不属本范式。
>
> 知识分两层：
>
> - **通用知识层** `generic/` — 语言无关的 bin 模型与 range 推导、bin index 计算、累加策略、线程切分与 每核分块、同步模型
> - **语言适配层** `adapters/<lang>/` — 强耦合具体开发语言的 API 真值与落地实现（提供 `ascendc` 适配层，覆盖混合 SIMT VF 形态；`dsl` 适配层，覆盖 CANNBotDSL）
>
> 范式路由（`routes.yaml` → 本文件）与语言路由（本文件 → 适配层）分离：范式路由由 `op.paradigms` 决定，语言路由由开发语言决定。

---

## 场景判定

满足以下条件的算子属于 Histogram 范式：

> **前置约束（芯片 SIMT 编程能力）**：本范式唯一实现形态为混合 SIMT VF（`__simt_vf__` + `asc_vf_call`），内核依赖芯片的 SIMT 编程能力。**目标芯片不支持 SIMT 编程能力时，不进入本范式**

1. **索引化分桶计数**——输出是一个**固定大小的直方图**。D=1 时为 `hist[nbins]`，每个输入元素经「bin index 计算」映射到 `[0, nbins)` 的一个桶，再累加到 `hist[idx]`；D≥2 时每样本 D 个特征各自映射到一维 bin index、组合为 D 维 bin 下标，按行优先展平为 `idx ∈ [0, ∏ nbins_axis)`，累加到 `hist[idx]`。多个元素可映射到同一桶，存在**跨元素的数据相关冲突累加**（原子累加或分批 DMA 归并）。
2. **元素→桶映射是数据相关的**——映射由元素值本身决定，而非由元素的固定位置（坐标）决定。这是与 Reduction 的根本区别：Reduction 沿固定轴确定性线性归约、输出 shape 由非归约轴决定；Histogram 输出固定 `hist[nbins]`，桶间存在数据相关冲突，需 `asc_atomic_add` 类原子原语（或 `SetAtomicAdd` DMA 原子归并）收敛。
3. **可缺省 range 的等宽/自定义边界分桶**——bins 为 int（等宽）或 tensor（自定义边界）；range 可由 `value_range` 显式给定，也可缺省取输入 `min/max`（后者需前置 min/max 归约）。可选输出 `bin_edges`（双输出）。D≥2 时各轴独立取等宽/自定义边界，或为**离散类标轴**（bin index 即类标，无需 range/边界）。

典型场景：`tf.histogram_fixed_width`（显式 `value_range`，D=1）、`tf.histogram`/`HistogramV2`（int 或自定义边界、可选双输出，D=1）、`torch.histc`（等宽，D=1）、`numpy.histogram`（自定义边界，D=1）。仅元素级无冲突的逐元素算子走 Elementwise 范式；仅沿固定轴归约的算子走 Reduction 范式。

---

## 语言路由

按开发语言选择适配层。语言由调用方显式提供，或读取 `spec.yaml` 的 `op.language`；缺失或未知时先报告输入缺口，不选择语言适配层。

| 开发语言 | 加载内容 | 入口 |
|------|------|------|
| `ascendc` | 通用知识层 + Ascend C SIMT 适配层（混合 VF） | [adapters/ascendc/index.md](adapters/ascendc/index.md) |
| `dsl`（CANNBotDSL） | 通用知识层 + Histogram DSL 适配层 | [adapters/dsl/index.md](adapters/dsl/index.md) |
| 其他语言 | 通用知识层；适配层待确认 | 报告当前没有对应的语言适配层，不读取其他语言的 API 或模板 |

**互斥规则**：一次任务只加载命中语言的适配层，其他语言的适配层不加载、不参考其 API 与代码模板。

> 说明：本范式的 `ascendc` 适配层指 **Ascend C SIMT**，唯一形态为**混合 SIMT VF**（regbase 框架 + `__simt_vf__` + `asc_vf_call`）。
> 混合 VF 即 regbase 框架 + SIMT 内核，属本范式覆盖；纯 SIMD regbase（`__aicore__` + `TBuf/TQue` + `ReduceXxx`，无 SIMT 内核）不属本范式。

---

## 通用知识层索引

语言无关的算法与模型，按需展开（完整检索表见 [generic/index.md](generic/index.md)）：

| 文档 | 覆盖 |
|------|------|
| [generic/scene.md](generic/scene.md) | 场景模型、术语、SIMT 并行计数三模板骨架（D≥1，多维与 1D 同构）、能力边界 |
| [generic/capability-contract.md](generic/capability-contract.md) | 能力谓词定义（通用层算法引用的硬件/语言能力参数，含局部直方图 DMA 归并与展平 index 谓词） |
| [generic/bin-model.md](generic/bin-model.md) | bin 规格模型：int 等宽 / tensor 自定义边界、range 与 min/max 推导、bin_edges/nbins 推导、多维 bins 模型与行优先展平 |
| [generic/bin-index.md](generic/bin-index.md) | bin index 计算：等宽公式 / 二分 lower_bound、min==max 特判、边界钳制与越界丢弃、多维每轴独立 + 展平 |
| [generic/accumulation.md](generic/accumulation.md) | 累加策略：局部直方图模板（UB 分批 bins + DMA 原子归并，`bins/inputsize<GM_ATOMIC_ADD_FACTOR`）、无局部直方图模板（GM 直接原子累加，`bins/inputsize≥GM_ATOMIC_ADD_FACTOR`）、多维局部直方图布局 |
| [generic/thread-mapping.md](generic/thread-mapping.md) | 线程组织与 每核分块跨步、元素→线程映射、warp 划分 |
| [generic/sync-model.md](generic/sync-model.md) | 线程栅栏同步点、RAW/WAR 依赖 |

---

## 消费规则

1. **按需加载**：Ascend C 先读适配层 index；DSL 先读 [adapters/dsl/index.md](adapters/dsl/index.md)，再按需展开。仅展开当前问题需要的叶子文档。
2. **叶子文档限额**：单个子问题最多展开 5 个叶子文档，仍未闭环则拆分为更小的子问题。
3. **能力真值**：Ascend C 读取 [capability-mapping.md](adapters/ascendc/capability-mapping.md)；DSL 读取 [adapters/dsl/capability-mapping.md](adapters/dsl/capability-mapping.md)，其「待验证」谓词须从调用方提供的 DSL API 与设备证据确认，未知项保留待验证。
4. **编码前检视参考实现**：Ascend C 以 `examples/add_example/` 为混合 VF 工程框架权威参考，SIMT 内核（每核分块 + 局部直方图 + 原子累加）以 `simt_histogram_story` 样例内核代码为 SIMT 内核参考；用于核实 SIMT API 用法与骨架，不提供当前算子的算法或资源默认值。
5. **设计依据**：根据规格读取通用算法与对应语言约束，给出分支、数据结构、资源预算、同步和验证结论；输出格式由调用方指定。
6. **范式优先级**：本范式的专用知识优先于 agent 通用知识；同一设计点结论冲突时，以本范式指向的必读文件为准。
