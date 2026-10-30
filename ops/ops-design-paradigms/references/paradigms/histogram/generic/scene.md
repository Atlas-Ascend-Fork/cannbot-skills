<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 场景模型

> 本文件给出 Histogram 范式的准入条件、术语、SIMT 并行计数三模板骨架与能力边界。语言无关；具体 API 真值见 [capability-contract.md](capability-contract.md) 与对应语言适配层的 `capability-mapping.md`。

## 1 准入条件

算子同时满足以下三点即属于本范式（详见 [../patterns.md](../patterns.md) §场景判定）。维度 `D` 为每样本特征轴数（`D ≥ 1`），D=1 即经典一维直方图，D≥2 为多维直方图：

1. 输出为固定大小的**分桶计数** `hist`：D 维总桶数 `` `∏ nbins_axis` ``，展平为 `hist[N_BINS]`（`N_BINS = ∏ nbins_axis`）；D=1 时退化为 `hist[nbins]`；
2. 元素（样本）→桶映射**由元素值决定**（数据相关冲突累加）：每个样本的 D 个特征各自映射到一维 bin index，组合成 D 维 bin 下标再累加；
3. 每轴分桶形态独立取**等宽**（int bins）或**自定义边界**（tensor bins），range 可显式给定或由输入 min/max 缺省推导；各轴亦可以是**离散类标轴**（bin index 即类标本身）。

## 2 术语

| 术语 | 含义 |
|------|------|
| **nbins** | 桶数，输出 `hist[nbins]` 的长度 |
| **EqualWidth（等宽）** | bins 为 int，`bin_edges = linspace(lo, hi, nbins+1)`；`lo/hi` 来自 `value_range` 或缺省 `min/max` |
| **CustomEdges（自定义边界）** | bins 为 tensor 递增边界，直接作为 `bin_edges`，`nbins = bins.size - 1` |
| **bin index** | 元素映射到 `[0, nbins)` 的桶下标；越界元素丢弃，`max` 边界值钳制到 `nbins-1` |
| **range 来源** | `value_range`（显式标量 min/max）或缺省 `[min(x), max(x)]`（需前置 min/max 归约） |
| **局部直方图模板** | 每核一份 UB 分批 bins 局部直方图：核内 SIMT 统计本核 input → DMA 原子归并到 GM；|
| **每核分块 + 核内跨步** | 输入按 core 切为连续块（`base = block×formerLength`），核内 `` `THREAD_COUNT_PER_BLOCK` `` 个线程以步长 `` `THREAD_COUNT_PER_BLOCK` `` 跨步遍历本核块，实现元素→线程均衡分配 |
| **无局部直方图模板** | 不做片上局部直方图，每元素直接 GM 原子累加到输出；|
| **边界条件** | `bins/inputsize < GM_ATOMIC_ADD_FACTOR`：bins 在 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍以内（atomicadd 冲突较大）用局部直方图模板，否则（冲突较小）用无局部直方图模板 |
| **确定性模板** | 第三模板：bins 并行 + input 串行 + 无原子累加；bins 分核后核内线程按 stride 遍历本核所有 bins（每 bin 恰被一个线程命中）、每线程遍历全部 input、每 bin 仅单线程累加，确保浮点累加顺序确定；由确定性开关选择，不参与 `bins/inputsize` 边界条件分派 |
| **weight / density** | 加权求和替代计数；density 完成后归一化（后处理） |
| **每样本特征轴 D** | 每样本有 D 个特征，D=1 为一维直方图，D≥2 为多维直方图；D 为规格参数（与 nbins 同级） |
| **每轴 bin index** | 样本第 d 个特征独立映射到 `[0, nbins_axis[d])` 的一维桶下标 `i_d`；各轴规则独立 |
| **多维 bin** | 由 D 个每轴 bin index 组成的 D 维下标 `(i0, i1, …, i_{D-1})`，对应直方图的一个 D 维桶 |
| **展平 index** | D 维 bin 下标按行优先展平为线性 index（见 [bin-index.md](bin-index.md) §多维）；输出 `hist` 按展平顺序存储 |
| **离散类标轴** | 特征取值为离散整数类标 `[0, nbins_axis)`，bin 边界天然是整数格点，bin index 即类标本身（无需 floor/二分） |
| **D_T / HistT** | 输入元素 dtype（float 族）；输出 hist dtype（int32 计数，或 float 权重） |

## 3 SIMT 并行计数骨架（三模板）

无论具体算子形态（FixedWidth / V2 / 多维），SIMT 实现收敛到三个模板——**局部直方图模板**（UB 分批 bins 计数 + DMA 原子归并）、**无局部直方图模板**（GM 直接原子累加）与**确定性模板**（浮点累加需确定时选用）。三模板的完整计算步骤（局部直方图五步 1–5、无局部直方图三步 1–3、确定性 1–4）与模板划分边界条件 `bins/inputsize < GM_ATOMIC_ADD_FACTOR` 见 [accumulation.md](accumulation.md) §1/§2/§3/§4；Ascend C 落地骨架见 [kernel 骨架](../adapters/ascendc/kernel-skeleton.md)。

**两模板 SIMT 统计内核同构**：每核分块 + 核内跨步 + bin index + 计数完全一致，差异仅在「累加去向」——局部直方图模板累加到 UB 分批 bins 再 `` `LOCAL_HIST_FLUSH` `` 归并；无局部直方图模板直接 GM 原子（`` `GLOBAL_MERGE_ATOMIC` ``）。两模板均要求输出 `hist` 预清零（host 启动前清零，或 kernel 内 每核分块清零，见 [accumulation.md](accumulation.md) §2.2/§3），否则原子累加/归并结果错误。

**多维与 1D 同构**：D≥2 时骨架结构与 D=1 完全一致，仅三处增量——（1）bin index 由 1D 变为「每轴独立计算 + 行优先展平」；（2）输入由单特征变为每样本 D 个特征（D 个 GM 数组或 N×D 数组）；（3）局部直方图大小由 `nbins` 变为 `` `∏ nbins_axis` ``。因此后续章节对 1D 的所有结论（同步点、分批加载、原子原语、驻留判定）可直接以「展平 index 替换 1D idx、`∏ nbins_axis` 替换 nbins」的方式推广。

三个维度的同步点、分批加载、原子原语选择分别由 [sync-model.md](sync-model.md)、[thread-mapping.md](thread-mapping.md)、[accumulation.md](accumulation.md) 展开。

## 4 能力边界

- **模板由边界条件决定**：`bins/inputsize < GM_ATOMIC_ADD_FACTOR`（bins 在 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍以内，atomicadd 冲突较大）用局部直方图模板，否则（冲突较小）用无局部直方图模板。局部直方图模板单批 bins 数量（D 维为 `` `∏ nbins_axis` ``，1D 为 `nbins`）受片上容量预算 `` `ONCHIP_HIST_CAPACITY` `` 约束，超限时按批次切分（每批清 0 → 核内统计 → DMA 归并，见 [accumulation.md](accumulation.md) §2 与 [bin-model.md](bin-model.md) §4 分批），不再以「nbins 整体是否驻片上」为硬门槛。多维直方图的总桶数是各轴 `nbins_axis` 的乘积，`∏ nbins_axis` 增长快于 1D，单批容量判定需先算总桶数。
- **计数与归并的语言形态**：两模板的**计数段**均为 SIMT（GM 直接访问，不依赖 MTE 数据搬运）；局部直方图模板的**归并段**为 DMA 原子归并（`` `LOCAL_HIST_FLUSH` ``，具体符号见适配层 capability-mapping）。
- **数据规模下限**：数据规模极小时，线程调度与线程间协调开销可能超过计算收益，应回退串行或低并行度。
- **仅 float 族输入**：参考样例只覆盖 float32；int32/int8/fp16 等 dtype 属未验证分支。
- **需要更精细的多级流水**：需要 MTE2（输入搬运）/Scalar/Vector 三级全重叠，或更复杂的多阶段软件流水时，应改用目标语言的自定义算子核函数或核函数/向量混合编程模型（具体符号见适配层 capability-mapping），本范式不覆盖；本范式只覆盖局部直方图片上队列 depth=2 双缓冲带来的向量单元（计数）与 DMA 单元（归并）重叠（见 [accumulation.md](accumulation.md) §2.1 与适配层 kernel-skeleton）。
