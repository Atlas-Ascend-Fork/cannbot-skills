<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 范式通用知识层索引

> **层级**：通用知识层（generic/），语言中立。与语言适配层（`../adapters/<lang>/`）组合消费：先经 [../patterns.md](../patterns.md) 的语言路由确定适配层，再按下表按需展开。

## 检索表

| 文档 | purpose | read_when | keywords |
|------|---------|-----------|----------|
| [scene.md](scene.md) | 场景模型：准入条件、术语、SIMT 并行计数三模板骨架、能力边界 | 判断算子是否属于本范式；建立全局心智模型 | 直方图、分桶、bin、等宽、自定义边界、SIMT、每核分块、并行计数、双输出、加权、density、多维、二维、展平 index、每轴 bin index、离散类标 |
| [capability-contract.md](capability-contract.md) | 能力谓词定义：通用层算法引用的硬件/语言能力参数 | 编写新语言适配层；理解算法中的能力依赖 | 谓词、能力、局部原子累加、全局原子归并、线程栅栏、warp、向量化宽、片上容量、对齐粒度、展平 index |
| [bin-model.md](bin-model.md) | bin 规格模型：int 等宽 / tensor 自定义边界、range 与 min/max 推导、bin_edges/nbins 推导、多维 bins 模型与行优先展平 | 设计 bin_edges 与 nbins 的来源和合法性；range 缺省判定；多维总桶数与展平布局 | bin_edges、nbins、linspace、min/max 前置归约、等宽、自定义边界、value_range、range、多维 bins、总桶数、行优先展平 |
| [bin-index.md](bin-index.md) | bin index 计算：等宽公式 / 二分 lower_bound、min==max 特判、边界钳制、越界丢弃、多维每轴独立 + 展平 | 设计元素→桶映射与越界处理；多维展平 index | bin index、floor、invWidth、lower_bound、二分定位、min==max、钳制、越界丢弃、展平 index、行优先、离散类标 |
| [accumulation.md](accumulation.md) | 累加策略：局部直方图模板（UB 分批 bins + DMA 原子归并，`bins/inputsize<GM_ATOMIC_ADD_FACTOR`）、无局部直方图模板（GM 直接原子累加，`bins/inputsize≥GM_ATOMIC_ADD_FACTOR`）、多维局部直方图布局 | 设计 hist 累加与跨核归并路径；按边界条件选模板；多维展平累加 | 局部直方图、分批 bins、DMA 原子归并、无局部直方图、GM 直接原子、边界条件、weight、density、多维局部直方图、展平、BATCH_BINS |
| [thread-mapping.md](thread-mapping.md) | 线程组织与 每核分块跨步：元素→线程映射、warp 划分、遍历步长 | 设计线程排布、元素分配与并行度 | 线程切分、每核分块、线程号、block 号、warp 划分、并行度、负载均衡 |
| [sync-model.md](sync-model.md) | 线程栅栏同步点、RAW/WAR 依赖、内存可见性 | 同步设计；从持有法则推导同步点 | 线程栅栏、同步、RAW、WAR、累加前哨、归并前 |

## 使用规则

1. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分为更小的子问题。
2. 通用层结论中的能力依赖一律引用 [capability-contract.md](capability-contract.md) 的谓词；具体真值以命中语言适配层的 `capability-mapping.md` 为准。
3. 通用层不含任何具体语言 API；能力依赖一律以 [capability-contract.md](capability-contract.md) 的谓词名引用，具体真值按各自语言适配层映射。
4. 具体语言的实现（代码骨架、API 真值、调优方法）在适配层，不在本文档族内。
