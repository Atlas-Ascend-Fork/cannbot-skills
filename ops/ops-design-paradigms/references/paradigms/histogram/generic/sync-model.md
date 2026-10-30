<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 同步模型

> 语言中立。定义两模板并行计数骨架中的线程栅栏同步点、RAW/WAR 依赖与内存可见性。同步原语真值见对应适配层 `` `THREAD_BARRIER` `` / `` `MEMORY_FENCE` `` / `` `VECTOR_MTE3_SYNC` ``（Ascend C 见 [../adapters/ascendc/capability-mapping.md](../adapters/ascendc/capability-mapping.md)）。

## 1 依赖分析

### 1.1 局部直方图模板（每批内部）

每批（[accumulation.md](accumulation.md) §2 步骤 2-4）内部的读写依赖：

| 依赖段 | 写（生产者） | 读（消费者） | 依赖类型 |
|--------|-------------|-------------|----------|
| 清 0 → 计数 | `` `LOCAL_HIST_CLEAR` `` 写本批局部直方图初值 0 | 计数（SIMT 内核）原子累加 | WAR（先清零后累加） |
| 计数 → 归并 | 各线程原子累加到 UB 局部直方图 | `` `LOCAL_HIST_FLUSH` `` DMA 归并读 UB 本批 bins | RAW（先累加后归并） |

- **清 0 → 计数**：清 0 用 `` `LOCAL_HIST_CLEAR` ``（批量清零），在计数启动前顺序完成，**无需 `` `THREAD_BARRIER` ``**（真值见适配层：AscendC 为 SIMD 批量填零，DSL 为 SIMT 线程跨步清 0 + `` `THREAD_BARRIER` ``）。
- **计数 → 归并**：跨引擎 RAW——计数段在 SIMT 内核（向量单元）写 UB，归并段在 DMA 单元读 UB；`` `THREAD_BARRIER` `` 只同步 block 内线程、不足以让 DMA 读到向量单元的 UB 写，需 **计数→归并跨引擎同步**（`` `VECTOR_MTE3_SYNC` ``，真值见适配层：AscendC 由队列交接携带、DSL 由 SIMT 内核边界 + 编译器自动同步，见 §2/§3）。

### 1.2 无局部直方图模板

无片上局部直方图，无阶段间依赖；跨核同 bin 收敛由 `` `GLOBAL_MERGE_ATOMIC` `` 原子语义保证（非栅栏），输出清 0 可见性见 §3。

## 2 同步点

局部直方图模板**每批**的同步点（清 0 无需 `` `THREAD_BARRIER` ``；计数→归并一个跨引擎 `` `VECTOR_MTE3_SYNC` ``）：

1. **清 0 → 计数**：`` `LOCAL_HIST_CLEAR` `` 在计数启动前顺序完成，无显式同步点（DSL 侧若清 0 用 SIMT 线程跨步，则还须 `` `THREAD_BARRIER` ``）。
2. **计数后、归并前**（`` `VECTOR_MTE3_SYNC` ``）：保证向量单元（SIMT 内核）写 UB 完成后，DMA 单元才读 UB 做归并（真值见适配层，见 §3）。

```
（每批） LOCAL_HIST_CLEAR 清0 → 计数（LOCAL_HIST_ATOMIC） → VECTOR_MTE3_SYNC → 归并（LOCAL_HIST_FLUSH）
```

无局部直方图模板无 `` `THREAD_BARRIER` ``：单遍 每核分块 + 核内跨步计数 + GM 原子，无片上阶段切换，也无 VECTOR→MTE3 归并段。

## 3 内存可见性

- UB 局部直方图的读写都在核内，`` `THREAD_BARRIER` `` 已保证核内可见性。
- GM 输出 `hist` 的跨核收敛依赖 `` `LOCAL_HIST_FLUSH` `` / `` `GLOBAL_MERGE_ATOMIC` `` 的原子语义，而非栅栏；输出 `hist` 由 host 在启动前清零。
- **计数 → 归并的跨引擎可见性**：计数段（SIMT 内核，向量单元）写 UB 后、归并段（DMA 单元）读 UB 前，须 `` `VECTOR_MTE3_SYNC` `` 保证向量单元写 UB 对 DMA 可见。真值由适配层给出——AscendC 由局部直方图 UB 队列交接携带，DSL 由 SIMT 内核边界 + 编译器自动同步（均无需显式跨引擎同步原语，见对应适配层 capability-mapping §2/§2.9）。`` `THREAD_BARRIER` ``/`` `MEMORY_FENCE` `` 是 SIMT 线程/内存栅栏，**不能替代跨引擎同步**。

## 4 与纯 SIMD regbase 流水线同步的区别

| 维度 | 本范式（混合 SIMT + 局部直方图 DMA 归并） | 纯 SIMD regbase（自定义算子） |
|------|--------------------------------------|--------------------------|
| 同步原语 | 计数段 `` `THREAD_BARRIER` ``；计数→归并跨引擎 `` `VECTOR_MTE3_SYNC` ``；归并段 DMA 原子原语（`` `LOCAL_HIST_FLUSH` ``） | 硬件事件（标志位/流水栅栏，具体符号见适配层 capability-mapping） |
| 同步粒度 | 计数段 block 内线程；归并段 MTE/DMA 单元 | 流水线（数据搬运/标量/向量单元）之间 |
| 数据一致性 | UB 静态数组 + 原子 + 栅栏；归并经 DMA 原子 | 流水线队列入队出队 + 互斥锁 |

本范式的混合形态仅在**归并段**使用 DMA 原子归并，计数段仍是 SIMT；不引入纯 SIMD regbase 的全流水线队列，二者不可混用。
