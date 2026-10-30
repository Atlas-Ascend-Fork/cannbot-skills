<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 能力谓词契约

> 本文件定义通用层算法引用的**能力谓词**（语言/硬件能力参数）。通用层只引用谓词名，不写具体真值；
> 真值一律由命中语言适配层的 `capability-mapping.md` 填值（AscendC 见 [../adapters/ascendc/capability-mapping.md](../adapters/ascendc/capability-mapping.md)）。
> 编写新语言适配层时，必须为本表每个谓词给出该语言的真值；无法确定者标「待验证」并承担验证义务。

## 谓词语义

| 谓词 | 语义 | 引用位置 |
|------|------|----------|
| `WARP_SIZE` | warp 内线程数（lane 数），用于 QB 排布与可选 warp 级原语 | thread-mapping |
| `THREAD_COUNT_PER_BLOCK` | 每个 block 的线程数（编译期常量），决定每 block 的 warp 数 | thread-mapping |
| `GRID_DIM` | grid 维度（= 参与算子的 core 数，运行时），决定参与输入的 core 数 | thread-mapping |
| `LOCAL_HIST_ATOMIC` | 片上（UB）局部直方图上的原子累加能力，处理核内多线程同 bin 并发写（局部直方图模板计数步） | accumulation |
| `LOCAL_HIST_CLEAR` | 片上局部直方图批量清零能力（每批 bins 清零，供分批复用） | accumulation、sync-model |
| `LOCAL_HIST_FLUSH` | 把片上 UB 分批 bins 局部直方图原子归并（累加）到 GM 输出的 DMA 归并能力，处理跨核同 bin 收敛（局部直方图模板归并步） | accumulation |
| `GLOBAL_MERGE_ATOMIC` | GM 输出上的逐元素原子累加能力，处理跨核同 bin 并发写（无局部直方图模板） | accumulation |
| `THREAD_BARRIER` | block 内线程栅栏，使局部直方图累加对所有线程可见 | sync-model |
| `MEMORY_FENCE` | 内存栅栏，保证片上写入对后续 DMA 归并/原子操作可见（可选） | sync-model |
| `VECTOR_MTE3_SYNC` | 计数段写 UB 完成 → 归并段读 UB 前的跨引擎同步能力，保证计数写 UB 对归并可见（真值由适配层给出：AscendC 由队列交接携带，DSL 由 SIMT 内核边界 + 编译器自动同步） | sync-model |
| `VECTOR_LOAD_WIDTH` | 单次向量化加载的元素个数，降低访存指令数（真值由适配层给出） | thread-mapping、bin-index |
| `ONCHIP_HIST_CAPACITY` | 片上局部直方图可用字节预算，约束单批加载 bins 数量上限（分批加载） | bin-model |
| `ALIGN_GRANULARITY` | 片上数据搬运/存储的对齐粒度（单位：字节）；局部直方图行宽 padding 按该粒度对齐，填充元素数 = 粒度 / `sizeof(HistT)`，避免非对齐访问冲突与越界 | bin-model、accumulation |
| `PREFIX_MINMAX_REDUCE` | range 缺省时对输入做前置 min/max 归约的能力 | bin-model |
| `FLAT_INDEX_CALC` | 将 D 维 bin 下标 `(i0, …, i_{D-1})` 按行优先展平为线性 index 的能力；D=1 时恒等 | bin-index、accumulation |

## 使用约束

1. 通用层算法文本中的能力依赖一律以 `code` 形式引用本表谓词名（如 ``需 `LOCAL_HIST_ATOMIC` ``），不内联语言 API。
2. 真值只出现在 `capability-mapping.md`，不在通用层重复。
3. 谓词真值随语言/平台变化；同一范式不同语言适配层可给出不同真值，但语义必须一致。
   实现时**只取命中语言适配层的真值，禁止跨层取值**——例：`` `THREAD_COUNT_PER_BLOCK` `` 在 Ascend C 侧为 512（默认，按性能测试结果调整）、DSL 侧为 256（默认，按性能测试结果调整），不得把 DSL 的 256 写进 Ascend C 代码。
4. 命中语言没有定义某谓词（能力缺失）时，该能力不可用于设计，需回退替代方案或在设计中标注能力缺口。
