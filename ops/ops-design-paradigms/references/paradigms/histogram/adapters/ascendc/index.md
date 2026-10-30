<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 范式 Ascend C SIMT 适配层（混合 VF）

> **层级**：Ascend C 语言适配层（adapters/ascendc/），覆盖 **Ascend C SIMT** 唯一形态：**混合 SIMT VF**。
> 语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。

## 1 定位

本目录承载 Histogram 范式中**强耦合 Ascend C SIMT** 的知识：SIMT API 真值、kernel 骨架与原子累加落地、访存与调优，覆盖唯一形态：

- **混合 SIMT VF**：regbase 框架（`op_host` TilingFunc + `__global__ __aicore__` 入口）+ SIMT VF（`__simt_vf__` + `asc_vf_call`），参考 `examples/add_example/`（工程框架）与 `simt_histogram_story`（SIMT 内核参考）。

> **模型边界**：**不覆盖**纯 SIMD regbase（`TBuf/TQue`/`ReduceXxx` 全流水线，无 SIMT 内核）。混合 VF 形态**可用** `kernel_operator.h`（regbase 框架编译环境），计数段走 SIMT VF 不走向量化流水线；**局部直方图模板的归并段**用 SIMD `DataCopyPad` + `SetAtomicAdd`/`SetAtomicNone` DMA 原子归并（见 [kernel-skeleton.md](kernel-skeleton.md) §2.2），属本范式混合形态的一部分，不构成纯 SIMD regbase。

## 2 能力映射

本语言对通用层能力谓词的真值与 API 实体见 [capability-mapping.md](capability-mapping.md)。通用层算法引用的谓词（`` `LOCAL_HIST_ATOMIC` ``、`` `LOCAL_HIST_FLUSH` ``、`` `GLOBAL_MERGE_ATOMIC` ``、`` `THREAD_BARRIER` ``、`` `WARP_SIZE` ``、`` `VECTOR_LOAD_WIDTH` `` 等）一律以该文件填值为准。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| API 真值 | [capability-mapping.md](capability-mapping.md) | SIMT 内核 API 实体（§1~§3：`asc_atomic_add`/`asc_syncthreads`/`__ubuf__`/`float4`/`__launch_bounds__`）+ SIMD DMA 归并（§2.8：`SetAtomicAdd`/`DataCopyPad`/`SetAtomicNone`）+ regbase 框架 API（§4：`__simt_vf__`/`asc_vf_call`/`ASCENDC_TPL_*`/`REGISTER_TILING_DEFAULT`/`SetBlockDim` 等） |
| Kernel 落地 | [kernel-skeleton.md](kernel-skeleton.md) | 三模板骨架（局部直方图 §2：SIMT VF 计数 + SIMD DMA 归并；无局部直方图 §8：GM 直接原子；确定性 §9：bins 并行 + input 串行）、bin index 分支、二维（多维）展平落地、regbase 工程框架（§7） |

## 4 公共知识依赖

| 主题 | 位置 |
|------|------|
| Cast 规则 | [../../../common/cast-rules.md](../../../common/cast-rules.md) |
| 同步与一致性 | [../../../common/sync-and-consistency.md](../../../common/sync-and-consistency.md) |

> regbase 专属公共知识的使用：本范式引用 regbase **工程框架**（OpDef/TilingFunc/`asc_vf_call`，见 [capability-mapping.md](capability-mapping.md) §4）。计数段用 SIMT VF 的 GM 直接访问 + 原子累加；局部直方图模板的归并段用 SIMD `SetAtomicAdd`/`DataCopyPad`/`SetAtomicNone` DMA 原子归并（见 [capability-mapping.md](capability-mapping.md) §2.8），不引入 `ReduceXxx` 全流水线。`vf-programming-rules.md` 的 `__simd_vf__`（SIMD Reg API）与本范式 `__simt_vf__`（SIMT）不同，仅启动机制（`asc_vf_call`）可对照。

## 5 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载其他语言适配层。
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题。
3. 按 [capability-mapping.md](capability-mapping.md) 核实 API 存在性与签名，按 [kernel-skeleton.md](kernel-skeleton.md) 落地。
4. 编码前必须检视参考样例——工程框架检视 `examples/add_example/`，SIMT 内核写法检视 `simt_histogram_story` 内核代码，核实 SIMT API 用法，不提供当前算子的算法或资源默认值。
