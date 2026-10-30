<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 线程组织与每核分块跨步

> 语言中立。定义 SIMT 线程排布、元素→线程映射、warp 划分与并行度。真值（`` `WARP_SIZE` ``、`` `THREAD_COUNT_PER_BLOCK` ``、`` `GRID_DIM` ``、片上容量）见对应适配层 capability-mapping。
> 排布形式为**每核（block）连续分块 + 核内（block 内线程）跨步**，而非全局交错。

## 1 线程模型

SIMT 内核由 host 侧启动，block 数（grid）由 tiling 决定（= 参与算子的 core 数 `` `GRID_DIM` ``），每 block 线程数为编译期常量 `` `THREAD_COUNT_PER_BLOCK` ``。内核内：

```
block    // block 号 = 物理 core 号 ∈ [0, GRID_DIM)（host 侧传核数，真值见适配层）
thread   // block 内线程号 ∈ [0, THREAD_COUNT_PER_BLOCK)
```

- **block 号即物理 core 号**：block 号对应物理 core 编号（区别于逻辑 block），无需核状态初始化。
- **早退**：grid 可能大于某阶段实际需要的 core 数（输入统计只需 `needXCoreNum` 个 core、输出清零只需 `clearYCoreNum` 个 core），内核开头以 `if (block >= needXCoreNum) return;` 让多余 core 早退，只让前 `needXCoreNum` 个 core 参与。

## 2 每核分块 + 核内跨步循环

输入 `N` 个元素由 host 侧 tiling 预切分为**每核一块的连续分块**（非全局交错），每个 core 处理自己的连续块：

```
formerLength   // 每核块长度（host tiling 预计算）
tailLength     // 末核尾块长度 = N - (needXCoreNum - 1) × formerLength
base           = block × formerLength                     // 本核连续块起始下标
length         = (block == needXCoreNum - 1) ? tailLength : formerLength

for (int64_t thread = threadIdx.x; thread < length; thread += THREAD_COUNT_PER_BLOCK) {
    i = base + thread;                                    // 核内 THREAD_COUNT_PER_BLOCK 个线程从 threadIdx.x 起始跨步访问本核块
    ...
}
```

- **每核连续块**：各 core 的块连续且互不重叠；`base` 为块起始，块内下标连续，末核尾部不足 `formerLength` 的部分取 `tailLength`。
- **核内跨步**：`` `THREAD_COUNT_PER_BLOCK` `` 个线程以步长 `` `THREAD_COUNT_PER_BLOCK` `` 遍历本核块，每线程处理 `ceil(length / THREAD_COUNT_PER_BLOCK)` 个元素，线程间负载差异 ≤ 1。
- **局部性**：同一 core 的线程访问本核连续块，核内访存连续、配合片上缓存提升局部性（比全局交错更强）。
- 这是 SIMT 并行计数的核心教学点：把串行 `for (i=0; i<N; i++)` 改写为「每核分块 + 核内跨步」即可并行化，且不影响后续优化叠加。

## 3 lane 划分与局部直方图清零

按 `` `WARP_SIZE` `` 得到 lane 号，用于可选 warp 级原语：

```
laneId = thread % WARP_SIZE                 // lane 号（可选 warp 级原语用）
```

- 局部直方图**每核一份**（非每 warp 一份），核内多线程写同 bin 靠 `` `LOCAL_HIST_ATOMIC` `` 收敛（见 [accumulation.md](accumulation.md) §2）。
- **局部直方图每批清零**：用 `` `LOCAL_HIST_CLEAR` `` 一次清本批 bins（真值见适配层：AscendC 用 SIMD 批量填零，DSL 用 SIMT 线程跨步清 0）；是否需 `` `THREAD_BARRIER` `` 见 [sync-model.md](sync-model.md)。

## 4 并行度与配置选择

| 量 | 语义 | 说明 |
|----|------|------|
| `` `THREAD_COUNT_PER_BLOCK` `` | 每 block 线程数 | 编译期常量；受每线程寄存器预算约束（过大导致寄存器溢出） |
| `` `GRID_DIM` `` | grid 维度（= 参与算子的 core 数） | 运行时决定；真值由 adapter 的能力映射给出 |
| `` `VECTOR_LOAD_WIDTH` `` | 向量化宽度 | 每线程一次读多元素时，核内跨步改为 `` `THREAD_COUNT_PER_BLOCK` `` × WIDTH |

**调优点**：每核块长 `formerLength` 决定每 core 计算负载；`needXCoreNum`（= `` `GRID_DIM` ``）越大，每核块越小、每线程负载越低（收益），但 block 调度开销随 grid 增大（代价）。存在收益拐点，需现场扫描 `` `GRID_DIM` `` 找拐点。

## 5 向量化的线程映射

按 `` `VECTOR_LOAD_WIDTH` ``（记为 `WIDTH`）向量化读入时，同一线程一次读 `WIDTH` 个连续元素，核内跨步放大 `WIDTH` 倍：

```
vecEnd = (length / WIDTH) * WIDTH
for (int64_t thread = threadIdx.x; thread * WIDTH < vecEnd; thread += THREAD_COUNT_PER_BLOCK) {   // 主循环：每线程读 WIDTH 个连续元素
    向量 v = 向量加载(x + base + thread * WIDTH);
    // 对 v 的每个分量各自计算 bin index 并原子累加
}
for (int64_t i = vecEnd + threadIdx.x; i < length; i += THREAD_COUNT_PER_BLOCK) {                 // 收尾：末尾不足 WIDTH 的元素标量处理
    ...
}
```

向量化只改访存，不改每核分块与 bin 语义。
