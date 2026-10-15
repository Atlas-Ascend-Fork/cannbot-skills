# contraction 范式 IterBatch Kernel 模板

任务与布局参数由 [Tiling](tiling.md) 提供。

## 1 Kernel 概述

完整矩阵组驻留 L1，按保留轴坐标子组及 M/N/K 窗口消费。贯穿实例与物理份数见 [DAG](dag-buffers.md)。矩阵乘加始终分别读取当前输出坐标映射到的 A/B 子区，不改变公共指标映射。

## 2 Kernel 骨架代码

```text
initialize_slot_ownership()
tasks = batch_group_tasks_for_core()
pending = none
if tasks.has_next():
    pending = prefetch_full_group(tasks.next())
while pending is not none:
    task, l1_slot = pending.task, pending.slot
    acquire_l1_for_reads(l1_slot)
    pending = prefetch_full_group(tasks.next()) if tasks.has_next() else none
    for batch_l0 in batch_subgroups(task.valid, BatchL0):
        for n_tile in windows(N, NL0):
            for m_tile in windows(M, ML0):
                c_slot = acquire_next_l0c_slot_for_compute()
                for k_tile in windows(K, KL0):
                    ab_slot = acquire_next_l0_input_slot()
                    l12l0_batch_tiles(l1_slot, batch_l0, m_tile, n_tile, k_tile, ab_slot)
                    publish_l0(ab_slot)
                    acquire_l0_for_compute(ab_slot)
                    for batch_local in range(batch_l0.valid):
                        issue_mmad(ab_slot, c_slot, batch_local, init=(k_tile.offset == 0))
                    release_l0_after_all_batch_compute(ab_slot)
                publish_l0c_for_output(c_slot)
                acquire_l0c_for_output(c_slot)
                l0c2gm_batch_tiles(task, batch_l0, m_tile, n_tile, c_slot)
                release_l0c_after_all_batch_output(c_slot)
    release_l1_after_last_read(l1_slot)
drain_all_pending_operations()
```

`prefetch_full_group` 取得下一 L1 槽的可写权，提交完整矩阵组 gm2l1 并发布就绪，返回任务及槽编号；发布不要求主流水等待搬运完成。下一任务来自本核任务序列；当前组开始消费后，可向另一空闲 L1 槽提交下一组预取，不能按全局相邻任务猜测下一组 GM 地址。所有 release 均是相应流水消费完成后的交接。

各层槽计数在初始化时置0；每次取得新槽使用 `counter % slotCount` 并递增。计数跨任务继续，不因GM任务基址改变而重置；本例L1及L0输入各2槽，L0C为2槽。L1计数每装入一个任务组递增一次，L0输入计数每准备一个 K 段的整个 batch 子组递增一次，L0C计数每完成一个 batch 子组的 M/N 输出块递增一次。取得槽时等待旧消费者完成，发布后消费侧显式等待生产完成。

## 3 地址偏移计算

### 3.1 核间偏移计算

任务 batch 起点由 [Tiling §7](tiling.md#7-多核切分) 给出。实际 batch `b=batchTaskOffset+batchL0Offset+batchLocal`。GM 地址：`base + elemBytes*(b*batchStride + matrix_coordinates_dot_strides)`；转置只改变矩阵 stride，不改变 b。

### 3.2 核内循环间偏移计算

统一用 `offsetBytes(layout,coordinates)` 表示实际物理布局函数：

```text
AL1 = aL1Base + l1Slot*aL1SlotBytes
    + (batchL0Offset+batchLocal)*aL1BatchPitchBytes
    + offsetBytes(aL1Layout, mTileOffset+mLocal, kTileOffset+kLocal)
BL1 = bL1Base + l1Slot*bL1SlotBytes
    + (batchL0Offset+batchLocal)*bL1BatchPitchBytes
    + offsetBytes(bL1Layout, kTileOffset+kLocal, nTileOffset+nLocal)
AL0 = aL0Base + abSlot*aL0SlotBytes + batchLocal*aL0BatchPitchBytes
    + offsetBytes(aL0TileLayout, mLocal, kLocal)
BL0 = bL0Base + abSlot*bL0SlotBytes + batchLocal*bL0BatchPitchBytes
    + offsetBytes(bL0TileLayout, kLocal, nLocal)
CL0 = cL0Base + cSlot*cL0SlotBytes + batchLocal*cL0BatchPitchBytes
    + offsetBytes(cL0TileLayout, mLocal, nLocal)
```

L1 布局保留完整矩阵父 pitch；L0 tile 内偏移与 batch 间距是两个不同参数。L0C 在同一 batch 子组及 M/N 块的完整 K 中不换位置；切换 batch 子组或 M/N 块时选择下一个结果槽。L0C 输出读取必须使用相同 batch pitch，不能套紧凑连续 batch。

### 3.3 贯穿实例的地址追踪

batch组 `[2,3]` 中 `batchLocal=1`，取 `(m,n,k)=(33,34,65)`。GM A元素偏移 `3*64*128+33*128+65=28865`，B为 `3*128*64+65*64+34=28770`，C为 `3*64*64+33*64+34=14434`。L1 batch 子区偏移为 16 KiB，L0 三个空间的 batch 子区偏移也各为 16 KiB；矩阵 tile 内坐标为 `(33,34,1)`。

本例选择 A/B L1 为 NZ、L0A 为 NZ、L0B 为 ZN、L0C 为 NZ，分形内边长为16；坐标公式引用 [公共 NZ/ZN 约定](../../common-contract.md#21-nzzn-布局坐标约定)，接口表达见 [所选编程接口映射](../../../patterns.md#编程接口选择)。下表均为相对所选槽基址的字节偏移，已包含 batchLocal=1 的固定子区间距：

| 空间 | 父布局或 tile 内坐标 | 字节偏移 |
|---|---|---:|
| A L1 | 完整矩阵 `(33,65)` | 25634 |
| B L1 | 完整矩阵 `(65,34)` | 26660 |
| L0A | 当前 K 段 `(33,1)` | 17442 |
| L0B | 当前 K 段 `(1,34)` | 17474 |
| L0C | 输出 `(33,34)` | 26696 |

例如 A L1 的元素偏移为 `(65//16)*64*16+33*16+65%16=4625`，故字节偏移为 `16384+4625*2=25634`。L0A 的 K 局部坐标回到1，batch pitch仍为16384字节。核1首次处理该任务，L1 与 L0C 使用槽0；k=65 位于第二 K 段，L0A/B 使用槽1。表中偏移需再叠加相应槽基址：本例 A L1 为0、B L1 为65536、L0A/B 为32768、L0C 为0字节。

## 4 CopyIn MTE2 逻辑

装入当前组的完整 A/B 矩阵；按每个真实输入batch stride读取，尾组只搬有效batch。首次物化矩阵padding负责零填充，不能越读下一个batch来凑齐。

## 5 CopyIn MTE1 逻辑

源为完整矩阵L1子窗口，目的为各batch独立L0子区。每次可以批量发射搬运，但槽所有权要覆盖所有batch。最后一次M/N/K消费者完成后才归还整个L1槽。

## 6 Compute 计算逻辑

逐batch调用MMAD；每batch同一M/N块的首K段初始化，后续累加。MMAD顺序为K段外、batch内，不跨batch共享累加值。VF不涉及；类型与舍入遵守 [公共数值契约](../../common-contract.md#1-语义与数值)。

## 7 CopyOut MTE3 逻辑

不涉及。

## 8 CopyOut 输出通路 逻辑

逐batch读取当前L0C子区，写入各自GM有效M/N窗口；可使用等价批量输出表达，但源batch pitch和目的batch stride独立。本例每batch源pitch为16KiB（FP32），GM目的stride为8KiB（BF16）；二者均覆盖64×64元素，但字节跨度不同。

## 9 流水同步

L0C acquire 在当前batch子组/M/N块完整K循环外；全部batch末K计算完成后交给输出。L0A/B每槽直到本轮最后一个batch MMAD完成才可复用。双L0C槽允许一个输出、另一个计算；同槽仍等最后一个batch输出完成。L0C 采用 [双槽整块交接](../../execution-strategies.md#43-双槽整块交接)，交接单位为当前 batch 子组的全部结果子区。L1 完整矩阵组保持驻留，输入槽计数按组推进。具体锁/事件绑定见 [所选编程接口映射](../../../patterns.md#编程接口选择)，不额外添加稳态全流水屏障。

## 10 Kernel 产出校验

验收接口遵循 [公共契约 §5](../../common-contract.md#5-产出与验收)，本节给出本模板的覆盖与数值检查。

用 `A[b,m,k]=b+1`、`B[b,k,n]=b+2`，期望 `C[b,m,n]=128*(b+1)*(b+2)`；另外使用非对称m/n/k数据验证布局。跟踪每槽及每batch子区地址并放置间隙哨兵，重点发现把16KiB完整矩阵pitch误换成tile跨度的错误。对照独立逐元素Golden，检查5个batch共10次MMAD、5个输出tile及尾组不读写batch5；扩展测试覆盖输入转置/类型、非整 M/N/K 和 BatchL1/BatchL0 尾组。扩展候选先按 Tiling 检查布局与容量。

### 10.1 跨任务槽轮转

两核例中，核0先处理 batch `[0,1]`，再处理 `[4]`。其 L1 槽依次为0、1；第一任务 K 段使用 L0 槽0、1，第二任务仍为0、1；L0C 槽依次为0、1。核1只处理 `[2,3]`，从自己的槽0开始。尾组仅一个有效 batch，固定槽跨度保持不变。

### 10.2 批次子区负例

令 BatchL1=3、BatchL0=2，首任务包含子组 `[0,1]` 和 `[2]`，后者的 L1 batch 起点为2，L0 batch 起点重新为0。分别注入提前释放输入槽、将 batchL0Offset 叠到 L0 子区地址、尾组仍发射两个 batch、用紧凑 K tile 跨度替换完整矩阵间距，要求覆盖或地址检查能识别错误。此候选需重新核算 L1 容量。

### 10.3 无任务核

任务数小于核数时，空任务序列保持 pending 为空，跳过预取和计算循环；收尾只回收实际已初始化的资源，不等待从未发出的生产事件。检查零任务、单任务及多任务，记录预取次数等于本核任务数。

## 11 Elementwise/Broadcast Bias[N] 组合扩展的地址与计算

组合语义为 `Y[...,m,n]=convert_out(acc[...,m,n]+bias[n])`。GM 元素偏移为 `nTileOffset+nLocal`，按 Bias[N] 的真实 `strideN` 换算字节地址；所有保留轴坐标共用该输入向量，不增加坐标或 batch 偏移。

使用[公共 Bias[N] 组合扩展首段融合骨架](../../execution-strategies.md#8-biasn-组合扩展的首段融合策略)：每个输出坐标的首 K 段以对应 `bias[n]` 启动累加，后续 K 段只累加乘积；无组合扩展时保留原零初始化骨架。IterBatch 在 `batch_local` 执行循环中依序应用同一 N 子区，BiasTable 只能在该子组最后一个消费者完成后复用。L0C 发布与输出协议保持所选完整路线。

取 `bias[n]=n+1`，Golden 为每个保留轴坐标的收缩结果加同一个 `bias[n]`，再按 Tout 转换。检查 N 偏移、每坐标恰好应用一次、K 分窗不重复应用、无组合扩展不读 bias，以及 K=0 时按公共预处理产生 `convert_out(bias[n])`。MergeBatch 的扩大 N 映射另按其模板检查。
