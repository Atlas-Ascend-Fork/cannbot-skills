# contraction 范式 MergeBatch Kernel 模板

公共语义、地址单位、资源描述、所有权与验收接口统一引用 [公共契约](../../common-contract.md)。

## 1 Kernel 概述

A沿M拼接、B沿N拼接，计算D再输出对角块；定义和贯穿实例见 [DAG](dag-buffers.md)。一个合并任务拥有完整K归约，D中包括非对角乘积。

## 2 Kernel 骨架代码

```text
initialize_slot_ownership()
for task in merged_batch_tasks_for_core():
    geometry = bind_merged_geometry(task.valid_batch_count)
    c_slot = acquire_next_l0c_slot_for_compute()
    for k_l1 in windows(K, KL1):
        l1_slot = acquire_next_l1_slot()
        gm2l1_merged_inputs(task, geometry, k_l1, l1_slot)
        publish_l1(l1_slot)
        acquire_l1_for_reads(l1_slot)
        for k_l0 in windows(k_l1.valid, KL0):
            ab_slot = acquire_next_l0_input_slot()
            l12l0_merged_inputs(geometry, k_l0, l1_slot, ab_slot)
            publish_l0(ab_slot)
            acquire_l0_for_compute(ab_slot)
            init = (k_l1.offset == 0 and k_l0.offset == 0)
            issue_merged_mmad(geometry, ab_slot, c_slot, init)
            release_l0_after_compute(ab_slot)
        release_l1_after_last_read(l1_slot)
    publish_l0c_for_output(c_slot)
    acquire_l0c_for_output(c_slot)
    for batch_local in range(task.valid_batch_count):
        l0c2gm_diagonal(task, geometry, batch_local, c_slot)
    release_l0c_after_all_diagonal_output(c_slot)
drain_all_pending_operations()
```

这些是职责伪操作，具体指令和同步接口由 [所选编程接口映射](../../../patterns.md#编程接口选择) 绑定；单次扩大MMAD计算所有batch配对，输出循环只选择对角。

各层槽计数在初始化时置0；每次取得新槽使用 `counter % slotCount` 并递增。计数跨任务继续，不因GM任务基址改变而重置；本例L1及L0输入各2槽，L0C槽数由对应DAG定义。取得槽时等待旧消费者完成，发布后消费侧显式等待生产完成。

## 3 地址偏移计算

### 3.1 核间偏移计算

任务映射引用 [Tiling §7](tiling.md#7-多核切分)：`taskId=coreId+taskRound*coreCount`，仅处理 taskId<taskCount。`batchTaskOffset=taskId*BatchMerge`，`batchTaskValid=min(BatchMerge,batchCount-batchTaskOffset)`，当前输入 batch 为 `b=batchTaskOffset+batchLocal`。

GM 原始基址与当前任务基址分开：`aTaskBase=aGmBase+inputElemBytes*batchTaskOffset*aBatchStride`；B、C 分别使用自身 batch stride 和元素字节数。任务起点来自核任务循环，不能按 coreId 固定一个 batch 基址。

### 3.2 核内循环间偏移计算

#### 3.2.1 GM 输入与输出

所有 stride 以元素为单位，base 为字节地址；`k=kL1Offset+kL0Offset+kL0Element`。

```text
A = aTaskBase + inputElemBytes*(batchLocal*aBatchStride + m*aStrideM + k*aStrideK)
B = bTaskBase + inputElemBytes*(batchLocal*bBatchStride + k*bStrideK + n*bStrideN)
C = cTaskBase + outputElemBytes*(batchLocal*cBatchStride + m*N + n)
```

A 的 [M,K]/[K,M] 与 B 的 [K,N]/[N,K] 分别决定对应 stride；C 固定 [M,N]。CopyIn MTE2 使用 kL1Offset 作为源窗口起点，L0 相对偏移用于后续 MTE1。

#### 3.2.2 片上槽与合并坐标

令 `mu(batchLocal)=mergedMStart(batchLocal)`、`nu(batchLocal)=mergedNStart(batchLocal)`，由物理拼接布局推导，不假定始终为 batchLocal*M、batchLocal*N。下式 b、i、j 均为任务内批次编号，不是全局 batch。槽间距是分配后的固定字节跨度，各层有独立原始基址；L1 和 L0 布局映射相同合并坐标：

```text
AL1 = aL1Base + l1Slot*aL1SlotBytes + offsetBytes(aL1Layout, mu(b)+m, kInL1)
BL1 = bL1Base + l1Slot*bL1SlotBytes + offsetBytes(bL1Layout, kInL1, nu(b)+n)
AL0 = aL0Base + abSlot*aL0SlotBytes + offsetBytes(aL0Layout, mu(b)+m, kInL0)
BL0 = bL0Base + abSlot*bL0SlotBytes + offsetBytes(bL0Layout, kInL0, nu(b)+n)
D   = cL0Base + cSlot*cL0SlotBytes + offsetBytes(cL0Layout, mu(i)+m, nu(j)+n)
```

读取 L1 时 `kInL1=kL0Offset+kInL0`，使用当前 L1 父窗口布局；写入 L0 后 kInL0 从局部 0 开始。D在K循环中不变址；输出令`i=j=batchLocal`，不能用`batchLocal*M*N`代替扩大布局中的对角起点。

### 3.3 贯穿实例的地址追踪

任务batch `[2,3]` 中batchLocal=1，取 `(m,n,k)=(1,2,65)`。GM A元素偏移 `3*16*128+1*128+65=6337`，B为 `3*128*16+65*16+2=7186`，C为 `3*16*16+1*16+2=786`。对应扩大坐标为A(17,65)、B(65,18)、D(17,18)，当前KL0段内K=1。D逻辑(17,18)的物理字节由cL0Layout求得，不按GM行主序强行计算。

### 3.4 贯穿实例的片上数据布局

本例 A/B L1 为 NZ，L0A 为 NZ，L0B 为 ZN，L0C 为 NZ；布局能力见 [所选编程接口映射](../../../patterns.md#编程接口选择)。本例 BF16 输入和 FP32 累加均采用 16×16 分形。元素偏移与跨度引用 [公共 NZ/ZN 函数](../../common-contract.md#21-nzzn-布局坐标约定)。布局参数中的行列数取当前合并窗口有效形状的物理对齐值；槽间距仍取 [Tiling 分配](tiling.md#106-资源约束的输入与实例分配)。

对 §3.3 的乘积，当前核首任务槽号均从 0 开始；k=65 位于第二 KL0 窗口，因此 L0 输入槽为 1。以下均为相对各存储层原始基址的字节偏移：

| 数据 | 布局内元素偏移 | 加槽基址后的字节偏移 |
|---|---:|---:|
| A L1(17,65) | NZ(17,65;32)=2321 | 4642 |
| B L1(65,18) | NZ(65,18;128)=3090 | 16384+6180=22564 |
| A L0(17,1) | NZ(17,1;32)=273 | 4096+546=4642 |
| B L0(1,18) | ZN(1,18;32)=289 | 4096+578=4674 |
| D L0C(17,18) | NZ(17,18;32)=786 | 3144 |

D 两个对角起点分别为 NZ(0,0;32)=0、NZ(16,16;32)=768 个累加元素；后者不是紧凑结果的 256 元素偏移。尾组仅一个 batch 时按 16×16 建立当前 D 布局，但仍保留最大 32×32 的槽跨度。

## 4 CopyIn MTE2 逻辑

### 4.1 GM 输入窗口

各batch根据真实GM stride搬到拼接布局指定位置；K尾块与布局空隙由首次物化负责填零。末组只搬有效batch，不越读下一batch。转置输入的搬运保持逻辑mu/nu映射不变，具体物理padding取所绑定布局。

### 4.2 UB 数据布局

不涉及。

### 4.3 L1 目的数据布局

使用 §3.2 的 batch 合并坐标和当前 L1 父布局。A/B 的合并顺序相同，padding 范围包含在对应物理窗口内；槽起点使用固定分配结果。

## 5 CopyIn MTE1 逻辑

从合并L1父窗口读取当前K段，写合并L0A/L0B；扩大M/N形状在全部K段保持一致。不能在某段使用紧凑拼接、另一段使用每batch对齐拼接。

## 6 Compute 计算逻辑

`D[mu(i)+m,nu(j)+n]=sum_k A[batchTaskOffset+i,m,k]*B[batchTaskOffset+j,k,n]`。i≠j也计算，但不属于C的有效输出。首K段初始化整个D，后续完整累加；仅在完整K后发布结果。类型和输出转换遵守 [公共数值契约](../../common-contract.md#1-语义与数值)。

### 6.1 MMAD 计算顺序

每任务按 L1 窗口起点递增，其内按 KL0 起点递增；所有段写同一扩大 D。`kL1Offset==0 && kL0Offset==0` 决定初始化；最后有效 K 段完成后才发布整块。无独立 firstMmad 同步状态。

### 6.2 VF 实现

不涉及。

## 7 CopyOut MTE3 逻辑

不涉及。

## 8 CopyOut 输出通路 逻辑

对每个有效batch读取D中`[mu(b):mu(b)+M,nu(b):nu(b)+N]`，写该batch的GM结果。支持批量对角提取的后端可合并发射，否则按合法子窗口输出；两者必须使用同一源布局函数。目的为 C[batch,M,N]，按该 batch 的起点和 N 行步长写回。非对角块与padding不写回GM。

## 9 流水同步

L0C所有权从任务首段计算前持续到完整K结束，随后交给对角输出；最后一个对角块消费完成后才归还槽。每次扩大MMAD独立持有当前L0A/B槽；L1等最后K子窗口读取完成再复用。双L0C不解除同槽的M→输出→M依赖，本例使用 [双槽整块交接](../../execution-strategies.md#43-双槽整块交接)，按 [公共同步契约](../../common-contract.md#4-所有权与同步) 及实现映射绑定事件或流水锁；不混入单槽结果单元交接。

L1 的 A/B 槽各自等待最后读取完成，L0A/B 按公共输入双缓冲规则归还。槽计数跨任务保持：本例核 0 的任务 0 消耗一个 L1 窗口、两个 L0 窗口和一个 L0C 槽，随后任务 2 从 L1 槽 1、L0 输入槽 0、L0C 槽 1 开始。所有槽回收是相应流水上的完成依赖，发射循环不通过主机阻塞等待模拟并行；收尾等待全部有效对角输出。

## 10 Kernel 产出校验

验收接口遵循 [公共契约 §5](../../common-contract.md#5-产出与验收)，本节给出本模板的覆盖与数值检查。

取 `A[b,m,k]=b+1`、`B[b,k,n]=b+2`，Golden为`128*(b+1)*(b+2)`；扩大D的非对角块为`128*(iGlobal+1)*(jGlobal+2)`，刻意与对角值不同以暴露错提取。增加随m/n/k变化的输入验证转置与K段地址；在输出边界放哨兵，确认只写5个16×16结果。

覆盖BatchMerge=1、完整组和尾组、多KL1/KL0段、各方向布局及对齐空隙。逐元素独立Golden检查有效结果，地址包围盒检查完整D写入；不能因输出精度通过就忽略非对角区域的越界写入风险。


地址校验逐点枚举每层布局，验证单槽内映射无重叠、所有访问落入预留跨度，且尾组不读写第 5 个 batch 之后的 GM 区域。以“将第二对角起点误设为 M*N”和“按尾组有效字节重算槽间距”构造错误变体，分别要求数值检查和固定分配区间检查识别。

## 11 Elementwise/Broadcast Bias[N] 组合扩展的地址与计算

组合语义为 `Y[...,m,n]=convert_out(acc[...,m,n]+bias[n])`。GM 只有一份 `[N]` 向量；合并执行的扩大列坐标 `nMerged` 使用 `bias[nMerged % N]`，即在每个扩大 N 子区重复映射 Bias[N]。因此每个对角输出块都加上同一局部列向量，非对角结果仍不写回。

使用[公共 Bias[N] 组合扩展首段融合骨架](../../execution-strategies.md#8-biasn-组合扩展的首段融合策略)：扩大矩阵计算的首 K 段按上述映射以 bias 启动累加，后续 K 段只累加乘积；若执行映射不支持首段融合，须使用等价 Elementwise/Broadcast 输出阶段保持同一数值契约。无组合扩展时保留原零初始化骨架。L0C 发布与输出协议保持所选完整路线。

取 `bias[n]=n+1`，Golden 为每个保留轴坐标的对角收缩结果加 `bias[n]` 后按 Tout 转换；非对角块即使参与扩大矩阵计算也不输出。检查所有扩大 N 子区重复映射、尾部有效区不越界、每个有效输出恰好应用一次、K 分窗不重复应用、无组合扩展不读 bias，以及 K=0 时按公共预处理产生 `convert_out(bias[n])`。
