# contraction 范式 batch Kernel 模板

公共语义、地址单位、资源描述、所有权与验收接口统一引用 [公共契约](../../common-contract.md)；本页只展开本模板的任务和布局变化。

## 1 Kernel 概述

数据流和贯穿实例见 [DAG](dag-buffers.md)。一个任务计算一个 batch 的 M/N 块及完整 K；任务映射见 [Tiling §7](tiling.md#7-多核切分)。本页定义普通广播分块，批次分层和合并分别由独立模板负责。每任务采用常规 KL1 窗口、两级输入双槽与单 L0C 整块交接；本例所有任务的 K 窗口均为 256、256、128。

## 2 Kernel 骨架代码

```text
initialize_slot_ownership()
for task_id in tasks_for_core(core_id):
    parent_id, m_part, n_part = resolve_global_tail_task(task_id, core_id)
    batch_id, matrix_task_id = divmod(parent_id, matrix_task_count)
    p = unravel_batch(batch_id)
    m_task, n_task = resolve_swat_rectangle(matrix_task_id, m_part, n_part)
    if m_task.valid == 0 or n_task.valid == 0:
        continue
    a_task, b_task, c_task = bind_broadcast_task_bases(p, m_task, n_task)
    acquire_l0c_for_compute()
    for k_l1 in windows(K, KL1):
        l1_slot = next_l1_slot()
        acquire_l1_for_write(l1_slot)
        gm2l1(a_task, b_task, k_l1, l1_slot)
        publish_l1(l1_slot)
        acquire_l1_for_reads(l1_slot)
        for k_l0 in windows(k_l1.valid, KL0):
            l0_slot = next_l0_slot()
            acquire_l0_for_write(l0_slot)
            l12l0(l1_slot, k_l0, l0_slot)
            publish_l0(l0_slot)
            acquire_l0_for_compute(l0_slot)
            init = (k_l1.offset == 0 and k_l0.offset == 0)
            final = (k_l1.offset + k_l0.offset + k_l0.valid == K)
            issue_mmad(l0_slot, init, final)
            release_l0_after_compute(l0_slot)
        release_l1_after_last_read(l1_slot)
    publish_l0c_for_output()
    acquire_l0c_for_output()
    l0c2gm(c_task)
    release_l0c_after_output()
drain_all_pending_operations()
```

伪操作表示带完成语义的资源交接，不是同步阻塞的主机调用；具体 API 见 [所选编程接口映射](../../../patterns.md#编程接口选择)。槽计数跨任务持续轮转，L0C 数值初始化由当前 K 坐标决定。

各层槽计数在初始化时置0；每次取得新槽使用 `counter % slotCount` 并递增。计数跨任务继续，不因GM任务基址改变而重置；本例L1及L0输入各2槽，L0C槽数由对应DAG定义。取得槽时等待旧消费者完成，发布后消费侧显式等待生产完成。

## 3 地址偏移计算

### 3.1 核间偏移计算

先按 [Tiling §7.2](tiling.md#72-切分公式) 将发射编号还原为全局父编号，再解码 batch 和矩阵父块，最后应用尾轮子块偏移。以下 mTaskOffset/nTaskOffset 是最终子块坐标，有效长度也使用子块长度；每个子块独立完成完整 K。

对输出前导坐标 `p[d]`，`aIndex[d]=0 if aShape[d]==1 else p[d]`；B 同理。所有 stride 以下均以元素计：

```text
aBatchElems = sum(aIndex[d] * aBatchStride[d])
bBatchElems = sum(bIndex[d] * bBatchStride[d])
cBatchElems = sum(p[d] * cBatchStride[d])
aTaskBase = aGmBase + inputElemBytes * (aBatchElems + mTaskOffset*aStrideM)
bTaskBase = bGmBase + inputElemBytes * (bBatchElems + nTaskOffset*bStrideN)
cTaskBase = cGmBase + outputElemBytes * (cBatchElems + mTaskOffset*cStrideM + nTaskOffset*cStrideN)
```

### 3.2 核内循环间偏移计算

GM A/B 窗口分别加 `inputElemBytes*kL1Offset*aStrideK/bStrideK`。片上地址始终是 `layerBase + slotId*slotBytes + layout_offset_bytes(local_coordinates)`：L1 使用父窗口的布局和 kL0Offset；L0A/B 的局部 K 从 0 开始；L0C 只使用局部 m/n，不随 K 或全局 batchId 平移。槽外 batch 不在片上常驻，不能额外叠加 batchStride。

### 3.3 贯穿实例的地址追踪

输出 batch `(1,2)`、矩阵点 `(m,n,k)=(129,257,513)`，行主序下 A 元素偏移 `1*256*640+129*640+513=246913`，B 偏移 `2*640*384+513*384+257=688769`，C 偏移 `5*256*384+129*384+257=541313`。字节偏移各乘 2。该点属于 m/n 起点 `(128,256)`、kL1 起点 512 的任务窗口；L1/L0 坐标由相对偏移求得，不能使用上述 GM 数字。

## 4 CopyIn MTE2 逻辑

UB 不涉及。搬运当前任务的有效 A/B 窗口到 L1，并在首次物化物理尾块时填零。广播只改变源基址；各次搬运仍遵守真实源 stride 与有效边界。

## 5 CopyIn MTE1 逻辑

按父 L1 数据布局读取 kL0 子窗口，写当前 L0A/B 槽的目的布局；源地址保留父窗口 K 跨度，目的地址从当前计算窗口的局部 K=0 开始。L1 只能在最后一次读取完成后释放。

## 6 Compute 计算逻辑

每输出元素只归约自身 batch 对应的 K。首段初始化、后续累加、末段发布；一次 K 段同时可能是首段和末段。FP32/INT32 累加及 Tout 转换遵守[公共数值契约](../../common-contract.md#1-语义与数值)。VF 不涉及。

## 7 CopyOut MTE3 逻辑

不涉及。

## 8 CopyOut 输出通路 逻辑

从当前 L0C 有效 m/n 写到 cTaskBase 对应窗口，按 C 的物理轴顺序取目的 stride。广播输入不会使输出地址广播；只写逻辑有效元素。

## 9 流水同步

遵循 [公共同步契约](../../common-contract.md#4-所有权与同步)，本例选择 L1 成对双槽与 L0 输入双槽，具体规则见 [公共执行策略 §2–3](../../execution-strategies.md#2-l1-成对分区双缓冲)。L0C 所有权覆盖当前任务完整 K；每次 MMAD 交接 L0A/B。跨 batch 没有额外的全流水屏障，但新任务首次写 L0C 必须等前任务输出完成。输入槽复用和任务基址更新是两项独立约束。

## 10 Kernel 产出校验

验收接口遵循 [公共契约 §5](../../common-contract.md#5-产出与验收)，本节给出本模板的覆盖与数值检查。

令本例 `A[p,0,m,k]=p+1`、`B[0,q,k,n]=q+1`，则每个 C 元素应为 `640*(p+1)*(q+1)`；该例检查广播，另用随 m/n/k 变化的非对称输入检查转置和地址。分别覆盖 4 种输入存储轴组合、C[M,N] 输出、各类型、单/多段 K、非整尾块、多任务复用。以逐元素独立 CPU 归约为 Golden，并检查输出外哨兵和任务唯一覆盖；浮点误差按公开契约判断。


## 11 Batch 与矩阵轴交错存储

当逻辑输入仍是 A(b,m,k)、B(b,k,n)，但物理A为 `[M,Batch,K]`、输出C为 `[M,Batch,N]` 时，已有Batch任务不变，只修改存储映射：

`offsetA=m*Batch*K+b*K+k`，`offsetC=m*Batch*N+b*N+n`。

例如 `Batch=3,M=2,N=5,K=4`，逻辑(b,m,k)=(1,1,2)对应A元素偏移18；C(b,m,n)=(1,1,3)对应偏移23。不能继续使用batch连续矩阵的 `b*M*K+m*K+k`；这不是A矩阵内ND/DN标志能够完整表达的变化。GM→L1操作必须能表达跨行的真实stride；若需要物理重排，应把该搬运阶段显式加入设计，而不是假定转置免费。

若该实现还有与A同行、按K分组的输入描述，其地址独立使用 `[M,Batch,auxGroupsPhysical]` 布局；不以payload字节stride替代描述的stride。此处只描述同一 contraction 的地址变化，不新增“转置算子模板”。

## 12 Bias[N] 组合扩展的地址与计算

原贯穿实例保持 hasBias=false；以下为同一形状接入 Bias[N] 组合扩展。所有保留轴坐标共享同一 N 向量。

GM 元素偏移为 `nTaskOffset + nLocal`，按 Bias[N] 的真实 strideN 换算地址；同一向量区间供当前任务中各输出 M 行使用。

尾轮子任务继承父任务的 N 起点，只裁剪有效 N 长度，不改变向量索引。

使用 [公共 Bias[N] 组合扩展融合骨架](../../execution-strategies.md#8-biasn-组合扩展的首段融合策略) 替换原首段零初始化：有组合扩展时首段以 bias 启动，后续累加；无组合扩展时保留原骨架。L0C 发布与输出协议保持所选完整路线。

在原实例输入上取 `bias[n]=n+1`，Golden 为原收缩结果加 `bias[n]` 后按 Tout 转换。检查完整 K 只应用一次组合扩展，尾列不越界；以 N 偏移错误、每个 K 段重复应用和无组合扩展误读 bias 作为负例。K=0 走公共预处理规定的输出路径。
