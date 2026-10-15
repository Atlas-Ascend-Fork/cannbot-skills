# contraction 单侧 L1 驻留模板：Kernel 设计

## 1 Kernel 概述

沿用 [DAG 的 B 整矩阵驻留实例](dag-buffers.md)：本核预载 B，随后执行多个 M 条带任务；每个任务沿 N 输出块逐一完成完整 K 归约。当前输出块的 A 按 KL1 流式搬运，L0A/B 按 KL0 轮转。与基础模板的关键差异是 B 的初始化和释放位于整个任务循环外。

## 2 Kernel 骨架代码

以下为语言无关的依赖伪代码。`afterReads`、`afterMmad`、`afterOutput` 表示在相应流水访问完成后交接；不是每一步阻塞主机或全流水等待。

```text
initializeSlotOwnership()
tasks = tasksForCore(coreId, coreCount)
if tasks.empty():
    return
acquireResidentBForWrite()
copyGmToResidentB(fullB)
publishResidentBReady()
acquireResidentBForReads()
aL1WindowIndex = 0
l0WindowIndex = 0
for task in tasks:
    for nL0Offset in range(0, N, NL0):
        output = outputTile(task, nL0Offset)
        acquireL0CForCompute()
        for kL1Offset in range(0, K, KL1):
            aSlotId = aL1WindowIndex % 2
            acquireAStreamSlotForWrite(aSlotId)
            copyGmToAStreamSlot(task, kL1Offset, aSlotId)
            publishAStreamSlotReady(aSlotId)
            acquireAStreamSlotForReads(aSlotId)
            kL1Valid = min(KL1, K - kL1Offset)
            for kL0Offset in range(0, kL1Valid, KL0):
                kL0Valid = min(KL0, kL1Valid - kL0Offset)
                l0SlotId = l0WindowIndex % 2
                acquireL0InputsForWrite(l0SlotId)
                copyL1ToL0(aSlotId, residentB, output,
                           kL1Offset, kL0Offset, l0SlotId)
                publishL0InputsReady(l0SlotId)
                acquireL0InputsForCompute(l0SlotId)
                initialize = (kL1Offset == 0 and kL0Offset == 0)
                accumulate(output, l0SlotId, kL0Valid, initialize)
                releaseL0InputsAfterMmad(l0SlotId)
                l0WindowIndex += 1
            releaseAStreamSlotAfterReads(aSlotId)
            aL1WindowIndex += 1
        publishL0CFinalResult()
        acquireL0CForOutput()
        copyL0CToGm(output)
        releaseL0CAfterOutput()
releaseResidentBAfterAllReads()
drainIssuedWorkAndReleaseSyncResources()
```

L0C 计算侧持有范围覆盖整个 K 循环；初始化由当前 K 坐标推导。B 的消费侧获取一次，覆盖后续所有读取；不能对一次预载就绪事件逐条等待，也不能每轮读取后把唯一驻留槽归还给写侧。L0 输入轮转引用[公共双缓冲策略](../../execution-strategies.md#3-l0-输入双缓冲)。实际机制绑定见[公共同步契约](../../common-contract.md#4-所有权与同步)与[所选编程接口映射](../../../patterns.md#编程接口选择)。

## 3 地址偏移计算

### 3.1 核间任务基址

任务映射见 [Tiling §7](tiling.md#7-多核切分)。令 `mTaskOffset=taskId*ML1`，条带内输出块 N 起点为 `nL0Offset`；`mTaskValid=min(ML1,M-mTaskOffset)`、`nOutputValid=min(NL0,N-nL0Offset)`。

GM 步长均以元素为单位：A 为 `(sAm,sAk)`，B 为 `(sBk,sBn)`，C 为 `(sCm,sCn)`。输出 `(sCm,sCn)=(N,1)`；输入轴顺序按[公共存储地址契约](../../common-contract.md#2-存储与地址)绑定，不能假设当前核只有一个任务。

### 3.2 GM 地址

```text
aGmAddr = aGmBase + ((mTaskOffset + mLocal)*sAm
                      + (kL1Offset + kLocal)*sAk)*sizeof(Tin)
bGmAddr = bGmBase + (kGlobal*sBk + nGlobal*sBn)*sizeof(Tin)
cGmAddr = cGmBase + ((mTaskOffset + mLocal)*sCm
                      + (nL0Offset + nLocal)*sCn)*sizeof(Tout)
```

B 的 GM 地址只用于整矩阵预载，不再为每个任务重新读取。A 的任务基址随 taskId 改变，C 同时包含任务 M 起点和当前 N 子块起点。

### 3.3 L1 地址

记 `Offset(layout,i,j)` 为父数据布局中的元素偏移，布局本身含完整父形状和物理步长：

```text
aL1Addr = aL1SlotBase[aSlotId]
        + Offset(ALayoutL1, mLocal, kL0Offset + kInner)*sizeof(Tin)
bL1Addr = bResidentBase
        + Offset(BResidentLayout, kL1Offset + kL0Offset + kInner,
                                  nL0Offset + nInner)*sizeof(Tin)
```

A 源父形状是本例固定流式 `[ML1,KL1]` 布局，B 源父形状是完整 `[K,N]` 驻留矩阵。B 在 K/N 方向均带完整坐标；不能把 B 当成从零起点的 `[KL0,NL0]` 紧凑源。尾窗口仍使用此父布局，物理槽基址使用 [Tiling §10.6](tiling.md#106-资源约束的输入与实例分配)；有效长度不缩短槽间距。

### 3.4 L0A、L0B 和 L0C 地址

```text
aL0Addr = aL0SlotBase[l0SlotId]
        + Offset(ALayoutL0, mLocal, kInner)*sizeof(Tin)
bL0Addr = bL0SlotBase[l0SlotId]
        + Offset(BLayoutL0, kInner, nInner)*sizeof(Tin)
cL0Addr = cL0Base
        + Offset(CLayoutL0, mLocal, nInner)*sizeof(Tacc)
```

L0 输入 K 从本次子窗口的零起点开始，L0C 不随 K 偏移。GM 的任务偏移不再叠加到 L0 地址。所有槽基址由资源规划给出，轮转索引跨 N 子块与任务持续递增。

### 3.5 贯穿实例的地址追踪

取核 0 的第二个任务 `taskId=4`，其第二个 N 输出块，乘积项 `A[273,149]*B[149,83]`：`mTaskOffset=256,nL0Offset=64,kL1Offset=128,kL0Offset=0`，局部坐标为 `mLocal=17,nInner=19,kInner=21`。

采用 [所选编程接口映射](../../../patterns.md#编程接口选择) 对应的 BF16 NZ/ZN 布局。元素偏移采用 [公共 NZ/ZN 函数](../../common-contract.md#21-nzzn-布局坐标约定)，本例 R/C 已对齐；A L1/L0A 为 NZ，B L1 为 NZ、L0B 为 ZN，L0C 为 NZ。

| 位置 | 相对矩阵或所选槽的元素偏移 | 字节偏移 |
|---|---|---:|
| GM A | `273*256+149=70037` | 140074 |
| GM B 预载源 | `149*128+83=19155` | 38310 |
| A L1 | `NZ(17,21;64)=1301` | 2602 |
| 驻留 B L1 | `NZ(149,83;256)=22867` | 45734 |
| L0A | `NZ(17,21;64)=1301` | 2602 |
| L0B | `ZN(21,19;64)=1333` | 2666 |
| L0C | `NZ(17,19;64)=1299` | 5196 |
| GM C | `273*128+83=35027` | 70054 |

表中片上地址还需加各槽字节基址。B L1 使用全局 K/N 坐标，L0B 使用局部 K/N 坐标；这一区别是驻留模板的核心地址约束。FP16、INT8 或其他转置布局按各自能力重算，不能机械复用该 BF16 数值表。

### 3.6 A 整矩阵驻留的地址变化

按 [Tiling §8.1](tiling.md#81-a-驻留的对称变化)，令 `nTaskOffset=taskId*NL1`；任务内沿 M 遍历 `mL0Offset`。有效输出尺寸为 `min(ML0,M-mL0Offset)`、`min(NL1,N-nTaskOffset)`。A 预载读取完整 `[M,K]`，B 流式窗口从 `(kL1Offset,nTaskOffset)` 搬入。

```text
aL1Addr = aResidentBase + Offset(AResidentLayout, mL0Offset + mInner,
                                kL1Offset + kL0Offset + kInner)*sizeof(Tin)
bL1Addr = bL1SlotBase[bSlotId] + Offset(BLayoutL1,
                                       kL0Offset + kInner, nInner)*sizeof(Tin)
cGmAddr = cGmBase + ((mL0Offset + mInner)*N
                     + nTaskOffset + nInner)*sizeof(Tout)
```

A 父布局包含完整 M/K，B 父布局只包含当前 KL1/NL1 窗口。L0 地址仍使用局部 M/N/K。驻留 A 在本核所有任务的最后一次 MTE1 读取后释放，流式 B 按每窗口最后读取归还；不能仅交换指针而保留 B 驻留的坐标公式。

## 4 CopyIn MTE2 逻辑

### 4.1 预载与流式搬运

B 完整搬入发生在本核首任务之前；如果目标需要拆成多个合法搬运，全部完成后才发布整份矩阵就绪。A 每个输出块按 KL1 搬入，写槽前等待该槽上次所有 MTE1 读取完成。

### 4.2 UB 数据布局

不涉及。

### 4.3 L1 目的数据布局

分别使用驻留 B 的完整父布局和流式 A 的窗口布局。两类目的区域不重叠，GM 转置由对应步长和搬运转换表达。首次物化尾块时，GM→L1 只读逻辑有效范围，将后续计算会读取的无效输入位置置零；L1→L0 按当前 `mTaskValid/nOutputValid/kL0Valid` 取数，并补齐目的计算读取范围。MMAD 使用这些逻辑有效尺寸，输出只写有效 M/N。

## 5 CopyIn MTE1 逻辑

从 A 窗口与完整 B 父布局取相同逻辑 K 区间，写到一对空闲 L0A/B 槽。B 的 N 子窗口由当前输出块确定；A 最后一个 KL0 子窗口读取结束后释放当前 A 槽，B 保持可读。L1 源地址和 L0 目的地址分别使用 §3.3、§3.4。

## 6 Compute 计算逻辑

每个 `[mTaskValid,nOutputValid]` 输出块按 K 顺序累加，首段初始化，末段发布完整结果。FP16/BF16 保留 FP32 部分和，INT8 保留 INT32 部分和；数值契约引用[公共数值契约](../../common-contract.md#1-语义与数值)，MMAD 结果单元生成顺序与输出消费顺序按目标能力配对。B 驻留不改变累加顺序。VF 不涉及。

## 7 CopyOut MTE3 逻辑

不涉及。

## 8 CopyOut 输出通路 逻辑

将当前输出块从 L0C 写到 §3.2 的 GM C 地址，只写有效 M/N 范围。完整条带需要多次输出，不能用 NL1 代替单次 NL0 输出宽度。Tout 转换只在最终写回执行，输出按 C 的 M/N 步长写回。

## 9 流水同步

### 9.1 所有权交接范围

| 资源 | 生产和首次可读 | 最后消费者与归还边界 |
|---|---|---|
| 驻留 B L1 | 完整预载完成后交给 MTE1 | 本核全部任务最后一次 MTE1 读取完成 |
| 流式 A L1 | 当前 KL1 窗口搬完后交给 MTE1 | 当前窗口最后一次 MTE1 读取完成 |
| L0A/B | 当前 KL0 搬完后交给计算侧 | 当前 MMAD 读取结束 |
| L0C | 完整 K 归约后交给输出侧 | 当前输出块写回消费结束 |

首次输入槽归生产者，驻留 B 首轮没有前一次消费可等待。B 释放并不要求它参与的所有 MMAD 已结束，因为 MTE1 已将所需值复制到 L0B；Kernel 退出仍需等待所有已发射计算与输出完成。若实现将多个资源合并到同一同步对象，释放范围必须覆盖其中最晚的消费者，不能沿用 A 窗口级归还去覆盖 B。

### 9.2 发射与等待

预载就绪后可以按目标流水语义提交 A 搬运、L0 搬运和 MMAD，依赖只等待相应资源。单槽 L0C 的取得位于当前输出块 K 循环前，释放位于输出后；不在 K 循环内重复交接。本例明确采用单槽整块交接，甘特图及骨架均在完整结果后发射输出；若改用[单槽结果单元交接](../../execution-strategies.md#42-单槽结果单元交接)，应同时替换写权限、末段发布和输出消费，不保留整槽等待。两者的依赖规则引用[公共契约](../../common-contract.md#4-所有权与同步)，具体机制引用接口映射，不能用额外全流水屏障代替跨流水依赖。

bank 资源竞争按 DAG §1.7 检查；驻留 B 单槽长期存活不意味着 MTE1 读它与 MTE2 写 A 必然无竞争。

## 10 Kernel 产出校验

用非恒定 A/B 检查四核任务表、二十个输出块和全部 K 覆盖；每个 C 元素只写一次。单独统计每核 B 预载次数应为一，A 每个 N 输出块重新遍历 K。对 §3.5 乘积项逐层解码，检查驻留 B 的父跨度未被子窗口跨度覆盖。

同步模型覆盖首轮、A 和 L0 槽重复复用、同核第二任务、最后任务；构造“B 第一 K 窗口后提前释放”和“B 子窗口改用局部父跨度”的错误变体，确保依赖检查或地址校验能拒绝。数值覆盖三类输入/累加组合、A/B 四种轴顺序、C[M,N] 输出、M/N/K 尾部和单段 K。设备检查进一步验证合法搬运参数、精度和越界，不能用逻辑模型代替异步设备执行。

## 11 双侧完整 K 面板的地址与执行

该变体使用 [DAG §3](dag-buffers.md#3-双侧完整-k-面板驻留变体) 的任务与局部形状。GM输入包含任务M/N起点；L1面板只保留任务内坐标。继续使用 §3 的 `Offset(layout,i,j)` 元素偏移，所有 base 均为字节基址：

```text
aL1Addr = aPanelBase + Offset(ALayoutL1, mL0Offset + mLocal,
                              kL0Offset + kInner)*sizeof(Tin)
bL1Addr = bPanelBase + Offset(BLayoutL1, kL0Offset + kInner,
                              nL0Offset + nInner)*sizeof(Tin)
aL0Addr = aL0SlotBase[l0SlotId] + Offset(ALayoutL0, mLocal, kInner)*sizeof(Tin)
bL0Addr = bL0SlotBase[l0SlotId] + Offset(BLayoutL0, kInner, nInner)*sizeof(Tin)
cL0Addr = cL0Base + Offset(CLayoutL0, mLocal, nInner)*sizeof(Tacc)
cGmAddr = cGmBase + ((mTaskOffset + mL0Offset + mLocal)*N
                     + nTaskOffset + nL0Offset + nInner)*sizeof(Tout)
```

L1父布局使用完整K，L0目的从局部K=0开始；局部输入pitch不能用于全局C的目的stride。英文伪操作如下，均表示对应流水完成后的交接：

```text
l0WindowIndex = 0
for task in tasks_for_core():
    acquire_panel_slots_for_write()
    load_full_k_panels(task)
    publish_and_acquire_panels_for_reads()
    for output_tile in mn_subtiles(task, ML0, NL0):
        acquire_l0c_for_compute()
        for k_window in windows(K, KL0):
            l0SlotId = l0WindowIndex % 2
            acquire_l0_inputs_for_write(l0SlotId)
            load_panel_subwindows(output_tile, k_window, l0SlotId)
            publish_and_acquire_l0_inputs_for_compute()
            issue_mmad(init=(k_window.offset == 0))
            release_l0_inputs_after_compute(l0SlotId)
            l0WindowIndex += 1
        publish_and_acquire_l0c_for_output()
        write_valid_output(output_tile)
        release_l0c_after_output()
    release_panels_after_last_read()
drain_pending_operations()
```

L0槽计数初始为0，跨输出块/任务递增取模2。L1面板不能在首个输出块算完后覆盖；L0C不能在每个KL0内交给输出。例中任务起点(64,64)、子块起点(32,32)的结果点(97,98)来自局部(1,2)；GM C元素偏移为12514，片上C只用(1,2)。验证四任务、每任务四输出块覆盖128×128一次，每输出块完整消去K；面板容量与全局/局部stride分开检查。

## 12 可选 bias 的地址与计算

原贯穿实例保持 hasBias=false；以下为同一形状的 hasBias=true 扩展。共享 bias[N]。A 驻留时各 N 块取不同列区；B 驻留时不同 M 块可重用同一列区。

GM 元素偏移为 `nTaskOffset + nLocal`，按 Bias[N] 的真实 `strideN` 换算字节地址；所有保留轴坐标共享该输入向量。MergeBatch 扩大布局按 `nMerged % N` 映射到原向量，不改变逻辑 bias shape。

bias 复用域独立于 A/B 驻留域；每个条带内输出块仍独立首段注入，驻留值不表示已经为新输出加入 bias。

使用[公共 Bias[N] 组合扩展首段融合骨架](../../execution-strategies.md#8-biasn-组合扩展的首段融合策略)替换原首段零初始化：有组合扩展时每个输出块在首 K 段以 `bias[n]` 启动，后续累加；无组合扩展时保留原骨架。IterBatch 按保留轴坐标顺序复用同一 N 子区；MergeBatch 按扩大 N 映射重复同一向量。L0C 发布与输出协议保持所选完整路线。

取 `bias[n]=n+1`，Golden 为每个输出坐标的收缩结果加同一 `bias[n]` 后按 Tout 转换。检查不同 M/N 子块的 N 起始偏移、每个输出只应用一次、K 分窗不重复应用、无组合扩展不读 bias，以及 K=0 输出 `convert_out(bias[n])`；MergeBatch 另查扩大列重复映射与对角提取。
