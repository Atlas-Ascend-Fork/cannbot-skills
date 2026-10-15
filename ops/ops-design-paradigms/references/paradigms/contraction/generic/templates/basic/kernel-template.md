# contraction 范式 basic 模板

## 1 Kernel 概述

### 1.1 数据流图

本实例计算 `C[m,n] = sum_k A[m,k]*B[k,n]`。完整边界和 Buffer 所有权见 [DAG](dag-buffers.md)，块尺寸与分核见 [Tiling](tiling.md)。

```text
GM A/B → MTE2 → L1 双槽 → MTE1 → L0A/B 双槽
                                      ↓
                                 MMAD，完整 K 累加
                                      ↓
                                 L0C 单槽 Tacc
                                      ↓
                             输出通路 → GM C Tout
```

## 2 Kernel 骨架代码

### 2.1 Kernel 骨架代码

以下函数名均为依赖伪码，不是 API；“等待/发布/归还”表示依赖和所有权交接，由选定后端实现。为了便于阅读按依赖列出，不表示每步都等待所有执行资源完成。

```text
l1WindowIndex = 0
l0WindowIndex = 0
outputTaskIndex = 0
initialize_writable_slots()  # 每核建立输入槽与累加槽可写初态
for task in assigned_output_tasks:
    mTaskOffset, nTaskOffset, mTaskValid, nTaskValid = task
    aTaskBase, bTaskBase, cTaskBase = task_gm_bases(task)  # 见 §3.1
    cSlotId = outputTaskIndex % cSlotCount
    begin_accumulation(cSlotId, outputProtocol)  # 整块等待或逐单元写权限
    for kL1Offset, kL1Valid in l1_windows(K, KL1, KL0, outputTaskIndex == 0):
        l1SlotId = l1WindowIndex % 2
        wait_writable(L1, l1SlotId)
        gm2l1(aTaskBase, bTaskBase, kL1Offset, kL1Valid, l1SlotId)
        prepare_l1_read_region(l1SlotId)  # 按物理窗口规则建立所有将被读取的值
        publish_ready(L1, l1SlotId)
        wait_ready(L1, l1SlotId)  # 一次取得当前窗口全部子窗口的读权限
        for kL0Offset in range(0, kL1Valid, KL0):
            kL0Valid = min(KL0, kL1Valid-kL0Offset)
            l0SlotId = l0WindowIndex % 2
            wait_writable(L0AB, l0SlotId)
            l12l0(l1SlotId, kL0Offset, kL0Valid, l0SlotId)
            prepare_l0_read_region(l0SlotId)  # 完成计算输入所需 padding
            publish_ready(L0AB, l0SlotId)
            wait_ready(L0AB, l0SlotId)
            final = (kL1Offset+kL0Offset+kL0Valid == K)
            initialize = (kL1Offset == 0 and kL0Offset == 0)
            accumulate_mmad(l0SlotId, cSlotId, initialize=initialize, final=final, protocol=outputProtocol)
            if final:
                publish_result(cSlotId, outputProtocol)
            release_after_mmad_read(L0AB, l0SlotId)
            l0WindowIndex += 1
        release_after_last_mte1_read(L1, l1SlotId)
        l1WindowIndex += 1
    consume_result(cSlotId, outputProtocol)
    l0c2gm(cTaskBase, mTaskValid, nTaskValid, Tacc, Tout, cSlotId)
    release_after_output_read(cSlotId, outputProtocol)
    outputTaskIndex += 1
finish_pipeline()  # 外部调用者按 stream 契约等待输出
```

窗口生成使用 [首块二分](../../execution-strategies.md#1-首块-l1-窗口二分)；begin/publish/consume/release 按 [L0C 交接](../../execution-strategies.md#4-l0c-选择与结果交接) 成套展开。单槽逐单元协议不额外插入整槽屏障。槽数与 KL1/KL0 由 Tiling 联合求解。L1/L0 槽轮转遵循各自的完整生命周期，不能把不同层槽号相同当作它们有相同生命周期。实际异步发射应允许下一窗口搬运与当前计算重叠。

## 3 地址偏移计算

### 3.1 核间偏移计算

用元素步长表示逻辑坐标到 GM 的映射：

```text
offsetA(m,k) = m*aStrideM + k*aStrideK
offsetB(k,n) = k*bStrideK + n*bStrideN
offsetC(m,n) = m*cStrideM + n*cStrideN
```

连续存储时，步长由 [DAG 轴顺序表](dag-buffers.md#11-基础场景与数据流) 得到：

| 存储 | 逻辑步长 |
|---|---|
| A[M,K] / A[K,M] | `(aStrideM,aStrideK)=(K,1)` / `(1,M)` |
| B[K,N] / B[N,K] | `(bStrideK,bStrideN)=(N,1)` / `(1,K)` |
| C[M,N] | `(cStrideM,cStrideN)=(N,1)` |

当前核的任务基址由 [分核策略](tiling.md#7-多核切分) 给出，不能仅按核号乘一个固定字节块推算。本模板先分配任务序号，再按 SWAT 映射到输出坐标：

```text
coreId = get_core_id()
coreCount = schedule.core_count
taskRound = 0  # 当前核的任务轮次
taskId = coreId + taskRound*coreCount                     # taskId < taskCount 时有任务
(mTaskOffset,nTaskOffset,mTaskValid,nTaskValid) = resolve_swat_task(taskId,coreId,schedule)  # 使用调度说明的父块与尾轮映射
```

调度器若采用其他遍历或尾块拆分方式，直接使用它返回的 `(mTaskOffset,nTaskOffset,mTaskValid,nTaskValid)`。后续地址推导只依赖这些坐标，不再自行重建另一套任务映射。

设 `aGmBase/bGmBase/cGmBase` 是各张量的 GM 字节基址，则当前核本轮任务的基址为：

```text
aTaskBase = aGmBase + offsetA(mTaskOffset,0)*sizeof(Tin)
bTaskBase = bGmBase + offsetB(0,nTaskOffset)*sizeof(Tin)
cTaskBase = cGmBase + offsetC(mTaskOffset,nTaskOffset)*sizeof(Tout)
```

本模板每个任务完成完整 K，所以任务起点的逻辑 K 为 0。一个核可以执行多个输出任务，每轮必须从张量原始基址和本轮坐标重新得到 aTaskBase/bTaskBase/cTaskBase；这些不是该核整个执行期间不变的地址。

### 3.2 核内循环间偏移计算

#### 3.2.1 GM 输入与输出

在本轮任务基址上叠加 K 窗口偏移：

```text
aL1WindowGmBase = aTaskBase + kL1Offset*aStrideK*sizeof(Tin)
bL1WindowGmBase = bTaskBase + kL1Offset*bStrideK*sizeof(Tin)
```

对当前 L1 窗口内的逻辑坐标 `(mInTask,kInL1)` 和 `(kInL1,nInTask)`，完整 GM 字节地址为：

```text
aGmAddr = aL1WindowGmBase + (mInTask*aStrideM + kInL1*aStrideK)*sizeof(Tin)
bGmAddr = bL1WindowGmBase + (kInL1*bStrideK + nInTask*bStrideN)*sizeof(Tin)
cGmAddr = cTaskBase + (mInTask*cStrideM + nInTask*cStrideN)*sizeof(Tout)
```

这里 `mInTask/nInTask` 是输出块内坐标，`kInL1` 是相对 kL1Offset 的 K 坐标。若继续按 KL0 分段，`kInL1=kL0Offset+kL0Element`，其中 kL0Offset 是 L0 窗口在当前 L1 窗口内的起点、kL0Element 是该 L0 窗口内坐标。展开后 A/B/C 地址分别对应全局逻辑坐标 `(mTaskOffset+mInTask,kL1Offset+kL0Offset+kL0Element)`、`(kL1Offset+kL0Offset+kL0Element,nTaskOffset+nInTask)`、`(mTaskOffset+mInTask,nTaskOffset+nInTask)`。

这些 GM 基址只用于全局存储访问。完成 gm2l1 后，l12l0 的源地址改为“当前 L1 物理槽基址 + 父 L1 布局的切片偏移”，目的地址为“当前 L0 物理槽基址 + 目的布局偏移”；不能把 aTaskBase/bTaskBase 继续加到片上地址。具体公式见下列各存储层偏移。

每个输出元素的 K 区间无重叠、无遗漏地覆盖 `[0,K)`。跨 KL1 只更新输入窗口；当前任务的 cTaskBase 与 L0C 部分和保持不变。进入下一个输出任务时更新三者任务基址，并在获得累加槽后重新初始化部分和。

#### 3.2.2 片上槽基址与轮转

`aL1Base[slot]`、`bL1Base[slot]`、`aL0Base[slot]`、`bL0Base[slot]`、`cL0Base[cSlotId]` 表示各自存储空间内的字节基址，由 Buffer 分配结果给出。固定等跨度分配可以用“区域基址 + slot×实际预留跨度”计算，不能用当前尾窗口的有效字节数替代预留跨度。

本模板维护两个跨任务持续递增的计数器：`l1WindowIndex` 在每个实际 L1 窗口后递增（首块二分的两段分别计数），`l0WindowIndex` 在每个 KL0 窗口后递增。槽号分别为 `l1WindowIndex%2`、`l0WindowIndex%2`；使用前等待对应槽归还。GM 的任务轮次与这两个窗口计数器分别管理。

下面的 `aL1OffsetElems(...)` 等是已选物理布局的元素偏移函数，使用当前窗口的形状、对齐和实际步长；它们是数学记号，不是新增 API。输入布局与累加布局分别绑定 Tin、Tacc。

#### 3.2.3 L1 A/B 偏移

```text
aL1Addr = aL1Base[l1SlotId]
          + aL1OffsetElems(mInTask,kL0Offset+kL0Element)*sizeof(Tin)
bL1Addr = bL1Base[l1SlotId]
          + bL1OffsetElems(kL0Offset+kL0Element,nInTask)*sizeof(Tin)
```

这里使用父 L1 窗口的布局。KL0 子窗口起点体现在源逻辑坐标中；换 KL0 窗口不换当前 L1 槽。gm2l1 写入时，目的地址使用同一布局函数，K 坐标为 `kInL1`。

#### 3.2.4 L0A 偏移

```text
aL0Addr = aL0Base[l0SlotId]
          + aL0OffsetElems(mInTask,kL0Element)*sizeof(Tin)
```

KL0 窗口搬入所选计算槽后，K 局部坐标从 0 开始。源 L1 的 kL0Offset 不再叠加到 L0A 目的地址；更换计算窗口时更新槽号和当前有效长度。

#### 3.2.5 L0B 偏移

```text
bL0Addr = bL0Base[l0SlotId]
          + bL0OffsetElems(kL0Element,nInTask)*sizeof(Tin)
```

L0B 与 L0A 配对使用同一个逻辑 K 区间及轮转槽号，但属于不同存储空间，独立使用各自槽基址、预留跨度和布局函数。

#### 3.2.6 L0C 偏移

```text
cL0Addr = cL0Base[cSlotId]
          + cL0OffsetElems(mInTask,nInTask)*sizeof(Tacc)
```

同一任务的所有 KL1/KL0 窗口更新相同输出元素位置，地址不包含 K 偏移。本贯穿实例选择一个累加槽；切换输出任务时，等待前一任务输出消费完成，按新任务有效 M/N 绑定累加布局并重新初始化数值。写回 GM 使用 §3.2.1 的 cGmAddr。

#### 3.2.7 BF16 实例的源与目的布局

当前 L1 窗口有效 K 长度记为 `kL1Valid`，其中一段 L0 的相对起点为 kL0Offset。源坐标和目的坐标分别是：

| 搬运 | L1 源元素偏移 | L0 目的元素偏移 |
|---|---|---|
| A 的 `(mInTask,kL0Element)`，其中 `0<=kL0Element<kL0Valid` | `NZ(mInTask,kL0Offset+kL0Element;mTaskValid)` | `NZ(mInTask,kL0Element;mTaskValid)` |
| B 的 `(kL0Element,nInTask)` | `NZ(kL0Offset+kL0Element,nInTask;kL1Valid)` | `ZN(kL0Element,nInTask;nTaskValid)` |

L1 源地址按父窗口布局求值，L0 目的地址按目的布局求值。不能把 L1 切片视为新建的紧密子矩阵：第二段 K 的源地址需要 kL0Offset，B 的源列分形步长仍取父窗口的 `ceil(kL1Valid/16)*16*16`。

首任务四个 L1 窗口的 B 列分形步长依次为 `2048,2048,4096,2048` 个元素；后续任务三窗口为 `4096,4096,2048`。本例按当前有效窗口建立数据布局，槽预留跨度保持不变。

#### 3.2.8 类型、转置与尾块的物理窗口

每次实例化先从目标接口能力取得输入/累加布局、各轴对齐、搬运有效范围及输出布局转换能力。对每个存储层分别记录逻辑有效范围、操作实际读写范围和槽预留范围，满足“有效范围包含于操作范围、操作范围包含于预留范围”；GM 读取仍须位于张量合法范围。

| 类型组合 | 输入元素宽度 | 累加元素宽度 | 布局及步长求值 |
|---|---:|---:|---|
| FP16→FP32 | 2 字节 | 4 字节 | 由 FP16 输入布局与 FP32 累加布局分别求值 |
| BF16→FP32 | 2 字节 | 4 字节 | 由 BF16 输入布局与 FP32 累加布局分别求值 |
| INT8→INT32 | 1 字节 | 4 字节 | 由 INT8 输入布局与 INT32 累加布局分别求值 |

轴顺序先决定 GM 元素步长，再由搬运能力决定如何写入片上目的布局。对同一逻辑输入元素，A[M,K]/A[K,M] 及 B[K,N]/B[N,K] 的 GM 地址不同，但在相同计算输入布局中的逻辑位置一致。输出同理：由累加布局取值，再按 C[M,N] 的步长写入。

每段 KL0 的有效范围为 `[0,mTaskValid)×[0,kL0Valid)` 与 `[0,kL0Valid)×[0,nTaskValid)`；计算实际访问范围由目标能力决定：

| 能力条件 | 搬运与 padding 的责任 | 乘加与输出 |
|---|---|---|
| 操作支持有效尺寸，且不读取范围外元素 | gm2l1/l12l0 仅搬有效区域；槽内其余字节不作为输入 | 传有效 M/N/K，只输出有效区域 |
| 操作读取对齐后的物理区域 | 首次物化该区域的搬运阶段填零，或在同一目标槽中显式清零；再发布就绪 | 使用已定义的物理范围计算，仅写回有效 M/N |

需要填零时，A/B 实际被读区域中超出有效 M/N/K 的元素都置为数值零。这样扩展 K 的乘积为零，扩展 M/N 的结果不会影响有效输出。gm2l1 能完成所需填充时由它负责；只有 l12l0 才引入的计算输入 padding 由 l12l0 或其前置同槽清零负责。槽复用时每轮重新建立所有将被读取元素的值，不能依赖首次初始化留下的零。

逻辑 kL0Offset 按有效 K 区间推进，物理补齐长度不参与下一段的逻辑偏移计算。若目标不能在本模板的数据通路和既有槽内完成所需的尾部填充、输入重排或有效输出，就将该候选判为不适用；不能越界读取 GM 或静默丢弃尾部。

### 3.3 贯穿实例的 GM 地址

本节固定 A[M,K]、B[K,N]、C[M,N] 的存储顺序。取核 3 第一轮任务 `taskId=3`，输出起点 `(128,128)`。A/B/C 的 GM 起始元素偏移分别是 `81920`、`128`、`49280`。因此本轮 `aTaskBase=aGmBase+81920*2`、`bTaskBase=bGmBase+128*2`、`cTaskBase=cGmBase+49280*2`。该核首任务四个 L1 窗口 K 起点为 `0,128,256,512`，A 起点为 `81920,82048,82176,82432`，B 起点为 `128,49280,98432,196736`。这些值都按元素计，BF16 字节偏移再乘 2。

每次搬 A 的 GM 行跨度为 640，B/C 的 GM 行跨度为 384；块宽为 128 不意味着 GM 行跨度也变成 128。片上物理地址仍须使用真实 NZ/ZN 布局，以上偏移不用于替代片上布局映射。

### 3.4 贯穿实例的片上布局公式

本例 A/B L1 都采用 NZ，L0A 采用 NZ，L0B 采用 ZN，L0C 采用 NZ。B 的逻辑形状为 (K,N)，先在 L1 形成 NZ 数据，再由 MTE1 转换为 L0B 的 ZN 数据。

布局类型与 C0 规则引用 [所选编程接口映射](../../../patterns.md#编程接口选择)。本例采用 BF16 输入 C0=16、FP32 累加布局显式 C0=16；下文只展开这些选择对应的地址推导。

NZ/ZN 元素偏移与物理跨度使用 [公共布局函数](../../common-contract.md#21-nzzn-布局坐标约定)，本例代入对应父窗口形状。

NZ 中 j 的低 4 位连续，跨列分形的步长为 `Ra*16`；ZN 中 i 的低 4 位连续，跨行分形的步长为 `Sa*16`。两者都不是 GM 的普通行主序。贯穿实例所有片上维度都是 16 的倍数，故本例无额外布局 padding，DAG 的有效字节表也就是对应数据的物理跨度；固定槽预留仍可能大于实际数据。

### 3.5 一个乘积项的完整地址追踪

取输出任务 taskId=3 中 `C[145,149]` 的乘积项 `A[145,403]*B[403,149]`：块内 `mInTask=17,nInTask=21`，第三个 L1 窗口 `kL1Offset=256`，第二个 L0 窗口 `kL0Offset=128`，其内 `kL0Element=19`。

| 位置 | 相对该矩阵或数据槽的元素偏移 | 相应字节偏移 |
|---|---:|---:|
| GM A | `145*640+403 = 93203` | 186406 |
| GM B | `403*384+149 = 154901` | 309802 |
| A L1 | `NZ(17,147;128) = 18707` | 37414 |
| B L1 | `NZ(147,21;256) = 6453` | 12906 |
| L0A | `NZ(17,19;128) = 2323` | 4646 |
| L0B | `ZN(19,21;128) = 2387` | 4774 |
| L0C 的累加结果位置 | `NZ(17,21;128) = 2325` | 9300 |
| GM C 的最终结果位置 | `145*384+149 = 55829` | 111658 |

GM 偏移相对各矩阵起点；片上偏移相对所选数据槽起点，需要再加槽基址。L0C/GM C 两行是该乘积项参与的完整累加结果位置，不能在只算完这一个乘积项时写出。此例同时体现 BF16 与 FP32 的地址宽度差异。

### 3.6 启用 SWAT 尾轮重新切分时的任务地址

任务由 [Tiling §7.4](tiling.md#74-swat-的可选尾轮重新切分) 展开；§2 的 assigned_output_tasks 消费展开后的任务集合。父任务不再执行，骨架仍对每个任务完成完整K。子任务的 mTaskOffset/nTaskOffset 等于父任务起点加拆分偏移，不能用子任务序号乘原ML1/NL1推算。

例中子任务5起点为(64,256)，ND布局下A/B/C任务起点分别为 `64*640=40960`、`256`、`64*384+256=24832` 个元素。GM地址继续使用§3.2.1的步长公式；L1从子任务局部原点开始，源切片保留父窗口布局，L0目的K从0开始，L0C不随K偏移。

固定槽间距保持不变，有效M/N及操作布局按当前任务绑定；只写64×128有效结果，不能仍写原128×128。L1最后读取、L0输入MMAD完成、L0C完整K与输出完成的交接均沿用§9，无需新增跨核同步。

## 4 CopyIn MTE2 逻辑

### 4.1 MTE2 指令选型

gm2l1 根据逻辑窗口、GM 轴顺序和目的布局搬运有效数据，并完成所需布局转换。这里确认 GM→L1 的形状、步长、对齐和尾部能力。

cache hint 绑定具体 A/B GM 访问，采用 [Tiling 策略](tiling.md#8-切分策略微调) 的决策。hint 不替代同步，也不修改地址或计算语义。

转置不是在 GM 偏移上交换两个数就结束：GM 连续方向变化后，搬运的行列长度、源步长、目的布局及计算输入解释必须配套。选定通路需能把逻辑 A(m,k)、B(k,n) 放到乘加操作要求的位置；具体指令是否直接支持该转换由能力引用确定。

### 4.2 UB 数据布局

不涉及。

### 4.3 L1 目的数据布局

根据 GM 轴顺序和 gm2l1 的目标布局，计算 A/B 输入窗口在 L1 的物理跨度与地址；本例布局公式见 [片上布局推导](#34-贯穿实例的片上布局公式)。L1 源窗口与 L0A/L0B 目的数据布局之间的转换见下一节 MTE1 搬运。

## 5 CopyIn MTE1 逻辑

### 5.1 MTE1 搬运职责

MTE1 负责 L1→L0A/L0B。等待当前 L1 窗口就绪及目标 L0 槽可写后，按 KL0 子窗口搬运并转换为计算输入布局；搬运完成后发布 L0 输入就绪。L1 源槽在最后一次 MTE1 读取完成后可复用，L0 目的槽须等对应乘加消费完成后才可复用。

### 5.2 L1 子窗口与 L0A/L0B 目的数据布局

源、目的地址统一使用 [核内各存储层偏移](#32-核内循环间偏移计算) 中的 aL1Addr/bL1Addr 和 aL0Addr/bL0Addr。MTE1 将源 L1 子窗口写入目标 L0 槽的局部窗口，并按目的布局重排。两端各用自己的槽基址与布局；搬运完成后的就绪发布和槽归还按本节职责执行。

## 6 Compute 计算逻辑

### 6.1 算法描述

MMAD 的 A/B 输入地址分别使用 §3.2.4/§3.2.5，累加地址使用 §3.2.6。

每次 MMAD 计算 `[mTaskValid,kL0Valid] × [kL0Valid,nTaskValid]` 并累加到同一 `[mTaskValid,nTaskValid]` 输出。第一次 MMAD 初始化累加，后续追加部分和；只有最后一个 KL0 窗口完成，结果才完整。结果就绪状态与数值初始化分别管理。

FP16/BF16 输入使用 FP32 部分和；INT8 输入使用 INT32 部分和。每段 K 的部分和保持 Tacc 类型，完整 K 结束后按 Tout 输出契约写回。浮点路径的累加顺序、输出舍入，以及整数路径的累加溢出行为，分别按算子数值契约处理。

类型变化不改变首轮初始化、后续累加和末轮结果交接。它会改变输入字节地址、布局分形与 K 对齐，需重新计算物理搬运参数。§3.4 显式 C0=16 的公式属于 BF16 数值例；INT8 输入布局按引用的布局能力取值。

#### 6.1.1 数值契约

遵循 [公共数值契约](../../common-contract.md#1-语义与数值)。本模板采用 [DAG 类型表](dag-buffers.md#11-基础场景与数据流) 中的 FP16/BF16→FP32、INT8→INT32 组合；贯穿实例为 BF16→FP32→BF16。每个输出块的完整 K 只由一个任务归约，直到最终输出才转换为 Tout。

#### 6.1.2 类型对计算输入的要求

用所选布局的元素偏移函数和物理跨度描述每层数据；它们的输入包括 Tin/Tacc、窗口尺寸、对齐和实际步长。先以这些函数推导地址与容量，再代入 §3.4 的 BF16 具体布局。

INT8 的输入元素字节数变小，但输入分形与 K 对齐也可能改变；因此只能复用逻辑分块与所有权推导，不能直接将 BF16 的所有地址除以二。累加和输出各用自身布局与元素宽度。

### 6.2 VF 实现

不涉及。

### 6.3 MMAD 计算顺序

区分三种顺序：核间输出块遍历顺序、同一输出块的 K 累加顺序、一次 MMAD 内部生成 M/N 方向结果单元的顺序。它们分别影响缓存局部性、累加依赖/精度以及与 输出通路 的协同。

最后一种顺序应与输出读取和格式转换的消费顺序匹配。根据目标提供的可选生成顺序及输出路径，选择能够及时交接就绪结果的组合。改变内部生成顺序不改变数学 M/N，也不改变 K 累加顺序。

输出采用细粒度交接时，逐单元检查生产和消费顺序；具体方向与参数映射引用目标能力。

## 7 CopyOut MTE3 逻辑

### 7.1 UB 数据布局

不涉及。

### 7.2 CopyOut 指令

不涉及。

## 8 CopyOut 输出通路 逻辑

### 8.1 L0C 源数据布局与 GM 目的数据布局

L0C 使用 Tacc 的片上布局，GM C 使用 Tout，按 [M,N] 轴顺序连续存储。

### 8.2 输出搬运与类型转换

输出源地址使用 §3.2.6 的 cL0Addr，目的地址使用 §3.2.1 的 cGmAddr。输出通路 按所选输出布局和类型转换写回有效区域；输出 N 方向连续，M 步长为 N。按 [物理窗口规则](#328-类型转置与尾块的物理窗口) 限制写回区域，不将 padding 写入 GM。

## 9 流水同步

### 9.1 Sync 知识

| 交接 | 就绪依赖 | 复用依赖 |
|---|---|---|
| L1 | MTE2 写完 → MTE1 读取 | 最后一次 MTE1 读完 → MTE2 覆盖 |
| L0A/B | MTE1 写完 → MMAD 读取 | MMAD 用完 → MTE1 覆盖 |
| L0C | 最终累加单元就绪 → 输出通路 读取 | 输出通路 消费完成 → 下一输出累加写入 |

遵循 [公共所有权与同步契约](../../common-contract.md#4-所有权与同步)。声明顺序和双 buffer 本身不能证明可见性。由后端托管的依赖不重复加外围事件；也不能删除它们后寄希望于指令自然串行。

### 9.2 范式特殊的同步

遵循 [公共所有权与同步契约](../../common-contract.md#4-所有权与同步)。本模板中 L1 A/B 配对窗口在 MTE2 与 MTE1 之间交接，L0A/B 配对槽按每个 KL0 窗口在 MTE1 与 M 之间交接。L0C 按选定的完整协议交接：单槽逐单元发布和归还，双槽按整块发布和归还。

本例单槽使用 [结果单元交接](../../execution-strategies.md#42-单槽结果单元交接)，不在 K 循环内反复取得整槽锁；首段用 K 坐标初始化，最后一段发布，只有一段 K 时同时完成这两件事。具体同步接口由 [所选编程接口映射](../../../patterns.md#编程接口选择) 维护；§9.3 给出本模板槽状态。

### 9.3 贯穿实例的槽状态演化

初始化时，L1 和 L0 输入槽都可供生产者写，L0C 为可写状态；实际事件方案必须建立这一初始条件，不能第一次就等待一个尚未发生的消费事件。L1/L0 的轮转计数跨输出任务持续递增。

一个输出块的窗口使用如下。表中序号表示逻辑消费次序，不代表各流水完成于同一时刻。

| L1 窗口 | L1 槽 | 完整 K 区间 | 依次使用的 L0 槽 | 累加与交接 |
|---|---:|---|---|---|
| 0 | 0 | [0,128) | 0 | 首条初始化 |
| 1 | 1 | [128,256) | 1 | 保留部分和 |
| 2 | 0 | [256,512) | 0、1 | 保留部分和 |
| 3 | 1 | [512,640) | 0 | 发布最终结果 |

- 启动：先生产 L1 槽 0，再生产 L0 槽 0，首条 MMAD 初始化 C；未就绪的输入不能被消费。
- 稳态：MTE1 读 L1 槽 0 时，MTE2 可写 L1 槽 1；MMAD 读 L0 槽 0 时，MTE1 可写 L0 槽 1。复用 L1 槽 0 搬第三窗口前，必须等第一窗口最后一次 MTE1 读完。复用 L0 槽 0 前，必须等使用该槽的上一次 MMAD 完成。
- 收尾：最后一条 MMAD 发布最终结果，输出侧消费并归还对应存储；没有下一段 K 时不再预取。
- 跨输出块：首块用过 4 个 L1 窗口和 5 个 L0 窗口，下一块从 L1 槽 0、L0 槽 1 开始。下一块首条 MMAD 再初始化数值，但写入前仍须取得 L0C 对应单元的写权限。输入预取可以先进行，不能因输入已准备好就覆盖尚未搬出的 C。

这里的“可重叠”由依赖允许；实际重叠程度还取决于发射位置、bank 和指令耗时。发射归还事件与完成归还是两件事，事件必须排在相应流水的最后一次访问之后。退出时等待仍在途的生产/消费完成，并按后端要求回收事件；不能等待未发射的下一轮事件。

### 9.4 发射顺序、流水等待与收尾

记 `Gx` 为第 x 个 L1 窗口的两条 GM→L1 搬运，`Ly` 为第 y 次 MMAD 的两条 L1→L0 搬运，`My` 为该次 MMAD，`F` 为输出 输出通路。本输出块的一种逻辑提交顺序是：

```text
G0 → L0 → M0 → G1 → L1 → M1 → G2 → L2 → M2 → L3 → M3 → G3 → L4 → M4 → F
```

箭头在这里表示程序提交次序，不表示上一条在所有流水上完成才提交下一条。数据等待由目标流水的同步指令表达，不能把它实现成每步全流水等待。

| 发射位置 | 执行前等待 | 归还条件 |
|---|---|---|
| G0、G1 | 各自 L1 阶段可写 | L0、L1 分别读取后归还 |
| G2 | L0 读完 L1 阶段 0 | L2、L3 都读完后归还 |
| G3 | L1 读完 L1 阶段 1 | L4 读完后归还 |
| Ly | 对应 G 完成，L0 槽前次 MMAD 已读完 | My 消费后归还 |
| My | Ly 完成且累加目标可写 | 最终 M4 按协议发布 |
| F | 对应结果范围就绪 | 输出读完归还 |

M0…M4 对同一 L0C 的部分和更新依次进行；不能为了流水重叠重排这条累加依赖。初态将输入槽交给生产者，首轮不等待不存在的上一轮消费；后续复用均由消费完成建立依赖。

最后一个输出块之后不发射新的 G/L/M。完成条件是已发射的搬运、计算和 F 全部结束，且所使用的同步资源已按后端契约释放；外层 Kernel 与运行时负责落实该完成条件。不能等待未发射的下一轮任务。

### 9.5 并发访问的存储竞争

G1 写阶段 1 与 L0 读阶段 0、G2 写阶段 0 与 L1 读阶段 1、G3 写阶段 1 与 L2/L3 读阶段 0，需结合实际时间区间检查。分区地址由 [L1 配对分区](../../execution-strategies.md#2-l1-成对分区双缓冲) 给出；bank 映射取自硬件事实源。

## 10 Kernel 产出校验

本节的任务、地址和槽状态枚举用于设计/实现一致性检查，证据承接见 [公共契约 §5.2](../../common-contract.md#52-设计一致性与证据选择)。将模型与真实 Kernel 源码、编译产物及所选 adapter 的依赖绑定对照，不自动要求额外的 task/slot 事件接口或独立取证能力。已选双缓冲和完整归约承诺保持，设备 Golden 仍覆盖多块、尾块和重复调用。

### 10.1 贯穿实例的物理布局校验

校验模型采用非恒定整数输入 `A[m,k]=(m+3*k)%7-3`、`B[k,n]=(2*k+n)%5-2`，独立 Golden 直接执行完整矩阵乘；执行模型按以下顺序展开，避免只检验数学切 K：

1. 按 taskId=coreId+taskRound*coreCount 枚举八个任务，按 SWAT 尾轮分派求父块和子块坐标，并维护跨任务的 L1/L0 轮转计数。
2. 每个 L1 窗口：从 GM 取值，按 NZ 元素偏移写入一维 A/B 槽。
3. 每个 L0 窗口：按父 L1 布局取值，分别按 NZ/ZN 写入一维 L0A/B 槽。
4. 从 L0 槽按目的布局读回矩阵，执行该段乘加。
5. 完整 K 结束：按 NZ 写入一维 L0C，再映射回 GM 的 ND 输出。
6. 对照完整 Golden，检查所有输出恰好写一次。

同时检查各 NZ/ZN 映射在本例尺寸上为槽内无重叠、无遗漏的双射，并复核 [完整地址追踪](#35-一个乘积项的完整地址追踪) 的元素偏移。该模型检查数据布局和分段计算，不模拟 MMAD 的硬件微执行顺序、BF16 输出舍入或异步事件时序。

### 10.2 调度与边界校验

先按贯穿实例枚举 `taskId=coreId+taskRound*coreCount`，经 SWAT 尾轮映射后检查八个任务的覆盖和 §3.3 的扁平地址；再按 §9.3 枚举五次 MMAD 的 K 区间、输入槽和首轮初始化与末轮发布。四个核的第二个任务用于检查跨输出块的槽轮转及尾轮有效尺寸；额外用不满末轮的配置检查无后继任务时的收尾。资源账本按 DAG 的物理跨度前提复核。

槽检查还展开到同核同槽再次使用，核对计数跨任务延续、实际独立分配、固定间距及最后消费者到下一次生产的完成依赖。设备测试用实际选块配置产生多输出任务和 M/N 尾块，覆盖单段/多段 K；重复调用变更输入并逐次比较 Golden，形状变化遵守入口契约。枚举能证明模型的地址与覆盖，不能模拟实际异步完成，Host 按预期拼装的序列不能充当设备证据。

换到另一槽与回到同槽分别派生覆盖，遵循 [候选合法性](../../tiling-selection.md#2-候选生成与合法性)。整块交接的累加池每输出任务推进一次，深度为 d 时至少需要同核 d+1 个有效输出任务触发首次回绕；K 窗口增加只推进对应输入池，不增加该累加池的输出任务计数。用实际核数和尾轮展开后的任务分布核对，不能只数全局任务或跨 launch 累计；受尺寸限制无法触发时依选块规则重新选择或返回设计缺口，不取消已选槽复用保证。

以下例子用非恒定值检查 M/N 地址、M/N 尾块及两级 K 分段。逻辑 tile 仅用于枚举，不是可下发的硬件配置。

```python
M, N, K = 3, 5, 7
ML1, NL1, KL1, KL0 = 2, 3, 4, 3
A = [[m + k - 2 for k in range(K)] for m in range(M)]
B = [[2*k - n for n in range(N)] for k in range(K)]
golden = [[sum(A[m][k]*B[k][n] for k in range(K))
           for n in range(N)] for m in range(M)]
C = [[0]*N for _ in range(M)]
coverage = [[0]*N for _ in range(M)]
for mTaskOffset in range(0, M, ML1):
    for nTaskOffset in range(0, N, NL1):
        for m in range(mTaskOffset, min(mTaskOffset+ML1, M)):
            for n in range(nTaskOffset, min(nTaskOffset+NL1, N)):
                acc = 0
                visited = []
                for kL1Offset in range(0, K, KL1):
                    kL1End = min(kL1Offset+KL1, K)
                    for kL0Offset in range(0, kL1End-kL1Offset, KL0):
                        for k in range(kL1Offset+kL0Offset, min(kL1Offset+kL0Offset+KL0, kL1End)):
                            acc += A[m][k]*B[k][n]
                            visited.append(k)
                assert visited == list(range(K))
                C[m][n] = acc
                coverage[m][n] += 1
assert C == golden
assert all(count == 1 for row in coverage for count in row)
assert C == [[98, 91, 84, 77, 70],
             [140, 126, 112, 98, 84],
             [182, 161, 140, 119, 98]]
```

此例只验证逻辑分段和覆盖，不模拟 bank、异步槽复用或输出舍入。设备验收还须覆盖实际合法 tile 的满块/尾块、单/多 K 窗口、多轮输出任务、各槽重复复用，并检查真实参数、输出精度和越界。bank、MMAD 顺序及 cache hint 的收益要用匹配配置的性能证据验证。

### 10.3 类型与数值校验

| 检查项 | 覆盖方式 |
|---|---|
| 类型参数 | 分别实例化 FP16/FP32、BF16/FP32、INT8/INT32，独立确定 Tout |
| 字节与布局 | 按每层实际类型重算偏移、物理跨度和 K 对齐；检查 padding 和有效区 |
| 首末段 | 单段 K 同时执行初始化和最终发布；多段 K 只在首段初始化、末段发布 |
| 浮点数值 | 输入先转换为 Tin，覆盖正负值、消去及输出舍入边界，按数值契约比较 |
| 整数数值 | 更宽整数 Golden；覆盖负值、零和接近累加边界的输入，检查中间部分和 |
| 输出转换 | Tacc 与 Tout 相同及不同的合法组合，确认只在最终写回阶段转换 |

### 10.4 输入输出轴顺序校验

固定逻辑 A、B 和 Golden，独立枚举 A/B 的两种轴顺序，共 4 种组合，C 固定 [M,N]。每组先按轴顺序打包输入，再用 §3.1 的步长读取，按逻辑 M/N/K 分块计算，最后按 C 步长写回；将物理输出还原为逻辑 C 后与同一 Golden 比较。

取 M=3、N=5、K=7，使三个长度不同，覆盖 M/N 尾块和 K 多段。输入采用非对称、非恒定值，避免方阵或相同元素掩盖交换轴错误。检查每个输出物理位置恰好写一次。此校验覆盖语义和地址映射；实例化时另外核对选定通路的转置搬运、输出转换和对齐能力。

## 11 Bias[N] 组合扩展的地址与计算

原贯穿实例保持 hasBias=false；以下为同一形状接入 Bias[N] 组合扩展。共享向量按最终子任务 nTaskOffset 取 N 区域，并广播到 M 轴的每个输出行。

GM 元素偏移为 `nTaskOffset + nLocal`，按 Bias[N] 的真实 strideN 换算地址；同一个向量区间供当前输出块的所有 M 行使用。

本核每个有效输出子块独立首段注入；不能把每核首任务与每个输出首 K 段混淆。

使用 [公共 Bias[N] 组合扩展融合骨架](../../execution-strategies.md#8-biasn-组合扩展的首段融合策略) 替换原首段零初始化：有组合扩展时首段以 bias 启动，后续累加；无组合扩展时保留原骨架。L0C 发布与输出协议保持所选完整路线。

在原实例输入上取 `bias[n]=n+1`，Golden 为原收缩结果加 `bias[n]` 后按 Tout 转换。检查每个输出完整 K 只应用一次组合扩展、尾列不越界；用 N 偏移错误、每个 K 段重复应用及无组合扩展分支误读 bias 作为负例。K=0 走公共预处理规定的输出路径。
