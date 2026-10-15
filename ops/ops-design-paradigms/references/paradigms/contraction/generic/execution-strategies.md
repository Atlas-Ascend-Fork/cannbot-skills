# Contraction 窗口与缓冲执行策略

本页维护可被多个模板引用的完整执行规则。模板声明选择、参数及实例结果；[公共契约](common-contract.md#4-所有权与同步) 维护共同依赖，具体同步指令由实现映射表达。

## 1 首块 L1 窗口二分

适用输入为：每核首个输出任务、共同 K 窗口、完整 K 归约，以及能够独立搬入两个子窗口的输入布局。令 `windowK=min(K,KL1)`。当 `floor(windowK/KL0)>=2` 时，将该任务原首窗口拆为两段，其余窗口保持原推进方式。

```text
firstLength = align_up(windowK // 2, kMoveAlign)
secondLength = windowK - firstLength
```

要求两段为正并满足搬运能力。先处理 `[0,firstLength)`，再处理 `[firstLength,windowK)`，随后处理 `[windowK,2*windowK)` 等剩余区间；最后截到 K。每个子窗口独立完成生产、消费和归还，L1 计数各递增一次。每个窗口内部按 KL0 重新切分；不能假定总 MMAD 条数永远不变，对齐后的边界可能改变条数。

二分策略需要同时评价首次输入等待的减少与新增搬运、事件和槽轮转开销，不能仅凭更早发出首条 MMAD 判定收益。

首块身份由每核任务循环确定，不能用全局 taskId=0 代替。之后的任务不拆首窗口，且两级槽计数继续轮转。数值初始化由全局 K 起点是否为零决定，完整结果由最后有效 K 位置决定。

例：K=640、KL1=256、KL0=128、kMoveAlign=16。首任务窗口长为 128、128、256、128，内部 MMAD 条数为 1、1、2、1；之后任务为 256、256、128，条数为 2、2、1。L1 双槽下首任务结束计数为 4，下一任务从槽 0 开始；L0 双槽计数为 5，下一任务从槽 1 开始。

## 2 L1 成对分区双缓冲

每个阶段同时容纳当前 A/B 窗口，两个阶段交替生产和消费。输入为两个合法分区的基址 `stageBase[0/1]`、跨度 `stageSpan` 和 A/B 单槽预留字节 `aSlotBytes/bSlotBytes`。

```text
aL1Base[s] = stageBase[s]
bL1Base[s] = stageBase[s] + aSlotBytes
requiredStageBytes = aSlotBytes + bSlotBytes
```

`requiredStageBytes<=stageSpan`，两个分区不重叠且落在合法地址范围内。分区间距由实际存储分配提供，不由当前有效窗口长度推算。每个阶段分别等待 MTE1 最后读取完成后再由 MTE2 覆盖；A/B 只有最后消费者一致时才共用阶段交接。

每次 L1 槽复用周期只发布一次就绪，消费侧取得一次读权限，并持有到当前窗口的全部 MTE1 子窗口读取完成。不能将一次发布对应的消耗型事件等待放到每个子窗口中重复执行。

窗口二分和尾块改变有效数据量，不改变槽预留或分区基址。bank 分析使用真实分区地址和并发区间，不能仅凭“两个分区”判定无冲突。

## 3 L0 输入双缓冲

L0A/B 各两槽，槽间距由各自存储分配给出。一次“输入准备批次”定义为：写入一对槽，并由指定的一组 MMAD 消费这些输入。生产侧写前等待旧批次全部消费者完成，写完发布就绪；计算侧等待就绪，全部消费者读完后归还。A/B 共用槽编号的前提是对应同一准备批次且归还覆盖两者最后消费者。

计数按输入准备批次递增一次，不能机械按 MMAD 条数推进。模板明确每批消费者：basic、保留轴分块和驻留模板每批一次 MMAD；IterBatch 每批包含当前有效保留坐标子组的全部 MMAD；MergeBatch 每批一次扩大矩阵 MMAD。计数跨 L1 窗口和输出任务保持连续。

校验一个消费者和多个消费者两种情况：在最后消费者完成前重新写入同槽应被依赖检查拒绝。槽数、每批有效数据及消费者数量分别记录，尾组减少消费者不改变槽间距。

## 4 L0C 选择与结果交接

### 4.1 选择规则

按常规最大输出块的实际累加布局跨度计算 `cSlotBytes`。若两个独立槽均满足容量、起址和输出能力，采用双槽整块交接；否则在单槽容量及细粒度交接能力成立时采用单槽结果单元交接。若单槽也不合法，回退分块候选。

本节给出单槽结果单元/双槽整块这一具体选择策略；槽数本身不唯一决定交接方式。其他完整策略可对双槽也使用结果单元交接，须保证每槽、每结果单元的发布与复用配对，具体版本绑定见 [所选编程接口映射](../patterns.md#编程接口选择)。

选择在 Host 完成，并将槽数与交接方式作为一份配置传给执行端。尾轮变小不临时改变槽数；每核使用同一配置，固定槽间距来自分配结果。

结果槽按每核的 `resultAllocationIndex` 轮转：初值为 0，每取得一个新的完整 K 归约结果区域递增一次，跨调度任务保持连续；同一区域的各 K 段不递增。它不是全局发射编号。basic 和保留轴分块模板每个有效输出任务对应一次分配；IterBatch 每个保留坐标子组的每个 M/N 输出块对应一次分配，同槽内多个结果共同归还；驻留模板每个条带内输出块对应一次分配；MergeBatch 每个扩大矩阵输出任务对应一次分配。模板已有等价计数时直接复用，例如 basic 的每核 `outputTaskIndex`，不增加重复计数器。

### 4.2 单槽结果单元交接

所有 K 段更新同一累加槽。中间段保留部分和，最后段按结果单元生成顺序发布最终值，输出侧仅消费已就绪单元。输出读取完成后该单元才允许下一任务写入。

这里取得写权限和发布就绪可以由单元交接机制隐式表达，不要求额外插入整槽等待；否则会消除最后 MMAD 与输出、输出与下次计算之间本可存在的重叠。计算与输出必须采用相同的交接粒度和顺序。具体机制表达见 [所选编程接口映射](../patterns.md#编程接口选择)。

### 4.3 双槽整块交接

任务按 `resultAllocationIndex%2` 选择 L0C 槽。首次累加前等待该槽上次输出读取完成；持有整个 K 归约；最后 MMAD 完成后发布整块，输出读完归还。下一任务使用另一槽，因此可与前一任务输出重叠。

整块事件与细粒度结果单元交接是两套完整方案；不能仅改变槽数而保持不匹配的发布和复用条件。

### 4.4 双槽结果单元交接

输出任务按 `resultAllocationIndex%2` 选择固定 L0C 半区，每个任务的所有 K 段写同一半区。中间段保留部分和，最后段按结果单元生成顺序发布；输出按相同顺序消费，全部有效结果消费后该半区才可安全复用。所选机制必须保证两任务后回绕到同一半区时的读取完成依赖。

这条路线同时确定半区地址轮转和结果单元协议，不从双槽数量推导整块事件。MMAD 与输出的结果单元标志始终配对，不再叠加另一套整槽交接；初态、结果单元顺序及结束回收按目标接口绑定。跨三个输出任务验证槽0→槽1→槽0，覆盖单段及多段 K。

## 5 策略联合校验

先决定合法槽布局，再展开窗口序列，最后按依赖发射。首块拆分改变 L1 事件次数、源布局尺寸和下一任务槽号；L0C 双槽改变跨任务输出依赖。均需重新核算首块、稳态、尾轮和收尾，不能只检查单个完整矩形。

校验输出覆盖、K 覆盖、源窗口父布局、固定槽跨度及每次生产前的复用条件。跨两任务检查 L1/L0/L0C 槽号，并展开到同槽回绕；用“拆首块后仍按原计数选槽”和“有效尺寸替换固定槽间距”的错误变体检查模型能否识别差异。

这里的槽序列和错误变体是设计/实现一致性检查，按 [公共证据选择](common-contract.md#52-设计一致性与证据选择) 对照真实源码、分配和生成的依赖；不要求算子把序列作为事件输出。已选的两份存储、窗口粒度、首块二分和最后消费者完成条件继续作为强制约束，不能仅以 Golden 通过、depth=2 或 Host 枚举结果放行。设备 Golden 仍覆盖多块、尾块、K 分段及重复调用；真实运行时观测另按其契约选择证据。

## 6 等分容量分区

本策略将 L1 分成两个容量相等的阶段，各阶段容纳一对 A/B 窗口；L0A/B 各分成两槽。设目标提供可用容量 `L1Bytes/L0ABytes/L0BBytes/L0CBytes`，各地址为对应存储空间内的字节偏移：

```text
l1StageStride = L1Bytes / 2
aL1Base[s] = s * l1StageStride
bL1Base[s] = aL1Base[s] + aSlotBytes
aL0Base[s] = s * (L0ABytes / 2)
bL0Base[s] = s * (L0BBytes / 2)
cL0Base[s] = s * (L0CBytes / 2)  // 双 L0C；单槽使用槽0
```

要求各等分边界满足目标对齐，`aSlotBytes+bSlotBytes<=l1StageStride`，每份 L0 数据跨度不超过所分半区；容量由已有平台接口取得。槽间距来自分区，不由有效 tile 字节数替代。保留相同 tiling 做受控对照时也重新绑定目标地址；自然 Host 选择则同时重新求解分块和槽数。其他分配策略使用其实际基址表，不套用本节等分公式。

## 7 双层 K 窗口与附加输入

本策略适用于输入准备同时消费 payload 与按 K 索引的附加输入，且一份附加输入覆盖多个 payload 窗口的情形。输入为任务 K 范围、payload 窗口 `KL1A/KL1B`、计算窗口 `KL0`、附加输入窗口 `auxKL1` 及各类槽数与布局。窗口长度均以原始 K 元素计，附加输入的编码与组粒度由其数值契约给出。

任务 K 起点必须与附加输入的分组边界一致；若从既有矩阵的中间组起步，应保留原组坐标并采用另一套边界映射，不能套用相对零点公式。

本策略选择同原点、完整嵌套的窗口：`auxKL1>=max(KL1A,KL1B)`，且 `auxKL1 % KL1A == 0`、`auxKL1 % KL1B == 0`；窗口边界还须满足附加输入的组粒度及搬运对齐。每个计算窗口必须完全落在当前 payload 与附加输入覆盖内，必要时在边界截短。尾窗裁到有效 K，物理补齐按各自布局处理。

```text
auxWindowIndex = kTaskRelative // auxKL1
auxWindowStart = auxWindowIndex * auxKL1
kInAuxWindow = kTaskRelative - auxWindowStart
auxSlotId = auxWindowCounter % auxStageCount
```

GM 索引使用任务 K 基址加窗口起点，片上索引使用窗口内相对坐标及父布局；附加输入的组索引再按其组粒度换算。附加输入计数仅在换附加输入窗口时递增，跨任务继续轮转。它不能由 payload 窗口计数替代。

每类输入单独记录逻辑角色、物理槽及字节跨度；总需求是各类 `stageCount*slotSpanBytes` 之和。若多个角色嵌入同一分配区，按组合布局实际跨度计一次。驻留输入另按其实际覆盖范围核算，不因其 payload 驻留而推断附加输入槽数。

```text
for aux_window in windows(K, auxKL1):
    aux_slot = acquire_next_aux_slot_for_write()
    load_and_publish_aux(aux_window, aux_slot)
    for preparation in input_preparations_inside(aux_window):
        acquire_payload_and_aux_ready(preparation, aux_slot)
        prepare_l0_inputs(preparation, aux_slot)
    release_aux_after_last_input_preparation(aux_slot)
```

附加输入生产完成后，所有依赖它的输入准备才能执行；该窗口最后一次读取完成后才允许覆盖。独立空槽可预取下一份 payload 或附加输入，伪码只规定依赖，不要求流水串行等待。L0 输入及结果交接沿用所选模板。

校验每个有效 K 的窗口与组索引同直接绝对索引一致，覆盖等长窗口、多 payload 窗口复用、尾窗及跨任务回绕。负例分别按每个 payload 窗口轮转附加输入槽、在首个 payload 窗口后提前归还，检查地址错配和未完成消费者。

## 8 Bias[N] 组合扩展的首段融合策略

本策略是 Elementwise/Broadcast Bias[N] 组合扩展的一种执行融合方式，适用于完整 K 的输出所有者及支持以 bias 启动累加的后端。逻辑组合语义为 `acc[...,m,n]+bias[n]`；模板可在每个输出区域的首个 K 窗口将 `bias[n]` 融入累加初始化一次，后续窗口仅累加。转换及舍入按组合扩展数值契约与目标能力绑定。

完整通路为 Bias[N] GM→L1→BiasTable→首段累加。按当前 N 子区取向量切片；同一切片广播到该输出块的 M 轴及其他保留轴坐标。L1 bias 槽由搬运生产，在最后一次 L1→BiasTable 读取完成后可复用；BiasTable 槽在最后一条引用它的计算完成后可复用，不必持有到完整 K 或输出结束。其生命周期不由 A/B 槽编号或结果槽编号代替。

```text
if constexpr has_bias:
    bias_region = bias_n_vector[nTaskOffset:nTaskOffset+nTaskValid]
    bias_l1 = acquire_bias_l1_for_write()
    load_bias_and_publish(bias_region, bias_l1)
    acquire_bias_l1_for_read(bias_l1)
    bias_slot = acquire_bias_table_for_write()
    convert_copy_and_publish_bias(bias_l1, bias_slot)
    release_bias_l1_after_last_copy(bias_l1)
for k_window in reduction_windows:
    ab_slot = prepare_and_acquire_ab(k_window)
    first = (k_window.logical_offset == 0)
    if constexpr has_bias:
        if first:
            acquire_bias_for_compute(bias_slot)
            compute_with_bias_seed(ab_slot, c_slot, bias_slot)
            release_bias_after_last_seed(bias_slot)
        else:
            compute_accumulate(ab_slot, c_slot)
    else:
        compute_zero_init_or_accumulate(ab_slot, c_slot, first)
    release_ab_after_last_compute(ab_slot)
```

此片段嵌入模板已取得结果写权限的完整 K 循环；保留原末段发布与输出协议。每个输出结果子区分别应用首段判定，不能用整个任务仅一次的标志替代。伪操作表达流水依赖，不要求主线程阻塞等待。

单 bias L1 槽和单 BiasTable 槽足以顺序执行；预取下一 bias 或多个结果子区并发时，按实际生命周期选择槽数。需求为 `biasL1Stages*span(biasL1Layout)` 与 `biasStages*span(biasComputeLayout)`，分别记入 L1 与 BiasTable；后者按转换后的元素类型计算。新区域与原 A/B 分配互不重叠，增加 bias 后重新检查容量与分块，不能仅在旧分区末尾无条件追加。

各模板的可选组合扩展实例采用 Bias[N] 向量视图及目标支持的 bias/Tacc 类型转换；L1/BiasTable 各一槽，按当前有效 N 搬入并建立目标要求的 padding，N stride 按实际输入布局绑定。资源跨度由目标布局推导，不硬编码硬件对齐；需要更高并行度时按相同所有权规则选择多槽。
