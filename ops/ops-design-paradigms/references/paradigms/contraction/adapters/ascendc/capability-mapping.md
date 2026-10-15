# Contraction AscendC 能力与接口映射

本文件将模板产出映射到 AscendC 基础 API。共享设计输入见 [公共适配契约](../../generic/common-contract.md#7-设计与接口适配契约)，构建绑定见 [Kernel 入口](kernel-entrance.md)，调度、Buffer 数量及窗口策略仍由场景模板定义。此处采用 `AscendC` 命名空间，基础接口由 `kernel_operator.h` 或目标版本对应的公共头文件提供。

下表指向本文件的能力绑定章节；支持条件须结合目标架构、SDK 版本、类型及布局核对，文档中的 API 名称不表示全部组合已支持或已验证。

| 能力要求 | 本 adapter 的绑定依据 |
|---|---|
| Tin/Tacc/Tout 组合 | §1 类型与布局、§2 乘加及结果转换 |
| LayoutX、SpanX 与对齐 | §1 的布局和地址单位；公共契约与模板给出跨度推导 |
| 各层容量与地址空间 | §1 物理存储绑定及已有硬件事实源 |
| 搬运与计算通路 | §2 的目标版本重载、操作 trait 与有效范围 |
| 同步与结果交接 | §3 的依赖映射及所选输出协议 |
| 可选 Bias[N] | 本文件的 Bias 章节；仅启用组合扩展时核对 |

## 1 张量与数据布局

A/B 的 ND/DN 各自来自 [TilingKey 编译期布局维度](../../generic/template-overview.md#3-tilingkey)，实现实例据此选择 GM 地址解释及搬运路径；逻辑形状仍为 A(M,K)、B(K,N)，不新增重复的运行时方向标志。

| 范式对象 | AscendC 表达 | 应保留的约束 |
|---|---|---|
| GM 地址与类型 | `GlobalTensor<T>`、`SetGlobalBuffer` | 使用当前任务的基址和真实 GM 步长；张量偏移按接口的元素单位求值 |
| 片上槽 | `LocalTensor<T>` | 分别绑定 L1、L0A、L0B、L0C 的合法存储位置和已分配容量 |
| 资源分配 | 目标支持的静态局部存储 | 按 DAG 的物理槽数与跨度分配，并显式管理槽所有权 |
| 数据布局 | 搬运、加载及输出参数 | `LocalTensor` 不替代布局推导；参数必须描述真实物理排列、分形步长及有效范围 |
| 子窗口 | 张量偏移与操作参数 | L1 子窗口使用父窗口步长，L0 目的 K 坐标从局部零点开始 |

片上地址和完整偏移公式引用 [基础 Kernel 地址计算](../../generic/templates/basic/kernel-template.md#3-地址偏移计算)。字节槽基址、元素偏移和指令字段单位分别换算；LocalTensor 视图不负责自动推导搬运和计算接口的参数。

本映射不使用 `TPipe`、`TBuf` 接口。静态局部存储按所选 SDK 的公共接口绑定为 `LocalTensor`；各存储层的槽基址、容量和跨度来自 DAG/Buffer 与 Tiling 的产出。共享存储空间的槽区必须互不重叠，数据访问满足目标对齐约束，槽的就绪、消费和复用由第 3 节的显式流水同步管理。

## 2 搬运与计算

| 模板操作 | AscendC 基础 API | 参数承接内容 |
|---|---|---|
| gm2l1 | `DataCopy` 的 GM→L1 重载；ND→NZ 路径使用 `Nd2NzParams` | 当前有效矩阵范围、GM 源步长、L1 目的分形步长和矩阵间隔 |
| l12l0 | `LoadData` 的目标版本适用重载 | L1 子窗口起点、重复或分形步进、源与目的跨度及转置控制；A/B 分别选型 |
| mmad | `Mmad`、`MmadParams` | 当前有效 M/N/K、累加类型、初始化及结果交接参数 |
| l0c2gm | `Fixpipe` 及目标版本适用的参数结构 | L0C 源布局、GM 目的步长、有效输出范围和 Tacc→Tout 转换 |

API 重载和参数结构不能跨架构照搬。具体规则复用 [DataCopy 指南](../../../../../../../ops/ascendc-api-best-practices/references/api-datacopy.md)、[LoadData 指南](../../../../../../../ops/ascendc-api-best-practices/references/api-loaddata.md) 与 [硬件能力](../../../../../../../ops/npu-arch/references/npu-arch-guide.md)。`Fixpipe` 的参数类型随版本和格式变化，应从同版本 `basic_api/kernel_operator_fixpipe_intf.h` 及对应结构声明取值。

### 2.1 输入搬运与转置

基础无转置实例将 GM 的 A[M,K]、B[K,N] 搬入所选 L1 布局，再分别加载到计算要求的 L0A/L0B 布局。源张量行跨度取完整 GM 矩阵步长，不取 ML1/NL1。改变 A/B 的存储轴顺序时，需同时调整 GM 取址、搬运参数和 L1→L0 转换；不能把一个转置开关当作全部布局映射。

L1 源窗口、L0 目的窗口的有效范围与物理范围分别记录，字段单位按目标重载核对。GM→L1 的格式转换若提供所需 padding，由其建立物理尾部数值；不能从 UB 的 DataCopyPad 能力推定所有 Cube 通路都具有相同边界行为。完整尾块处理遵循 [物理窗口规则](../../generic/templates/basic/kernel-template.md#328-类型转置与尾块的物理窗口)。

### 2.2 累加与结果输出

基础 API 使用 `AscendC::Mmad(cL0, aL0, bL0, params)` 这一类张量重载。将 `params.m/n/k` 绑定当前窗口有效尺寸，按输出任务的首段 K 设置 `cmatrixInitVal`；后续段保留部分和。初始化条件由 `kL1Offset == 0 && kL0Offset == 0` 求得，不与锁生命周期混合。FP16/BF16→FP32、INT8→INT32 分别绑定正确的 LocalTensor 类型与目标支持的计算配置。

`Fixpipe` 在完整 K 结束后读取累加结果。源步长来自 L0C 物理布局，目的步长来自 C 的 GM 存储轴顺序，输出范围仅包含有效 M/N。C 按 [M,N] 写回，类型转换选择支持的重载和配置；不支持当前组合时回到候选选择，不能静默套用普通行主序写回参数。

调用形式与字段依据：[官方 Cube 计算指南](https://asc.gitcode.com/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/cube_matrix_computation.html)。接口签名核对同版本 `basic_api/kernel_operator_data_copy_intf.h`、`kernel_operator_load_data_intf.h`、`kernel_operator_mm_intf.h`、`kernel_operator_fixpipe_intf.h`；这些入口提供事实源，本文不复制硬件规格表。

## 3 同步与资源所有权

通用所有权与交接边界见 [公共所有权与同步契约](../../generic/common-contract.md#4-所有权与同步)。本节只维护 AscendC 接口映射。

### 3.1 按槽映射流水同步

下述 L0C Mutex 行用于整块交接方案；基础单槽细粒度方案使用 §4.2，不能将整槽锁叠加在该方案外层。

稳态窗口循环按依赖同步，不能在每次搬运或乘加后使用 `PipeBarrier<PIPE_ALL>()`；全流水排空会阻断不相关流水的重叠。Kernel 退出所需的一次收尾与稳态交接分别处理。

在提供 Mutex 的目标版本上，使用配套流水锁表达槽所有权。当前核对版本的实际符号为 `AscendC::Mutex::Lock<pipe>(id)` / `Unlock<pipe>(id)`；接口拼写以目标版本的基础 API 头文件为准。接口约束引用 [Mutex::Lock](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/Lock.html)。

| 资源 | 生产阶段 | 消费阶段 | 释放边界 |
|---|---|---|---|
| 每个 L1 A/B 配对槽 | MTE2 写入前 lock，写入后 unlock | MTE1 首次读取前 lock | 最后一个 KL0 子窗口搬出后由 MTE1 unlock |
| 每个 L0A/L0B 配对槽 | MTE1 写入前 lock，写入后 unlock | M 流水读取前 lock | MMAD 读取结束后由 M unlock |
| L0C 整块交接槽 | 完整 K 循环前 M lock，循环后 M unlock | 完整 K 结束后 FIX 读取前 lock | 输出消费结束后 FIX unlock |

双槽使用不同锁 ID，L1、L0A/B、L0C 也分别分配 ID；同一资源的生产者和消费者使用相同 ID。Lock/Unlock 按代码提交顺序配对，同 ID 不嵌套；不同资源同时持有锁时，获取顺序不得形成循环等待。槽计数跨任务延续，下一次同槽生产需等待前次消费释放。锁 ID 的分配和可用范围遵守目标接口资源管理契约。

L0C 锁覆盖当前任务的完整 K 循环，循环中不发射 l0c2gm，所以不逐次获取和释放 L0C 锁。循环内每次 MMAD 仍由对应 L0A/B 槽的 M 流水 Lock/Unlock 包围；[Unlock](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/Unlock.html) 等待该流水前序指令完成，当前方案已具有逐段 MMAD 的完成边界，不额外叠加 `PipeBarrier<PIPE_M>()`。若后续合并或移动输入槽的 Unlock，必须重新核对部分和依赖，不能照搬这个结论。

L0C 的 M Unlock 与 FIX Lock 位于完整 K 循环之后；FIX Unlock 归还累加槽，供下一任务的 M Lock 获取。首段数值初始化由 `kL1Offset == 0 && kL0Offset == 0` 决定，不需要另维护 `firstMmad` 状态，也不参与加锁条件。

采用 Mutex 完整交接结果时，不再叠加一套未配对的 UnitFlag 协议。合法的依赖只提供并行机会，实际重叠还取决于指令发射顺序。

### 3.2 目标能力与同步机制选择

Mutex 方案仅用于目标接口明确支持的架构。其他目标按 [流水管理](../../../../../../../ops/ascendc-api-best-practices/references/api-pipeline.md) 使用匹配的 `SetFlag/WaitFlag` 表达就绪与归还，事件 ID 和初态遵守该实现的分配协议。不同机制承接相同资源依赖，但不能在同一资源上无配对地混用。

## 4 实例化与校验

从相同模板产出绑定 SWAT 任务坐标、每核多任务循环、两级窗口和物理槽；接口映射不重新选择块尺寸或调度顺序。入口及 Host/Device 的参数定义保持同源，AIC/MIX 路由与所选执行路径一致。

复用 [基础 Kernel 产出校验](../../generic/templates/basic/kernel-template.md#10-kernel-产出校验)：检查 GM 与片上地址、类型及四种输入存储轴顺序的候选适用性，再验证单段/多段 K、尾块、跨任务复用及收尾。具体工程需对所选 API 重载完成编译和设备 Golden；文档映射不代表各组合已设备验证。

基础调度的源码标识 `BuiltInAswtScheduler` / `BlockSchedulerAswtBuiltIn` 对应本文的 SWAT 策略。源码标识保留原拼写，正文统一使用 SWAT。绑定时同时传递尾轮拆分份数、边缘再均衡尺寸及 launch 核数。

### 4.1 基础策略参数绑定

公共选择算法见 [选块 §7](../../generic/tiling-selection.md#7-基础双侧流式策略的参数派生)，执行策略见 [窗口与缓冲](../../generic/execution-strategies.md)。本节仅将已选策略参数绑定到 Host/Device 字段。

具体常数、枚举顺序与阈值统一见 [选块参数配置](../../generic/tiling-selection.md#8-基础双侧流式策略参数配置)。

字段映射：ML0/NL0/KL0 对应 baseM/baseN/baseK；ML1/NL1 为形状对齐后与 baseM*stepM/baseN*stepN 的较小值，基础 stepM=stepN=1；KL1 对应 `baseK*min(stepKa,stepKb,maxKSteps)`，其中 maxKSteps 取所选策略配置；cSlotCount 对应 l0cDB；L1 阶段数对应 l1BufferNum。mTailCnt/nTailCnt、边缘块参数与 usedCoreNum 交给调度；缓存选择写入 l2CacheDisable，由 A/B GM 张量分别消费。

### 4.2 基础单槽与双槽输出映射

本节按 §4.3 核对目标接口，绑定基础单槽结果单元/双槽整块执行策略。对采用 ROW_MAJOR Fixpipe 的该实现，在首次 MMAD 前调用 `AscendC::SetMMLayoutTransform(true)`，使 MMAD 的结果生成顺序与输出消费顺序匹配；全部相关计算和输出完成后调用 `AscendC::SetMMLayoutTransform(false)` 恢复模式。该模式是核内状态，不能只设置 Fixpipe 的输出格式而遗漏初始化或退出恢复。收尾使用所选协议的完成依赖，不在稳态循环增加全流水屏障。

```text
AscendC::SetMMLayoutTransform(true);
execute_selected_window_and_output_protocol();
drain_pending_compute_and_output();
AscendC::SetMMLayoutTransform(false);
```

上述两个职责伪操作由模板循环与收尾实现；`SetMMLayoutTransform` 的可用架构和接口声明按所用 SDK 的 `kernel_operator_mm_intf.h` 核对。

单槽方案的中间 MMAD 使用 NON_FINAL_ACCUMULATION，最后 MMAD 与输出使用 FINAL_ACCUMULATION；输出与后续累加依靠目标 UnitFlag 机制交接对应结果单元，不增加 L0C 的整槽 Mutex。双槽方案的 UnitFlag 为 0，改用每槽完整结果就绪与输出完成的配对事件或等价 Mutex。两套路线共享 K 数值初始化条件，但不同步混用。

L1/L0 输入槽仍分别建立就绪与归还依赖；事件/锁初态、最后消费者及结束回收由 §3 的适用方案实现。保持同一粒度是性能复现条件之一，整槽等待不能冒充细粒度交接。

### 4.3 目标接口与执行策略绑定

同一任务映射和分块参数在不同 SDK/API 版本上可能对应不同窗口搬运、累加交接或输出协议。实例化时按目标版本的公开头文件和 API 声明绑定实际重载、参数单位、同步机制及完成边界；接口映射不重新选择模板策略。记录所用头文件、编译宏、源码/构建版本和产物身份，避免仅凭版本号或字段同名推断内部实现。实现身份与验证证据遵循[公共验收](../../generic/common-contract.md#51-实现身份与性能对照)。

### 4.4 按实际分区绑定缓冲地址

平台接口提供各空间可用字节数，LocalTensor 偏移按元素宽度换算。接口按所选模板和执行协议绑定实际分区，单槽与双槽选择仍由模板给出，不在接口层重新计算策略。

## 5 Elementwise/Broadcast Bias[N] 组合扩展接口绑定

Bias[N] 组合扩展沿输出 N 轴读取向量，广播到同一输出的其他坐标；Elementwise 加法和 Broadcast 坐标语义引用对应范式。`hasBias` 只选择组合扩展是否实例化。有组合扩展时可通过目标支持的 BiasTable/累加接口将 bias 融入首个 K 窗口，或采用其他满足数值契约的融合映射；无组合扩展实例保持零初始化路径，不构造 bias 地址、存储或同步。具体张量类型、转换参数和重载按目标 SDK 的公开声明绑定，不由某一类型示例推断其他类型能力。

首段融合时以 bias 启动输出累加，后续 K 窗口只累加输入乘积。bias 的 MTE2→MTE1 与 BiasTable→计算消费者分别按最后读取完成后复用；同一 `[N]` 向量映射到各保留轴输出坐标。不得增加稳态全流水屏障，且不得在无组合扩展分支访问 bias 地址。
