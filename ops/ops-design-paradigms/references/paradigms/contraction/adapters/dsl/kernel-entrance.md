# Contraction CANNBotDSL Kernel 入口

本文件将 [generic 的编译分派与参数产出](../../generic/template-overview.md#3-tilingkey)绑定到 CANNBotDSL 的公开入口、Host 与 Kernel。能力及操作绑定见 [capability-mapping.md](capability-mapping.md)。适配任务所需的入口、编译与启动规则直接列在本文，只读取 Contraction 内部资料。本 adapter 消费 generic 的选择结果，五类模板的算法与契约由原文件维护。

## 1 代码组织与执行域

| 位置 | Contraction 的承接内容 | 接口依据 |
|---|---|---|
| 公开 Python 入口 | 校验逻辑 shape、存储 stride、类型和设备；按 generic 预处理及选块确定当前配置；分配输出与必要临时资源 | 本文 §3、§4 |
| 构建函数 | 将影响代码与静态分配的配置绑定到 Kernel 和 TensorSpec；按需显式编译 | 本文 §2 |
| `@host` | 接收本次 Tensor 与运行时参数，直接执行 `kernel[core_num](...)` | 本节的装饰器边界 |
| `@kernel` 及其 `@jit` helper | 执行模板任务映射、窗口、搬运、累加、输出与资源交接 | [能力与接口映射](capability-mapping.md) |
| `@export` | 用同一构建函数声明可发布的编译实例 | 本文 §6 |

Host 只处理元数据、分配、已确认的零拷贝视图及启动。规格要求的数值转换、padding、Bias[N] 物化、零结果生成及最终结果合成由 DSL 设备路径完成，不以 Host torch 计算代替设备阶段。

Host 的任务计划可用于选块、独立复算及选择测试输入，不代表设备实际任务/槽事件；不为满足普通算子的设计一致性检查在 Host 拼装“已执行”序列，也不默认增加 trace 返回 ABI。真实源码、分配和编译器同步的检查由 [能力映射 §7](capability-mapping.md#7-实例化与验证) 承接，设备 Golden 仍检查每次真实调用的输出。

`host`、`kernel`、`jit` 从 `cannbotdsl` 导入。`@host` 可由普通 Python 调用，应返回 `None`，输出 Tensor 通过形参传入。`@kernel` 不能作为普通 Python 函数调用，方括号 launch 必须直接位于 `@host` 函数体；`@jit` 沿用调用方执行域，不能作为编译入口或 Kernel 启动中转。Kernel 类构造参数只绑定编译期配置，本次 Tensor、标量和核数通过入口形参传入。API 约束与编译/设备结果的区别见 [能力映射 §7](capability-mapping.md#7-实例化与验证)。

## 2 TilingKey、Params 与 DSL 参数的对应

generic 的 TilingKey 章节规定“合法编译实例及其唯一入口”，TilingData/Params 章节规定“参数来源、单位和消费者”。DSL 通过既有编译配置、TensorSpec、Kernel 形参及 Host launch 承接这些保证；无需为其新增数字 key、C++ TilingData 或镜像运行时结构。

| generic 产出 | DSL 中的承接位置 | 保留的保证 |
|---|---|---|
| 模板及执行策略 | 已选 Kernel 配置和对应的设备控制流 | 每个合法组合有唯一实现；不在内层重新选择策略 |
| Tin/Tacc/Tout、A/B 独立 ND/DN、hasBias | 编译配置、TensorSpec dtype/layout 与实际消费这些配置的实例 | 类型及方向不重复编码；不支持组合明确拒绝；无 bias 实例消除全部 bias 代码 |
| ML1/NL1/KL1/KL0、槽数、物理容量、结果交接方式 | 影响静态分配或生成代码的配置 | 同一个选择结果驱动分配、循环和操作绑定，不能捕获不相容的旧配置 |
| 本次 M/N/K、真实 stride、输入输出地址 | Tensor 及其规格；确有独立消费者的标量按形参传入 | 从同一元数据来源取得，保留元素/字节单位，不增加同义维度字段 |
| SWAT、边缘尺寸、尾轮拆分及任务总数 | Host/Kernel 的既有配置、形参或派生值；[份数选择](../../generic/tiling-selection.md#9-尾轮拆分份数选择) 输出 requestedParts，再按 [SWAT](../../generic/swat-scheduler.md#4-尾轮分派与拆分) 由本次 M/N 及边缘再均衡所得末块实际长度逐轴求 actualParts，最后派生 S、expandedCount 与父子映射 | 发射及坐标派生消费同一组实际份数；不能直接用请求份数或标称 ML1/NL1 代替末块归一化。请求搜索、核数政策及正常空槽防护保留；未启用参数不进入镜像大结构 |
| launch 核数 | 本次 `@host` 参数用于方括号 launch；Kernel 用 `get_block_num()` 读取实际数量 | 分配任务与实际 launch 使用同一核数 |
| BatchL1/BatchL0/BatchMerge、pitch、驻留范围 | 模板确实消费的配置、Tensor 布局与子窗口参数 | batch 子区与物理槽分开，尾组不改变最大预留跨度 |
| Bias[N] | 有扩展实例的输入 Tensor、N stride 与实际资源配置 | 一维共享映射、首段恰好一次与独立资源交接 |

`TensorSpec`、`Dim`、`dtypes` 从 `cannbotdsl` 导入。TensorSpec 只描述真实 Tensor 元信息，不保存数据、不支持读取或搬运：

```python
TensorSpec(
    shape: Iterable[int | Dim], dtype: DType, *,
    stride: Iterable[int | Dim] | None = None,
    storage_format: str = "nd",
) -> TensorSpec
```

| 规格项 | 约束 |
|---|---|
| `shape` | ND 静态维度非负；NZ 仅 rank 2/3、维度为正，动态 Dim 的下界至少为 1。共享逻辑维度复用同一 Dim 约束 |
| `dtype` | 使用 DSL dtype 描述符，并与本次真实 Tensor 一致；计算类型仍由能力映射绑定 |
| `stride` | 元素单位，项数与 rank 一致；ND 省略时为紧凑 stride，非连续输入显式描述；NZ 仅支持与逻辑 shape 对应的紧凑规格 |
| `storage_format` | `"nd"` 或 `"nz"`，描述真实 GM Tensor，不代替片上 Channel 的物理布局 |

`cannbotdsl.compile(host_entry, *parameter_specs)` 提前编译 `@host` 函数或绑定方法。规格顺序与形参一致：Tensor 使用 TensorSpec，运行时标量使用对应 DSL 类型描述符；执行返回的程序时传入当前真实 Tensor 和标量值。接受的动态尺寸与 stride 必须位于规格范围内，并被 Kernel 的取址、布局和搬运支持。影响布局、容量或代码的值才进入编译配置，不为每个本次尺寸或地址生成无意义变体。

形状或平台信息变化时，先按 generic 重算选择。若分块、静态容量、布局或协议改变，取得匹配的编译实例；若仍满足同一编译契约，传入当前 Tensor 和运行参数复用程序。不得因缓存可命中而沿用上一次输入的任务数、尾轮或核数。

显式编译返回的程序由持有方管理，在最后一次使用结束后通过上下文管理器或 `close()` 释放；关闭后不能继续调用。并发或异步路径确认最后使用完成，程序、Tensor 和临时存储在此之前保持有效。连续复用时将创建和释放放在整组调用之外；编译产物缓存不等于程序句柄生命周期。

## 3 公开入口与退化输入

校验与归一化沿用 [generic 预处理](../../generic/tiling-preprocess.md)：从规格取得 A/B 逻辑矩阵轴、各自存储方向与真实 stride；前导保留轴按两输入独立映射。零拷贝 reshape/转置视图须证明地址对应关系，不能用一次 `.contiguous()` 隐藏接口不支持的存储组合。

| 输入情况 | 入口与设备行为 |
|---|---|
| 输出为空 | 形成框架约定的空输出，不发起 `core_num=0` 的非法 launch |
| K=0 且输出非空，无 Bias[N] | 启动满足数值与类型契约的 DSL 写零路径，覆盖全部有效输出 |
| K=0 且输出非空，有 Bias[N] | DSL 按同一 N 坐标映射产生 `convert_out(bias[n])`，不进入正 K 的 Cube 循环 |
| 正尺寸 | 使用已选模板的编译实例、真实任务映射及能力绑定 |

退化分支仍属于公开能力契约；普通 `matmul` 的零尺寸 NOP 与预先分配输出不证明结果已经生成。需要 Vector 的设备写出或转换时按对应 DSL 能力与组合范式绑定，不能通过 Host torch 小算子补齐，也不能把验证缺项静默改成拒绝合法 K=0。

Bias 仅接受公共契约规定的一维 `[N]`，类型、N stride 与目标转换条件独立校验。输入 device、输出 device 与启动 stream 一致；无 Bias 编译实例不传入假 bias 作为设备计算输入。

## 4 平台、选块与 launch

平台查询从 `cannbotdsl` 导入：

```python
def get_platform_info(
    force_refresh: bool = False,
    stream: torch.npu.Stream | None = None,
) -> PlatformInfo: ...

def get_mem_size(mem_type: str) -> int: ...
```

`get_platform_info` 返回 `available`、`soc_version`、`npu_arch`、`cube_core_num`、`vector_core_num` 和 `core_num` 等属性，核数不超过设备与指定 stream 的相应上限。查询与提交使用同一设备和 stream；传入 `None` 时查询当前 stream，获取失败可能退回硬件核数，不能将其直接视为本次 stream 的配额。

`get_mem_size` 只接收存储类型字符串，不接收 stream 参数；在相同当前设备上下文中查询。可选键为 `ub`、`l1`、`l0a`、`l0b`、`l0c`、`bt`、`fb0`，单位为字节，平台信息不可用时返回 0。必要容量、可用核数非正或 `available=False` 时报告缺项，不能替换成开发设备常数后继续执行。

纯 Cube 正尺寸路径依据 `cube_core_num` 选择 launch。所选策略的 coreCount、任务总数、尾轮参数与 launch 同源；不能在启动处独立按固定核数或线性 tile 数重新截断。L2 容量、吞吐与带宽等未由接口提供的模型量由调用方提供，记录来源及单位，不从函数名字推定可查询。

选块使用 [generic 的确定策略](../../generic/tiling-selection.md)，同一候选的窗口、槽布局、容量和最终任务联合核验。Python Host 可以实现这些元数据计算，但不能用固定 tile、L1 depth 或其他选块算法替代该策略。缺少合法能力绑定时返回具体缺项，保留策略与验证义务。

`@host` 直接发起已选 Kernel，Tensor 与运行时标量按声明的签名同序传入。`@jit` helper 不作为 launch 中转。Kernel 从 `cannbotdsl.ops.arch` 导入 `get_block_idx()/get_block_num()` 取得实际 launch 信息，再执行所选模板的每核任务循环，不另传同义核数字段。

## 5 任务、窗口与设备阶段

下面各行只绑定入口和数据组织，完整算法读取对应 generic 文档；能力、布局及所有权使用 [能力映射](capability-mapping.md#5-五类模板的接口承接)。

| 模板 | 入口之后的 DSL 承接职责 |
|---|---|
| basic | 按 [SWAT](../../generic/swat-scheduler.md) 求父块、边缘尺寸与尾轮子块；使用真实元素起点建立 A/B/C 视图；按每核任务轮次展开首窗口和后续窗口 |
| l1-resident | 为有任务的核建立完整驻留父视图，在复用域开始前预载；条带内各输出块分别建立流式与驻留侧的 L0 子窗口 |
| batch | 从 generic 全局 parentTaskId 解码输出 batch 和矩阵内任务；A/B 各自映射到真实输入 batch，随后建立矩阵与 K 窗口视图 |
| iterbatch | 任务组及下一组预取来自本核任务序列；完整 L1 组与当前 L0 子组使用各自 batch 起点和 pitch，按模板循环发射逐 batch MMAD |
| mergebatch | 用当前任务的 mu/nu 映射建立扩大输入、累加矩阵与对角输出视图；保留原物理父跨度及扩大计算量 |

DSL 控制流得到的 `(mTaskOffset,nTaskOffset,mTaskValid,nTaskValid)` 与 generic 任务表一致。`tile_slice` 不能以任意元素偏移冒充 tile 编号；非标准起点与固定父布局的承接条件见 [视图映射](capability-mapping.md#22-gm-与片上视图)。

每核资源在对应模板的生命周期边界建立，L1、L0 输入和 L0C 的槽推进各按其分配事件发生；任务基址变化不重建全部 Channel 或重置计数。驻留矩阵与完整任务组保持父视图，局部 K 目的窗口从零起点建立。无任务核不消费不存在的预载，也不等待从未发出的生产事件。

generic 的 gm2l1、l12l0、mmad、l0c2gm 阶段可由 `@jit` 组织，但阶段方法边界不替代真实同步。首轮、稳态、最后消费者和排空按 [所有权绑定](capability-mapping.md#4-所有权与同步绑定)核对；流水发射保留原窗口顺序及允许的重叠，不把提交顺序当作异步完成。

Bias[N] 首段融合沿用 [公共策略](../../generic/execution-strategies.md#8-biasn-组合扩展的首段融合策略)及其 [DSL 绑定](capability-mapping.md#6-biasn-组合扩展)。各模板的输出区域分别判断首 K，IterBatch 子组与 MergeBatch 扩大列映射不新增另一套 bias 语义。

## 6 构建与验收绑定

在已有设计产物中关联“模板/策略 → 编译实例 → 参数来源与消费者 → 操作及同步 → 验证证据”。generic 的 key/kernel 对应在 DSL 中体现为编译配置与实际 Kernel 的对应；字段消费表体现为构建配置、TensorSpec、Host/Kernel 形参和派生值，不新增第二份注册表或协议文件。

发布需要时，从 `cannbotdsl.aot` 导入 `export`，用 `@export("name")` 声明同步、无参数回调。回调枚举支持的编译实例，通过同一构建函数实际发起编译请求；导入时只标记入口，直接调用回调只编译。`cannbotdsl.aot.build(collectors=..., target=..., output=...)` 收集编译请求并构建 Native 产物。冷/热缓存都必须覆盖声明的变体，不能因运行时句柄已缓存而跳过请求；回调释放自己取得的程序，运行时持有的句柄仍由原调用方管理。

| 公共验收要求 | DSL 证据 |
|---|---|
| Host 选择 | 当前形状/stride、平台信息、策略及模型输入、最终分块/槽数/任务配置 |
| 编译实例 | DSL/编译器版本、源码身份、静态配置、TensorSpec、生成产物及实际分配/依赖 |
| 运行入口 | 实际程序与 Kernel、当前 Tensor/标量、launch 数、任务配置和设备执行记录 |
| 覆盖与数值 | 独立任务/地址复算、模板所需边界与槽回绕、输入、Golden、Tout、结果和比较规则 |
| 性能 | 相同数值契约与运行条件下的编译实例、输入和实测；JIT/Host 时延与设备时延分别记录 |

复用 [公共验收](../../generic/common-contract.md#5-产出与验收)和各模板检查。文档链接、API 名称或编译成功不证明设备同步与 Golden 通过；Contraction 的现有 C++ 静态检查也不成为 DSL 通过凭据。实际映射无法核对的项保留能力缺项与验证义务，不能修改 generic 的原有保证来使检查通过。
