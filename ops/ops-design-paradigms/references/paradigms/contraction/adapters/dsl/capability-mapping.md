# Contraction CANNBotDSL 能力与接口映射

本文件承接 [公共适配契约](../../generic/common-contract.md#7-设计与接口适配契约)，将 generic 已确定的类型、布局、窗口、任务和依赖映射到 CANNBotDSL。入口与参数绑定见 [Kernel 入口](kernel-entrance.md)。选块、调度及五类模板的执行规则仍由 generic 维护。

本映射覆盖非量化 Matmul、BatchMatmul 及已有的一维 Bias[N] 组合扩展，基础接口约束的目标为 NPU ARCH 3510 / Ascend 950PR、950DT。适配任务只读取 Contraction 内部的 generic 和本 adapter；所需基础接口签名与约束直接列在本文及入口文档中，不参考完整算子实现或外部 Matmul 专题资料。实例化时记录实际 DSL 包、编译器与运行目标，核对其接口是否满足本文约束；文档支持与设备验证分别记录。

## 1 能力事实与绑定条件

| 公共契约要求 | DSL 表达与接口依据 | 实例化时必须核对 |
|---|---|---|
| Tin/Tacc/Tout | `dtypes`、`matmul` 输入/累加类型、`mem_copy` 输出转换；见 §2、§3 | 当前类型组合、计算模式、舍入和误差契约 |
| LayoutX、SpanX、对齐 | `Tensor.layout/physical_layout`，`Buffer/Channel` 的 shape、capacity、物理格式；见 §2 | 父布局、物理跨度、地址对齐、batch pitch 与尾部访问 |
| 各层容量与地址空间 | `MemLoc.L1/L0A/L0B/L0C/BIAS`，`get_mem_size`；见 §2 | 各独立槽、驻留数据、padding、保留区及实际分配范围 |
| gm2l1、l12l0、mmad、l0c2gm | `make_copy_engine`、`mem_copy`、`matmul`；见 §3 | 所需类型、方向、有效尺寸与布局转换是否可表达 |
| 生产、消费与归还 | 实际 Tensor 访问的依赖分析，必要的流水同步及 UnitFlag；见 §4 | 首轮、最后消费者、槽回绕、所选交接粒度和收尾 |
| 可选 Bias[N] | L1→`MemLoc.BIAS` 搬运及 `matmul(..., bias=...)`；见 §6 | 仅一维共享向量、Tacc 类型、首段初始化与恰好一次应用 |

“存在公开接口”只证明可以开展对应绑定。分配地址、特定子窗口搬运或编译器依赖尚未核对时，记录待验证项及其影响。最终候选仍须满足 [公共选块合法性](../../generic/tiling-selection.md#2-候选生成与合法性)，不能以数学结果等价代替策略、布局或同步等价。

## 2 类型、布局与物理资源

### 2.1 数值类型

本文的 [基础 MMAD 接口](#31-基础-mmad-api)承接 FP16/BF16 输入、FP32 累加，以及 FP32 输入/累加的组合。Tout 单独绑定到最终搬出，不随 Tin 或 Channel 类型隐式决定。FP32 路径的计算模式必须明确；要求关闭 HF32 时，按 §3.1 的 `set_fp32_mode()` 绑定设备配置，并由 §7 核对实际编译结果，不能从 FP32 dtype 或 FP32 累加类型推定该配置已生效。HF32 仅在公开数值契约允许时使用，不能静默以较低精度模式替换 FP32。

generic 中其他类型的语义与模板保留原约束，本 adapter 不提供量化类型的接口绑定。浮点输入解释、分段累加与 Golden 继续引用 [公共数值契约](../../generic/common-contract.md#1-语义与数值)，比较规则来自当前算子的数值契约。

### 2.2 GM 与片上视图

generic 的 A(m,k)、B(k,n)、C(m,n) 是逻辑坐标；DSL Tensor 的轴序、stride 和物理格式分别绑定。以下关系核对的是接口表达，不重新选择分块：

| 输入存储方向 | 可核对的 DSL 表达 | 必须保持的信息 |
|---|---|---|
| A[M,K] | GM `[M,K]` → ND2NZ L1 → L0A | M/K 元素步长、当前任务和 K 窗口起点 |
| A[K,M] | GM `[K,M]` → DN2NZ，L1 采用适配的 ZN 表达 → L0A | 逻辑 A(m,k) 与存储轴序的映射，L1 父窗口跨度 |
| B[N,K] | 连续 GM `[N,K]`，stride=`(K,1)` → ND2NZ → L1 `(NL1,KL1)`/NZ → `(nTaskValid,kL0Valid)` 源子视图 → 普通加载 → L0B `(NL1,KL0)`/NZ，rhs 视图 `(nTaskValid,kL0Valid)` | 逻辑 B(k,n) 的 GM 地址 `n*K+k`、L1 父物理 pitch 与 MMAD 有效 K/N |
| B[K,N] | 连续 GM ND `[K,N]`，stride=`(N,1)` → ND2NZ → L1 `(KL1,NL1)`/NZ → `(kL0Valid,nTaskValid)` 源子视图 → `mem_copy(b_l0_view, b_l1_view, transpose=True)` → 硬件 L0B ZN | GM 源轴序/stride、L1 父物理 pitch、有效 K/N 与下述 DSL 右输入声明 |

表中的 NL1、KL1/KL0 和有效尺寸沿用所选 generic 模板；槽容量及实际 Span 仍按 §2.3 核算。源子视图按当前 L1 父窗口内的 `kL0Offset` 建立，保留父物理 pitch。四种 A/B 组合逐一绑定，不以默认路径代替未核对的组合。

§3.1 的 K×N/ZN 描述硬件 B(k,n) 计算布局，不规定所有 DSL L0B 唯一采用 `(K,N)`/`data_format="zn"`。上述 B[K,N] 转置加载可绑定 L0B `(NL1,KL0)`/默认 NZ、rhs 有效视图 `(nTaskValid,kL0Valid)`；采用 `(KL0,NL1)`/显式 ZN 时，rhs 有效视图为 `(kL0Valid,nTaskValid)`。每种声明都须与对应加载形成的物理布局及 `matmul` 推导的 M/N/K 成套核对，尤其保持加载与 MMAD 的 N 分形步长一致；不能只交换 shape 或格式声明就宣称完成转换。

上表连续 GM B[K,N] 路线中，L1 源子视图为 `(k,n)`；采用 L0B `(n,k)`/NZ 及同轴序 rhs 视图时，`mem_copy(b_l0_view, b_l1_view, transpose=True)` 执行对应的 L1→L0B 物理加载转换。`(n,k)` 是 DSL 右输入的视图轴序，数学仍按 generic 的逻辑 B(k,n) 计算 `C=A@B`，GM 中 `B[k,n]` 的元素地址仍为 `k*N+n`。

`tile_slice` 的 coord 表示 tile 编号，不能直接传入任意元素偏移。SWAT 子任务、首窗口二分、驻留子窗口等起点可能不等于编号乘常规 tile 长度；先按 generic 的真实坐标建立合法视图，再应用所需切块。GM 的 `reinterpret(..., offset=...)` 使用字节偏移，stride 使用目标 dtype 的元素单位；运行时 offset 要显式给出 shape 且保持 dtype。片上视图需保留父物理布局，不能用紧凑子窗口 stride 重解释 NZ/ZN。

`Tensor.reinterpret(dtype=None, shape=None, *, stride=None, offset=0)` 返回同一存储的别名。显式 shape 而省略 stride 会推导紧凑 stride，非连续 GM 必须保留真实跨度；运行时字节 offset 使用 64 位整数并保证访问范围和对齐。`layout` 描述逻辑坐标，`physical_layout` 描述物理排布，建立片上子视图时保留父布局及容量形成的 pitch。创建视图不执行数据重排、数值转换或 padding。

### 2.3 槽数、跨度与地址

`Channel` 的 depth 承接模板的独立槽数；槽内 batch 子区数量另由 shape/layout 表达。`Buffer` 承接无需轮转的存储，例如完整驻留输入；它不提供同步。动态有效 shape 配静态 capacity，物理跨度按 capacity 与真实格式求得。尾任务缩短有效尺寸，不缩短已分配槽间距。

`Channel`、`Buffer` 从 `cannbotdsl` 导入，存储位置使用 `MemLoc` 枚举。这里只列本适配所需的构造参数：

| 参数 | 约束与作用 |
|---|---|
| `mem_loc`、`shape`、`dtype` | 指定存储层、逻辑形状和元素类型；Channel 的 shape/dtype 在语义上必选，Buffer 使用 shape 或显式布局形式。资源在 Kernel 或其调用的 `@jit` 中建立 |
| `capacity` | 静态正整数 tuple，与 shape 同 rank；动态 shape 必须显式提供，运行时有效尺寸不得超过对应容量 |
| `depth` | Channel 独立物理槽数，正整数；普通构造申请连续存储并划成等大槽。Buffer 只有一块存储，无轮转游标 |
| `data_format` | L1/L0A/L0B/L0C 默认 NZ，BIAS 默认 ND；L0B 的构造轴序与真实 ZN 计算排布仍按 §2.2 绑定 |
| `stride`、`n1_pad` | ND stride 使用元素单位；NZ/ZN Channel 的 stride 必须为 `None`。`n1_pad` 是 NZ 相邻 N1 切片的额外元素间隔，增加容量但不初始化数值 |
| `kind` | Channel 默认为 `ChannelKind.SameCore`；本适配承接核内资源交接，不引入跨核队列 |
| `layout`、`physical_layout` | Buffer 的显式布局形式，与 shape 形式互斥，须同时绑定 dtype 和静态 capacity；不提供同步 |

NZ/ZN Channel 支持 rank 2/3，由静态 capacity 推导物理布局，不能通过任意 stride 改写其物理 batch pitch。BIAS 为一维 ND、32 位元素。IterBatch 的固定子区间距、MergeBatch 的拼接坐标必须与实际物理布局逐点核对；其他布局用 Buffer/视图表达后仍须满足 §3 的搬运条件，不能在容量计算中假定接口已实现该布局。

输入分形粒度为 `K0=32B/sizeof(Tin)`，FP16/BF16 为 16，FP32 为 8；L0C 分形固定为 16×16。NZ 为外层列优先、分形内行优先，ZN 为外层行优先、分形内列优先。容量与子窗口地址使用对应层的真实分形，不把 generic 的 16×16 数值实例套到全部类型。

容量从 [入口中的平台查询](kernel-entrance.md#4-平台选块与-launch)取得。普通 Channel 自动申请存储不证明实际槽基址满足 generic 已选的等分容量分区或 bank 排布。所选策略需要指定地址、槽间隔或并发 bank 关系时，必须从实际分配和编译产物核对这些关系。不能把紧凑自动分配当作已有分区策略的等价实现。地址绑定未闭环时保留能力缺项，不改写 generic 的分配规则。

`cannbot-dsl 0.7.0+gd8dca30` 的 `dav-3510` 目标提供以下公开 Cube 空间绑定，相关对象和 `make_channel` 从 `cannbotdsl` 导入。此处承接已选分区，不重新选择分块、槽数或地址策略：

| 公开接口 | 静态绑定条件 |
|---|---|
| `L1/L0A/L0B/L0C.view(*shape, dtype=None)` | shape 的元素积等于该空间完整容量，返回 uint8 字节根；dtype 省略或为 `dtypes.uint8`。容量须与当前目标/包的完整空间一致；同一 Kernel 重复调用同空间时复用同一根 |
| `root.reinterpret(dtype=..., shape=..., offset=..., data_format=...)` | 明确静态正整数 shape、非负静态字节 offset、Tin/Tacc 和 ND/NZ/ZN；NZ/ZN 为 rank 2/3。视图完整物理跨度、对齐和范围必须合法；它只解释存储，不搬运、转换数值或转置已有字节 |
| `make_channel(views, *, dtype=None, kind=None)` | 非空 tuple/list 中各槽都是上述根派生的未绑定静态视图，dtype、shape、stride、物理布局及空间一致；默认核内 `ChannelKind.SameCore`。槽数取视图个数，沿用原 Channel 的 produce/consume 语义；不另申请各槽存储 |

先按 [已选分区公式](../../generic/execution-strategies.md#6-等分容量分区) 派生每个空间中的槽基址，再用原常规最大窗口的 typed shape/format 构造静态槽。下面仅示出 A[M,K] 的 L1 两槽绑定，位于 Kernel 或其设备域 `@jit` 中；全部变量来自已有静态容量、分块和基址推导，不是新增 Host 字段：

```python
from cannbotdsl import L1, make_channel

l1_root = L1.view(L1Bytes)
a_l1 = make_channel((
    l1_root.reinterpret(dtype=Tin, shape=(ML1, KL1),
                        offset=aL1Base[0], data_format="nz"),
    l1_root.reinterpret(dtype=Tin, shape=(ML1, KL1),
                        offset=aL1Base[1], data_format="nz"),
))
```

L1 的 A/B 槽共用该字节根，但分别建立自己的 Channel；各阶段 A/B 邻接、两阶段间距按所选公式绑定。L0A/B/C 分别取本空间字节根，按 `aL0Base/bL0Base/cL0Base` 和 §2.2 已选轴序/格式构造；这些基址及槽间距是静态派生值。不得扩大 typed 父 shape 来制造半区间距，或把派生槽间距变成无人消费的运行时参数。当前有效子窗口在选槽后建立，保留原父 pitch 和预留跨度。

完整字节根容量、typed 数据 Span、地址包络和实际槽间距分别核算；根与其别名不重复计作独立分配，同一根上的不同资源仍须证明全部槽区间不交叠。已绑定 Buffer/Channel 的资源不能转移给 `make_channel`。同一 Kernel、同一空间的 manual 根与普通自动 Buffer/Channel 不能混用，声明顺序不能解除限制；存在任何 manual 空间根的 Kernel 不支持 `channel_rewind`，无论 rewind 位于根声明前或后。rewind 也不是等待、归还或槽游标复位，不能用于跨任务回绕。

此版本的公开绑定已有静态原语 frontend、分配及生成指令证据，但它不自动证明完整 K、首窗口二分、尾块、SWAT、各池不同推进频率或最后消费者完成。按 §4/§7 使用真实实现的同次产物逐项承接。Basic 的实际基址表用于记录所选分配结果，不是改变方案的授权；改用自动紧凑地址时，须返回已有设计/分区策略，明确半区分隔、同阶段 A/B 邻接与固定间距各项保证的去向，不能从容量合法或独立双槽推定等价。

## 3 搬运、累加与输出

| generic 操作 | DSL 绑定 | 消费的模板信息 |
|---|---|---|
| gm2l1 | `mem_copy`，按布局选择 `make_copy_engine(format_transform="nd2nz"/"dn2nz")` | 本任务输入基址、真实 stride、当前有效窗口、L1 父布局及预留容量 |
| l12l0 | `mem_copy` 的 L1→L0A/L0B 路径及适用的 transpose 配置 | 父窗口中的源坐标、局部目的布局、有效 M/N/K 与补齐范围 |
| mmad | `cannbotdsl.ops.matmul.matmul` | L0A/B Tensor、同一输出区域的 L0C Tensor、首段初始化、所选结果交接方式 |
| l0c2gm | `mem_copy` 的 L0C→GM 路径，按目标格式选择 NZ2ND 与类型转换 | 完整 K 结果、L0C 源布局、C 的目的 stride、有效输出范围与 Tout |

操作名 `mmad` 沿用 generic 的职责命名。下列基础接口承接所选模板的单次搬运与计算，不提供完整 Matmul/BMM 实现或另一套调度算法。

### 3.1 基础 MMAD API

从 `cannbotdsl.ops.matmul` 导入 `matmul`。这是 AIC 上的 Cube 矩阵乘加原语，使用 `PIPE_M`：

```python
def matmul(
    dst: Tensor, lhs: Tensor, rhs: Tensor, *,
    bias: Tensor | None = None, init=True,
    unit_flag: int = 0, disable_gemv: bool = True,
) -> None: ...
```

| 操作数/参数 | 本适配使用的约束 |
|---|---|
| `lhs` | L0A，硬件逻辑矩阵 M×K，NZ 分形 16×K0，基址 512B 对齐 |
| `rhs` | L0B，硬件逻辑矩阵 K×N，ZN 分形 K0×16，基址 512B 对齐；Tensor 轴序与加载绑定见 §2.2 |
| `dst` | L0C，硬件逻辑矩阵 M×N，NZ 分形 16×16，基址 1024B 对齐 |
| dtype 组合 | FP16×FP16→FP32、BF16×BF16→FP32、FP32×FP32→FP32；bias 若存在为 FP32 |
| `bias` | BIAS 中共享 N 向量，广播到 M；基址 64B 对齐，由 L1 预先搬入，长度 `ceil(N*4/64)*64` 字节 |
| `init` | `True` 清零 L0C；`False` 不清零，初值来自已有 L0C 或 BiasTable。Bias 首段与后续段的绑定见 §6 |
| `unit_flag` | `0` 关闭，`2` 保持占用，`3` 更新交接标志；计算与对应搬出成对配置，见 §4.1 |
| `disable_gemv` | 默认 `True`；M=1 且为 `False` 时 L0A 按 ND 读取，仍需 512B 对齐，须单独证明布局绑定 |

单次指令的有效 M/N/K 单位为元素，范围为 `[0,4095]`；任一为零时为 NOP，不写出退化结果。物理存储按 M/N 向上补齐到 16、K 向上补齐到 K0 申请，基址及完整跨度均不能超出对应存储层。全局尺寸超过该范围时由 generic 的窗口分段，不能截断尺寸。

3510 上向同一 L0C 累加的相邻 MMAD 无需额外流水同步；这不替代输入加载、输出和槽复用的依赖。基础接口提示 NaN 输入可能导致执行报错；数值契约包含此类输入时须独立验证，不能凭类型支持宣称满足全部特殊值要求。

FP32 输入的显式模式接口从 `cannbotdsl.ops.cube` 导入：

```python
def set_fp32_mode() -> None: ...
```

该接口无参数、无返回值，在本适配的 950PR/950DT 目标上使用 `PIPE_S`，仅在 AIC 生效，非 AIC 调用直接返回。它关闭 HF32，使 L0A/L0B 中的 FP32 输入参与 MMAD 前不做 HF32 舍入；Tin、FP32 累加及最终 Tout 转换仍按已有数值契约绑定。

在 Kernel 或其调用的设备域 `@jit` 内，首条需要该模式的 MMAD 前显式调用。模式持续生效，后续 `enable_hf32()` 会改变该状态；分支、循环及内联调用须核对所有实际进入 FP32 MMAD 的路径，保证所选模式未被覆盖，不能依赖前次设备调用或 Host 配置。动态控制流后的状态若无法由当前编译器明确承接，按 §7 报告具体缺口。

编译绑定核对 `asccube.set_fp32_mode` 及对应的 `asc_set_fp32_mode()` 控制指令，并追溯其作用范围内的实际 FP32 MMAD；不能只检查设置调用存在而忽略后续模式变更。优化后的等价配置仍须由同一编译实例证明，源码声明或静态 dtype 不作为设置生效的证据。

### 3.2 搬运 API 与转换条件

`mem_copy`、`make_copy_engine` 从 `cannbotdsl` 导入。本适配使用单源、单目标 Tensor；以下列出相关参数，未列出的量化、多目标、原子等模式不作为本适配的绑定：

| 接口参数 | 作用与约束 |
|---|---|
| `mem_copy(dst, src, ...)` | 目标在前、源在后；其他参数用关键字传入，真实访问范围由 Tensor 视图确定 |
| `engine=None` | 按存储位置与布局选择普通搬运或 GM→L1 的 ND2NZ/DN2NZ；不自动启用转置、分区或布局折叠 |
| `make_copy_engine(format_transform=...)` | `"identity"` 不转换；`"nd2nz"` 为 GM/UB→L1 的 ND→NZ；`"dn2nz"` 为 GM→L1 的 DN→ZN；`"nz2nd"` 为 L0C→GM/UB 的分形→ND；`"transpose"` 用于 L1→L0A/L0B |
| `transpose=True` | L1→L0A/L0B 转置加载简写，须与 engine 的显式转换一致 |
| `unit_flag=0` | L0C 搬出的默认交接方式；结果单元路线与 MMAD 对应，见 §4.1 |
| `pad_value`、`left_padding`、`right_padding` | 填充值及左右 padding 数量；必须满足下面的方向、维度和字节限制 |

`make_copy_engine` 只声明静态配置，不建立窗口、不执行搬运。源/目的 dtype、有效 shape、stride 和物理布局必须匹配所选通路，不能以 engine 名称推定任意子窗口或类型转换合法。

对 §2.2 的连续 B[K,N] 的 ND2NZ→L1 NZ 路线，普通 L1→L0B 加载不会自动将 NZ 字节转换成 ZN，须显式执行所列转置加载。仅交换 shape、重解释视图或修改目标 format 声明不执行字节重排，不能代替该转换。

该 B 路线实例化时，将逻辑 B(k,n)、GM 源轴序/元素 stride/窗口起点、Channel 声明轴序/物理格式/容量、L1 父物理 pitch/子窗口地址、加载转换以及 MMAD 的 rhs 有效视图作为同一条路径核对；沿用 §7 的 review_evidence 对照实际源码及生成指令，确认 ND2NZ 后对应 L1→L0B 转置加载，加载与 MMAD 使用匹配的 N 分形步长和有效 M/N/K。有效子窗口不能取代父物理 pitch，ZN 声明不能转换已有 NZ 字节。

连续 B[K,N] 的 DN2NZ→L1 `(KL1,NL1)`/ZN→L0B `(KL0,NL1)`/ZN 普通加载是另一条成套绑定；选择它时，须按所选 L1 布局重新派生父 pitch、切片地址、容量和验证模型，不能沿用 generic 中 L1 NZ 贯穿实例的布局公式。单输出任务、单 K 段的设备数值通过不证明多窗口、多任务及尾块已完成；完整 K、唯一输出、固定槽跨度与最后消费者完成后复用仍由所选 generic 模板及 §4/§7 的实现和验收承接。

| 所需通路 | 接口边界 |
|---|---|
| GM→L1 | 普通搬运、ND2NZ、DN2NZ；真实存储方向决定转换，GM 读取不得越界 |
| GM→UB→L1 | 需要多维 padding 时可用 GM→UB NDDMA，再用 UB→L1 普通搬运或 ND2NZ；增加的 UB 存储和依赖仍进入资源核算 |
| L1→L0A/L0B | 普通或转置加载；保留源父布局，结果须满足 §3.1 的 MMAD 格式和对齐 |
| L1→BIAS | 连续一维 FP16/BF16/FP32→FP32，目的不超过 4096B；先完成输入转换和长度补齐 |
| L0C→GM | 普通或 NZ2ND 搬出及受支持的 FP32→Tout 转换；有效目的范围、舍入及结果交接分别核对 |

NDDMA padding 仅适用于 GM→UB，同 dtype、同 rank，rank 为 1–5；左右数量各为 0–255，序列长度与 rank 相同，目标 shape 等于源 shape 加 padding。非 NDDMA 的 padding 仅支持 Identity GM→L1，每侧仅有最内轴且不超过 32B 数据量，不能将此选项套到 ND2NZ。所需填充若不在当前搬运通路内，在 DSL 设备路径物化后再加载。

同一有效输出区域只选择一次累加槽，各 K 段使用其别名。无 bias 时，首段 `matmul(..., init=True)`，后续 `init=False`；判断来自当前输出区域的全局 K 起点，不从 Channel 游标或锁状态推定。最后一个有效 K 段结束后才输出，Tout 转换不进入中间 K 段。`matmul` 的零尺寸 NOP 不负责写出 K=0 的零或偏置结果，退化处理见 [入口](kernel-entrance.md#3-公开入口与退化输入)。

搬运根据当前有效视图执行，槽 capacity 按最大合法窗口保留。必须确认 GM 读取范围和首次物化的尾部数值；裁剪逻辑 shape 不证明全部被 MMAD 读取的物理 padding 已初始化。短 N、K 尾部、转置加载和 BiasTable 长度约束分别核对，所需 padding 在设备路径内完成。

## 4 所有权与同步绑定

generic 的 acquire/publish/consume/release 伪操作表达 [资源依赖](../../generic/common-contract.md#4-所有权与同步)。DSL 的 `Channel.produce()/consume()` 只选择槽位并推进各自游标，不等待、不发布数据，也不检查队列满空。调用位置与槽位选择须对应真实分配事件；默认核内 Channel 与单独的 Buffer 均不能凭容器名称证明下面的完成依赖。

| 资源 | DSL 必须表达的真实访问关系 | 槽位选择边界 |
|---|---|---|
| L1 输入窗口 | 搬入完成→所有相关 L1→L0 读取；最后读取→同槽下一次搬入 | 每个实际 L1 窗口选择一次，内层子窗口沿用该 Tensor |
| 驻留 L1 输入 | 完整预载→复用域全部读取；最后读取后才可覆盖 | 预载时取得一次存储，保持完整父视图至复用域结束 |
| L0A/B 输入批次 | L1→L0 完成→本批全部 MMAD；最后 MMAD 读取→同槽下一次加载 | 按输入准备批次选择，不按 MMAD 条数机械推进 |
| L0C 完整结果 | 当前输出区域首次写入→全部 K 段→输出最后读取→同槽下一次写入 | 按模板的结果区域分配次数推进，K 循环内保持同一 Tensor |

编译器依赖分析必须看到真实读写、存储别名与循环复用。自动同步只有覆盖对应访问、参与者和代次时才承接伪操作；看不到的别名或手工协议必须依据当前版本公开语义补足。不能以 produce/consume 次数相同代替访问依赖证明，也不额外叠加稳态全流水屏障。

上述资源依赖分别由算子表达和编译器生成承接，不要求算子实现另一套逐槽取证系统：

| 责任 | 算子表达的内容 | 必须核对的生成结果 |
|---|---|---|
| 任务与窗口 | SWAT 整数控制流、有效边界、KL1/KL0 窗口、首块二分及真实 Tensor 读写 | 对应设备分支、循环、地址与完整 K 累加链 |
| 资源与选槽 | Channel 的容量/槽数、produce/consume 的推进位置、Buffer/别名的持有范围；每层计数按自身生命周期跨任务延续 | 编译器分配的实际独立槽、基址、跨度与对齐；K 段保持同一累加槽 |
| 就绪、消费与复用 | 让编译器看到每次搬运、计算、输出及其真实别名/循环依赖；选择对应的 UnitFlag 或整块路线 | 编译器生成的流水等待/归还或等价完成依赖，覆盖最后消费者到同槽下一次写入、首轮和收尾 |

produce/consume、槽号和代次是实现/依赖分析中的概念，不自动成为公开返回值或探针事件。检查以本次源码及对应 IR、生成代码或可解释的编译产物为依据；能证明的约束按 §7 的 review_evidence 承接。若产物不能解释关键别名、槽分配或完成依赖，写明缺失边和受影响资源并返回本节设计；不声称 depth 或 Host 计划已证明同步，也不把编译器无法证明的依赖默认为串行安全。

### 4.1 结果交接路线

| generic 已选协议 | DSL 表达与核对条件 |
|---|---|
| 整块交接 | `unit_flag=0`，通过实际 M→输出→下一次 M 的完成依赖交接完整槽；双槽按结果区域轮转，各槽最后消费者分别核对 |
| 结果单元交接 | 中间 MMAD 使用 `unit_flag=2`，最后有效 K 的 MMAD 与对应搬出使用 `unit_flag=3`；保持同一槽、交接范围及生成/消费顺序 |

结果单元路线要求 MMAD 与对应搬出同时启用 UnitFlag，计算范围、消费范围及分形顺序一致。同一槽由多条计算或搬出指令持续操作时，前序使用 `2` 保持占用，最后一条使用 `3` 更新标志；不能只为其中一侧开启。

结果生成方向接口从 `cannbotdsl.ops.cube` 导入：

```python
def set_mmad_direction(direction: Literal["m", "n"]) -> None: ...
```

`"m"` 为逐列、`"n"` 为逐行生成结果分形，只改变顺序。UnitFlag 的 NZ2ND 搬出设置 `"n"`，本适配的其他非量化搬出设置 `"m"`；关闭 UnitFlag 时默认 `"m"`。在 MMAD 前配置，该状态持续生效，初始化和收尾仍处理所选路线的待完成访问。

UnitFlag 输出使用参与累加的同一 Tensor 别名。不能为了“消费”再推进游标，使输出读到另一结果槽；也不能同时把整槽等待叠到结果单元交接外层。双槽数量不自动决定采用哪套协议，须保持 generic 已选路线。

IterBatch 的输入槽归还覆盖子组最后一次 MMAD，结果槽归还覆盖全部子区最后输出。MergeBatch 的整块路线保持 `unit_flag=0`，输出只提取对角块；不能为扩大矩阵开启仅部分搬出的 UnitFlag 协议，使未消费的结果单元无法归还。

## 5 五类模板的接口承接

本表给出各模板已经确定的设计需求在 DSL 中的承接位置。模板算法、循环顺序、容量推导和贯穿实例仍读取 generic；是否满足接口条件由能力证据判断。

| generic 模板 | DSL 承接方式 | 必须保持的模板保证 |
|---|---|---|
| [basic](../../generic/templates/basic/kernel-template.md) | DSL 整数控制流执行 SWAT；Tensor 视图表达其任务与两级 K 窗口；L1/L0 Channel 和所选 L0C 路线承接物理槽 | 包含边缘再均衡、尾轮替换和每核首窗口二分；每个输出任务完整 K，槽计数跨任务延续 |
| [l1-resident](../../generic/templates/l1-resident/kernel-template.md) | 完整驻留输入使用保留父布局的 Buffer 或已选单槽 Tensor；流式侧按窗口轮转；L0 子窗口分别从两类父视图建立 | 预载与归还位于整个复用域外；完整父跨度、条带内 M/N 遍历和最后读取边界不变 |
| [batch](../../generic/templates/batch/kernel-template.md) | 先解码 generic 全局父任务的 batch，再执行矩阵 SWAT；A/B 分别按真实广播映射建立 GM 视图 | 全局尾轮与单矩阵调度同源；输入 batch 与输出 batch 分开，保留 SWAT 的任务次序与尾轮映射 |
| [iterbatch](../../generic/templates/iterbatch/kernel-template.md) | 任务组存入带独立 batch 子区的 L1 槽；子组的 L0A/B/L0C 视图保留各自固定 pitch；按 K 外、batch 内发射 MMAD | L1 按组推进、L0 按输入批次推进；子组全部消费者结束后归还，尾组不改变槽跨度 |
| [mergebatch](../../generic/templates/mergebatch/kernel-template.md) | 以模板的 mu/nu 拼接坐标构造扩大矩阵视图，执行扩大 MMAD；从原 L0C 父布局提取各有效对角子窗口 | 保留非对角计算与完整容量核算；只写有效对角结果，全部对角输出后归还整槽 |

五类模板均按 §2–§4 逐项绑定。rank-3 物理 pitch、驻留子窗口、扩大矩阵及对角搬出需要目标版本的视图、加载与输出接口闭环；不能跳过模板条件，也不能把文档中的绑定需求写成已通过设备验证。

## 6 Bias[N] 组合扩展

沿用 [公共 Bias[N] 数值契约](../../generic/common-contract.md#62-可选-bias-的语义与数值)和 [首段融合策略](../../generic/execution-strategies.md#8-biasn-组合扩展的首段融合策略)。GM 输入仅为共享一维 `[N]`；当前 N 区间经 `mem_copy` 搬入 ND L1，再进入 `MemLoc.BIAS`，BiasTable 类型与 Tacc 一致。FP16/BF16/FP32→FP32 的 L1→BIAS 转换条件见 §3.2，基址、补齐长度与容量计入资源表。

每个输出区域的首 K 段以当前 Bias[N] 子区启动累加，后续只累加乘积。`bias` 与 `init` 的具体组合按目标版本的初始化语义核对，不能仅复用无 bias 的清零参数或把 bias 当作每段加法。BiasTable 可复用的边界是其最后一次计算读取；IterBatch 覆盖子组所有首段消费者。

MergeBatch 依模板将同一向量重复映射到扩大 N 子区；所需物化和类型转换在 DSL 设备路径完成，保持原 `bias[nMerged % N]` 语义。不能用新的逐 batch bias 输入替代这个映射。无扩展编译实例消除 bias 地址、分配、搬运与同步，保持原计算。

## 7 实例化与验证

交付当前模板的操作与依赖绑定，并关联实际源码、DSL/编译器版本、静态配置、运行参数及编译产物。按 [公共证据选择](../../generic/common-contract.md#52-设计一致性与证据选择) 区分设计一致性检查和设备测试，复用对应模板校验：

1. 独立复算 SWAT 尾轮、驻留复用域和 batch 子组，对照实际 Kernel 控制流和对应生成产物，核对完整 K 及每个有效输出恰好写入一次；Host 任务表只作为复算依据。
2. 检查 GM、父子视图、真实分配和槽地址，覆盖四种轴序、物理 padding、尾组及跨任务回绕；有效 shape 不代替固定物理跨度。
3. 核对编译产物中的实际独立槽和完成依赖；按实际窗口计数展开到同核同槽再次使用，分别检查 UnitFlag 与整块路线。双缓冲不能退化成单槽以回避检查，也不能由 depth 或 Host 序列推定同步正确。
4. 用非恒定输入做多块、M/N 尾块、首/末 K、单段/多段 K、有/无 Bias[N] 与 K=0 的设备 Golden。重复调用变更输入并逐次比较，形状变化限于已支持的入口域；实例和输入须实际触发所声明的任务规模与复用路径。
5. 所选模板的编译检查和设备 Golden 分别记录通过；性能需同条件测量，公开 API、源码或静态文档检查均不构成设备或性能证据。

普通 MatMul/BMM 的任务映射、已选双缓冲和槽复用是该实现能力的设计约束，由其已有 Unit 的 `design_constraints.review_evidence` 承接；关键未决项使用同一 Unit 的前置 `feasibility_checks.review_evidence`。不把逐槽取证派生成独立算子能力或必经 Unit。数值与边界行为写入对应检验义务，设计审查项不机械转成需要 probe_events/probe_predicates 的结构义务。

review_evidence 沿用现有字段，产物必须来自本次实际编译实例，不能填写预期快照冒充检查结果：

| artifact | check | pass_criteria |
|---|---|---|
| 实际 Kernel 源码、对应 IR 及同次生成的模式控制/MMAD 指令 | FP32 输入所选计算模式、设备调用位置及覆盖路径 | 要求关闭 HF32 时，§3.1 的显式设置在 AIC 上先于所需 MMAD 生效，全部相关路径保持该模式；核对对应控制指令或可追溯的等价配置及后续 MMAD，不能以 FP32 dtype、Host 设置或非 AIC 调用代替证明。无法确认时记录具体模式/路径缺口，不静默转为 HF32 |
| 实际 Host/Kernel 源码及同一编译实例的生成控制流/地址代码 | 请求与实际拆分份数、尾轮父子映射、任务/K 区间、首块二分、KL1 复用及输出所有者 | 逐项对照末块实际长度→actualParts→S→expandedCount 的来源、参数传递与实际消费者；保留请求搜索。正常空槽防护或数值通过不豁免归一化，不能以删空压紧替换发射映射。输出覆盖无遗漏/重复，K 无遗漏/重叠；不能换成逐 KL0 重新搬入来绕过 KL1 承诺 |
| 同一编译实例的 typed 布局、分配 IR 与最终地址代码或资源报告 | 各层独立槽、根容量、基址、数据 Span、父 pitch、实际间距、别名及资源身份 | 逐池对照所选分区公式；选用等分容量分区时，核对 L1 阶段 A/B 邻接和阶段间距、L0A/B/C 半区基址。完整范围、对齐、槽数与间距一致，两个槽不能别名为一份存储。§2.3 的显式绑定须追溯静态视图到对应资源及最终字节地址；只检查独立性/容量或引用另一片段产物不能证明当前所选地址关系 |
| 对应生成的 IR/同步代码或可解释的指令产物 | 每槽就绪、最后消费者、同槽再生产、输出归还与收尾 | 关键完成依赖有可追溯表达，跨任务回绕受保护，输出协议与 §4.1 一致，不添加稳态全流水屏障 |

若编译产物不足以证明关键约束，记录具体能力或证据缺口，返回 §2–§4 及相应 generic 章节；检查未通过前该能力及其后继不能放行。只有实际要求运行时观察的契约才产生真实事件/设备 trace 采集义务，先明确来源与可执行判据；不默认新增 GM trace 系统。Host 按预期组装 task/copy_out/slot_* 或 generation 不能证明设备执行、同步完成或重叠。
