# Contraction 范式入口

Contraction 是按指标映射定义的高层计算范式：输入张量在保留指标和收缩指标上对齐，逐项相乘，并对收缩指标求和。对两个输入可写为：

`Y[o] = Σ_{r∈R} A[φA(o,r)] × B[φB(o,r)]`

其中 `o` 是输出保留指标，`r` 是收缩指标，`φA/φB` 将逻辑指标映射到各输入的坐标；shape、stride 和允许的广播关系共同确定物理地址。二维矩阵乘及带前导保留轴的矩阵乘只是该定义的具体实例，不构成不同的 Contraction 语义类别。

当前模板覆盖稠密输入、规则坐标映射和完整收缩轴归约的实现组织。矩阵形实例用 M/N 表示输出轴、K 表示收缩轴；这些是模板变量，不是范式分类。依赖动态索引选择、IndexGather 或 Sparse 语义的执行模板尚未定义。本目录未提供 MX 编码、scale 布局及其接口适配；此类需求须补齐专项契约和目标能力依据。类型由公开契约与目标能力判断，示例不构成 dtype 白名单。

## 编程接口选择

复用调用方明确提供的 `op.language` 选择开发方式/编程接口。缺失或未知时返回输入缺项，不默认选择其他接口。

| op.language | 所选适配资料 | 范围 |
|---|---|---|
| `ascendc` | [AscendC 能力映射](adapters/ascendc/capability-mapping.md)、[运行时入口](adapters/ascendc/kernel-entrance.md) | 基础 API 的张量、搬运、乘加、输出及同步绑定 |
| `dsl` | [DSL 能力映射](adapters/dsl/capability-mapping.md)、[Kernel 入口](adapters/dsl/kernel-entrance.md) | generic 已选模板、参数与依赖的 DSL 绑定；覆盖非量化 Matmul、BatchMatmul 和 Bias[N] |

只展开所选 adapter 及其实际引用的配套章节。未登记的示例返回空，不回退到其他接口示例。

## Read → Write → Check

1. Read：[公共契约](generic/common-contract.md) → [模板总览](generic/template-overview.md) → [预处理](generic/tiling-preprocess.md)。按编程接口选择核对目标能力，再读 [选块](generic/tiling-selection.md) 和所选模板的 DAG、Tiling、Kernel；按问题展开执行策略与调度细节。
2. Write：在现有设计产物中记录收缩指标、输出保留指标、两输入坐标映射、shape/stride、Tin/Tacc/Tout、buffer、任务划分和源码接口对应关系。按 [策略绑定](generic/tiling-selection.md#11-策略绑定与实现对应) 记录策略、模型输入及选择依据；adapter 承接同一套选择结果。新策略先更新公共规则和对应执行模板。
3. Check：验证公式和指标映射、物理地址、任务覆盖、完整归约、参数绑定和资源生命周期。所选策略需能独立复算并与 Host 输出一致；尾轮按 [份数选择](generic/tiling-selection.md#9-尾轮拆分份数选择) 与 [SWAT](generic/swat-scheduler.md#4-尾轮分派与拆分) 逐项核对请求值、末块实际长度、实际份数、S/expandedCount 及父子映射。按 [候选合法性](generic/tiling-selection.md#2-候选生成与合法性) 派生逐池同核同槽覆盖，记录尺寸限制内的可达实例或设计缺口，不以多轮任务或重复调用替代回绕。按 [证据选择](generic/common-contract.md#52-设计一致性与证据选择) 用真实源码/编译产物承接设计一致性检查，保留多块、尾块和重复调用的设备 Golden；模板的槽检查不自动新增取证接口或独立能力。编译、设备同步、Golden 精度及性能对照分别记录，不能相互替代。
4. 输入若含当前模板没有表达的索引、稀疏或编码语义，指出具体能力缺项；不得仅凭算子名称或 dtype 名称声称已有执行路径。

## 主题索引

| 当前问题 | 参考 |
|---|---|
| 术语、数据范围与章节约定 | [术语表](terminology.md) |
| 语义、数值、布局、资源与适配边界 | [公共契约](generic/common-contract.md) |
| 五类模板与组合条件 | [模板总览](generic/template-overview.md) |
| 指标归一化、shape/stride 与空维 | [预处理](generic/tiling-preprocess.md) |
| 候选、评分、参数配置与独立复算 | [选块](generic/tiling-selection.md) |
| 窗口、槽轮转与结果交接 | [执行策略](generic/execution-strategies.md) |
| 任务遍历、边缘再均衡与尾轮重切分 | [SWAT](generic/swat-scheduler.md) |

当前矩阵形模板可通过 Elementwise/Broadcast 组合扩展表达 `A @ B + bias`。bias 是一维 `[N]` 向量，沿输出 N 轴相加并广播到其他输出轴；语义见 [公共契约](generic/common-contract.md#62-可选-bias-的语义与数值)，执行融合见 [Bias 首段策略](generic/execution-strategies.md#8-biasn-组合扩展的首段融合策略)。

## 静态校验与证据

[validate_gates.py](validate_gates.py) 提供 `validate_gates(design_text) -> list[str]`，消费设计文档中的 C++ 代码块。检查明确的非正拆分份数、已识别的指针绑定错误及 BF16 输出字节数错误；注释、字符串和类名不作为策略选择证据。核数、请求搜索和归一化按 [选块](generic/tiling-selection.md#74-kl0kl1槽数及任务参数) 与 [SWAT](generic/swat-scheduler.md#4-尾轮分派与拆分) 对照实际参数来源和消费者，不因 `min`、`CeilDiv` 或变量名判为违例，也不要求份数固定为 1。

测试覆盖这些静态诊断的正反例与语言路由。符号表达式和目标版本的接口约束仍需源码/编译产物审查；DSL 的布局、模式及分区绑定另按其 adapter 验证。空错误列表不证明完整策略、覆盖、ABI、同步或设备精度通过。

设备结论绑定生成源码、依赖版本、构建命令、输入、Golden 和实际设备结果，见 [公共验收契约](generic/common-contract.md#5-产出与验收)。

资料迁入基线：[skills-dev MR 200](https://gitcode.com/opencann/skills-dev/merge_requests/200)，提交 `ed6e28f80041dc175cccb535d38111ae499b84e4`。本目录的公共规则和模板分别由上述主题文件维护；硬件与公共 API 继续引用已有事实源。
