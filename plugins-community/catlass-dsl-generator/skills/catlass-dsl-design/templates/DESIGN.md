# CATLASS DSL 算子设计

> 本文档用于设计评审和用户批准。请用完整句子说明设计，让不了解状态机内部格式的
> 读者也能理解。状态机所需的严格字段直接冻结在 develop run 根目录的 `state.json` 中。

## 设计摘要

- **算子名称：** `<算子名称>`
- **算子类别：** `<matmul、reduction、elementwise、attention 等>`
- **目标：** `<这个算子解决什么问题，以及用户可以观察到的行为>`
- **版本：** `1`
- **状态：** 草稿

## 背景与范围

说明为什么需要该算子、它会用于什么场景，以及本次设计明确包含和不包含的内容。
实现必须使用 CATLASS Python DSL（`python/tla_dsl`），采用 Python 函数、
`@tla.kernel`、`tla.Tensor` 和已有的 `tla.*` API，不设计 C++/CUDA C++ CATLASS
class 或模板。

## 知识依据

记录查询条件、知识版本、实际读取并采用的路径、适用约束，以及检索覆盖不足时的重查结果。

## 输入与输出

用业务含义描述每个张量。shape 可以使用符号维度，但必须在正文中解释约束。

| 名称 | 方向 | Shape | Dtype | Layout | 含义 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入 | `<符号 shape 与约束>` | `<支持的 dtype>` | `<逻辑和物理布局>` | `<张量含义>` |
| `output` | 输出 | `<符号 shape>` | `<输出 dtype>` | `<输出布局>` | `<结果含义>` |

## 计算语义

写出数学定义或清晰的伪代码，并说明：

- 广播、空输入、尾块、对齐以及非法输入如何处理；
- 累加精度、舍入方式、容差以及 NaN/Inf 行为；
- `reference.py`、`definition.json`、`workload.jsonl` 的相对路径和 SHA-256。

## 实现方案

### 算法

描述准备采用的 Python DSL 算法、关键 `tla.*` API，以及仍需验证的假设。

### 融合范围与性能依据

默认 `fusion.mode=performance`，关闭全部 anti-hack；只有用户明确要求单个融合 kernel
时才设为 `single_kernel` 并启用严格校验。记录用户约束，比较数据复用、GM 读写、launch
开销、存储容量、同步成本和并行度，说明各 kernel 的计算范围、调用顺序及中间张量。
没有实测数据的性能判断必须标记为待验证假设。

### 数据流

按执行顺序说明各 kernel 的加载、计算、写回和中间数据依赖。默认允许多个 executor
及 GM 中间张量；严格模式要求所有融合阶段在一个 kernel 内完成。

### Tiling 与 Layout

说明初始 tile、线程/核映射、逻辑布局与物理布局策略，以及选择这些参数的理由。

### 内存与同步

说明 GM/L1/L0/UB 等内存空间、buffer、pipeline、flag、mutex 或 barrier 的使用方式。

## 正确性验证

列出必须通过的代表性、边界和安全用例。这里展示便于评审的输入和预期行为；
精确的机器输入保存在 `workload.jsonl`。

另外列出 `focused_cases` 和实际执行命令。没有专用 focused 命令时执行批准的全量命令；
执行器应输出 argv、退出码、逐 case 状态和验证上下文摘要。覆盖全部 required cases 时，
该执行证据可在评审后复用，不再重复全测。

| 用例 | 类别 | 输入摘要 | 预期结果 |
| --- | --- | --- | --- |
| `correctness-main` | 正确性 | `<shape、dtype 和输入分布>` | `<与 Torch reference 的比较方式>` |

## 性能目标

无论是否设置性能门槛，都要说明最终 benchmark 方法、采样参数、聚合指标和 candidate
profiling 原始数据的保存位置。如果没有门槛，请明确写“本次不设置性能门槛，但仍执行
最终 benchmark 和 profiling”。默认结果记录 `anti_hack.status=disabled`；严格模式要求
`anti_hack.status=passed`，证明每次只 launch 一个声明的 kernel。如果有门槛，继续说明方向、阈值、最大优化轮数
和停止条件。

## 修改范围

列出允许修改的仓库相对路径，并用自然语言解释每个路径的用途。

- `<safe/repository-relative/operator.py>`：CATLASS Python DSL 算子实现。
- `solution_path`：完整源码包的仓库相对路径，默认与主实现文件同目录的 `solution.json`。

develop 一次生成全部 kernel、host 编排及完整 `solution.json`，随后才进行调试验证；
不进行任务拆分或逐任务上板。不涉及远程连接及文件同步实现。

## 风险与评审重点

说明 DSL 执行语义、同步、内存、layout、精度、公共接口或构建配置方面的风险，
以及 reviewer 应重点检查的内容。没有特殊风险时也应明确写出。

## 交付与证据

- 证据目录：`.catlass-dsl/develop-runs/<name>-YYYYMMDD-HHMMSS`
- 交付形式：工作树修改
- 知识收录：完成时批量准入

## 批准记录

- **批准状态：** 待批准
- **批准人：** `<批准后填写>`
- **批准时间：** `<批准后填写 RFC-3339 时间>`

批准表示评审者认可本文档描述的接口、语义、范围、验证方法和性能策略。设计发生变化
时，应建立新的 develop run、重新生成 state config 并重新批准。
