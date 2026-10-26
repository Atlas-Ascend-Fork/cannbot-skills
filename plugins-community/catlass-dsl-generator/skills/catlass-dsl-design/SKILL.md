---
name: catlass-dsl-design
description: Use when specifying a new CATLASS Python DSL operator and producing its Torch oracle, definition, workloads, executable reference evidence, and approved Markdown preliminary design
---

# CATLASS DSL Design

本 Skill 负责输出算子规格、初步设计和 Torch 精度标杆，产物是供用户阅读和批准的
`DESIGN.md`、供 develop 初始化的 run 根 `state.json`、可导入的 `reference.py`、
`definition.json`、`workload.jsonl` 及 reference case 执行证据。不实现 CATLASS DSL
solution，不执行任务拆分、review、性能测试或优化。三个 JSON/JSONL benchmark 输入
是算子数据规格，不是设计文档的并行表达。

## 语言与实现边界

CATLASS DSL 是以 Python 语法编写的算子 DSL，不是 C++ 算子库。设计必须面向
`python/tla_dsl` 的 Python 接口和执行模型：

- 算子入口采用 `@tla.kernel`、Python 函数、`tla.Tensor` 和 `tla.*` API；
- 算子实现文件应为 `.py`，代码模式必须给出 Python DSL，而不是 C++、CUDA C++、
  Ascend C 或伪装成 DSL 的 C++ 模板代码；
- 不得设计 C++ class/template、头文件、namespace、宏、指针算术、RAII 或
  `<<<grid, block>>>` launch；
- GM/L1/L0/UB、tiling、layout、copy、MMAD、vector、flag 和 mutex 等概念必须映射到
  CATLASS Python DSL 的现有接口；
- C++、MLIR 或生成代码只能作为后端实现和问题定位资料，不能成为算子公开接口或
  初步实现方案。
- 默认 `fusion.mode=performance`，按整体性能最大化确定融合范围。比较数据复用、GM
  读写、launch 开销、存储容量、同步成本与并行度，允许多个 `@tla.kernel`、多个 executor、
  GM 中间张量及 shape/dtype/layout 分派；计算主体仍使用 CATLASS Python DSL。
- 只有用户明确要求单个融合 kernel 时设置 `fusion.mode=single_kernel`：一次 host `run`
  只 launch 一个 kernel，solution 只声明一个 `@tla.kernel`，所有编译期变体复用它，
  不得为编译期变体声明独立 kernel，不得通过 GM 临时张量连接多个 executor。
- 默认关闭全部 anti-hack；严格单融合模式才启用源码与逐次 launch 的完整 anti-hack。
- 每个 `@tla.kernel` 必须自包含：除形式参数、函数内局部值、Python 内建符号和受信任的
  `tla` API 命名空间外，不得读取模块级 tile、shape、dtype、params 对象、开关或其他
  用户定义全局值，也不得捕获 closure 配置。
- 固定 tile/dtype/params 在 kernel 函数内定义；实际 shape 和其他运行时配置通过显式
  scalar/tensor 参数、tiling data 或 tensor metadata 取得。shape/dtype/layout 等编译期
  特化由 decorated kernel 的形式参数类型和 metadata 驱动，host 不得通过
  `global`、模块属性或可变容器改写语义。默认模式允许显式选择不同 kernel。

若目标仓库同时包含 C++ CATLASS 与 CATLASS DSL，必须先定位 Python DSL 样例和
`catlass.tla` 导入方式。没有真实 Python DSL API 支持的设计项应标记为待验证假设，
不得用相似的 C++ API 补齐。

## Torch 精度标杆

设计阶段必须在当前 workspace 中生成项目专用 `reference.py`。它只能使用公开
`torch` API 表达算子数学语义，不得 import CATLASS DSL candidate、复用 candidate
输出或调用待实现算子。reference 默认在 CPU 上执行；只有语义确实依赖设备且 Torch
CPU 无等价表达时才使用目标 device，并在环境缺失时阻止设计批准。

`reference.py` 必须导出与 `catlass-dsl-bench` 完全相同的 `run`：

```python
def run(*inputs):
    """Return one Tensor or a flat tuple/list ordered like definition.outputs."""
```

用例不在 `reference.py` 中声明。设计阶段按
[`catlass-dsl-bench`](../catlass-dsl-bench/SKILL.md) 接口生成：

- `definition.json`：输入输出规格及完整 Torch `reference` 源码，其中 `axes` 固定为空对象，
  `reference` 必须与同批生成的 `reference.py` 逐字一致；
- `workload.jsonl`：每行一个 required case，`uuid` 必须等于 `case_id`；
- `solution.json` 由 develop 在实现完成后生成，不属于 design 输出。

case 规则：

- 不使用 axis binding：`definition.json` 的 `axes` 必须为 `{}`，每行 workload 的
  `axes` 必须为 `null`；禁止用 axis 名称、`const`、`var` 或 `expr` 间接表达 shape；
- `inputs`、random/scalar/null 来源和 tolerance 全部使用公共 `bench.py` 字段，不增加
  design 私有字段；每个 tensor input 必须在 workload 中直接写出数值 `shape` 和
  `dtype`，标量直接写 `value`，可选空输入直接写 `null`；
- 每个 case 的 shape、dtype、layout selector、序列边界、布尔开关、标量参数以及初始
  state 是否存在等所有决定用例配置的数据，必须直接保存在该 case 对应的
  `workload.jsonl` 行中；`uuid` 只标识用例，不得承担配置查表功能；
- 允许使用 `custom_inputs_entrypoint` 和 `type: custom` 生成需要特殊分布或约束的输入，
  但不得借此恢复 axis binding；shape、dtype 和其他用例配置仍须在 workload 中显式保存。
  禁止在 `reference.py` 中维护 `_CASE_SPECS` 等 case table，也禁止通过 `CASE`、
  `case_id`、`case_index`、序号或其他不透明配置索引选择 shape、dtype、layout 或开关；
- 边界、尾块和广播尺寸分别用独立 workload 表达；随机输入必须固定 CLI `seed`；
- tolerance 必须来自数值设计，不能为通过测试而放宽；
- definition/workload 不保存 solution 输出或伪造 expected。

使用公共校验器运行设计用例：

```bash
python3 skills/catlass-dsl-design/scripts/validate_reference.py \
  --reference <reference.py> \
  --definition <definition.json> \
  --workload <workload.jsonl> \
  --state <develop-run>/state.json
```

校验器只导入用户信任的 `reference.py`，该模块可以执行任意 Python 代码。所有设计
case 会直接复用 `bench.py` 的 definition/workload 校验、设备初始化、输入
生成和 Tensor 输出树规则。全部通过后，证据记录 `reference.py`、`definition.json`
和 `workload.jsonl` 的 SHA-256；把 case 状态、输出结构、输入路径和完整摘要归一化到
`state.json.config.reference_validation`，不生成 `reference-cases.json`。同时把相对路径和
完整摘要写入 state config 的 `semantics.computation`，并在 `DESIGN.md` 中用可读
文字展示，随设计一同获批。`reference.py`、`definition.json`、`workload.jsonl` 的
校验口径（用例集、容差、输入契约）一旦获批即冻结；任一变化必须重新运行全部 case、
更新摘要并增加 revision。**但新增或修改 `DESIGN.md` 的实现方案不需要重新批准**——
见下方"设计是工作草稿"。

## 过程

1. 只读探索目标仓库，确认 `python/tla_dsl` 来源、Python 导入方式、已有 `.py` 算子
   模式、构建/测试入口和参考实现。
2. 明确算子名称、family、用途和可观察行为。按 `catlass-dsl-knowledge` 初始化并查询与
   当前算子、目标架构及设计问题相关的知识，完整读取实际采用的条目；检索覆盖不足时按
   该 Skill 的通用规则显式重查。将查询条件、知识版本、采用路径和适用约束写入 `DESIGN.md`。
3. 固定每个输入输出的 symbolic shape、dtype、layout 和语义。
4. 写清数学计算、广播/尾块/空输入等边界行为，以及累加精度、舍入和 NaN/Inf 规则。
5. 生成 Torch `reference.py` 和 `definition.json`，并把每个非 performance required
   case 写成 `workload.jsonl` 中的一行。
6. 运行全部 reference cases，将结构化结果归一化进 `state.json` 并冻结三个输入文件的摘要。
7. 使用 CATLASS Python DSL API 给出融合范围、各 kernel 的计算职责与调用顺序、
   中间张量、tiling/layout、内存空间、pipeline 和同步方案，解释性能选择依据。
   tiling 的块数按核数取整（块数接近核数整数倍），行块大小对齐 cube fractal（16 行）；
   依据见 `dsl/pattern-gemm-optimization-ladder`。
   写明 `fusion.mode` 及单融合要求是否由用户明确提出；没有实测支持的性能判断标记为
   待验证假设。明确固定配置的局部定义和运行时配置的显式来源。
8. 明确正确性 oracle、required cases、`focused_cases`、对应的实际 focused/full_test 命令、
   风险、允许路径、必需 benchmark/profiling 命令和可选性能门槛。没有专用 focused 命令时
   使用批准的 full_test；要求执行器输出实际 argv、退出码及逐 case 结果，以便完整覆盖时
   复用验收证据，协议见 [证据复用](../catlass-dsl-develop/references/verification-reuse.md)。
9. 从 [`templates/DESIGN.md`](templates/DESIGN.md) 创建面向用户的设计文档。
10. 在 develop run 根目录的 `state.json` 中冻结配置，并写入
    `DESIGN.md` 的 SHA-256。
11. 执行 state 结构与仓库身份校验，向用户展示 `DESIGN.md`、Resolved Plan 和 reference
    case 结果并取得批准。

```bash
python3 skills/catlass-dsl-develop/scripts/develop_state.py validate \
  --state <develop-run>/state.json
```

初步设计不是已验证实现结论，不得虚构 benchmark 或 profiling 数据。涉及 DSL
执行语义、同步、内存、layout、精度 oracle、公共接口或构建配置时，
`risk_level` 必须为 `high` 并填写 targeted review focus。

**设计是工作草稿，不是合同。** state config 中的评测口径（allowed paths、required cases、
benchmark argv、融合模式、anti-hack 策略）初始化后不可修改；但 `DESIGN.md` 中的实现方案
（tiling、数据流、buffer 规划、同步策略、kernel 划分）允许在 develop 阶段随实测修改：

- 修改后不新建 develop run、不需重新批准；
- 在 `ITERATIONS.md` 记录一条"设计修订"（改了什么、为什么改、依据哪次实测）；
- 只有最终交付（`finish`）时才把 DESIGN.md 固化为只读快照。

设计应当在"写→跑→看数"的循环中收敛，而不是在动笔前一次性消灭所有不确定性。

批准前必须再次检查：算子主体和 reference 都是 Python 文件；算法和数据流引用的
`tla.*` 接口可以从本地源码或 OKF 知识中定位；标准 definition/workload 覆盖并通过
全部非 performance required cases；正文包含三个文件的实际 SHA-256；没有把 C++
CATLASS API 当作 CATLASS DSL API；每个 decorated kernel 自包含；融合边界有明确性能
依据。仅在 `single_kernel` 模式检查唯一声明、单次 launch 和无独立变体分派。

所有设计都必须批准 benchmark argv、metric JSON path，并设置 `profiling_required=yes`，
以便 develop 保存最终性能现状和 candidate 原始 profile。声明性能门槛时还必须批准方向、
阈值、迭代上限、停滞阈值和最小提升比例；optimization config 的 allowed paths
必须且只能声明一个 kernel 文件，不再声明额外工作目录。没有性能门槛只表示不触发 optimize，
不表示跳过 benchmark 或 profiling。

输出只有获批的可读 `DESIGN.md`、与其摘要绑定的 state config、冻结的
`reference.py`、`definition.json`、`workload.jsonl`、执行证据及其摘要。
`catlass-dsl-develop` 读取 state config 后一次生成完整实现与 `solution.json`，再进行
focused_test、知识查询与调试修改、review、完整测试、最终 benchmark/profiling 和可选优化。
代码仍集中在一个主 Python 实现文件，允许在文件内定义多个 kernel。冻结 `solution_path`，
并在 benchmark argv 中设置 `--anti-hack off`（默认）或 `--anti-hack single-fused`（严格模式）；
后续 optimize 必须继承该设置。
