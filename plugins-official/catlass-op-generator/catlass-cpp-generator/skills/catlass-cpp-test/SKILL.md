---
name: catlass-cpp-test
description: CATLASS C++ 已注册算法族五阶段工作流的 05 算子测试，也可用于其他类型算子工程的用例泛化与用例设计覆盖评估。默认不询问用户、按工程类型分流：CATLASS C++ 五阶段工程只执行全量精度、功能、异常和性能验收的编排、失败恢复与性能候选迭代；其他类型算子工程直接执行泛化模式，由用户提供算子源码与入口后对多类型算子生成用例、执行用例并评估用例设计覆盖。CATLASS 五阶段工程下，仅当用户主动强调泛化、泛化性、泛化功能、泛化用例、覆盖评估或指定执行后端时，才追加泛化模式。触发条件：CATLASS 五阶段工程进入 validation 阶段且 validation_scope=full，或用户主动要求泛化与覆盖测试。
---

# 05 算子测试

## 模式判定（默认路径与触发条件）

**默认不询问用户，按工程类型直接分流。**

- **CATLASS C++ 五阶段工程** → 默认只执行「默认流程」，本文件中所有标注「泛化模式」的
  章节一律不进入。
- **其他类型的算子工程** → 默认执行「泛化模式」，「默认流程」不适用、直接跳过。
- **触发** → CATLASS C++ 五阶段工程下，用户主动强调泛化性、泛化功能、用例泛化、
  覆盖评估，或指定执行后端时，在「默认流程」之外启用「泛化模式」：执行全部标注
  「泛化模式」的章节。

触发词（必须是用户主动提出的需求，而非顺带提及）：泛化、泛化性、泛化功能、泛化用例、
覆盖评估、指定执行后端。

### 泛化模式的适用范围（可脱离 CATLASS 五阶段工程独立运行）

- **CATLASS C++ 五阶段工程**：泛化模式与「默认流程」并行执行，精度证据仍归「默认流程」
  的判定口径所有。
- **其他类型的算子工程**（如 Ascend C 直调工程）：跳过「默认流程」，只执行泛化
  模式章节，可对多类型算子生成用例并评估用例设计覆盖。此时不需要 `workflow.json`
  状态，也不需要 `reference/precision-policy.json` 或家族知识。

非 CATLASS 工程下的最小启动输入：

1. 算子源码根目录；
2. 入口文件名，或设计文档路径（至少提供其一；源码路径与入口文件名不得推测，必须由用户
   提供或确认候选）；
3. 测试工程落盘位置（用户指定，未指定时用当前工作目录）；
4. 执行后端可用性（在流程起点主动检测，不依赖用户口述）。后端不可用时退化为「仅生成
   用例」，仍交付用例配置、约束修正与覆盖评估报告。

判定结果写入本次任务的执行记录。非 CATLASS 工程下不得因缺少 workflow 状态或
`reference/precision-policy.json` 而拒绝执行泛化模式。

## 默认流程

进入条件：workflow 状态为 `validation`、`validation_scope=full`。
读取 `workflow/precision-policy.md` 和
`workflow/development-and-validation.md`。

`sparse_flash_mla` 另读家族 `validation.md` 与 `performance.md`；普通 SWA
同时应用其中的动态诊断、完整输出和流水验收规则。分别验证独立 metadata、主核与成对调用。
原 ATK 报告的实际 accuracy、LSE 全量比较和最终构建身份才是正式精度证据；
诊断脚本 PASS、进程返回0或仅 wheel 安装均不等同于精度 PASS。
手写基线从用户指定历史 JSON 读取，不重新执行；候选按任务规定的
5 次预热和 5 次正式样本取完整 device 主核均值，逐 case 比较。

所有阶段复用冻结的 `reference/precision-policy.json` 和本 Skill 唯一
`scripts/compare_precision.py`。全量验收覆盖功能、边界、异常、TilingKey/模板/运行分支及受影响
算子；性能使用统一模型用例和 `msprof op_summary` 的 `Task Duration(us)`。

单用例连续运行 60 秒无返回视为 kernel 超时。清理进程和设备资源后，将现象、Stage、
TilingKey 和 `blockDim` 写入 `docs/validation.md`；该用例不得计为通过，按失败证据进入对应
恢复分支。

失败按 workflow validator 的恢复矩阵一次更新全部状态字段。性能不达标保持
`stage=validation`、`issue_type=performance_optimize`；每轮只改变一个主要变量，候选先过精度
再同条件比较。修改设计假设返回 03，只改实现返回 04。最终候选必须重新执行 fresh 全量验收。

只有 full 精度、功能和要求的性能验收实际通过后，才能设置 `stage=complete`，清空问题字段并
整理验收 README。未达到目标时如实保留未达标状态，不降低门禁。

## 绝对不要做的事（NEVER 清单，泛化模式）

以下规则任何场景下不得违反：

| # | NEVER 规则 | 理由 |
|---|---|---|
| 1 | **禁止依赖记忆或历史版本读精度标准** | 精度标准持续更新，记忆会导致用过时数值 |
| 2 | **禁止推测算子源码路径/入口文件名/设计文档路径** | 算子目录名与入口文件名无可靠映射，推测极易出错；必须由用户提供或确认候选（置信度 ≥ 95%） |
| 3 | **禁止跳过源码定位流程直接分析算子** | 源码缺失会导致约束设计错误 |
| 4 | **禁止用 PyTorch 语义推断 C++ assert 条件** | C++ assert 与 PyTorch 行为不一致，必须从源码提取 |
| 5 | **禁止使用 `range: [1, N]` 范围方式定义 shape 维度值** | 执行后端按笛卡尔积展开会爆炸、生成卡死；必须用 `values` 离散列表 |
| 6 | **禁止硬编码 dtype_numbers、算子特定值域、rtol/atol 等数值** | 必须从接口契约或执行后端报告取得；全局默认分布见「用例生成约定」 |
| 7 | **禁止在 YAML/配置中写 `1e-5`，必须写 `1.0e-5`** | YAML 会将 `1e-5` 解析为字符串 |
| 8 | **禁止 NPU aicore 异常时尝试修复算子** | 这是算子实现缺陷，应收集日志报告算子开发人员 |
| 9 | **禁止用约束修正掩盖 INT32 溢出缺陷** | 溢出是算子 bug，约束应条件化跳过 2^31 修正，而非抹掉 |
| 10 | **禁止在 YAML 骨架阶段填入具体 dtype_numbers/extra_numbers 数字** | 必须填占位符，在约束清单完成后根据覆盖需求计算回填 |
| 11 | **禁止修改算子源码或执行后端代码** | 本 Skill 仅生成测试用例，不修改被测代码或测试框架代码 |
| 12 | **禁止定位和解决算子精度问题** | 精度问题属于算子实现缺陷，应收集日志报告；仅允许修复用例设计本身的问题（shape、dtype、数据生成错误等） |
| 13 | **禁止从 kernel_name 正则推断 dtype** | dtype 必须从接口契约/C++ 签名提取，不得从名称正则猜测 |
| 14 | **禁止在约束修正逻辑中定义参数范围** | YAML/配置负责定义参数候选空间（dtypes/values/ranges/shapes），约束修正只做"修正"不做"定义"；禁止在修正逻辑中硬编码 dtype 列表、shape 取值范围、attr 候选值 |
| 15 | **禁止将末维约束泛化为全维度上限** | 若 C++ `assert(dim[-1] == N)` 仅约束末维，只修正 `shape[-1]`，不得对其他维度做相同裁剪 |
| 16 | **禁止依赖用户口述判断执行后端可用性** | 必须在流程起点主动执行 `python -c "import atk"`（或等效命令）检测后端状态 |

## 环境阶段确认（泛化模式）

流程起点必须用 `AskUserQuestion` 主动询问用户：

1. **当前环境阶段**：
   - `仅生成用例（无 NPU）` → 仅执行用例生成与覆盖评估，产出可移交交付物
   - `生成 + 评估（无 NPU）` → 执行用例生成、覆盖评估，不执行精度比对
   - `完整流程（含 NPU）` → 执行生成、比对、评估全流程
   - 若用户选完整流程但观察到 NPU 不可用迹象（`torch_npu` 导入失败、`npu-smi` 无输出），立即暂停并提示确认

2. **算子关键信息**：源码根目录路径、入口文件名或设计文档路径。若用户未提供，可做候选搜索后交用户确认，禁止自行推测。

## 用例生成约定（泛化模式，每次任务必读）

本节只定**生成侧**取值；精度判定以执行后端报告为准，本 Skill 不重算阈值（NEVER #1）。

1. **Tensor 值域分布**：均匀/正态默认各占 50%，取值与 YAML 写法见下节「YAML 设计要点」；
   仅在需要偏离默认比例时才写 `custom_dist_ratio`。
2. **Attr 覆盖规则**：标量覆盖全部等价类；布尔覆盖 True/False；枚举覆盖全部取值；各类正交组合。
3. **特殊场景**：空 Tensor、标量 Tensor（shape `[1]`）、上/下边界、inf / -inf / nan。
4. **INT32 溢出**：由本 Skill 按 size/offset 乘法 pattern 自动推断触发，不由阈值表驱动。
5. **dtype 映射**：`torch.bfloat16→bf16`、`torch.half→fp16`、`torch.float32→fp32`、
   `double→fp64`；dtype 从接口契约提取（NEVER #13）
6. **精度标准名（`standard.acc`）**：取后端的精度标准注册项，本工作流对应
   `mixed_tolerance_bm_v2`（缺省 `default` 即生态开源混合容差标准）；独立算子无 workflow 时
   按后端默认，并在覆盖评估报告 §1「精度策略路径」注明取值来源。

## 用例生成流程（泛化模式）

### 步骤 0-2：源码定位与约束提取

**核心原则**：算子源码路径、入口文件名、设计文档路径属于关键信息，禁止自行推测（NEVER #2）。必须由用户明确提供，或候选搜索后由用户确认（置信度 ≥ 95%）。

**候选搜索策略**（辅助用户确认）：
- 按目标工程的入口文件命名约定搜索（如 aclnn 工程用 `Glob **/aclnn_*.cpp`，直调工程用
  `Glob **/*_host.cpp`），并配合 `Glob **/*<op>*.cpp`，按算子名 token 筛选，交用户选择
- `Glob **/<op>*/` 处理版本号（如 `xxx_v3`），多版本目录交用户选择最新
- `Glob **/docs/*.md` 交用户选择设计文档

**拒绝执行条件**：
- 用户未提供源码路径且候选未确认 → 拒绝执行
- 用户未提供入口文件名且候选未确认 → 拒绝执行
- 设计文档缺失但有完整 C++ 源码 → 警告"可能遗漏语义级约束"，继续执行，标注"来源：C++ 源码"
- C++ 源码和设计文档都缺失 → 拒绝执行

**局部推断**（用户确认源码根目录后允许）：
- 在该目录下按工程约定的目录继续定位子文件（如 `Glob **/op_tiling/*.cpp`、
  `Glob **/op_kernel/*.cpp`）
- `Grep` 搜索 `assert` / `TORCH_CHECK` / `OP_LOGE`
- 扫描 INT32 溢出风险 pattern
- 若算子支持 N 维但实际按低维计算，用例只覆盖实际计算维度

**创建测试工程目录**：`test_<算子名>/`，算子名用 snake_case。CATLASS 五阶段工程按 workflow 约定落盘；其他类型算子工程用用户指定位置，未指定时落在当前工作目录下。

**提取内容**：逐行提取所有 `assert` / `TORCH_CHECK` / `OP_LOGE` 等拦截语句，记录：
- 条件表达式（如 `hidden_dim == 128`、`dim % 16 == 0`）
- 错误消息
- 需要约束修正做什么（固定值、对齐、枚举限制、上限）

**约束来源优先级**：C++ assert > Python 校验 > 执行后端框架限制。C++ assert 是硬拦截，必须全部处理。多来源冲突时按此顺序取源，
并在约束清单中记录该冲突。

**产出约束清单表格**：约束名、来源（标注 C++ 源码，NEVER #4）、约束表达式、约束类型（shape/dtype/attr/align）、YAML/约束修正映射方式。

**✅ 检查点**：约束清单表格产出，每条约束来源标注为 C++ 源码。不通过 → 拒绝执行，提示用户补充源码或确认路径。

### 步骤 3：创建 YAML 配置与约束修正

**职责边界（NEVER #14）**：
- **YAML = 参数空间声明**：dtypes 候选、shape 维度值列表（`dim_values` 离散列表，NEVER #5）、值域范围、attr 候选值、分布配置
- **约束修正 = 修正逻辑**：将随机生成的参数修正到不触发 C++ assert 的合法范围；只做修正，不做定义；禁止硬编码参数范围

**最小 YAML 骨架**（真实 schema 为 `extra='forbid'`，未列出的字段一律报错）：

```yaml
name: torch.xxx
api: pytorch            # 框架默认值；该字段仅用于报表展示/兼容，aclnn 路径沿用默认
api_type: function      # 自定义算子须改为执行插件 @register 的注册名，见下方说明
version: v2.1
generate: <op>_constraint   # 须等于 constraint 中 @GENERATOR_REGISTRY.register 的注册名
# aclnn 路径（算子经 aclnn 调用时必填，见下方「执行后端」章节）
# aclnn_name: <算子名>          # 无 acl 前缀时后端自动补成 aclnn<name>
# aclnn_api_type: aclnn_function # 默认值；仅自定义 aclnn 执行插件时才改
standard:
  acc: <精度标准名>
  perf: not_key
shape_distributions:
  - [0, 1.0]              # 默认必须含 [0, 1.0]，缺失易导致生成死循环
dtype_numbers: <占位符>
extra_numbers: <占位符>
inputs:
  - name: input
    type: tensor          # 参数类型取值见下方
    required: true
    dtypes:
      values: [bf16, fp16, fp32]
    ranges:
      valid:
        values: [[-5, 5]]
        # random_types 只属于 ranges.valid / ranges.invalid，不在顶层或 inputs 层
        # random_types: [{name: nd, mean: [-5, 5], std: [0.1, 2]}]
      invalid:
        values: [["nan"], ["inf"], ["-inf"], ["null"]]
    shapes:
      dim_numbers:
        values: [1, 2]
      dim_values:
        values: [1, 64, 128] # 必须 RandomConfig；禁止裸二维列表与 range: [1, N]
      max_length: 4096       # 未设时后端默认 2^32（enhanced 策略下为 2^32-1）
    boundary:
      has_empty: true
      has_infnan: true
      has_scalar: true       # 标量 tensor（shape=[1]）
      has_lower_border: true
      has_upper_border: false
# outputs 为 out 变体专用：结果写回哪个入参（int 下标 / str 关键字名），不填表示直接 return
# outputs: 1
```

- 顶层只认 `name`/`aclnn_name`/`version`/`api`/`api_type`/`generate`/`standard`/`inputs`/
  `shape_distributions`/`dtype_numbers`/`extra_numbers` 等字段；`op`、`source_root`、`attrs`、
  `reduction` 等均不存在，出现即报 `Extra inputs are not permitted`。
- `inputs[].type` 仅接受：`tensor`/`tensors`/`tensor_tuple`/`scalar`/`scalars`/
  `scalar_tuple`/`attr`/`attrs`/`attr_tuple`。
- `dtypes`/`ranges.valid`/`ranges.invalid`/`shapes.dim_numbers`/`shapes.dim_values` 等取值字段
  统一用 `{values: [...]}` 形式，不能直接写裸列表。
- `api_type` 必须等于执行插件 `@register("<name>")` 的注册名；`function` 是后端内置执行器
  （按 YAML `name` 求值），仅适用于内置 torch 函数，自定义算子写成 `function` 会在执行期报
  `NameError`。自定义执行方式见「执行后端」章节的 `03 扩展开发/自定义执行方式.md`。
- aclnn 路径：算子经 aclnn 调用时 `aclnn_name` 必填（无 `acl` 前缀时后端自动补成
  `aclnn<name>`），`aclnn_api_type` 默认 `aclnn_function`。`api_type` 与 `aclnn_api_type`
  是两条独立路由，`atk aclnn` 只走后者；默认值够用条件见「执行后端」章节。
- aclnn 路径不在 YAML 声明出参：后端按参考输出自动生成 out 张量，再按 c++ 原型顺序追加到
  入参尾部。把 out 写成 `inputs` 一项会打乱参数顺序；`outputs` 只服务于 torch out 变体。
- `generate` 必须等于 constraint 中 `@GENERATOR_REGISTRY.register("<name>")` 的注册名，
  否则该约束生成器不会被加载。

**YAML 设计要点**：
- `dtype_numbers` 先填占位符（NEVER #10），约束清单完成后回填
- `dim_values` 用离散列表：2^n 与 2^n-1 边界对，逐对覆盖到大 shape
  （如 `[1, 2, 3, 4, 7, 8, ..., 524287, 524288, 1048575, 1048576]`，勿止步于 2^16）
- 单输入元素数上限 `max_length` 未设时后端默认 2^32（enhanced 策略下为 2^32-1），
  在约束修正中做二次校验
- 均匀分布：`ranges.valid.values: [[-5, 5]]`
- 正态分布：`ranges.valid.random_types: [{name: nd, mean: [-5,5], std: [0.1,2]}]`
- 混合分布：`ranges.valid.custom_dist_ratio: [[0.5, 0.0], [0.5, 0.0]]`
- 布尔参数：`type: attr`, `dtypes: {values: [attr_bool]}`, `ranges.valid.values: [true, false]`
- 枚举参数：`type: attr`, `dtypes: {values: [string]}`, `ranges.valid.values: [所有枚举值]`
- 标量参数：`type: attr`, `dtypes: {values: [int/float]}`, `ranges.valid.values: [等价类值]`
- 边界声明：`boundary` 字段启用 5 类边界场景（`has_empty` 空 tensor、`has_scalar` 标量、`has_infnan`
  inf/-inf/nan、`has_upper_border` 上边界、`has_lower_border` 下边界）；边界用例数量由 `extra_numbers` 控制
  （无特殊要求时取 8，按需覆盖的边界场景数调整；留空则后端默认 `all`，规模易失控）
- 极端值覆盖：`ranges.invalid.values` 列出 `[["nan"], ["inf"], ["-inf"], ["null"]]`，覆盖 happy path
  之外的健壮性取值

**dtype 等价类剪枝**：若 C++ 内核对 FP16/BF16 走相同计算路径（如统一转 FP32），两者精度等价，可合并为"低精度"类取并集，
`dtype_numbers` 相应降低。

**多 attr pairwise 剪枝**：多 attr 全组合易爆炸（3 attr × 4 值 = 64 组合），改用 pairwise（2-way）覆盖数组可降至 ~16
组合。算子特性必测组合强制保留。

**约束修正模板**（每条 C++ assert 必须有对应修正逻辑）：

| C++ assert 模式 | 修正代码 |
|---|---|
| `assert(dim == N)` | `shape[idx] = N` |
| `assert(dim % N == 0)` | `shape[idx] = (shape[idx] // N) * N` |
| `assert(dim >= N)` | `if shape[idx] < N: shape[idx] = N` |
| `assert(dim <= N)` | `if shape[idx] > N: shape[idx] = N` |
| `assert(is_tensor_in_dtypes<A, B>(x))` | `if dtype not in ["a", "b"]: dtype = "a"` |
| `assert(!param)` | YAML 固定 `values: [false]` |
| `assert(param)` | YAML 固定 `values: [true]` |
| `assert(aligned)` | `shape[idx] = (shape[idx] // align) * align; if < align: = align` |
| `assert(dim_a % dim_b == 0)` | `dim_a = (dim_a // dim_b) * dim_b` |

**约束生成器与执行插件**：完整模板见官方 `03 扩展开发/自定义参数约束.md` 与
`03 扩展开发/自定义执行方式.md`。两条注册名必须与 YAML 对齐：`generate` ↔
`@GENERATOR_REGISTRY.register`，`api_type` ↔ `@register`。文件名 `<op>_constraint.py` /
`execute_<op>.py`（aclnn 路径通常无需自写执行插件）。离线时按「符号 ← 模块」导入（执行后端专有）：
`GENERATOR_REGISTRY` ← `atk.case_generator.generator.generate_types`；
`CaseGenerator` ← `atk.case_generator.generator.base_generator`；
`CaseConfig` ← `atk.configs.case_config`；`register` ← `atk.tasks.api_execute`；
`BaseApi`/`AclnnBaseApi` ← `atk.tasks.api_execute.base_api`、
`atk.tasks.api_execute.aclnn_base_api`；`InputDataset` ← `atk.configs.dataset_config`。
约束生成器继承 `CaseGenerator` 并改写
`after_case_config(self, case_config: CaseConfig) -> CaseConfig`（就地修改并返回该对象）；
执行插件入口签名为 `__call__(self, input_data: InputDataset, with_output: bool = False)`。

约束修正体内通过 `case_config.inputs[i]` 读写 `name`/`type`/`dtype`/`shape`/`range_values`
（另有 `is_range_null()`/`numel()`）；执行插件体内通过 `input_data.args` / `input_data.kwargs`
取入参。aclnn 执行插件继承 `AclnnBaseApi`（`@register` 名 = YAML 的 `aclnn_api_type`），
默认 `aclnn_function` 一般无需自定义。

**维度索引精确性（NEVER #15）**：若 `assert(shape[-1] == N)` 只约束末维，只修正 `shape[-1]`，禁止泛化为全维度上限。

**✅ 检查点**：
- YAML 与接口契约签名一致（dtype 映射正确、参数类型正确）
- 职责边界：YAML 声明参数空间，约束修正仅做修正
- YAML：`dim_values` 用离散列表 + 含非对齐值 + `dtype_numbers` 占位
- 约束修正：每条 C++ assert 有对应修正逻辑 + 维度索引精确

### 步骤 4：生成用例

**执行后端检测（NEVER #16）**：必须主动执行 `python -c "import atk"` 检测后端可用性：
- **输出版本号 → 可用**：正常执行
- **报错 `ModuleNotFoundError` → 不可用**：进入降级模式（仅生成 YAML/约束修正源码，跳过用例生成），提示用户"需在后端环境执行步骤 4"
- **其他报错** → 交用户判断

**门禁原则**：必须先 dry-run，再正式生成；命令加 `timeout 60` 防卡死。

**执行流程**：
1. **Dry-run**：验证 YAML + 约束修正组合能正常工作
2. **回填 `dtype_numbers`**：按约束清单回填（经验范围 20000，根据 dtype 等价类数、shape 取值个数、
   attr 等价类数调整，确保组合充分覆盖）
3. **正式生成**

```bash
# 以下命令属执行后端专有；更换后端时整块替换
timeout 60 atk case -f test_<op>.yaml -p <op>_constraint.py -dt 1 -en 2  # dry-run
atk case -f test_<op>.yaml -p <op>_constraint.py                         # 正式生成
```

其余参数见 `atk case --help` 与官方 `01 基础操作/用例生成.md`。产物落在执行目录的
`result/<yaml 名去后缀>/json/all_<yaml 名>.json`。

**✅ 检查点**：dry-run 通过（无报错），`dtype_numbers` 已回填。不通过 → 修正后重跑。

### 步骤 5：编译算子（用户环境完成）

算子编译部署由用户在 NPU 环境完成。

**✅ 检查点**：算子已部署，可正常导入执行。不通过 → 检查编译环境变量与工程编译错误。

### 步骤 6：精度比对

**策略**：用例生成至少 100 条；冒烟测试随机抽取 25 条常规用例 + 智能分类抽取边界用例（每类 1 条，约 5-8 条），总计约 30-33 条。当前冒烟采样暂不含
`upper_border`，保留在全量集中。

```bash
python scripts/sample_quality_cases.py \
  -i <全量用例.json> -o <冒烟用例.json> -n 25 -b smart -s 42
```

`-b smart` 按边界类别各抽一条，`-s` 固定随机种子保证可复现。

**失败分类**：
- **用例设计问题**（shape 不匹配、dtype 不一致、数据生成错误）→ 可修复
- **算子实现问题**（精度差异、NPU aicore 异常）→ 不修复，收集日志报告

**NPU aicore 异常识别**（关键词命中即判定）：

| 异常关键词 | 含义 |
|---|---|
| `aic-error` / `aicore error` | aicore 执行错误 |
| `DDR address out of range` | DDR 地址越界 |
| `aic overflow` / `aic underflow` | aicore 溢出/下溢 |
| `TSCH timeout` | 任务调度超时 |
| `hccl error` | 多卡通信异常（单卡算子一般不会出现） |
| `ASCEND_RT_AICPU_PATH` | aicpu 路径错误 |
| `Device kernel error` | NPU kernel 加载/执行失败 |
| `bad address` | 非法地址 |

**识别方式**：
1. **自动**：执行插件在 `except` 中匹配上表关键词，命中时抛出带 `[NPU_AICORE_ERROR]` 前缀的 `RuntimeError`
2. **手动**：在日志中搜索 `[NPU_AICORE_ERROR]` 或上表关键词

**处理流程**：
1. 收集触发异常的 shape/dtype/参数组合与完整错误日志
2. 标记为「NPU 实现缺陷，需算子开发人员修复」
3. **不尝试修复**（NEVER #8）
4. 继续执行其余用例，收集完整失败分布

**报告模板**：
```ini
⚠️ NPU aicore 异常报告
算子名称: <op_name>
异常类型: <如 DDR address out of range>
触发条件: shape=<如 [256,128]>, dtype=<如 bf16>, 参数=<如 trans=True>
错误日志: <完整错误信息>
结论: 算子 NPU 实现存在缺陷，需算子开发人员修复
```

**✅ 检查点**：精度报告生成，含 NPU vs CPU 比对结果。不通过 → 区分精度问题 vs aicore 异常，按对应路径处理。

### 步骤 7：用例设计覆盖评估报告

**前置条件**：用例生成完成（或降级模式下约束清单已产出）。不依赖 NPU，可在生成后立即执行。

**8 维度评估**（每项先结论后明细）：

1. **基本信息**：算子名、源码路径、入口文件、测试工程目录、用例 JSON 路径、总用例数、生成时间、后端可用性、精度标准版本
2. **约束覆盖分析**：逐条比对 C++ assert 与约束修正逻辑；覆盖率 = 已覆盖/总 assert 数；未覆盖清单（若有）+ 原因 + 建议
3. **Shape 覆盖分析**：维度取值是否为离散列表；是否含 2^n 与 2^n-1 边界对（覆盖 tiny/large/上下边界）；
   `boundary` 字段是否启用（覆盖空 tensor、标量、inf/nan）；`ranges.invalid` 是否配置极端值；
   8 类边界场景（empty_shape/zero_dim/scalar/tiny_shape/large_shape/dtype 边界/
   upper_border/lower_border）逐类覆盖情况；ND 维度处理说明
4. **dtype 覆盖分析**：等价类剪枝依据（C++ 计算路径）；各 dtype 用例分布；零覆盖 dtype（若有）
5. **Attr 覆盖分析**：等价类值覆盖；pairwise 前后组合数；强制保留的算子特性 case
6. **INT32 溢出看护状态**：是否扫描 size/offset pattern；是否命中 int32；看护开关取值；
   触发依据（源码位置）；是否对超大 shape 跳过 2^31 修正
7. **用例规模统计**：`dtype_numbers` 回填值；`extra_numbers`；总用例数；是否在预期区间
8. **汇总与建议**：各维度评级（充分/一般/不足）；遗漏点 + 影响 + 改进建议；是否建议进入全量执行

**降级模式**（后端不可用、无用例 JSON）：「当前覆盖情况」列填 YAML/约束修正设计值并标注「降级（设计值）」；依赖用例 JSON 的统计项填 N/A。

**输出**：按本 Skill 的 `templates/test_case_design_report.md` 的 8 个章节填写。CATLASS 五阶段
工程写入 `docs/validation.md`；其他类型算子工程输出到用户指定路径，未指定时写入测试工程
目录下的 `coverage_report.md`。

**✅ 检查点**：8 维度全部填充，无未替换占位符；约束覆盖逐条核验。不通过 → 补充缺失数据后重新生成。

## Bug-Hunting 用例设计（泛化模式）

用例好坏的衡量标准是**能否识别代码中的 bug**，而非通过率。三种模式配合使用，缺一不可：

| 模式 | 做法 | 落地字段 |
|---|---|---|
| 边界 | 覆盖空 tensor、标量、inf/-inf/nan、上边界（某维取最大值）、下边界（所有维为 1）5 类场景 | YAML `boundary` 字段 + `extra_numbers` |
| 极端值 | 覆盖 `nan` / `inf` / `-inf` / `null`（空 tensor）等 happy path 之外的取值；`null` 用 `expected_error_msg` 校验算子是否正确报错 | YAML `ranges.invalid.values` |
| 算子特性 | 按算子类型补必测项，见下表 | 约束修正中的强制保留组合 |

| 算子类型 | 必测项 |
|---|---|
| GEMM 类 | 转置分支（trans_a / trans_b 各组合）、内积维度匹配（K 维对齐） |
| Attention 类 | GQA 比例（q_head_num / kv_head_num）、KV cache block 数量、block_size 对齐 |
| Reduction 类 | reduce 维度为 1、reduce 全部维度、输入全相等（极差为 0）、空 reduce |
| LayerNorm 类 | hidden_dim 对齐（16/32/64）、normalize 维度全 1、affine=True/False |
| 激活函数类 | 输入为 0、负数极大、nan 透传、inplace 行为 |

## INT32 溢出看护（泛化模式）

**背景**：算子 C++ 内核若用 `int32` 计算 shape 维度乘积（`total = M * N`、`offset = idx * stride`），乘积超
2^31 时溢出。INT32 溢出是需要主动构造的看护场景（非拦截条件），约束不修正而是保留以暴露 bug（NEVER #9）。

**扫描范围**（按风险层级）：

| 源码层级 | 是否扫描 | 说明 |
|---|---|---|
| `op_host/op_api/*.cpp` | 必扫 | 入口层，含 workspace size 计算 |
| `op_host/op_tiling/*.cpp` | 必扫 | tiling 层，含 M*N*K 等 shape 乘法（最高风险） |
| `op_host/*_infershape.cpp` | 选扫 | 通常用 int64，风险较低 |
| `op_kernel/*.cpp` | 选扫 | 通常用 uint64，但仍有风险 |

**扫描 pattern**（命中任一即启用看护）：

| Pattern | 含义 |
|---|---|
| `int32_t.*\*.*int32_t` | int32 乘 int32 |
| `int [a-z_]+ = .* \*` | int 变量赋值为乘法 |
| `(size|offset|total|workspace).* = .*\*` | size/offset/total 计算 |
| `.* = .*shape.*\*` / `.* = .*\*.*shape` | shape 乘法 |

**「2^31 修正」**：未启用看护时，约束修正把 shape 乘积裁剪到 2^31 以内，规避内核 int32 乘法溢出；
启用看护时不做该裁剪，让溢出用例真实构造出来（NEVER #9）。开关即本节「是否启用看护」的判断结果，
不额外引入 YAML 字段。

**启用后**：约束修正对 shape 乘积 > 2^31 的用例跳过 2^31 修正，仅保留 OOM 硬上限防护（如 2^33）。

**保守策略**：无法确定是否溢出时，默认启用看护。多生成几条溢出用例的代价远小于漏掉一个溢出 bug。

## 执行后端（泛化模式）

用例由执行后端在 NPU 与 CPU 上双端执行，逐 case 比对并生成报告。后端未安装时先安装再执行，不得跳过。

本 Skill 的默认执行后端是 ATK：命令名为 `atk`，导入路径以 `atk.` 开头，执行期行为与报告字段
均由该后端定义。本文件标注「执行后端专有」的命令、导入路径与配置面在更换后端时需整块替换。

YAML 字段以「步骤 3」内联的最小 schema 为准（离线可用）；以下官方仓库文档作为补充参考，
用于更完整的语法、constraint 模板、执行插件编写与命令行参数。
该仓库及其文档属执行后端专有：
`https://gitcode.com/Ascend/ATK.git` 中的 `docs/ATK使用指南/`：
- `01 基础操作/用例设计.md`：YAML 配置与字段说明
- `01 基础操作/用例生成.md`：用例生成流程与 dry-run
- `01 基础操作/任务执行.md`：执行命令与参数
- `02 参考资料/用例设计文件说明.md`：YAML 完整语法参考
- `02 参考资料/常用执行命令说明.md`、`任务执行参数说明.md`：执行命令与参数
- `03 扩展开发/自定义参数约束.md`：constraint 编写
- `03 扩展开发/自定义执行方式.md`：执行插件编写
- `05 后端资料/fusion.md`：自定义算子执行插件 + constraint 注册完整示例

**执行命令（执行后端专有）**：torch 路径 `atk pytorch <用例.json> -p <执行插件> --task accuracy`，aclnn 路径
`atk aclnn ...`，两者都需 `PYTHONPATH=<test_dir>`，可选 `-s/-e` 限定起止 case、`--devices`
指定卡。前者默认主后端 npu、对比后端 cpu，后者默认对比 cpu。完整参数见官方
`02 参考资料/常用执行命令说明.md`、`任务执行参数说明.md` 与 `examples/用例执行/README.md`，
或本地 `atk pytorch --help`。

精度判定以执行后端报告为准：执行后自动生成精度比对报告，含「精度比对通过率」「精度是否达标」列，
该结论为最终精度证据。`workflow/precision-policy.md` 与 `scripts/compare_precision.py` 作为 catlass
工作流的补充验证手段，不替代执行后端判定。

## 常见失败速查（泛化模式）

部分条目涉及执行后端专有的命令与报错（如 `atk case`、`oversize_case`）。

| 现象 | 原因 | 处置 |
|---|---|---|
| `Tried to instantiate class ... but it does not exist`（NPU 路径） | 算子未编译进 .so | 重新编译并部署到 publish 目录 |
| `hidden dim must be divisible by N` | 约束修正漏了对齐 | 补对齐逻辑后重新 `atk case` |
| `Input 1e-5 is invalid` | YAML 将 `1e-5` 解析为字符串 | 改用 `1.0e-5`（NEVER #7） |
| `Permission denied` 写文件，或文件属主与当前用户不一致 | 宿主机与容器内用户不一致 | 用 `docker exec <容器名> bash -c "..."` 执行，或在容器内 `chown -R root:root <目录>` |
| 修完约束修正仍报同一错误 | 未重新生成用例 JSON | 重新执行 `atk case -f <yaml> -p <plugin>` |
| 总输出元素不足 100,000 | `dim_values` 范围过小 | 下探维度下限或增大上限 |
| `--limit is not a valid option` | 旧参数已废弃 | 改用 `-e N` |
| 用例全是 happy path | 未启用 `boundary` 与 `ranges.invalid` | 见「Bug-Hunting 用例设计」 |
| NPU 行 accuracy 为 `-`（执行成功但无输出） | aicore 异常致输出未保存，或特定 shape/dtype 下静默失败 | 用 Python 在该 shape/dtype 上直接复测；命中异常则按 aicore 路径处理 |
| INT32 看护用例 OOM | 溢出档 shape 过大（如 2^31 单元素 tensor） | 降至 `2^31 + 2^28`，或多维分解（如 `[65536, 32769]`） |
| `oversize_case: true` 警告但用例仍生成 | 框架对 > 2^31 用例的软限制提示 | 正常现象；启用看护时属预期，否则检查 2^31 修正是否生效 |
| INT32 看护用例 NPU 结果 ≠ CPU | 内核未做 int64 提升，shape 乘积超 2^31 溢出 | 算子缺陷，收集日志报告，不用约束修正掩盖（NEVER #9） |

## 职责边界（泛化模式）

- 不修改算子源码与执行后端代码（NEVER #11）
- 不定位、不修复算子精度问题；只修复用例设计自身缺陷（NEVER #12）
- 不在约束修正中硬编码精度策略已提供的阈值（NEVER #6）
- 不在约束修正中定义参数范围，只做修正（NEVER #14）
- 不将末维约束泛化为全维度上限（NEVER #15）
- 不依赖记忆读精度标准，每次任务必读最新版（NEVER #1）
- 不推测算子源码路径，必须由用户提供或确认（NEVER #2）
- 不用 PyTorch 语义推断 C++ assert，必须从源码提取（NEVER #4）
- 不用 range 范围定义 shape 维度值，必须用离散列表（NEVER #5）
- 不在 YAML 中写 `1e-5`，必须写 `1.0e-5`（NEVER #7）
- 不在 YAML 骨架阶段填具体 dtype_numbers（NEVER #10）
- 不从名称正则推断 dtype（NEVER #13）
- 不依赖用户口述判断后端可用性，必须主动检测（NEVER #16）
- NPU aicore 异常不修复，收集日志报告（NEVER #8）
- INT32 溢出不用约束修正掩盖，保留以暴露 bug（NEVER #9）