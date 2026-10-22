---
name: ops-error-code-remediation
description: 为新增或已完成的CANN C/C++算子设计、接入、审计和整改错误码上报路径，覆盖实际log.h解析、失败出口治理、最小编译、负向测试与Markdown报告。触发：用户要求新增算子错误码开发，或扫描、重构、验证既有算子的错误码与失败返回；
---

# CANN算子错误码开发与重构整改

本Skill消费cann-error-code-standard提供的唯一基准，负责把基准落实到具体算子源码。不要在本目录复制错误码清单、来源或通用开发规范。

## 模式选择

- **方案咨询**：用户只询问设计方法、核验流程或判定依据时，按相关基准和参考给出方案、所需证据与验证命令，区分“建议执行”和“实际已验证”。未提供目标算子或源码时，不自行选择样例算子、全仓/全盘扫描或构建；明确缺失输入后交付当前可回答的方案，避免无目标地反复探索。需要实际实施时再进入以下模式。
- **错误码开发**：面向新增算子或新增失败路径。枚举所有可观察失败出口，按根因选择合法正式码，接入最小上报入口，并完成编译和负向测试。
- **重构整改**：面向已有算子。扫描废弃码、内部码、仅日志失败返回及重复首报，解析真实宏叶子，逐点分类和修改，再验证整改前后差异。
- **只读审计**：用户只要求分析或评审时，只生成候选、证据、建议和Markdown报告，不修改源码。

方案咨询仍须实际读取可用的基准与适用参考，不能凭记忆给出具体码；但“说明如何验证/能否完成”不等于授权执行整改。没有目标源码时，只能交付有依据的方案、验证要求和缺失输入，不能虚构台账、目标版本注册结论、构建结果或整改报告。实际开发/整改仍须完成下述全部验证后才能关闭。

用户未指定扫描边界时，以当前算子或当前仓库为范围，排除`.git`、构建产物和三方依赖；是否排除`experimental`遵循用户要求和仓库规范，并在报告中声明。

## 必须读取的基准与参考

1. 先阅读cann-error-code-standard skill，并按其中路由读取：
   - development-standard.md (cann-error-code-standard/references/development-standard.md)：选码、首报、消息字段和验证规范。
   - documented-error-code-catalog.md (cann-error-code-standard/references/documented-error-code-catalog.md)：人工选码索引。
   - documented-error-codes.json (cann-error-code-standard/references/documented-error-codes.json)：审计和静态检查的机器可读基准。
2. 整改废弃码或内部码时，读取[整改路由](references/remediation-routing.md)。
3. 扫描、审计E*9***或回答入口覆盖咨询时，读取[触发入口覆盖契约](references/trigger-coverage.md)，说明静态errno映射与目标版本实际实现的证据边界，不能只复述映射结果。
   整改模式还必须读取[逐候选处置闭环](references/candidate-dispositions.md)，冻结整改前候选并维护处置台账。
4. 生成报告时，读取[报告规范](references/status-report-schema.md)。
5. 修改源码后，读取[编译与测试](references/build-and-test.md)和[验证清单](references/validation.md)。

目标版本源码注册表、可见宏和实际安装包是“当前版本能否调用”的事实来源；基准码表命中只证明官方资料收录，不能代替注册和Arglist核对。

## 工作流

### 1. 建立扫描边界与版本证据

记录仓库、commit、算子目录、CANN版本、构建target、排除项和已有未提交修改。只修改本任务涉及的错误路径，不覆盖用户无关改动。

### 2. 解析真实上报入口

1. 从修改文件所属CMake target解析`target_include_directories`、`include_directories`和编译选项；先按实际构建上下文展开`CMAKE_CURRENT_SOURCE_DIR`、`ASCEND_CANN_PACKAGE_PATH`等CMake变量，形成有序候选include目录，再解析头文件。前者对应当前CMakeLists.txt所在目录，后者取目标构建实际配置的CANN路径，不凭安装目录名称猜值。
2. 递归展开本地include和宏别名，确认同名`OP_LOGE`实际来自哪个`log.h`。
3. 将入口追到`REPORT_PREDEFINED_ERR_MSG`、`REPORT_INNER_ERR_MSG`或仅PLOG叶子。
4. 静态解析存在生成头、条件编译或变量缺口时，从真实编译命令提取`-I/-isystem/-D`，必要时用编译器`-E/-H`复核。

回答同名`log.h`来源的核验咨询时，同样说明“先展开CMake变量 → 形成候选include目录 → 递归include与宏别名 → 真实编译/预处理复核”的流程，以及`definition_path/include_chain`证据。无法展开的变量保留为未解析项，不把含`${...}`的字面路径当作已确认目录；静态推断与目标编译结果冲突时，以真实编译结果为准。

可先运行审计器：

```bash
python scripts/audit_operator_error_codes.py \
  --root <operator_dir> \
  --cmake-root <repo_or_build_root> \
  --cann-root <installed_cann_root> \
  --output error_code_return_status.md
```

默认码表来自`cann-error-code-standard/references/documented-error-codes.json`；需要验证另一版基准时可显式传入`--documented-codes <path>`。

### 3. 枚举失败出口

开发模式不能只搜索错误码字符串。遍历参数校验、InferShape、Tiling、workspace/内存、执行器创建、Kernel选择、Runtime/Driver调用及异常边界，记录：

- 根因与失败条件；
- 当前层是否拥有`actual/expected/reason/rc`；
- 下层是否已经产生正式首错；
- 公开返回值与兼容性约束；
- 用户可以执行的下一步动作。

整改模式先运行候选扫描器：

```bash
python scripts/scan_error_code_candidates.py \
  --root <repo_or_operator> --format csv \
  --output error_code_candidates.csv
```

扫描器只产出候选。对多行宏、别名和条件编译继续阅读完整函数与调用链，不以字符串命中直接决定替换。

首次修改算子源码前，运行 `check_candidate_dispositions.py init` 冻结同范围候选、源文件哈希和扫描规则。每个候选必须落入“已整改、经证据确认可保留、删除重复/不可达检查、阻塞待确认”之一；初始化的 PENDING 不算处置。不得只筛选显式 OP_LOGE 而跳过 OP_CHECK_NULL_WITH_CONTEXT 等隐式入口，也不得在整改后重建基线掩盖遗漏。只读审计不要求修改源码或把所有候选关闭。

候选扫描器与审计器共用`scripts/error_code_triggers.py`中的入口定义，覆盖上下文型和errno型`OP_LOGE`、空指针/Vector/Tiling/Cube/Parser封装、`TR_EXECUTE`、`CHECK_RET/CHECK_COND`以及`Report*Message`直接入口。入口范围和静态errno映射见[触发入口覆盖契约](references/trigger-coverage.md)。

混合大小写的`ReportErrMessage`、`ReportInterErrMessage`、`ReportErrorMessage`也必须进入“调用与返回明细”，不能只在回归测试覆盖说明中出现。逐条记录`entry`、`final_code/code_status`、`report_status`、`return_expr/return_marker`、实际宏或函数定义来源和`include_chain`；动态码、errno映射或未解析项须说明依据，不能一律归为内部上报。

解释errno候选或审计结果时，将静态映射的适用条件与结果一并给出：`ACLNN_ERR_RUNTIME_ERROR → EZ9903`、`ACL_ERROR_FAILURE → EZ9999`只用于发现和静态审计，不授权机械改码。即使脚本将结果写入`final_code`，也不代表目标版本已经核验；必须沿实际include链核对`ReportErrorMessage`实现及构建分支，冲突时以目标版本实际实现为准并记录差异。尚无实现证据时标明“静态候选、目标实现待核验”，不得断言运行时必然上报该码。

修改入口识别（包括共享注册表、候选扫描器或审计器）后，必须执行 `python scripts/test_error_code_trigger_coverage.py`，验证上述入口族、混合大小写调用、多行调用、errno映射和候选—审计对账。回答“如何证明入口覆盖”的咨询时，也应给出这项条件与执行命令，而不只是描述测试文件的用途；未执行不得声称回归通过。

入口覆盖咨询需把证据链说完整：以已读取的触发入口覆盖契约为依据，说明共享注册表`error_code_triggers.py`、同边界候选—审计对账和逐入口明细；“未进入审计明细为0”只证明发现链路一致。随后说明实际整改关闭还要求目标include与宏实现、注册/Arglist、最小受影响编译、代表性负向测试和首错唯一性验证，不能只讲对账或静态errno映射就结束。

### 4. 按基准决策并实施

对每个失败出口执行以下顺序：

1. 按根因在基准清单中寻找候选，不按`op_api/op_host/op_kernel`目录机械选码。
2. 核对目标版本注册项、Arglist、模块归属和可见宏。
3. 优先使用目标版本opbase已有正式语义宏；无合适宏但有码时，使用本模块正式ErrorManager入口并严格匹配Arglist。
4. 在拥有根因事实的最内层首报；下层已首报时，外层只补PLOG并原样透传。
5. 资料和注册表均无匹配语义时，只形成结构化正式码类别提案，编号分配前不得编造错误码。

开发模式同步实现错误提示和失败返回。整改模式按[整改路由](references/remediation-routing.md)处理废弃码、E*9***、重复首报和无上报返回，不能做机械编号替换。

逐项更新台账中的根因、处置、证据及整改后位置；行号漂移必须显式映射，不以候选数量减少代替逐项处理。保留宏调用不等于合规：需要实际宏叶子、调用链、保留依据及负向证据。无法确认的项标为 BLOCKED 并报告，不机械改码或静默跳过。

若用户输入错误与内部契约异常无法区分、下层首报状态未知、多个候选会导致不同修复动作，或需要改变公开errno/API返回行为，则停止该点自动修改，保留证据并向用户提出一个具体问题。

### 5. 编译与负向测试

按[编译与测试](references/build-and-test.md)执行：

1. 从修改文件反查最小受影响target，复用仓库已有构建入口。
2. 验证宏可见、Arglist数量和类型、实际头文件来源以及链接结果。
3. 优先扩展已有UT/ST和fixture；按不同根因与上报路由补代表性负向用例。
4. 同时断言公开返回值、正式码、关键诊断字段和“一次故障一个ErrorManager首错”。
5. 缺少环境、设备、mock或数据时，记录命令和首个阻塞证据，标记未验证，不得写成通过。

### 6. 复扫并生成单份报告

以与整改前相同的参数重跑扫描器和审计器。审计器会自动执行候选—审计对账；任一候选未进入审计明细时，先修复入口或include解析，不能宣称全量覆盖。最终只维护一份Markdown报告，包含扫描边界、入口对账、include解析、调用与返回状态、修改决策、基准/注册证据、编译、测试、遗留风险和“是否可关闭”。

完成整改前还必须运行 `python scripts/check_candidate_dispositions.py check --ledger <处置台账.json>`，将退出码、未处置/阻塞项和复扫残留写入上述同一份报告。入口对账为零仅证明“已发现”，处置对账通过才证明“逐项有结论”；任一检查未通过/未运行均不可宣称整改完成。脚本只校验处置结构和源码一致性，不能替代证据真实性、include/注册、编译与负向测试评审。

回答“候选已发现，能否说整改完成”或“完成检查是什么”时，先读取[逐候选处置闭环](references/candidate-dispositions.md)，按其中“完成结论的最小证据”逐项解释，不能仅列处置状态和退出码0而省略记录字段、位置映射、报告退出码及语义证据边界。咨询只解释要求，不虚构执行结果。

报告标记和关闭条件见[报告规范](references/status-report-schema.md)。存在未对账候选、未解析include、未核对注册/Arglist、适用构建或测试未通过/未执行时，均不可关闭。修改Skill自身扫描逻辑后还必须运行`python scripts/test_error_code_trigger_coverage.py`。
