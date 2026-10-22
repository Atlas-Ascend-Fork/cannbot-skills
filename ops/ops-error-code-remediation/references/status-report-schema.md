# 算子错误码返回状态报告规范

## 1. 报告用途

`audit_operator_error_codes.py`用于已经完成开发的算子。它同时检查错误上报入口、最终错误码、失败返回值、CMake include目录和宏定义来源，并把候选扫描结果与审计明细自动对账，生成Markdown报告的静态审计部分。源码发生修改时，执行者还必须在同一报告中补充编译验证、测试用例和最终关闭结论。

报告中的“返回标记”写入审计结果，不自动把标记注释插入源码。用户明确要求整改时，再依据标记修改真实根因层代码。

## 2. 返回标记定义

| 标记 | 含义 | 完成条件 |
|---|---|---|
| `FORMAL_REPORTED_RETURN` | 已解析到资料收录的正式错误码并观察到失败返回 | 注册项、Arglist、宏可见性、诊断字段和首报唯一性均通过验证 |
| `FORMAL_REPORTED_NO_RETURN_OBSERVED` | 已正式上报，但静态窗口未观察到返回 | 阅读完整函数或异常路径，确认控制流终止或由调用者返回 |
| `DEPRECATED_CODE_RETURN` / `DEPRECATED_CODE_NO_RETURN` | 使用治理范围内的EZ0001/2/3/5/6/7 | 必须整改，不得作为完成状态 |
| `INTERNAL_E9_RETURN` / `INTERNAL_E9_NO_RETURN` | 直接或经宏隐式生成E*9*** | 按根因改成正式码、下层首错透传或形成正式码提案 |
| `LOWER_LAYER_RETURN` | 当前层只写PLOG并返回下层`rc/ret/status/errno` | 负向日志证明下层已经产生正式首错，并且关联字段完整 |
| `LOG_ONLY_RETURN` / `LOG_ONLY_NO_RETURN_OBSERVED` | 仅写日志，不能证明存在ErrorManager首错 | 补查下层调用链；无首错时在根因层补正式上报 |
| `RETURN_WITHOUT_REPORT` | 失败返回附近没有错误码上报入口 | 确定首报责任；如果不是外层透传则补正式上报 |
| `UNDOCUMENTED_CODE_RETURN` / `UNDOCUMENTED_CODE_NO_RETURN` | 解析到具体码但参考资料未收录 | 核对目标版本源码注册表；未注册或无文档时不得直接使用 |
| `UNRESOLVED_RETURN` / `UNRESOLVED_NO_RETURN` | 同名宏、条件编译或include链未解析出叶子 | 补充`--cmake-root`、`--cann-root`或`--include-dir`后重跑 |

## 3. 错误码状态

| 状态 | 定义 |
|---|---|
| `DOCUMENTED_FORMAL` | `final_code`存在于`cann-error-code-standard/references/documented-error-codes.json` |
| `GOVERNANCE_DEPRECATED` | 属于EZ0001、EZ0002、EZ0003、EZ0005、EZ0006、EZ0007 |
| `INTERNAL_E9` | 符合`E*9***`模式 |
| `NOT_IN_DOCUMENT_REFERENCE` | 解析到具体码，但`cann-error-code-standard`文档基准未收录 |
| `NO_CODE` | 当前入口没有解析出具体错误码 |

“资料已收录”只说明可进入进一步验证，不等价于目标安装包已经注册，也不代表当前模块可以直接使用。

## 4. 静态与最终判定

审计脚本只输出“静态审计是否通过：是/否”，不提前判断整个整改是否可关闭。

- 任一治理候选未进入审计明细、任一调用未达到`FORMAL_REPORTED_RETURN`，或存在未解析CMake/include链时，静态审计不通过。
- `LOWER_LAYER_RETURN`经负向日志证明下层已产生正式首错后，可由评审者改判为已完成路径。
- 最终Markdown由执行者补充“是否可关闭：是/否”。只有静态审计、目标版本注册/Arglist、最小受影响构建、适用负向UT/ST和“一次故障只产生一个ErrorManager首错”全部通过，才可填写“是”。
- 整改模式同时要求[逐候选处置闭环](candidate-dispositions.md)校验通过。每条整改前候选记录处置、根因、证据和整改后位置；PENDING/BLOCKED 或未经证据确认的保留项使最终结论为“否”。不能把“未进入审计明细为0”改写成“未整改为0”。

## 5. 报告字段

| 字段 | 内容 |
|---|---|
| `path/line/function/stage` | 调用点及op_api、op_host、op_kernel或shared阶段 |
| 候选—审计对账 | 候选总数、未进入审计明细数量及未对账列表 |
| 候选—处置对账 | 台账路径、基线摘要、各处置状态数、未关闭列表、复扫残留和检查退出码 |
| `entry` | 源码入口宏或直接`return` |
| `final_code/code_status` | 宏展开后的错误码及资料/治理状态 |
| `report_status` | `FORMAL_REPORTED`、`INTERNAL_REPORTED`、`LOG_ONLY`、`RETURN_ONLY`或`UNRESOLVED` |
| `return_expr/return_marker` | 观察到的返回表达式和最终分类标记 |
| `macro_leaf` | 解析得到的最小上报叶子 |
| `definition_path/definition_line` | 实际宏定义来源 |
| `include_chain` | 从调用文件到宏定义文件的已解析include链 |
| `recommended_action` | 当前状态的下一步处理建议 |
| 编译验证 | target、构建命令、环境、通过/失败/未验证及首个有效错误 |
| 测试用例 | 修改文件、场景、预期码与字段、执行命令、结果及未覆盖项 |
| 最终结论 | 是否可关闭及全部未关闭原因 |

errno映射所得的`final_code/code_status/report_status`仍是静态工具结论，不因字段名含“final”而视为已核验的运行时事实。执行者须在报告中区分“静态候选”与“目标实现已确认”，给出实际`ReportErrorMessage`定义和生效构建分支证据；冲突时记录差异并以目标版本实际实现为准。静态映射只用于发现和审计，不授权机械改码；缺少实现证据时保留待核验状态，不宣称该码必然发生。

## 6. 使用限制

- C/C++预处理器包含条件编译、生成头文件和target级属性。脚本提供可复核的静态近似，不能替代编译器的`-E/-H`输出。
- 同一头文件在不同target或编译分支中可能解析到不同实现。报告必须记录构建target、编译选项和CANN版本。
- `FORMAL_REPORTED_NO_RETURN_OBSERVED`不等价于缺少返回；可能通过宏、异常或上层控制流终止，需要人工复核。
- `RETURN_WITHOUT_REPORT`可能是合法的包装层透传，但必须给出下层首错证据才能关闭。
- 对账采用`path + line`静态关联；同一行存在多个嵌套入口时仍需人工核对调用顺序和首报责任。
