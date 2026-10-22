# E*9***触发入口覆盖契约

## 1. “覆盖全量”的判定口径

本Skill的全量覆盖是指：在声明的源码扫描边界内，能够识别已知直接错误码、封装宏、errno映射、宏别名和失败返回入口，并使每条治理候选进入最终审计明细。它不代表脱离目标构建分支即可证明所有条件编译和生成头路径；这类路径仍须用真实编译命令或预处理结果确认。

候选扫描器和最终审计器必须共同引用`scripts/error_code_triggers.py`。不得在两个脚本内分别维护入口清单，否则入口新增时容易只出现在候选报告或只出现在最终审计中。

## 2. 当前必须识别的入口族

| 入口族 | 代表入口 | 审计重点 |
|---|---|---|
| 直接内部上报 | `REPORT_INNER_ERR_MSG`、`REPORT_INNER_ERROR`、`REPORT_INTER_ERR_MESSAGE`、`ReportInterErrMessage` | 直接码、动态码参数、返回路径 |
| op_common上下文型 | `OP_LOGE(opName/context, ...)` | 沿实际`log.h`展开到内部上报还是仅PLOG |
| nnopbase errno型 | `OP_LOGE(errno, ...)`、`ReportErrorMessage(errno, ...)` | errno到EZ1001/EZ9903/EZ9999的实际映射 |
| 空指针封装 | `OP_CHECK_NULL_WITH_CONTEXT`、`OPS_CHECK_NULL_WITH_CONTEXT[_RET]`、`OPS_CHECK_NOT_NULL_WITH_CONTEXT`、`MOE_OPS_CHECK_NULL_WITH_CONTEXT_RET` | 宏叶子、错误码、是否直接返回 |
| Vector/Tiling封装 | `VECTOR_INNER_ERR_REPORT_TILIING`、`OPS_REPORT_VECTOR_INNER_ERR`、`MOE_VECTOR_INNER_ERR_REPORT_TILIING` | 参数根因、Tiling执行失败和内部契约异常 |
| Tiling注册/执行包装 | `REGISTER_OP_TILING_V3/V4_*`、`TR_EXECUTE`及经include链解析出的别名 | 生成函数中的空对象检查、下层是否已首报、包装层是否覆盖首错 |
| Parser/GE/InferAxis | `ONNX_PLUGIN_LOGE`、`GE_OP_LOGE_WITH_REPORT`、`INFER_AXIS_TYPE_ERR_REPORT` | 解析、推导和图构建失败语义 |
| Cube封装 | `OPS_REPORT_CUBE_INNER_ERR`、`CUBE_INNER_ERR_REPORT[_PLUGIN]` | 能力不支持、参数约束或实现异常 |
| 返回检查 | `CHECK_RET`、`CHECK_COND` | errno返回是否在本层或下层已有正式首报 |
| 正式注册入口 | `REPORT_PREDEFINED_ERR_MSG`、`REPORT_CALL_ERROR`、`ReportErrMessage` | 具体码是否资料收录、目标版本已注册且Arglist匹配 |
| 仅日志入口 | `OP_LOGE_WITHOUT_REPORT`、`OPS_LOG_E`等 | 不误判为ErrorManager上报；确认下层首错 |

入口名只是发现线索。最终分类必须读取调用语句、函数上下文、宏定义、include链、失败返回和下层调用链，不能根据名字直接批量替换。

### 混合大小写直接入口的审计证据

`ReportErrMessage`、`ReportInterErrMessage`、`ReportErrorMessage`不是仅供测试覆盖的名字。每个调用都须进入报告的“调用与返回明细”，保留实际大小写和入口类型：

| 入口 | 必须核对的事实 |
|---|---|
| `ReportErrMessage` | 正式注册上报入口的实际实现、传入码及目标版本注册/Arglist，不因名字含Report就归为内部码 |
| `ReportInterErrMessage` | 内部上报实现及传入码，动态码必须保留待解析说明 |
| `ReportErrorMessage` | 完整errno实参、实际映射实现及最终码；静态映射不能代替目标实现 |

三类入口均记录`path/line`、`entry`、`final_code/code_status`、`report_status`、
`return_expr/return_marker`、`definition_path/definition_line`和`include_chain`；经宏包装时还记录`macro_leaf`。没有宏叶子的函数调用如实说明，不编造宏定义。返回未观察到或include未解析时保留相应状态，不因候选对账为零而省略证据。

## 3. errno映射核对

对nnopbase实现，审计器按已知符号和十进制errno首位给出静态候选映射：

| errno类别 | 静态映射 | 含义 |
|---|---|---|
| `ACLNN_ERR_PARAM_*`、`ACL_ERROR_INVALID_PARAM`或1开头 | EZ1001 | 参数错误正式码；仍需核对消息字段和目标实现 |
| `ACLNN_ERR_RUNTIME_ERROR`或3开头 | EZ9903 | Runtime内部摘要码；优先保留下层具体首错并仅日志透传 |
| `ACLNN_ERR_INNER*`、`ACL_ERROR_FAILURE`或5开头 | EZ9999 | 泛化内部码；必须按实际根因重新分类 |

静态映射只用于发现和审计，不授权机械改码。宏实现与上述规则不一致时，以目标版本实际`ReportErrorMessage`实现为准，并在报告中记录差异。

审计器的`final_code`是静态解析结果，字段名不代表运行时已经验证。回答入口覆盖或对账问题时，应在列出errno映射的同时说明上述限定，而不是仅在末尾泛称“仍需验证include”。核对实际include链、`ReportErrorMessage`函数体和生效编译分支后，才能将候选映射作为目标实现结论；源码不可得时明确“静态候选、目标实现待核验”。合成回归验证的是工具既有映射规则，不是所有CANN版本的运行时行为。

## 4. 候选—审计对账

运行`audit_operator_error_codes.py`时，脚本会在相同根目录、后缀和排除规则下再次执行候选扫描，并按`path + line`与审计明细对账。

- `未进入审计明细的候选 = 0`是静态关闭的必要条件。
- 非零时必须先补入口识别、include路径或动态码解析，不得把该候选从报告删除来规避。
- 对账为零只证明发现链路一致；仍需逐条完成注册/Arglist、编译、负向测试和首错唯一性验证。

## 5. 回归验证

修改入口识别（共享入口注册表、候选扫描器或审计器任一项）后，必须实际执行以下回归，并在最终报告记录命令、退出码和结果；仅描述“测试覆盖这些入口”不等于执行通过。回答入口覆盖的流程咨询时，也要说明这一触发条件和命令：

```bash
python scripts/test_error_code_trigger_coverage.py
```

该测试用合成源码覆盖上述入口族、混合大小写`Report*Message`、多行`OP_LOGE`和errno映射。新增入口族时，必须同时扩充共享注册表和本测试。
