# 既有算子错误码重构路由

先按cann-error-code-standard确认通用选码与首报规范。本文件只描述已有代码的候选识别、废弃码迁移、内部码拆解和特定调用链决策。

## 1. 整改不是编号替换

每个候选都先确认：实际宏定义、最小上报叶子、完整函数上下文、根因层、下层首报和公开返回值。`REPORT_INNER_ERR_MSG`允许未注册内部码，不能用它承接正式码治理。

## 2. 六个废弃错误码的替换路由

| 待废弃码 | 常见旧宏/语义 | 首选opbase路由 | 选择条件 |
|---|---|---|---|
| EZ0001 | `OP_LOGE_WITH_INVALID_INPUT_SHAPE` | EZ0008 `OP_LOGE_FOR_INVALID_SHAPE` | 单参数且明确正确shape |
| EZ0001 | 同上 | EZ0009 `OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON` | 单参数，约束用reason更清楚 |
| EZ0001 | 同上 | EZ0010 `OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON` | 多Tensor shape关系错误 |
| EZ0002 | `OP_LOGE_WITH_INVALID_ATTR` | EZ0024 `OP_LOGE_FOR_INVALID_VALUE` | 单属性且明确正确值 |
| EZ0002 | 同上 | EZ0026 `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON` | 单属性，用范围或条件描述 |
| EZ0002 | 同上 | EZ0027 `OP_LOGE_FOR_INVALID_VALUES_WITH_REASON` | 多属性或属性与输入组合约束 |
| EZ0003 | `OP_LOGE_WITH_INVALID_ATTR_SIZE` | EZ0025 `OP_LOGE_FOR_INVALID_LISTSIZE` | 单列表属性且明确正确size |
| EZ0003 | 同上 | EZ0038 `OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON` | size由条件决定或用reason描述 |
| EZ0005 | `OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE` | EZ0014 `OP_LOGE_FOR_INVALID_SHAPESIZE` | 单参数且明确正确元素数 |
| EZ0005 | 同上 | EZ0015 `OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON` | 单参数，约束用reason描述 |
| EZ0005 | 同上 | EZ0016 `OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON` | 多参数shape size关系错误 |
| EZ0006 | `OP_LOGE_WITH_INVALID_INPUT_FORMAT` | EZ0017 `OP_LOGE_FOR_INVALID_FORMAT` | 单参数且明确正确format |
| EZ0006 | 同上 | EZ0035 `OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON` | 单参数且支持集合/条件复杂 |
| EZ0006 | 同上 | EZ0018 `OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON` | 多参数format关系错误 |
| EZ0007 | `OP_LOGE_WITH_INVALID_INPUT_DTYPE` | EZ0019 `OP_LOGE_FOR_INVALID_DTYPE` | 单参数且明确正确dtype |
| EZ0007 | 同上 | EZ0020 `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON` | 单参数且支持集合/条件复杂 |
| EZ0007 | 同上 | EZ0021 `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON` | 多参数dtype关系错误 |

替换时用算子原型中的稳定参数名代替input index。无法从原型或注册定义确认名称时，不自动替换。

```cpp
// 前：EZ0001，只告诉用户第0个输入。
OP_LOGE_WITH_INVALID_INPUT_SHAPE(opName, 0, actualShape, expectedShape);

// 后：使用参数名和匹配的正式语义。
OP_LOGE_FOR_INVALID_SHAPE(opName, "query", actualShape, expectedShape);
```

```cpp
// 前：EZ0002。
OP_LOGE_WITH_INVALID_ATTR(opName, "sparse_mode", actual, expected);

// 后：范围或组合条件由reason表达。
OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "sparse_mode", actual,
    "supported values are 0, 1, 2 and 3 for the current layout");
```

## 3. E*9***最小入口识别

本节给出整改路由；完整入口族、errno映射和候选—审计一致性要求见[触发入口覆盖契约](trigger-coverage.md)。

| 候选入口 | 常见最终码 | 核对重点 |
|---|---:|---|
| `op_common::OP_LOGE(opName/context, ...)` | EZ9999 | 宏内是否调用`REPORT_INNER_ERR_MSG` |
| `nnopbase::OP_LOGE(errno, ...)` | EZ1001/EZ9903/EZ9999 | errno首位映射；必须读取完整errno和目标实现 |
| `OP_CHECK_NULL_WITH_CONTEXT`及别名 | EZ9999 | 是否展开到op_common的`OP_LOGE` |
| Vector/Tiling包装宏 | E89999 | 同名宏有的只写PLOG，必须查实际定义 |
| Cube包装宏 | E69999/E59999 | 区分算子、插件和编译分支 |
| GE/FE/Parser包装 | E19999/E29999/E79999 | 是否已有正式图或解析错误码 |
| `REPORT_INNER_ERR_MSG("E?9???", ...)` | 调用方传入 | 直接内部码，无正式注册约束 |
| `OP_LOGE_WITHOUT_REPORT`、`OPS_LOG_E` | 无 | 只写PLOG，不应当作错误码调用 |

识别到内部码后，按基准Skill的根因路由选择“正式语义、下层首错透传、删除不可达检查或新增正式码类别”，不能全部改成EZ1009。

## 4. 典型重构

### 4.1 opName/context型参数错误

```cpp
// 前：op_common::OP_LOGE隐式EZ9999。
OP_LOGE(context, "input rank must be 4");

// 后：根因是单参数rank不满足约束。
OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
    std::to_string(actualRank).c_str(), "rank must be 4");
```

### 4.2 errno型包装覆盖下层首错

```cpp
// 前：完整errno经包装映射成泛化内部码。
OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TransData returned nullptr");

// 后：TransData根因层负责正式首报；本层只补上下文并兼容返回值。
OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR,
    "TransData returned nullptr; propagate lower-layer first error");
return ACLNN_ERR_INNER_NULLPTR;
```

### 4.3 明确能力不支持

只有调用链已经确认当前SoC、kernel或组合没有实现时，才使用目标版本已注册的EZ1009入口；reason包含actual/supported SoC、kernel、dtype、layout和选择key。当前分支无对应宏时，通过本模块正式ErrorManager入口上报准确Arglist，不回退到内部码。

## 5. 保留内部码的准入条件

同时满足以下条件才可保留：

1. 已证明输入、环境和资源契约合法。
2. 不是已知能力不支持、配置缺失或下层已有首错。
3. 异常代表真实内部不变量破坏、设备执行异常或当前层无法解释的实现缺陷。
4. 日志包含可关联的op/kernel、阶段、原始rc、关键shape/dtype/tiling key；必要时触发故障现场采集。
5. 台账记录保留理由、责任边界和后续正式码治理计划。

不满足条件且找不到现有正式码时，形成结构化正式码提案，不为清零而误分类。

## 6. 已验证调用链决策

以下结论用于识别同类场景，应用前仍需复核目标版本：

| 场景 | 调用链证据 | 路由 |
|---|---|---|
| 当前SoC/kernel无实现 | host检查发生在executor/Runtime之前，或opbase确认无binary | EZ1009；reason包含actual/supported SoC、kernel、dtype/format和选择key |
| 合法输入下Tiling失败 | Tiling函数不存在、执行失败、内部workspace计算或序列化失败 | EZ1008；包含原始ret、required/actual、tilingKey |
| Caffe Parser工厂为空 | Parser按optype创建失败 | E11009，Arglist=`opname,optype` |
| ONNX Parser未注册 | 无法由原始optype取得OM类型 | E16002，Arglist=`optype` |
| IR工厂未注册 | 创建Operator后名称/类型不成立 | E10501，Arglist=`opname,optype` |
| `CREATE_EXECUTOR`后检查null | `UniqueExecutor`使用普通`new`，失败抛`bad_alloc` | 删除死检查；公共C API异常边界和实际分配层承接资源首报 |
| launcher聚合失败 | 包装只返回`GetWorkspace`等下层聚合状态 | 追到Tiling、kernel选择、资源或Runtime根因；外层仅日志透传 |
| GE模块无法引入opbase头 | 语义码已注册但头依赖不允许 | 用GE现有ErrorManager入口正式上报，Arglist逐字匹配 |
| `aclrtMemcpyAsync`等返回rt失败 | 调用宏可能只返回rc，是否首报取决于目标Runtime实现 | Runtime/ACL层完成rc到正式码映射；算子层不生成泛化摘要 |
| AIPP JSON字符串解析失败 | 字符串来自OpDesc属性而非已定位磁盘文件 | 用户来源用E20101；框架生成用E20007并补详细PLOG；不伪造file字段 |

## 7. 新增正式码类别的门槛

只有同时满足以下条件才提出新增类别：

1. 已读实际失败分支和上游校验，证明不是用户参数错误、下层已有首错或不可达代码。
2. 已检索基准清单、目标版本公开注册表和可见公共宏，确认不存在语义与字段均匹配的正式码。
3. 已确认跨模块复用不会违反模块归属；若只是头依赖问题，优先通过本模块入口复用现有码。
4. 提案写明模块、标题、消息模板、Arglist、原因、解决方案、首报层、责任人和UT，并回答What、Where、Why、What Next。
