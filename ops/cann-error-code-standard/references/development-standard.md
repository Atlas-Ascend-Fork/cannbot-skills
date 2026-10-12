# CANN算子错误码开发使用规范

## 1. 目标

错误输出应使开发者快速得到`What → Where → Why → What Next`，同时保证同一故障只有一个ErrorManager首错。错误码表达用户可见根因，不表达源码目录或发生阶段。

## 2. 不可破坏的规则

1. 在最接近根因、掌握完整事实的层完成一次正式首报；外层只补PLOG上下文并透传。
2. 优先使用目标版本opbase已经提供的正式语义宏；宏不存在时才考虑本模块正式ErrorManager入口。
3. 任何直接上报都必须与目标版本注册项的Arglist名称、数量和顺序一致。
4. `E*9***`是内部码模式，不是新开发路径的可选语义码；不得用泛化内部码兜底。
5. 禁止新增`EZ0001/EZ0002/EZ0003/EZ0005/EZ0006/EZ0007`。
6. 错误码上报与公开返回值分开治理。没有兼容性评审，不改变公开errno或API返回行为。
7. 资料和注册表没有匹配项时，只提出正式码类别和所需字段；编号正式分配前不得编造。

## 3. 选码决策流程

1. **识别根因**：区分用户参数、能力不支持、InferShape/Tiling、资源、Runtime/Driver、下层透传和真实内部不变量。
2. **确定首报层**：当前层是否拥有根因事实；下层是否已上报正式首错。
3. **查文档清单**：在`documented-error-code-catalog.md`或JSON中查找语义匹配项并保留URL。
4. **无匹配时查源码索引**：按`component-error-code-source-index.md`定位归属组件的公开注册文件，不跨模块借码。
5. **查目标版本实现**：确认注册项、Arglist、模块归属和可见宏；解析调用点实际include链。
6. **选择入口**：优先语义宏，其次本模块正式上报入口；下层已首报则只写PLOG。
7. **设计消息**：输出足以执行修复的字段，不只写“failed”或“invalid”。
8. **验证**：最小受影响编译、代表性负向测试和首错唯一性检查。

## 4. 根因到错误码语义的推荐路由

| 根因 | 候选正式语义 | 必须携带的信息 |
|---|---|---|
| 单Tensor shape/rank/size | EZ0008/9、EZ0011/12、EZ0014/15 | 参数名、actual、expected或reason |
| 多Tensor shape/rank/size关系 | EZ0010/EZ0013/EZ0016 | 参数集合、实际值、关系约束 |
| format/layout | EZ0017/EZ0035；多参数EZ0018 | actual、支持集合、layout条件 |
| dtype | EZ0019/EZ0020；多参数EZ0021 | actual、支持集合 |
| attr/value/list size | EZ0024/25/26/27/38 | 参数名、actual、合法范围或关系 |
| stride | EZ0028 | actual、expected |
| 文件/配置 | EZ0029～EZ0034 | 文件路径、配置项、actual、原因 |
| 图节点/一般参数关系 | EZ0036/EZ0037 | node/参数名、违反的约束 |
| 必选输入为空 | EZ0004；ACLNN参数空且目标宏适用时EZ1011 | API/算子名、参数名 |
| 合法输入下InferShape失败 | EZ1014 | opName、具体reason、下层rc |
| 合法输入下Tiling失败 | EZ1008 | opName、Tiling函数/key、reason/rc |
| 当前SoC/kernel/组合不支持 | EZ1009 | kernel、actual/supported SoC、dtype/layout、选择key和reason |
| 下层已报告具体首错 | 不新增正式码；仅日志并透传 | 当前stage、下层API、原始rc、关联ID |
| 内存/资源不足 | 由实际allocator、ACL或Runtime资源码首报 | 资源类型、required/available bytes、API和rc |

表中编号是候选语义，不是无条件可用清单。使用前必须在目标版本注册表和实际头文件中确认。发生在Tiling代码中的shape错误仍属于参数错误；只有合法输入下Tiling执行失败才考虑EZ1008。返回对象为null也不能直接等价为内存不足。

## 5. 各开发阶段的责任边界

| 阶段 | 应首报的典型问题 | 通常只补上下文的问题 |
|---|---|---|
| op_host / InferShape / Tiling | 已掌握actual与约束的参数错误；合法输入下本层确定的InferShape/Tiling失败；明确的kernel/SoC不支持 | 下层解析器、Runtime或allocator已产生正式首错 |
| op_api / aclnn包装 | 公共API边界独有的空指针、参数契约或异常转换 | executor、Runtime、Driver和算子host已首报的失败 |
| op_kernel / 启动边界 | host侧已知的launch配置或能力不支持 | 设备侧异常、MTE越界、Runtime/Driver执行错误应由对应运行时诊断链路首报，算子包装层保留kernel/task/rc关联信息 |

不要为“每个阶段都要有错误码”而重复上报。同一根因跨多层传播时，只保留一个正式首错。

## 6. 消息字段规范

消息至少回答：

- **What**：哪个API、算子、参数或资源失败；
- **Where**：阶段、callee、kernel或关键选择路径；
- **Why**：actual与expected/reason，或下层原始rc；
- **What Next**：用户能采取的输入修改、能力核对、资源调整或证据采集动作。

按场景增加：

- 参数错误：稳定的语义参数名，不使用易漂移的input index；
- 容量错误：`requiredBytes`和`actual/availableBytes`；
- 下层调用：API名称和未经丢失的原始rc；
- kernel选择：SoC、dtype、layout、tiling key和支持集合；
- 关联诊断：op/kernel、stream/task或correlationId。

## 7. 合法入口优先级

1. 目标版本opbase正式语义宏。
2. 目标模块已有且最终调用正式注册接口的语义宏。
3. 本模块直接调用正式注册接口，并逐字匹配Arglist。
4. 结构化新增正式码提案。
5. 只有经过证明的真实内部不变量或无法由当前层解释的设备异常，才允许保留内部诊断码，并必须关联可追踪上下文和治理计划。

`OP_LOGE_WITHOUT_REPORT`、`OPS_LOG_E`、`D_OP_LOGE`等通常只写日志，不能当作正式错误码上报；同名宏在不同头文件中可能不同，必须解析实际定义。

## 8. 完成判据

一条错误路径只有同时满足以下条件才可判为合规：

- 文档语义与根因匹配；
- 目标版本已注册，Arglist与调用一致；
- 实际include链和宏叶子已确认；
- 根因层正式首报且外层不重复；
- 公开返回值兼容；
- 最小构建通过；
- 代表性负向测试验证错误码、关键字段和首错唯一性。

静态检查只能给出候选或缺口，不能替代编译和运行证据。
