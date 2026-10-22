# 逐候选处置闭环

候选发现、进入审计和完成整改是三件事。仅替换显式日志，留下同函数内的
`OP_CHECK_NULL_WITH_CONTEXT` 等隐式入口且没有处置依据，不能称为完整整改。
本契约适用于用户明确要求整改的范围，不扩大只读审计或用户限定的任务边界。

## 1. 整改前冻结

在 Skill 目录运行，产物放在任务工作目录，不放入被扫描的源码目录：

```bash
python scripts/check_candidate_dispositions.py init \
  --root <repo_or_operator> --output <task_dir>/candidate-dispositions.json
```

`--exclude-dir`、`--suffix`、`--include-definitions` 与候选扫描器同义；按真实源码补充
后缀（如 `.asc`）。初始化会运行候选扫描器，冻结根目录、后缀、排除项、入口定义版本、
整改前源文件哈希和完整候选；每项以整改前记录的摘要作为稳定 ID。
输出文件已存在时拒绝覆盖。保留 baseline 和 baseline_sha256，只编辑 decisions。
基线必须在首次源码修改前生成；补审已有 patch 时从真实基准版本扫描，不能把整改后
扫描当作整改前基线。无法取得基准时说明覆盖证据缺失，不宣称全量完成。

## 2. 处置状态与证据

每条 decision 都填写 `root_cause`、`reason`，证据值应是可复核的报告章节、
源码位置、注册项或测试日志路径，不能只是“已确认”“通过”。

| status | 含义 | evidence 必填字段 |
|---|---|---|
| FIXED | 根因层改为正式语义上报或包装层改为仅日志透传 | macro_definition、call_chain、registration、build、negative_test、first_error |
| RETAINED | 经目标实现和调用链证明可以保留当前入口 | macro_definition、call_chain、retention_basis、negative_test |
| REMOVED | 删除重复或不可达的检查/上报路径 | macro_definition、call_chain、removal_basis |
| BLOCKED | 语义、版本、首报或验证条件仍不明确 | root_cause、reason 记录阻塞和下一步；不可关闭 |
| PENDING | 初始状态、尚未处置 | 不可关闭 |

`FIXED`、`RETAINED` 的 `after` 必须填写整改后 `path`（根目录相对路径）、
`line`（入口起始行）和 `entry`（实际宏/函数入口）；`REMOVED` 的 after 为 null。
源码移动、行号变化应显式映射，不能用整改后行号重新生成 ID。整改把一个候选拆成多条
路径时，主入口写 after，其余路径及审计证据写在 reason/evidence，复扫中的每个残留
候选仍必须有唯一的保留处置；复杂拆合需人工复核，不用自动相似匹配猜测。

RETAINED 不是“暂不处理”的别名。内部异常须满足整改路由的保留准入条件；误报须有
实际只写日志等源码证据。废弃码不得通过该状态关闭。无证据、缺环境或需要新增正式码
注册时应为 BLOCKED，不能批量标成 RETAINED。下层透传路径的 registration 证据引用
下层实际首码注册，不能写成当前层另加首码。

## 3. 整改后强制复核

```bash
python scripts/check_candidate_dispositions.py check \
  --ledger <task_dir>/candidate-dispositions.json
```

脚本从冻结的范围复扫当前源文件，不接受临时缩小范围；校验基线未被误改、扫描器版本
一致、每个基线候选恰有一条处置、证据字段齐全及整改后位置真实存在。
FIXED/REMOVED 对应原文件完全未变化、FIXED 位置仍是治理候选、残留/新增候选没有
唯一且有依据的 RETAINED 映射、PENDING/BLOCKED 均不通过。源文件读取失败、基线损坏、
扫描器变化或扫描期间源码变化返回错误，不允许零差异放行。

退出码：0 处置结构与复扫对账通过；1 未完成；2 校验不完整。
它不会自动证明注册语义、保留理由和日志真实有效，也不是 C/C++ 编译器或安全防篡改工具。
检查通过只表示“处置对账通过”，整体关闭仍要求入口审计、注册/Arglist、include、
适用构建和负向测试全部满足。不得把此脚本通过替代现有审计器的静态判定。

最终仍只维护一份 Markdown 报告：附台账路径、基线摘要、各状态数、未处置/阻塞项、
复扫残留、命令退出码和“是否可关闭”。JSON 是机器校验台账，不是第二份结论报告。

## 4. 完成结论的最小证据

实际整改报告及解释完成规则的咨询均须覆盖下列证据；咨询说明“需要什么”，不伪称已经取得证据。

| 检查层次 | 必须说明或记录的内容 |
|---|---|
| 逐候选记录 | 每项均有根因`root_cause`、理由`reason`、可复核`evidence`及整改后位置`after`；行号漂移须从原候选显式映射到整改后`path/line/entry`，不能重建ID掩盖遗漏；删除项`after=null`并提供删除证据 |
| 复扫与处置 | 同冻结范围运行check；未改却标FIXED、仍是候选却标FIXED、处置缺失/重复、无依据保留或未对账残留均失败。残留候选须有唯一且满足准入依据的RETAINED映射，不能把“候选必须清零”代替逐项处置 |
| 单份报告 | 执行者将check命令及实际退出码、未处置/阻塞项、复扫残留和是否可关闭写入同一份Markdown；仅说“退出码0才通过”不能代替报告记录要求 |
| 证据真实性 | 校验器只验证处置结构与源码一致性，不能证明语义证据真实性；整体关闭仍须核对真实include/宏叶子、目标版本注册/Arglist、适用构建、负向测试及首错唯一性 |

任一PENDING/BLOCKED、复核失败或适用验证未执行，均不能宣称完成。保留入口不等于合规，须按真实根因和准入证据判断，不能把所有内部码一概允许保留或一概当作废弃码处理。
