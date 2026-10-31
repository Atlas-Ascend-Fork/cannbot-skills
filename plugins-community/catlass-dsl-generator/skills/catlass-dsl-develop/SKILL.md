---
name: catlass-dsl-develop
description: Generate a complete CATLASS Python DSL operator from an approved DESIGN.md, then debug, review, test, benchmark, profile, and optionally optimize it with knowledge queries and resumable state. Use for CATLASS DSL operator implementation, debugging, and delivery from an approved design.
---

# CATLASS DSL Develop

读取获批设计，一次生成全部代码，再进行验证和增量调试。Agent 负责实现与决策，
`develop_state.py` 固化配置、验证门禁、推进 `next_action` 并保存紧凑证据。

## First action

先读用户提示、获批 `DESIGN.md` 和 `state.json`；可发现的信息不要询问。展示：

**📋 Resolved Plan**

- **Workspace / Design** — 仓库根目录、设计路径和 SHA-256。
- **Implementation** — 一个主 Python 文件 `kernel_path`，可包含多个 DSL kernel。
- **Fusion** — `performance`（默认）或用户明确指定的 `single_kernel`。
- **Benchmark inputs** — `solution_path`、`definition.json`、`workload.jsonl`。
- **Commands / required cases** — 获批 argv、case IDs、设备与环境。
- **Risk / Performance / Knowledge** — 评审重点、性能目标或只测量、查询条件。

配置使用 [state template](templates/state.json)，状态位于：

```text
.catlass-dsl/develop-runs/<run-id>/state.json
```

`fusion.mode=performance` 默认关闭全部 anti-hack，允许多个 kernel、executor、
GM 中间张量及显式 shape/dtype/layout 分派。只在用户明确要求单个融合 kernel 时使用
`single_kernel`，要求只声明一个 `@tla.kernel`、每次 `run` 只 launch 一个 kernel，
不得为编译期变体声明独立 kernel。benchmark argv 分别使用 `--anti-hack off` 或
`--anti-hack single-fused`，optimize 的初始化和全部复测必须继承它。

每个 kernel 保持自包含：固定配置在函数内定义，运行时配置走形式参数或 tensor metadata；
host 不通过 `global`、模块级可变状态或 closure 注入语义。默认模式的普通代码评审仍检查
实现是否符合获批设计，但不执行源码 anti-hack 扫描、launch/name 审计或 manifest 校验。

```bash
python3 skills/catlass-dsl-knowledge/scripts/record_knowledge.py initialize --project-root <workspace>
python3 skills/catlass-dsl-develop/scripts/develop_state.py validate --state <run>/state.json
python3 skills/catlass-dsl-develop/scripts/develop_state.py start --state <run>/state.json
```

## 骨架先行，增量扩展

默认从 `implement` 开始。**先写出可编译、可运行的最小端到端路径**：单 tile / 单分块、
单一路径、host 可调用；**数值允许错误**，但必须能在 NPU 上完整跑完一次并返回。

骨架完成后**立即进入 `focused_test`**（编译 + 最小 shape 运行），拿到第一个真实反馈；
再按「正确性 → 边界 → 性能」顺序增量扩展。每次扩展都必须重新通过编译与 focused test。

仍保持一个主实现文件（复用 SHA-256 / 快照 / best 恢复机制）；`solution.json` 随每次
增量同步更新，不做"攒到最后一次性生成"。

代码生成阶段只做静态检查（Python 语法、`run` 入口、`solution.json` 与源码一致性，
UTF-8 内容保留原始换行使源码包 SHA-256 与文件字节一致），不导入 CATLASS、不要求本地
有 NPU；真正的能力验证由紧随其后的 focused test 承担。

**调研预算（可数，不依赖时钟）**：骨架跑通之前，同一文件读取不超过 2 次，总读取不超过
20 次；达到任一上限必须停止调研并落笔写骨架。骨架跑通后进入调试循环，正常开发的重心应
落在"改→编译→上板→看数"上，而不是继续查阅资料。本地只做生成与静态检查；编译、上板、benchmark 在远端完成，见「执行环境」。

## 执行环境

开发机没有 NPU 时可采用"本地生成、远端验证"：本地负责设计、代码生成、`solution.json`
与静态检查；编译、上板、精度比对、benchmark 在远端完成，连接与文件同步由用户自备。
两种模式的证据要求一致：`result.evidence` 必须指向实际执行日志，没有执行记录不得记为
`passed`。

**编译与运行必须拆成两条独立命令**，禁止把「改代码 + 编译 + 上板」串进同一条命令：

```bash
# 编译：不绑定设备，只做 lowering/构建
unset ASCEND_RT_VISIBLE_DEVICES; timeout 300 <compile>

# 运行：占卡执行，必须自带硬超时
export ASCEND_RT_VISIBLE_DEVICES=<id>; timeout -k 5 60s <run>
```

单次算子执行超过 **60 秒**即判定超时或不可用，立即停止该次执行，不得继续等待。
运行命令必须由 `timeout -k 5 60s` 包裹、不得裸跑：设备同步超时
（`rtDeviceSynchronizeWithTimeout`）默认要等 8 分钟以上，裸跑会让每轮调试固定损失这段
时间；返回码 124/137 或等价超时信号即判定 60 秒超时。计时范围为单个 case 完整 `run`
调用至设备同步完成，涵盖 correctness、warmup、性能采样与 profiling。触发后暂停后续
case、采样、优化轮次与最终验收，保存触发用例、输入规格、设备、源码摘要、耗时与日志后
转入 `debug` 定位（同步阻塞、死循环、访存或运行环境），修复后重试；不得用该次执行生成
性能结论或可复用验收证据。

## 机制验证预算

骨架成形前允许验证少数不确定机制，但验证必须有计数上限、有出口、可复用：

- **默认不写独立探针**。优先使用 `python/tla_dsl/examples/` 与 `catlass/examples/` 中已
  覆盖的写法；examples 未覆盖、但能通过编译（frontend lowering）或一次数值对比判断的，
  直接写入主实现后再验证。
- 只有同时满足三条才允许写独立探针：examples 未覆盖；失败会推翻已批准设计（而不是只需
  改常量或 tiling）；无法用编译检查或分段运行判断。
- 单个探针必须：只回答一个是/否命题；自带断言与退出码；单次运行有 ≤60 秒超时；同一
  探针修改 3 次仍不通即判定该机制不可用，按 DESIGN.md 的备选路径推进，禁止继续调试探针
  本身。
- 探针总数不超过 5 个。超出即停止验证，未验证机制在 `ITERATIONS.md` 中标注为假设，
  留待实现阶段的 focused test 验证。
- 探针通过的代码必须直接提升为主实现的骨架函数，禁止"探针 → 记笔记 → 重写"；探针得到的
  容量、对齐、语义结论必须立即回写 DESIGN.md 的契约约束清单。

## 高风险接口选型

`tla.gather` 在当前 DSL 版本上属于高风险接口：表宽受单寄存器限制（f32 ≤64 lane，超限报
`valid_lanes must be <= 64`），索引必须来自向量运算（裸载入的向量会静默返回全 0），多块
组合时还容易触发后端 legalize 失败。**默认不要用它，能不用就不用。**

需要按索引从大缓冲区里取数时，采用已实测性能最优的做法：

**预计算索引（bucket + 链表）+ SIMT 标量读**

- 阶段 1（每个 M 块一次）：把每个 id 映射到 V 方向桶，桶内用链表串起同桶的 k 位置；
- 阶段 2（每个 V 块一次）：沿链表游标，用 SIMT 标量读 `x[row, tok - v_block_offset]`
  直接写 payload；
- 每个 id 在整个 V 循环里**只被访问一次**，总量 `O(M·K + M·V/BN)`，与 V 无关。

约束与实现细节见 `knowledge/dsl/pattern-precomputed-index-structure`。

只有在索引方案确实不可行时才考虑其他路径：逐 lane 标量读（性能次之），最后才是 `gather`
（须遵守 2.1.0 契约并用已知表做数值对拍）。选型理由写入 DESIGN.md。

## Agent loop

状态推进：`skeleton → implement → focused_test →（数值收敛后）targeted_review（high risk）
→ final_review → full_test → benchmark → optional optimize → finish`。

**第一份提交必须是 `skeleton`**：可编译、能在 NPU 上完整跑完一次的最小端到端路径
（数值允许错误）。骨架未提交前，不允许进入额外的源码调研或编写探针。

1. **Validate.** 完整首版生成后，执行获批 focused cases，覆盖完整计算链路及受影响用例。
   没有独立 focused_test 命令时执行获批 full_test 命令，不跳过首轮验证。
   编译、运行、精度、review 或 full test 失败转 `debug`；失败后不测性能。
2. **Analyze and query.** 遇到问题先读日志、DESIGN、当前源码及 `ITERATIONS.md`，按算子族、
   API、错误类型、shape/dtype/layout 查询 `debug`、`dsl` 或 `operator`，性能问题查询
   `optimization`。先 `query --compact`，再对实际采用条目调用 `get` 读取全文。
   记录规范化查询、匹配模式、score/matched_fields/matched_terms；零命中时调整条件重查，
   不扫描整个知识库，不把建议直接作为已采用知识。
3. **Debug.** 一轮验证一个假设，写明 Expected 和判伪条件，然后增量修改完整实现。
   每次修改重新生成 `solution.json`，不修改 reference、workload、容差或计时边界换取通过。
   `debug` 完成后返回 `focused_test`。
4. **Milestone submission.** 只在里程碑提交 `advance`：骨架跑通、focused test 通过、
   数值收敛、或判定 blocked。探索阶段的每次小改动不必提交，也不产生新的 iteration；
   提交时绑定当前 kernel SHA。审计链由里程碑快照 + 最终快照保证，不在探索期逐轮收税。
5. **Continue.** 读取状态机的 `status`、`next_action` 和已失效门禁，不凭记忆跳步。

```bash
python3 skills/catlass-dsl-develop/scripts/develop_state.py advance \
  --state <run>/state.json --result <run>/traces/iter-NNN-<stage>/submission.json
```

NPU/CANN/CATLASS 不可用时记录 `blocked/not_run` 和精确重跑 argv，保留完整实现及当前
`next_action`；环境就绪后可重试该阶段，无需重启 run 或重新生成代码。未执行不是通过。

## Review, test, and benchmark

- review 在**数值收敛之后**执行：`focused_test` 通过且数值稳定后，才做 targeted review
  （high risk）和 final review，然后跑全部 required cases。编写与调试阶段不因 review
  未做而阻塞——写之前能拿到的唯一真实反馈是"跑一次"。
- review 必须检查每个阶段的复杂度标注（实现中一行注释即可，不要求展开分析）。
  选择/索引类阶段若为 O(M·K·nv)（按 V 块重新扫描全部 K），必须要求给出理由与实测代价，
  否则判不通过并要求改为 O(M·K)。
- review 默认检查接口、数学语义、数据流、融合范围及 DSL 实现；严格单融合模式额外检查
  自由名字、唯一声明、单 launch、无独立 dispatch 和无 Torch/Tensor 计算替代。
- full test 通过后始终调用 `catlass-dsl-bench`，要求正确性、有限正数 candidate mean 和
  profiling 通过。默认 `anti_hack.status=disabled`，不要求审计 manifest；严格模式要求
  `anti_hack.status=passed`，并复检原始逐 iteration CSV 和 manifest，缺失即失败。
- 多 kernel 时测量完整 `run`，包括所有阶段；不挑选其中一个 kernel 的耗时作为整体性能。
- 代码变化使旧测试、review、benchmark 和知识准入失效；每个结果绑定当前 kernel SHA-256。
- 有性能目标且未达标才调用 `catlass-dsl-optimize`，继承融合/anti-hack 策略。采用 best 后
  优先接收其 finalize 的完整验收记录，再 final_review；证据完整且评审未修改代码时直接
  finish。缺少可复用证据时才重新 benchmark、full test，不能采用普通轮次或最后一次尝试。
- 选择空闲 NPU，执行前 source 项目 `env.sh`。candidate profiler 每轮重新采集，Torch
  reference profile 复用项目共享缓存；普通 trace 归档只保留 CSV。严格模式的 optimize
  最终验收移交另外保留复核所需的审计分片，不保存完整 profiler trace。

## 性能收口：无性能目标时无需询问，直接执行至少三轮优化

**触发条件**：用户提示词或获批 DESIGN.md 中没有给出明确的性能目标（耗时上限、加速比、
"必须快于参考实现"之类）。此时 benchmark 全绿只代表"能跑对"，不代表"做完"。

**为什么必须继续**：实测出现过验收形状下 candidate 慢于 reference（加速比 0.8x 量级）
仍首个全绿就交回的情况——它自己已定位三处优化（L1/L0 软件流水、A tile 一次载入服务多个
V 块、修好 L0C→UB 去掉 GM 中转）却一处未做。**慢于参考实现说明数据通路或流水仍有明显
空间，直接交付等于把已知可改进项留给用户。**

**必须完成的动作**：benchmark 与 profiling 全绿后，**无需询问用户、无需等待确认，直接执行
至少 3 轮**性能优化迭代。除下面的提前停止条件成立或用户主动叫停外，不得就
"是否继续优化""要不要再优化一轮"征询用户意见。

**评测规模上限**：这三轮优化的**性能计时**最多使用 30 条代表性用例，不跑全量 workload 计时。
代表性用例由你按该算子的计算特征自行选取，并固化到 `workload_perf.jsonl`，三轮共用同一份，
不得每轮换样本；workload 总用例数超过 30 条时，代表性用例取 20~30 条，不足 30 条时全部使用。
正确性门禁仍按原 workload 全量执行，只有性能计时收敛到该子集。`bench.py` 每次运行都会先做
正确性再计时，没有只跑正确性的开关，因此全量正确性由 full_test 阶段的全量 bench 保证，该次
顺带产生的计时不参与优化前后对比。

性能计时同样受「单个 case 一次运行 60 秒」约束：批量执行不得把整批包进一次性 `timeout`，
必须按固定分片逐段执行，每段写独立 `--output`；`bench.py` 没有 `--resume` 参数，续跑依赖
已有分片产物而不是单进程恢复。禁止用 `rm -rf` 丢弃已完成结果后从头重跑。
「机制验证预算」中的独立探针单次运行同样按 60 秒硬超时执行。

每轮按下列顺序执行，任一步缺失视为该轮未完成：

1. **瓶颈取证**：第一步是零成本算术：`makespan = ceil(总块数 / 核数)`。实测耗时 ≈
   makespan × 单块理想耗时即为任务铺放空转，不是数据通路问题——用"块数恰等于核数的
   形状耗时台阶式减半"一次运行即可坐实，修法是把工作单元再切一维、在整除因子里选
   组数，禁止把实验轮次消耗在带宽/流水/分块上（判据与修法见
   `knowledge/optimization/makespan-scheduling-gap`）。排除铺放空转后，再从本轮 fresh
   profiling 的原始数据指出主导瓶颈——哪个 pipe 占比最高、属于搬运受限还是计算受限、
   该 pipe 的理论下限是多少——写入 `ITERATIONS.md`；profiling 计数器不可用（如全 0）
   时改用单变量消融计时实验取证。禁止只写"还能优化"这类无证据结论。
2. **单元假设**：针对该瓶颈提出**一个**改动假设，写明预期收益与判伪条件
   （例："L1/L0 双缓冲后 MTE2 与 MAC 可重叠，预期总时长降到 max(MTE2, MAC)"）。
   一轮只改一个变量，不混合多个改动轴。
3. **针对性修改**：只改与假设相关的实现；保持正确性门禁可重跑。
4. **前后对比**：同一 `workload_perf.jsonl`（≤30 例）、同一设备重跑 benchmark，记录 candidate
   mean 前后值、相对参考的加速比、是否达成预期；未达成时说明原因，不得粉饰或只报聚合值。

**允许提前停止的唯一条件**：某轮 profiling 证明剩余时间已接近硬下限（例如 >80% 时间
落在单一 pipe 且该 pipe 已达理论带宽/算力），并给出量化依据。"感觉差不多了"不算。

**交付前检查**：`ITERATIONS.md` 必须含 `## Performance Rounds` 表（至少 3 行，或附硬下限
证据），每行含：瓶颈、改动、before→after、是否达成预期。`finish` 时随交付物一并提交。

**与既有规则的关系**：有明确性能目标且未达标时仍走 `catlass-dsl-optimize`；本条适用于
**没有目标**的情形，目的是把"能跑"推进到"接近该实现方式的极限"。

## Verification reuse

默认启用证据复用。design 明确 `focused_cases` 与实际命令；focused_test 生成的逐 case
执行报告覆盖全部 required cases 和获批命令时，final_review 后不重复 full_test。
optimize 的 finalize 完整验收也可移交给 develop，复核后只补 final_review。
两者均绑定源码、规格、命令、环境和测量参数，finish 前重新核对，变化后补跑门禁。
实际测试执行器报告、当前环境快照、`export-verification-context`、optimize 的
`--verification-context` 和 submission 扩展见
[证据复用协议](references/verification-reuse.md)。不能用 submission 的覆盖声明替代报告。
模板：[focused-test submission](templates/focused-test-result.json)、
[执行器报告](templates/test-report.json)；报告中的状态必须由真实执行结果填写。

## Finalization and knowledge

`finish` 必须证明当前源码的 full test、final review、fresh benchmark 和 profiling 均通过；
源码包也必须一致。保存真实源码到 `final/kernel.py`，以及紧凑的 `final/result.json`。
`ITERATIONS.md` 是唯一人类入口，表头为：

```markdown
| Iter | Title | Score | Passed | Notes |
```

最终用 `## Workload Speedups` 表格列出每个 workload UUID、candidate/reference mean ms
和 `reference_mean / candidate_mean`，不只报告聚合指标。

没有声明性能目标时，`finish` 还要求 `ITERATIONS.md` 含 `## Performance Rounds` 表且
至少三轮（或附证明已达硬下限的量化证据），否则不予交付。

仅将完整测试或 profiling 直接支持的经验在完成时批量交给 `catlass-dsl-knowledge record`，
绑定最终 kernel SHA-256。证伪假设、静态推断及 `not_run` 不入库。

## Compact artifacts

```text
.catlass-dsl/develop-runs/<run-id>/
├── state.json
├── ITERATIONS.md
├── traces/iter-NNN-<stage>/
│   ├── kernel.py
│   ├── result.json
│   ├── failure.txt                  # 仅 failed/blocked/not_run
│   ├── validation/                  # 可选，参与复用的执行报告
│   └── profile/case-NNNN/{kernel_details.csv,step_trace_time.csv}
└── final/
    ├── kernel.py
    ├── result.json
    └── profile/case-NNNN/{kernel_details.csv,step_trace_time.csv}
```

每轮快照必须与结果 SHA-256 一致。submission 解析后删除，日志仅保留有界失败摘要。
只接受 `catlass.dsl.workflow.v4`；旧 run 不自动迁移，需重新创建设计配置并批准。
