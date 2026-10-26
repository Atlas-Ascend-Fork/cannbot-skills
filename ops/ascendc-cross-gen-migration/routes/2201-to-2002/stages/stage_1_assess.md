# 阶段 1：评估定级（含 m200 风险扫描）

**前置条件**：`[GATE-0] ENV_CONFIRMED` 已输出。

## LOADED 检查点

| 序号 | MUST READ 文件 | LOADED Token |
|------|---------------|-------------|
| 1 | `references/m200-port-guide.md` | `[LOADED] m200-port-guide` |
| 2 | `references/m200-cube-guide.md`（**cube 类算子** MUST，见 Step 1.2） | `[LOADED] m200-cube-guide` |

**★ 本文件 MUST 在算子源码读取前加载。** m200-port-guide 定义了特性缺失/硬件 errata/预算核数三个维度的风险扫描清单。3510 路线的 api-diff-guide / cube-migration-guide / l2-guide / api-mapping / regbase sub-skill 在本路线**禁止加载**（RegBase/MicroAPI 在 2002 不存在）。

## Step 1.1：读取算子源码

MUST 读取以下文件：

| 文件 | 检查内容 |
|------|----------|
| `_def.cpp` | SoC 列表、DataType 配置（**bf16/fp64/fp8 命中标记**，310P 全部无硬件支持） |
| kernel `.cpp` / `.h` | API 使用模式、条件编译、**跨核同步原语（CrossCoreSetFlag/WaitFlag、IBSet/IBWait、SyncAll、NotifyNextBlock/WaitPreBlock）** |
| `op_host/CMakeLists.txt` | 是否已有 ascend310p 分支 |
| Tiling `.cpp` / `.h` | 核数/UB 是否硬编码来源平台值（如 910b 核数、`GetBlockNum()*GetSubBlockNum()` 分片——fuse 核 GetSubBlockNum 实测恒 1） |

## Step 1.2：查阅全量仓参考（按需）

310P 路线下全量仓降级为按需（2201→3510 迁移指南不适用；310P 平台知识真源是本包 `references/m200-port-guide.md`）：

- **查看同类算子样例**：`$DEVKIT_PATH/examples/` 目录下按编程模型分类查找
- **查阅算子实践参考**：`$DEVKIT_PATH/docs/zh/guide/operator_practice/` 下查找同类算子的实现和优化案例
- **Cube 类算子**（Step 1.1 中发现 Mmad/LoadData/Fixpipe/DataCopyCO12DstParams 等 cube API）：MUST 先读 `references/m200-cube-guide.md`（输出 `[LOADED] m200-cube-guide`）

读取全量仓文件后输出：`[LOADED] $DEVKIT_PATH/<相对路径>`

### Step 1.2.5：★ 知识库卡检索（MUST）

m200-port-guide 的 errata 目录是**入口知识**而非全集——cannbot-knowledge 有 150 张 `platforms: ["310p"]` runbook 卡（compilation/performance/precision）且持续增长。本步强制执行：

1. 经 `knowledge-query`（配置 `CANNBOT_KNOWLEDGE_ROOT`）按 **算子族 + 平台 310p** 检索（如 conv/matmul/gather/datacopy + 本算子用到的关键 API 名）
2. 命中卡逐张阅读，与 Step 1.3 扫描结果对照：**卡中有而 errata 目录没有的命中点 MUST 补入扫描结果**（分析摘要 §7.2 注明来源卡路径）
3. 未命中任何新信息也要在分析摘要注明「已检索，无增量」
4. 卡与本包 errata 目录冲突时以本包为准（本包内容经本路线演练实证），冲突点上报

**证据**：分析摘要中含「知识库检索」小节（检索词、命中卡路径清单、增量结论）。

## Step 1.3：★ m200 风险扫描

**基于已加载的 `references/m200-port-guide.md`，对算子源码做三维度风险扫描。** 扫描结果记入算子源码分析摘要。

### 1.3.1 特性缺失扫描（P0）

按 `references/m200-port-guide.md` §2 清单逐项 grep：

| 扫描项 | 检测方法 | 命中后的处理 |
|--------|---------|-------------|
| bf16 使用点 | grep `bfloat16_t` / `bf16` | 移除 dtype 或 host widen fp32（§2.1） |
| fp64 / fp8 / int4 | grep `double` / `fp8` / `hifloat8` / `int4b_t` | 按 §2.2 处置；int4+cube 须先实证 int8 Mmad（U5） |
| fp32 matmul | kernel 含 Mmad/Matmul 且契约 fp32 | MUST 按 §2.3 决策树选路线并记录理由（含门禁语义依据） |
| 3510 新增设施残留 | grep `RegTensor` / `__VEC_SCOPE__` / `PRECISION_1ULP_FTZ_FALSE` / `FixpipeParamsArch3510` | 命中说明来源不是纯 arch22——stop 并上报用户 |

### 1.3.2 errata 命中扫描（P0）

逐项 grep 特征扫描（处置知识在 cannbot-knowledge 的 310P 卡中，命中后经 `knowledge-query` 按卡名读取；卡路径相对知识仓 `knowledge/ops/ascendc/runbooks/`）：

| errata | grep 特征 | 处置卡 |
|--------|----------|--------|
| E1 | `DataCopyPad` / `Copy<` | compilation/pb_105 + platform_path_capability_fallbacks |
| E2 | `Brcb` | precision/brcb_mask_residue_restore_full_mask |
| E3 | 标量 UB 读写（`GetValue`/`SetValue` 或标量指针解引用） | runtime/dav_m200_scalar_path_sync_visibility |
| E4 | tiling 中 `KTILE` / K 轴分片 =128 | precision/dav_m200_cube_output_readout_traps |
| E5 | kernel 内直接解引用 tiling GM 指针 | runtime/dav_m200_scalar_path_sync_visibility |
| E6 | DataCopy 后紧跟 vector 消费且无 PipeBarrier | runtime/dav_m200_cross_pipe_sync_event_primitives |
| E8 | `1UL <<` / `1UL<<` 移位 mask | compilation/dav_m200_kernel_semantic_walls |
| E9 | UB→L1 `DataCopy` | performance/dav_m200_data_movement_primitive_selection |
| E10 | `CAST_RINT`（fp32↔fp16） | compilation/pb_119 / pb_120 |
| E11 | `LoadData2D` / `Fixpipe` | compilation/pb_117 / ol_544 |
| E12 | Mmad 多分形 tile（L0C 读出寻址） | precision/dav_m200_cube_output_readout_traps |
| E13 | L0C→UB 读出 / UB→GM 逐行 64B 存储 | precision/dav_m200_cube_output_readout_traps |
| E14 | `GatherMask` 源非 32B 对齐 | precision/gathermask_vreduce_src_32b_alignment |
| E15 | 纯 vector kernel 手工 SetFlag/WaitFlag 事件链 | runtime/dav_m200_vector_kernel_event_chain_hang |
| E16 | 多核写出路径含 `SetValue` on GlobalTensor | runtime/dav_m200_multicore_scalar_gm_write_drop |
| E17 | fp16 标量 GM 读 | runtime/dav_m200_fp16_scalar_gm_read_stale |
| E18 | 裸 Mmad 链（无 `REGIST_MATMUL_OBJ`）/ 无冷启动 warmup | runtime/dav_m200_mmad_long_chain_cold_start_warmup |

注：E7（64-bit 标量单指令）是**优化点**、E19（对齐窗口+gather 表+零哨兵）是**解法模式**，非风险项，不入本表——分别见 performance/dav_m200_scalar_emission_floor 与 performance/dav_m200_load_side_vectorization_pattern。

命中即处置（阶段 2，读卡后按卡的规避方案）；未命中在分析摘要标注"未命中"。

### 1.3.3 预算与核数重估（P1）

UB 248KB→240KB、核数→8（`references/m200-port-guide.md` §4）：tile 尺寸与 block_dim 分片是否溢出重算。

### 1.3.4 性能形态评估（P0，route.md 性能形态三约束的前置）

| 评估项 | 内容 | 产出 |
|--------|------|------|
| 核数打满可行性 | 逐 kernel 分析分片维度：典型 shape 下总 tile 数 vs 8 核；tile 数不足时的拆维/合并方案 | 每 kernel 的分片方案（保证 block_dim=8） |
| 向量化风险点 | E1/E14 命中处的替代方案选型：layout 重排天然对齐 / UB 内 vector gather·shift / 对齐直通——**禁止标量逐元素环**（route.md 约束） | 每个命中点的向量化模式选择 |
| cube 化判定 | 含矩阵乘结构的计算阶段哪些上 cube（高阶 Matmul API）、哪些留 vector（仅限搬运/elementwise/softmax 类）、理由 | cube 化判定结论（写入分析摘要） |

### 1.3.5 subnormal 扫描（标注待实证，不预设立场）

310P 是否裁剪 subnormal **未实证**（U1）。证据链要求（dtype/输入范围/中间计算/常量/计算链五段分析）与既有纪律一致保留——

当算子使用 Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt 时，MUST 基于证据链分析输入是否可能含 subnormal（含 eps/常量 dtype 可表示性检查）；但结论标注为「**待 stage 4 实证**」：

- **不预设 FTZ 裁剪结论**（不假设 310P 与 A5 一样裁剪）
- **不做预防性适配**（`PRECISION_1ULP_FTZ_FALSE` 是 3510 设施，2002 不存在）
- 阶段 4 用 subnormal 用例实证实际行为，回写 `references/m200-port-guide.md` §6（知识闭环）

## ## Step 1.4：判定迁移层级（310P 语义）

按本路线 `route.md` 的决策树判定 L1/L2/L3。★ 注意本路线层级语义与 3510 路线**不同**：

| 层级 | 310P 语义 | 触发条件 |
|------|----------|---------|
| **L1** | 纯 host 适配（`AddConfig("ascend310p")` + `config/ascend310p/` + 构建） | kernel 无 errata 命中、无特性缺失、UB 富余——**适用面比 3510 路线宽**（同为 MemBase，kernel 可零改动） |
| **L2** | kernel 适配：dtype 降级 + errata 规避 + UB 收紧 + 核内流水同步盘点 | 命中 E1-E15 任一 / 有 dtype 降级 / UB 超预算。**310P 无 RegBase，L2 ≠ RegBase 重写** |
| **L3** | cube 深度重写（fp32 路线选型、Fixpipe 消除、ND2NZ 操作数路径）或性能重写 | cube 类 fp32 算子 / 性能关键算子。SIMT 在 2002 未实证（U2），默认不适用 |

cube 类算子定级后 MUST 按 `references/m200-cube-guide.md` 执行；3510 路线的 cube-migration-guide（低阶直跑评估/分离模式/flagId 避让区）**不适用**——310P 低阶 cube API 大面积缺席（E11）、fuse 核无 AIC/AIV 分离。

## Step 1.5：同步协议盘点

**310P fuse 核前置说明**：310P `cube_vector_combine=fuse`，无 AIC/AIV 分离——3510 路线的 AIC↔AIV 参与集合粒度问题域（模式 0/1/2/4 语义对照、flagId 避让区）**整体不适用**；发现跨核同步原语时先注明"fuse 核，AIC/AIV 集合粒度不适用"。盘点重点转为**核内流水同步**（E3 scalar 无隐式 auto_sync、E6 MTE2_V early-fire、E15 vector-only 事件链挂死——处置卡见 Step 1.3.2；同步语义背景见 `references/m200-port-guide.md` §5）。CrossCore* 原语语义仍以 310P 平台文档为准（"模式号相同 ≠ 语义相同"约束不变）。

**MANDATORY**：Step 1.1 中发现任何跨核同步原语（CrossCoreSetFlag/WaitFlag、IBSet/IBWait、SyncAll、NotifyNextBlock/WaitPreBlock）时，MUST 执行本步；未发现时跳到 Step 1.6。

跨核同步协议**按原语逐个核对**，不要整体判定。四步法：

**① 枚举同步点**：列出 kernel 中所有核间同步原语调用，每个同步点记录：

| 字段 | 内容 |
|------|------|
| 同步点位置 | 文件 + 行号 + 所在函数 |
| 原语 | CrossCoreSetFlag/WaitFlag、IBSet/IBWait、SyncAll 等 |
| 模式号 | 如 `CrossCoreSetFlag<2, PIPE_MTE3>` 的模式 2 |
| flagId 表达式 | 如 `SYNC_C1_V1_FLAG[(taskId + 1) & 1][blockIdx & 1]`——**含核身份索引（taskId/blockIdx/subblock 号参与索引）时重点标记** |
| 调用方角色 | AIC / AIV / 核内 subblock 序号（`ASCEND_IS_AIC`/`ASCEND_IS_AIV` 分支）。无核类型分支的"全执行"结构（所有核无条件执行 bmm/vector/写输出）在 A2/A3 常见，310P 为 fuse 核、无分离模式，该问题域不适用（见本节顶部 fuse 核前置说明） |

**② 推断参与集合假设**：写出每个同步点"谁等谁"的粒度——单对（AIC↔指定 AIV）/ 组内全集合（AIC↔全部 AIV）/ 全核（所有 AIC 或所有 AIV）。**flagId 含核身份索引的，即隐式假设了"flagId 级配对粒度"**（用不同 flagId 区分不同对端）。

**③ 对照目标平台语义**：读 `$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.md` 与 `CrossCoreSetFlag_ISASI.md`，核对：
- 所用模式在目标平台的参与集合定义与 flagId 计数器驱动条件（模式 0 = 全核、模式 1 = 组内全部 AIV、模式 2 = AIC↔组内全部 AIV、模式 4 = AIC↔单个 AIV）
- 型号支持范围（**模式 4 仅 950PR/950DT 支持**）
- **模式号相同 ≠ 语义相同**：同一模式跨架构可能语义变化（如模式 2 从 flagId 级配对收紧为全集合），必须以目标平台文档为准，不得假设与旧平台一致
- **参数写法按目标平台文档生效**：旧平台被忽略的参数在目标平台可能生效——无参调用（如裸 `CrossCoreWaitFlag(flagId)`）在 950 按默认模式 0（全核）执行，必须与配对 SetFlag 的模式号显式核对；"不写参数，语义也在变"（实战：910b 无参 WaitFlag 照常工作、950 上生效致模式错位死锁——310P 语义以 310P 平台文档为准）
- **参与集合恒定**：同一 flagId 的 Set/Wait 是否在所有参与核的**无条件路径**上执行——被 needExec/needPair/if 条件分支包裹、循环 0 次（分片边界）、空闲核都可能造成参与核缺 Set（同上，沿用决策规则第 4 条）

**④ 处置决策**：每个同步点给出结论——

| 结论 | 条件 |
|------|------|
| 沿用 | **4 项全部静态论证通过**：① 集合粒度 ≥ 目标平台文档定义粒度（含等号）② 模式在目标平台支持 ③ 参数写法按目标平台文档生效 ④ 参与集合恒定（Set/Wait 在所有参与核的无条件路径上） |
| 禁用（走标准路径） | 4 项中任一项无法静态论证（如粒度 < 文档粒度、无参调用默认语义错位、Set 可能缺位）→ 死锁风险 |
| 改用其他模式 | 需支持单对粒度时改用模式 4——但须核对型号支持范围（仅 950PR/DT） |

**通用判定规则：假设集合粒度 < 目标平台文档定义的集合粒度 → 死锁风险 → 禁用该路径走标准流程，禁止原样移植。默认值反转：沿用是论证结论，不是赌注——4 项中任一项无法静态论证即禁用/降级走标准路径，禁止"沿用 + 运行时验证"式风险转移（死锁失败从方案阶段 1 次静态检查前移到运行时排查=成本后置）。**

## Step 1.6：生成算子源码分析摘要

**MANDATORY**：无论算子有无文档，都 MUST 基于被迁移源码生成摘要，并在对话中展示完整内容，同时写入文件 `${OP_DIR}/analysis_<op_name>.md` 持久化保存。

摘要内容：

```markdown
# 算子源码分析摘要：算子名

## 1. 算子功能描述
- 计算逻辑（从 kernel 代码逆向提取）
- 算子类别：elementwise / reduction / attention / matmul / 其他

## 2. 接口签名
- 输入参数（名称、类型、shape 约束）
- 输出参数（名称、类型、shape 计算规则）
- 属性参数

## 3. 支持的数据类型
- 从 _def.cpp 和 kernel 代码提取
- 列出：FP16 / BF16 / FP32 / INT8 / FP8 / HiFloat8 等

## 4. 计算逻辑伪代码
- 从 Compute 函数提取 API 调用序列
- 标注升精度路径、条件编译分支

## 5. 关键 API 使用
- 列出所有 CANN API 调用（DataCopy/Cast/Reduce/Sort 等）

## 6. 边界条件与特殊值
- 输入域约束、NaN/Inf 处理

## 7. ★ m200 风险扫描结果
### 7.1 特性缺失
- bf16/fp64/fp8/int4 命中情况与处置：
- fp32 matmul 路线决策（含门禁语义依据）：
### 7.2 errata 命中
- E1-E15 逐项：命中（处置方式）/ 未命中
### 7.3 预算重估
- UB 预算重算结论（≤240KB）；block_dim 按 8 核分片结论
### 7.4 subnormal（待实证）
- 是否使用 Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal：是/否（是时标记「待 stage 4 实证」并附证据链：dtype 分析 + 输入范围 + 中间计算 + 常量 eps 可表示性 + 计算链）
### 7.5 性能形态（三约束前置评估）
- 核数打满：逐 kernel 分片方案与 block_dim=8 可行性
- 向量化：E1/E14 命中点的向量化替代模式选择（禁止标量环）
- cube 化判定：各计算阶段 cube/vector 归属与理由

## 8. 同步协议盘点结果（有跨核同步原语时）
- 同步点清单：位置 / 原语 / 模式号 / flagId 表达式 / 调用方角色
- 参与集合假设与目标平台语义对照结论
- 处置决策：沿用 M 个 / 禁用 K 个 / 改用 R 个（按 Step 1.5 ④）
- 输出格式：`[SYNC-AUDIT] 同步点 N 个：沿用 M 个 / 禁用 K 个 / 改用 R 个`
```

### 逆向分析方法

> 完整五步逆向方法论（接口签名提取 → 计算逻辑逆向 → 输入域约束 → 边界条件 → PyTorch 对标）见 `../../references/precision-testing/source-code-reverse-analysis.md`。

| 方法 | 适用场景 |
|------|---------|
| 代码逆向分析 | 所有算子——从 Compute 函数逐行提取 |
| 同类算子类比 | 算子名可推断功能时（如 `fast_gelu` 类比 `gelu`） |
| 测试用例反推 | 有已有 UT/ST 时 |
| PyTorch 对标 | 有 PyTorch 同名接口时 |

**有文档算子**：先从源码提取，再与文档交叉对比，记录差异。测试用例始终以源码为准。

## Step 1.7：输出迁移方案

基于 Step 1.4 判定的层级、Step 1.3 的 API 差异扫描结果、Step 1.5 的同步协议盘点和 Step 1.6 的源码分析摘要，生成迁移方案并在对话中展示：

| 项目 | 内容 |
|------|------|
| 迁移层级 | L1/L2/L3 |
| 涉及文件 | 列出所有需要新增/修改的文件 |
| 改造范围 | 简述需要改什么（如"移除 V220 guards"、"Compute 函数改为 RegBase"） |
| 同步协议处置 | Step 1.5 各同步点的沿用/禁用/改用结论（有同步点时必填） |
| 预期风险 | 如 UB 预算、特性缺失处置、errata 替代路径开销、核内流水同步（E3/E6/E15）等 |
| ★ m200 风险适配项 | 特性缺失/errata/预算三个维度的适配清单 |
| ★ m200 风险回归 shape | errata 触发路径两侧、fp32-via-fp16 性能档等高风险 shape 清单（供阶段 5 回归使用） |

迁移方案在对话中展示后即可进入阶段 2（展示供用户审阅留痕，无需等待确认）。

## Gate 输出条件

- [ ] 算子源码已全部读取
- [ ] **★ `m200-port-guide` 已加载并输出 LOADED Token**（cube 类加 `m200-cube-guide`）
- [ ] **★ m200 风险扫描（特性缺失/errata/预算重估/subnormal 待实证标记）已完成**
- [ ] 迁移层级已判定并标注理由
- [ ] 同步协议盘点已完成（有跨核同步原语时：同步点已枚举、参与集合假设已推断、目标平台语义已对照、处置决策已输出）
- [ ] 算子源码分析摘要（含 API 差异扫描结果与同步协议盘点结果）已在对话中展示并写入文件
- [ ] 迁移方案（含 API 差异适配项与同步协议处置）已在对话中展示
- [ ] 明确了算子支持的 dtype 和 shape 范围
- [ ] **★ 已列出 m200 风险回归 shape 清单**
- [ ] **★ 性能形态评估已完成（核数打满分片方案 / 向量化风险点 / cube 化判定，route.md 性能形态三约束）**
- [ ] **★ 知识库卡检索已完成（Step 1.2.5），增量结论已入分析摘要**

**全部通过 → 输出 `[GATE-1] LEVEL=L1/L2/L3` → 进入阶段 2**
