# 路线 2201→2002：A2/A3（DAV_2201）→ 310P（DAV_2002 / dav-m200）

本文件是路线包 `routes/2201-to-2002/` 的入口，承载该架构对的全部路线专属内容，由 `../../SKILL.md`「Step R：迁移路线判定」路由进入。
**回退校验**：若实际源/目标平台不是 DAV_2201→DAV_2002，立即停止并退回 `../../SKILL.md` Step R 重新判定，禁止沿用本文件的差异清单与决策树。
进入本路线后，本文件的约束与 `../../SKILL.md` 的全局约束、Gate 协议同等强制；阶段执行细则在本包 `stages/`、路线专属知识在本包 `references/`、路线无关公共资产在 `../../references/`。本包自包含，禁止引用其他路线包的内容。

## 路线性质（先理解，再动手）

2201→2002 是**下移植**（down-port）：310P 与 arch22 同为 MemBase 编程模型，**无 RegBase/MicroAPI**。因此：

- L2 语义 ≠ RegBase 重写（RegTensor/MicroAPI/`__VEC_SCOPE__` 在 2002 不存在）
- L3 ≠ SIMT（SIMT 在 2002 未实证，默认不适用）
- 主要风险不是"新架构新行为"，而是**特性缺失**（无 bf16/fp8/fp64、cube 仅 fp16）+ **dav-m200 硬件 errata 群**（多处静默错误，编译无感）+ **预算/核数收缩**（UB 工作预算 ~240KB、8 个 AIV fuse 核）
- fuse 核（cube_vector_combine=fuse）无 AIC/AIV 分离——跨核握手问题域消失，风险转移到核内流水同步

## 平台事实速查（详表见 `references/m200-port-guide.md` §1）

| 属性 | 值 |
|------|-----|
| SoC 名（npu-smi / acl.get_soc_name） | `Ascend310P3` |
| `build.sh --soc=` / `AddConfig()` / `op_host/config/` 目录 | `ascend310p`（小写） |
| bisheng `--npu-arch` 直调 | `dav-2002`（`dav-m200`/`dav-2201` 均被拒） |
| 平台宏 / kernel 变体目录 | `__NPU_ARCH__ == 2002` / `arch20/` |
| 核数 / UB 工作预算 / L1 / L0C | 8 AIV（fuse）/ ~240KB / 1MB / 256KB |
| cube | 16×16×16，**仅 fp16 输入**（fp32 累加器）；无 bf16/fp8/fp64 |

## 阶段执行流程

**MUST 按顺序执行。每个阶段开始时先读取本包对应 stage 文件，完成时输出 Gate Token（判据见 `../../SKILL.md` Gate 协议）。**

| 阶段 | 文件（本包内） | Gate Token |
|------|------|------------|
| 0 环境验证 | `stages/stage_0_env.md` | `[GATE-0] ENV_CONFIRMED` |
| 1 评估定级 | `stages/stage_1_assess.md` | `[GATE-1] LEVEL=L1/L2/L3`（层级语义见本文决策树，与 3510 路线不同） |
| 2 代码改造 | `stages/stage_2_implement.md` | `[GATE-2] FILES_MODIFIED=[...]` |
| 3 编译安装 | `stages/stage_3_build.md` | `[GATE-3] BUILD=PASS INSTALL=PASS` |
| 4 精度验证 | `stages/stage_4_precision.md` | `[GATE-4] PRECISION=N/N_PASS`（仅当 N_FAIL==0 时允许输出 PASS；否则输出 `PRECISION=N/M_PASS BLOCKED`） |
| 5 性能验证 | `stages/stage_5_performance.md` | `[GATE-5] PERF=COLLECTED`（证据判据见本包 stage_5，与 3510 路线有变体） |
| 6 性能优化（可选，owner 指令触发） | `stages/stage_6_optimize.md` | `[GATE-6] OPT=<达标/上限收敛/契约冲突升级> SPEEDUP=<值>` |

## 本路线覆盖的两大迁移能力

**① m200 差异迁移适配**（所有层级 MUST 执行），覆盖三个维度：
- **特性缺失**：bf16/fp64/fp8 无硬件；cube 仅 fp16（fp32 matmul 有 fp16-internal 精度地板，路线决策树见 `references/m200-port-guide.md` §2.3）
- **硬件 errata**（E1-E15）：DataCopyPad/`Copy<float,false>` 静默 no-op、CAST_RINT 垃圾、Fixpipe/LoadData2D NOT_SUPPORT、Mmad C 分形 −1024B 偏移等——**编译无感，只能靠精度用例暴露**
- **预算与核数**：UB ~240KB、8 核重分片

详见 `references/m200-port-guide.md`。

**② Cube 类算子迁移**（kernel 含 Mmad/高阶 Matmul API/LoadData 等 cube API 时）：cube fp16-only 的 dtype 降级决策、标准 opp 路线高阶 Matmul API 适配、ND2NZ/L0C 路径，详见 `references/m200-cube-guide.md`（调试方法论见 `references/cube-debug-lessons.md`，其案例来自 3510 但排查方法通用）。

## 迁移层级决策树（310P 语义，与 3510 路线不同）

```
算子是否命中 m200 风险（特性缺失 / errata E1-E15 / UB 超预算）？
  │
  ├─ 否 → L1: 纯 host 适配（AddConfig("ascend310p") + config/ascend310p/）
  │        ——适用面比 3510 路线宽：同为 MemBase，kernel 可零改动
  │
  └─ 是 → 是否为 cube 类 fp32 算子或需 cube 深度重写？
      ├─ 是 → L3: cube 深度重写（fp32 路线选型、Fixpipe 消除、ND2NZ 操作数路径，
      │        按 references/m200-cube-guide.md）
      └─ 否 → L2: kernel 适配（dtype 降级 + errata 规避 + UB 收紧 + 核内流水同步盘点）
               ——明确：310P 无 RegBase，L2 ≠ RegBase 重写

  ★ 无论选择哪个层级，都 MUST 额外执行 m200 风险扫描与适配（见下方）
```

### m200 差异适配（所有层级 MUST 执行）

无论算子判定为 L1/L2/L3，都 MUST 在阶段 1 评估时做 m200 风险扫描，在阶段 2 改造时逐项处置：

| 差异维度 | 扫描时机 | 适配时机 | 参考文件 |
|---------|---------|---------|---------|
| 特性缺失（bf16/fp64/fp8/fp32-matmul） | 阶段 1 | 阶段 2 | `references/m200-port-guide.md` §2 |
| 硬件 errata（E1-E18） | 阶段 1 | 阶段 2 | `stages/stage_1_assess.md` Step 1.3.2（处置卡列） |
| 预算与核数（UB/8 核） | 阶段 1 | 阶段 2 | `references/m200-port-guide.md` §4 |

## 路线专属 Reference 索引

### m200 差异适配（★ 本路线所有层级 MUST READ）

| 文件 | 用途 | LOADED Token |
|------|------|-------------|
| `references/m200-port-guide.md` | 310P 差异适配指南：平台事实卡、特性缺失扫描、errata 目录（E1-E15）、预算与核数、同步语义差异、未实证清单 | `[LOADED] m200-port-guide` |
| `references/m200-cube-guide.md`（cube 类算子 MUST） | cube fp16-only dtype 降级决策树、标准 opp 路线高阶 Matmul API 适配清单、ND2NZ/L0C 路径 | `[LOADED] m200-cube-guide` |

**本路线禁止加载**（RegBase/MicroAPI 在 2002 不存在，跨路线取材即违反自包含约定）：3510 路线包的 `l2-guide.md` / `api-mapping.md` / `l3-guide.md` / `api-diff-guide.md` / `cube-migration-guide.md`，以及 `ascendc-regbase-best-practice` 知识 skill。

### 实战经验（按需加载）

| 文件 | 内容 | LOADED Token |
|------|------|------|
| `references/cube-debug-lessons.md` | 死锁/卡死/结果错误排查方法论（案例来自 3510，方法通用；阶段 4 测试失败/精度异常时 MUST 查阅） | `[LOADED] cube-debug-lessons` |
| `references/multi-stage-guide.md` | 多阶段 UB 复用模式（方法论通用；**数值为 3510 取值，310P 预算以 m200-port-guide §4 为准**） | `[LOADED] multi-stage-guide` |

### 外部知识库（cannbot-knowledge，按需查阅）

本路线包自包含核心知识；**cannbot-knowledge 仓另有 150 张 `platforms: ["310p"]` 卡**（`knowledge/ops/ascendc/runbooks/{compilation,performance,precision}/`，含 pb_119 cast 只能 CAST_NONE、pb_127 MTE2 DataCopy 字节约束、ec_203/204 raw launch 契约等，与本包 errata 目录同源），按需经 `knowledge-query` skill 检索（配置 `CANNBOT_KNOWLEDGE_ROOT`，用法见 `../../SKILL.md` 知识依赖节）：

- **阶段 1（Step 1.2.5，MUST）**：按算子族+平台 310p 强制检索，增量入分析摘要
- **阶段 3/4 报错时**：按错误码/症状检索（507014/507015/PB-/EC- 系列）
- **阶段 6（Step 6.0，MUST）**：按瓶颈模式强制检索 performance/precision 卡，负结果用于排除优化方向
- **命中卡片与本包 errata 目录冲突时**：以本包为准并在卡上标注（本包内容经本路线演练实证）

**知识卡化（MR !93 起已落地）**：本路线的知识正文已全部沉淀为 cannbot-knowledge 卡（`platforms: ["310p"]`）——`references/m200-port-guide.md` 是扫描程序入口（不含知识正文），E↔卡对照表在 `stages/stage_1_assess.md` Step 1.3.2：
- 平台事实：`compilation/ascend310p_platform_facts_quick_reference`（+ `common/platforms/concepts/target_ascend310p` 互链）
- E↔卡对照：见 `stages/stage_1_assess.md` Step 1.3.2 扫描表
- U 系列实证：U4→`performance/profiling_toolchain_310p_pipe_reading`；U6→`compilation/fuse_core_getsubblocknum_always_one`；U8→`compilation/neff_ge_64_archive_rule_also_required_m200`；U9→`performance/dav_m200_data_movement_primitive_selection`；U10→`performance/ol_15`
- eager-parity 门公式：`precision/fp16_cube_fp32_semantics_310p`

## 路线专属约束（与 SKILL.md 全局约束同等强制）

1. **★ bf16 禁用**——dav-m200 无 bf16 硬件：dtype 覆盖矩阵、def.cpp、测试用例一律移除 bf16；业务必须时按 `references/m200-port-guide.md` §2.1 host widen 处置
2. **★ 命名纪律**——kernel 变体目录用 `arch20/`、平台宏 `__NPU_ARCH__ == 2002`、`AddConfig("ascend310p")`/`config/ascend310p/`（小写）；**禁止用 `arch35/` 目录名装 2002 代码**；SoC 字符串三套拼写（`--soc=ascend310p` / `Ascend310P3` / bisheng `dav-2002`）以 stage 0 实证落盘为准，不得混用
3. **★ 禁止以编译通过推断正确**——E1（DataCopyPad/`Copy<float,false>` 静默 no-op）与 E10（CAST_RINT 垃圾）编译无感，只能靠阶段 4 精度用例暴露
4. **★ 产物 MUST 含精度与性能测试数据**——产物即算子工程本体，其中必须落盘：精度报告 `<op>_precision_report.md`+`.json`（与测试脚本同目录）、性能报告 `perf/<op>/perf_report_<op>.md`、profiler/计时原始数据（`perf/<op>/case_*/` 或等效目录）——以 stage 4/5 的 Gate 证据约束为准
5. **★ fp32 matmul 路线选型 MUST 写明门禁语义依据**——绝对真值门（CPU 标杆）下 hi/lo 拆分合法；参考相对门下"更准的候选"反而会挂（vendor-parity 教训），见 `references/m200-port-guide.md` §2.3
6. **★ subnormal 不预设立场**——310P 是否裁剪 subnormal 未实证（U1）：含 Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt 的算子在阶段 4 实证记录实际行为并回写 `references/m200-port-guide.md` §6
7. **核数/SubBlockNum 以实证为准**——`GetSubBlockNum()` 在 fuse 核实测恒 1（U6）；来源算子 `GetBlockNum()*GetSubBlockNum()` 分片逻辑需改为仅 `GetBlockNum()`
8. **★ 绑定层组合语义 fail-loud**——绑定层有"超阈值自动拆段"等组合语义的算子，组合语义 MUST 有内部属性显式声明（如 partial）；tiling 对"超阈值 && 未声明"fail-loud（MBG 线实测事故：裸单节点静默算错，不写 y 只写 workspace）
9. **★ 绑定层 dtype fail-loud**——wrap/binding 必须 dtype-aware 或 fail-loud（ALN 线整网事故：fp16 residual 被按 fp32 静默重读，误差∝|x|+越界 507015）；验证电池必须含真实张量+离群通道+尾批

## ★ 性能形态三约束（MUST，产物形态要求而非优化建议）

owner 硬性要求（2026-10-10 起），优先级高于"正确性优先"的保守实现选择。三约束的判据与例外经 SDPA310p 实证线校准：

### 约束 1：核数打满（禁单核）

- **规则**：每个 kernel 的每次 launch，block_dim 必须 = 可用核数（310P3=8，以 stage 0 实证 CORE_NUM 为准）
- **分片方法（三件套）**：① 波次模型——units=ceil(units/cores) 波次，尾波缺额即空转，先算清波次账 ② 自然单元数 < 2×cores 或不能整除时，拆**内层最大维**凑满（选拆分维使 units ≡ 0 mod cores，或 units ≥ 2·cores 且尾 ≥ cores/2）③ **切点必须对齐**——拆分点钉在使每核输出区间整块 32B 对齐的边界上（防跨核共享 32B 块竞态）
- **豁免（均需文档显式声明）**：① **小形状 `min(tasks, cores)` 合法**——实在分不了满核的可以不分（owner 裁决 2026-10-12），禁止硬凑 per-core 不满一 tile 的碎活；报告注明 tasks<cores ② 输出元素数 < 一个 32B 量子（物理上无可切分）③ CV-precheck 回落路径按 `GetBlockNum()` 实测均分（blockDim=cvCoreNum<8 合法）
- **取证方式**：tiling 加 env 守卫的 debug 打印，每个验收 case 打印 path + 核数
- **多核化正确性警示**：多核流水深度是正确性约束不是性能调参（ALN 线实测：depth-3 轮转是唯一安全输入流水线，depth-2 无冷却期会错）；多 block 化不一定免费（MBG 线：MatmulImpl 多 block 结果错被弃用）——分片/流水改造后 MUST 重跑精度+确定性
- **证据**：stage 5 每 kernel 每 case 的 Block Num 记录

### 约束 2：全程向量化（禁标量化正确性兜底）

- **判据（频率制，实测校准）**：per-launch ≤7 元素 × 低频（每 launch 一次）的标量操作可忍（如张量尾）；**per-tile / per-iteration 的任何标量逐元素操作零容忍**——必须向量化或改布局
- **向量化替代模式（按优先级）**：① **布局先行**——重排/重打包存储使 tile 天然 32B 对齐（NZ/fractal 布局让搬运天然对齐）；**消拷贝优于快拷贝**（AsStrided 直读视图替代 contiguous 拷贝）② UB 攒批 + 整数个 8 元素 quanta 整块 MTE3 flush，标量只留 ≤7 元素尾 ③ Duplicate 填充代替写循环 ④ strided DataCopy block 模式
- **★ Gather 不是默认解**：310P 的 Gather 吞吐有场景依赖（实测反例：ALN 线 V 管道 Gather 比标量环慢 3.5×；SDPA 线 Brcb{2,16} 561ns vs Gather 978ns）——非连续访问的替代选型 MUST 三形态探针实测（标量环 / Gather / 直读重排）后定，禁止默认假设
- **★ 拷贝类 API 要输出侧对拍**：m200 的"看起来调用了实际没搬"不止 DataCopyPad（Copy API 也有静默 no-op 实证）——验证拷贝类替代必须用输出数据对拍，不能看调用成功
- **310P 特有警示**：标量环不只是慢——标量 GM 读混 DataCopy 会断 MTE3；标量写 UB 对 MTE3 不可见需 V 管道物化（Adds0/Or0）。标量环在热路径会引入正确性雷区
- **证据**：热循环零标量逐元素的静态检查（kernel 主循环 grep `GetValue`/`SetValue`；必需标量表读 hoist 到 prologue）

### 约束 3：能 Cube 绝不纯 Vector

- **规则**：含 **GEMM 语义**的计算 MUST 走 cube（310P 高阶 Matmul API）；Vector 只承担 cube 不能做的（搬运/elementwise/在线 softmax 类/epilogue）
- **适用边界**：本约束只适用于有 GEMM 语义的算子——**纯 element-wise 族（LN/RoPE 类）本来就不该 cube**，门禁不得误伤
- **算法形态先行**：cube 化有时要先改算法形态（如注意力在 310P 无 FlashAttention 二进制，走 dense bmm——MSDA 线实证）
- **L0C 长链警示**：cube 族在 310P 的 L0C 链长度有实测上限（MBG 线：长链 bare-Mmad per-launch 概率 fault；多 block 化不一定免费——MatmulImpl 多 block 结果错被弃用过）——长链 GEMM（如 conv im2col 大 K）要警惕同族问题，必要时分段+归约
- **fp32 契约的路线选型（owner 裁决 2026-10-12：eager-parity 优先级最高）**：
  ① **默认 = fp16-internal 直送（eager-parity）**——torch eager 的 fp32 matmul 在 310P 内部就降 fp16（实测 max err 2.3e-2），fp16-internal 与基线语义相同，不是妥协；精度门用 eager-parity 公式（ours vs fp64 误差 ≤ 2× eager vs fp64 误差），仍锚定绝对真值
  ② 契约**显式要求** fp32-exact（vs fp64 golden ≤1e-6 级）→ cube + hi/lo 拆分 3-Mmad（实测 5.35e-8；代价 3× cube 工作——cube 即使 3× 工作也碾压 VEC GEMM）。工程细节：固定项序保确定性（hi@hi+hi@lo+lo@hi）、CAST_NONE、每 case range precheck（maxAbs 全核同判，超域 pow2 prescale，非有限回落通用路径）
  ③ VEC 精确路径只作 inf/nan tile/odd-D 的 fallback，禁入热路径
- **契约冲突升级**：默认口径已是 eager-parity；仅当调用方显式要求 fp32-exact 且性能不达标时才构成冲突——遇到即升级 owner 裁决，**不是 kernel 层问题**
- **包络制豁免（唯一例外）**：定义 CV 包络（UB 容量/对齐约束），验收电池全部落包络内并取证 cvEligible=1；**包络外小形状（<1K 元素）允许纯 VEC 兜底**——cube 在那里反而慢 0.7~0.8×（GEMM warmup + L1 bounce 固定成本摊不薄，mfg-ops-vision 线实测）。除小形状包络豁免外，GEMM 场景没有"退 vector 更划算"（SDPA310p 线）
- **cube 侧工程纪律**：单静态 matmul 对象 + 每 tile IterateAll/End 成对（禁裸 Mmad 长链，E18）；raw-launch 路径每核冷启动做一次丢弃的 warmup GEMM（首 IterateAll 产全零，E18；managed/opp 路径不需要——边界见卡 runtime/dav_m200_mmad_long_chain_cold_start_warmup）；分组 GEMM 按组只换 GM 基址、共享 matmul 对象（coPerTile 取组内通道约数）
- **证据**：stage 5 math 类 kernel 的 mac_ratio 记录（参考画像，不设门槛）

### 三约束的门禁化

stage 5 报告含「性能形态合规」节（逐条给证据/豁免理由）；GATE-5 证据表含对应条目；阈值定值见 `stages/stage_5_performance.md`「形态阈值」。

## 路线专属「绝对不要做」（每条含替代方案）

| 禁止 | 应该做 |
|------|--------|
| **★ 加载 RegBase 资料（3510 路线 l2-guide/api-mapping/regbase sub-skill）** | **dav-m200 无 RegBase/MicroAPI——层级语义见本包决策树，cube 类按 `references/m200-cube-guide.md`** |
| **★ 在 dtype 矩阵/测试中保留 bf16** | **移除（硬件无）；业务必须时 host widen fp32（`references/m200-port-guide.md` §2.1）** |
| **★ 用 `dav-m200` 拼写直调 bisheng** | **CANN 9.0 只接受 `dav-2002`；标准路线用 `build.sh --soc=ascend310p`（本包 stage_3）** |
| **★ 用 `arch35/` 目录或 `__NPU_ARCH__==3510` 宏** | **用 `arch20/` + `__NPU_ARCH__ == 2002` + `AddConfig("ascend310p")`** |
| **★ 编译通过即声称"实现无误"** | **E1/E10 静默缺陷编译无感；精度用例是唯一暴露面（约束 3）** |
| **★ 沿用来源的 Fixpipe/DataCopyPad/CAST_RINT** | **按 stage_1 Step 1.3.2 处置卡替代（Fixpipe→L0C→UB→GM；DataCopyPad→DataCopy+手工 padding；CAST_RINT→CAST_NONE）** |
| **★ fp32 matmul 盲目上 hi/lo 拆分或直接 fp16 直算** | **按 §2.3 决策树结合门禁语义选型，报告中声明路线与预期精度档** |
| 跨核同步盘点套用 AIC↔AIV 分离核假设 | fuse 核无分离——注明"不适用"，重点转核内流水同步（E3/E6/E15） |
| **★ tile 数 < 核数就放任部分核空转** | **换维度拆/核内多 tile 合并/降 tile 粒度——block_dim 必须打满（性能形态约束 1）** |
| **★ 用标量逐元素环兜底正确性（E1/E14 替代路径）** | **向量化替代：layout 重排天然对齐 / UB 内 vector gather·shift / 对齐直通+尾段批量（性能形态约束 2）** |
| **★ 矩阵乘结构留纯 Vector 实现** | **上 cube（高阶 Matmul API；fp32 契约默认 hi/lo 3-Mmad）——vector 只做 cube 不能做的（性能形态约束 3）** |

## 实证记录

本路线包经 7_ConvStandard2d 端到端演练验证（Ascend310P3 + CANN 9.0.0，2026-10-10 ~ 10-15）：

- **精度**：32/32 PASS + 逐位确定性（eager-parity 门，CPU fp64 锚定三列证据）
- **性能**：quick 口径（ops-profiling，Task_Duration）0.029 → 0.0536；wall 0.154 → 0.268
- **探针判决**：硬件 img2col（LoadData3D img2col 形态）在 2002 不可用（全零/507014）；direct-conv vector 直卷在 dense case 死刑（25× 慢）；vendor 形态走 v350 conv 指令（AscendC 不可达）——三者在案，后续移植不必重试
- **关键纠错**：高阶 Matmul API 在 310P+CANN9.0 可用（matmul_intf.h 的 2201/3510 门只切 kfc 头，#else 分支给 2002 matmul.h）；E18 冷启动 warmup 触发条件 = raw-launch + Matmul 对象（managed/opp 路径不需要）
- 全部实证已沉淀为 cannbot-knowledge 卡（platforms:["310p"]，见「外部知识库」节）
