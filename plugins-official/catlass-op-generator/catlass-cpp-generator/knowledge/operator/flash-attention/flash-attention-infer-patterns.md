---
type: "operator"
title: "FA dense 前向 A2/A3 实现模式"
description: "dense flash attention 前向（QK^T→online softmax→PV，MHA/GQA/fused attention）的 BNSD 接口契约、AIC/AIV 分核、online softmax 状态递推、workspace 与同步协议、PAGED/尾块/mask 处理、aclnnFlashAttentionScore 精度标杆、性能 gate 口径与 FA0-FA13 检查表。"
tags: ["catlass-cpp", "flash-attention", "fa", "flash-attn", "mha", "gqa", "fused-attention", "atlas_a2_a3", "online-softmax", "bnsd"]
status: "stable"
generated: {"by": "process:catlass-cpp-fa-migration-curation", "at": "2026-10-22T00:00:00Z"}
verified: [{"by": "process:fa-source-cross-check", "at": "2026-10-22T00:00:00Z"}]
sources:
  - {"id": "fa-design-route", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-design/references/kernels/flash-attention.md", "title": "FlashAttention 类算子设计路由：场景定义、组件契约、shape 矩阵与陷阱", "kind": "repository"}
  - {"id": "fa-baseline-ref", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-design/references/kernels/flash-attention-npu-reference.md", "title": "FA 标杆接口参考：aclnnFlashAttentionScore/npu_fusion_attention 参数映射与继承边界", "kind": "repository"}
  - {"id": "fa-handcraft-manual", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-kernel-handcraft.md", "title": "FA kernel 手搓实现手册：四文件结构、任务切分算法、AIC/AIV 双入口主循环、同步编排、HardEvent 预置、BlockMmad/softmax epilogue 内部、host 直调模板、变体映射法与自查表", "kind": "repository"}
  - {"id": "fa-recipe-book", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-kernel-recipes.md", "title": "FA kernel 算子配方库：FFA 手搓实证与 §12.5.14 终态配方、s1s2 单核融合算法、S1 sink 流程与 kfc 单 TU 直调配方", "kind": "repository"}
  - {"id": "fa-compound-features", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-kernel-compound-features.md", "title": "FA 复合特性叠加册：器件级 API 墙、sink/varlen/q8/causal/paged 手搓步骤、测量与泛化坑", "kind": "repository"}
  - {"id": "fa-dev-patterns", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/flash-attention.md", "title": "FA 实现规则：kernel 设计经验、BNSD 接口、PAGED 方案、精度/性能归档与 FA0-FA13 检查表", "kind": "repository"}
  - {"id": "fa-stage-design", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/a2-a3-flash-attention-stage-design.md", "title": "A2/A3 FA Stage 设计经验：四段流水、workspace 协议、buffer 预算", "kind": "repository"}
  - {"id": "fa-handcraft-pitfalls", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-handcraft-pitfalls.md", "title": "FA 族手搓坑清单：架构级坑、多块 softmax 修复、多 batch bug、观测元坑", "kind": "repository"}
  - {"id": "fa-perf-gate", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-perf-gate-evaluation.md", "title": "FA 族性能评测与 gate 判定方法论：计时四陷阱与序列长分桶套件", "kind": "repository"}
  - {"id": "fa-precision-verify", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/precision-verification.md", "title": "精度验证补充：FA/MHA/GQA 双口径验证与 mixed tolerance 报告字段", "kind": "repository"}
  - {"id": "fa-shape-constraints", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/shape-constraints.md", "title": "shape 覆盖约束 Δ6：FA 类 shape 按算法维度生成", "kind": "repository"}
  - {"id": "fa-compilation", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/architecture/03-compilation.md", "title": "独立 kernel 直调构建：CATLASS_ARCH/--npu-arch/include/link 注入清单", "kind": "repository"}
  - {"id": "fa-architect-clauses", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:plugins-official/catlass-op-generator/agents/catlass-op-architect.md", "title": "设计 agent FA 条款：FA 类型识别、设计前置链与 catlass/examples 例外", "kind": "repository"}
  - {"id": "fa-precheck-clauses", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:plugins-official/catlass-op-generator/workflows/task-prompts.md", "title": "主体工作流 FA 预检条款：布局/D%16/SoC/baseline 确认", "kind": "repository"}
  - {"id": "fa-generator-clauses", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:plugins-official/catlass-op-generator/agents/catlass-op-generator.md", "title": "开发 agent FA 条款：内核来源锁定、BNSD host 转换、精度双口径", "kind": "repository"}
  - {"id": "fa-reviewer-clauses", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:plugins-official/catlass-op-generator/agents/catlass-op-reviewer.md", "title": "审查 agent FA 专项门禁条款", "kind": "repository"}
  - {"id": "fa-design-skill-route", "resource": "git:cann/cannbot-skills@585201016db01d5c9854402ca5086bd53d2ea46e:ops/catlass-op-design/SKILL.md", "title": "ops 设计 skill FA 检索路由：命中词表、When-to-Use 与索引表中 flash-attention.md 条目（L82-83/L101-102/L227）", "kind": "repository"}
  - {"id": "fa-dev-skill-route", "resource": "git:cann/cannbot-skills@585201016db01d5c9854402ca5086bd53d2ea46e:ops/catlass-op-develop/SKILL.md", "title": "ops 开发 skill FA 检索路由：FA 族形态全覆盖路由句、实现注意事项与性能 gate 路由（L80/83/86/107-108）", "kind": "repository"}
  - {"id": "fa-tune-skill-route", "resource": "git:cann/cannbot-skills@585201016db01d5c9854402ca5086bd53d2ea46e:ops/catlass-op-perf-tune/SKILL.md", "title": "ops 调优 skill FlashAttention Stage Tuning 整节：四段流水归因表与 A2 参考经验", "kind": "repository"}
  - {"id": "fa-profiling-faq", "resource": "git:cann/cannbot-skills@df779a092621865154e62b23f5fc34aed6c9b6fd:ops/ops-profiling/SKILL.md", "title": "ops-profiling !992 增量：高频问答速查（采集/对比/瓶颈判定）与阅读纪律（+17 行）", "kind": "repository"}
  - {"id": "fa-profiling-evals", "resource": "git:cann/cannbot-skills@df779a092621865154e62b23f5fc34aed6c9b6fd:ops/ops-profiling/evals/evals.json", "title": "ops-profiling !992 增量：evals 模型 token 配置（+20 行，参考性质）", "kind": "repository"}
  - {"id": "fa-profiling-script", "resource": "git:cann/cannbot-skills@df779a092621865154e62b23f5fc34aed6c9b6fd:ops/ops-profiling/scripts/msprof_perf_summary.py", "title": "ops-profiling !992 增量：_collect_compute_kernels/_launch_wall_min——launch wall 仅计 compute kernel 时间跨度的性能计量口径", "kind": "repository"}
operator_families: ["flash-attention"]
architectures: ["atlas_a2_a3"]
---

# 接口与概念

本 family 收录 **dense flash attention 前向**：`O = softmax(scale · Q·Kᵀ [+ mask]) · V`，KV 块全选、
无块稀疏 mask。覆盖 MHA（Hq==Hkv）、GQA（Hq>Nkv，Nq%Nkv==0）与 fused attention 形态。
块稀疏（Q 块 × KV 块 0/1 mask）属 `block_sparse_attention`；MLA（latent 压缩 + 共享 KV）属
sparse-flash-mla；反向（dO/LSE 输入出 dQ/dK/dV）属 flash-attention-grad。本族结论不得外推到这三族。

**命名约定**：FA 算子统一命名 `FA-<layout>[-<mode>]`，如 `FA-BNSD`（全注意力）、`FA-BNSD-Causal`
（下三角因果）、`FA-TND`（varlen）。设计/实现/验证全程同一算子名。[^fa-design-route]

**公开接口必须是严格 BNSD**：`Q=[B,H,Sq,D]`、`K/V=[B,H,Sk,D]`、`O=[B,H,Sq,D]`，全部连续
fp16。kernel 内部只消费一种布局，转换全部放 host，kernel 无运行时布局分支。[^fa-dev-patterns]

- `D % 16 == 0`（cube 友好；常用 64/128），`scale = 1/√D` 约定。
- GQA/MQA：query head `h` 使用 KV head `h // group`（`group = Nq/Nkv`，必须整除）。
- `actualQseqlen/actualKvseqlen` 是 **per-batch 原始长度**（int64 数组长度 B，非累加和）；
  dense BNSD 传 `[Sq]×B` / `[Skv]×B`。标杆 `npu_fusion_attention` 的 `actual_seq_*` 是**累加和**，
  两者约定不同，host 必须转换。[^fa-dev-patterns]

**精度标杆锁定**：`aclnnFlashAttentionScore`（CANN op_api 内置，Python 侧
`torch_npu.npu_fusion_attention(q, k, v, H, "BNSD", scale=scale)` 封装）。设计期冻结基准 contract
（`input_layout`、maskType、`scale`、seqlen 约定）；不需要 clone 外部仓库，也不要要求用户重复提供
参考路径。dropout/`pse`/`softmax_max/sum` 反向中间量/`inner_precise` 等标杆字段不在本族前向范围，
命中时按「用户 contract 优先」处理，不得静默继承。[^fa-baseline-ref]

预检清单（命中 FA/MHA/GQA/fused attention 时必做）：[^fa-precheck-clauses]

1. 输入布局确认：Q/K/V 是否 BNSD `[B,H,S,D]`；不满足则记录 host 需做的布局转换。
2. `D % 16 == 0` 校验、`scale = 1/√D` 约定确认。
3. 目标 SoC 确认：A2 必须走 `PAGED=true` + 恒等 block_table；950PR 需单独评估。
4. baseline 确认：冻结 `aclnnFlashAttentionScore` 作为精度标杆。

**FA 类型识别与设计前置链（主体 architect 条款，原样收录）**：[^fa-architect-clauses]

- **类型识别**：Step 1 结构化需求时，算子特征命中 FlashAttention / MHA / GQA / fused attention /
  QK^T-softmax-V，即归入 FA 类型，先执行 FA 设计路由，再进入具体 catlass 组件选型。
- **FA 专属前置（进入组件选型前必须完成）**：① 先走 design 路由（`/catlass-op-design` 的
  flash-attention 路由，即本册「设计路由全文」）；② 冻结 full-flow baseline
  （`aclnnFlashAttentionScore`）；③ FA 内核来源锁定（标准 FA 复用 FAInferKernel；FA 变体用高阶
  API）；④ 组件选型 BlockMmadQK/PV + EpilogueOnlineSoftmax/RescaleO；⑤ BNSD 接口与 host 布局
  转换方案（Q→BSND、K/V→块格式、O→BNSD）。
- **catlass/examples 例外**：Step 2 组件选型的通用规则是「在 `catlass/examples/` 找形态最接近的
  样例」；**FlashAttention 是例外，不走 example 对照**，直接用 FA kernel 设计知识。

## 场景路由与检索入口（ops skill FA 命中词与路由，原文）

以下为三个 ops skill（catlass-op-design / catlass-op-develop / catlass-op-perf-tune）`SKILL.md` 中
FA 相关检索/路由入口段落的原文（正文原样，仅链接已内联化）。这些是「检索/路由入口」性质的内容：
使用者按命中词进入对应参考文件，本 concept 已收录这些参考文件的全量正文，路由入口本身保留以
维持检索语义完整。

**FA 命中词表**（来源：catlass-op-design SKILL.md 路由树 + catlass-op-develop SKILL.md 全形态路由句）：
FlashAttention / MHA / GQA / fused attention / MLA / latent / paged attention / QK^T-softmax-V，
以及 FA 族形态枚举：标准 FA / BNSD / Causal / TND-varlen / paged / FFA / sink / 复合特性（q8 叠加、
causal/paged 叠加）、fa-sageattention（Ascend950/dav-3510 量化注意力）。

**catlass-op-design SKILL.md「When to Use Each Source」FA 条目（L82-83 原文）**：[^fa-design-skill-route]

- **FlashAttention / MHA / GQA / MLA / latent / paged attention 类算子设计路由 → [references/kernels/flash-attention.md（见本册「设计路由全文」节）**
- **A2/A3 FlashAttention stage 分层、workspace/flag、online softmax 状态 → [../catlass-op-develop/references/patterns/a2-a3-flash-attention-stage-design.md（见本册「A2/A3 Stage 设计经验全文」节）**

**catlass-op-design SKILL.md 场景路由树 FA 分支（L101-102 原文）**：[^fa-design-skill-route]

```
├── Attention / State Recurrence
│   ├── 命中 FlashAttention / MHA / GQA / fused attention / MLA / latent / paged attention / QK^T-softmax-V
│   │   → 读取 [kernels/flash-attention.md（见本册「设计路由全文」节）
```

（同树中未命中 FA 命中词的 linear attention / GDN / KDA / retention / RWKV / state recurrence 分支
路由到 linear-attention.md，不属本族。）

**catlass-op-design SKILL.md 索引表 flash-attention.md 条目（L227 原文）**：[^fa-design-skill-route]

| 参考文件 | 条目描述 |
|---|---|
| [kernels/flash-attention.md（见本册「设计路由全文」节） | **FlashAttention（MHA/GQA）设计路由**：full-flow 判定与 baseline（aclnnFA）冻结、BlockMmadQK/PV + OnlineSoftmax/RescaleO 组件选型、AIC/AIV 协作、BNSD 接口转换、shape 覆盖矩阵、mixed tolerance 与性能基准、PAGED 平台陷阱 |

**catlass-op-develop SKILL.md「When to Use Each Source」FA 条目（L80/83/86 原文）**：[^fa-dev-skill-route]

- **FlashAttention / MHA / GQA 类实现注意事项 → [patterns/flash-attention.md（见本册「设计路由全文」节）**（BNSD 接口、PAGED 方案、host 布局转换、aclnn 精度对比）
- **任何 FA 族形态（标准 FA / BNSD / Causal / TND-varlen / paged / FFA / sink / 复合特性）→ 一律按 [patterns/fa-kernel-handcraft.md（见infer 册「手搓手册全文」节） 手搓单 kernel**：**标准 FA/FA-Paged/GQA/causal（CrossCoreFlag 组件路径）优先按 [patterns/fa-paged-handcraft-recipe.md（见本册「FA-Paged 手搓完整配方全文」节） 逐行写**（四文件端到端：kernel Step 1-7 + mask Step 8 + common/tiling/host Step 9-11，NO_MASK 闭卷实证全 PASS；手册 §1–9 是其机制解读层）；kfc 单 TU 路径走 **[fa-kernel-recipes.md（见infer 册「配方集全文」节） §12.13 终态配方**（fa/fa-sink，geomean 0.693 vs 标杆，序列长至 458752，含 F0 十一步逐步生成流程），[fa-kernel-compound-features.md（见infer 册「复合特性叠加册全文」节） §12.12 = TND-varlen/q8 叠加、**§12.14 = causal/paged 逐步叠加**，配方册 §12.5.14 = FFA 完整配方——**全程只用本 skill 文件**
- **FA 族性能评测 / gate 判定 / 泛化套件 → [patterns/fa-perf-gate-evaluation.md（见infer 册「性能评测与 gate 判定全文」节）**（计时四陷阱、gate=geomean device 口径、序列长分桶、B>1 探 bug）

**catlass-op-develop SKILL.md 索引表 FA 条目（L107-108 原文）**：[^fa-dev-skill-route]

| 参考文件 | 条目描述 |
|---|---|
| [patterns/flash-attention.md（见本册「设计路由全文」节） | FlashAttention（MHA/GQA）实现注意事项：BNSD 公开接口 + host 布局转换、A2 PAGED=true+恒等 block_table、尾块填充、AIC/AIV 协作、dump O + aclnn 标杆精度对比、FA1–FA11 检查表 |
| [patterns/a2-a3-flash-attention-stage-design.md（见本册「A2/A3 Stage 设计经验全文」节） | A2/A3 FlashAttention stage 设计细则：C1→V1→C2→V2 四段流水、online softmax 状态递推、GM workspace/CrossCoreFlag、尾块处理、AIC/AIV 流水重叠 |

## 算子算法

FA-2 在线 softmax（不物化完整 `[Sq,Sk]` 分数矩阵），每个 KV 块 `j` 四段流水：

```text
C1 (AIC)  S_j = scale · Q · K_jᵀ                → L0C → Fixpipe → UB
V1 (AIV)  m_new = max(m, rowmax(S_j))
          P_j   = exp(S_j − m_new)
          l_new = l·exp(m − m_new) + rowsum(P_j)
C2 (AIC)  O_j   = P_j · V_j                     → L0C → Fixpipe → UB
V2 (AIV)  O_acc = O_acc·exp(m − m_new) + O_j
末块      O     = O_acc / l
```

要点：`m`（running max）、`l`（running sum）、`O_acc` 跨 KV 块递推，不可中断或重置；rescale 因子
`exp(m_old − m_new)` 对 `O_acc` 与 `l` 都要乘；除以 `l` 后移到全部 KV 块处理完之后。**禁止**重排
C1→V1→C2→V2 顺序，或把跨块递推状态当简单 buffer 复用。[^fa-stage-design]

mask 模式（内核 `MaskType`）：[^fa-dev-patterns]

| 值 | 名称 | 含义 |
|---|------|------|
| 0 | `NO_MASK` | 全注意力 |
| 1 | `MASK_SPEC` | mask 张量逐位屏蔽（`ApplyMask = score += mask·(−3e38)`，mask≠0 屏蔽、0 保留，与 npu `atten_mask` 的 1=不参与 一致） |
| 2 | `MASK_CAUSAL` | 下三角 causal：结构性跳过对角块以上整 KV 块（省约一半计算），仅对角块读压缩 mask；勿自实现 causal |

npu 标杆 `sparse_mode` → `maskType`：不传 mask→`NO_MASK`；allMask/任意逐位 mask→`MASK_SPEC`；
leftUp/rightDownCausal（下三角）→`MASK_CAUSAL`；band/prefix 等由 host 预生成 `atten_mask` 后走
`MASK_SPEC`。标杆 causal 的 `atten_mask` 必须是压缩下三角 `[2048,2048]`；自研内核的压缩掩码张量是
固定 `[1024,1024]` fp16（`mask[q][k]=1 当 k>q`，host 一次预置常驻复用）——同约定不同尺寸，勿混用。[^fa-baseline-ref]

## 设计路由全文：FlashAttention 类算子（catlass-op-design）
> 以下为设计路由源文件全量正文（Step 0-5、常见陷阱与输出设计章节补充），供设计期逐条对照。

> 本指南覆盖 FlashAttention（FA-2）前向算子的 catlass 设计入口，包括 MHA / GQA 两种 head 形态与 MLA（multi-head latent，KV latent 压缩 + 分页）变体。它不替代普通 Matmul、Grouped Matmul 路由；只有需求命中 attention 融合特征时才启用。

---

### 场景定义

FlashAttention 类算子的共同特点是：Q/K/V 三输入，计算图为 **Q·K^T → softmax → ·V** 的两段矩阵乘 + 中间 softmax 的融合，且为了长 KV 序列采用**分块 + online softmax（FA-2）**，避免物化完整 `[Sq, Sk]` 分数矩阵。典型 I/O 维度是 `(B, H, Sq/Sk, D)`（BNSD）。

触发信号包括：`flash attention`、`fused attention`、`attention`、`scaled_dot_product_attention`、`QK^T softmax V`、`MHA`、`GQA`、`multi-head attention`、`MLA`、`latent attention`、`paged attention`、`page attention`。

> **命名约定（强制）**：本 skill 生成的 FA 算子统一命名为 `FA-<layout>[-<mode>]`，如 `FA-BNSD`（BNSD 全注意力）、`FA-BNSD-Causal`（BNSD 下三角 causal）、`FA-TND`（TND varlen）。设计 / 实现 / 验证全程使用同一算子名。

> **FA kernel 设计知识（本 skill 内化，不依赖外部样例）**：
> - **标准 FA / FA-Paged / GQA / Causal（组件路径）按 fa-paged-handcraft-recipe.md 逐行写出**（四文件端到端：kernel Step 1-7 + mask Step 8 + common/tiling/host Step 9-11；NO_MASK 闭卷实证 kv=128~204800 全 PASS，max_abs≤7.6e-06）；机制解读层为 fa-kernel-handcraft.md §1–9；备选 kfc 单 TU 直调路径见同配方册 §12.13
> - **fa-sink（S1 模板）/ fa-varlen（TND）/ FFA 双 KV** 按 逐行配方生成：sink=fa-sink-handcraft-recipe.md（S1 模板逐步骤，sink=SoftmaxFlashV2 isUpdate 播种）、varlen=fa-varlen-handcraft-recipe.md（varlen kernel 逐步骤）、FFA=配方册 §12.5.14 终态配方 + ffa-handcraft-recipe.md（算子结构契约）；其余 FA 变体（FlashSoftmax 分片等）用 AscendC 高阶 API 按 FA kernel 设计经验 §0.2 原则自行开发 kernel
> - **kfc 单 TU 直调（`__mix__(1,2)` + `REGIST_MATMUL_OBJ` + userWs 独立参数）的 fa/fa-sink 完整配方见配方册 §12.13**（第三轮纯 skill 实证 geomean 0.693 vs npu_fusion_attention(sink=)，序列长至 458752）——含 +4GB 窗口坑/运行时 transB/每 chunk 单 IterateAll/SoftmaxFlashV2 正确用法/原子累加竞态/尾块 pad+mask 全套事实
> - **MLA 手搓路径（已实测可一次跑通，配方见 fa-mla-handcraft-recipe.md）**：内核形态 = 通用 MLA（H=16/32/64）+ H=128 特化（TP1）+ AMLA 特化，dispatch 固化 `MmadAtlasA2MLAQK`/`MmadAtlasA2MLAPV`；语义 `O=softmax(scale·(Q·c_kv+q_rope·k_rope))·V_latent`，`scale=1/√576`、`Dc=512/Dr=64/blockSize=128/V=latent`、`q∈[1,4]`、`kv≥128`（接口 gen_data 声明 ≤16384 为验证包络，实测可至 20万+）。host 契约见 patterns/fa-mla-paged.md §1-4。**★手搓必读 recipe 的两条纠错**：①AIC/AIV 分流用 `#ifdef __DAV_CUBE__`/`#ifdef __DAV_VEC__`（非 `__mix__`+g_coreType）；②主循环 prologue+延迟 PV（非同块锁步），否则 CrossCoreFlag token 不平衡→死锁
>
> FA kernel 设计经验（AIC/AIV 分工、在线 softmax 状态、workspace 槽位、同步协议、Tile 策略）已内化到 本册「FA 实现规则全文」 §0.2，不依赖外部样例存在。

---

### Step 0: 自动锁定标杆参考

命中本路由时，先读取 flash-attention-npu-reference.md。该文件锁定 FA 标杆接口（`aclnnFlashAttentionScore` / `torch_npu.npu_fusion_attention`），维护「标杆参数 → FA 内核参数」映射（layout / scale / `sparse_mode`↔`maskType` / seqlen 累加↔per-batch）、可继承/不可继承边界与生成契约。**不要要求用户在 prompt 中重复提供参考路径**；标杆是 CANN/torch_npu 内置 API（有文档），无需 clone 外部仓库。

DESIGN.md 的 baseline/reference 章节必须写明：标杆来源（`aclnnFA` / `npu_fusion_attention`）、本次冻结的 contract（`input_layout`、`maskType`、`scale`、seqlen 约定），以及哪些语义来自标杆、哪些来自本 skill kernel 设计知识。

---

### Step 1: 冻结目标形态与同语义 baseline

```
目标是什么？
├── full-flow fused op（推荐）
│   ├── 必须冻结同语义 full-flow baseline
│   └── 首选 baseline: aclnnFlashAttentionScore（CANN 内置，见 op_api 文档）
└── stage operator（C1/QK^T 或 C2/PV 单独交付）
    ├── 必须说明上下游 stage 输入/输出契约
    └── golden 可用 stage-aware reference，但上游中间量必须先被 full-flow 独立验证
```

**禁止**把局部 helper、临时 NumPy 片段当作 full-flow baseline。baseline 的语义、dtype、layout、scale 规则必须与交付目标一致（`scale = 1/sqrt(D)`，`O = softmax(scale * QK^T) · V`）。

> 与 FA-2 数学模型的对应：`C1: S = Q·K^T`、`V1: P = exp(S − m)`（online softmax）、`C2: O_acc += P·V`、`V2: O = O_acc / l`。详见 `ascendc-tiling-design/references/flashattention/foundation/fundamentals.md`（skill 仓内方法论，不变量 I1–I5）。

---

### Step 2: FA kernel 策略与组件契约

FA 是 AIC/AIV 跨核协作的混合 kernel。**标准 FA（Q·Kᵀ→softmax→·V，单 KV）按 fa-paged-handcraft-recipe.md 逐行写出**（四文件端到端逐步骤，NO_MASK 闭卷实证全 PASS；机制解读 = 手册 fa-kernel-handcraft.md §1-9；备选 §12.13 kfc 单 TU 直调）。**MLA 按 fa-mla-handcraft-recipe.md 手搓**（MLAKernel 逐行配方，`#ifdef __DAV_CUBE__/__DAV_VEC__` 分流 + prologue+延迟 PV，已实证一次跑通）。**其余 FA 变体（FlashSoftmax 分片等）用 AscendC 高阶 API（MatMul/SoftmaxFlashV2）按 FA kernel 设计经验 §0.2 原则自行开发**——高阶 API 用 Event 同步（非 CrossCoreFlag 黑盒），错误可观测可调试。

> **★实现策略（最高优先级）**：FA 族一律按手搓手册/recipe 写 kernel（§1–9 / §12.13），host 侧按配方自写。
> **FA/MLA 用 `BlockMmad + Epilogue + CrossCoreFlag` 从零拼装是可行的**（paged/MLA 两条 recipe 即此结构），
> **前提**：①分流用 `#ifdef __DAV_CUBE__/__DAV_VEC__`（MLA）或 `operator()<AIC>/<AIV>` 特化（FA），
> ②主循环 prologue+延迟 PV（非同块锁步），③HardEvent 预置/drain 按 §8.6 全表。
> 缺任一即 CrossCoreFlag token 不平衡→死锁（本会话实测：`__mix__`+g_coreType+锁步→必死锁）。
> 详见 本册「FA 实现规则全文」 §0 与 `patterns/fa-mla-handcraft-recipe.md`。

标准 FA 内核固化的组件（仅供理解 FAInferKernel 的设计原理，FA 变体开发参考 FA kernel 设计经验 §0.2）：

| 组件 | 取值 | 说明 |
|------|------|------|
| ArchTag | `Arch::AtlasA2`（A2/A3） | 由目标 SoC 定 |
| BlockMmadQK | `MmadAtlasA2FAIQK` | C1：Q·K^T，K 按 ColumnMajor 消费（host 不预转置）|
| BlockMmadPV | `MmadAtlasA2FAIPV` | C2：P·V |
| BlockMmadQKTail/PVTail | `MmadAtlasA2FAITailQK` / `MmadAtlasA2FAITailPV` | KV 非 128 对齐时的 tail 处理 |
| EpilogueOnlineSoftmax | `EpilogueAtlasA2OnlineSoftmax` | V1：online softmax + running max/sum |
| EpilogueRescaleO | `EpilogueAtlasA2RescaleO` | V2：按 `exp(m_old − m_new)` rescale O |
| Kernel | `FAInferKernel<...>`（AIC/AIV 双特化，复用 FAInferKernel） | `operator()<AIC>` + `operator()<AIV>` |
| TileShape | `L1=L0=GemmShape<128,128,128>`（`L1TileShape::K==D`） | 内核固定，性能再迭代 |
| PAGED_CACHE_FLAG | `true`（A2） | 恒等 block_table 等价非 Paged |

> **组件名核验**：不同 catlass 版本组件名可能不同。设计/实现前 grep 核验真实组件名与是否有特化体：
> ```bash
> grep -rn "MmadAtlasA2.*QK\|MmadAtlasA2.*PV" catlass/include/catlass/gemm/dispatch_policy.hpp
> grep -rn "struct BlockEpilogue<EpilogueAtlasA2.*RescaleO" catlass/include/
> ```

**数据流**：`QK^T → L0C → Fixpipe → UB(S) → EpilogueOnlineSoftmax(P, m, l) → P·V → L0C → Fixpipe → UB(O_acc) → EpilogueRescaleO → O`。AIC 出 S/O，AIV 做 softmax/rescale，AIC↔AIV 用 CrossCoreFlag 握手。

---

### Step 3: workspace 与 AIC/AIV 协作

- 中间量 `S`/`P`/`O_tmp`/`O_update` 用**独立 GM workspace 段**（3 级轮转），用 GM buffer 指针透传，**禁 `GetUserWorkspace`（直调路径会丢弃入参返回 kfc 地址）**。
- AIC/AIV 跨核同步：A2 支持 `CrossCoreSetFlag/WaitFlag` modeId 0/1/2（950PR 另有 4）。Set 与 Wait 两侧 modeId 必须一致，否则死锁。
- 对 BNSD 接口：Q 经 host 转 BSND `[B,Sq,H,D]`，K/V 转块格式 `[numBlocks,128,H,D]` 后进 kernel；O 由 BSND 转回 BNSD。布局转换放 host，kernel 只消费一种内部布局。
- 详细 stage 设计见 `本册「A2/A3 Stage 设计经验全文」`。

---

### Step 4: Shape 覆盖矩阵

FlashAttention 的测试 shape 按算法维度构造，不按随手枚举：

| 类别 | 构造要点 |
|------|----------|
| B/Sq/Sk 组合 | 单 batch、多 batch；`Sq == Sk`、`Sq < Sk`、`Sk` 非 128 倍数（尾块）|
| H / head 形态 | MHA（Hq==Hkv）、GQA（Hq > Hkv）；单头/多头 |
| D（headSize） | 64 / 128 等 cube 友好值，必须 `D % 16 == 0` |
| 对齐边界 | `Sk` 恰为 128 倍数 / 非倍数；尾块 0 填充路径 |
| 数值边界 | 全零 Q/K/V、近零 softmax 输出、极值 scale |

shape 来源写进 DESIGN/PLAN：用户实网规模、baseline 支持范围、cube 整数倍、边界类别。禁止硬编码无说明的 shape tuple。

---

### Step 5: 精度与性能基准在设计阶段冻结

#### 精度
- 标准优先 `ops-precision-standard`，FA 浮点输出按 mixed tolerance：`abs(actual − golden) <= atol + rtol * abs(golden)`，再看 `matched_ratio` / `max_abs`。
- golden 首选 **`aclnnFlashAttentionScore`**（标杆接口）实测输出；次选独立 CPU 参考（numpy/torch fp32 累加）。
- 报告每 case 输出 `shape`、`dtype`、`atol`、`rtol`、`matched_ratio`、`max_abs`、`pass/fail`。

#### 性能
| 优先级 | 基准 |
|:---:|------|
| 1 | 同语义 baseline（aclnnFA）实测 Task Duration |
| 2 | 本 skill 整经验证内核同 shape 实测 |
| 3 | Cube/MTE/Vector 理论上限 |

性能报告至少含：custom/baseline 时长、speedup、launch count、主导流水（Cube/MTE/Vector/同步等待）、cube_util、workspace peak、profiler 路径。

---

### 常见陷阱

| 陷阱 | 后果 | 正确做法 |
|------|------|----------|
| ① **A2 上 `PAGED_CACHE_FLAG=false`（真非 Paged）路径触发 aicore exception** | 507015 运行崩溃 | A2 上固定 `PAGED=true` + **恒等 block_table**（语义等价非 Paged BNSD 连续 KV 块）；不要尝试修 Paged 路径 |
| ② 接口写成 BSND/块格式而不是 BNSD | 与用户 BNSD 期望不符 | host 做 BNSD→内部布局转换（Q→BSND、K/V→块格式、O→BNSD），公开接口保持 BNSD |
| ③ KV 非 128 对齐不做尾块填充 | 越界/错误结果 | 尾块 0 填充；bin 读入尺寸（BNSD）与 device 尺寸（块格式）分开 |
| ④ 直调路径用 `GetUserWorkspace` | 拿到 kfc 地址越界 | 用独立 GM buffer 指针透传 |
| ⑤ 未冻结 baseline 就开写 | 精度无从对标 | 设计期先冻结 `aclnnFlashAttentionScore` 语义 |
| ⑥ 把中间 S 物化到 HBM | 失去 FA 分块意义 | 走 L0C→Fixpipe→UB，保持分块 online softmax |

---

### 输出设计章节补充

命中本路由时，DESIGN.md 除 catlass 通用选型表外，必须额外包含：
1. full-flow 判定与 baseline 路径（aclnnFA 或 CPU 参考）
2. 组件选型表（BlockMmadQK/PV、EpilogueOnlineSoftmax/RescaleO、TileShape、PAGED 方案）
3. 数据流与 workspace（3 级轮转、AIC/AIV 同步）
4. BNSD 接口与 host 布局转换方案
5. shape 覆盖矩阵与每类 shape 依据
6. mixed tolerance 精度标准与 golden 来源
7. 性能对标口径

## 标杆接口参考全文：flash-attention-npu-reference（catlass-op-design）
> 以下为标杆对齐源文件全量正文（标杆定位、参数映射、sparse_mode→maskType、可继承边界与用户 contract 优先规则）。

本文件用于 FA 类 Catlass 算子设计/实现前的**标杆对齐**。命中 FA 路由（见 flash-attention.md Step 0）时，先读本文锁定标杆接口与「标杆参数 → FA 内核参数」映射，**不要要求用户在 prompt 中重复提供参考路径**。

### 标杆接口定位

| 标杆 | 位置 | 用途 |
|---|---|---|
| `aclnnFlashAttentionScore` | CANN `op_api` 文档（ACL 单算子） | 底层标杆：语义 / dtype / layout / scale / mask 权威来源 |
| `torch_npu.npu_fusion_attention` | torch_npu op-plugin（封装 `aclnnFlashAttentionScore`） | Python 标杆：可直接 NPU 实测逐元素对比 |
| FA kernel 设计知识 | 本 skill `develop/patterns/flash-attention.md` §0.2（AIC/AIV 分工 + 在线 softmax + workspace + 同步协议） | FA kernel 开发指导（标准 FA 复用 FAInferKernel；FA 变体按原则自研）|

> 本 skill 前向只对齐 `attention_out = softmax(scale·Q·Kᵀ + mask)·V`。训练侧 dropout / `softmax_max`/`softmax_sum` 反向中间量 / `pse` / `sink` / `inner_precise` 等 `npu_fusion_attention` 字段**不在本 skill 前向生成范围**，命中时按「用户 contract 优先」处理（见下）。

### 标杆参数 → FA 内核参数映射

| 标杆参数 | FA 内核处理 |
|---|---|
| `input_layout`（`BNSD`/`TND`/`BSH`/`SBH`/`BSND`） | 内核内部固定：Q=`BSND [B,Sq,H,D]`，K/V=块格式 `[numBlocks,128,H,D]`。**host 做转换**，kernel 不做运行时分支（TND 下 Q/O 可直透传，仅 K/V 需转 paged） |
| `head_num=Nq`、`Nq%Nkv==0`（MHA/GQA） | 内核 `numHeads=Nq`、`kvHeads=Nkv`；query head `h` 用 kv head `h//group`（`group=Nq/Nkv`） |
| `scale`（默认 `1/√D`） | `FATilingData.scaleValue`。内核 `FAInferTiling` 默认按 `1/√D` 自算；若标杆传**自定义 scale**，host 须覆盖 `scaleValue` |
| `atten_mask` + `sparse_mode` | → 内核 `maskType`（见下表） |
| `actual_seq_qlen`/`actual_seq_kvlen`（npu 为**累加**和） | 内核 `actualQseqlen`/`actualKvseqlen` 为 **per-batch 原始长度**（host 做累加→原始的差分；dense 等长直接 `[S]×B`）。**两者约定不同，必须转换** |
| `keep_prob`/`dropout`（前向） | 本 skill 前向 `keep_prob=1`（无 dropout） |
| 输出 `attention_out` | 内核 `O`（BSND，host 转回公开 layout） |

`sparse_mode` → `maskType` 映射（重点）：

| npu `sparse_mode` | 含义 | 内核 `maskType` |
|---|---|---|
| 不传 `atten_mask` / `0`(defaultMask) 无掩码 | 全注意力 | `NO_MASK`(0) |
| `1`(allMask) / 任意 `atten_mask` 逐位 | 逐位掩码 | `MASK_SPEC`(1) |
| `2`(leftUpCausal) / `3`(rightDownCausal) — **下三角 causal**（Sq==Skv 时一致） | 因果/下三角 | **`MASK_CAUSUAL`(2)**（内核结构性跳过对角块以上整块 + 仅对角块读 mask，勿自实现 causal） |
| `4`(band) / `5-8`(prefix/varlen外切) | 由 host 按 `pre_tockens`/`next_tockens` 预生成 `atten_mask` 后走 `MASK_SPEC` | `MASK_SPEC`(1) |

mask 张量语义：内核 epilogue `ApplyMask = score += mask·(−3e38)` → **mask≠0 屏蔽、0 保留**（与 npu `atten_mask` 的 `1=不参与` 一致）。具体张量形状/stride/取值由内核 `LayoutMask` 与对角块索引决定，**以 本册「FA 实现规则全文」 §0.2.8 与 `a2-a3-flash-attention-stage-design.md` 为准**。

> npu 标杆侧：`npu_fusion_attention` causal（sparse_mode 2/3）的 `atten_mask` 必须是压缩下三角 `[2048,2048]`（否则 tiling 报 `161001 "set atten_mask_shape to [2048,2048]"`）。
> 自研 kernel 侧的 FAInfer 级内核（手搓手册 §1–9 骨架）causal 掩码张量 = **固定 `[1024,1024]` fp16**（`mask[q][k]=1 当 k>q`，host 一次预置常驻复用）——与标杆的 [2048,2048] 同约定不同尺寸，勿混用。

### 可继承 vs 不能直接继承

| 来源 | 可继承 | 不能直接继承 |
|---|---|---|
| `aclnnFA`/`npu_fusion_attention` | 数学语义、`scale=1/√D`、mask 语义（`atten_mask 1=不参与`）、layout、head 映射、baseline 对齐口径 | 把 npu 的**累加** `actual_seq` 当内核 seqlen；把 dropout/`pse`/`sink`/`inner_precise` 等**本 skill 未支持**字段静默继承 |
| 本 skill §0.2 kernel 设计知识 | AIC/AIV 分工、在线 softmax 状态、workspace 槽位、同步协议、Tile 策略（FA 变体开发基础） | 标准 FA 手搓 catlass Block 组件（CrossCoreFlag 黑盒必死锁）|

### 生成契约：用户 prompt 优先

FA 同族算子常存在多个相近但不等价的 contract（`scale` 作用位置、mask 模式与对齐、seqlen 累加/原始、输出 dtype、是否产出反向中间量）。生成时遵循：

- 用户 prompt 给的**数学公式与接口语义优先于标杆实现**。
- 标杆只用于对齐语义 / layout / mask / scale / baseline 口径；不能把标杆里本 skill 不支持的 contract 静默继承到目标算子。
- DESIGN.md 必须冻结本次 contract（`input_layout`、`maskType`、`scale`、seqlen 约定），并在 golden、verify、README、精度/性能报告中保持同一语义。
- 若标杆、本 skill 旧算子、用户公式三者不一致，**不自行择优**：在 DESIGN.md 记录差异并以用户公式为准，除非用户明确改口。

# 用法

## 分核策略与基本块切分

FA kernel 是 AIC（Cube）+ AIV（Vector）跨核协作的混合 kernel：

- AIC：两个 GEMM（Q·Kᵀ 与 P·V），L0C 累加 → Fixpipe → 写 GM workspace。
- AIV：online softmax（max/sum/exp/P）与 O rescale，从 GM workspace 读 AIC 结果，跨 KV 块维护
  `m/l/O_acc`。AIC↔AIV 用 CrossCoreFlag 握手。[^fa-dev-patterns]

任务切分单位是 `(batch, qSBlockIdx, qNBlockIdx)`：[^fa-dev-patterns]

```text
qSBlockSize = 128（固定，Sq 按 128 分块，尾块取余）
qNBlockTile = min(max(⌊128/qSeqlen⌋/2*2, 1), groupSize)   // N 维分块，保证偶数为 AIV 双子核负载均衡
rowNum = qSBlockSize × qNBlockSize                        // N 维 pack 进 GEMM 的 M 维
任务总数 = Σ_b qNBlockNumPerGroup × kvHeads × qSBlockNum
```

N 维 pack 进 M 维是 FA 能用普通 GEMM 组件的关键技巧：Q 按 `[qSBlockSize, qNBlockSize, D]` 当一个
`rowNum×D` 矩阵做 GEMM，O 写回时按 qNBlockSize 分头拆回。GQA 时 K/V 平面用 `kvHeadIdx`、Q/O 平面
用 `qHeadIdx = kvHeadIdx·group + 组内偏移` 索引，两套偏移勿混。

Tile 参数（FAInferKernel 固化）：`L1=L0=GemmShape<128,128,128>`（K 维必须=D，D≤128）；Q 分块 128；
KV 分块（blockSize）=128，与 PAGED 块大小一致；`preLaunch=2`（AIC 领先 AIV 2 个 KV 块）。[^fa-dev-patterns]

shape 覆盖矩阵按算法维度构造，禁止随手枚举无说明的 shape tuple：[^fa-shape-constraints]

| 类别 | 构造要点 |
|------|----------|
| B/Sq/Sk | 单/多 batch；`Sq==Sk`、`Sq<Sk`、`Skv` 非 128 倍数（尾块） |
| H 形态 | MHA（Hq==Hkv）、GQA（Hq>Hkv）；单头/多头 |
| D | 64/128 等 cube 友好值，必须 `D%16==0` |
| 对齐边界 | `Skv` 恰为/非 128 倍数；尾块 0 填充路径 |
| 数值边界 | 全零 Q/K/V、近零 softmax 输出、极值 scale |
| 对标 shape | 与 `aclnnFlashAttentionScore` 完全同 shape/layout/dtype |

## 数据路径与存储层级

内部布局（标准 FAInferKernel）：Q 取 BSND `[B,Sq,H,D]`（`strideQO=H*D`）；K/V 取块格式
`[numBlocks,128,H,D]`；O 出 BSND。host 转换：Q BNSD→BSND（三重循环转置）；K/V BNSD→块格式
（按 128 块重排，尾块 0 填充）；O BSND→BNSD。**bin 读入尺寸与 device 尺寸分离**：BNSD K 的 bin 是
`B*H*Sk*D*2`，块格式是 `numBlocks*128*H*D*2`。golden 用原始 BNSD K/V 计算，不受块格式影响。[^fa-dev-patterns]

片上预算（A2/910B3）：[^fa-stage-design]

| Region | 上限 | FAInferKernel 装什么 |
|---|---|---|
| UB | 192 KB | AIV：m/l、S（fp32 ping-pong）、P（fp16 ping-pong）、O_acc rescale、causal mask |
| L1 | 512 KB | AIC：Q/K/V/P tile（128×128×128，K 维=D≤128）、ping-pong 操作数 |
| L0A/L0B | 64 KB 各 | Q tile（PV 阶段复用为 P）/ K tile（复用为 V） |
| L0C | 256 KB | QK 的 fp32 累加器（PV 写 (M,D) 子区） |

任一 region 超限都是 runtime 设备错误（编译干净）。FA 最常踩 UB：S/P ping-pong + m/l + mask 同时
驻留，rowNum/stackSeqTile 一放大就超 192KB。预算紧时压缩顺序：① 减 `stackSeqTile`；② 减 `rowNum`；
③ aliasing 生命周期不重叠的 UB region；不要指望 cast 被编译器折叠省 UB。

GM workspace 4 段独立 buffer，指针透传，**禁 `GetUserWorkspace`/`SetSysWorkspaceForce`**（直调路径
丢入参返回 kfc 地址致 MTE 越界）：[^fa-stage-design]

| 段 | 用途 | A2 尺寸（blockDim=aicCoreNum） |
|---|---|---|
| s | QK 输出 S（AIC→AIV） | blockDim × 131072 × 12 B |
| p | softmax 输出 P（AIV→AIC） | blockDim × 131072 × 6 B |
| oTemp | PV 输出 O_block（AIC→AIV） | blockDim × 131072 × 12 B |
| oUpdate | rescale O 更新量 | blockDim × 131072 × 12 B |

3 级轮转（`preLaunch=2`）：每核 `coreIdx × WORKSPACE_BLOCK_SIZE_DB × (preLaunch+1)` 定址，
`WORKSPACE_BLOCK_SIZE_DB=131072`。跨 stage 边规则：同一逻辑块在所有 stage 用同一 slot 公式；
producer 完整写完才 set ready；consumer 在内层循环入口前 wait，不在 row/tile 循环内反复握手；
`O_acc` 初值 0、`m` 初值 −inf 的初始化路径明确。

## 实现顺序

1. 预检（布局/D/scale/SoC/baseline）→ 冻结基准 contract 写入 DESIGN.md。
2. 选内核来源（见「代码模式」）：标准 FA 复用 `FAInferKernel`（catlass examples/23）；变体走高阶 API
   或手搓（见 fa-variants-patterns.md）。
3. host 布局转换 + 独立 GM workspace + tiling 三件套。
4. 双口径精度验证（内部 golden + aclnn 标杆）。
5. msprof device 口径性能对比 + 归档。

## A2/A3 Stage 设计经验全文（a2-a3-flash-attention-stage-design）
> 以下为 stage 设计源文件全量正文：何时读取、四段流水、workspace 协议、online 状态管理、双特化与流水重叠、TileShape、buffer 预算与 tally 模板、checklist。

本文沉淀 FlashAttention（FA-2）算子在 A2/A3 上的 stage 设计经验。适用于同时包含 Cube/Vector 协作、KV 分块、online softmax 状态递推、跨 stage workspace 或性能流水调优的 Catlass FA 算子（配套 fa-kernel-handcraft.md 手搓手册，stage 契约与 buffer 预算供 kernel 设计时核对）。

### 何时读取

命中 FlashAttention 路由后，如果出现以下任一情况，设计/实现前必须读取本文：
- 算子含多个 stage（QK^T → softmax → PV），需 AIC/AIV 异核协作；
- KV 序列长、需要分块 + online softmax 状态（max/sum/O_acc）跨块递推；
- 涉及 CrossCoreFlag、workspace slot 复用、尾块处理；
- 性能优化涉及 Cube/Vector 流水重叠。

### 1. 四段流水与状态递推（FA-2）

每个 KV block 的四段流水（数据依赖方向严格）：

```
C1 (AIC)  S_j = Q·K_j^T · scale        → L0C → Fixpipe → UB
V1 (AIV)  m_j = max(m, rowmax(S_j))
          P_j = exp(S_j − m_j)
          l_j = l · exp(m − m_j) + rowsum(P_j)
C2 (AIC)  O_j = P_j · V_j               → L0C → Fixpipe → UB
V2 (AIV)  O_acc = O_acc · exp(m − m_j) + O_j
末块      O = O_acc / l
```

**禁止**改变 C1→V1→C2→V2 的顺序，或把 online 状态（m/l/O_acc）当作简单 buffer 复用以覆盖跨块依赖。跨 KV block 递推的状态必须完整、正确地累积。

### 2. GM workspace 与 CrossCoreFlag 协议

A2/A3 上 Cube→Vector 跨 stage 中间量默认经 GM workspace 中转。每条跨 stage 边满足：

| 规则 | 说明 |
|------|------|
| 稳定 slot | 同一个逻辑块在所有 stage 使用同一 workspace slot 公式（3 级轮转：s/p/oTemp/oUpdate）|
| stage 末尾 set | producer 完整写完该 slot 后再 set ready flag |
| stage 入口 wait | consumer 进入内层循环前 wait，不在 row/tile 循环内反复握手 |
| slot 不混用 | GM slot 与 UB ping/pong 概念分离 |
| modeId 一致 | `CrossCoreSetFlag/WaitFlag` 两侧 modeId 必须一致（A2 modeId 0/1/2；950PR 另有 4）|
| 清零明确 | O_acc 初值 0、m 初值 -inf 的初始化路径明确 |

### 3. Online softmax 状态管理（m / l / O_acc）

- `m`（running max）、`l`（running sum）、`O_acc`（输出累加）是跨 KV block 递推状态，必须按 block 更新：
  - rescale 因子 `exp(m_old − m_new)` 对 `O_acc` 和 `l` 都要乘；
  - 每个 block 先 rescale 旧状态，再加新 block 贡献（`O_acc = O_acc·exp(m_old−m_new) + P_j·V_j`）。
- 尾块（`Skv % 128 != 0`）0 填充后，softmax 语义不受影响（0 填充位置的 exp 贡献为 0），但必须保证 0 填充后的数值不引入 NaN/Inf。

### 4. AIC/AIV 双特化与流水重叠

catlass `FAInferKernel` 用 `operator()<AIC>` / `operator()<AIV>` 双特化：
- AIC 特化负责两个 Mmad（QK^T 与 PV）及 Fixpipe；
- AIV 特化负责 softmax/rescale；
- 两者用 CrossCoreFlag 同步。经验：A2 上正确重叠后 `cube_util ~ 96%`、AIC/AIV 高度并行（长上下文 1159μs vs 参考 1166μs）。
- 若 Cube/Vector 互等明显，优先检查 flag 是否放在内层循环、workspace slot 是否跨 stage 冲突，而不是调 TileShape。

### 5. TileShape 与分块

- 首个可跑选型参考 `L1 = L0 = GemmShape<128, 128, 128>`；Bq（query 分块）、Bk（KV 分块）由 UB/L1 预算推导。
- `Bk=128`（blockSize）与 block_table 的 KV 块粒度一致；改 Bk 需同步改 host 块重排。
- 小 `Sq*H`（少于核数）时先评估核调度/分块策略，不直接调 TileShape。

### 6. BNSD 接口转换的 stage 归属

BNSD→内部布局转换（Q→BSND、K/V→块格式、O→BNSD）属于 **host 侧 stage**，不进 kernel：
- host 转换与 kernel 四段流水解耦，kernel 只消费一种内部布局；
- 转换循环的 shape 校验（B/H/S/D、`Skv%128`）在 host 完成，避免 kernel 分支。

### 7. buffer 预算（FA 跨核流水最易踩 UB/L1 越界，写 kernel 前先纸面算）

#### 硬件上限（A2 / dav-2201 / 910B3）

| Region | 上限 | FAInferKernel 里装什么 |
|---|---|---|
| UB | 192 KB | AIV：online softmax 的 m/l（每行标量）、S（QK 输出 fp32，ping-pong）、P（softmax 输出 fp16，ping-pong）、O_acc rescale、mask（causal）|
| L1 | 512 KB | AIC：Q/K/V/P tile（`L1TileShape 128×128×128`，K 维 = D ≤ 128），preload/ping-pong 操作数 |
| L0A | 64 KB | Q tile（PV 阶段复用为 P）|
| L0B | 64 KB | K tile（PV 阶段复用为 V）|
| L0C | 256 KB | QK 的 fp32 累加器（PV 写 `(M,D)` 子区）|

> 任一 region 超上限的表现都是 **runtime 设备错误**（编译干净）。FA 最常踩的是 **UB**：S/P ping-pong + m/l + mask 同时在 UB 里，rowNum/stackSeqTile 一放大就超 192KB。

#### GM workspace（4 段独立 buffer，指针透传，禁 `GetUserWorkspace`）

| 段 | 用途 | 尺寸（`blockDim = aicCoreNum`）|
|---|---|---|
| `s` | QK 输出 S（AIC→AIV）| `blockDim × 131072 × 12` B |
| `p` | online softmax 输出 P | `blockDim × 131072 × 6` B |
| `oTemp` | PV 输出 O_block | `blockDim × 131072 × 12` B |
| `oUpdate` | rescale O 更新量 | `blockDim × 131072 × 12` B |

#### 预算 tally 模板（生成算子时在 DESIGN.md 填这张表）

```
| Buffer        | shape           | dtype | depth | bytes |
|---------------|-----------------|-------|-------|-------|
| S_ub (QK out) | (rowNum, N_qk)  | fp32  | 2     |  ...  |
| P_ub (softmax)| (rowNum, N_qk)  | fp16  | 2     |  ...  |
| m / l (状态)  | (rowNum, 1)     | fp32  | -     |  ...  |
| O_acc (rescale)| (rowNum, D)    | fp32  | 1     |  ...  |
| mask_ub       | (128, 1024)     | fp16  | 1     |  ...  |
| TOTAL (UB)    |                 |       |       |  ≤192K?|
```

预算紧时的压缩顺序：① 减 `stackSeqTile`（KV 块栈数 blockStackNum）；② 减 `rowNum`（Q 块行 qSBlockSize/qNBlockSize）；③ aliasing 生命周期不重叠的 UB region。**不要**靠"vf/cast 会被编译器折叠掉"来省 UB（可能不折叠，实测占 UB）。

### 8. Checklist

- [ ] 四段流水顺序严格 C1→V1→C2→V2，不重排
- [ ] online 状态（m/l/O_acc）跨块正确 rescale/累积
- [ ] GM workspace slot 公式跨 stage 稳定，flag 只在 stage 入口/末尾
- [ ] CrossCoreFlag Set/Wait modeId 一致
- [ ] 尾块 0 填充且不引入 NaN/Inf
- [ ] A2 上 `PAGED=true` + 恒等 block_table
- [ ] BNSD 接口转换在 host，kernel 无运行时布局分支
- [ ] 性能报告区分 Cube/Vector 互等、调度欠利用、尾块 3 类瓶颈

# 代码模式

**内核来源锁定（最高优先级）**：[^fa-generator-clauses]

- **标准 FA**（Q·Kᵀ→softmax→·V，单 KV）：**默认**复用 catlass `examples/23` 的 `FAInferKernel`
  组件链。若实测性能未达需求门槛（典型为短序列/少头数等任务切分粒度受限场景），允许回退到本册
  「手搓手册全文」「配方集全文」的族内手搓流水路径（不违反内核来源锁定，属同族实现路径切换）；
  切换后须以同口径精度与性能实测数据作为选路依据。
- **FA 变体**（paged/sink/varlen/FFA/MLA/复合特性等）：用 AscendC 高阶 API（`MatMul`/`SoftmaxFlashV2`）
  或按 fa-variants-patterns.md 的手搓配方开发；高阶 API 用 Event 同步（非 CrossCoreFlag 黑盒），错误可观测。
- **FA 不走 catlass/examples 样例对照**：其它算子类型按「找最接近 example」选型，FA 是例外，直接用
  FA kernel 设计知识。[^fa-architect-clauses]

标准 FA 组件选型表（仅供理解 FAInferKernel 设计原理）：[^fa-design-route]

| 组件 | 取值 | 说明 |
|------|------|------|
| ArchTag | `Arch::AtlasA2`（A2/A3） | 由目标 SoC 定 |
| BlockMmadQK | `MmadAtlasA2FAIQK` | C1：Q·Kᵀ，K 按 ColumnMajor 消费（host 不预转置） |
| BlockMmadPV | `MmadAtlasA2FAIPV` | C2：P·V |
| Tail 组件 | `MmadAtlasA2FAITailQK/FAITailPV` | KV 非 128 对齐尾块 |
| EpilogueOnlineSoftmax | `EpilogueAtlasA2OnlineSoftmax` | V1：online softmax + running max/sum |
| EpilogueRescaleO | `EpilogueAtlasA2RescaleO` | V2：按 `exp(m_old−m_new)` rescale O |
| Kernel | `FAInferKernel<...>` | `operator()<AIC>` + `operator()<AIV>` 双特化 |
| PAGED_CACHE_FLAG | `true`（A2） | 恒等 block_table 等价非 Paged |

数据流：`QK^T → L0C → Fixpipe → UB(S) → EpilogueOnlineSoftmax(P,m,l) → P·V → L0C → Fixpipe →
UB(O_acc) → EpilogueRescaleO → O`。组件名随 catlass 版本可能不同，生成前 grep 核验真实组件名与是否
有特化体（`grep -rn "MmadAtlasA2.*QK" catlass/include/catlass/gemm/dispatch_policy.hpp`）。
**catlass 来源必须用 gitcode.com/cann/catlass**（HEAD 兼容 FAIQK 旧版 API）；canndev/CANN opp 自带
catlass 是 FAQK 新版 API（6→9 参 BlockMmad、无 Tail），编译报 ~20 个 `MmadAtlasA2FAIQK 未定义` 即此因。[^fa-dev-patterns]

**独立 CMake 直调构建**（不走 examples 模板时必须显式注入，易漏）：[^fa-compilation]

```cmake
target_compile_definitions(<op> PRIVATE CATLASS_ARCH=2201)   # ★架构头 #if 依赖此宏，--npu-arch 不能替代
target_include_directories(<op> PRIVATE ${CATLASS_INCLUDE} $ENV{ASCEND_HOME_PATH}/include)
target_compile_options(<op> PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)
target_link_libraries(<op> PRIVATE ascendc_runtime ascendcl runtime profapi mmpa c_sec
                       error_manager graph_base tiling_api register platform unified_dlog dl m)
```

host 侧要点：核数用 `aclGetDeviceCapability(dev, ACL_DEVICE_INFO_AI_CORE_NUM, &v)`（enum 属
`aclDeviceInfo`）；`aclrtGetHardwareSyncAddr(&hwSync)` 传内核首参，内核入口 `SetSyncBaseAddr(hwSync)`；
`aclrtMalloc` 返回 `void*` 须 `(GM_ADDR)` 显式强转。host main 用 `.asc` 扩展名按 ASC 编译。[^fa-compilation]

## FA 实现规则全文（patterns/flash-attention.md）
> 以下为实现规则源文件全量正文：§0 kernel 设计经验（四大门槛/AIC-AIV 分工/在线 softmax/同步协议/workspace/Tile/HardEvent/PAGED/causal 跳块/开发策略分流）、§1-8 BNSD 接口、尾块、精度、性能基线、坑表与 FA0-FA13。

> **导航**：设计路由见本册「设计路由全文」。本文聚焦实现期规则：kernel 设计经验、BNSD 接口、PAGED/block_table 方案、host 布局转换、AIC/AIV 协作、精度/性能归档。

---

### 0. FA kernel 设计经验总结（★最高优先级——先读本文再写 kernel）

#### 0.1 FA kernel 手搓的四大门槛（历史坑，已全部破解并沉淀为手搓手册）

标准 FA-2（Q·Kᵀ→softmax→·V）的 kernel 在 Ascend A2 上是 **AIC（Cube）+ AIV（Vector）跨核协作**的混合 kernel。早期（无手搓手册时）用 catlass Block 级组件从零拼装，以下四项时序是**黑盒**、盲拼必死锁（现已全部破解，见下方现状）：

| 黑盒项 | 为什么手工对不齐 | 错了的后果 |
|---|---|---|
| **CrossCoreFlag Set/Wait modeId 配对** | AIC set flag 和 AIV wait flag 的 modeId（A2: 0/1/2）必须精确配对，但哪些 stage 用哪个 modeId 没有文档，只能从已验证 kernel 逆向 | **静默死锁 507015**——AIC 已 set，AIV 卡在 `CrossCoreWaitFlag` 不返回，无报错无日志 |
| **HardEvent flag 预置** | kernel 入口必须预置 `M_MTE1×8 + FIX_M×2 + MTE1_MTE2×8` 共 18 个 `SetFlag`，对应 kernel 出口 18 个 `WaitFlag`。漏一个或顺序错 → 流水死锁 | 同上 |
| **online softmax m/l UB 布局** | m（running max）和 l（running sum）在 UB 中的位置、与 S/P buffer 的 aliasing 关系是 EpilogueOnlineSoftmax 内部实现，外部不可见 | 精度静默错误（S 被置零或 rescale 错） |
| **preLaunch 流水时序** | preLaunch=2 意味着 AIC 比 AIV 领先 2 个 KV 块，flag 的 set/wait 要匹配这个偏移 | 流水断流 → 死锁或性能崩溃 |

**现状（已破解）**：上表四项黑盒已全部从已验证 kernel 源码**逆向推导为显式规则**，沉淀进 fa-kernel-handcraft.md（§3 同步编排、§5 事件预置、§6-§7 组件流水与 softmax 内部、§12.5.3 正确同步架构）。**手搓是默认路径**——FA 族任意形态（含标准 FA）均可凭手册从零写出，并经两轮纯 skill 实证（FFA 全家族 1.68× vs 标杆、复合变体一次成型）。
FA 族任意形态（含标准 FA）默认按手搓手册从零写出——§1–9 即 FAInfer 级内核的全量逆向知识，§12.13 为 kfc 直调终态配方；**只用本 skill 文件，不读任何外部内核源**。

#### 0.2 FA kernel 设计经验（从 FAInferKernel 提炼——FA 变体开发的基础知识）

以下是从已验证 FA 内核逆向提炼的**核心设计知识**——是手搓任何 FA 族 kernel（标准 FA 或 FFA 等变体）的设计原则，实现层细则见 fa-kernel-handcraft.md。**MLA 例外**：有整经验证内核，优先采用（见 fa-mla-paged.md，手搓慢一个数量级）。

##### 0.2.1 AIC/AIV 双核分工模型

FA kernel 在 A2 上分两个逻辑核运行：
```
AIC（Cube 核）：
  - 负责 GEMM（矩阵乘）：Q·Kᵀ（stage C1）和 P·V（stage C2）
  - 使用 L0C 累加 → Fixpipe → 写 GM workspace
  - 每个 KV 块做两轮 GEMM（QK 和 PV），通过 CrossCoreFlag 通知 AIV

AIV（Vector 核）：
  - 负责逐元素运算：online softmax（max/sum/exp/P）和 O rescale
  - 从 GM workspace 读 AIC 的 GEMM 结果
  - 维护跨 KV 块递推的 softmax 状态（m, l, O_acc）
  - 通过 CrossCoreFlag 通知 AIC
```

##### 0.2.2 在线 softmax 状态管理（核心算法——所有 FA 变体共用）

FA-2 的核心是不物化完整 S×S 得分矩阵，改用**在线 softmax**逐块递推：
```
初始化：m = -inf, l = 0, O_acc = 0

对每个 KV 块 j：
  S_j = scale · Q · K_jᵀ                         # AIC GEMM
  m_new = max(m, rowmax(S_j))                     # AIV
  P_j = exp(S_j - m_new)                          # AIV
  l_new = l · exp(m - m_new) + rowsum(P_j)        # AIV（rescale + accumulate）
  O_acc = O_acc · exp(m - m_new) + P_j · V_j       # AIV rescale + AIC GEMM
  m = m_new, l = l_new

最终：O = O_acc / l                                 # AIV 除法
```

**关键设计要点**：
- `m`（running max）和 `l`（running sum）跨 KV 块递推，**不可中断或重置**
- rescale 因子 `exp(m_old - m_new)` 对 `O_acc` 和 `l` 都要乘
- 除法（÷l）**后移到所有 KV 块处理完之后**（避免每块做除法）

##### 0.2.3 跨核同步协议（A2 平台）

AIC↔AIV 用 `Arch::CrossCoreSetFlag/WaitFlag` 握手：
```
AIC 产出 S_j → CrossCoreSetFlag<PIPE_FIX>(qkReady)     → AIV 等 qkReady
AIV 产出 P_j → CrossCoreSetFlag<PIPE_MTE3>(softmaxReady) → AIC 等 softmaxReady
AIC 产出 O_j → CrossCoreSetFlag<PIPE_FIX>(pvReady)     → AIV 等 pvReady
```
- **modeId 配对**：每个 flag ID（qkReady=1, softmaxReady=2, pvReady=3）在 AIC Set 和 AIV Wait 两侧 modeId 必须一致。A2 支持 modeId 0/1/2。
- **硬件同步基址**：kernel 首参 `hardwareSyncAddr`，由 host `aclrtGetHardwareSyncAddr` 获取，kernel 内 `SetSyncBaseAddr(hwSync)` 设置。

##### 0.2.4 workspace 槽位设计

FA 流水需要 4 段 GM workspace，用独立 buffer 指针透传（**禁 `GetUserWorkspace`/`SetSysWorkspaceForce`**——直调路径会丢入参返回 kfc 地址致 MTE 越界）：

| 段 | 用途 | A2 尺寸（blockDim = aicCoreNum） |
|---|---|---|
| s | QK 输出 S（AIC→AIV）| blockDim × 131072 × 12 B |
| p | softmax 输出 P（AIV→AIC）| blockDim × 131072 × 6 B |
| oTemp | PV 输出 O_block（AIC→AIV）| blockDim × 131072 × 12 B |
| oUpdate | rescale O 更新量 | blockDim × 131072 × 12 B |

**3 级轮转**（preLaunch=2）：每核用 `coreIdx × WORKSPACE_BLOCK_SIZE_DB × (preLaunch+1)` 定址，3 个 slot 循环使用，实现 AIC/AIV 流水重叠。`WORKSPACE_BLOCK_SIZE_DB = 131072`。

##### 0.2.5 Tile / 分块策略

| 参数 | 取值 | 约束 |
|---|---|---|
| L1TileShape | `GemmShape<128, 128, 128>` | K 维必须 = D（head_dim）；D ≤ 128 |
| Q 分块（qSBlockTile） | 128 | 固定（`GetQSBlockTile`） |
| N 分块（qNBlockTile） | 由 `GetQNBlockTile(qSeqlen, groupSize)` 推导 | 保证 vec 核负载均衡 |
| KV 分块（blockSize） | 128 | 与 PAGED 块大小一致 |
| preLaunch | 2 | AIC 领先 AIV 2 个 KV 块 |

##### 0.2.6 硬件事件预置（HardEvent flag pattern）

kernel 入口必须预置以下 flag（否则流水死锁）：
```
SetFlag<M_MTE1>(EVENT_ID0~7)   × 8
SetFlag<FIX_M>(EVENT_ID0~1)    × 2
SetFlag<MTE1_MTE2>(EVENT_ID0~7) × 8
```
kernel 出口对应 `WaitFlag` 各 8/2/8 个。**这些 flag 是 in-pipe 流水同步的基础**——AIC 的 M（Mmad）→ MTE1（L0→UB）→ MTE2（UB 操作）三段流水依赖这些 flag 推进。

##### 0.2.7 PAGED 方案（A2 平台）

A2 上 `PAGED_CACHE_FLAG=false`（真非 Paged）触发 aicore exception（507015），**必须 `PAGED=true`** + 恒等 block_table：
- `block_table[g] = g`（连续 KV 等价非 Paged）
- 每 batch `maxNumBlocksPerBatch = ⌈maxSkv/128⌉` 槽
- K 按 `LayoutK=ColumnMajor` 在 kernel 内消费，**host 不预转置 K**

##### 0.2.8 Causal mask 结构性跳块（自研掩码张量 = 固定 [1024,1024] fp16 压缩下三角）

下三角 causal（`maskType=MASK_CAUSUAL`）**不逐位算 mask**，而是**结构性跳过**：
- 内核按 query 块 `qSBlockIdx` 推导 `noSkipKvS = (qSBlockIdx+1) × curQSBlockTile + diffS`（diffS=Skv-Sq）
- 对角块以上的整 KV 块**完全跳过**（不算 QK、不算 softmax、不算 PV）
- 仅对角块调用带 mask 的 softmax epilogue（`mask[r][c]=1 当 c>r`）
- 效果：causal 比 full attention **省约一半计算**（上三角整体跳过）

#### 0.3 FA kernel 开发策略（按算子形态分流）

| 算子形态 | kernel 策略 | 原因 |
|---|---|---|
| **标准 FA / FA-Paged / GQA / Causal**（CrossCoreFlag 组件路径） | **首选 fa-paged-handcraft-recipe.md 逐行配方**（四文件端到端：kernel Step 1-7 + mask Step 8 + 三件套 Step 9-11，NO_MASK 闭卷实证 kv 至 20万 全 PASS）；机制解读回手册 §1–9 | 逐行级配方比骨架级手册少一层"再发明"；mask/causal 的结构跳块与 QKTail/PVTail 只有 recipe 有完整步骤 |
| 标准形态（kfc 单 TU 直调）/ fa-sink（S1 模板）/ varlen / q8 复合 | 手搓：**fa-sink → fa-sink-handcraft-recipe.md**（S1 模板逐步骤，sink=SoftmaxFlashV2 isUpdate 播种）；**varlen → fa-varlen-handcraft-recipe.md**（varlen kernel 逐步骤，TND 直读+批界截断）；任意序列长/复合 → §12.13 kfc 配方 + §12.12/§12.14 叠加 | **skill 自包含**；§12.13 实测 geomean 0.693 vs 标杆 sink 算子；§12.11 S1 模板 0.80-0.92x |
| **其他 FA 变体**（双 KV、FFA、FlashSoftmax 分片等） | 按 fa-kernel-handcraft.md 手搓单 kernel：套用骨架模式（任务切分/逐块流水/事件预置/host 模板），按变体映射法增改 GEMM 数/flag 数/workspace 段/softmax 相位特性；FFA 直接用 §12.5.14 终态配方 | 手册含全部同步规则与 12 个实锤坑，手搓能力已经两轮纯 skill 实证；变体数学没有现成 kernel |
| MLA（latent 压缩 + 分页 KV） | **手搓**（fa-mla-handcraft-recipe.md 端到端配方：kernel §1-8 + tiling §11 + 槽位/host §12；fa-mla-paged.md §8 为设计依据） | recipe 为端到端逐行配方，闭卷实证一次跑通；§8.9-8.12 五大性能支柱解释为什么快 |
| FFA / FIA（双 KV TensorList + sharedPrefix，DeepSeek 形态） | **生成走 §12.5.14 终态配方**；算子结构契约见 ffa-handcraft-recipe.md（sink-in-epilogue/FD/LSE/nZ/量化入参族语义） | §12.5.14 实证几何 2.19×；kernel 与 paged recipe 同构，差异已有清单 |

---

### 1. 公开接口必须是 BNSD，布局转换放 host

> **命名**：FA 算子统一命名 `FA-<layout>[-<mode>]`（如 `FA-BNSD` / `FA-BNSD-Causal` / `FA-TND`），见 design 路由「命名约定」。

FlashAttention 算子的**公开接口**取严格 BNSD：`Q=[B,H,Sq,D]`、`K/V=[B,H,Sk,D]`、`O=[B,H,Sq,D]`，全部连续 fp16。kernel 内部消费一种布局，**host 负责转换**，kernel 只认一种内部布局、不做运行时分支。

标准 FA（FAInferKernel）的内部布局：
- Q 取 **BSND** `[B,Sq,H,D]`（`strideQO = H*D`）
- K/V 取**块格式** `[numBlocks, blockSize, H, D]`（`blockSize=128`）
- O 出 BSND，host 转回 BNSD

host 转换要点：
```text
Q:   BNSD [B,H,Sq,D] → BSND [B,Sq,H,D]      （三重循环转置）
K/V: BNSD [B,H,Sk,D] → 块 [numBlocks,128,H,D]（按 128 块重排，尾块 0 填充）
O:   BSND [B,Sq,H,D] → BNSD [B,H,Sq,D]
```
**bin 读入尺寸与 device 尺寸分离**：BNSD K 的 bin 是 `B*H*Sk*D*2`，kernel 的块格式是 `numBlocks*128*H*D*2`。

---

### 2. KV 非 128 对齐的尾块处理

`numBlocksPerBatch = ceil(Skv / 128)`。`Skv % 128 != 0` 时尾块 0 填充：
- 块 buffer 先 memset 0，再按有效长度填充
- gen_data 与 host 都要用同一套 block 映射
- golden 用原始 BNSD K/V 计算，不受块格式影响

---

### 3. 精度验证：dump O + aclnn 标杆对比

- kernel 算出 O 后，**必须 dump 输出**到 `o.bin`，供外部对比
- 精度双口径：
  1. 内部 golden（CPU numpy/torch fp32 累加 softmax attention）逐元素对比
  2. **标杆接口对比**：`torch_npu.npu_fusion_attention`（封装 `aclnnFlashAttentionScore`），与算子 O 逐元素比
- 判据：`atol=0.02, rtol=0.1`，通过率 ≥ 99.9% 且 `max_abs < 0.05`

---

### 4. 性能归档

#### 度量口径（device time，不是 wall time）
- **device kernel 时间**用 cannbot `ops-profiling`（msprof）的 `task_time(us)`，**取 min of N**
- 对比标杆：`npu_fusion_attention` 的 `FlashAttentionScore` kernel（同 msprof 口径）
- **不要用 host 侧 wall-clock**比 kernel 性能（含 launch/sync/布局转换开销）

#### FA skill Step 4 覆盖矩阵系统性枚举的实测基线（910B3 device 1）

| 算子 | 小 shape (Sq≤256) | 大 shape (Sq≥1024) | 总均比 |
|---|---|---|---|
| FA-BNSD（全注意力）| 1.49×（catlass 快） | 0.87×（npu 快） | 1.16× |
| FA-BNSD-Causal | 1.68× | 0.98× | 1.23× |
| FA-TND（varlen）| 1.72× | 0.88× | 1.35× |

> 小 shape catlass 快：npu aclnn FA 的框架调度开销在小 shape 占主导，kernel 直调无此开销。大 shape npu 快：aclnn FA 的大 shape 优化更激进。Sq=512 基本持平。

#### 性能字段（归档到 `docs/perf/round_NNN/`）
| 字段 | 说明 |
|------|------|
| baseline | `aclnnFlashAttentionScore` / `npu_fusion_attention` 同 shape 实测（msprof device time）|
| Task Duration | custom kernel vs baseline，同 shape、同 msprof 口径 |
| cube_util | AIC 利用率（A2 经验 ~96%，AIC/AIV 高度重叠）|
| dominant pipeline | Cube / MTE / Vector / 同步等待中的主瓶颈 |
| workspace peak / profiler path | 原始 msprof trace 路径 |

---

### 4.5 手搓 FA kernel

- **标准 FA / FA-Paged / GQA / Causal（组件路径）→ 直接按 fa-paged-handcraft-recipe.md 逐行写**（含 mask 路径与三件套，勿跳过 Step 6 的 3 条修复与 Step 8 的结构跳块公式）。
- **fa-sink（S1 模板）→ 按 fa-sink-handcraft-recipe.md**（sink 用 SoftmaxFlashV2 isUpdate 播种法，勿手写 max/add 链）。
- **fa-varlen（TND）→ 按 fa-varlen-handcraft-recipe.md**（seqlen 累加和 kernel 内差分；TND 直读不转块；批界用 s2End 截断）。
- **FFA 双 KV → §12.5.14 生成 + ffa-handcraft-recipe.md 语义对照**。
- **MLA → 按 fa-mla-handcraft-recipe.md 逐行写**（分流必须 `__DAV_CUBE__/__DAV_VEC__`、主循环必须 prologue+延迟 PV——两种锁步/`__mix__` 变体均实测死锁）。
- **其他 FA 变体 / kfc 直调路径 → 先读 fa-kernel-handcraft.md**：4 个文件结构、任务切分算法、AIC/AIV 双入口与主循环、跨核同步编排、HardEvent 预置、BlockMmad 流水、online softmax/rescale 内部实现、host 直调模板、变体映射法、自查表。它是 §0.2 的实现层补充，§12.14-C 有全家族覆盖路由表。

---

### 5. 已知问题与踩坑（实战总结，生成前必读）

| # | 现象 | 根因 | 对策 | 来源 |
|---|------|------|------|------|
| T1 | AIC 已 set flag，AIV 卡在 softmax 内 `CrossCoreWaitFlag`/MTE2 wait 不返回（507015） | **从零拼装标准 FAKernel，flag 时序未对齐** | 标准 FA 复用 FAInferKernel（§0.1）；FA 变体用高阶 API（§0.3）| 实战复现 |
| T2 | A2 上 `PAGED_CACHE_FLAG=false` 跑出 aicore exception | 组件库该路径在 A2 未验证 | 固定 `PAGED=true` + 恒等 block_table（§0.2.7）| A2 平台复现 |
| T3 | 报某组件"未定义"/"只有声明无特化体" | 不同 catlass 版本组件名不同；某些 DispatchPolicy 只有 `struct X{}` 声明、无 `block_epilogue_*.hpp` 特化体 | 生成前 grep 核验真实组件名 + 是否有特化体（design §Step2）| catlass 版本差异实测 |
| T4 | 编译报 `MmadAtlasA2FAIQK 未定义` / `AscendC 未声明`（~20 errors） | 用了系统自带 catlass（canndev / CANN opp，**FAQK 新版 API**），与 FAInferKernel 要的 **FAIQK 旧版 API** 不兼容（6→9 参 BlockMmad、无 Tail） | 用 gitcode 的 catlass：`git clone https://gitcode.com/cann/catlass.git`（HEAD 兼容 FAIQK）；**不**用 canndev 自带 catlass | 部署实证（canndev catlass 不兼容）|
| T5 | gen_data 与算子/对比脚本的参数顺序不一致 | 参数顺序不统一 | 统一参数顺序，三方对齐 | 测试工程实证 |
| T6 | `o.bin` reshape 失败 | data 目录残留旧 shape 数据 | 每次 gen 前清 data 目录 | 测试工程实证 |
| T7 | 首次 launch 偶发卡死 | NPU 首次初始化竞态 | 重试复现即排除 | NPU 经验 |
| T8 | workspace 越界（MTE DDR） | 直调路径用了 `GetUserWorkspace` | 独立 GM buffer 指针透传（§0.2.4）| 实战实证 |

---

### 6. 强制检查表（FA0–FA13）

| # | 检查项 |
|---|--------|
| FA0 | **kernel 来源正确**：FA 族形态默认**按 fa-kernel-handcraft.md 手搓单 kernel**（骨架模式 + 变体映射法，过 §11 自查表 K1–K10）；标准 FA 复用整经验证内核仅当语义完全一致且零定制（快捷方式，非能力依赖）；**MLA 按 fa-mla-paged.md §8 手搓（达标架构，五大性能支柱缺一不可；§8.13 已闭卷实证）** |
| FA1 | 公开接口为严格 BNSD（Q=[B,H,Sq,D]，K/V=[B,H,Sk,D]），布局转换在 host |
| FA2 | 冻结同语义 baseline（`aclnnFlashAttentionScore`），语义/dtype/layout/scale 一致 |
| FA3 | A2 上固定 `PAGED=true` + 恒等 block_table（标准 FA）；FA 变体按算子需求设计 KV 布局 |
| FA4 | KV 非 128 对齐时尾块 0 填充；bin 读入尺寸与 device 尺寸分离 |
| FA5 | workspace 用独立 GM buffer 指针透传，禁 `GetUserWorkspace` |
| FA6 | AIC↔AIV 同步（CrossCoreFlag 或 Event）modeId/ID 一致，无死锁 |
| FA7 | 标准 FA 组件选型为 BlockMmadQK/PV + EpilogueOnlineSoftmax/RescaleO；FA 变体按 §0.2 原则选高阶 API |
| FA8 | shape 覆盖矩阵含 B/Sq/Sk 组合、非 128 对齐尾块、MHA/GQA、D=64/128、数值边界 |
| FA9 | 精度双口径（内部 golden + aclnn 标杆），mixed tolerance，`max_abs`/`matched_ratio` 记录 |
| FA10 | 性能报告含 baseline、Task Duration、launch count、cube_util、profiler path |
| FA11 | 多 stage/AIC-AIV 场景已读取并执行 `a2-a3-flash-attention-stage-design.md` checklist |
| FA12 | **mask 模式正确**：掩码按内核 `MaskType` 选型（`NO_MASK`/`MASK_SPEC`/`MASK_CAUSUAL`）；下三角 causal 用 `MASK_CAUSUAL`，勿自实现 causal。mask 张量语义/契约见本册「标杆接口参考全文」 |
| FA13 | **actualQseqlen/Kvseqlen 是 per-batch 原始长度（非累加）**；dense BNSD 传 `[Sq]×B`/`[Skv]×B` |

**审查门禁（主体 reviewer Step FA，FA1–FA11 编号体系）**：[^fa-reviewer-clauses]

主体审查 agent 的 FA 专项门禁：命中 FlashAttention / MHA / GQA / fused attention 时追加执行
FA1–FA11 检查表，重点核查：

- **FA3**：A2 上是否固定 `PAGED=true` + 恒等 block_table（`PAGED=false` 触发 aicore exception 应判定为必须修复）
- **FA1**：公开接口是否严格 BNSD、布局转换是否在 host（kernel 无运行时布局分支）
- **FA4**：KV 非 128 对齐尾块 0 填充、bin 读入尺寸与 device 尺寸分离
- **FA9**：精度是否双口径（内部 golden + `aclnnFlashAttentionScore` 标杆对比），记录 `max_abs` / `matched_ratio`
- 组件选型是否 BlockMmadQK/PV + EpilogueOnlineSoftmax/RescaleO（标准 FA 复用 FAInferKernel；FA 变体按 §0.2 原则用高阶 API）

**编号对应关系**：主体 reviewer 的 FA1–FA11 与上表 FA0–FA13 同源同义（FA1↔FA1 … FA11↔FA11 逐条
对应）；FA0（kernel 来源正确）、FA12（mask 模式 MaskType）、FA13（seqlen per-batch 约定）为 ops
活体表扩展项，主体 FA1–FA11 未单列（语义部分含于其组件选型/精度核查）。执行以 FA0–FA13 全表为准，
FA1–FA11 为其最小子集。

---

### 7. mask 模式（MaskType）

| 值 | 名称 | 含义 |
|---|------|------|
| 0 | `NO_MASK` | 全注意力，无 mask |
| 1 | `MASK_SPEC` | 用 mask 张量逐位屏蔽 |
| 2 | `MASK_CAUSUAL` | **下三角 causal**：结构性跳过 + 对角块读 mask |

- 下三角 causal（npu `sparse_mode` 2/3）用 `MASK_CAUSUAL`(=2)：内核按 query 块 `qSBlockIdx` 结构性跳过对角块以上整 KV 块（§0.2.8），仅对角块调用带 mask 的 softmax。
- mask 语义：`ApplyMask = score += mask·(-3e38)` → mask≠0 屏蔽、0 保留（与 npu `atten_mask` 的 `1=不参与` 一致）。
- 标杆对比：`npu_fusion_attention(input_layout="BNSD", sparse_mode=2, atten_mask=triu(ones,bool,1))`（封装 `aclnnFlashAttentionScore`），与算子 O 逐元素比。

### 8. seqlen 契约：per-batch（非累加）

`actualQseqlen/actualKvseqlen` 是**每 batch 原始长度**（int64 数组，长度=B），内核 `qBOffset += qSeqlen·strideQO` 自行累加偏移。
- dense BNSD：`[Sq]×B` / `[Skv]×B`
- varlen TND：传每 batch 原始 Sq/Skv（非累加和）

## 手搓手册全文（fa-kernel-handcraft）
> 以下为手搓手册源文件全量正文：四文件结构、任务切分、AIC/AIV 主循环、同步编排、HardEvent 预置、BlockMmad/softmax/rescale epilogue 内部机制、host 直调模板、变体映射法、自查表、常量速查与全家族覆盖路由表。 [^fa-handcraft-manual]

> **定位**：这是 §0.2「FA kernel 设计经验」的**实现层**补充——§0.2 讲"怎么设计"，本手册讲"怎么写出代码"。读完本手册应能**不参考任何外部文件**，从零写出一个标准 FA（或 FA 变体）的单 kernel。
>
> **适用对象**：需要在 Ascend A2（910B/910C，AtlasA2）上手搓 FA 族 kernel 时。标准 FA 与 FA 变体（FFA/MLA/复合特性 sink/varlen/q8 等）**全部按本手册骨架（§1-11）+ 配方册/特性册（§12 系列，编号不变）手搓**（自包含，不依赖任何外部 kernel 库/示例源码）。
>
> **事实来源**：整经验证 FA 内核全量 4 文件逆向（`fai_kernel.cpp` 797 行 + `fai.cpp` 319 行 + `fai_tiling.cpp` 163 行 + `kernel_common.hpp` 154 行）+ 库层 `include/catlass/gemm/block/block_mmad_fai_qk_normal.hpp`(325 行) + `include/catlass/epilogue/block/block_epilogue_online_softmax_no_mask.hpp`(821 行) + `block_epilogue_rescale_o_no_split_row.hpp`(274 行)。

---


> **分册导航**：本册是 FAInfer 级 kernel 的**骨架手册**（§1-11 + 常量速查 + §12.14-C 全家族覆盖路由表）。
> §12 系列配方已分册（编号不变）：**§12.5（FFA 实证库）/ §12.10（s1s2 算法）/ §12.11（S1 sink）/ §12.13（kfc 终态配方）→ fa-kernel-recipes.md**；
> **§12.6/§12.7/§12.12/§12.14（复合特性叠加：sink/varlen/q8/causal/paged 与测量坑）→ fa-kernel-compound-features.md**。

### 1. 你要写的 4 个文件（一个 FA kernel = 4 份代码）

> **★生成入口**：标准 FA/FA-Paged/GQA/causal（CrossCoreFlag 组件路径）**直接按
> fa-paged-handcraft-recipe.md 逐行写**（四文件端到端全量提取，本节
> §1-9 是其机制解读层——recipe 写不出来时回来读这里理解为什么）；MLA 按
> fa-mla-handcraft-recipe.md；kfc 单 TU 路径按 §12.13。

| 文件 | 内容 | 运行侧 | 行数参考 |
|---|---|---|---|
| `fai_kernel.cpp` | kernel 本体：AIC/AIV 双入口 `operator()`、任务切分、主循环、同步编排 | device | ~800 |
| `fai.cpp` | host 直调：参数解析、读数据、建 workspace、tiling、`<<<>>>` 启动 | host | ~320 |
| `fai_tiling.cpp` | host 侧 tiling：任务数/workspace 尺寸/scale 计算 | host | ~160 |
| `kernel_common.hpp` | 常量（flag ID、BLOCK_SIZE、workspace 块尺寸）、`FATilingData`、`FAIKernelParams` 结构体、对齐工具 | 共用 | ~150 |

**一个 kernel = 单一 C++ 源码里两个 `extern "C" CATLASS_GLOBAL` 入口（Fp16/Bf16），各自实例化同一个模板 `FAInferKernel<>`，模板内按 `g_coreType` 分流 AIC/AIV 的 `operator()`。** host 一次 launch 同时拉起 AIC 核（Cube）与 AIV 核（Vector）协作。

---

### 2. 任务切分算法（多核并行 + N 维 pack 进 M）

这是 FA kernel 能用普通 GEMM 组件的关键技巧。标准 FA 是 BNSD：`Q=[B,H,Sq,D], K/V=[B,H,Sk,D]`。**切分单位是 (batch, Sq 块, N 块)**：

```
每个任务 = 一个 (batch, qSBlockIdx, qNBlockIdx)
  qSBlockSize = 128（固定，Sq 按 128 分块，尾块取余）
  qNBlockTile = min(max(⌊128/qSeqlen⌋/2*2, 1), groupSize)   // N 维分块，保证偶数为 AIV 双子核负载均衡
  qNBlockNumPerGroup = ⌈groupSize / qNBlockTile⌉
  任务总数 += qNBlockNumPerGroup × kvHeads × qSBlockNum   // 逐 batch 累加
```

**核心技巧——N 维 pack 进 GEMM 的 M 维**：

```
rowNum = qSBlockSize × qNBlockSize   // 把 (S 块 × N 块) 拍平成 GEMM 的行
Q 读入后按 [qSBlockSize, qNBlockSize, D] 当作一个 rowNum×D 的矩阵做 GEMM
O 写回时按 qNBlockSize 分头拆分回 [Sq 块][H] 位置
```

**GQA（Nq>Nkv）手搓要点**：`groupSize = Nq/Nkv`（必须整除）；任务切分第 2 行的 `qNBlockTile` 按 groupSize 界定；`kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup`、`qHeadIdx = kvHeadIdx·groupSize + 组内偏移`——K/V 平面用 kvHeadIdx、Q/O 平面用 qHeadIdx 索引（两套平面偏移勿混）。

多核分发（AIC/AIV 两侧相同的循环骨架）：

```cpp
uint32_t coreIdx = AscendC::GetBlockIdx();          // AIC 直接取；AIV 要 /GetSubBlockNum()
uint32_t coreNum = AscendC::GetBlockNum();
uint32_t curTotalTaskNum = 0, preTotalTaskNum = 0, curBatch = 0;
for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += coreNum) {
    while (taskIdx >= curTotalTaskNum) {            // 跨 batch 推进：偏移累加
        ++curBatch; preTotalTaskNum = curTotalTaskNum;
        qBOffset += qSeqlen * strideQO;             // strideQO = H*D（BSND）
        kvBOffset += kvSeqlen * strideKV;           // strideKV = kvHeads*D
        qSeqlen = gActualQseqlen.GetValue(curBatch);
        kvSeqlen = gActualKvseqlen.GetValue(curBatch);
        curTotalTaskNum += /* 本 batch 任务数 */;
    }
    uint32_t t = taskIdx - preTotalTaskNum;
    uint32_t qSBlockIdx = t / curQNBlockNum;        // → Sq 块
    uint32_t qNBlockIdx = t % curQNBlockNum;        // → N 块（含 head 分组）
    uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
    uint32_t kvHeadIdx  = qNBlockIdx / qNBlockNumPerGroup;          // GQA：head 分组归属哪个 KV head
    uint32_t qHeadIdx   = kvHeadIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
    uint32_t rowNum = qSBlockSize * qNBlockSize;
}
```

---

### 3. AIC/AIV 双入口 + 主循环（软件流水骨架）

#### 3.1 入口分流

```cpp
template <int32_t CORE_TYPE = g_coreType>
CATLASS_DEVICE void operator()(FAIKernelParams const& params);

template <> CATLASS_DEVICE void operator()<AscendC::AIC>(FAIKernelParams const& params)
{ /* Cube：loadQGM → 主循环(GEMM QK → flag → GEMM PV) */ }

template <> CATLASS_DEVICE void operator()<AscendC::AIV>(FAIKernelParams const& params)
{ /* Vector：等 qkReady → online softmax → flag → 等 pvReady → rescale O */ }
```

#### 3.2 AIC 主循环（关键常量与流水结构）

```cpp
uint32_t blockStackNum = 4;      // 4 个 KV 块叠成一个 [rowNum, 512] 大块（充分利用 Cube）
uint32_t preLaunch   = 2;        // AIC 领先 AIV 2 个大块（延迟消费，软件流水深度）
int32_t  totalStackSeqNum = ⌈noMaskKvS / (4*pagedBlockSize)⌉ (+1 当 masked);
uint32_t pagedBlockSize = 128;   // KV 分块（PAGED 块大小）

// 第一循环：无 mask 的 KV 块（全注意力部分）
for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumNoMask; kvSIdx += blockStackNum) {
    stackSeqTile = (尾块) ? noMaskKvS - kvSIdx*pagedBlockSize : pagedBlockSize*blockStackNum;
    // ① QK GEMM：S = Q·Kᵀ（写 workspace 槽 A）
    blockMmadQK(..., stackSeqCount % (preLaunch+1) 槽);      // 槽 = stackSeqCount%(preLaunch+1)
    CrossCoreSetFlag<0x2,PIPE_FIX>(qkReady);                 // 通知 AIV：S 就绪
    // ② PV GEMM：延迟 preLaunch 个大块消费 P（写 O_tmp 槽 B）
    if (kvSIdx >= preLaunch*blockStackNum) {
        blockMmadPV(..., (stackSeqCount-preLaunch)%(preLaunch+1) 槽);  // 读 P，等 softmaxReady
        CrossCoreSetFlag<0x2,PIPE_FIX>(pvReady);
    }
    stackSeqCount++;
}
// 第二循环：masked KV 块（causal/对角，+ 尾部 preLaunch 个块的收尾 PV）
for (kvSIdx = maskedStartIdx; kvSIdx < kvSLoopNumTotal + preLaunchStackNum; ) {
    // 同结构：QKTail（带 mask）+ 延迟 PVTail（最后一块带 mask）
    kvSIdx += (特殊边界) ? noMaskTailInteStackNum : blockStackNum;
}
```

**因果 mask 不逐位算**：`noSkipKvS = min(kvSeqlen, (qSBlockIdx+1)*qSBlockTile + diffS)` 结构性跳块；仅对角块调带 mask 的 softmax epilogue。

#### 3.3 AIV 主循环（对应地消费）

```cpp
for (kvSIdx = 0; kvSIdx < kvSLoopNumNoMask; kvSIdx += blockStackNum) {
    CrossCoreWaitFlag(qkReady);                       // 等 AIC 的 S
    epilogueOnlineSoftmax(gP, gS, ..., isFirst=(stackSeqCount==0));   // S→P
    CrossCoreSetFlag<0x2,PIPE_MTE3>(softmaxReady);    // 通知 AIC：P 就绪
    if (kvSIdx >= preLaunch*blockStackNum) {
        CrossCoreWaitFlag(pvReady);                   // 等 AIC 的 O_block
        epilogueRescaleO(gO, gOTmp, ..., isFirst, isLast);            // 累加+最终÷l
    }
    stackSeqCount++;
}
```

---

### 4. 跨核同步完整编排（AIC ↔ AIV）

flag 注册在 kernel 类成员：`Arch::CrossCoreFlag qkReady{QK_READY_ID=1}, softmaxReady{2}, pvReady{3};`

| 数据流 | AIC 动作 | AIV 动作 | pipe |
|---|---|---|---|
| S（QK 输出） | `SetFlag<0x2,PIPE_FIX>(qkReady)` | `WaitFlag(qkReady)` 后读 S | AIC 写 GM workspace |
| P（softmax 输出） | `WaitFlag(softmaxReady)` 后读 P | `SetFlag<0x2,PIPE_MTE3>(softmaxReady)` | AIV 写 GM workspace |
| O_block（PV 输出） | `SetFlag<0x2,PIPE_FIX>(pvReady)` | `WaitFlag(pvReady)` 后 rescale | AIC 写 GM workspace |

- **modeId 配对**：`SetFlag<0x2, ...>` 的 modeId=2 必须与 AIV `CrossCoreWaitFlag` 内部一致（A2 支持 0/1/2）。**Set/Wait 两侧不一致 → 静默死锁 507015**。
- **硬件同步基址**：kernel 首参 `hardwareSyncAddr`（host `aclrtGetHardwareSyncAddr` 获取），入口第一行 `AscendC::SetSyncBaseAddr(hardwareSyncAddr)`。

---

### 5. HardEvent 硬件事件预置（in-pipe 流水同步）

**kernel 入口**必须预置的 `SetFlag`（漏一个或顺序错 → 流水死锁）：

```
AIC 侧： M_MTE1(EVENT0~7)×8 + FIX_M(EVENT0~1)×2 + MTE1_MTE2(EVENT0~7)×8
AIV 侧： MTE3_V(EVENT0~1)×2 + MTE3_MTE2(EVENT2~5)×4 + V_MTE2(EVENT0~3)×4 + MTE3_V(EVENT2)×1
```

**kernel 出口**对应 `WaitFlag`（逐一配平）。这些 flag 对应 A2 的物理流水依赖：
- `M_MTE1` = Cube(MMAD) 完 → L0C→UB（FixPipe→MTE1）
- `MTE1_MTE2` = UB 内操作完 → MTE2 搬运（GM/UB 间）
- `FIX_M` = FixPipe 完成 → 下一轮 MMAD 可清零 L0C
- `V_MTE2` / `MTE2_V` / `MTE3_V` / `V_MTE3` = Vector 与搬运的握手

---

### 6. BlockMmad（GEMM 块）内部机制——手搓 QK/PV 时必须复刻的流水

以 `block_mmad_fai_qk_normal.hpp`（325 行）为准，它是**一个普通 A2 GEMM + FA 专属的 KV 块堆叠**：

#### 6.1 内存分级与缓冲

```
L1：l1ATensor（Q，整块一次载入，rowNum×D）
    l1BTensor[2]（K/V，2 级乒乓，每级 L1TileShape::N×K = 128×128）
L0：l0ATensor[STAGES]、l0BTensor[STAGES]（乒乓）、l0CTensor（累加器）
```

#### 6.2 loadQGM：Q 的多头 Nd→L1 装载

```cpp
// Q 是 BSND [B,Sq,H,D]，一个任务取 qHeadIdx 起的 qNBlockSize 个头
// 按 [qNBlockSize, D] 一块块拷贝，nStr=H*D（跳头）
copyGmToL1A(l1ATensor, gA, layoutAInL1, layoutSingleANd, tokenNumPerGroup, qHeads*embed, ...);
SetFlag<MTE2_MTE1>(EVENT_ID3);  WaitFlag<MTE2_MTE1>(EVENT_ID3);  // 搬运完成屏障
```

#### 6.3 内层计算循环（QK：M=rowNum, N=KV列, K=embed）

```cpp
for (blockStackIdx=0; blockStackIdx<4 && (nIdx+blockStackIdx)<nLoop; ++blockStackIdx) {
  for (kIdx=0; kIdx<kLoop; ++kIdx) {          // kLoop = ⌈embed/128⌉
    nowNIdx = nIdx + blockStackIdx;
    getBlockShape(...); getKVOffset(...);      // 实际 tile 尺寸 + PAGED block_table 偏移
    // 每 2 个 KV 块一个输出槽：gCOffset = (blockStackIdx/2*2)*128
    computeQK(..., firstItr=(blockStackIdx%2==0 && kIdx==0),
                  endItr  =(blockStackIdx%2==1 || nowNIdx==nLoop-1) && kIdx==kLoop-1,
                  initMmad=kIdx==0, firstQItr=blockStackIdx==0, ...);
  }
}
```

#### 6.4 computeQK：单次 GEMM tile 的精确事件序列（★复刻顺序）

```cpp
// ① K 的 GM→L1（双缓冲，kIdx==1 时先发第一块，此后一直提前发下一块）
copyGmToL1B(l1BTensor[pong], gB, ...); SetFlag<MTE2_MTE1>(pong);
// ② Q 的 L1→L0（firstQItr 才做一次；此后 l0ATensor[0] 复用）
WaitFlag<M_MTE1>(0); WaitFlag<M_MTE1>(1); copyL1ToL0A(l0ATensor[0], l1ATensor, ...);
// ③ K 的 L1→L0（乒乓）
WaitFlag<MTE2_MTE1>(pong); WaitFlag<M_MTE1>(pong+2); copyL1ToL0B(l0BTensor[pong], l1BTensor[pong], ...);
SetFlag<MTE1_MTE2>(pong);
// ④ MMAD（initMmad=true 时清零 L0C；firstItr 需先等 FIX_M 0/1——等上一轮 FixPipe 写完 L0C→GM）
if (firstItr) { WaitFlag<FIX_M>(0); WaitFlag<FIX_M>(1); }
tileMmad(l0CTensor[pong*mRound*128], l0ATensor[0], l0BTensor[pong], mRound, nActual, kActual, initMmad, unitFlag);
// ⑤ L0C→GM（endItr 才搬；FIX 前必须 M_FIX 同步，写 zN→RowMajor）
if (endItr) { SetFlag<M_FIX>(pong); WaitFlag<M_FIX>(pong);
              copyL0CToGm(gC, l0CTensor, layoutC, layoutInL0C);   // zN 布局
              SetFlag<FIX_M>(0); SetFlag<FIX_M>(1); }
```

**K 每 4 个块堆成一个 512 列大块**（`gCOffset = (blockStackIdx/2*2)*128`），S 在 GM 的 stride 用 512 对齐（`LayoutC(rowNum, stackTile, 512)`）——让 AIV 侧一次处理 512 列。

---

### 7. Online softmax epilogue（AIV 核心算法）——手搓实现

`block_epilogue_online_softmax_no_mask.hpp`（821 行）。**核心是 m/l/O 状态在 UB 的布局 + 一组向量运算**。

#### 7.1 UB 布局（AIV 私有 buffer）

```
LS（S 输入，fp32）  @ 0x00        ← AIC 的 QK 输出，pingpong×MAX_UB_S_ELEM_NUM(8192)
LP（P 输出，fp16）  @ 4*16384
LM/LL（本块 rowmax/rowsum）@ 10*16384+8*1024 起   （8 字节/行 × MAX_ROW_NUM_SUB_CORE=128）
HM/GM（全局 max）/ DM（diff）/ GL（全局 sum）@ 9~13*1024 偏移
TV（临时/tv）@ 10*16384
```

#### 7.2 逐块计算序列（对应数学：m_new=max, P=exp, l_new 递推）

对每个输入行子块（按 `MAX_ROW_NUM_SUB_CORE=128` 行 + `preLoad=1` 乒乓）：

```
① CalcLocalRowMax    lm = rowmax(S)                    // BlockReduceMax，512 列走专用 3 级归约
② UpdateGlobalRowMax hm = max(lm, gm)                  // 第一块：hm=lm（copy）；否则：
                      dm = gm - hm; dm = exp(dm)        //   rescale 因子 = exp(m_old - m_new)
                      gm = hm
③ CalcExp            tv = Brcb(hm) 广播成块;  ls -= tv; ls = exp(ls)   // S←exp(S-m_new)
④ DownCastP          lp = cast_fp32_fp16(ls)           // P
⑤ CalcLocalRowSum    ll = rowsum(ls)                   // BlockReduceSum
⑥ UpdateGlobalRowSum gl = dm*gl + ll                   // 第一块：gl=ll（copy）；否则累加
```

**关键算子**：`BlockReduceSum/Max`（块内逐行归约，512 列 3 级 / 256 列 2 级 / 尾块 mask 处理）、`Brcb`（1×N 状态向量广播成整块）、`SetVectorMask`（尾部对齐）。

#### 7.3 双子核切分（GetSubBlockIdx）

`qNBlockSize==1` 时按行对半（每子核 ≤64 行）；`>1` 时按 head 分（子核 0 拿前 `qNBlockSize/2` 头）。两子核对同一 GM S 分区消费。

#### 7.4 数据流同步（AIV 内部）

```
WaitFlag<V_MTE2>(pong) → CopyS → SetFlag<MTE2_V>(pong)     // S GM→UB（preLoad 提前 1 行）
SubCoreCompute 内：      WaitFlag<MTE2_V> → 计算 → SetFlag<V_MTE3>(P写GM)
                         SetFlag<V_MTE2>(ll写GM) → WaitFlag<V_MTE3> → CopyP → SetFlag<MTE3_V>
mask 版：额外 EVENT_ID2 管 mask 的 GM→UB + ApplyMask（mask32*=-3e38 再加到 S）
```

---

### 8. Rescale O epilogue（AIV）——手搓实现

`block_epilogue_rescale_o_no_split_row.hpp`（274 行）：

```
// isFirstStackTile：O_tmp 直接拷贝进 go（不 rescale）
// 否则：go = go*dm  +  lo（lo 是上一大块 O_tmp）—— 在线累加 O_acc
// isLastStackTile：go = go / gl（Brcb 广播 gl → Div）；cast fp32→fp16；写 GM
```

```
WaitFlag<V_MTE2>(EVENT3) → DataCopy(lo) → SetFlag<MTE2_V>(EVENT0) → WaitFlag<MTE2_V>(EVENT0)
tv = Brcb(dm); go *= tv;  go += lo                    // 非首块
tv = Brcb(gl); go /= tv;  go = cast(go); CopyOToGm    // 末块（÷l 只做一次）
SetFlag<V_MTE2>(EVENT3)
```

---

### 9. Host 侧直调模板（fai.cpp 的骨架）

```cpp
auto aicCoreNum = PlatformAscendCManager::GetInstance()->GetCoreNumAic();
uint32_t blockDim = aicCoreNum;
// workspace 4 段（每段 = aicCoreNum × 131072 × 3 个元素 × 元素字节）
//   s=fp32, p=fp16, oTemp=fp32, oUpdate=fp32（§0.2.4 已有）
// tiling：FAInferTiling::GetFATilingParam(faInfo, blockDim, faTilingData) → memcpy 到 device
uint64_t hardwareSyncAddr;
aclrtGetHardwareSyncAddr((void**)&hardwareSyncAddr);
FAInferFp16<<<blockDim, nullptr, stream>>>(
    hardwareSyncAddr, q, k, v, mask, blockTables, o,
    qSeq, kvSeq, s, p, oTemp, oUpdate, tiling);
```

**kernel 入口签名（14 参数，顺序固定）**：
```cpp
extern "C" CATLASS_GLOBAL void FAInferFp16(
    uint64_t hardwareSyncAddr, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o,
    GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR s, GM_ADDR p, GM_ADDR oTemp, GM_ADDR oUpdate,
    GM_ADDR tiling)
{ AscendC::SetSyncBaseAddr(hardwareSyncAddr); FAInferKernel<> kernel; kernel(params); }
```

---

### 10. 变体映射法：怎么把一个新 attention 形态手搓成 kernel

拿到一个新 FA 变体（FFA、MLA、双 KV…），按以下顺序映射到骨架：

| 步骤 | 问自己 | 对应修改点 |
|---|---|---|
| 1 数学 | 分几步 GEMM？每步 A/B/C 是什么、S 还是 P 语义？ | 新增/修改 `blockMmad*` 调用；`MmadAtlasA2FAIQK`/`FAIPV` 换成对应语义 |
| 2 flag | 数据依赖 AIC↔AIV 有几条？每步产物谁消费？ | 新增 `CrossCoreFlag`（ID 递增）；每数据流一对 Set/Wait（modeId 一致） |
| 3 workspace | 中间产物几类？fp16/fp32？ | `FATilingData` 加字段；`fai.cpp` 加段；`FillWorkSpaceTilingData` 加尺寸 |
| 4 主循环 | 每步 GEMM 的 (M,N,K)、按哪个维分块、有无跳块 | 主循环里按顺序插 QK/PV 调用的对（流水：产 S→flag→等 P→PV→flag） |
| 5 host 布局 | 公开接口 BNSD/其他？输入布局怎么转 | host 转换（§1 已有）；`FAIKernelParams` 加指针 |
| 6 epilogue | softmax 变体（分片、对角）？ | 仿 §7/§8 写新 epilogue，复用 BlockReduce/Brcb 模式 |
| 7 精度/性能 | 冻结 baseline、mixed tolerance、msprof 口径 | FA9/FA10 检查表 |

**FFA 实例**（`S=scale·(Q·K1ᵀ+Q·K2ᵀ对角)→softmax→O=W·V1+W·V2对角`）：
- 数学是 **4 个 GEMM**（QK1、QK2、PV1、PV2），其中 K2/V2 带对角语义（Q 的 M 与 K2 的 M 共享）；实现上把对角 matmul 的 `M` 维也 pack 进 GEMM 行（与 N-pack 同思路），或用 permute 消除对角（host 做）。
- flag 需要 **5 条**（QK1Ready、QK2Ready、PReady、PV1Ready、PV2Ready）而非 3 条；workspace 相应加段。
- 主循环每轮：AIC 产 QK1/QK2 → flag → AIV 先加再 softmax → flag → AIC 产 PV1/PV2 → flag → AIV rescale O。
- 其余骨架（任务切分/双循环/preLaunch/事件预置/host 模板）原样复用。

---

### 11. 手搓自查表（写完 kernel 后逐条过）

| # | 检查项 |
|---|---|
| K1 | AIC/AIV 两个 `operator()` 特化都存在；入口 18 个 SetFlag（AIC）/10 个 SetFlag（AIV）与出口 WaitFlag 一一配平 |
| K2 | `SetSyncBaseAddr(hardwareSyncAddr)` 在入口第一行；host 传对 `aclrtGetHardwareSyncAddr` |
| K3 | 每个 `CrossCoreFlag` 的 Set（AIC/AIV 侧）与 Wait（对侧）modeId 一致；ID 全局唯一且 < 硬件支持数 |
| K4 | workspace 槽位 `coreIdx×(preLaunch+1)+slot`，slot = `stackSeqCount%(preLaunch+1)`，两侧算法一致 |
| K5 | `rowNum = qSBlockSize×qNBlockSize` 的 N-pack 在 Q 装载、S 布局、O 拆分三处对称 |
| K6 | tail 块：`stackSeqTile = noMaskKvS - idx*pagedBlockSize`；非 128 对齐时 RowReduce 用 SetVectorMask 处理 |
| K7 | 除法 ÷l 只在最后一个 KV 块（isLastStackTile）；rescale 因子 exp(m_old-m_new) 对 O 和 l 都乘 |
| K8 | causal 用结构性跳块（noSkipKvS），不是逐位 mask |
| K9 | host 与 kernel 的 tiling 字段、参数顺序、bin 尺寸三方一致 |
| K10 | 精度双口径（内部 golden + aclnn 标杆）+ msprof device time 归档 |

---


---

### 12. 关键常量速查

```
BLOCK_SIZE = 16（元素对齐基数）
WORKSPACE_BLOCK_SIZE_DB = 131072（≈128K 元素槽，fp32）
L1TileShape = GemmShape<128,128,128>（K 必须 = head_dim D，D≤128）
pagedBlockSize = 128；blockStackNum = 4；preLaunch = 2
qSBlockTile = 128（GetQSBlockTile 固定）
QK_READY_ID=1, SOFTMAX_READY_ID=2, PV_READY_ID=3
MAX_UB_S_ELEM_NUM = 8192；MAX_ROW_NUM_SUB_CORE = 128
UB_UINT8_BLOCK_SIZE = 16384；UB_UINT8_VECTOR_SIZE = 1024
FLOAT_VECTOR_SIZE=64；FLOAT_BLOCK_SIZE=8；VECTOR_SIZE=128（fp16）
```


---

#### C. 覆盖对照（全家族 → 手搓路径）

| 算子 | 手搓路径 |
|---|---|
| 标准 FA / FA-BNSD / **FA-Paged**（CrossCoreFlag 组件路径） | **fa-paged-handcraft-recipe.md**（四文件端到端逐步骤：kernel Step 1-7 + mask Step 8 + 三件套 Step 9-11，NO_MASK 闭卷实证全 PASS）；§1–9 为其机制解读层 |
| 标准 FA（kfc 单 TU 直调路径） | §12.13（含 F0 十一步逐步生成流程） |
| FA-BNSD-Causal | fa-paged-handcraft-recipe.md Step 8（组件级 MASK_CAUSAL）或 §12.13 + §12.14-A（kfc 逐行尾部法） |
| FA-TND varlen | **fa-varlen-handcraft-recipe.md**（varlen kernel 差异逐步骤：累加差分/TND 直读/s2End 截断/L1Carry/SameAB + 两路对照）或 §12.13 + §12.12-A2（kfc + 转块格式路径） |
| FA-paged（kfc 路径） | §12.13 + §12.14-B（block_table 查表改块级） |
| FA-sink（S1 模板路径） | **fa-sink-handcraft-recipe.md**（S1 模板 kernel 逐步骤：三深流水/SoftmaxFlashV2 isUpdate 播种 sink 正确用法/workspace 精确公式/sparse 跳块）+ §12.11（算法语义/tiling 公式/基线）；kfc 路径 = §12.13（内建 sink） |
| FFA（双 KV） | §12.5.14 完整配方（生成路径）+ ffa-handcraft-recipe.md（FFA/FIA 算子契约：TensorList/sharedPrefix/sink-in-epilogue/FD/LSE/nZ 语义与差异清单） |
| 复合（sink+varlen+q8 任意组合） | §12.6 总纲 + §12.6-E + §12.12（互不冲突，四合一 +4.3%） |
| GQA | recipe Step 3d/4d 双平面索引（kvHeadIdx/qHeadIdx；§2 机制解读） |
| fa-sageattention（A5/950 量化注意力） | **fa-sageattention-recipe.md**（Ascend950/dav-3510：开发流程七步 + architecture/rules/troubleshooting/precision 四参考，自包含） |
| MLA | **fa-mla-handcraft-recipe.md**（端到端：kernel §1-8 + tiling §11 + 槽位/host §12；§12 含 FD rescale kv-split 归并全码）；fa-mla-paged.md §1-4 为 host 契约、§8 为设计依据 |

## 配方集全文（fa-kernel-recipes）
> 以下为配方册源文件全量正文（§ 编号保持原状）：§12.5 FFA 全程实证与终态配方、§12.10 s1s2 单核融合算法、§12.11 fa-sink S1 模板流程、§12.13 kfc 单 TU 直调终态配方（含 F0 逐步生成流程）。 [^fa-recipe-book]

> **分册说明**：本册是 fa-kernel-handcraft.md §12 系列的**算子配方部分**（原手册拆分，§ 编号不变）：
> §12.5 FFA 全程实证与终态配方、§12.10 s1s2 单核融合算法剖析、§12.11 fa-sink S1 模板流程、§12.13 kfc 单 TU 直调终态配方。
> 复合特性叠加（sink/varlen/q8/causal/paged）见 fa-kernel-compound-features.md；骨架见手册 §1-11。

> **简称**：本文档称「配方册」；fa-kernel-compound-features.md 称「特性册」。

---

### 12.5 FFA 手搓实测（已验证 + 卡点）

> 实测工程（本地自写 FFA kernel，catlass BlockMmadQK + 自写任务/softmax/rescale/host）。**90% 环节已验证正确，卡在跨核整幅同步**。以下是有实据的经验，直接复用。

#### 12.5.1 FFA 数学 → 任务映射（已验证正确）

FFA `S=scale·(Q·K1ᵀ+Q·K2ᵀ对角)→softmax→O=W·V1+W·V2对角` 的**两个 GEMM batch 维正交**（a0 的 B 按 n 分组、a1 的 B 按 m 分组），**同一行排布下不可能都高效**。解法（**已验证 S0/S1 max_abs=0.0000**）：

```
a0 = Q·K1ᵀ  任务(n,m0) 行=M  B=K1[n]共享   → S0   （标准 GEMM）
a1 = Q·K2ᵀ  任务(m,n0) 行=N  B=K2[m]共享   → S1   （对角 → 按对角维 batch 拆成标准 GEMM）
o1 = P·V1   任务(n,m0) 行=M  B=V1[n]共享   → O1
o2 = P·V2   任务(m,n0) 行=N  B=V2[m]共享   → O2
```
- **host 预处理**（`scripts/gen_data.py`）：Q0[N,M,D]、Q1[M,N,D]（Q 转置）、K1/V1[N,K,D]、K2/V2[M,K,D]（K2/V2 转置 [K,M,D]→[M,K,D]）、V1ᵀ[N,D,K]/V2ᵀ[M,D,K]。
- **workspace 布局**（全连续，AIV 逐 (n,m) 向量加和）：S0[N,M,K]fp32、S1[M,N,K]fp32、P0[N,M,K]fp16、P1[M,N,K]fp16、O1[N,M,D]fp32、O2[M,N,D]fp32、L[N,M]fp32。
- S0 和 S1 用**不同布局**（S0 按 [N,M,K]、S1 按 [M,N,K]），AIV 逐 (n,m) 读两个 K 向量相加（不同地址，各自连续）。

#### 12.5.2 已解决的踩坑（全部实测确认）

| # | 现象 | 根因 | 正确做法 |
|---|---|---|---|
| F1 | Add/Muls/Exp 只处理前 2 个元素 | **4 参版 `Add(dst,src0,src1,count)` 的 count 是元素数**（非 repeat 数） | 传 **K**（元素数），不是 K/64 |
| F2 | ReduceMax 的 dst[0] 不是全局 max | `ReduceMax(dst,src,tmp,count)` 每 64 元素输出 1 个（多 repeat） | **分段**：`ReduceMax(...,64)` 循环 K/64 次，再合并 |
| F3 | O1 错（PV 阶段） | PV 复用 BlockMmadQK（ColumnMajor B），B 物理须为 **V^T=[D,K]**（host 转置），不是原始 V=[K,D] | gen_data 输出 V1ᵀ[N,D,K]；kernel B 偏移 n*D*K |
| F4 | count 语义统一：Exp/Cast/Adds 也是元素数 | 同 F1 | 全传 K/D |

#### 12.5.3 ✅ 已解决：跨核同步用"逐块流水 + 全核参与"（实测 PASS）

**现象**：整幅同步（AIC 全算完 S → 通知 AIV 全算 softmax → 通知 AIC 全算 PV）在 `aclnrtSynchronizeStream` 死锁。

**根因**：AscendC `CrossCoreSetFlag/WaitFlag` 是 **1 Set 配 1 Wait 的核间配对**。整幅同步下：
- 只有核 0 Set → 20 个 AIC Wait 只有 1 个通过 → **死锁**。
- 所有核无条件 Set（含没做事的核）→ AIC 提前读半成品 P0 → **精度错**。

**已解决的正确架构（实测 7+ shape 全 PASS）**：
1. **逐块流水（FAI 模式）**：每个 (n0,m0) tile 内 AIC 做 A0/A1 → Set qkReady → AIV softmax → Set softmaxReady → AIC C1/C2 → Set pvReady → AIV rescale。flag 按块 1:1 配对。
2. **全核参与**：每个拥有 tile 的 AIC 核 / AIV 子核对都参与 Set+Wait；**无 tile 的核直接 return，绝不碰 flag**。
3. **AIC 与 AIV 的 taskIdx 必须对齐**（都用 `taskIdx = coreIdx` / `aicIdx`，步进 `coreNum`，处理相同 tile），每对核间 1:1 flag。
4. **多核分发**：`for (taskIdx = coreIdx; taskIdx < totalTiles; taskIdx += coreNum)`，支持 tiles > 核数。

**另一个关键：纯 AIC 方案不可行**（AIC 核无 UB/VECCALC）。AIC 核用 BlockMmad 的 L1/L0（cube 内存）做 GEMM 完全正确，但 **AIC 核上跑 AscendC vector softmax（DataCopy/Add/Exp 到 UB）失败（P0 全 0）**——`resource.ubBuf`（VECCALC）是 AIV 核的内存，AIC 核访问无效。**softmax/rescale 必须在 AIV 核做**（框架 Matmul API 封装路径可绕过此限制，手搓路径不可）。

#### 12.5.4 ✅ 已解决：BlockMmad 连续调用需 PipeBarrier（异步残留）

**现象**：GEMM 循环（A0/A1/C1/C2 连续多次 `blockMmadQK`）出现**前几个调用错**（D=32 时 S1 的 m=1,2 错；O2 的 m=1 错），之后稳定。

**根因**：标准 `BlockMmad` 连续调用都复用 `l1ATensor[0]`（isFirst 时载入）。前一次调用的异步 MTE（copyL1ToL0A 读 l1ATensor[0]）未完成时，下一次 `copyGmToL1A` 覆盖 → 前几次"接力"边界错。

**已解决**：**每次 `blockMmadQK` 调用后加 `AscendC::PipeBarrier<PIPE_ALL>()`**（强制本次 GEMM 完全串行）。代价是损失流水重叠（性能下降），但正确。**性能优化时**应改用乒乓 buffer / 减少全屏障。

#### 12.5.5 ✅ 已跑通：精度与性能实测

- **精度**：9 shape（D=32/64/128，N/M 跨 16~1024）**matched_ratio=1.0，max_abs<0.016**，全 PASS。
- **性能演进**（N=16 M=128 K=128 D=64，标杆 aclnn 274μs）：
  - v1（逐行 softmax + 全屏障）：1614μs（0.17×）
  - v2（批量 softmax，见 §12.6）：685μs（**0.40×**）
  - 大 shape 最好 case（256×1024×64）：**3561μs vs 标杆 5943μs = 1.67× 反超标杆**
- **几何平均 0.38×**（11 case）。剩余差距主因：tile 数 < 核数时单核串行 + 同步头开销（profiler 显示 Task 611μs 中核最长仅 43μs，**头开销 92.9%**）。

#### 12.5.6 性能优化实战经验（v1→v4，全部实测验证，最终几何 1.68× 反超标杆）

**优化演进路径**（16×128×64 case，标杆 272μs）：

| 版本 | 关键改动 | 该 case 用时 | vs 标杆 |
|---|---|---|---|
| v1 | 逐行 softmax + 每 GEMM 后 PIPE_ALL | 1614μs | 0.17× |
| v2 | softmax/rescale 批量化（16 行/批）+ 屏障分层 | 687μs | 0.40× |
| v2' | **MTE1_MTE2 token dance**（替代长循环的 PIPE_ALL） | 454μs | 0.60× |
| v3 | + **动态 tile**（TILE 128→32/64 按形状选择） | **134μs** | **2.04×** |

**关键优化技术（全部实测）**：

| # | 技术 | 收益 | 实现要点 |
|---|---|---|---|
| P1 | softmax/rescale 批量化（16 行/批） | 2.3× | 向量算子 count 传批量元素数；同 n 内 m 连续 → S0 连续 1D 拷贝、S1 用 2D strided load；每 (row,seg) Reduce 用**私有 tmp 槽（≥256 floats）**（槽小读出垃圾→exp 溢出 inf） |
| P2 | 屏障分层 | 含 P1 | **必须 PIPE_ALL**：DataCopy load 后（MTE2→V）、Cast→store 前（V→MTE3，竞态实例：sub0 赢竞态 sub1 读旧零）；纯向量间用 PIPE_V |
| P3 | **MTE1_MTE2 token dance**（W2 的正确解） | 1.5×（687→454） | FAQK 组件每次调用后 Set(MTE1_MTE2, pp)（L1A 读排空信号，内部供 k+2 次调用消费）。调用间插入 `Wait(MTE1_MTE2, ppPrev); Set(MTE1_MTE2, ppPrev)`——**先等读方排空再补回 token**，既消除覆盖竞态又不破坏组件内部事件配对。**短循环（<32 次）用 PIPE_ALL 更稳，长循环（128 次）用 dance**；跨相位边界必须 PIPE_ALL 隔离；循环尾残留 SET 会挂起 profiling 版内核（token 账目要平） |
| P4 | **动态 tile 大小** | 3.4×（454→134） | tile 从编译期常量改为 tiling 字段。实测最优：默认 32；64 当 `tiles₆₄∈[12,32]`（中 shape）。tile 越小核越多但单 GEMM 越小——按形状调优。**注意：TILE≠128 曾在旧代码（全屏障版）触发 AIV 静默写零；token dance 版 32/64 全 PASS——该坑与同步结构相关，非 TILE 本身** |

**最终基线（v4，11 个 Step-4 case，手搓=事件计时 min-of-10，标杆=msprof）**：

| case | 手搓 (μs) | 标杆 (μs) | 加速比 |
|---|---|---|---|
| 16×128×64 (L0) | 133.8 | 272.5 | **2.04×** |
| 32×128×64 | 234.8 | 255.7 | 1.09× |
| 16×128×32 | 124.0 | 372.2 | **3.00×** |
| 16×128×128 | 150.9 | 476.7 | **3.16×** |
| 16×256×64 | 143.6 | 206.3 | 1.44× |
| 64×256×64 | 236.8 | 221.0 | 0.93× |
| 16×512×64 | 135.0 | 226.4 | 1.68× |
| 128×512×64 | 833.0 | 370.2 | 0.44× |
| 256×1024×64 (L3) | 2981.5 | 5852.3 | **1.96×** |
| 16×1024×64 | 239.9 | 1041.9 | **4.34×** |
| 64×256×128 | 262.1 | 441.5 | 1.68× |

**几何平均 1.68×，9/11 反超标杆，10/11 ≥0.6×，唯一弱项 128×512（0.44×）**——该 shape tile 内 GEMM 链太长（每核对 256 次 GEMM 深度 1 串行），需要更深流水（官方用框架 Matmul API 的多级任务环流水）。

**坑（W 系列，实测确认）**：

| # | 现象 | 根因 | 对策 |
|---|---|---|---|
| W1 | 2D strided DataCopy **UB→GM store**（fp16）只写第一个 block | MTE3 2D store 不可靠（MTE2 load 正常） | store 退化为逐行 1D |
| W2 | 无屏障连续 GEMM：前几次调用数据错（L1A 覆盖竞态） | 下一调用的 copyGmToL1A 覆盖 l1A[0] 时，上一调用 MTE1 读未完成；组件内部 Set(MTE1_MTE2,pp) 由 k+2 次调用消费，保护滞后 2 拍 | **P3 token dance** |
| W3 | 全屏障版 TILE≠128 多 tile 时 AIV 静默写零 | 与同步结构相关的旧问题（token dance 版已消失） | 用 v2'+ 同步结构后 TILE 32/64 安全 |
| W4 | 批量寻址漏 tile 偏移 | 全局坐标 = tile 相对坐标 + tile 原点 | 统一 `nG = n0*TILE + r0/mBlk` 换算 |
| W5 | profiler 采集偶发全空（host/device_1 目录空） | 设备多次 core dump 后分析子系统劣化（全设备、官方 exe 也复现） | 降级 **aclrtEvent 流计时**（kernel 前后 event、warmup 3 + 10 次取 min；与 msprof 口径偏差 <5%）；aclnn 每次 launch 有 ~1.4s host 开销（executor 复用无效）不能事件计时标杆侧，官方基线沿用 profiler 健康期的 msprof 数据 |

#### 12.5.10 ✅ 纯 skill 生成复合功能变体（sink/varlen/q8，2026-08-17 第二轮验证）

**验证协议**：同 §12.5.8（只读本手册），一次生成 3 个可组合特性 + 全组合，32/32 PASS（20 老 case 回归 + 12 特性组合）。

**特性实现（全部 v4 路径 K=128）**：
| 特性 | 数学 | kernel 落点 |
|---|---|---|
| **FA-Sink**（StreamingLLM 语义） | 虚拟 sink 列（score=sinkScore, V=0）：m'=max(m,sink)，l'+=exp(sink-m')，O 归一化自动正确 | softmax 相位：m 合并处 `if(m<sink)m=sink`；l 存储处标量 exp 加成 |
| **FA-Varlen**（核内前缀 mask，无 mask 张量） | k≥kvLen[b,n] → -3e38 | softmax 相位：`gLen.GetValue` 标量 + K 次 SetValue 构造掩码行 + Add |
| **FA-Q8**（W8A8 式核内反量化） | int8 输入 ×(1/scale) → fp16 staging → 正常 GEMM | **新增 dq 相位**：AIV tile 内反量化本 tile 操作数切片 → Set(dqReady id=4) → AIC Wait(dq) 再 A 相位（第 4 个配对 flag，逐 tile 1:1） |

**开销实测**（16×128×64，基线 157.5μs）：sink +5%、varlen ±0%、q8 +0.3%、四合一 +4.3%——特性近乎免费。

**新坑（W10-W13，全部实锤）**：
| # | 现象 | 根因 | 对策 |
|---|---|---|---|
| W10 | 特性 kernel 输出全零、kernel 正常返回（AIV 静默死亡） | **UB 向量算子 buffer 必须 32B 对齐**（标量 Exp 的 scratch 放 93188=4 偏移 → 核异常；流式路径 93184 对齐所以没事） | 所有 UB 偏移 %32==0；对齐坑的签名=输出全零+正常返回 |
| W11 | 量化输出大 ~750×（=sc²） | gen 的 scale 语义是 int8=x·s（乘），dequant 应 **除**：kernel Muls 需传 **1/s** | scale 契约：gen 写 s（乘法系数），host 求倒数传入 |
| W12 | dav_c220 无 int8→float、无 bf16←half Cast | 该架构 Cast 支持表有限 | int8 链路固定 half 中转；**q8 仅支持 fp16**（bf16+q8 host 拒绝）；另：**量化 golden 必须用 round(x·s)/s（量化-反量化值），不能用 x/s（未量化缩小值）**——这是测试侧最容易犯的语义错 |
| W13 | q8 dequant 输出"看似合法但全错"（反量化值对不上 int8 输入；某 plane 甚至全零），Cast/scale 单测都对 | **AIV DataCopy(GM→UB int8) 与后续 Cast 之间缺 PipeBarrier**——MTE2 加载未完成，V 管 Cast 抢先读 ubI8 陈旧值；fp16 路径同款代码没炸只因碰巧没踩同拍竞态 | DataCopy 加载后、Cast 前必须 `PipeBarrier<PIPE_ALL>`（MTE2→V）；改法：dequantPlane 在 DataCopy 与 Cast 之间补一条屏障即可，其余链（V→V、V→MTE3）原样 |

**特性设计通用法**（可复用到其它变体）：① mask 类特性（varlen/三角/局部窗口）= softmax 相位构造 -3e38 掩码行，与 mask 张量可叠加（双 -3e38 等价 max 语义）；② softmax 状态类特性（sink/logit-bias）= 修改 m/l 递推，不动 GEMM；③ 输入格式类特性（量化/低精度）= 新增 AIV 预处理相位 + 配对 flag，GEMM 主路径零改动。



#### 12.5.9 ✅ P1/P2：spec 全覆盖（bf16/多BH/aux输出/K块流式，2026-08-17）

**P1（增量扩展，全部一次或近一次通过）**：
- bf16：kernel 元素类型全走 `ElementQ`，入口克隆 `FAFFABf16`（`bfloat16_t`）；gen/verify 用 torch `view(uint16)` 转 numpy
- B/H：`totalTasks = BH*nBlocks*mBlocks` 展平，`bh = taskIdx/tilesPerBH`；所有张量加 plane 偏移（**V1ᵀ plane 步长 = N·D·K（含 N！），V2ᵀ = M·D·K**）；mask 按 `b = bh/H` 共享
- softmax_max/sum [B,H,N,M,8]：softmax 相位直接 SetValue（host 预清零保证 [1..7]=0）
- 实测坑：P1 store 漏 op1 偏移 → 全 plane 写 plane0 互踩（smax 对但 O 全错——辅助输出对+主输出错=偏移在 store 侧）

**P2（K 块流式 + 在线 softmax，攻 3 个新坑后全通）**：

架构（KB=128，nK=K/128；tileDim 强制 32）：
```
per tile per block j（块级锁步，复用 3 个 flag nK 次）:
  AIC: A0_j(nBlk GEMM→S0t) + A1_j(mBlk→S1t) → Set(qk) → Wait(sm) → C1_j(P0t·V1ᵀcol_j→O1b) + C2_j(→O2b) → Set(pv)
  AIV: Wait(qk) → online softmax_j → Set(sm) → Wait(pv) → Oacc_j
online: mNew=max(mOld,blkMax); α=exp(mOld-mNew)(j=0 置 0); P=exp(S-mNew); l=α·l+ΣP
        Oacc = Oacc·α + O1b + O2b（j=0 直写防 NaN×0）; 末块: O=Oacc/l + 存 smax/ssum
workspace：per-core 槽（20×TILE²×KB，S/P/O 独立槽 + Oacc 专缓冲）——内存与 K/N/M 总量解耦
标量 exp 用向量 API 单元素（`Exp(ubSc,ubSc,1)`），标量 expf 在核内不可用
```

**P2 新坑（全实测）**：

| # | 现象 | 根因 | 对策 |
|---|---|---|---|
| W6 | m/l 跨行混乱（probe 显示单行递推正确、存储值错） | **tile 局部 buffer 的 2D strided LOAD 也混行**（W1 只知 store；全局大 stride 的 2D load 正常、tile 局部小 stride 的会混） | S1t/O2b 全部逐行 1D |
| W7 | C 相位 GEMM 结果乱 | **B 是 [D,K] 平面的列切片：列 stride=K≠KB**；`MakeLayout(KB,D)` 声称 stride=KB 错（v4 中 K==KB 掩盖） | 布局传全宽 `MakeLayout(K,D)`，块范围由 `GemmCoord.k=KB` 表达 |
| W8 | Oacc 阶段 chunk 起始行(r=0)对、r≥1 全错 | O2b 行 stride=nBlk·D 被当连续 1D 拷贝 | 逐行 1D load |
| W9 | 偶发 Bus error（加 printf 即好） | 块级锁步边界缺隔离 | 每块起止 + Wait 后各补 PIPE_ALL（保守但稳，5×压测通过） |

**P2 验证**：K=256/512/1024 × mask/bf16/多BH/D32/64/128/大N/M 共 8+8 组合全 PASS（O+smax+ssum 三输出）；K=128 全量回归 20/20。流式性能：K=1024 1252μs，workspace 仅 ~10MB（非流式方案同 shape 需 1.3GB）——**大 K 只有流式可行**。



#### 12.5.8 ✅ skill 闭环验证：纯 skill 生成 FFA-Mask 变体（2026-08-17）

**验证协议**：只读本手册 + FFA 需求说明，不打开任何外部仓 / catlass 组件源码。目标 = 全新功能（atten_mask 掩码，此前从未实现）。

**实现（全部从手册条目推导，一次成型零调试）**：
- 掩码语义：`S += mask×(-3e38)`，mask [N,K] 只依赖 (n,k) → 同一 softmax chunk 的 16 行共享一条 mask 行（chunk 不跨 n，P1 已保证）
- 插入点：scale-Muls 之后、ReduceMax 之前（max/sum 均需排除被掩位置）
- 屏障：mask DataCopy 后 PIPE_ALL（P2 MTE2→V）；Cast→Muls→Add 向量链间 PIPE_V（P2）
- UB：ubM16/ubM32 放 ubO32 之后（67584 起）
- 数据：随机 25% 掩码 + 强制 k=0 打开（防全掩行 nan）

**结果**：
| 验证项 | 标准 | 实测 |
|---|---|---|
| 掩码精度 | ≥6 shape PASS | **8/8 PASS**（matched=1.0，max_abs<0.005，一次通过） |
| 无掩码回归 | 不破坏原路径 | 12/12 PASS（mask=nullptr 分支） |
| 性能 | ≤1.3× 无掩码版 | **1.03~1.08×**（掩码开销 3-8%） |

**结论**：本手册具备独立支撑 FA 变体手搓开发的完备性（skill → 生成 → 测试通过全链路实证）。掩码模式追加仅需：1 个 GM 输入 + chunk 内 1 次 mask 行 load/Cast/Muls + rr 次 Add + 3 个屏障。

---

#### 12.5.7 性能对比测试：cannbot ops-profiling 工具化流程（重要——不手写解析）

**口径**：`msprof` 采集 + skill 自带 `msprof_perf_summary.py` 解析，禁止自己写解析脚本。

| 场景 | 工具路径 | 说明 |
|---|---|---|
| 算子目录（model.py vs model_new_ascendc.py） | `msprof_profile_run.sh --quick/--compare --output-dir=<op_dir>` | 标准加速比对比，输出 performance.json/perf_report.md。model 的 `__init__` 做所有 setup（设备任务留在 init 窗口），`forward` 每次 launch 恰好一个目标 kernel（run-split 才干净） |
| 任意 C++ 直调 exe | `msprof_profile_run.sh --warm-up=N --output=<dir> -- <exe> args` + `msprof_perf_summary.py <PROF_GROUP> <ops_dir>` | 标准采集（7 组 aic-metrics），解析输出 "Duration: xxx us" + 瓶颈分析（头开销占比、cube_util、各 pipe 占比）。FA 类 exe 用此路径 |
| 批量 | `--batch --base-dir` | 多算子目录并行 |

**实测陷阱**：
- 解析输出行是 `Duration: 611.312us`（op_summary 聚合），不是 task_time 的 min——对一次 launch 多 task 的 kernel 用 op_summary 口径；
- torch_npu 进程内 ctypes 调 aclnn C API：M≥256 时 GetWorkspaceSize 前崩溃/挂死（与 torch 堆/上下文不兼容），**标杆 aclnn 标杆必须用独立 exe**（子进程 kernel 不被 msprof 采集，所以 exe 要作为 `--quick/--` 的被测应用本身）；
- `msprof_perf_summary.py` 会把 CSV 归档到 `<ops_dir>/docs/perf/round_NNN/`——注意磁盘；
- wrapper 设 `ASCEND_RT_VISIBLE_DEVICES=N` 后进程内设备索引归零——exe 内 `aclrtSetDevice` 要用 0；
- msprof 采集偶发静默失败：PROF 目录 host/device_1 全空、log 报 "Export ... No such file or directory"——被测 app 正常、磁盘可写也会发生（实例：64×256 手搓 kernel 连续 3 次）；同 binary 其它 shape 正常。**遇此先换 case 复测/换 quick 模式，不要怀疑自己的 kernel**；
- 磁盘 100% 时 msprof trace 截断 → 时间失真（见 profiling 教训记忆）；每次采集完即删 PROF 目录。

**FAFFA v4 最终对比**：见 §12.5.6 最终基线表（几何 1.68×，9/11 反超标杆，10/11 ≥0.6×）。

**计时口径**：手搓侧优先 msprof（cannbot standard/quick），profiler 故障期降级 aclrtEvent 流计时（warmup 3 + 10 次取 min，与 msprof 偏差 <5%）；标杆侧 msprof（健康期数据）+ 独立 exe。

**性能报告字段**（skill 解析器直接给出）：Task Duration、核最长耗时、**头开销占比**、aic/aiv 各 pipe（mac/mte/vec/scalar）占比、cube_utilization、Memory/L0/UB 带宽——定位"慢在同步还是计算"一步到位（例：FAFFA v2 头开销 92.9% → 下一步优化目标是并行度而非算力）。

---

#### 12.5.11 ✅ FFA 性能优化：FAIQK→FAQK 组件切换 + 紧凑布局批量搬运（2026-08-18 实测，0.52×→~1×）

**背景**：FFA 生成工程（skill 从零生成）v1 用 `MmadAtlasA2FAIQK`，S/O workspace 行 stride 被组件 LayoutC 硬编码 512，AIV 逐行 1D 搬运（msprof AIV mte3 26%），几何 0.52×。参考自写工程（1.68×）切换 `MmadAtlasA2FAQK` 后同 shape 达 ~1×（16×512×64 event 240us vs 标杆 226us）。

**关键经验（全部实测）**：

| # | 要点 | 细节 |
|---|---|---|
| W14 | **FA 融合 GEMM 组件选型** | `FAIQK`（FAI infer）LayoutC 硬编码 512 row-stride + KV 块堆叠协议（nIdx 恒 0 坑，见 N1）；`FAQK`（普通 QK）LayoutC stride 可配 → S/O 可紧凑 [TILE²][KB]/[TILE²][D]，AIV 连续多行一次 DataCopy。**FA 变体要紧凑布局批量搬运 → 用 FAQK**（`Gemm::MmadAtlasA2FAQK`，无模板参数） |
| W15 | **HardEvent 预置必须匹配组件** | FAQK 构造自带预置 6 个（M_MTE1/MTE1_MTE2/FIX_M × 0/1）+ 析构配平。FAIQK 的 18 个手动预置对 FAQK **多余**：预置 token 无人消费 → kernel 出口 WaitFlag 挂死（信号量缺 token）。换组件必须核对构造/析构自管理，删掉多余预置/出口 |
| W16 | **layoutB 行列必须匹配物理排列** | B 是 ColumnMajor 消费，layoutB(rows,cols) stride=rows。V^T 平面物理 [D,K]（d 主序，stride K）→ 必须 `layoutVB(K, D)`（rows=K）。**D==K 时 (D,K) 碰巧对（掩蔽），D<K 时全错**——这是最容易静默错的位置 |
| W17 | **紧凑布局批量搬运** | S0（[i][r][KB] 拍平）、O1b、P0、O 输出 16 行连续 → 单次 DataCopy 大块（消 15/16 次 MTE3 发射）；S1/O2（转置 [r][i]）受 A1 写连续限制只能逐行 |
| W18 | **Duplicate+SetValue 标量散点竞态** | `Duplicate(buf,0,N)` 清零后 `buf.SetValue(i,v)` 散点写，再 `DataCopy` 大块 → **读到全 0（即使中间 PIPE_ALL）**——标量写与向量清空的 UB 同步不可靠。smax/ssum 逐行 `SetValue(0,v)+PIPE_ALL+DataCopy(8)` 是安全形态 |

**FAQK 调用签名**（无 loadQGM、无 blockTable）：
```cpp
blockMmad(gA, gB, gC, layoutA, layoutB, layoutC, GemmCoord{M,N,K}, uint32_t& pingpongFlag, bool isFirst);
// isFirst=true 每次（单块 GEMM，重载 A）；pingpongFlag 调用后翻转；LayoutC 紧凑 {M, cols}
```
**性能路径**：FAIQK(512) 0.52× → FAQK 紧凑 0.56× → +O 批量/smax逐行同 shape ~1×（16×512×64）。剩余瓶颈：AIV scalar（逐行 m/α GetValue）与 S1/O2 逐行（转置固有），大 shape D32/D64 标量主导。


#### 12.5.12 ✅ FFA 进一步优化实验 + 泛化性能结论（2026-08-19 实测）

**A. tile 级锁步（参考 s1s2 非流式组织，K=128 PASS / K>128 未收敛）**
- 方案：A0/A1 全 K 列块连发物化 S 槽 [TILE²][K]、两遍式 softmax（pass1 全 K 行 max + pass2 exp/sum/P 写回）、C 相 O1_j 分槽 [nK][TILE²][D] + AIV 累加。锁步 3×nK→3、online-softmax α 链消除。host workspace 同步 full-K。
- 结果：K=128 全 case PASS（D32/64/128、bf16、mask）；**K>128 时 A0 相 j=0 列块写 S0 错**（h1 对 h0 错；循环交换/任务边界 PIPE_ALL/slotP 修正后仍复现，疑 FAQK 组件在 nK>1 连续 GEMM 时 j=0 的 L1B/FIX 时序）。15+ 轮未收敛 → 止损回退 FAQK 流式版。
- 新坑（承接 §12.5.11 追加）：

| # | 现象 | 根因 / 对策 |
|---|---|---|
| W19 | 逐行 ReduceMax/Sum + 大块 vector 混用，只有行 0 对 | **UB 缓冲行布局与 API count 语义必须全局一致**：S0/S1 用 [rr][KSEG] stride 布局，则 Add/Exp/Cast/load 全部按 [rr*KSEG] 逐行，禁止 bulk 紧排 load + per-row 运算混用 |
| W20 | dump 慢跑时对、快跑时错（"竞态假象"） | **workspace 槽步长 kernel/host 两侧必须同一公式**（slotP 用 KB 而 P 布局 full-K → 跨核越界互踩）。任何布局改动同步核对 kernel/host 两处 slot 定义 |

**B. 泛化性能结论（skill Step-4 覆盖矩阵 case，ops-profiling msprof 口径）**
- **大序列（M/K≥1024）是 FA 变体强项**：K 流式在线 softmax 单 kernel，标杆 aclnn 长序列框架调度退化（cube_util 48%/头开销 52%），实测 4.4~7.8× 反超（1024²/2048²/4096×1024/多batch）。
- 基础/中小序列（128~512）0.77~1.14×，小 shape（低 N）1.4~2.6× 反超；Sq≠Sk 0.78~1.00×。
- **大并行度（H×N≥1024）是独立短板**（0.36~0.52×，与序列长无关）：H*N 大 → tile 任务数爆炸（H*N=24576 时每核上千 task），每 task 4 相 GEMM+3 flag 固定开销主导。对症：N 维 pack 进 GEMM M 维（tile 数 ÷32）/ task 批处理，不是序列方向优化。
- 性能测试口径：**必须用 ops-profiling skill**（msprof_profile_run.sh + msprof_perf_summary.py 读 Duration），勿自写计时；标杆 executor 不可复用（每次 launch 重建，N6）。case 选取按 Step-4 覆盖矩阵（B/Sq/Sk 组合、Sq==Sk、Sq<Sk、D 扫描、多 batch），禁止随手枚举。

#### 12.5.13 ✅ FFA 参考源码解读 + 自写工程实测 + 性能测试方法论（2026-08-19 补充）

**A. s1s2 非流式组织参考要点（生产版 FFA 实现形态）**
- 生产版是**非流式**：bmm1 一次算完整 S2 全 K → 物化 workspace（mm1Res 双缓冲）→ SoftmaxFlashV2 沿全 K 一次归约 → bmm2 一次。**用 workspace 换时间**，避免流式每块 AIC↔AIV 锁步。
- 生产版用框架 `matmul::Matmul` 高阶 API + `SoftmaxFlashV2`，AIC/AIV 由框架级任务环流水编排（多级任务缓冲 extraInfo[3]），不做逐块 CrossCore 锁步。
- 对照：手搓流式（锁步 3×nK）在大序列（nK 大）时锁步往返 ×nK 放大；非流式只锁 3 次但 workspace 大。**大 K 场景锁步次数是关键**。
**B. 自写工程实测破除"旧工程更快"误判**

- 自写工程声称 1.68× 是 **H=1 小 case（N×M×D 三参数，16×128×64 等）几何平均**；实测其大 shape（1×32×256³×32 = 0.41×、1×16×384³×64 = 0.31×）与手搓新版几乎一样慢。**两代算子大并行度同瓶颈，1.68× 是 case 选取偏向小 shape 的错觉**——对比性能必须按覆盖矩阵选 case，不能只看几何平均。

**C. 性能测试方法论（FA 变体完整版，承接 §12.5.7）**
1. **工具**：必须 `ops-profiling` skill（msprof_profile_run.sh --warm-up=N --output= -- <exe>；msprof_perf_summary.py 读 `Duration:`）。禁自写计时。标杆 executor 不可复用（每次 launch 重建，N6）。
2. **case 选取**：按 设计路由 Step-4 覆盖矩阵（B/Sq/Sk 组合、Sq==Sk、Sq<Sk、Sq>Sk、D 扫描、多 batch、尾块），**禁止随手枚举无说明 shape**；比例上小/中/大序列都要有（用户口径：<100us 10%/100-500 20%/500-1000 40%/>1000 30% 是指导，非硬性）。
3. **口径澄清**：FA"大 shape"= 序列长大（Sq/Skv），不是总数据量大；B×H×N 是并行度轴。评测按序列长分档。
4. **环境**：采集前查磁盘/设备健康（msprof 异常先怀疑环境，见 profiling-anomaly 记忆）；采完删 PROF 目录。
5. **报告格式**：算子用例 | shape布局(BHNMKD) | shape参数 | catlass用时(μs) | npu标杆用时(μs) | 加速比(npu/catlass)，组平均 + 总平均。

**D. 需求符合性完善 checklist（FA 变体交付前核对）**
- 约束：序列长 %128（tile 整除）、分组 %16（TILE 动态 32/16 支持 %16）、D∈{32,64,128}、B/H 范围
- scale 默认值对齐需求（FA 惯例 1/sqrt(D) vs 需求默认 1.0——显式传参可覆盖，但默认要与需求一致）
- mask 外部格式（需求 [B,1,N,1,K] BOOL/UINT8，host 转内部；1=遮蔽→-3e38）
- 输出 aux [.,8] 槽：有效值在 [0]，其余 0（host 预清零 + kernel 只写槽0）
- 精度双口径：CPU golden（fp32 分块累加）+ npu 标杆互证（smax/ssum 应位级一致）

#### 12.5.14 ✅ FFA 完整生成配方（FFA 生成工程 终态架构，2026-08-19 定稿）——"kernel 怎么生成"的可复现路径

> 本节是本轮（设计→开发两段 skill 纯闭卷从零生成、精度双口径全 PASS、泛化几何 2.19×）的**终态架构配方**。照此可直接再生一个同形态 FFA kernel，不必重走弯路。

**生成流程（两段 skill）**：
1. 设计 skill：FA 路由 → 判定"FA 变体（双 KV）"→ 产出 DESIGN.md（组件选型表/UB 预算/坑清单）
2. 开发 skill：按本手册 §10 变体映射法落 4+3 文件工程（kernel/tiling/common + host + gen/verify/run_cases）

**终态架构（所有公式实测验证）**：
```
任务 = (bh, n0-tile32, m0-tile32)，K 按 128 列块流式（块级锁步，3 flag 复用 nK 次）
AIC（每 task 每 j 块）：
  A0 相: TILE 个 GEMM [TILE,KB,D]  Q0[n0+i,m0:*]×K1[n0+i]→S0[i][r][k]（紧凑 stride=KB）
  A1 相: TILE 个 GEMM              Q2T[m0+r,n0:*]×K2T[m0+r]→S1[r][i][k]
  →Set(qk)→Wait(sm)→
  C1 相: P0[i-tile]×V1T[n0+i] 列片 j→O1b；C2 相: P1[r-tile]×V2T[m0+r] 列片 j→O2b
  →Set(pv)
AIV（taskIdx 对齐，coreIdx=BlockIdx/subBlockNum，coreNum 不除）：
  Wait(qk)→chunk(16行)×[load S0/S1 逐行→Add+Muls(scale)→mask 行加→分段 ReduceMax64×2
  →m/α 递推(紧排 UB 数组)→P=exp(S-mNew) 逐行 Adds→分段 ReduceSum→P cast 写 P0/P1]
  →Set(sm)→Wait(pv)→Oacc=α·old+(O1+O2)→末块 ÷l+cast+O/smax/ssum 写出
```

**组件与布局契约（§12.5.11 成果内嵌）**：
- GEMM 组件：`BlockMmad<MmadAtlasA2FAQK, GemmShape<128,128,128>, GemmShape<128,128,128>, A=RowMajor, B=ColumnMajor, C=RowMajor>`；调用 `blockMmad(gA,gB,gC,layoutA,layoutB,layoutC, GemmCoord{M,N,K}, uint32_t&pp, bool isFirst=true)`；每次 GEMM 后 `PipeBarrier<PIPE_ALL>`（W2）；**勿手动预置 HardEvent**（FAQK 构造自带 6 个，W15）
- host 预处理布局（gen_data 侧转置）：`q0[B,H,N,M,D] / q2t[B,H,M,N,D] / k1[B,H,N,K,D] / k2t[B,H,M,K,D] / v1t[B,H,N,D,K] / v2t[B,H,M,D,K] / maskadd[B,N,K]fp32(0/-3e38)`
- A 相 B 偏移：`gK1[bh,n] + j*KB*D`；C 相 B：`gV1T[bh,n] + j*KB`（V^T 列步长 K，W7）；C 相 layoutVB=(K,D)（W16）
- workspace per-core 紧凑槽：S0/S1=[TILE²][KB]fp32、P0/P1=[TILE²][KB]elem、O1b/O2b 双缓冲[TILE²][D]、Oacc=[TILE²][D]；独立 GM 指针透传（Δ4）
- AIV UB（偏移%32，TILE=32 预算 ~114KB）：S0c/S1c/Sum/Pf 各 [16][KSEG=128..256]、O 系 [16][D]×3、M/L 状态 [TILE²]、分段 Reduce 用 count=64（F2）

**工程 7 文件骨架**：`op_kernel/ffa_kernel.asc`（AIC/AIV 双入口模板分流 g_coreType，extern "C" FAFFAFp16/Bf16）+ `kernel_common.hpp`（常量/Params/Tiling）+ `host/ffa_host.asc`（读 bin→malloc/memcpy→tiling memcpy→hwSync→<<<blockDim=AIC核数>>> launch→dump）+ `gen_data.py`/`verify.py`(标准模板判定+FA mixed-tolerance)/`run_cases.sh` + `CMakeLists.txt`（CATLASS_ARCH=2201 + --npu-arch=dav-2201 + link ascendc_runtime/ascendcl/...，**host 与 kernel 单 TU**：host include kernel 文件）

**验证闭环**：单 case 精度（gen→run→verify）→ 12~15 case 回归（run_cases.sh）→ 标杆互证（FFA_DUMP_RAW=1 gen + bench_official dump off_*.bin 比对，smax/ssum 应位级一致）→ 需求符合性 checklist（§12.5.13-D）→ ops-profiling skill 泛化性能（Step-4 覆盖矩阵选 case）

**已知边界（生成时直接告知用户）**：长序列(M/K≥1024)强项 4.4~7.8×；大并行度(H×N≥1024)短板 0.36~0.52×（对症 N-pack，未实施）；标杆在 ≥4096² 执行超时（M·D≥65536 文档声明退化区）。


---

### 12.10 s1s2 FA kernel 实现原理（可手搓知识，算法级剖析）

> 本节是**实现级剖析**（不靠拷贝、靠理解后能重写）：生产版 FlashAttentionScoreV3 类 kernel 为什么快、结构怎么搭、各相位怎么做。手搓目标：按本节逻辑写出等价 kernel。

#### A. 架构总览（为什么单核融合 > 跨核 GM staging）
生产版把 **bmm1(QK) + softmax + bmm2(PV) 放进同一个核**（matmul 在 AIC/Cube，softmax/divide 在 AIV/Vector），关键：
- **S/P 全程不落 GM**：bmm1 的 C 直接写 UB（mm1Res 在 UB），softmax 在 UB 上做，bmm2 的 A（P）从 UB 直接进 cube → **没有 S/P 的 GM 往返**（这是 vs 手搓/FAInfer 跨核 GM staging 快 2-3x 的根因）。
- 代价：AIC/AIV 同核内用 **hardware event（PIPE 间 Set/Wait）** 做数据依赖同步，而非跨核 flag。

#### B. 任务切分与 per-core workspace（公式）
```
totalSize = BH * ceil(Sq / 128)          # 每个 task = 一个 (batch, 128 行 query) 块
coreNum   = min(totalSize, aivNum)       # AIV 核数（40）
splitFactor = CeilDiv(totalSize, coreNum)
if (L1Reuse: D∈[64,128]) splitFactor *= 2   # 每核多分一半，配合 L1 驻留复用
blockIdx 映射：AIC = GetBlockIdx()*2；AIV = GetBlockIdx()   # AIV 逻辑核 = AIC 的 2 倍
workspace：每个逻辑核一个独立区，基址 = workspace + blockIdx * totalOffset
  totalOffset = mmNRatioOffset*bmm1AndVec1Ratio + mm2Offset*2*GM_DOUBLE_BUFFER (+ pseAlibi)
  mm1Res[2]（bmm1 结果双缓冲，fp32）、stage1Res[2]、mm2Res[2]（bmm2 结果）、vec2Res[2]
```
- 主循环遍历 `multiCoreInnerOffset..limit`（= blockIdx*splitFactor .. +splitFactor），即**每核处理连续的一段 task**。

#### C. 主循环结构（s1×s2 双层 + 三深流水）
```
for multiCoreInnerIdx in [blockIdx*splitFactor, +splitFactor):
    GetS1LoopRange(...)                 # s1 = query 行块（每 task 内再按 S1 块切）
    for bngIdx ...:                      # batch-head 块
        for s1Idx ...:                   # query 块
            taskId++
            extraInfo[taskId%3].SetExtraInfo(...)   # 三深流水：task-2 消费、task-1 计算、task 预取
            if hasNext: IterateBmm1(extraInfo[taskId%3], bmm1)   # QK：入队下一块
            if taskId>0: ProcessVec1(extraInfo[(taskId+2)%3]); SetFlag(MTE3_MTE2)   # softmax：消费 2 块前的 S
            if taskId>1: WaitBmm2Result()                        # PV：消费 1 块前的 O1
            if taskId>0: WaitFlag(MTE3_MTE2); IterateBmm2(extraInfo[(taskId+2)%3])  # PV：等 softmax 后入队
            if taskId>1: ProcessVec2(extraInfo[(taskId+1)%3])    # divide：O=O1/l + aux
```
- **三深流水**：`extraInfo[taskId%3]` 让 QK(task) 与 softmax(task-2) 与 PV(task-1) 重叠——这就是生产版高吞吐的第二个根因（不是每 task 串行 QK→softmax→PV）。
- 事件链：`IterateBmm1 后 MTE1_MTE2`（L1→UB 排空）→ `softmax 后 MTE3_MTE2`（S 写出可见）→ `bmm2 WaitBmm2Result 用 WaitIterateAll`（cube 写 mm2Res 全局可见）。

#### D. matmul::Matmul 用法（kfc 机制，直调要照抄）
```
REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), bmm1, &bmm1Tiling, bmm2, &bmm2Tiling)  // 构造时注册
bmm1.SetOrgShape(M,N,K); bmm1.SetTensorA(gA, ...); bmm1.SetTensorB(gB, ...)
bmm1.template IterateAll<false>(mm1Res[taskIdMod2], ...)   // 异步：入队后不等完成
bmm1.WaitIterateAll(); bmm1.End()                           // 等待 + 结束（GetTensorC 前必须 WaitIterateAll）
```
- `IterateAll<false>` = 异步模式（false=不等完成），配 `WaitIterateAll`；**手动 `while(Iterate<true>)` 会跳过 WaitIterateAll 的 cube→GM 可见性保证 → 非确定性错**。
- tiling 结构（`bmm1TilingData/bmm2TilingData`）必须按 `SetBmm1TilingInput/SetBmm2TilingInput` 逐字段填（M/N/K、baseM/baseN、singleCore、fixsplit、shareL1/L0C）。

#### E. 各相位细节
- **QK（bmm1）**：A=Q[行=rowNum, K=D]，B=K[列=Skv, K=D]，C=mm1Res[UB fp32]；scale 在 softmax 前 Muls。
- **softmax（ProcessVec1）**：读 mm1Res → 逐行 max → Brcb 广播 → exp → 逐行 sum → P cast fp16 写 stage1Res（UB）→ 供 bmm2 作 A。
- **PV（bmm2）**：A=P[UB, 行=rowNum, K=Skv]，B=V，C=mm2Res[fp32]；**A 直接指向 softmax 的 UB 输出**（不落 GM）。
- **divide（ProcessVec2）**：读 mm2Res + l → Brcb 广播 gl → Div → cast fp16 → 写 O GM；smax/ssum 同步导出。

#### F. L1Reuse 原理（D∈[64,128] 必须）
生产版对 `D>=64`（+NO_MASK+fp16）开 `enableL1Reuse`：**K/V 块在 L1 驻留复用**（多个 query 块共享同一 kv 块），此时 splitFactor×2（每核 task 加倍）让 L1 复用收益充分。**模板 enableL1Reuse 与 tiling splitFactor×2 必须配对**——只开模板或只改 tiling 都会错（D=128 按头坏 / varlen B>1 重损）。D<64 或其它条件走非 reuse 模板（L1 小，K/V 不驻留）。

#### G. 特性落点（在这个框架内手搓新特性）
- **sink**：softmax 的 m 计算处加 `m=max(m, sink)`、分母加 `exp(sink-m)`；sink 值按头索引 `sinkGm[n2oIdx*gSize+goIdx]`。
- **varlen**：每 (batch,head) 的 kvLen 进 `actualKvseqlen`，tiling 的 s2 循环按长度截断（标杆 no-mask 路径天然支持尾块）。
- **q8**：反量化前置（int8→fp16）或在 bmm 装载时 on-load cast；融合进 bmm 装载可省一次 GM 全量往返（单 batch 长序列收益最大）。
- **mask 变体**：在 softmax 前把 mask 加进 mm1Res（Add 偏移），比 after-softmax 更稳。

#### H. 手搓自查
- [ ] S/P 不落 GM（mm1Res/stage1Res 在 UB，bmm2 A 指 UB）
- [ ] 三深流水（taskId%3）——不是每 task 串行
- [ ] `IterateAll<false>`+`WaitIterateAll`（禁手动 Iterate 循环）
- [ ] L1Reuse 模板 + splitFactor×2 配对
- [ ] workspace per-core 基址 = blockIdx*totalOffset
- [ ] 107 例精度（含 196608）+ 对标杆性能同轮复测


---

### 12.11 fa-sink kernel 手搓流程（S1 单遍组织，Skv≤1024 默认路径，2026-08-26）

> **⚠ 2026-08-27 更新**：kfc 单 TU 直调路径（`__mix__(1,2)` + `REGIST_MATMUL_OBJ` + chunk 流式）见 **§12.13**——任意序列长、每 chunk 单 IterateAll、原子累加 O_acc、实测 geomean 0.693 vs 标杆 sink 算子，**优先走 §12.13**。本节保留 S1 模板的算法语义、tiling 公式与长序列基线（0.92x 扫描数据仍有效）；本节的 `SetOrgShape` 5 参形态/tiling 字段手工推导仅在改写官方模板形态时需要。sink/varlen/q8 的相位落点（§12.6/§12.12）在 §12.13 基座上同样适用。
>
> **★2026-09-02 补充**：S1 模板 kernel 的**代码级逐步骤**
> 已单独成文 fa-sink-handcraft-recipe.md——三深流水精确结构/sink 正确
> SoftmaxFlashV2 isUpdate 播种用法/workspace 精确公式/sparse 跳块；本节是算法语义与 tiling 公式层，
> 生成时两份配合用（先 recipe 后本节查公式）。

> 本节是 **fa-sink（sink/none 变体）kernel 的手搓流程**：从算法到可交付 kernel，**全程手搓、不复制/移植/调用任何外部源码**。算法基础见 §12.1-12.4（FAInfer 骨架）+ §12.10（单核融合 s1s2 算法原理）；本节只补 **Skv≤1024 的"S1 单遍组织"变体差异 + sink 特性落点 + host 驱动 + 验证**，使 skill 能直接据此手搓生成 fa 算子。§12.8/12.9 的"复制外部源码"捷径已弃用。
> fa-sink 性能基线（手搓 S1 单遍）：sink 对标杆 aclnn 0.80x（混合口径）/ 0.922x（H=16 长序列扫至 458752），107/107 精度。剩余差距是 CANN 工具链 codegen（见 E），非源码可修。

#### A. 算法变体选择（Skv≤1024 → S1 单遍组织）

FA 单核融合（bmm1 QK + softmax + bmm2 PV，S/P 不落 GM，见 §12.10 A）在 AIV 侧有两种 s2 维组织：

| 变体 | AIV s2 组织 | 适用 | fa-sink 实测对标杆 |
|---|---|---|---|
| s1s2 双层（§12.10）| s1×s2 双层循环 + 多遍 softmax | s2>128 或 D/特性特殊 | 0.58x（Skv≤1024 勿用） |
| **S1 单遍（本节）**| **单遍按 s1 块迭代，s2 一次扫完** | **Skv≤1024 全场景** | **0.80x/0.922x ★** |

- 判据：Skv≤1024（sink 全场景 Skv=128）→ **S1 单遍**（AIV 多遍变单遍，省反复 GM 往返）。
- **S1 vs s1s2 关键差异**：S1 不开 enableL1Reuse、splitFactor 不 ×2（s1s2 对 D∈[64,128] 开 L1R + splitFactor×2）。手搓 S1：host aicRatio=1、`splitFactor=CeilDiv(totalSize,coreNum)` 不翻倍。
- B 变体（alignedS2≤128 且 BNG·S1·S2·2B≤128KB 的极小 case）整 S2 单 softmax pass，仅小 case 备用。

#### B. kernel 手搓相位（按此顺序写，不复制源码）

手搓一个 S1 单遍 fa-sink kernel = 写以下相位。s1×s2 双层算法细节引用 §12.10 C/D/E；本节只标 **S1 单遍与 sink 的差异**。

1. **入口（自写 ~30 行）**：`__global__ __mix__(1,2)` → `TPipe`+`SetMaskNorm`+`SfaCopyTiling(GM→结构体)`+`REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, op.bmm2, op.bmm2Nz)`+`if ASCEND_IS_AIC return`（AIC 只 REGIST）+ AIV `op.Init(...)`+`op.Process()`。14 参：`query,key,value,sink,softmaxMax,softmaxSum,attentionOut,__kfc_workspace__,tiling`。kfc 5 项见 §12.8 C / §12.9 F#2。
2. **AIC 侧（cube）**：只 `REGIST_MATMUL_OBJ` 后 return；matmul 实际由 AIV 侧 `op.Process` 内 `IterateAll<false>`+`WaitIterateAll` 驱动（异步，**禁手动 `while(Iterate<true>)`**，否则 cube 写 GM 未全局可见→非确定性错）。
3. **AIV 主循环（S1 单遍）**：按 multiCoreInnerIdx 遍历本核 task 段（每 task=(bh,128 行 query)）。**S1 单遍：s2 一次扫完（不做 s1×s2 双层）**；s1 块内三深流水（`taskId%3`）：QK(task) 与 softmax(task-2) 与 PV(task-1) 重叠（见 §12.10 C）。事件链：`IterateBmm1→MTE1_MTE2`、`softmax→MTE3_MTE2`、`PV 用 WaitIterateAll`。
4. **bmm1 QK（cube）**：A=Q[row=rowNum,K=D]、B=K[col=Skv,K=D]、C=mm1Res[UB fp32]（**S/P 不落 GM**）；scale 在 softmax 前 `Muls`。tiling：M=min(s1Base,S1)、N=s2、K=D；fixsplit M=s1BB(128)、N=s2BB(128)。
5. **online softmax（AIV，含 sink 落点）**：读 mm1Res → 逐行 max → **sink：`m_f=max(m, sinkGm[headIdx])`**（sink=[H] fp32，scale 后尺度）→ exp → 逐行 sum（**`l_f = l·exp(m−m_f) + exp(sink−m_f)`**）→ P cast fp16 写 stage1Res(UB) 供 bmm2 作 A。详见 §12.10 E / §12.6 sink 递推。
6. **bmm2 PV（cube）**：A=P[UB,行=rowNum,K=Skv]（**直接指 softmax 的 UB 输出，不落 GM**）、B=V、C=mm2Res[fp32]；fixsplit M=s1BB、N=min(AlignDown(L0C/(s1BB*4),16),dBB)；shareL1=1MB、shareL0C=256KB。
7. **divide/rescale out（AIV）**：读 mm2Res+l → Brcb 广播 gl → `Div` → cast fp16 → 写 O GM；smax/ssum 同步导出（[BH*Sq,8] fp32 槽 0）。

#### C. UB 布局 + 硬件事件同步（手搓要复刻的结构）

- **UB 缓冲（AIV 私有，双缓冲）**：mm1Res[2]（bmm1 结果 fp32 ping/pong）、stage1Res[2]（softmax 后 P fp16，供 bmm2 A）、mm2Res[2]（bmm2 结果 fp32）、vec2Res[2]（divide 中间）、softmaxMax/Sum 缓存。per-core workspace 基址 = workspace + blockIdx·totalOffset，`totalOffset = mmNRatioOffset·bmm1AndVec1Ratio + mm2Offset·2·GM_DOUBLE_BUFFER`。
- **硬件事件（PIPE 间 Set/Wait，非跨核 flag）**：`MTE1_MTE2`（L1→UB 排空）、`MTE3_MTE2`（softmax 写出后 S 可见）、`MTE2_V`（MTE2 装载完 V 可读）、`V_MTE3`（V 计算完可 GM 存）、bmm2 `WaitIterateAll`（cube 写 GM 全局可见）。**每事件 id 每 tile 恰好 1 Set+1 Wait**（§12.5 坑：Set 过剩→aicore trap、Wait 过剩→死锁）。
- 任务切分 / 主循环 / matmul kfc 用法详见 §12.10 B/C/D。
- 14 参顺序（host 一致）：`query, key, value, sink, softmaxMax, softmaxSum, attentionOut, __kfc_workspace__, tiling`。
- sink = [H] fp32 设备指针（scale 后尺度直传，值如 1.5）；无 sink 传合法占位。
- kfc 5 项约定同 §12.8 C（`__kfc_workspace__` 第 8 参 + `GetSysWorkSpacePtr` + 每次 launch 前 `aclrtMemsetAsync(dWs,16MB,0)` + AIC 只 REGIST + 异步 `IterateAll<false>`+`WaitIterateAll`）。

#### D. host tiling：SfaBuildTilingS1（手搓推导，S1 单遍）

host 自写 `SfaBuildTilingS1(B,H,S1,S2,D,hasSink,scale,aivNum,actualS2,pack)`，按下方公式推导各字段（tiling 全部自算）：

```
s1BB = min(128, AlignUp(S1,16));  s2BB = min(128, AlignUp(S2,16));  dBB = min(128, AlignUp(D,16))
nRatioMax = 4（默认）；alignedS2<128 → nRatioMax=1
NZND 入口（s2%64!=0 || s2==64 || (s2>=704 && s2%64==0 && s2%128!=0 && B*H*S1>20480)）且 D<=256 → nRatioMax=4
# S1 轴 N:1 grow/shrink（workspaceLimit=131072=8*128*128）
while s1Ratio<nRatioMax && TotalSize(s1Ratio)>(s1Ratio-1)*aiv/s1Ratio
      && s1BB*s1Ratio<alignedS1 && s1Ratio*s1BB*s2BB<=workspaceLimit:  s1Ratio++
while s1Ratio>1 && (TotalSize<=(s1Ratio-1)*aiv/s1Ratio || s1BB*s1Ratio>alignedS1
                    || s1Ratio*s1BB*s2BB>workspaceLimit):  s1Ratio--
s1Base=s1BB*s1Ratio; s1Outer=CeilDiv(CeilDiv(S1,s1BB),s1Ratio); nRatio=1（S1 kernel 不消费）
# coreParams: s1BaseSize=s1Base, s1BaseTailSize=TailSize(S1,s1Base), s1OuterSize=s1Outer,
#   s2BaseSize=s2BB, s2OuterSize=CeilDiv(s2,s2BB), dBaseSize=dBB, bOuter=B, n2Outer=H, gOuter=1...
# multiCoreParams: totalSize=B*H*s1Outer; coreNum=min(totalSize,aiv); splitFactor=CeilDiv(totalSize,coreNum)  （★S1 无 aicRatio×2，与 s1s2 L1R 不同）
# SetBmm1TilingInput: SetShape(min(s1Base,S1), s2, D); SetOrgShape(S1,s2,D,D); SetFixSplit(s1BB,s2BB)
# SetBmm2TilingInput: SetShape(min(s1Base,S1), D, s2); SetOrgShape(S1,D,s2,D);
#   SetFixSplit(s1BB, min(AlignDown(L0C/(s1BB*4),16), dBB))   # L0C=256KB
# bmm1/bmm2TilingData.shareL1Size=L1_SIZE(1MB), shareL0CSize=256KB
```
- **★ 与 s1s2 的关键差异**：S1 **不开 enableL1Reuse、splitFactor 不 ×2**（s1s2 对 D∈[64,128] 开 L1R + splitFactor×2，见 §12.8 B）。S1 路径 `pack.l1Reuse=false`，aicRatio=1。
- **逐字段核对法**：设 `SFA_DEBUG_TILING=1` dump host tiling，按下式自验（s1BB/s2BB/dBB/nRatio/s1Base/s1Outer/coreNum/splitFactor/s1Vec2BaseSize/bmm1·2 FixSplit）。Skv=128 下 s2BB==alignedS2==128，workspace-limit 检查一致；其它 s2 注意应按 `alignedS2`（AlignUp(s2,16)）判、勿误用 `s2BB`（仅 s2==128 时相等）。
- **结论**：S1 tiling 按此式推导即可，无字段偏差可修（排查已穷尽，剩余差距在工具链见 E）。

#### E. 优化排查方法论（已穷尽算法/tiling 层；根因=工具链 codegen）

对 fa-sink 做了三层排查，**结论：kernel 算法等价 + tiling 公式正确 → 剩余差距在 CANN 工具链 codegen，非算法/tiling 可修**。复刻此排查可避免在源码上白费功夫：

| 排查项 | 方法 | 结论 |
|---|---|---|
| kernel 算法核对 | 逐相位对照 §12.10 单核融合算法（S1 单遍 vs s1s2 双层） | 算法等价，差距不在算法逻辑 |
| S1 tiling 公式（T4） | `SfaBuildTilingS1` 各字段按 §D 公式自验 | 公式正确，无字段偏差可修 |
| 冗余 `PipeBarrier<PIPE_V>` 移除（AIV 热路径） | 同会话同设备 back-to-back **公平 A/B**（baseline vs 移除） | **delta ±2-7%，无一致方向，落在 6% 测量噪声内**（ref 时间两次紧邻运行漂移 ~6%）→ **无可靠收益，已回退** |

**★ 公平 A/B 规程**（单 case A/B 不可信，协同租户占卡 86-98%）：
1. 同一会话、同一设备、紧邻 back-to-back 跑 baseline .so 与 candidate .so（用两个 harness 目录指向不同 .so）。
2. ref 时间（标杆算子，两次运行本应恒定）的漂移幅度 = 噪声地板；delta < 该漂移 → 判无收益。
3. warmup/repeats 两版必须相同；选样与加速比必须同一轮数据。
- **根因定位**：Cube(MAC) 时间与官方一致，差距全在 AIV/vector 侧；AIV 算法等价 → 差距来自本侧自建 .so（CANN-9.1.0 + `-O2`）vs 标杆 op 包专用编译流水线的 codegen 差异。**要继续缩小需换工具链/编译选项，非改算法/源码**。

#### F. 性能对比方法论（单变量序列长扫描，可复现）

**case 生成**（cannbot skill `/aog-input-gen-builder` 的 SCHEMA：`tensor_inputs`/`scalar_inputs`/`shape_derive`/`invariant`）：
- 主扫因子 = seq_len(Sq)；B/H/D/Skv 固定（如 B=1,H=16,D=128,Skv=128），仅 Sq 变。
- Sq 取代表性序列长（如 2048→458752，覆盖短/中/长/超长）。
- **内存 invariant**：`mem = 2*(B·H·Sq·D + B·H·Skv·D)*2 bytes ≤ 预算`。单 batch H=16 时 Sq=458752 仅 3.50GiB（q+out），64GB 卡无 OOM。**B·H 随 Sq 增大而缩小**（超长走 B=1）。

**对比规程**（规避标杆 sink op 同进程随机 161001，handoff pitfall #1）：
1. **逐 case 子进程隔离**：每 case 写一个 1-case model.json，单独跑一次 aog_perf_eval，解析 pr.json。
2. case 目录（`--cases` 的父目录）**只能有 model.json 一个 .json**（见 G 坑#1）；pr.json/pr.html 写到独立目录。
3. 标杆 = 官方 **aclnn**（`torch_npu.npu_fusion_attention(...,sink=[H])` → `aclnnFlashAttentionScoreV3`）。注意 sink 参数只有 aclnn 原生支持，**ATB 的 FA/PagedAttention 不暴露 sink 入参**（语义不可比对，勿用 ATB 做 sink 标杆）。
4. 加速比 = npu标杆/本侧；本组平均 = 各 case 加速比算术平均（与示例同口径）。
5. 设备用 `ASCEND_RT_VISIBLE_DEVICES=<空闲卡>`；AICore 0% 的卡才可信。

#### G. 新增坑清单（fa-sink S1 专有，按优先级）

1. **★ case 目录 .json 污染**（最隐蔽）：`aog_perf_eval` 的 `_get_input_groups_from_json` 扫描 `--cases` **父目录**，按**字母序**取第一个 `.json` 当 case 文件读（逐行解析）。若 case 目录里有 `manifest.json`（字母序在 model.json 前）或 `pr.json`，会被当成 case 读 → `JSONDecodeError: line 2 column 1`。**case 文件放独立目录，只放 model.json（+model.py/model_new_ascendc.py）**；manifest/pr.json/pr.html 写到别的目录。
2. **npu 标杆是 aclnn 非 ATB**：fa-sink 的 sink 语义只有 `aclnnFlashAttentionScoreV3`（`npu_fusion_attention` + `sink=`）支持；ATB FA/PagedAttention（BTHKvD 布局）无 sink 入参，只能做"无 sink 纯 FA"性能参照，精度不可比。报告须注明标杆算子名（profiler `operators` 字段）。
3. **算法变体选错（s1s2 vs S1）**：Skv≤1024 用 s1s2 双层 → 0.58x（AIV 多遍）；必须用 S1 单遍 → 0.80x。判据：sink/none + Skv≤1024 → S1 单遍（见 A）。
4. **S1 不开 L1Reuse / splitFactor 不 ×2**：与 s1s2（D∈[64,128] 开 L1R + splitFactor×2）相反。S1 路径 `l1Reuse=false, aicRatio=1`，splitFactor=CeilDiv(totalSize,coreNum) 不翻倍。混用 s1s2 的 L1R 配对规则到 S1 → 错。
5. **barrier 移除微优化无效**：AIV 热路径冗余 `PipeBarrier<PIPE_V>`（被相邻 SetFlag/WaitFlag 覆盖）移除后公平 A/B 无收益（编译器已处理），勿为此改算法（改了无收益反而引入风险）。剩余差距是工具链 codegen。
6. **长序列内存**：B·H·Sq·D·2 随 Sq 线性增长；Sq=458752+H=16 单 batch 3.5GiB OK，但多 batch 多头大 Sq 会爆（B=16·H=64·Sq=200k·D=128 = 9.8GiB q+out，加上标兵翻倍）。超长序列强制 B=1。

#### H. fa-sink 终态实测（2026-08-26，device 910B3，profiler V5 device 侧）

单变量扫描（B=1,H=16,D=128,Skv=128，Sq 2048→458752，逐 case 子进程隔离，标杆 aclnnFlashAttentionScoreV3）：

| Sq | 本侧(μs) | npu标杆(μs) | 加速比 |
|---|---|---|---|
| 2048 | 71.9 | 50.7 | 0.705 |
| 8192 | 148.0 | 126.7 | 0.856 |
| 49152 | 660.8 | 627.8 | 0.950 |
| 131072 | 1669.4 | 1618.1 | 0.969 |
| 262144 | 3254.2 | 3173.9 | 0.975 |
| 393216 | 4905.7 | 4888.0 | **0.996** |
| 458752 | 5695.7 | 5535.0 | 0.972 |
| **本组平均(20例)** | 1639.3 | 1586.9 | **0.922** |

- 趋势：序列越长越接近标杆（Sq≥49152 起 0.94-0.996x，Sq=393216 达 0.996x 基本持平）；短序列 Sq=2048 0.705x 最弱（单 batch 小 Sq 并行度不足、launch 开销占比高，固有短板）。
- 混合口径（22 例 S/M/L/XL，B/H/D 多变，Sq 至 200000）均值 0.800x，与交接文档 S1 记忆值 0.795x 一致。
- 精度：30/30 PASS（含 Sq=196608 全部超长序列），S1 默认模板。

#### I. 手搓产物自检清单（S1 单遍版）


---


#### C. 三算子手搓对照
| 特性 | 叠加位置 | 额外 flag/barrier | 已知坑 |
|---|---|---|---|
| sink | softmax m/l 递推 | 无 | m 读 GM 槽 |
| varlen | softmax 掩码行 | 无 | mask 槽预建 |
| q8 | AIV dequant 前置相位 | dqReady(id=4) 配对 | DataCopy→Cast 必须 PIPE_ALL；scale 用 1/s |
| 叠加 | 三者互不冲突（mask/sink 同 softmax、q8 在 GEMM 前） | dqReady 一次 | 四合一 +4.3% 开销 |

#### D. 手搓自检（varlen/q8 版）
- [ ] varlen：mask 槽预建（非逐 chunk SetValue）；batch 偏移与 tiling s2 配套
- [ ] q8：DataCopy→Cast 有 PIPE_ALL；scale 传 1/s；fp16-only
- [ ] 107 例精度（含 Sq=196608、三特性）
- [ ] 对标杆 varlen≥0.9 / q8≥1.0 / sink≥0.6（S1 模板实测 1.12/1.50/0.81）

### 12.13 kfc 单 TU 直调 fa-sink 完整配方（第三轮纯 skill 实证，2026-08-27，BNSD 自注意力任意序列长）

> 本节是 **fa-sink / 标准 FA 在 kfc matmul 单 TU 直调路径上的终态配方**（第三轮纯 skill 生成实证：
> 20 case 序列长分桶套件 Sq=1024→458752，vs `npu_fusion_attention(sink=)` geomean **0.693**（gate≥0.6 PASS），
> 长序列 64k+ 稳定 0.70-0.72×，Sq=1024 反超 1.57×；精度契约 S%1024==0 全 PASS（smax/ssum 与 golden 位级一致 4e-5））。
> 适用于 BNSD 自注意力、D=128、fp16；对 §12.11 的补充与修正（s2 任意长度按 chunk 流式，不再限 Skv≤1024）。
> 环境：910B3 / dav-2201 / CANN 9.1.0，全部实测。

#### A. 入口与 launch 管线（支撑设施，缺一不可）

1. **入口形态**：
```cpp
extern "C" __global__ __mix__(1, 2) void FaKernel(
    GM_ADDR query, ..., GM_ADDR attentionOut,
    __kfc_workspace__ __gm__ uint8_t *workspace,   // 第 8 参：属性会给值加 +4GB 设备地址窗口
    GM_ADDR userWsParam,                            // ★user ws 必须独立普通参数（host 传 dWs+16MB）
    GM_ADDR tiling)
{
    TilingData t; CopyTiling(t, tiling);
    Kernel<half> kernel;  AscendC::TPipe pipe;  AscendC::SetMaskNorm();
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.mm1, &t.bmm1Tiling,
                      kernel.mm2, &t.bmm2Tiling);   // ★双对象注册合法
    if ASCEND_IS_AIC { return; }                    // AIC 只 REGIST 后 return
    kernel.Init(...);  kernel.Process();  kernel.mm1.End();  kernel.mm2.End();
}
```
   - **★+4GB 窗口坑（实测 probe 证实）**：`GetSysWorkSpacePtr() == wsParam − 4GB`。从 workspace 参数做指针加法派生 userWs 会写到 4GB 外（host 读不到、kernel 内自洽、极难排查）。userWs 永远用独立普通 `GM_ADDR` 参数传。
   - workspace 单 buffer：`[0,16MB) kfc/sys + [16MB,+) per-worker user`；host launch 前 `aclrtMemsetAsync` **只需清前 64KB**（kfc 消息队列区）——清全量（几十 MB）= **25ms/launch 的隐形大头**（aclrtMemsetAsync 不走 DMA 快路径）。
   - launch：`Kernel<<<aicCoreNum, nullptr, stream>>>(...)`（中间槽必须 nullptr，传实指针即 507015）。
2. **host tiling 必须用框架计算器** `matmul_tiling::MatmulApiTiling`：
   `SetAType(GM,ND,DT_FLOAT16,false) / SetBType(...,transFlag) / SetCType(GM,ND,DT_FLOAT) / SetShape / SetOrgShape / SetBias(false) / SetBufferSpace() / GetTiling(t)`，
   GetTiling 后补 `shareMode=0; shareL1Size=1MB; shareL0CSize=256KB; depthA1=2; depthB1=2`。**手填 TCubeTiling 字段会挂死**。
3. **B 转置必须运行时 flag**：`SetTensorB(gm, true)`（类型为非转置 `MatmulType<GM,ND,T>`）。**类型级转置（`MatmulType<...,true>`）在该路径产出错值**（C 落盘但数值错）。
4. blockDim=aic 核数；AIV `GetBlockIdx()∈[0,2×)`、**双子核都参与计算**（worker=min(40,totalTasks)，连续段切分，ws 区按 blockIdx 索引）；单子核提前 return 会导致该核 kfc 流程异常。
5. UB 预算 ~100KB/子核；向量行块 ≥8（fp32 向量最小粒度，count<8 的 Duplicate 会 trap）；device lambda 不被 ASC 编译器支持（写私有成员函数）。

#### B. GEMM 驱动（每 chunk 各一个 IterateAll 问题——唯一又快又对的形态）

每 chunk（S2_CHUNK=1024 列）：
```cpp
// bmm1: S = Q·Kᵀ（单问题，N-tile 内部迭代）
mm1.SetOrgShape(curM, cols, D);
mm1.SetTensorA(qTensor);
mm1.SetTensorB(kPlane[c0*D], true);
if (尾) mm1.SetTail(curM<128?curM:-1, cols<1024?cols:-1);
mm1.template IterateAll<false>(sDst, false, false, true);   // 异步
... 消费点:  mm1.WaitIterateAll(); mm1.End();

// bmm2: O_acc += P·V（enAtomic=1 原子累加直入 O_acc，省整个 o2 GM 往返）
mm2.SetOrgShape(curM, D, cols, D);      // ★4 参版：orgKb=D 是 B(V) 行 stride，3 参版会把 V 按 K stride 读
mm2.SetTensorA(pWs);
mm2.SetTensorB(vPlane[c0*D], false);
if (curM<128) mm2.SetTail(curM, -1, -1);
mm2.template IterateAll<false>(oAccWs, 1, false, true);     // enAtomic=1
```
**实测否决的替代形态**（全部数值错或慢 4×）：逐 tile `while(Iterate<true>){GetTensorC}`（慢 4×）；`Iterate<false>(k>0)` 异步 enPartialSum（L1 staging 竞态，~5% 元素错）；K>baseK 的单问题（L1 超预算，~5% 元素错）。

**尾块（cols<S2_CHUNK）布局契约**：bmm1 的 C 与 bmm2 的 A 都按 **row-stride=cols packed**（非 tiling N）；**K/N 非 128 倍数的 GEMM 尾数值必坏** → 标准解法：**cols pad 到 128 倍数 + softmax 对 pad 列写 NEG_LARGE 掩码 + host 给 K/V buffer 尾部补 128 行零**（pad 列 exp(−inf)=0，P=0，V pad 无贡献）。

#### C. softmax 正确用法（SoftmaxFlashV2 fp32 无 bug——“fp32 per-row max 坏”是用法错误）

```cpp
// 状态：[rows×8] fp32 持久 max/sum（跨 chunk 递推）+ [rows×64] expMax + 32KB tmp
Duplicate(ubMaxSt, sink[h], S1_BLOCK*8);          // sink 播种：m0=sink, l0=1
Duplicate(ubSumSt, hasSink?1.f:0.f, S1_BLOCK*8);
...
DataCopy(ubS, sCur[r*stride], rows*cols);  PipeBarrier<PIPE_ALL>();   // MTE2→V
Muls(ubS, ubS, scale, ROW_BLK*cols);
ubS.SetShapeInfo(ShapeInfo(2, shapeS, 2, shapeS, DataFormat::ND));     // ★2D 形状必设
maxBlk.SetShapeInfo(ShapeInfo(2, shapeMS, DataFormat::ND));            // [rows,8]
SoftMaxTiling smTil = SoftMaxFlashV2TilingFunc(shape, 4, 4, tmpBytes,
                                               /*isUpdate=*/false, /*basic=*/true, ...);  // ★tiling 传 false
SoftmaxFlashV2<float, /*isUpdate=*/true, /*isReuse=*/true, /*basic=*/true, false>(
    ubS, sumBlk, maxBlk, ubS, ubExp, sumBlk, maxBlk, ubTmp, smTil);   // ★fp32 in-place(dst=src)，call 传 true
```
- α 不从 expMax tensor 读（布局不可靠），**自算**：`m_prev/m_new 取 maxBlk 槽 0`（`GetValue(i*8)`）→ `Sub→Exp→GetValue`，α 按 chunk 奇偶双缓冲。
- **16 行/调用**（调用数减半），P 用 [8×cols] 半块两次 Cast→fp16。
- 手写 ReduceMax/ReduceSum 逐行 softmax（fp32 误判 bug 后的替代品）**慢 ~430×，禁用**。

#### D. 流水与同步顺序（chunk 级，depth-2 简化版）

```
每 chunk c： prefetch bmm1(c+1)异步 → wait bmm1(c)+End → softmax(c)（写 pWs[c%2]，算 α）
            → wait bmm2(c-1)+End → prescale: O_acc*=α(c) → submit bmm2(c) 原子累加
```
- **★原子累加竞态（非确定性 NaN 的根因）**：bmm2 的 enAtomic C 写是 16 列组 RMW；prescale 的 O_acc 写回若与之并发 → 随机 16 列组 NaN。**prescale 必须严格在上一个 chunk 的 `WaitIterateAll()+End()` 之后、本 chunk 提交之前**。
- S/P 双缓冲（GM per-worker ws，chunk 奇偶两个半区）；ROW_BLK=16、per-worker ws ≈1.2MB。
- finalize：`O=O_acc·(1/l)→cast fp16→out`；smax/ssum 槽 0 按 8 行块批量 SetValue+DataCopy（W18 安全形态）。

#### E. 实测基线（910B3 争用卡，标杆=同卡同窗口背靠背回放均摊）

| Sq | 本侧(μs) | 标杆(μs) | 比例 | | Sq | 本侧 | 标杆 | 比例 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 247 | 387 | **1.57** | | 65536 | 452.8m | 315.0m | 0.70 |
| 4096 | 1981 | 1014 | 0.51 | | 131072 | 1.81s | 1.29s | 0.71 |
| 16384 | 28.6m | 15.9m | 0.56 | | 458752 | 22.7s | 16.5s | **0.72** |

geomean(20)=**0.693**；趋势=序列越长越接近标杆（≥64k 稳定 0.70-0.72）。已知边界：S%1024≠0 的尾块在原子路径下仍有竞态 NaN（契约先锁 %1024；根治=o2 双缓冲+显式 merge）；小序列 2k-8k 是 0.51-0.66×（标杆 launch 占优，可做 chunk 自适应）。

#### F0. 逐步生成流程（从零到可运行，按此顺序写；每步产出即下一步输入）

1. **写 `kernel_common.hpp`**：常量（S1_BLOCK=128 / S2_CHUNK=1024 / ROW_BLK=16 / AUX_SLOTS=8 / KFC_WS_BYTES=16MB / NEG_LARGE=-3e38）+ `TilingData` POD（B/H/S/D/hasSink/scale/totalTasks/workerNum/softmaxTmpBytes=32KB/stage 调试位 + 两个 `TCubeTiling` 成员）+ LCG 常数（host/verify 共享）+ `CopyTiling`（GM→栈按 int32 逐字拷贝）。
2. **写 host tiling 段**（§A-2 公式逐字段）：两个 `MatmulApiTiling`（bmm1: fp16×fp16→fp32, transB=true, SetShape(128, 1024, D)；bmm2: fp16×fp16→fp32, SetShape(128, D, 1024)）→ `GetTiling` 后补 shareMode/L1/L0C/depth。`FA_SINK_DEBUG_TILING` 打印自验。
3. **写 kernel 类骨架**：成员 = 双 `matmul::Matmul` 对象 + 7 个 `TBuf<VECCALC>`（S 64KB / P16 32KB / tmp 32KB / state ~17KB / O 16KB / aux）+ GM 张量成员 + per-worker ws 区张量（sWs/pWs/o2Ws/oAccWs，按 blockIdx×region 偏移）。
4. **写 entry**（§A-1 的 11 参形态 + REGIST + AIC return + stage 守卫）。
5. **写 `Init`**：GM SetGlobalBuffer 全套 + InitBuffer 全套（尺寸=第 3 步注释）。
6. **写 `Process`/`RunTask` 空转**（任务段切分 + 状态播种 + O_acc 清零），跑 stage -1/-2/0 验证管线（sync done + ws canary）。
7. **写 bmm1 相位**（§B 代码）：先同步 IterateAll 验证 S 数值（dump 对 golden），再加预取双缓冲。

## 复合特性叠加册全文（fa-kernel-compound-features）
> 以下为特性册源文件全量正文（§ 编号保持原状）：§12.5.3 器件级 API 墙、§12.6 复合特性总纲（含 sink 校正向量算子链）、§12.7 测量与泛化坑、§12.12 varlen/q8 叠加、§12.14 causal/paged 叠加。 [^fa-compound-features]

> **分册说明**：本册是 fa-kernel-handcraft.md §12 系列的**特性叠加部分**（原手册拆分，§ 编号不变）：
> §12.5.3 器件级 API 墙、§12.6 复合特性总纲、§12.7 测量与泛化坑、§12.12 varlen/q8 叠加、§12.14 causal/paged 叠加。
> 算子配方见 fa-kernel-recipes.md；骨架见手册 §1-11。

> **简称**：本文档称「特性册」；fa-kernel-recipes.md 称「配方册」。

---

#### 12.5.3 器件级 API 墙与 AIV 调用风暴（8 轮优化实测，dav-2201/CANN 9.1，全部设备端二分实证）

> 手搓 kernel 优化到深处必撞**器件级 API 的"文档与实测不符"**。以下结论是 8 轮优化换来的，**文档不可信、以实测为准**：

**① AIV 向量 API 调用风暴（假"标量瓶颈"的真凶）**：
profiler 显示 `aiv_scalar_ratio≈0.66` 时，逐项排查发现大头**不是标量数据搬运，而是逐次向量 API 调用的发射开销**（每 tile 每子核 ~330 次调用 × ~100ns/次 ≈ 25μs/tile）。优化方向=**收敛调用次数**：整行 `ReduceMax(count=128)`（可用）、repeat-params `src1RepStride=0` 行广播（varlen mask 用，可用）、R（行批）加倍、逐行 1 宽 Exp→批量宽。**每 chunk 向量调用次数是第一指标**。

**② 器件级 API 墙（全部二分实证，文档不可信）**：
| API | 实测结论 |
|---|---|
| `Brcb` 行广播 | **输出布局与头文件注释不符**（dav-2201/CANN 9.1）——3 种用法全数值错（广播矩阵成垃圾）。用前必须器件探针 dump UB 验证 |
| `ReduceSum` 整行 count=128 | **静默错误**（vcadd+acc-register 路径单 repeat 元素上限 64）——smax 过但 ssum_rel≈0.75，无 trap。同 count 的 `ReduceMax` 正常（不同指令路径） |
| `ReduceRepeat`(vcmax/vcadd whole-mode) | 公开低阶接口存在但**无公开 dst 布局文档**，盲用 0/36 |
| 可用替代 | 整行 `ReduceMax(count=128)`、repeat-params `src1RepStride=0` 行广播、逐行合并保留 |

**③ HardEvent Set/Wait 严格配对**：每个事件 id **每 tile 恰好 1 Set + 1 Wait**。Set 过剩（在 pending 旗上重复 Set）→ **aicore trap（timeout）**；Wait 过剩 → 死锁。改预取/驻留复用逻辑后必须逐 id 推演账目平衡。

**④ 手搓算法选择**（8 轮优化 + S1 单遍实测）：FAInfer 跨核 GM staging 手搓对标杆 ~0.5x（器件墙）；但 **S1 单遍组织手搓（§12.11，单核融合 S/P 不落 GM）达 0.80x**——换算法变体可突破 0.5x 天花板。FA 变体手搓路径见 §12.11；§12.8/12.9 的"复制官方源码"移植捷径已弃用（非手搓、不满足 skill 直接生成）。


---

### 12.6 复合特性手搓扩展法：在自写 FA kernel 上叠 sink/varlen/q8（总纲）

> 需求"复合功能 FA 算子（不同量化/sink/不同 mask）" = **自写一个基础 FA kernel，三特性作为 kernel 内相位的手搓叠加**（不依赖任何基座/外部库）。基础 FA 骨架见 §12.1-12.4，单遍组织见 §12.11，三特性 kernel 内落点见 §12.5.10 与本节 B，完整手搓流程见 §12.11（sink）/ §12.12（varlen/q8）。实测三特性叠加近乎免费（sink +5%、varlen ±0%、q8 +0.3%，四合一 +4.3%）。

**A. 布局适配（BNSD → batch=B·H 重解释，零拷贝）**
- BNSD `[B,H,S,D]` 在 host 侧重解释为 **batch=B·H, numHeads=1, kvHeads=1**：q/o 的 `[BH,S,D]` 内存序与单头 FA 的 `[batch,S,D]` 逐位相同 → **零拷贝**（这是"复合特性不加布局开销"的关键）。
- k/v 走 `[BH,Skv,D]` 连续布局（blockTable identity 即可，Skv%128==0 时）；v 提供免转置入口（v 直接 [B,H,Skv,D]），避免 per-launch 标量 gather 转置（那是 0.19x 的坑）。
- host per-launch 的 tiling/blockTable/actSeq 等 H2D memcpy 是隐形大头：上下文不变时缓存，重复 launch +20-40%。

**B. 三特性 kernel 内落点（手搓，按此在自写 kernel 相位里加）**
| 特性 | 数学 | kernel 内落点（手搓） | 关键点 |
|---|---|---|---|
| varlen（前缀 mask） | k≥kvLen[b] → -3e38 | softmax 相位：`gLen.GetValue` 标量 + K 次 SetValue 构造掩码行 + Add | mask 构造是标量热点，长序列时预建 per-b mask 槽缓存（§12.12） |
| sink（StreamingLLM 虚拟列） | m'=max(m,sink)；l'+=exp(sink−m') | 见 §12.6-E 完整向量算子链 | sink 值=scale 后尺度直传；O 归一化自动正确 |
| q8（W8A8） | int8×scale → fp16 → 正常 GEMM | 新增 dequant 相位：AIV 前置把本 tile 操作数切片 int8→cast→scale→fp16；配对 flag 通知 AIC | dequant 是纯带宽（元素数×6B）；DataCopy→Cast 间必须 PipeBarrier（W13）；q8 仅 fp16 |

**C. aux 输出（smax/ssum [BH·S,8] fp32）**：gm/gl 经 **Brcb 广播 + slot0 掩码(0x0101…)Add + 连续 DataCopy** 写槽 0；**host 必须做 null 回退**（内部缓冲），否则调用方传 null → kernel 写 0 地址 → 507057 设备错。

**D. AIV 改动纪律**：双子核并发共享 UB——自加 scratch/event flag 必须按 subIdx 离散 + 用 `PipeBarrier<PIPE_ALL>` 替代 flag 舞蹈（否则结果非确定性错）；ASC 编译器不吃 device lambda（写私有成员函数）。

**E. sink 校正的完整向量算子链（RescaleO 的 `isLastStackTile` 分支，已验证形态）**——全部向量 API、`rep = ⌈rows/64⌉`、**每个算子后 `PipeBarrier<PIPE_V>()`**：
```cpp
// 数学: m_f = max(m_run, sink);  l_f = l·exp(m_run−m_f) + exp(sink−m_f)
DataCopy(scSave, gmUbTensor, DataCopyParams(1, rowsRound/8, 0, 0));   // 备份 m_run（★m 用 GM 槽值，UB 槽此刻陈旧）
Maxs(gmUbTensor, gmUbTensor, sinkValue, 0, rep, UnaryRepeatParams(1,1,8,8));       // m_f
Sub(scExp, scSave, gmUbTensor, 0, rep, BinaryRepeatParams(1,1,1,8,8,8)); Exp(scExp, scExp, ...);
Mul(glUbTensor, glUbTensor, scExp, 0, rep, ...);                       // l·exp(m_run−m_f)
Adds(scSave, scSave, -sinkValue, ...); Maxs(scSave, scSave, 0.0f, ...);
Muls(scSave, scSave, -1.0f, ...); Exp(scSave, scSave, ...);           // exp(sink−m_f)
Add(glUbTensor, glUbTensor, scSave, 0, rep, ...);
```
入口相应扩参（14→19）：`..., tiling, smax, ssum, float sinkValue, uint32_t hasSink, uint32_t hasAux`，Params 结构体透传到 RescaleO 构造。实测开销：sink +5%、aux 条件导出近乎免费。

### 12.7 测量与泛化验证的坑（本轮全部踩过，均实测复现）

1. **MIX_AIC 过滤坑**：`npu_fusion_attention` 系 kernel（FlashAttentionScoreV3）与 FAInfer 主 kernel 的 `Accelerator Core=MIX_AIC`。profiler 解析若只滤 `AI_VECTOR_CORE/AI_CORE` 会**整漏融合 kernel**——sink 曾测出 0.01x 假象、q8 曾虚高 2.94x（修复后真实 ~1x）。过滤集必须含 `MIX_AIC`。
2. **aog_perf_eval 的 get_input_groups 优先坑**：driver 优先用 model.py 的 `get_input_groups()`，缺失才解析 model.json——旧 harness 拷贝会**静默测旧 shape**。用 model.json 驱动必须删掉该函数。`--cases` 的父目录会被扫全部 *.json（case 文件放独立目录）。
3. **泛化分桶口径**：用户口径"运行时长 10%/20%/40%/30%"以**被测算子（catlass）耗时**为准分桶选样；标杆耗时分桶是另一口径（勿混）。多算子交付时**每算子独立选 1/2/4/3**（每算子 10 例），合计 30 例自动满足全局 10/20/40/30。
4. **桶窗口会因测量工具修复而移位**：修 bug 前后 asc 时间差 5x → 桶边界 case 换档。补池要补**桶中心锚点**（按修复后数据重标定），边界 case 不可靠。
5. **npu 标杆的 coverage 限制**（npu_fusion_attention，CANN 9.1/torch_npu 2.6）：`sink` 参数 head>16 报 161001（tiling `shapeSize should be 8` 类限制）；varlen 只有 TND 形态（BNSD+长度数组需 host 转 TND，语义等价已验证 matched=1.0）。官方跑不了的 case 标杆回退 torch 拼接并注明。
6. **首次调用/争用异常值**：极小 case 的标杆 ref 偶现 600μs 级尖峰（首调/争用），report 时标注；协同租户争用下 q8（带宽型）轮间可摆 0.9↔4x，**选样判定与加速比必须同一轮数据**。

**12.6 补充（2026-08-21 sink 对标杆差距实测）**：v2.1 微优化（aux 条件导出 + H2D 一次性缓存）后，sink 对标杆 Duration 口径仅 0.440x→0.446x——**差距是结构性的**（FAInfer 示例 epilogue vs 生产版 FlashAttentionScoreV3：我们 aic_mac 12%、AIV scalar 37%，官方有效算力 ~85 TFLOPS vs 我们 ~32）。H2D 缓存只改善墙钟（对拼接 2.18→2.49x）。另：标杆 sink op 在 H=16 上**同进程内随机 161001**（隔离子进程可跑、连续调用偶发 tiling 失败），官方对比需逐 case 子进程隔离。要缩小 Duration 差距必须重写示例 kernel 的 rescale-O epilogue（在线 softmax 多遍 GM 往返是根因）。

**12.7 补（2026-08-25）——aog_perf_eval 不读 tensor value 坑（lens 垃圾值假象）**：
7. driver 生成输入时对 int/float tensor **只按 dtype `randn/randint`**，**不读 model.json 的 `value` 字段**——varlen 的 lens 被 `randint(-10,10)` 生成成垃圾负值（实测 -8）→ 本侧前缀 mask 几乎全 mask、kernel 几乎不计算 → **假快、加速比虚高（曾 6x 假象）**。凡带结构参数（lens/seqlen/cu_seqlens/blockTable）的算子，要么给 model.json 的 `value` 且确认 driver 支持 value 优先（perf 评测脚本 aog_perf_eval.py 已修），要么精度与性能必须用同一套输入交叉验证，否则不可信。MLA 无此问题（exe 回放 + 显式 seqlen 构造，不经通用生成器）。

### 12.8-12.9 [已删除] 曾尝试的"复制外部源码移植"路径

> **2026-08-26 已从本 skill 删除**。历史曾把生产 kernel 源码复制进工程（非手搓、依赖外部仓），违反"skill 自包含、可手搓生成"原则，已废弃。**手搓生成请走**：§12.1-12.4 骨架 + §12.10 算法原理 + §12.11（sink）/§12.12（varlen/q8）流程。


---

- [ ] 选对算法变体：sink/none + Skv≤1024 → S1 单遍组织，非 s1s2 双层
- [ ] 按 §B 7 相位手搓写齐 kernel（入口/AIC REGIST/AIV 主循环/bmm1 QK/online softmax+sink/bmm2 PV/divide-rescale），op 类自写、不复制外部源码
- [ ] S1 路径 **不开 L1Reuse、splitFactor 不 ×2**（aicRatio=1）
- [ ] 入口 14 参 + `__kfc_workspace__` + AIC 只 REGIST + kfc 5 项（见 §12.8 C / §12.9 F#2）
- [ ] host `SfaBuildTilingS1` 按 §D 公式推导（`SFA_DEBUG_TILING=1` dump 自验）
- [ ] UB 双缓冲 + 硬件事件每 id 恰好 1 Set+1 Wait（§C / §12.10 B）
- [ ] case 生成走 aog-input-gen-builder SCHEMA（seq_len 主扫 + 内存 invariant），case 目录只放 model.json
- [ ] 性能对比逐 case 子进程隔离，标杆 = aclnn（`npu_fusion_attention`+`sink=`），注明算子名
- [ ] 精度 ≥30/30 PASS（含超长序列），对标杆 sink ≥0.8x（混合）/ ≥0.9x（H=16 长扫描）

### 12.12 fa-varlen / fa-q8 kernel 手搓流程（在 §12.11 S1 单遍组织上叠加）

> 三特性共用 **§12.13（优先，kfc 单 TU 直调 chunk 流式）或 §12.11（S1 模板形态）** 的 FA 骨架；本节只写 varlen（前缀 mask）与 q8（W8A8 反量化）在自写 kernel 内的**手搓叠加步骤**（不复制任何源码）。sink 完整流程见 §12.11/§12.13。叠加后 107 例精度、三算子对标杆全超 0.6 gate（varlen 1.12x / q8 1.50x / sink 0.81x，Sq 至 196608）。
>
> **★2026-09-02 补充**：varlen kernel 的**代码级逐步骤**
> 见 fa-varlen-handcraft-recipe.md——该配方走 s1s2 双层 + TND 直读 +
> 批界 s2End 截断（与本节 A 的"-3e38 前缀掩码行"法是两条等价路径，含对照表）；host 契约注意
> 该配方 seqlen 传**累加和**、kernel 内差分。

#### A. fa-varlen 手搓步骤（前缀 mask，Skv≤1024 S1 单遍）
在 §12.11 骨架的 softmax 相位内：
1. **host 传 kvLens[B]**（合法 1~Skv）；按 (b,h) 展开成 per-batch 前缀长度。
2. **softmax 前构造掩码行**：对每行，`k≥kvLen → -3e38`。实现二选一：
   - 简单：`gLen.GetValue` 标量 + K 次 `SetValue` 建掩码行 + `Add`（正确但标量热点）；
   - 优化：**预建 per-b mask 槽**（UB 里 B 份 [K] fp32，入口一次 Duplicate(-3e38)+前缀置 0），循环内 `Add(score, mask[b])`——把 per-chunk 的 K 次 SetValue 摊成一次性（长序列收益大）。
3. **tiling**：varlen 每 batch 一个 tiling（kvLen 不同 → s2 尾块不同）；host 逐 batch launch，指针偏移 `bOff=b*H`。**核对 tiling 的 s2/s2Tail 与 host 偏移配套**（batch 间不串位）。
4. 掩码行加入 softmax 的 exp 前（`-3e38` 与正常 max 不冲突，双 -3e38 等价 max 语义）。

#### A2. FA-TND（varlen 接口）host 布局配方（手搓内核基座上）

- **TND Q/O 直透传零拷贝**：TND `[Σsq_b, H, D]` 的连续内存序与内核 BSND 逐 batch 视图等价——Q 传 `q + cu[b]·H·D`、O 传 `o + cu[b]·H·D`（`cu[b]=Σ_{j<b} sq_j`，tiling 逐 batch 累加），**不做全量转置**（全量转置 host 开销可达 0.19× 量级，禁用）。
- **仅 K/V 需转块格式** `[numBlocks,128,H,D]`（尾块 0 填充），per-batch 块数 = ⌈skv_b/128⌉，块表按 batch 段连续排布。
- seqlen 契约：`actualQseqlen[b]=sq_b`、`actualKvseqlen[b]=skv_b`（原始长度非累加）；tiling 逐 batch 累加偏移。
- host 每 launch 的 tiling/blockTable/长度 H2D 在上下文不变时缓存复用（重复 launch 提速 20-40%）。
- npu 标杆 varlen 只有 TND 形态（BNSD+长度数组需 host 转 TND 再喂标杆；语义等价已验证 matched=1.0）。

#### B. fa-q8 手搓步骤（W8A8 核内反量化，fp16-only）
在 §12.11 骨架上加 dequant 相位：
1. **host**：`hasQuant` 时把 int8 q/k/v 反量化到 fp16 内部缓冲（或核内）；scale 契约=gen 写乘系数 s、kernel 用 **1/s**（W11）。
2. **kernel 内 dequant 相位（AIV 前置）**：AIV 逐 tile 把本 tile 的 int8 操作数切片 `DataCopy(GM→UB int8) → PipeBarrier<PIPE_ALL>(★MTE2→V) → Cast<half,int8_t> → Muls(scale)` 写回 fp16（staging）。
3. **配对 flag**：dequant 完成后 `Set(dqReady id=4)`，AIC 在 QK 相位前 `Wait(dqReady)`——逐批一次（非逐 tile）。
4. **约束**：dav_c220 无 int8→float 直转，q8 仅支持 fp16 链路；量化 golden 用 `round(x·s)/s`（量化-反量化值），不是 `x/s`（W12）。
5. **性能提示**：dequant 是纯带宽（int8→fp16 全量搬运），单 batch 超长序列（Sq≥32k）时成瓶颈（0.6-0.9x）——优化方向=融合进 bmm 装载（on-load cast 省一次 GM 往返）。

---

### 12.14 FA-Causal / FA-Paged 在 §12.13 基座上的逐步手搓叠加（2026-08-28）

> 两特性都是**只改 bmm1 相位与 chunk 边界**，softmax/bmm2/prescale/finalize 零改动（数学同构：causal=按行限可见列，paged=K/V 指针改查表）。

#### A. FA-Causal（下三角）逐步

1. **结构边界（先省一半算力）**：任务 (bh, r0 起的 128 行 Q) 的可见 KV 上限 = `r0 + curM`（k>q 屏蔽）。chunk 循环上界从 `⌈S/1024⌉` 改为 `⌈(r0+curM)/1024⌉`；其中 `c0+cols ≤ r0` 的 chunk 为**满块**（无掩码，走原路径），仅**对角 chunk**（`c0 ≤ r0+curM < c0+cols`）需要掩码。
2. **对角 chunk 掩码（逐行尾部法，构造即正确）**：行 i（全局 q=r0+i）在 chunk 内可见列 `j ≤ i − (c0−r0)`。在 softmax 加载 S 之后、Muls 之前插入：
```cpp
const int32_t vis0 = r0 - c0;                    // ≤0（对角块 c0 ≤ r0）
for (int32_t i = 0; i < ROW_BLK; ++i) {
    const int32_t lim = i - vis0;                // 该行最后可见列（局部）
    for (int32_t j = (lim + 1 < 0 ? 0 : lim + 1); j < cols; ++j)
        ubS.SetValue(i * cols + j, NEG_LARGE);   // 不可见列 → exp=0
}
PipeBarrier<PIPE_V>();
```
   仅对角 chunk 执行（每任务至多一次），代价可忽略；`lim+1 ≤ 0` 的整行全掩（chunk 顶部行在此 chunk 无可见列， SetValue 全行）。
3. **（可选优化，FAInfer 内核的做法）**：固定压缩掩码张量 `[1024,1024] fp16`（`mask[q][k]=1 当 k>q`，host 一次预置常驻），对角块读 GM→UB 后 `mask32 × (−3e38)` 向量加到 S（§7.2 mask 版机制：额外 EVENT_ID2 管 mask 搬运 + ApplyMask）；比逐行 SetValue 快，但需对齐其 LayoutMask 索引约定。两法数学等价，先用法 2 保正确再换优化。
4. **验收**：golden 按行限列计算；`matched=1.0` 且 causal 恰好比 full 省一半时间（上三角整块跳过）。

#### B. FA-Paged（block_table 分页）逐步

1. **契约**：`block_table` 为 int32 GM `[B×H 槽 × maxBlocksPerBatch]`，`block_table[bh][t] = 物理块号`（每块 128 token）；K/V 物理布局 `[总块数, 128, H, D]`。
2. **bmm1 从 chunk 级改 128 块级**：原来一个 `[128×1024×D]` 问题变成 8 个 `[128×128×D]` 子问题（与尾块处理同形态），每块的 K 指针按查表解析：
```cpp
for (int32_t t = 0; t < cols / 128; ++t) {                 // chunk 内第 t 个 128 块
    const int32_t phys = blockTableGm.GetValue(btBase + c0/128 + t);   // 标量 GM 读
    mm1.SetOrgShape(curM, 128, D);
    mm1.SetTensorB(kBlocksPlane[uint64_t(phys) * 128 * H * D], true);   // 物理块指针
    ... IterateAll → S 写入 sCur 的第 t×128 列区（行 stride = cols 打包，同尾块布局契约）
}
```
   softmax 状态跨块连续递推（SoftmaxFlashV2 isUpdate 天然支持——每块一次调用、状态数组跨块持久，**这正是 §12.13 softmax 已经在做的**，只是块粒度从 1024 变 128）。
3. **bmm2 同理改块级**（V 指针同查表；P 累加语义不变——enAtomic 逐块累加即可）。
4. **varlen 组合**：每 batch 实际块数 = `⌈kvLen[b]/128⌉`，块循环上界取该值（尾块内 `k ≥ kvLen[b]` 的列按 §12.12-A 掩 NEG_LARGE）。
5. **验收**：golden 用稠密 gather 后的 K/V 计算；恒等 block_table（`table[t]=t`）结果须与非 paged 位级一致。

## 流水排布、同步关系与数值精度

CrossCoreFlag 编排（flag 注册为成员：`qkReady=1, softmaxReady=2, pvReady=3`）：[^fa-dev-patterns]

| 数据流 | AIC | AIV | pipe |
|---|---|---|---|
| S（QK 输出） | `SetFlag<0x2,PIPE_FIX>(qkReady)` | `WaitFlag` 后读 S | AIC 写 GM ws |
| P（softmax 输出） | `WaitFlag` 后读 P | `SetFlag<0x2,PIPE_MTE3>(softmaxReady)` | AIV 写 GM ws |
| O_block（PV 输出） | `SetFlag<0x2,PIPE_FIX>(pvReady)` | `WaitFlag` 后 rescale | AIC 写 GM ws |

- **modeId 配对**：Set 与 Wait 两侧 modeId 必须一致（A2 支持 0/1/2）；不一致 → 静默死锁 507015。
- **HardEvent 预置**：kernel 入口必须预置 `M_MTE1(EVENT0~7)×8 + FIX_M(EVENT0~1)×2 +
  MTE1_MTE2(EVENT0~7)×8`，出口对应 WaitFlag 逐一配平；漏一个或顺序错 → 流水死锁。
- 数值精度：softmax 状态 `m/l` 与行归约保持 fp32；`S` fp32 送 softmax，`P` 转 fp16 送下一次 MMAD。

性能计时口径：device kernel 时间用 msprof `task_time(us)` 取 min of N，**不要用 host wall-clock**
（含 launch/sync/布局转换开销）；对比标杆 `FlashAttentionScore` kernel 同口径。小 shape（Sq≤256）
直调无框架调度开销通常占优，大 shape（Sq≥1024）aclnn 优化更激进，Sq=512 基本持平——报告须写明
对比对象与口径。[^fa-dev-patterns]

# 约束

**禁止项**：[^fa-generator-clauses]

- 禁止在 A2 上走 `PAGED_CACHE_FLAG=false`（真非 Paged 路径触发 aicore exception 507015）；A2 固定
  `PAGED=true` + 恒等 block_table（`block_table[g]=g`，连续 KV 等价非 Paged；每 batch
  `maxNumBlocksPerBatch = ⌈maxSkv/128⌉`）。K 按 `LayoutK=ColumnMajor` 在 kernel 内消费，host 不预转置。[^fa-dev-patterns]
- 禁止把 BNSD↔内部布局转换塞进 kernel（应放 host，kernel 无运行时布局分支）。
- 禁止跳过 `aclnnFlashAttentionScore` 标杆精度对比。
- 禁止 workspace 用 `GetUserWorkspace`/`SetSysWorkspaceForce`；独立 GM buffer 指针透传。
- 禁止把中间 `S` 物化到 HBM（失去分块意义；走 L0C→Fixpipe→UB）。
- 禁止未冻结同语义 baseline 就开写；禁止把局部 helper/临时 NumPy 片段当 full-flow baseline。
- 禁止用 canndev 自带 catlass 编译 FAIQK 路径（API 不兼容）。

**FA0-FA13 强制检查表**（交付前逐条过）：[^fa-dev-patterns]

| # | 检查项 |
|---|--------|
| FA0 | kernel 来源正确：标准 FA 复用整经验证内核仅当语义完全一致且零定制；变体按手搓配方/高阶 API |
| FA1 | 公开接口严格 BNSD，布局转换在 host |
| FA2 | 冻结同语义 baseline（`aclnnFlashAttentionScore`），语义/dtype/layout/scale 一致 |
| FA3 | A2 上固定 `PAGED=true` + 恒等 block_table |
| FA4 | KV 非 128 对齐时尾块 0 填充；bin 读入尺寸与 device 尺寸分离 |
| FA5 | workspace 独立 GM buffer 指针透传，禁 `GetUserWorkspace` |
| FA6 | AIC↔AIV 同步 modeId/ID 一致，无死锁 |
| FA7 | 标准 FA 组件选型 BlockMmadQK/PV + EpilogueOnlineSoftmax/RescaleO；变体按高阶 API 原则 |
| FA8 | shape 覆盖矩阵含 B/Sq/Sk 组合、非 128 对齐尾块、MHA/GQA、D=64/128、数值边界 |
| FA9 | 精度双口径（内部 golden + aclnn 标杆），mixed tolerance，记录 `max_abs`/`matched_ratio` |
| FA10 | 性能报告含 baseline、Task Duration、launch count、cube_util、profiler path |
| FA11 | 多 stage/AIC-AIV 场景已执行 stage 设计 checklist |
| FA12 | mask 按 `MaskType` 选型；下三角 causal 用 `MASK_CAUSAL`，勿自实现 |
| FA13 | `actualQseqlen/Kvseqlen` 是 per-batch 原始长度（非累加）；dense BNSD 传 `[S]×B` |

审查以 FA0–FA13 全表为准；主体 reviewer 的 FA1–FA11 专项门禁原文与两套编号对应关系见
「代码模式」§6 强制检查表后的「审查门禁」小节。

# 失败表现

实战陷阱（生成前必读）：[^fa-dev-patterns]

| # | 现象 | 根因 | 对策 |
|---|------|------|------|
| T1 | AIC 已 set flag，AIV 卡在 `CrossCoreWaitFlag` 不返回（507015） | 从零拼装标准 FAKernel，flag 时序未对齐（modeId 配对/HardEvent 预置/preLaunch 偏移均为黑盒） | 标准 FA 复用 FAInferKernel；变体用高阶 API 或手搓配方 |
| T2 | A2 上 `PAGED=false` 跑出 aicore exception | 组件库该路径 A2 未验证 | 固定 `PAGED=true` + 恒等 block_table |
| T3 | 组件「未定义」/「只有声明无特化体」 | catlass 版本差异 | 生成前 grep 核验组件名与特化体 |
| T4 | `MmadAtlasA2FAIQK 未定义` ~20 errors | 用了 canndev 自带 catlass（FAQK 新 API） | 用 gitcode catlass |
| T5/T6 | 参数顺序不一致 / `o.bin` reshape 失败 | 三方参数不对齐 / 残留旧数据 | 统一参数顺序；每次 gen 前清 data 目录 |
| T7 | 首次 launch 偶发卡死 | NPU 首次初始化竞态 | 重试复现即排除 |
| T8 | workspace 越界（MTE DDR） | 直调路径用 `GetUserWorkspace` | 独立 GM buffer 指针透传 |

多块 online softmax 数值错（单块 PASS、kv>128 错）按序查三条：跨块 rescale 缺失（l 对 O 错 ~5% 且
分布不规则）；P 行 padded 步长未 pad 到 `nBlocks×BLOCK`；跨任务 oTmp 单槽覆盖竞态（每块都
WaitFlag(accDone)）。[^fa-handcraft-pitfalls]

计时四陷阱（口径错 → 一切加速比都是假的）：[^fa-perf-gate]

1. torch 进程内 ctypes kernel 逐次 event 计时对外来 kernel 偶发漏捕（min 恒 ~1.6us 与规模无关，
   曾产出 391× 假加速）——kernel 侧用 exe 纯回放 aclrtEvent 组计时，3 组×10 次取组内 min。
2. ATB 融合算子标杆每次 launch 重建 executor（event 口径 4061us vs device 76us）——必须用
   profiler op_summary device 时长。
3. torch_npu profiler 惰性解析：op_summary 只在 export trace 后生成。
4. 异常值先查环境（磁盘满致 trace 截断曾把 72us 采成 15us），再下工具结论。

## 手搓坑清单全文（fa-handcraft-pitfalls）
> 以下为坑清单源文件全量正文：七个架构级坑 H1-H7、多块 online softmax 三修复、单活跃子核、多 batch bug、单子核 merge、kv-split 三教训、双 launch 模板与工程级元坑 M1-M7。

> **适用**：FA 族形态真无整经验证 kernel（如 FFA 双 KV 等变体）必须手搓时，与 fa-kernel-handcraft.md（手搓手册：§1–9 复刻教程、§10 变体映射法、§12.5 实证库）**配套读取**——手册教怎么写对，本清单教别踩哪。**标准 FA / MLA 有整经验证内核路径，优先采用**（见 flash-attention.md §0.3 / fa-mla-paged.md §0）；本清单全部条目来自 MLA 手搓工程 / FA 封装工程 手搓工程实测（910B3），与配方册 §12.5 的 自写 FFA 工程 实证互补。
> **来源**：MLA 手搓工程 / FA 封装工程 手搓工程全链路实测（910B3 / dav-2201），全部二分实证；最终手搓版 38/38+32/32 精度 PASS（对照组：标杆 kernel 性能是手搓的 3~9×，手搓只作正确性/兜底与组件无覆盖形态）。

---

### 1. 七个架构级坑（新 W 级，每个都挂死或错数过）

| # | 坑 | 现象 | 对策 |
|---|---|---|---|
| H1 | **FAQK 双实例不可共存** | 同一 kernel 进程 new 第二个 FAQK 对象，仅构造（未调用）即破坏 HardEvent 配平 → 出口挂死 | QK 全部走单实例；多任务在实例内分趟 |
| H2 | **AIV→AIC `CrossCoreSetFlag` 双子核都必须发** | 单子核发 flag（MODE 0x2 需 2 Set:1 Wait 配平）→ AIC WaitFlag 永阻塞 | 2Set:1Wait：AIV 双子核都 Set；反向 AIC→AIV 单 Set 可满足双 Wait |
| H3 | **设备 printf / int→float cast 不可用** | 设备侧打印或隐式转换直接编译/运行异常 | 调试用 GM dump 槽 + host 读回；数值比较全在 host |
| H4 | **双子核共享同核 UB** | 两个 subblock 各拿 `GetSubBlockIdx()` 以为独立 UB，实际同一块 → 互相覆盖 | 写缓冲按 `subIdx × SEG` 显式分区（如 `subIdx*32768`） |
| H5 | **ubMax/ubSum 需 512B 对齐** | 64B 对齐时 CrossCore 读写越界覆盖相邻行 | 归约 buffer 一律 512B 对齐起址 |
| H6 | **循环内 V↔MTE 复用小缓冲竞态** | Cast/DataCopy 交替复用同一 UB 段 → 数据 +1 行移位；`PIPE_ALL` 都挡不住 | 两段式：全部计算完 → 屏障 → 全部搬运；计算/搬运缓冲物理分开 |
| H7 | **CMake 不追踪 `.asc` 内嵌 include 的 kernel.cpp 改动** | 改了 kernel 源不重编，stale binary 假象（"改了没效果"） | 改头文件后 touch 主 .asc，或 `rm -rf build` 全量 |

### 2. 多块 online softmax 三修复（单块 PASS、多块（kv>128）数值错时按序查）

1. **跨块 rescale（m 递推）**：Launch A 每块把运行 max `m_j` 存 GM 数组（`mArr[task][blk][16][rLo]`）；Launch B 用 `m_final = 末块 m + Σ m_j`，按 `w = exp(m_j − m_final)` 缩放各块 `O_j` 再累加。只做块内 softmax 不做跨块 rescale → l 对、O 错 ~5% 且分布不规则。
2. **P 行 padded 步长**：P 行步长必须 pad 到 `nBlocks×BLOCK`（尾块写满 BLOCK、masked 位置 0 正确写入）；按 `valid` 截断会丢列，FA 宽容差（atol=0.02）掩盖、严格 atol（1e-3）下暴露。
3. **跨任务 oTmp 单槽覆盖竞态**：末块不 Wait 累加完成就复用 oTmp，多任务同核时下一任务块 0 覆盖上一任务末块未读数据（fp16 靠时序侥幸、bf16+MTP 必现）→ **每块都 WaitFlag(accDone)**（flag 配平仍 2Set:1Wait）。

### 3. 单活跃子核陷阱（与 H2 同源，单独列出因为文档曾写反）

`if (subIdx != 0) return` 单活跃子核 + flag MODE 仍 0x2 → AIC WaitFlag 只收 1 Set 永阻塞。修复：**双子核都活跃**（UB 按 subIdx 分区、各处理一半行），保留 0x2 双 Set。设计文档若写"单活跃子核 + flag 1:1"而代码 MODE 是 0x2，**以 MODE 实测配平为准**。

### 4. 多 batch bug（B=1 矩阵永远测不出，见评测 doc §4）

- **S-slot 核间地址撞车**：双缓冲需 2 槽，`sBase = coreIdx×TILE×BLOCK` 单槽 → core1-slot0 与 core0-slot1 同地址，非确定性时对时错。修复：槽位 ×2。
- **host BuildTiling 缺 `cu += qSeq[b]`**：B>1 全 batch 用 cu=0，读错 Q 写错 O（b0 对、b1 错）。从宿主工程复制 tiling 代码时最容易丢这一行。

### 5. 轻量纯向量 kernel 是单子核（勿假设双子核）

merge/归一化类轻量纯向量 kernel 编译为**单上下文**，`GetSubBlockIdx()` 恒 0——与主 kernel（双子核）行为不同。按双子核写 merge 会 rows 8-15 全零。写法：单上下文一次处理全部 16 行。

### 6. kv-split 并行优化的三条教训（v4 亲测）

1. **host seg planning 写死 SEGS=1/BPS=1 → 虚假性能**：只处理 block0，实测 45.6us（53.7×）全是假的，tiling dump 才发现 sed 没生效。**split 优化先 dump tiling 验证分段真写入了再谈性能**。
2. **sed 批量改 kernel 源不可回滚**：v3 源码被 sed 覆盖无法 diff、v4 bug 无从对比。kernel 迭代用编辑器逐处改 + 备份（`.bak`），或从基线骨架重写。
3. **精度不过的优化不可用**：kv-split 53.7× 但多块 FAIL，最终回退双 launch V^T 版（32/32 PASS）。性能数字必须以精度 PASS 为前提归档。

### 7. 手搓架构模板（双 launch，最终全 PASS 形态）

```
Launch A（FAQK 流式）: 逐块 Q·K^T → 双子核在线 softmax（m/l 递推）
                      → 写 P_unnorm（padded 步长）+ mArr + l
Launch B（BlockGemm PV + AIV 累加归一）: 读 P/mArr/l → w=exp(m_j−m_final)
                      → PV 累加 → /l → O
AIV→AIC 就绪 flag 2Set:1Wait；accDone flag 每块 Wait
```

调试套路：环境变量分相位 dump（如 `MLAHC_PHASE=1/3`）把 P/l/O 中间量对 numpy 参考二分定位——比整链比对快一个量级。

---

### 8. 工程级元坑（第三轮 kfc 单 TU 实证补充，2026-08-27）——坑的不是 kernel，是你的观测

| # | 坑 | 现象 | 对策 |
|---|---|---|---|
| M1 | **stale binary 假验证** | `.asc` TU 的 include 依赖不被 CMake 跟踪；`rm -rf CMakeFiles/<target>.dir` 静默破坏 make 规则图（旧 exe 残留）→ 改了代码数字逐位不变 / 假 PASS | 重建脚本：`touch TU源 + cmake .. 再生成 + make` + **断言 exe mtime > 源 mtime**；exe 打 REV 版本串运行确认 |
| M2 | **stale dump 假验证** | kernel 挂死（timeout）后 verify 读上一次 dump → 逐位相同的假 PASS/FAIL，误导归因数小时 | **verify 前必删 dump 文件**；运行必查 sync 完成标记 |
| M3 | **memset 排队污染计时** | 争用卡上 aclrtMemsetAsync（全量 ws 几十 MB）= 25ms/launch 且与大小无关（纯排队）；把它算进 kernel 时间 → 全部归因错 | 事件对分离 `ev0→memset→ev1→kernel→ev2` 分别报 MEMSET_US/KERNEL_US；kfc 区只需清 64KB |
| M4 | **device lambda** | ASC 编译器不吃 device lambda（报 undeclared member） | 写私有成员函数 |
| M5 | **验证脚本自己的语法错** | 测试代码里 `replace('raise SystemExit(main())','')` 造成空 if 块 → 把文件语法错误误当被测对象失败 | 测试 harness 的字符串替换要落 'pass' 并单独语法校验 |
| M6 | **对标参数未生效** | 逐 case 精度与性能分别起两次 exe，第二次忘了带同一个 env 旋钮 → 测的是默认配置还当成旋钮效果 | 旋钮写入 tiling 并在运行日志回显；一次进程内完成"生成+计时" |
| M7 | **链接文件≠目录文件（tiling 来源坑）** | 工程根目录同名文件从未被 CMake 链接（实际链的是别处版本）→ 据目录文件打包/移植后行为完全不同（kvSplit 恒错档、小 kv 更慢） | 移植/打包前核对 build 实际链接的文件：CMake 源列表指向 + quoted-include 解析顺序 |

## 性能评测与 gate 判定全文（fa-perf-gate-evaluation）
> 以下为评测方法论源文件全量正文：计时四陷阱、gate 口径与实测参考、序列长分桶泛化套件、B>1 探测器、归档清单与争用设备计时口径。

> **适用**：FA 族（FlashAttention / MLA / FFA / paged 变体）生成后的性能评测与 gate 判定。本文全部来自 FA 封装工程（FA 内核封装）+ MLA 手搓工程（MLA 内核封装）两工程实测踩坑（910B3），每条都付过假结论的代价。
> **核心原则**：**计时口径错 → 一切加速比都是假的**。先读 §1 陷阱，再采数。

---

### 1. 计时四陷阱（按危害排序，全部实测复现过）

#### 陷阱 1：torch 进程内 ctypes kernel 逐次 event 计时**漏捕**（最危险，产出过 391× 假加速）

`record → ctypes launch → record → sync` ×10 取 min，对外来（非 torch_npu 注册）kernel **偶发漏捕**：min 恒 ~1.6-1.7us、与 case 规模无关 → 加速比虚高数百倍。median 表现正常，极具迷惑性。

**正解**：
- kernel 侧时长用 **exe 纯回放 aclrtEvent 组计时**：3 组 × 10 次/组，组均取 min（与 msprof Task Duration 同带，已互证）
- torch event 只用于纯 torch op 的 baseline（median + min 双报，min 单报必查是否触底恒定值）

#### 陷阱 2：ATB 融合算子标杆每次 launch 重建 executor（host 开销吞掉一切）

`npu_multi_head_latent_attention` 等 ATB 算子无 executor 复用：单次 event 计时 4061us vs device 实际 76us。对比这类标杆**必须用 profiler op_summary 的 device 时长**，event/墙钟口径一律无效。

> **★NNAL ATB env 前置（2026-09 实测）**：跑官方 ATB 标杆前必须 `source /usr/local/Ascend/nnal/atb/set_env.sh`
> （与 cann set_env 同 source），否则 `torch_npu.atb.npu_multi_head_latent_attention` 抛
> "Please check that the nnal package is installed"——bench 直接退化为 0/0。墙钟口径下还会把
> executor 重建算进标杆时长（短 kv 放大 5~40×），gate 虚高。

#### 陷阱 3：torch_npu profiler 惰性解析

op_summary 只在 `export_chrome_trace`（或等价导出）触发后才生成——只 start/stop 不导出会以为"没有 op_summary"。

#### 陷阱 4：profiling 异常值先查环境再下结论

曾出现 msprof 采出 FAInferFp16 15us（真实 72-78us）：根因是**磁盘满导致 trace 截断**，不是工具盲区。结论"工具有固有盲区"之前，先查磁盘/设备状态（`df -h`、npu-smi）。

**ATB 标杆采集附加**：逐 case **独立进程**跑（单进程累积采 20 case 会被 OOM 杀，采一半丢全部）。

---

### 2. gate 判定口径

```
gate = geomean(标杆 device 时长 / 本侧 device 时长)   # 20 case 泛化矩阵
```

- **对比对象必须在报告显式命名**：「vs torch 分解链」与「vs 融合算子标杆」是两个完全不同的数字（MLA 实测：分解链 33.5× 墙钟 vs 官方融合 1.16× device）。写"比标杆快 N 倍"前先确认口径。
- 公平下界 = **device 侧 sum**（op_summary 逐 kernel 求和）；墙钟含分解链 host 调度间隙（实测 d18：441 kernel/iter，19.7ms 墙钟中 8.3ms 是间隙）。
- 几何平均（geomean）为 gate，算术均值只在附录给（长尾 case 会拉高算术均值造成误读）。
- 同源对比（本侧即标杆 kernel 同源封装）gate ≈ 1.0 属正常（FA 内核封装差 1.025）；此时 gate 意义 = 回归检测，不是优化空间。

实测参考（910B3）：
| 算子 | kernel | gate vs 标杆 |
|---|---|---|
| fa-page-attention | FA 内核封装 | **1.025**（同源，封装差） |
| fa-mla | MLA 内核封装 | **3.047**（序列长扫描至 kv=204800 同设备复测版，长段 ratio 最高 7.16×；batch 放大口径 1.161） |
| （手搓版对照组） | 双 launch AIC/AIV | 0.107~0.364（正确性优先，预期内） |

---

### 3. 序列长分布泛化套件（当前口径；早期时长分布口径已废弃）

**当前口径（2026-08-24 起）**：case **按 kv 序列长分桶**，不按时长分桶。序列长尽可能长（≥20 万 token），以不爆内存为限：
- fa（单序列 kv 无上限，受内存限制）：20 case 升序 kv 扫描，桶 `≤16k ×5 / 16k~64k ×5 / 64k~200k ×5 / >200k ×5`，最大 kv 458752（k+v ≈229MB/case，实测安全）
- mla（接口 gen_data 声明范围仅 [128,16384]，**实测非硬约束，kernel/ATB 均可正确跑至 20万+**）：桶 `≤2k / 2k~16k / 16k~100k / 100k~204800(20万) ×5`；超 16384 属标杆未验证区，精度必须逐 case 自测；B×kv 总 token 放大（如 B16×16384）可另列演示 case，勿混进 kv 扫描
- B 固定小值（fa B=1、mla B=2）使 kv 成为唯一变量；B>1 约束仍满足（mla 侧）
- 内存预算：host 每 case bin ≈ `kv × D × 2B × 2(k+v)`；device 同量级 + workspace。生成前 `df -h` + `npu-smi` 双查

**历史口径（已废弃，仅存档）**：按时长分桶 `<100us:10% / 100-500:20% / 500-1000:40% / >1000:30%` → 20 case 精确 2/4/8/6，需用吞吐模型定标构造（fa `t≈0.004×kv`、mla `U=Σ T·H·kv_b` 吞吐表），模型 2 轮迭代收敛。废弃原因：定标迭代成本高、且时长是导出量（随 kernel 优化漂移，case 集要反复重建）；序列长是输入量，一次构造长期有效。

**case 重生成坑**：新旧 case 同名时脚本会"先 gen 新、后清理误删新留旧"——重生成前先手动清旧名目录；换口径用新目录（如 `cases_len/`）并保留旧目录存档。

---

### 4. ★泛化套件的隐藏价值：强制 B>1 是多 batch bug 探测器

单 batch（B=1）语义矩阵会**掩盖**多 batch bug——泛化/性能套件强制构造 B>1 case 后，FA 封装工程 一次暴露两个此前 10/10 PASS 从未触发的 bug：

| bug | 现象 | 根因 |
|---|---|---|
| **S-slot 核间地址撞车**（非确定性：时对时错） | `sBase = coreIdx×TILE×BLOCK`，但双缓冲需 2 槽 → core1-slot0 与 core0-slot1 同地址 | 双缓冲槽位 ×2（`sBase×2`） |
| **host BuildTiling 缺 `cu += qSeq[b]`**（确定性：b0 对 b1 错） | token 偏移不按 batch 递增，B>1 全部 batch 用 cu=0 | 补累积行（MLA 手搓工程 有、FA 封装工程 漏——复制宿主代码时最容易丢的一行） |

**规则**：任何 FA 族 kernel 交付前，评测矩阵必须含 B>1 且各 batch qSeq/kvSeq 不等的 case。

---

### 5. 归档清单

评测完成后归档（两工程模板：`FA 封装工程/perf_test_list.md`）：
- 逐 case 表：case 名（`<op>-NNN` 三位编号）/ shape 布局与参数 / 本侧 device us / 标杆 device us / 加速比
- 末行 gate（几何平均）+ 均值（算术，标注口径）
- 时长分布命中表（2/4/8/6 ✓）、精度 N/N PASS、三方互证 err 范围
- 数据 tsv 留 `/tmp` 副本（case 名与目录名、meta.json `case` 字段三处同步，重命名用脚本一次改齐）

---

### 6. 争用设备计时口径（2026-08-27 第三轮实证补充）

协同租户占卡 87-95% 时的公平口径，否则归因全错：

1. **本侧**：exe 内事件对把 memset 与 kernel **分开计时**（`ev0→memsetAsync→ev1→launch→ev2`，各报 MEMSET_US / KERNEL_US，取 3 组×10 次的组内 min）。争用排队发生在第一个操作上，不分离时 25ms 级的 memset 排队会把 kernel 归因完全污染。
2. **标杆侧**（torch_npu op-plugin 路径）：同卡同窗口**背靠背回放均摊**——warmup 3-5 次 → sync → `perf_counter` 包住 N 次连发（N=5-10）→ sync，除以 N，取 3 组 min。torch `cuda.Event` 在 NPU 是 dummy 类直接抛异常，不可用；`time.perf_counter` 单发含 host 调度间隙（几十至几百 μs）必须均摊。
3. **逐 case 同窗口交替**：本侧与标杆紧邻背靠背跑（同争用水平），不要隔着几分钟各跑各的——争用波动 > 30%。
4. **旋钮一致性**（M6）：性能与精度若分别起进程，必须带同一套 env 旋钮；旋钮值打进运行日志回显。
5. 序列长分桶套件的标杆侧同样可能受 H≤16（sink 参数限制）/同进程随机 161001 约束：逐 case 子进程隔离（见配方册 §12.11-F）。

## FlashAttention Stage Tuning（catlass-op-perf-tune SKILL.md 整节原文）

> 以下为 `ops/catlass-op-perf-tune/SKILL.md`「FlashAttention Stage Tuning」节全文（L105 起，
> 正文原样；其指向的 `a2-a3-flash-attention-stage-design.md` 全文已收录于本 concept
> 「A2/A3 Stage 设计经验全文」节）。[^fa-tune-skill-route]

当目标是 FlashAttention（MHA/GQA）融合算子时，先读取 `catlass-op-develop/references/patterns/a2-a3-flash-attention-stage-design.md`。FA 的常见瓶颈不是单个 TileShape，而是 C1→V1→C2→V2 四段流水的 AIC/AIV 重叠、CrossCoreFlag、workspace slot、online softmax 状态递推的组合问题。

| 现象 | 优先归因 | 优先检查 |
|---|---|---|
| AIC/AIV 互等明显 | stage 边界或 flag 放在内层循环 | wait/set 是否只在 stage 入口/末尾；workspace slot 是否冲突 |
| Cube 利用率低但 MTE 不高 | QK^T 与 PV 未重叠 | AIC/AIV 双特化重叠、Bk 分块、TileShape（参考 `GemmShape<128,128,128>`）|
| 长上下文变慢 | KV 分块 / block_table 粒度 | blockSize=128 块重排是否与 kernel 消费一致 |
| 非 128 对齐 tail 慢或错 | 尾块未 0 填充 | 尾块填充路径、bin/device 尺寸分离 |
| 小 `Sq*H` 慢 | 核调度不足 | 先评估分块/核分配，不只调 TileShape |
| GQA 场景慢 | head 合并策略 | 同 kvHead 的 query head 是否合并到 Mmad m 维 |

性能报告必须把判断写进 `docs/perf/round_NNN/`：说明瓶颈属于 stage 重叠、flag、workspace、分块、尾块还是 baseline unsupported；A2 参考经验：正确重叠后 cube_util ~96%、与参考 example 长上下文 parity（1159μs vs 1166μs）。

# 验证方法

**精度双口径**（缺一不可）：[^fa-precision-verify]

1. 内部 golden：CPU numpy/torch fp32 累加 softmax attention 逐元素对比。
2. 标杆对比：算子 dump BNSD 输出 O 到 `o.bin`，与 `npu_fusion_attention`（封装
   `aclnnFlashAttentionScore`）输出逐元素比。

判据（mixed tolerance，A2 实测经验阈值）：`abs(actual−golden) <= atol + rtol*abs(golden)`，
`atol=0.02, rtol=0.1`，通过率 ≥ 99.9% 且 `max_abs < 0.05`（A2 实测 custom vs aclnn max_abs ~1e-4）。
每 case 报告 `shape`、`dtype`、`atol`、`rtol`、`matched_ratio`、`max_abs`、`pass/fail`。验证前必删旧
dump（防 stale 假验证）。

**性能 gate**：`gate = geomean(标杆 device 时长 / 本侧 device 时长)`（20 case 序列长分桶泛化矩阵）；
对比对象必须在报告显式命名（vs torch 分解链 ≠ vs 融合算子标杆）；算术均值只附录给。同源封装
gate≈1.0 属正常（回归检测语义）。[^fa-perf-gate]

**B>1 强制**：单 batch 矩阵会掩盖多 batch bug——评测矩阵必须含 B>1 且各 batch qSeq/kvSeq 不等的
case（曾一次暴露 S-slot 核间地址撞车、host tiling 缺 `cu += qSeq[b]` 两个 10/10 PASS 未触发的 bug）。[^fa-perf-gate]

## 性能计量口径与采集指引（ops-profiling !992 增量，df779a09）

**性能计量口径：launch wall 仅计 compute kernel 时间跨度**。来源：`ops/ops-profiling/scripts/msprof_perf_summary.py`
新增的 `_collect_compute_kernels` + `_launch_wall_min`（!992，+43 行）。脚本代码本身不入 concept，
过滤与聚类规则如下（方法论描述，规则与脚本行为一致）：[^fa-profiling-script]

- **只计 compute kernel**：排除 meta 任务（`kernel_type` 命中 `_META_KERNEL_TYPES`）与名字为空的行；
  排除数据搬运——`kernel_name` 含 `Memcpy` / `MEMCPY` / `Rdma` / `RDMA` 的任务一律剔除（含 H2D/D2D
  tiling 拷贝，某些 host 在 launch 前逐次下发的 H2D 拷贝不计入 launch wall）；`task_time(us)` 解析失败
  或 `<= 0` 的行剔除。
- **launch wall 定义**：单次 launch 的 wall = 该次 launch 内 compute 任务的 `max(end) − min(start)`。
  对一次 launch 含多任务的 kernel（AIC/AIV 双流任务），按任务间 gap 聚类（gap 阈值 8 μs：launch 内
  任务紧叠/紧接，launch 之间 gap 超过阈值）把任务聚类成 launch，再取每个聚类的 wall。
- **取值**：丢弃 warmup 后，取活跃的末 `repeats` 次 launch 中 wall 的最小值（抗争用噪声的 min 口径）。

**ops-profiling SKILL.md !992 增量原文（高频问答速查 + 阅读纪律，+17 行）**：[^fa-profiling-faq]

> ## 高频问答速查（先查这里——本节答案即为权威结论，已与脚本实际行为对齐，无需再读源码验证）
>
> | 问题 | 答案 |
> |---|---|
> | compare 模式下用例文件加载优先级？ | ① `<op>_perf_cases.jsonl`（标准命名）→ ② 任意 `*.jsonl` → ③ 任意 `*.json`（非 `.bak`）→ ④ `model.py` 的 `get_input_groups()`。同时存在时**选①**（详见下文「用例加载优先级」） |
> | 怎么做加速比对比？ | `msprof_profile_run.sh --compare`（全量）/ `--quick`（1 轮快速），输出 `performance.json` + markdown 报告即完成，**不**走可视化子技能 |
> | MC² / 多 rank（fork）算子用什么采集？ | **必须 `msprof`**，禁止 `msprof op`（详见 msprof-guide.md「MC² 多 rank 算子采集」） |
> | 采集结果如何判定瓶颈？ | 用 `msprof_perf_summary.py` 解析 `PROF_GROUP_*`，输出瓶颈类型与各 pipe 占比，判定规则见 `references/msprof-guide.md` |
>
> ### 阅读纪律（防上下文爆炸，必须遵守）
>
> - `scripts/msprof_perf_summary.py`（2400+ 行）与 `scripts/msprof_profile_run.sh` 是**可执行工具，不是文档**。需要确认某个行为时，只允许 `grep -n "<函数名/关键字>" scripts/msprof_perf_summary.py` 定位后**读命中处前后 ~30 行**；**禁止整文件通读**。
> - `references/` 下四份文档按文末「参考资源」表的"何时查阅"**按需加载**；回答本文件已写明的事实性问题时，不需要打开任何 reference 或 script。
> - 本 SKILL.md 覆盖全部使用决策；若与脚本实际行为发现不一致，以脚本行为为准并在本文件勘误。

**ops-profiling evals.json !992 增量（+20 行，参考性质）**：[^fa-profiling-evals] 该 diff 为五条既有评测
（MC² 多 rank 禁 msprof op、深度分析工具选择探测环境、对比模式用例加载优先级、quick 与标准采集用途、
两套性能产物目录不互喂）各补 `max_tokens_by_model` 配置（`deepseek-v4-flash: 120000`、`glm-5: 110000`），
无新增 FA 断言内容；作为评测配置口径参考收录。


[^fa-design-route]: 固定提交中的 FA 设计路由：场景定义、命名约定、组件选型表、数据流、shape 覆盖与常见陷阱。
[^fa-baseline-ref]: 固定提交中的 FA 标杆接口规范：aclnn/npu 标杆定位、参数映射、sparse_mode→maskType、可继承边界与用户 contract 优先规则。
[^fa-dev-patterns]: 固定提交中的 FA 实现规则：kernel 设计经验 §0（AIC/AIV 分工、在线 softmax、同步协议、workspace、PAGED）、BNSD 接口、尾块、双口径精度、性能基线、T1-T8 陷阱与 FA0-FA13 检查表。
[^fa-stage-design]: 固定提交中的 A2/A3 FA stage 设计经验：四段流水顺序、workspace slot 协议、online 状态管理、UB/L1/L0 预算表与压缩顺序、checklist。
[^fa-handcraft-pitfalls]: 固定提交中的 FA 手搓坑清单：七个架构级坑、多块 softmax 三修复、单活跃子核、多 batch bug 与 kv-split 教训。
[^fa-perf-gate]: 固定提交中的 FA 性能评测与 gate 方法论：计时四陷阱、gate 口径、序列长分桶泛化套件、B>1 探测器与争用设备计时口径。
[^fa-precision-verify]: 固定提交中的 FA/MHA/GQA 精度验证补充：baseline 层级冻结、双口径验证与 mixed tolerance 报告字段。
[^fa-shape-constraints]: 固定提交中的 shape 覆盖约束 Δ6：FA 类测试矩阵按算法维度构造的类别表。
[^fa-compilation]: 固定提交中的独立 kernel 直调构建说明：CATLASS_ARCH 宏、--npu-arch、ASC 入口、include/link/核数/硬件同步注入清单与最小 CMake 骨架。
[^fa-precheck-clauses]: 固定提交中的主体工作流 FA 预检条款（布局/D%16/SoC/baseline 四项确认）。
[^fa-architect-clauses]: 固定提交中的设计 agent FA 条款：FA 类型识别、设计前置链（design 路由/baseline 冻结/内核来源锁定/组件选型/BNSD host 转换）与 catlass/examples 例外。
[^fa-generator-clauses]: 固定提交中的开发 agent FA 条款：内核来源锁定、禁止项、BNSD host 转换与精度双口径。
[^fa-reviewer-clauses]: 固定提交中的审查 agent FA 专项门禁（Step FA，FA1–FA11 编号体系）：FA1/FA3/FA4/FA9 重点核查与组件选型；FA0/FA12/FA13 为 ops 活体表扩展，对应关系见 §6「审查门禁」。
[^fa-design-skill-route]: catlass-op-design SKILL.md 中的 FA 检索路由原文：When-to-Use FA 条目（L82-83）、场景路由树 FA 分支（L101-102）、索引表 flash-attention.md 条目（L227）。
[^fa-dev-skill-route]: catlass-op-develop SKILL.md 中的 FA 检索路由原文：FA 实现注意事项/FA 族形态全覆盖/性能 gate 三条路由（L80/83/86）与索引表 FA 条目（L107-108）。
[^fa-tune-skill-route]: catlass-op-perf-tune SKILL.md 的「FlashAttention Stage Tuning」整节（L105 起）：四段流水归因表与 A2 参考经验。
[^fa-profiling-faq]: ops-profiling SKILL.md !992（df779a09）+17 行增量原文：高频问答速查（用例加载优先级/加速比对比/MC² 采集/瓶颈判定）与阅读纪律。
[^fa-profiling-evals]: ops-profiling evals.json !992（df779a09）+20 行增量：五条既有评测的 max_tokens_by_model 配置（参考性质）。
[^fa-profiling-script]: ops-profiling msprof_perf_summary.py !992（df779a09）+43 行增量：_collect_compute_kernels/_launch_wall_min——launch wall 仅计 compute kernel 时间跨度、排除数据搬运与 meta 任务的性能计量口径。

## 共享参考的 FA 增量全文

以下为三个共享参考文件中 FA 相关部分的全量正文（标题层级按 OKF 章节适配，正文原样；链接已内联化）。

##### 独立 kernel 直调构建（脱离 catlass examples 模板时）

当用 `<<<>>>` kernel 直调（host main 含 launch，如 FA demo / 独立验证工程）且**不走 catlass `examples/` 构建模板**时，自包含 CMake 必须显式注入下列项（examples 模板里这些由 `examples/CMakeLists.txt` 全局提供，独立工程没有，**易漏**）：

| 项 | 做法 | 漏的后果 |
|----|------|----------|
| **`CATLASS_ARCH` 编译宏** | `target_compile_definitions(<op> PRIVATE CATLASS_ARCH=2201)` | catlass 架构头 `copy_l0c_to_gm.hpp` 等 `#if (CATLASS_ARCH==2201)` 不成立 → 不 include 定义 `ScaleGranularity` 等的架构头 → 级联报 `ScaleGranularity undeclared / PackedTileCopyTla too few template args`。**注意：`--npu-arch=dav-2201` 是编译器 flag，≠ `CATLASS_ARCH` 宏，两者都要** |
| `--npu-arch` | `target_compile_options(<op> PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)` | ASC 编译无目标架构 |
| ASC 语言 host 入口 | host main 用 `.asc` 扩展名（自动按 ASC 编译），或 `.cpp` + `set_source_files_properties(<f> PROPERTIES LANGUAGE ASC)` | `<<<>>>` 语法 / AscendC 符号无法编译 |
| include 路径 | `catlass/include` + skill 提供的算子内核目录（FA/MLA 内核随本 skill 交付）+ `$ENV{ASCEND_HOME_PATH}/include`（`acl/acl.h`、`tiling/platform/`）；`kernel_operator.h` 由 ASC 工具链自动提供 | 头文件找不到 |
| link 库 | `ascendc_runtime ascendcl runtime profapi mmpa c_sec error_manager graph_base tiling_api register platform unified_dlog dl m`（`$ASCEND_HOME_PATH/lib64`） | 链接失败 |
| `<<<>>>` 实参类型 | `GM_ADDR = __gm__ uint8_t*`；host `aclrtMalloc` 返回 `void*`，launch 实参须 `(GM_ADDR)dQ` 显式强转（`hwSync` 是 `uint64_t` 不转） | `cannot initialize '__gm__ uint8_t*' with 'void*'` |
| 核数 | `aclGetDeviceCapability(dev, ACL_DEVICE_INFO_AI_CORE_NUM, &v)`（enum 属 `aclDeviceInfo`，**不是** `aclCompileOpt::ACL_AICORE_NUM`）；fallback `PlatformAscendCManager::GetInstance()->GetCoreNumAic()` | 取错 enum → 编译错；blockDim=0 → kernel 不跑 |
| 硬件同步 | `aclrtGetHardwareSyncAddr(&hwSync)` 传给内核首参，内核内 `SetSyncBaseAddr(hwSync)` | AIC/AIV 跨核握手失败 |

最小骨架（FA 类，目标 A2/910B3）：

```cmake
cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(<op> LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)

set(CATLASS_INCLUDE "/path/to/catlass/include")
# FA 族：kernel 由手搓手册（fa-kernel-handcraft.md §1–9 + fa-kernel-recipes.md §12.13）指导从零写进工程 op_kernel/，无外部内核目录
# MLA：kernel 按 fa-mla-paged.md §8 手搓进工程 op_kernel/（组件全在 catlass include 层），无外部内核目录

add_executable(<op> <op>.asc)
target_compile_definitions(<op> PRIVATE CATLASS_ARCH=2201)          # ★必需，见上表
target_include_directories(<op> PRIVATE ${CATLASS_INCLUDE} ${SKILL_KERNEL_DIR} $ENV{ASCEND_HOME_PATH}/include)
target_compile_options(<op> PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)
target_link_directories(<op> PRIVATE $ENV{ASCEND_HOME_PATH}/lib64)
target_link_libraries(<op> PRIVATE ascendc_runtime ascendcl runtime profapi mmpa c_sec
                       error_manager graph_base tiling_api register platform unified_dlog dl m)
```

> 若走 catlass `examples/` 模板（`catlass_example_add_executable`），上述 `CATLASS_ARCH` / `--npu-arch` / 通用 link 库由 `examples/CMakeLists.txt` 全局注入，只需额外加 skill 组件 include 与 `target_link_libraries` 补充库即可。**本 skill 的 FA 族算子不走该模板**——一律手搓（手搓手册 §1–9/§12.13）+ 自写 host + 独立 CMake。

##### 9. FlashAttention / MHA / GQA 类精度验证补充

FlashAttention 融合算子的 golden 与测试覆盖还需满足：

1. **冻结 baseline 层级**：full-flow fused op 必须对齐同语义 full-flow baseline，**首选 `aclnnFlashAttentionScore`**（`torch_npu.npu_fusion_attention(q, k, v, H, "BNSD", scale=scale)`）。语义、dtype、layout、scale（`1/sqrt(D)`）必须与交付目标一致。
2. **双口径验证**：① 内部 golden（CPU numpy/torch fp32 累加 softmax attention）；② **aclnn 标杆对比**（算子 dump BNSD 输出 O 到 `o.bin`，与 aclnn 输出逐元素对比）。
3. **mixed tolerance 报告字段固定**：每 case 输出 `shape`、`dtype`、`atol`、`rtol`、`matched_ratio`、`max_abs`、`pass/fail`。浮点判据按 `ops-precision-standard`：`abs(actual-golden) <= atol + rtol*abs(golden)`。FA 经验阈值：`atol=0.02, rtol=0.1`，通过率 ≥99.9% 且 `max_abs < 0.05`（A2 实测 custom vs aclnn max_abs ~1e-4）。
4. **shape 覆盖按算法维度生成**：覆盖 B/Sq/Sk 组合、`Skv` 非 128 对齐尾块、MHA/GQA、D=64/128、数值边界。详见 shape-constraints.md Δ6。

##### Δ6：FlashAttention / MHA / GQA 类 shape 覆盖按算法维度生成

FlashAttention 的 shape 不应只按 GEMM 的 `(M,N,K)` 或历史调试 tuple 设计。测试矩阵必须说明每个 shape 对应的覆盖类别。

| 覆盖类别 | 构造方式 | 目的 |
|---------|---------|------|
| B/Sq/Sk 组合 | 单 batch/多 batch；`Sq == Sk`、`Sq < Sk` | 覆盖核调度与 KV 分块路径 |
| `Skv` 对齐边界 | `Skv` 恰为 128 倍数 / 非倍数（如 177） | 覆盖尾块 0 填充路径（bin 读入与 device 块格式尺寸分离）|
| H / head 形态 | MHA（Hq==Hkv）、GQA（Hq>Hkv）；单头/多头 | 覆盖 head 合并到 Mmad m 维的路径 |
| `D` / headSize | 64 / 128，`D % 16 == 0` | 覆盖 cube 友好分块 |
| 数值边界 | 全零 Q/K/V、近零 softmax 输出、极值 scale | 暴露 softmax 状态递推与 mixed tolerance 问题 |
| 对标 shape | 与 `aclnnFlashAttentionScore` 完全同 shape、同 layout、同 dtype | 标杆对比可比性 |

向调用方提的需求：脚本保留固定 shape tuple 时，必须在报告/注释标明每个 tuple 来自哪个覆盖类别；否则视为硬编码经验，不视为完整测试设计。

[^fa-handcraft-manual]: 固定提交中的 FA kernel 手搓实现手册全文：四文件结构、任务切分（N pack 进 M、GQA 双平面索引）、AIC/AIV 双入口与主循环、跨核同步编排、HardEvent 预置全表、BlockMmad/OnlineSoftmax/RescaleO epilogue 内部机制、host 直调模板、变体映射法、K1-K10 自查表、关键常量速查与全家族覆盖路由表。
[^fa-recipe-book]: 固定提交中的 FA 算子配方库全文：§12.5 FFA 全程实证（F1-F4/W1-W20 坑、P1-P4 优化技术、v1→v4 演进与最终基线 1.68×、§12.5.14 终态配方 2.19×）、§12.10 s1s2 单核融合算法原理、§12.11 fa-sink S1 模板流程与性能基线、§12.13 kfc 单 TU 直调终态配方（A-F0 全节与实测 geomean 0.693）。
[^fa-compound-features]: 固定提交中的 FA 复合特性叠加册全文：§12.5.3 器件级 API 墙与 AIV 调用风暴、§12.6 复合特性总纲（布局零拷贝重解释、三特性 kernel 内落点、sink 校正向量算子链）、§12.7 测量与泛化坑、§12.12 varlen/q8 叠加步骤、§12.14 causal/paged 叠加步骤。

