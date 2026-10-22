---
type: "operator"
title: "FA 变体族 A2/A3 手搓配方精要"
description: "FA 变体（FA-Paged 标准/GQA/Causal、fa-sink、fa-varlen TND、fa-q8、FFA 双 KV、MLA latent 压缩分页）的手搓配方精要：算法差异、分核组织（S1 单遍/s1s2 双层/flash-decoding）、契约转换、关键修复、死锁前提与验收口径。"
tags: ["catlass-cpp", "flash-attention", "fa-variants", "fa-paged", "fa-sink", "fa-varlen", "ffa", "mla", "q8", "atlas_a2_a3", "handcraft"]
status: "stable"
generated: {"by": "process:catlass-cpp-fa-migration-curation", "at": "2026-10-22T00:00:00Z"}
verified: [{"by": "process:fa-source-cross-check", "at": "2026-10-22T00:00:00Z"}]
sources:
  - {"id": "fa-handcraft-manual", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-kernel-handcraft.md", "title": "FA kernel 手搓手册：四文件结构、任务切分、同步编排、HardEvent 预置、softmax epilogue 内部与变体映射法", "kind": "repository"}
  - {"id": "fa-recipe-book", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-kernel-recipes.md", "title": "FA kernel 算子配方库：FFA 实证、s1s2 算法、S1 sink 流程与 kfc 直调配方", "kind": "repository"}
  - {"id": "fa-compound", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-kernel-compound-features.md", "title": "FA 复合特性叠加：sink/varlen/q8/causal/paged 手搓步骤、器件级 API 墙与测量坑", "kind": "repository"}
  - {"id": "fa-paged-recipe", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-paged-handcraft-recipe.md", "title": "FA-Paged 手搓完整配方：四文件端到端逐步骤、3 条关键修复与 mask 路径", "kind": "repository"}
  - {"id": "fa-sink-recipe", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-sink-handcraft-recipe.md", "title": "fa-sink/标准 FA（S1 模板）手搓配方：三深流水、SoftmaxFlashV2 isUpdate 播种、sparse 跳块", "kind": "repository"}
  - {"id": "fa-varlen-recipe", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-varlen-handcraft-recipe.md", "title": "fa-varlen（TND 变长）手搓配方：差分解码、TND 直读、批界 s2End 截断", "kind": "repository"}
  - {"id": "ffa-recipe", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/ffa-handcraft-recipe.md", "title": "FFA/FIA 双 KV TensorList 算子配方：37 参契约与 kernel 结构差异清单", "kind": "repository"}
  - {"id": "fa-mla-recipe", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-mla-handcraft-recipe.md", "title": "MLA 手搓端到端配方：__DAV 分流、HardEvent 全表、prologue+延迟 PV、FD rescale", "kind": "repository"}
  - {"id": "fa-mla-paged", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-mla-paged.md", "title": "MLA Paged Attention 实现契约：launch/tiling/workspace/shape 约束与五大性能支柱", "kind": "repository"}
  - {"id": "fa-dev-skill-route", "resource": "git:cann/cannbot-skills@585201016db01d5c9854402ca5086bd53d2ea46e:ops/catlass-op-develop/SKILL.md", "title": "ops 开发 skill FA 族形态全覆盖检索路由与性能 gate 路由（L83/86 原文）", "kind": "repository"}
operator_families: ["flash-attention"]
architectures: ["atlas_a2_a3"]
---

# 接口与概念

本 concept 收录 FA 族**变体**的手搓配方精要（标准 dense FA 前向的组件复用路径见
flash-attention-infer-patterns.md）。变体共用 FA 的在线 softmax 状态递推、BNSD 公开接口约定与
`aclnnFlashAttentionScore` 标杆口径；差异在 KV 布局、mask 语义、量化相位或双 KV 拼接。

**形态全覆盖检索路由（catlass-op-develop SKILL.md L83/86 原文，变体命中的总入口句）**：[^fa-dev-skill-route]

- **任何 FA 族形态（标准 FA / BNSD / Causal / TND-varlen / paged / FFA / sink / 复合特性）→ 一律按 [patterns/fa-kernel-handcraft.md（见infer 册「手搓手册全文」节） 手搓单 kernel**：**标准 FA/FA-Paged/GQA/causal（CrossCoreFlag 组件路径）优先按 [patterns/fa-paged-handcraft-recipe.md（见本册「FA-Paged 手搓完整配方全文」节） 逐行写**（四文件端到端：kernel Step 1-7 + mask Step 8 + common/tiling/host Step 9-11，NO_MASK 闭卷实证全 PASS；手册 §1–9 是其机制解读层）；kfc 单 TU 路径走 **[fa-kernel-recipes.md（见infer 册「配方集全文」节） §12.13 终态配方**（fa/fa-sink，geomean 0.693 vs 标杆，序列长至 458752，含 F0 十一步逐步生成流程），[fa-kernel-compound-features.md（见infer 册「复合特性叠加册全文」节） §12.12 = TND-varlen/q8 叠加、**§12.14 = causal/paged 逐步叠加**，配方册 §12.5.14 = FFA 完整配方——**全程只用本 skill 文件**
- **FA 族性能评测 / gate 判定 / 泛化套件 → [patterns/fa-perf-gate-evaluation.md（见infer 册「性能评测与 gate 判定全文」节）**（计时四陷阱、gate=geomean device 口径、序列长分桶、B>1 探 bug）

**内核来源分流（先选路再动手）**：[^fa-handcraft-manual]

| 算子形态 | 策略 |
|---|---|
| 标准 FA / FA-Paged / GQA / Causal（组件路径） | 按 fa-paged 配方四文件端到端逐行写（kernel + kernel_common + tiling + host）；NO_MASK 闭卷实证 kv=128~204800 全 PASS，max_abs≤7.6e-06 |
| fa-sink（S1 模板）/ varlen / 标准形态 kfc 直调 | 按 S1 模板/s1s2 双层配方逐相位写；kfc 单 TU 直调（`__mix__(1,2)`+`REGIST_MATMUL_OBJ`）实测 geomean 0.693 vs 标杆 |
| 其余 FA 变体（FlashSoftmax 分片等） | AscendC 高阶 API（MatMul/SoftmaxFlashV2）自研 kernel：Event 同步可观测 |
| MLA（latent 压缩 + 分页 KV） | 按 fa-mla 配方手搓（闭卷实证一次跑通；五大性能支柱缺一不可） |
| FFA/FIA（双 KV TensorList + sharedPrefix） | 终态配方生成（实证几何 2.19×），结构契约对照 |

MLA 语义：`O = softmax(scale·(Q·c_kv + q_rope·k_rope))·V_latent`，`scale=1/√576`、`Dc=512/Dr=64/
blockSize=128/V=latent`、`q∈[1,4]`、`kv≥128`（gen_data 声明 ≤16384 为验证包络，实测可至 20 万+）。
dispatch 固化 `MmadAtlasA2MLAQK/MLAPV`。[^fa-mla-paged]

FFA/FIA 语义（37 参入口）：
`O = softmax(scale·(Q[+qRope]·[KSharedPrefix; KTensorList][+kRope]ᵀ + pse + mask)[+sink]·
[VSharedPrefix; VTensorList])`。双 KV = sharedPrefix（全 batch 共享前缀）+ TensorList local（per-batch
池），kernel 侧统一按 paged blockTable 连续寻址（逻辑 kvSeqlen = sharedLen + localLen_b），不分两次
launch、不做 host 拼接拷贝。含 LSE 可选输出、nZ（K）/zN（V）KV 布局分支、量化 antiquant 入参族。[^ffa-recipe]

## 算子算法

各变体在标准 FA 四段流水（C1→V1→C2→V2）之上的算法增量：[^fa-compound]

| 特性 | 数学 | kernel 内落点 |
|---|---|---|
| varlen（前缀 mask） | `k ≥ kvLen[b] → −3e38` | softmax 相位构造前缀掩码行；或 per-batch s2End 截断（见下） |
| sink（StreamingLLM 虚拟列） | `m'=max(m,sink)`；`l' = l·exp(m−m') + exp(sink−m')` | softmax 播种：`maxUb←sink[head]`、`sumUb←1.0`，`SoftmaxFlashV2` isUpdate=true；O 归一化自动正确 |
| q8（W8A8） | int8×scale→fp16→正常 GEMM | 新增 dequant 相位：AIV 前置 int8→cast→scale→fp16 + 配对 flag 通知 AIC；仅 fp16 链路 |
| causal（下三角） | `j ≤ i − (c0−r0)` 行限可见列 | 对角 chunk 逐行尾掩（SetValue→NEG_LARGE + PipeBarrier）；或 [1024,1024] 压缩 mask 向量加 |
| paged（block_table 分页） | K/V 物理布局 `[总块数,128,H,D]`，`block_table[bh][t]=物理块号` | bmm1/bmm2 从 chunk 级改 128 块级，K/V 指针查表解析 |

MLA 反向不在此册（见 flash-attention-grad 家族）；MLA 前向含 flash-decoding kv-split 归并（FD
rescale：仅 `maxKvSplitCoreNum != 1` 时，drain 之后按 `w=exp(m_j−m_final)` 归并各 split 的 m/l/O）。[^fa-mla-recipe]

# 用法

## 分核策略与基本块切分

三种 kernel 组织（选型判据）：[^fa-recipe-book]

- **S1 单遍组织**（Skv≤1024 默认路径）：每 task = 128 行 query 一锅，s2 一次扫完；三深流水
  `QK(task) ‖ softmax(task−2) ‖ PV(task−1)+divide(task−1)`，extraInfo[taskId%3]；单核融合 S/P 不全
  落 GM（mm1Res/stage1Res GM ping-pong 双缓冲）。`enableL1Reuse`（D∈[64,128] 开：两核共享 task 对、
  blockIdx/2 切分）。S1 对标杆 0.80–0.92x。
- **s1s2 双层组织**（长序列/varlen）：s2LoopCount 内层多遍 softmax；`L1Reuse` 时 splitFactor×2。
- **跨核 GM staging（FAInfer 形态）**：AIC/AIV 分核 + CrossCoreFlag，见主 concept。

MLA 任务切分：flash-decoding 三重索引（AIC/AIV 共用同一遍历），kv-split 并行是长序列性能关键；
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`（一 AIC 两 AIV 核组）。[^fa-mla-paged]

BNSD 复合特性的零拷贝重解释：BNSD `[B,H,S,D]` 在 host 侧重解释为 `batch=B·H, numHeads=1,
kvHeads=1`——`[BH,S,D]` 内存序与单头 FA 逐位相同，零拷贝（复合特性不加布局开销的关键）。v 提供免
转置入口避免 per-launch 标量 gather 转置（那是 0.19x 的坑）；host per-launch 的 tiling/blockTable/actSeq
H2D memcpy 上下文不变时缓存（重复 launch +20-40%）。[^fa-compound]

## 数据路径与存储层级

varlen TND 路径（生产级口径）：[^fa-varlen-recipe]

1. **seqlen 传累加和，kernel 内差分**：`qListGm/kvListGm` 是 int64 累加数组 [B]，
   `GetSeqQlenKvlenByBoidx` 差分出 per-batch 原始长度（b=0 直取）。
2. **TND 直读，不转块格式**：Q/K/V 按 `[ΣS,N,D]` 连续内存 + `s1SizeAcc/s2SizeAcc` 前缀偏移直读；
   尾块 `SetTail(real)` + softmax 尾列填 −inf。Q/O 直透传 `q + cu[b]·H·D`，不做全量转置。
3. **batch 边界不用掩码行**：per-batch `s2EndIdx` 截断（每 batch KV 循环上界=自己的 kvLen），天然
   不越界读下一 batch。
4. `needL1Carry`（KV 驻留 L1 跨 batch 复用）为 varlen 专属优化；SameAB（自注意力 Q=K=V）有专用头。

kfc 直调 workspace：`workspaceLimit=131072=8*128*128`；per-core 槽位 `coreIdx×WS_DB×(PRE_LAUNCH+1)+
slot×WS_DB`；FA-Paged 组件路径同公式。MLA workspace 按 fa-mla 契约 §3 实测最小尺寸分配。[^fa-recipe-book]

## 实现顺序

1. 按分流表选配方路径；先写 kernel_common（常量/结构体）再写 kernel、tiling、host。
2. 变体特性按「算法」表落点叠加（实测叠加近乎免费：sink +5%、varlen ±0%、q8 +0.3%、四合一 +4.3%）。
3. 三条死锁前提逐条核对（见「约束」）。
4. 双口径精度 + 变体专项验收（见「验证方法」）。

# 代码模式

**FA-Paged 组件路径 3 条关键修复（照写即一次通过）**：[^fa-paged-recipe]

1. AIC PV 的 `LayoutP` 必须 3 参 `(rowNum, stackSeqTile, 512)`（stride=512=blockStackNum×pagedBlockSize），
   非 2 参——stride 不匹配则只读到 1/4 的 P 行（O 只 4 head 对）。
2. secondary rescale 的 `isLast = (stackSeqCount−preLaunch == totalStackSeqNum−1)`（仅末块 finalize），
   非恒 1（恒 1 提前 finalize → 多块精度错/hang）；primary rescale isLast=0。
3. `BlockMmadPV(resource, L1_QK_SIZE)`（L1 偏移=QK 的 L1 分配大小），非 0 偏移。

**sink 校正向量算子链**（RescaleO 的 isLastStackTile 分支，每个算子后 `PipeBarrier<PIPE_V>()`）：
`备份 m_run → Maxs(m, sink) → Sub/Exp 得 exp(m_run−m_f) → Mul l → Adds/Muls/Exp 得 exp(sink−m_f) → Add l`。
sink 值=scale 后尺度直传；入口 14→19 参（新增 smax/ssum/sinkValue/hasSink/hasAux）。aux 输出
（smax/ssum）host 必须做 null 回退（内部缓冲），否则传 null → kernel 写 0 地址 → 507057 设备错。[^fa-compound]

**q8 dequant 相位**：AIV 逐 tile `DataCopy(GM→UB int8) → PipeBarrier（★MTE2→V）→ Cast<half,int8_t> →
Muls(scale)` 写回 fp16 staging；完成后 `Set(dqReady)` AIC QK 前 Wait（逐批一次）。scale 契约=gen 写
乘系数 s、kernel 用 1/s；量化 golden 用 `round(x·s)/s`（量化-反量化值）不是 `x/s`。[^fa-compound]

**MLA 两条纠错（手搓必读）**：①AIC/AIV 分流用 `#ifdef __DAV_CUBE__`/`#ifdef __DAV_VEC__`（非
`__mix__`+g_coreType 运行期判断）；②主循环 prologue+延迟 PV（首任务只 QK+softmax 无 PV，task n 的
PV 在 n+1 拍执行），非同块锁步——两种锁步/`__mix__` 变体均实测 CrossCoreFlag token 不平衡 → 死锁。[^fa-mla-recipe]

**FFA 差异要点**（与 paged recipe 同构之外的增量）：`#ifdef` 同文件并列分流（MLA 式）；
`resetBlockStart(kvStart, pagedBlockSize)` 每 task 重置块内游标；`gSink` 直传 epilogueOnlineSoftmax
（8 处调用点全部带 sink 参）；`IS_FD` 模板位 flash-decoding 每核只扫自己的 s2 段；空任务时 VEC 侧
`epilogueInitOut` 防读未写；LSE 走 `LayoutLse(totalQTokens, qHeads)`。[^ffa-recipe]

## FA-Paged 手搓完整配方全文（fa-paged-handcraft-recipe）
> 以下为 paged recipe 源文件全量正文：Step 1-11 端到端逐步骤（includes/组件选型、kernel 骨架、AIC/AIV body、实例化 entry、3 条关键修复、GM 槽位与 launch/build/tiling、mask 双循环、kernel_common.hpp 全文、fai_tiling.cpp、host 11 步）与验证状态。

> **用途**：**标准 FA / FA-BNSD / FA-Paged / GQA**（`O=softmax(scale·Q·Kᵀ[+mask])·V`，分页 KV）手搓 kernel
> 的 skill 生成依据——**端到端四文件**（kernel + kernel_common + tiling + host）全部在本 recipe 内，
> 闭卷生成不依赖任何外部文件。
> 以下每一步均为**已验证可编译+跑通+精度 PASS 的真实代码**（kv=128~204800 全 100% PASS，max_abs≤7.6e-06）。
>
> **构成**：`FAInferKernel` 四文件从零手搓的完整步骤 +
> 3 条实测修复（LayoutP stride / isLast flag / L1_QK_SIZE 偏移）。
>
> **★与 MLA 共有的死锁关键**（必读 fa-mla-handcraft-recipe §0 实测纠错）：
> - 分流用 `operator()<AIC>/<AIV>` 模板特化（非 `__mix__`+g_coreType runtime if）。
> - 延迟 PV（preLaunch 深度 2），非同块锁步。
> - CrossCoreFlag Set `<0x2,PIPE_X>` + bare Wait。
> - AIV `coreIdx=GetBlockIdx()/GetSubBlockNum()`。
>
> **算子族覆盖**：Step 1-7+9-11 = NO_MASK 全注意力（标准 FA/Paged/GQA/变长 batch per-batch seqlen）；
> Step 8 = MASK_SPEC（逐位）/ MASK_CAUSAL（因果）叠加。变长=per-batch actualQseqlen/actualKvseqlen 天然支持
> （任务切分逐 batch 重算，无需改 kernel）。sink/varlen/q8 等 kfc 路径叠加见特性册 §12.6/§12.12/§12.13。

---

### Step 1: includes + 类型别名 + 组件选型

```cpp
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_common.hpp"
#include "kernel_operator.h"

using namespace Catlass;
namespace cg = Catlass::Gemm;
namespace ce = Catlass::Epilogue;
namespace cl = Catlass::layout;

// 类型
using ElementQ=half; using ElementK=half; using ElementV=half; using ElementP=half; using ElementO=half;
using ElementS=float; using ElementOTmp=float; using ElementUpdate=float; using ElementMask=half;
using LayoutQ=cl::RowMajor; using LayoutK=cl::ColumnMajor; using LayoutV=cl::RowMajor;  // ★FA: V=RowMajor（非 MLA 的 Col）
using LayoutS=cl::RowMajor; using LayoutP=cl::RowMajor; using LayoutO=cl::RowMajor;
using LayoutOTmp=cl::RowMajor; using LayoutUpdate=cl::RowMajor; using LayoutMask=cl::RowMajor;

// 组件
using L1TileShape=GemmShape<128,128,128>;     // ★K=embed=128（MLA 是 576）
using L0TileShape=L1TileShape;
using BlockMmadQK=cg::Block::BlockMmad<cg::MmadAtlasA2FAIQK<true,false>,L1TileShape,L0TileShape,
    cg::GemmType<half,LayoutQ>,cg::GemmType<half,LayoutK>,cg::GemmType<float,LayoutS>>;
using BlockMmadPV=cg::Block::BlockMmad<cg::MmadAtlasA2FAIPV<true,false>,L1TileShape,L0TileShape,
    cg::GemmType<half,LayoutP>,cg::GemmType<half,LayoutV>,cg::GemmType<float,LayoutOTmp>>;
using EpilogueOnlineSoftmax=ce::Block::BlockEpilogue<ce::EpilogueAtlasA2OnlineSoftmax,
    cg::GemmType<half,LayoutP>,cg::GemmType<float,LayoutS>,cg::GemmType<half,LayoutMask>>;
// ★EpilogueRescaleO 需要 3 个 GemmType 参（+OTmpType），非 2 个
using EpilogueRescaleO=ce::Block::BlockEpilogue<ce::EpilogueAtlasA2RescaleO,
    cg::GemmType<half,LayoutO>,cg::GemmType<float,LayoutUpdate>,cg::GemmType<float,LayoutOTmp>>;
```

---

### Step 2: kernel 类骨架 + Params

```cpp
template <class BQK, class BPV, class EpSoft, class EpRes>
class FAInferKernel {
public:
    using ArchTag=Arch::AtlasA2;
    // ★Params 成员必须每行单独声明（GM_ADDR=__gm__ uint8_t* 不能逗号列表——bisheng 拒）
    struct Params {
        GM_ADDR q; GM_ADDR k; GM_ADDR v; GM_ADDR mask; GM_ADDR blockTables;
        GM_ADDR actualQseqlen; GM_ADDR actualKvseqlen; GM_ADDR o; GM_ADDR s; GM_ADDR p;
        GM_ADDR oTemp; GM_ADDR oUpdate; GM_ADDR tiling;
        CATLASS_DEVICE Params() {}
        CATLASS_DEVICE Params(GM_ADDR q_,GM_ADDR k_,GM_ADDR v_,GM_ADDR mask_,GM_ADDR bt_,GM_ADDR aq_,GM_ADDR ak_,
            GM_ADDR o_,GM_ADDR s_,GM_ADDR p_,GM_ADDR ot_,GM_ADDR ou_,GM_ADDR t_)
            : q(q_),k(k_),v(v_),mask(mask_),blockTables(bt_),actualQseqlen(aq_),actualKvseqlen(ak_),
              o(o_),s(s_),p(p_),oTemp(ot_),oUpdate(ou_),tiling(t_) {}
    };
    // ★分流：主模板声明（无 body）+ 两个特化。entry 调 fa(params) 自动编译期分派。
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <> CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params) { /* Step 3 */ }
    template <> CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params) { /* Step 4 */ }
};
```

---

### Step 3: AIC body（cube 核：QK matmul + PV matmul + CrossCoreFlag Set）

```cpp
template <> CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params) {
    // 3a. HardEvent 预置（25 个 token，§8.6 AIC 全表）
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0); AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2); AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4); AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6); AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0); AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    for(uint32_t e=0;e<8;e++) AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(e);
    for(uint32_t e=0;e<6;e++) AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(e);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID0);

    // 3b. Resource + 组件构造
    Arch::Resource<ArchTag> resource;
    // ★修复3: BlockMmadPV 构造传 L1_QK_SIZE（L1 偏移），非 0
    static constexpr uint32_t L1_QK_SIZE =
        BQK::L1TileShape::M * BQK::L1TileShape::K * sizeof(ElementQ) +
        BQK::L1TileShape::N * BQK::L1TileShape::K * sizeof(ElementK) * 2;
    BQK blockMmadQK(resource);
    BPV blockMmadPV(resource, L1_QK_SIZE);

    // 3c. 读 FATilingData + GlobalTensor
    __gm__ FATilingData* td = reinterpret_cast<__gm__ FATilingData*>(params.tiling);
    uint32_t batch=td->batch, qHeads=td->numHeads, kvHeads=td->kvHeads, embed=td->embeddingSize;
    uint32_t pagedBlockSize=td->blockSize, maxNumBlocksPerBatch=td->maxNumBlocksPerBatch;
    uint32_t totalTaskNum=td->totalTaskNum; uint32_t maskType=td->maskType; float scaleValue=td->scaleValue;
    AscendC::GlobalTensor<ElementQ> gQ; gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
    AscendC::GlobalTensor<ElementK> gK; gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
    AscendC::GlobalTensor<ElementK> gV; gV.SetGlobalBuffer((__gm__ ElementK*)params.v);
    AscendC::GlobalTensor<int32_t> gBT; gBT.SetGlobalBuffer((__gm__ int32_t*)params.blockTables);
    AscendC::GlobalTensor<int64_t> gAQ; gAQ.SetGlobalBuffer((__gm__ int64_t*)params.actualQseqlen);
    AscendC::GlobalTensor<int64_t> gAK; gAK.SetGlobalBuffer((__gm__ int64_t*)params.actualKvseqlen);
    AscendC::GlobalTensor<ElementS> gS; gS.SetGlobalBuffer((__gm__ ElementS*)params.s);
    AscendC::GlobalTensor<ElementP> gP; gP.SetGlobalBuffer((__gm__ ElementP*)params.p);
    AscendC::GlobalTensor<ElementOTmp> gOTmp; gOTmp.SetGlobalBuffer((__gm__ ElementOTmp*)params.oTemp);
    uint64_t strideQO=qHeads*embed, strideKV=kvHeads*embed;
    uint32_t groupSize=qHeads/kvHeads;
    uint32_t coreIdx=AscendC::GetBlockIdx(), coreNum=AscendC::GetBlockNum();
    Arch::CrossCoreFlag qkReady(QK_READY_ID), pvReady(PV_READY_ID), softmaxReady(SOFTMAX_READY_ID);
    constexpr uint32_t blockStackNum=4; constexpr int32_t preLaunch=2;

    // 3d. 任务遍历初始化
    uint32_t curBatch=0, preTotalTaskNum=0, curTotalTaskNum=0;
    uint64_t qBOffset=0, blockBOffset=0; int64_t qSeqlen=0, kvSeqlen=0;
    qSeqlen=gAQ.GetValue(curBatch); kvSeqlen=gAK.GetValue(curBatch);
    uint32_t curQNBlockTile=GetQNBlockTile(qSeqlen,groupSize);
    uint32_t qNBlockNumPerGroup=CeilDiv(groupSize,curQNBlockTile);
    uint32_t curQNBlockNum=qNBlockNumPerGroup*kvHeads;
    int64_t curQSBlockTile=GetQSBlockTile(kvSeqlen);
    uint32_t curQSBlockNum=CeilDiv(qSeqlen,curQSBlockTile);
    curTotalTaskNum=curQNBlockNum*curQSBlockNum;

    // 3e. 任务遍历主循环
    for (uint32_t taskIdx=coreIdx; taskIdx<totalTaskNum; taskIdx+=coreNum) {
        while (taskIdx>=curTotalTaskNum) {
            ++curBatch; preTotalTaskNum=curTotalTaskNum; qBOffset+=qSeqlen*strideQO; blockBOffset+=maxNumBlocksPerBatch;
            qSeqlen=gAQ.GetValue(curBatch); kvSeqlen=gAK.GetValue(curBatch);
            curQNBlockTile=GetQNBlockTile(qSeqlen,groupSize); qNBlockNumPerGroup=CeilDiv(groupSize,curQNBlockTile);
            curQNBlockNum=qNBlockNumPerGroup*kvHeads; curQSBlockTile=GetQSBlockTile(kvSeqlen);
            curQSBlockNum=CeilDiv(qSeqlen,curQSBlockTile); curTotalTaskNum+=curQNBlockNum*curQSBlockNum;
        }
        uint32_t tcb=taskIdx-preTotalTaskNum;
        uint32_t qSBlockIdx=tcb/curQNBlockNum, qNBlockIdx=tcb%curQNBlockNum;
        uint32_t qNBlockIdxCurGroup=qNBlockIdx%qNBlockNumPerGroup;
        uint32_t kvHeadIdx=qNBlockIdx/qNBlockNumPerGroup;
        uint32_t qHeadIdx=kvHeadIdx*groupSize+qNBlockIdxCurGroup*curQNBlockTile;
        uint64_t gmQOffset=qBOffset+qSBlockIdx*curQSBlockTile*strideQO+qHeadIdx*embed;
        uint32_t qSBlockSize=(qSBlockIdx==(curQSBlockNum-1))?(qSeqlen-qSBlockIdx*curQSBlockTile):curQSBlockTile;
        uint32_t qNBlockSize=(qNBlockIdxCurGroup==(qNBlockNumPerGroup-1))?(groupSize-qNBlockIdxCurGroup*curQNBlockTile):curQNBlockTile;
        uint32_t rowNum=qSBlockSize*qNBlockSize;
        uint32_t noMaskKvS=kvSeqlen, kvSLoopNumNoMask=CeilDiv(noMaskKvS,pagedBlockSize);
        uint32_t kvSLoopNumTotal=kvSLoopNumNoMask;
        int32_t totalStackSeqNum=CeilDiv(noMaskKvS,blockStackNum*pagedBlockSize);
        int32_t stackSeqCount=0; uint32_t stackSeqTile;
        LayoutQ layoutQTemp(rowNum,embed); LayoutK layoutKTemp(strideKV,blockStackNum*pagedBlockSize);
        LayoutV layoutVTemp(blockStackNum*pagedBlockSize,strideKV);
        blockMmadQK.loadQGM(gQ[gmQOffset],layoutQTemp,rowNum,qNBlockSize,qHeads);

        // 3f. 主循环（QK + 延迟 PV）
        for (uint32_t kvSIdx=0; kvSIdx<kvSLoopNumNoMask; kvSIdx+=blockStackNum) {
            stackSeqTile=(kvSIdx+blockStackNum>kvSLoopNumNoMask-1)?(noMaskKvS-kvSIdx*pagedBlockSize):(pagedBlockSize*blockStackNum);
            uint32_t curStackTileMod=stackSeqCount%(preLaunch+1);
            uint64_t gmSOffset=coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1)+curStackTileMod*WORKSPACE_BLOCK_SIZE_DB;
            GemmCoord shapeQK{rowNum,stackSeqTile,embed};
            blockMmadQK(gQ[gmQOffset],gK,gS[gmSOffset],gBT[blockBOffset],layoutQTemp,layoutKTemp,
                        shapeQK,kvSIdx,kvSLoopNumNoMask,pagedBlockSize,noMaskKvS,strideKV);
            Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(qkReady);
            if ((int32_t)kvSIdx>=preLaunch*blockStackNum) {  // 延迟 PV（preLaunch 深度）
                uint32_t nowkvSIdx=kvSIdx-preLaunch*blockStackNum;
                stackSeqTile=(nowkvSIdx+blockStackNum>kvSLoopNumNoMask-1)?(noMaskKvS-nowkvSIdx*pagedBlockSize):(pagedBlockSize*blockStackNum);
                uint32_t pvPP=(stackSeqCount-preLaunch)%(preLaunch+1);
                uint64_t gmPOffset=coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1)+pvPP*WORKSPACE_BLOCK_SIZE_DB;
                // ★修复1: LayoutP 必须 3-参 stride=512（与 softmax 写 P 的 stride 一致）
                LayoutP layoutPTemp(rowNum,stackSeqTile,512);
                GemmCoord shapePV{rowNum,embed,stackSeqTile};
                blockMmadPV(gP[gmPOffset],gV,gOTmp[gmPOffset],gBT[blockBOffset],layoutPTemp,layoutVTemp,
                            shapePV,nowkvSIdx,kvSLoopNumNoMask,pagedBlockSize,noMaskKvS,strideKV,softmaxReady);
                Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(pvReady);
            }
            stackSeqCount++;
        }

        // 3g. 次循环（drain 延迟 PV 尾巴）
        uint32_t maskedStartIdx=AlignUp(kvSLoopNumNoMask,blockStackNum);
        uint32_t preLaunchStackNum=preLaunch*blockStackNum;
        for (uint32_t kvSIdx=maskedStartIdx; kvSIdx<kvSLoopNumTotal+preLaunchStackNum; ) {
            if (kvSIdx>=preLaunchStackNum) {
                uint32_t nowkvSIdx=kvSIdx-preLaunchStackNum;
                stackSeqTile=(nowkvSIdx+blockStackNum>kvSLoopNumNoMask-1)?(noMaskKvS-nowkvSIdx*pagedBlockSize):(pagedBlockSize*blockStackNum);
                uint32_t pvPP=(stackSeqCount-preLaunch)%(preLaunch+1);
                uint64_t gmPOffset=coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1)+pvPP*WORKSPACE_BLOCK_SIZE_DB;
                LayoutP layoutPTemp(rowNum,stackSeqTile,512);  // ★修复1: 3-参 stride=512
                GemmCoord shapePV{rowNum,embed,stackSeqTile};
                blockMmadPV(gP[gmPOffset],gV,gOTmp[gmPOffset],gBT[blockBOffset],layoutPTemp,layoutVTemp,
                            shapePV,nowkvSIdx,kvSLoopNumNoMask,pagedBlockSize,noMaskKvS,strideKV,softmaxReady);
                Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(pvReady);
            }
            kvSIdx+=blockStackNum; stackSeqCount++;
        }
    }

    // 3h. drain AIC HardEvent（反预置，§8.6 全表 Wait）
    for(uint32_t e=0;e<8;e++) AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(e);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    for(uint32_t e=0;e<8;e++) AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(e);
    for(uint32_t e=0;e<6;e++) AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(e);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID0);
}
```

---

### Step 4: AIV body（vector 核：softmax + rescale + CrossCoreFlag Wait）

```cpp
template <> CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params) {
    // 4a. HardEvent 预置（10 个 token，FA AIV 表）
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0); AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4); AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

    // 4b. Resource + epilogue 构造
    Arch::Resource<ArchTag> resource;
    __gm__ FATilingData* td=reinterpret_cast<__gm__ FATilingData*>(params.tiling);
    uint32_t batch=td->batch,qHeads=td->numHeads,kvHeads=td->kvHeads,embed=td->embeddingSize;
    uint32_t pagedBlockSize=td->blockSize,maxNumBlocksPerBatch=td->maxNumBlocksPerBatch;
    uint32_t firstBatchTaskNum=td->firstBatchTaskNum,totalTaskNum=td->totalTaskNum,maskType=td->maskType;
    float scaleValue=td->scaleValue;
    AscendC::GlobalTensor<int64_t> gAQ; gAQ.SetGlobalBuffer((__gm__ int64_t*)params.actualQseqlen);
    AscendC::GlobalTensor<int64_t> gAK; gAK.SetGlobalBuffer((__gm__ int64_t*)params.actualKvseqlen);
    AscendC::GlobalTensor<ElementO> gO; gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
    AscendC::GlobalTensor<ElementS> gS; gS.SetGlobalBuffer((__gm__ ElementS*)params.s);
    AscendC::GlobalTensor<ElementP> gP; gP.SetGlobalBuffer((__gm__ ElementP*)params.p);
    AscendC::GlobalTensor<ElementOTmp> gOTmp; gOTmp.SetGlobalBuffer((__gm__ ElementOTmp*)params.oTemp);
    uint32_t groupSize=qHeads/kvHeads, embedRound=RoundUp(embed,BLOCK_SIZE);
    EpSoft epSoft(resource,scaleValue); EpRes epRes(resource);
    // ★AIV coreIdx 必须除以 subBlockNum（否则双子核都跑 task0→flag 死锁）
    uint32_t coreIdx=AscendC::GetBlockIdx()/AscendC::GetSubBlockNum(), coreNum=AscendC::GetBlockNum();
    Arch::CrossCoreFlag qkReady(QK_READY_ID), pvReady(PV_READY_ID), softmaxReady(SOFTMAX_READY_ID);
    constexpr uint32_t blockStackNum=4; constexpr int32_t preLaunch=2;

    // 4c. 任务遍历初始化（同 AIC，但 oBatchOffset 替代 qBOffset/blockBOffset）
    uint32_t curBatch=0,preTotalTaskNum=0,curTotalTaskNum=firstBatchTaskNum;
    int64_t oBatchOffset=0,qSeqlen=gAQ.GetValue(curBatch),kvSeqlen=gAK.GetValue(curBatch);
    uint32_t curQNBlockTile=GetQNBlockTile(qSeqlen,groupSize),qNBlockNumPerGroup=CeilDiv(groupSize,curQNBlockTile);
    uint32_t curQNBlockNum=qNBlockNumPerGroup*kvHeads; int64_t curQSBlockTile=GetQSBlockTile(kvSeqlen);
    uint32_t curQSBlockNum=CeilDiv(qSeqlen,curQSBlockTile); curTotalTaskNum=curQNBlockNum*curQSBlockNum;

    // 4d. 任务遍历主循环
    for (uint32_t taskIdx=coreIdx; taskIdx<totalTaskNum; taskIdx+=coreNum) {
        while (taskIdx>=curTotalTaskNum) {
            ++curBatch; oBatchOffset+=qSeqlen*qHeads*embed; preTotalTaskNum=curTotalTaskNum;
            qSeqlen=gAQ.GetValue(curBatch); kvSeqlen=gAK.GetValue(curBatch);
            curQNBlockTile=GetQNBlockTile(qSeqlen,groupSize); qNBlockNumPerGroup=CeilDiv(groupSize,curQNBlockTile);
            curQNBlockNum=qNBlockNumPerGroup*kvHeads; curQSBlockTile=GetQSBlockTile(kvSeqlen);
            curQSBlockNum=CeilDiv(qSeqlen,curQSBlockTile); curTotalTaskNum+=curQNBlockNum*curQSBlockNum;
        }
        uint32_t tcb=taskIdx-preTotalTaskNum;
        uint32_t qSBlockIdx=tcb/curQNBlockNum, qNBlockIdx=tcb%curQNBlockNum, qNBlockIdxCurGroup=qNBlockIdx%qNBlockNumPerGroup;
        uint32_t kvHeadIdx=qNBlockIdx/qNBlockNumPerGroup, qStartNIdx=kvHeadIdx*groupSize+qNBlockIdxCurGroup*curQNBlockTile;
        int64_t gmOffsetO=oBatchOffset+qSBlockIdx*curQSBlockTile*qHeads*embed+qStartNIdx*embed;
        uint32_t qSBlockSize=(qSBlockIdx==(curQSBlockNum-1))?(qSeqlen-qSBlockIdx*curQSBlockTile):curQSBlockTile;
        uint32_t qNBlockSize=(qNBlockIdxCurGroup==(qNBlockNumPerGroup-1))?(groupSize-qNBlockIdxCurGroup*curQNBlockTile):curQNBlockTile;
        uint32_t rowNum=qSBlockSize*qNBlockSize;
        uint32_t noMaskKvS=kvSeqlen, kvSLoopNumNoMask=CeilDiv(noMaskKvS,pagedBlockSize), kvSLoopNumTotal=kvSLoopNumNoMask;
        int32_t totalStackSeqNum=CeilDiv(noMaskKvS,blockStackNum*pagedBlockSize), stackSeqCount=0;
        uint32_t stackSeqTile, stackSeqTilePad=blockStackNum*pagedBlockSize;

        // 4e. 主循环（softmax + 延迟 rescale）
        for (uint32_t kvSIdx=0; kvSIdx<kvSLoopNumNoMask; kvSIdx+=blockStackNum) {
            stackSeqTile=(kvSIdx+blockStackNum>kvSLoopNumNoMask-1)?(noMaskKvS-kvSIdx*pagedBlockSize):(pagedBlockSize*blockStackNum);
            LayoutS layOutS(rowNum,stackSeqTile,stackSeqTilePad); LayoutP layOutP(rowNum,stackSeqTile,stackSeqTilePad);
            GemmCoord shapeQK{rowNum,stackSeqTile,embed};
            uint32_t curStackTileMod=stackSeqCount%(preLaunch+1);
            uint64_t gmOffsetS=coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1)+curStackTileMod*WORKSPACE_BLOCK_SIZE_DB;
            uint64_t gmOffsetP=gmOffsetS;
            Arch::CrossCoreWaitFlag(qkReady);
            epSoft(gP[gmOffsetP],gS[gmOffsetS],layOutP,layOutS,shapeQK,(stackSeqCount==0),qSBlockSize,qNBlockSize,curStackTileMod);
            Arch::CrossCoreSetFlag<0x2,PIPE_MTE3>(softmaxReady);
            if ((int32_t)kvSIdx>=preLaunch*blockStackNum) {  // 延迟 rescale
                stackSeqTile=(kvSIdx>=kvSLoopNumNoMask)?(noMaskKvS-(kvSIdx-preLaunch*blockStackNum)*pagedBlockSize):(pagedBlockSize*blockStackNum);
                LayoutO layoutO(qSeqlen,embed*qHeads);
                // ★LayoutOTmp 3-参 stride=embedRound（与 PV 写 OTmp 的 stride 一致）
                LayoutOTmp layoutOTmp(rowNum,embed,embedRound);
                GemmCoord shapePV{rowNum,embed,stackSeqTile};
                uint32_t curStackTileMod2=(stackSeqCount-preLaunch)%(preLaunch+1);
                uint64_t gmOffsetOTmp=coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1)+curStackTileMod2*WORKSPACE_BLOCK_SIZE_DB;
                Arch::CrossCoreWaitFlag(pvReady);
                epRes(gO[gmOffsetO],gOTmp[gmOffsetOTmp],layoutO,layoutOTmp,shapePV,qSBlockSize,qNBlockSize,
                      (stackSeqCount-preLaunch==0),0,curStackTileMod2);  // primary isLast=0
            }
            stackSeqCount++;
        }

        // 4f. 次循环（drain 延迟 rescale 尾巴）
        uint32_t maskedStartIdx=AlignUp(kvSLoopNumNoMask,blockStackNum), preLaunchStackNum=preLaunch*blockStackNum;
        for (uint32_t kvSIdx=maskedStartIdx; kvSIdx<kvSLoopNumTotal+preLaunchStackNum; ) {
            if (kvSIdx>=preLaunchStackNum) {
                uint32_t nowkvSIdx=kvSIdx-preLaunchStackNum;
                stackSeqTile=(nowkvSIdx+blockStackNum>kvSLoopNumNoMask-1)?(noMaskKvS-nowkvSIdx*pagedBlockSize):(pagedBlockSize*blockStackNum);
                LayoutO layoutO(qSBlockSize,embed*qHeads); LayoutOTmp layoutOTmp(rowNum,embed,embedRound);
                GemmCoord shapePV{rowNum,embed,stackSeqTile};
                uint32_t curStackTileMod2=(stackSeqCount-preLaunch)%(preLaunch+1);
                uint64_t gmOffsetOTmp=coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1)+curStackTileMod2*WORKSPACE_BLOCK_SIZE_DB;
                Arch::CrossCoreWaitFlag(pvReady);
                // ★修复2: isLast = (stackSeqCount-preLaunch==totalStackSeqNum-1)（仅末块 finalize），非恒1
                epRes(gO[gmOffsetO],gOTmp[gmOffsetOTmp],layoutO,layoutOTmp,shapePV,qSBlockSize,qNBlockSize,
                      (stackSeqCount-preLaunch==0),(stackSeqCount-preLaunch==totalStackSeqNum-1),curStackTileMod2);
            }
            kvSIdx+=blockStackNum; stackSeqCount++;
        }
    }

    // 4g. drain AIV HardEvent（反预置，顺序照抄 4a 预置表）
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4); AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
}
```

---

### Step 5: 实例化 + entry

```cpp
using FaKernel = FAInferKernel<BlockMmadQK,BlockMmadPV,EpilogueOnlineSoftmax,EpilogueRescaleO>;

CATLASS_GLOBAL void FAInferFp16(uint64_t hwSync, GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR mask,
    GM_ADDR blockTables, GM_ADDR o, GM_ADDR qSeq,GM_ADDR kvSeq, GM_ADDR s,GM_ADDR p,GM_ADDR oTemp,
    GM_ADDR oUpdate, GM_ADDR tiling) {
    AscendC::SetSyncBaseAddr(hwSync);
    FaKernel::Params params{q,k,v,mask,blockTables,qSeq,kvSeq,o,s,p,oTemp,oUpdate,tiling};
    FaKernel fa;
    fa(params);  // operator()<g_coreType>() — 编译期分派（AIC on cube, AIV on vector）
}
```

---

### Step 6: 3 条关键修复（照写即一次通过）

1. **★修复1（FA 独有坑）**：AIC PV 的 `LayoutP layoutPTemp` **必须 3-参 `(rowNum, stackSeqTile, 512)`**（stride=512=blockStackNum*pagedBlockSize）。
   非 2-参 `(rowNum, stackSeqTileRound)`（默认 stride≠512）。因 AIV softmax **写** P 用 3-参 `LayoutP(rowNum, stackSeqTile, stackSeqTilePad=512)` stride=512，
   PV 读 P 的 stride 必须匹配，否则只读到 1/4 的 P 行（row 0,4,8,12）→ O 只 4 head 对。MLA 无此问题（MLA P layout 统一 3-参）。

2. **★修复2**：secondary rescale 的 `isLast` = `(stackSeqCount-preLaunch==totalStackSeqNum-1)`（仅末块 finalize），
   非恒 `1`（恒 1 提前 finalize → 多块精度错/hang）。primary rescale 的 isLast = `0`。

3. **★修复3**：`BlockMmadPV(resource, L1_QK_SIZE)`（L1 偏移=QK 的 L1 分配大小），非 `BlockMmadPV(resource, 0)`。
   `L1_QK_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementQ) + L1TileShape::N * L1TileShape::K * sizeof(ElementK) * 2`。

---

### Step 7: GM 槽位 + launch + build + tiling

```
GM workspace slots（preLaunch+1=3 槽 pingpong）:
  gmSOffset/gmPOffset/gmOTmpOffset = coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1) + curStackTileMod*WORKSPACE_BLOCK_SIZE_DB
  (WORKSPACE_BLOCK_SIZE_DB=131072)

workspace sizes:
  s  = aicCore * WS_DB * 3 * sizeof(float)
  p  = aicCore * WS_DB * 3 * sizeof(half)
  oTemp = aicCore * WS_DB * 3 * sizeof(float)
  oUpdate = aicCore * WS_DB * 3 * sizeof(float)

launch: FAInferFp16<<<aicCoreNum,nullptr,stream>>>(hwSync,q,k,v,mask,blockTables,o,qSeq,kvSeq,s,p,oTmp,oUpdate,tiling)
  14 params (hwSync + 13 GM_ADDR)

tiling: FAInferTiling::GetFATilingParam(FAInfo, blockDim, FATilingData)
  FAInfo: numTokens, numHeads, embeddingSize, numBlocks, blockSize, kvHeads, batch,
          qSeqlenList(int64_t*), kvSeqlenList(int64_t*), maskType

build: CATLASS_ARCH=2201 + --npu-arch=dav-2201
  host .asc #include "fai_kernel.cpp" 单 TU
  link: ascendc_runtime ascendcl runtime profapi mmpa c_sec error_manager graph_base tiling_api register platform unified_dlog dl m
  fai_tiling.cpp 单独 CXX TU（不 #include 进 host）
  fai_tiling.cpp 与 fai_kernel.cpp 都需要 kernel_common.hpp（定义 FATilingData + 常量）
```

---

### Step 8: mask 路径（MASK_SPEC=1 逐位 / MASK_CAUSAL=2 因果）——masked 双循环逐步

> Step 1-7 是 `maskType=NO_MASK(0)` 全注意力路径。mask 路径的改动集中在：
> **①任务级边界变量 ②次循环起点/推进 ③QKTail/PVTail 组件 ④AIV softmax 换带 mask 版调用**。
> 主循环（第一循环）体零改动（只是遍历域从 kvSeqlen 缩到 noMaskKvS）。

#### 8a. 组件增补（Step 1 追加）

```cpp
// ★Tail 组件名核对（catlass include dispatch_policy.hpp:175 实测存在）：
//   TailQK = MmadAtlasA2FAITailQK、TailPV = MmadAtlasA2FAITailPV
//   （"FAI"+"Tail"+"QK/PV"——勿写成 FAIQKTail/FAIPVTail，该名不存在，编译报未定义）
using BlockMmadQKTail=cg::Block::BlockMmad<cg::MmadAtlasA2FAITailQK<true,false>,L1TileShape,L0TileShape,
    cg::GemmType<half,LayoutQ>,cg::GemmType<half,LayoutK>,cg::GemmType<float,LayoutS>>;
using BlockMmadPVTail=cg::Block::BlockMmad<cg::MmadAtlasA2FAITailPV<true,false>,L1TileShape,L0TileShape,
    cg::GemmType<half,LayoutP>,cg::GemmType<half,LayoutV>,cg::GemmType<float,LayoutOTmp>>;
```

AIC 构造（与 QK/PV 并列，同款 L1 偏移）：
```cpp
BlockMmadQKTail blockMmadQKTail(resource);
BlockMmadPVTail blockMmadPVTail(resource, L1_QK_SIZE);
```

#### 8b. 任务级边界变量（替代 Step 3d/4d 的 `noMaskKvS=kvSeqlen`，写进任务循环内）

```cpp
uint32_t noSkipKvS = kvSeqlen;      // 因果上界：本 q 块最后一个 query 可见的 kv 长度
uint32_t noMaskKvS = kvSeqlen;      // 无掩码域长度（主循环遍历域）
uint32_t noMaskTailS = 0;           // 无掩码域的尾块残余（<128 部分）
if (maskType != 0) {
    uint32_t diffS = kvSeqlen - qSeqlen;                    // Sq≠Skv 时的 kv 前移量
    noSkipKvS = (qSBlockIdx + 1) * curQSBlockTile + diffS;  // 结构性跳块公式
    noSkipKvS = Min((uint32_t)kvSeqlen, noSkipKvS);
    noMaskKvS = noSkipKvS - qSBlockSize;                    // 对角块之前的整块域
    noMaskTailS = noMaskKvS % pagedBlockSize;
}
uint32_t maskedKvS = qSBlockSize;   // 对角（masked）块宽度 = q 块行数
uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);
int32_t totalStackSeqNum = (maskType != 0) ? (CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize) + 1)
                                           : CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize);
```

**语义**：`noMaskKvS` 域 = 全部可见无需 mask 的 KV 列（走主循环 + 普通 QK/PV 组件）；
`noSkipKvS..kvSeqlen` 的最后 `qSBlockSize` 列 = 对角块（走次循环 + QKTail/PVTail + mask）。
causal 不逐位算——整块跳过 + 仅对角块读 mask（`noSkipKvS` 公式即"跳块"）。

#### 8c. 次循环起点与推进规则（AIC/AIV 两侧同公式，替换 Step 3g/4f 的计算段）

```cpp
uint32_t maskedStartIdx = (maskType != 0) ?
        ((noMaskTailS != 0) ? (kvSLoopNumNoMask - 1) : kvSLoopNumNoMask)   // ★回退一格重算带尾块的对角前块
      : AlignUp(kvSLoopNumNoMask, blockStackNum);
uint32_t noMaskTailInteStackNum = (noMaskKvS / pagedBlockSize) % blockStackNum;
noMaskTailInteStackNum = (noMaskTailInteStackNum != 0) ? noMaskTailInteStackNum
                                                       : ((noMaskTailS != 0) ? 0 : blockStackNum);
uint32_t preLaunchStackNum = (maskType != 0) ? ((preLaunch - 1) * blockStackNum + noMaskTailInteStackNum)
                                             : (preLaunch * blockStackNum);
```

次循环 `for (kvSIdx = maskedStartIdx; kvSIdx < kvSLoopNumTotal + preLaunchStackNum; )` 的**推进**：
```cpp
if ((maskType != 0) && (stackSeqCount - preLaunch == totalStackSeqNum - 2)) {
    kvSIdx += noMaskTailInteStackNum;     // ★对角块前一次跳"残余整数块数"，非整 4 块
} else {
    kvSIdx += blockStackNum;
}
stackSeqCount++;
```

#### 8d. AIC 次循环体（masked 版）

```cpp
for (uint32_t kvSIdx = maskedStartIdx; kvSIdx < kvSLoopNumTotal + preLaunchStackNum;) {
    // ① QK（对角块）：stackSeqTile 恒 = maskedKvS；QKTail 多 2 个尾参 (noMaskTailS, 1)
    if ((kvSIdx < kvSLoopNumTotal) && (stackSeqCount <= totalStackSeqNum - 1)) {
        stackSeqTile = maskedKvS;
        uint32_t SPP = stackSeqCount % (preLaunch + 1);
        uint64_t gmSOffset = coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1) + SPP*WORKSPACE_BLOCK_SIZE_DB;
        GemmCoord shapeQK{rowNum, stackSeqTile, embed};
        blockMmadQKTail(gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset], gBT[blockBOffset], layoutQTemp,
                        layoutKTemp, shapeQK, kvSIdx, kvSLoopNumTotal, pagedBlockSize, noSkipKvS,
                        strideKV, noMaskTailS, 1);
        Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(qkReady);
    }
    // ② PV（延迟）：最后一块用 PVTail（带 mask 尾参），其余仍用普通 PV
    if (kvSIdx >= preLaunchStackNum) {
        uint32_t dKvSIdx = kvSIdx - preLaunchStackNum;
        stackSeqTile = (dKvSIdx + blockStackNum > kvSLoopNumTotal - 1 && maskType != 0) ? maskedKvS
                     : (dKvSIdx + blockStackNum > kvSLoopNumNoMask - 1) ? (noMaskKvS - dKvSIdx*pagedBlockSize)
                     : pagedBlockSize*blockStackNum;
        uint32_t PVPP = (stackSeqCount - preLaunch) % (preLaunch + 1);
        uint64_t gmPOffset = coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1) + PVPP*WORKSPACE_BLOCK_SIZE_DB;
        LayoutP layoutPTemp(rowNum, stackSeqTile, 512);          // ★同修复1
        GemmCoord shapePV{rowNum, embed, stackSeqTile};
        if ((stackSeqCount - preLaunch == totalStackSeqNum - 1) && (maskType != 0)) {
            blockMmadPVTail(gP[gmPOffset], gV[gmVOffset], gOTmp[gmPOffset], gBT[blockBOffset], layoutPTemp,
                            layoutVTemp, shapePV, dKvSIdx, kvSLoopNumTotal, pagedBlockSize, noSkipKvS,
                            strideKV, softmaxReady, noMaskTailS, 1);
        } else {
            blockMmadPV(gP[gmPOffset], gV[gmVOffset], gOTmp[gmPOffset], gBT[blockBOffset], layoutPTemp,
                        layoutVTemp, shapePV, dKvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS,
                        strideKV, softmaxReady);
        }
        Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(pvReady);
    }
    // ③ 推进（8c 规则）+ stackSeqCount++
}
```

#### 8e. AIV 次循环体（masked softmax 调用——签名不同，qkReady 传入 epilogue 内部 Wait）

```cpp
// NO_MASK 主循环（Step 4e）：外面显式 Wait —— CrossCoreWaitFlag(qkReady); epSoft(gP,gS,layOutP,layOutS,...);
// masked 次循环：gMask+layOutMask 进参，qkReady 作为最后参数传给 epilogue（内部按需 Wait，mask 懒加载）
if ((kvSIdx < kvSLoopNumTotal) && (stackSeqCount <= totalStackSeqNum - 1)) {
    stackSeqTile = maskedKvS;
    LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
    LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
    LayoutMask layOutMask(1024, 1024, 1024);        // ★固定压缩下三角 [1024,1024] fp16
    GemmCoord shapeQK{rowNum, stackSeqTile, embed};
    uint32_t SPP = stackSeqCount % (preLaunch + 1);
    uint64_t gmOffsetS = coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1) + SPP*WORKSPACE_BLOCK_SIZE_DB;
    epSoft(gP[gmOffsetS], gS[gmOffsetS], gMask, layOutP, layOutS, layOutMask, shapeQK,
           (stackSeqCount == 0), qSBlockSize, qNBlockSize, SPP, qkReady);
    Arch::CrossCoreSetFlag<0x2,PIPE_MTE3>(softmaxReady);
}
// rescale 段与 Step 4f 相同；isLast 判据同修复2：
//   (stackSeqCount - preLaunch == totalStackSeqNum - 1)
```

#### 8f. mask 张量契约

- mask = 压缩下三角 `[1024, 1024]` fp16 常驻（`mask[q][k] = 1 当 k>q 屏蔽`，语义 `score += mask·(−3e38)`，
  mask≠0 屏蔽、0 保留）；host 一次预置复用（勿每 launch 重建）。
- MASK_SPEC（逐位）：同一 epilogue 机制，mask 由 host 按 sparse band/prefix 语义预生成。
- MASK_CAUSAL：`noSkipKvS` 结构跳块 + 仅对角块走 QKTail/PVTail（数学同 §12.14-A 的 kfc 逐行尾部法，
  组件级实现）。Sq==Skv 时 leftUp/rightDown causal 等价。

---

### Step 9: kernel_common.hpp 全文（共用头，逐行照写）

```cpp
#ifndef KERNEL_COMMON
#define KERNEL_COMMON

constexpr uint32_t QK_READY_ID = 1;
constexpr uint32_t SOFTMAX_READY_ID = 2;
constexpr uint32_t PV_READY_ID = 3;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;

// 设备侧工具（CATLASS_DEVICE）
template <typename T> CATLASS_DEVICE T CeilDiv(T a, T b) { return (a + b - 1) / b; }
template <typename T> CATLASS_DEVICE T AlignUp(T a, T b) { return (b == 0) ? 0 : (a + b - 1) / b * b; }
template <typename T> CATLASS_DEVICE T RoundUp(T a, T b) { return (a + b - 1) / b * b; }
template <typename T> CATLASS_DEVICE T Min(T a, T b) { return (a > b) ? b : a; }
template <typename T> CATLASS_DEVICE T Max(T a, T b) { return (a > b) ? a : b; }

CATLASS_DEVICE uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    // N 维分块偶数化 trick：AIV 双子核行均衡（每子核 ≤64 行），辅助 rescale 编码
    uint32_t qNBlockTile = (128 / qSeqlen) / 2 * 2;
    qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
    qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
    return qNBlockTile;
}
CATLASS_DEVICE uint32_t GetQSBlockTile(uint32_t kvSeqlen) { return 128; }   // 固定 128

struct FATilingData {
    uint32_t numHeads = 0;            // Nq
    uint32_t embeddingSize = 0;       // D
    uint32_t numBlocks = 0;           // KV 物理总块数
    uint32_t blockSize = 0;           // pagedBlockSize（必须 128）
    uint32_t maxKvSeqlen = 0;
    uint32_t kvHeads = 0;             // Nkv（GQA group=Nq/Nkv）
    uint32_t batch = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    uint32_t firstBatchTaskNum = 0;
    uint32_t totalTaskNum = 0;
    uint32_t maskType = 0;            // 0=NO_MASK 1=MASK_SPEC 2=MASK_CAUSAL
    uint64_t mm1OutSize = 0;          // s workspace 字节数
    uint64_t smOnlineOutSize = 0;     // p workspace
    uint64_t mm2OutSize = 0;          // oTemp workspace
    uint64_t UpdateSize = 0;          // oUpdate workspace
    uint64_t workSpaceSize = 0;
    float scaleValue = 0.0;           // 默认 1/sqrt(D)（host 可覆盖）
};

struct FAIKernelParams {              // 13 个 GM_ADDR（tiling 在最后）
    GM_ADDR q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen,
            o, s, p, oTemp, oUpdate, tiling;
    // 构造函数逐成员初始化（bisheng 不吃逗号列表声明，每行一个成员）
};
#endif
```

> device 侧还有 `CeilDiv/RoundUp`（模板）；host 侧 tiling 用同名独立实现（见 Step 10）。
> `TILING_*` 索引常量表（TILING_BATCH=0…）是 **MLA kernel_common** 的内容，FA-Paged 不用（FA-Paged
> tiling 是结构体 `FATilingData` 整体 memcpy，MLA 是 fp32 数组 + 索引寻址——两算子契约不同勿混）。

---

### Step 10: fai_tiling.cpp（host 侧 tiling 计算，独立 CXX TU）

```cpp
// FAInfo：host 聚合输入
struct FAInfo {
    int32_t numTokens;        // Σ qSeqlen（生成 blockTable 用）
    int32_t numHeads;         // Nq
    int32_t embeddingSize;    // D
    int32_t numBlocks;        // KV 物理总块数 = Σ ceil(kvSeqlen_b/128)
    int32_t blockSize;        // 必须 128（否则报错返回）
    int32_t kvHeads;          // Nkv
    int32_t batch;
    int64_t* qSeqlenList;     // per-batch 原始长度（非累加！npu 累加口径 host 先差分）
    int64_t* kvSeqlenList;
    MaskType maskType;        // {NO_MASK=0, MASK_SPEC=1, MASK_CAUSAL=2}
};

int32_t GetFATilingParam(const FAInfo& faInfo, uint32_t blockDim, FATilingData& t)
{
    // ① 校验：seqlen 指针非空；blockSize==128（非 128 直接 -1）
    // ② maxKvSeqlen = max_b kvSeqlenList[b]
    // ③ FillBasicTilingData：
    //      maxNumBlocksPerBatch = ceil(maxKvSeqlen/blockSize)
    //      scaleValue = 1/sqrt(embeddingSize)（标杆自定义 scale 时 host 覆写此字段）
    //      直拷 batch/numHeads/kvHeads/embeddingSize/numBlocks/blockSize/maskType
    // ④ FillSplitCoreTilingData（与 kernel 侧 device 同公式，host 侧再算一遍）：
    //      for b in [0,batch):
    //        curQNBlockTile = GetQNBlockTile(qSeqlen_b, groupSize=Nq/Nkv)   // host 版 min/max 实现
    //        qNBlockNumPerGroup = ceil(groupSize/curQNBlockTile)
    //        curQNBlockNum = qNBlockNumPerGroup * Nkv
    //        curQSBlockNum  = ceil(qSeqlen_b/128)
    //        curTaskNum = curQNBlockNum * curQSBlockNum
    //        if b==0: firstBatchTaskNum = curTaskNum
    //        totalTaskNum += curTaskNum
    // ⑤ FillWorkSpaceTilingData（blockDim = aicCoreNum）：
    //      mm1OutSize     = blockDim * 131072 * 4 * 3   // s  : float, ×4B ×3槽
    //      smOnlineOutSize= blockDim * 131072 * 2 * 3   // p  : half, ×2B ×3槽
    //      mm2OutSize     = blockDim * 131072 * 4 * 3   // oTemp
    //      UpdateSize     = blockDim * 131072 * 4 * 3   // oUpdate
    //      workSpaceSize  = 四段之和（host 据此 malloc 4 块独立 buffer，指针透传，禁 GetUserWorkspace）
}
```

**host-kernel 一致性铁律**：④ 的任务数公式与 kernel device 侧 `while (taskIdx >= curTotalTaskNum)` 推进段
**逐字段相同**（同 GetQNBlockTile/同 ceil）——两侧任何一侧改动不同步 → totalTaskNum/firstBatchTaskNum 错位 →
AIC 与 AIV 解出不同的 (batch, qS, qN) → flag 死锁或精度全错。改切分公式必须双侧同改。

---

### Step 11: host fai.cpp 步骤（launch 管线，11 步顺序）

```cpp
// 1  aclInit(nullptr); aclrtSetDevice(deviceId); aclrtCreateStream(&stream);
// 2  aicCoreNum = PlatformAscendCManager::GetInstance()->GetCoreNumAic();   // A2=20
//    blockDim = aicCoreNum;
// 3  读输入 bin → aclrtMallocHost/aclrtMalloc(ACL_MEM_MALLOC_HUGE_FIRST) → ReadFile → aclrtMemcpy H2D：
//      q_ntokens.bin(int32×1)、q_seqlen.bin/kv_seqlen.bin(int64×batch)、q.bin/k.bin/v.bin、
//      block_table.bin(int32 × batch×maxNumBlocksPerBatch)、mask.bin（maskType!=0 时）
// 4  建 4 段 workspace（尺寸取自 tiling 的 mm1OutSize/smOnlineOutSize/mm2OutSize/UpdateSize）
//    + o（qoSize×2 bytes）+ tilingDevice（sizeof(FATilingData)）
// 5  tilingHost = aclrtMallocHost(sizeof(FATilingData)); 填 FAInfo → GetFATilingParam(faInfo, blockDim, faTilingData)
//    → aclrtMemcpy(tilingDevice, ..., tilingHost, ..., H2D)
// 6  aclrtGetHardwareSyncAddr(&hardwareSyncAddr);      // ★跨核同步基址，kernel 入参第 1 参
// 7  launch（fp16/bf16 双入口二选一）：
//      FAInferFp16<<<blockDim, nullptr, stream>>>(
//          hardwareSyncAddr, q, k, v, mask, blockTables, o,
//          actualQseqlen, actualKvseqlen, s, p, oTemp, oUpdate, tiling);   // ★14 参顺序固定
// 8  aclrtSynchronizeStream(stream);
// 9  aclrtMemcpy o D2H（fp16 元素 2B）
// 10 golden 比对（ReadFile golden.bin → 逐元素 max_abs/matched_ratio）
// 11 逆序 Free/FreeHost
```

> **中间槽必须 nullptr**（`<<<blockDim, nullptr, stream>>>`）——传实指针即 507015（同 §12.13 A-1）。
> workspace 4 段是**独立 malloc 的 4 个指针**分别传 s/p/oTemp/oUpdate（a2-a3 stage design §7：禁合并、禁
> GetUserWorkspace）。blockTable 生成：per-batch 物理块号连续递增（identity 时 `table[t]=t`）。

### 验证状态

> **★2026-09 闭卷验证（NO_MASK 全量通过）**：纯按本 recipe 手搓 `fai_kernel.cpp`（不参照任何外部文件），
> **编译通过 + 运行不死锁 + kv=128~204800(20万) 全部 100% PASS**（max_abs≤7.6e-06）。
> Step 1-7 + 3 条修复照写即可一次编译+跑通+精度 PASS。
>
> **Step 8（mask 路径）**：masked 双循环的边界变量/推进规则/组件调用/softmax 签名完整步骤；未单独闭卷回归——首次生成 masked 版时按 §8a-8f 逐步写，golden 用行限列因果参照。
> **Step 9-11（三件套）**：kernel_common.hpp / fai_tiling.cpp / fai.cpp
> 三个配套文件的完整内容与步骤，作为闭卷生成时的"第四块拼图"（此前 recipe 只覆盖 kernel 单文件，三件套要靠重新
> 发明——这是手搓翻车的主要残留点）。

## fa-sink / 标准 FA（S1 模板）配方全文（fa-sink-handcraft-recipe）
> 以下为 sink recipe 源文件全量正文：架构总览、成员与 UB 布局、三深流水 Process 主循环、轴解码、QK/softmax（sink 播种正确用法）/PV+divide 相位、sparse 跳块、workspace 精确公式、entry、tiling 与 kfc 选型对照。 [^fa-sink-recipe]

> **用途**：`O=softmax(scale·Q·Kᵀ[+mask])[+sink]·V` 在 **S1 单遍组织**（Skv≤1024 全场景默认路径，
> 配方册 §12.11 判据）下的 kernel 生成依据——生产级 S1 模板 kernel（~1800 行量级）的
> 核心相位完整拆解 + tiling 公式。
> **与 §12.11/§12.13 的关系**：§12.11 是本 kernel 的算法语义/tiling 公式/性能基线（S1 对标杆 0.80-0.92x）；
> §12.13 是 kfc chunk 流式自研配方（任意序列长，geomean 0.693）。本 recipe 补的是**kernel 类的真实
> 代码结构**（相位顺序/事件配对/sink 正确用法/workspace 公式精确版）——三份合起来闭卷可生成。
>
> **★sink 正确用法（本文件最有价值的一段，勿自创）**：sink 不加虚拟列、不改 GEMM——在 softmax 播种：
> `maxUb←sink[head]`、`sumUb←1.0`，然后 `SoftmaxFlashV2` 走 **isUpdate=true 模板参**（有 sink）vs
> `isUpdate=false`（无 sink，首块自初始化）。m/l 递推由 SoftmaxFlashV2 内部完成（见 §5 SoftMaxCompute）。
>
---

### 0. 架构总览（单核融合 + 三深流水，与配方册 §12.10 一致——此处为代码级精确版）

```
每核 task 段 = [multiCoreInnerOffset, multiCoreInnerLimit)（连续段，非 round-robin）
每 task = (boIdx, n2oIdx, goIdx, s1oIdx)：一锅 128 行 query（S1 单遍：s2 一次扫完）
三深流水（extraInfo[taskId%3]）：QK(task) ‖ softmax(task-2) ‖ PV(task-1)+divide(task-1)
S/P 不落 GM 全程？——否：S 落 per-core GM workspace（mm1Res 双缓冲），P 落 stage1Res（bmm2 的 A）。
  mm1Res/stage1Res/mm2Res 都在 GM，但 ping-pong 双缓冲让搬运与计算重叠。
```

**模板参数**（类 `FlashAttentionScoreS1Bn2gs1<implMode, layOutType, hasPse, hasAtten, hasDrop, …>`）：
`layOutType ∈ {LAYOUT_BSH, LAYOUT_SBH, LAYOUT_BNSD}`（三布局偏移公式在 §3/§6）；`enableL1Reuse`
（D∈[64,128] 开：两核共享 task 对、blockIdx/2 切分、奇偶跳过——见 §2）；bmm1Format/bmm2Source
（ND/NZ/CUBE 格式路由，dSize%16≠0 走 NZ 路）。

---

### 1. 成员与 UB 布局（InitBuffer 尺寸）

```cpp
TBuf<> maskTBufPing;   // 11K  atten mask 装载（hasSink 时复用作 softmax exp 输出！见 §5）
TBuf<> maskTBufPong;   // 11K  drop mask 装载
TBuf<> pseTBuf;        // 16K  pse 装载 / Vec2 阶段复用作 softmaxSum 装载
TBuf<> stage1PingBuf;  // 32K  softmax 工作区（fp32 S）+ cast 源
TBuf<> stage1PongBuf;  // 32K  bmm1 ND 格式直读 / Vec2 的 bmm2Res
TBuf<> vecOut;         // 16K  NZ 路输出
TBuf<> softmaxSumBuf;  // 8K   跨 splitN 积攒的 sum（[softmaxCopyOutLimit][vecS1BaseSize][8] fp32）
TBuf<> softmaxMaxBuf;  // 8K   同上 max
TBuf<> softmaxExpBuf;  // 8K   ★仅作 SoftmaxFlashV2 的占位参数（hasSink 时换 maskTBufPing）
TBuf<> commonTBuf;     // 32K  SoftmaxFlashV2 apiTmpBuffer / Vec2 输出 ping-pong
GlobalTensor<T>        mm1Res[2], mm2Res[2];         // GM 双缓冲
GlobalTensor<INPUT_T>  stage1Res[2];                 // P（bmm2 的 A）
```

---

### 2. Process()：三深流水主循环（逐段）

```cpp
int64_t offset = blockIdx * splitFactorSize;                  // 连续 task 段
if constexpr (enableL1Reuse) offset = blockIdx / 2 * splitFactorSize;   // L1R：两核一 task 对
int64_t limit = Min(offset + splitFactorSize, totalSize);
GetS1LoopRange(offset, limit);                                // sparse 负载均衡（sparseStartIdx 表，§7）
SplitS1dExtraInfo extraInfo[3];
int64_t taskId = 0;
event_t evMte3ToMte2 = FetchEventID(HardEvent::MTE3_MTE2);
bool notSecondLast = true, notLast = true;
limit += 2;                                                   // ★多跑 2 个空 task 位给流水 drain
bool unPair = false, lastNotPair = false;
if constexpr (enableL1Reuse) {
    limit += 2;                                               // L1R 再多 2（task 对奇偶）
    if ((limit - offset) % 2) { limit += 1; unPair = true; }
}
for (int64_t idx = offset; idx < limit; ++idx) {
    // 末尾标记（L1R 与非 L1R 的判据差 2——因 task 对）
    if constexpr (enableL1Reuse) {
        if (idx == limit - 4) notSecondLast = false; else if (idx == limit - 2) notLast = false;
        if ((idx - offset) % 2 != blockIdx % 2) continue;     // ★L1R 奇偶跳过：每核只跑自己奇偶的 task
    } else {
        if (idx == limit - 2) notSecondLast = false; else if (idx == limit - 1) notLast = false;
    }
    if (idx + 1 == limit - 4) lastNotPair = unPair;
    bool notLastTwoLoop = notSecondLast && notLast;

    ComputeAxisIdx(idx);                                      // §3
    GetS2LoopRange();                                         // §7 causal/band 跳块
    if (taskId >= 1 && notLast) WaitBmm1Result();             // bmm1.WaitIterateAll()+End()（task-1 的 QK）
    if (notLastTwoLoop) {
        SetExtraInfo(extraInfo[taskId%3], taskId, 0, 0, L1R? idx/2 : idx, lastNotPair);
        IterateBmm1(extraInfo[taskId%3]);                     // 提交 task 的 QK（异步 IterateAll<false>）
    }
    if (taskId > 0 && notLast) {
        ProcessVec1(extraInfo[(taskId+2)%3]);                 // softmax task-2 的 S
        SetFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2);          // P 写 GM 可见
    }
    if (taskId > 1) {                                         // PV task-1 的 P
        (dSizeAlign16 == dSize) ? WaitBmm2Result(bmm2) : WaitBmm2Result(bmm2Nz);
    }
    if (taskId > 0 && notLast) {
        WaitFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2);
        IterateBmm2(extraInfo[(taskId+2)%3], dSizeAlign16==dSize ? bmm2 : bmm2Nz);
    }
    if (taskId > 1) ProcessVec2(extraInfo[(taskId+1)%3]);     // divide task-1 的 O
    ++taskId;
}
```

**三深流水账目**（自查）：每 task 恰好 1×{WaitBmm1, IterateBmm1, ProcessVec1+Set(MTE3_MTE2),
Wait(MTE3_MTE2)+IterateBmm2, WaitBmm2, ProcessVec2}（首尾 task 按 taskId>0/>1 与 notLast 剪掉）。
多算的 2（L1R 4）个空 task 只走 Wait/ProcessVec 消费侧——**不是 bug，是 drain**。

---

### 3. 轴解码：ComputeAxisIdx

```cpp
boIdx  = idx / n2GS1o;                       // batch
n2oIdx = idx % n2GS1o / gS1o;                // kv head（n2=N2 kvHeads）
goIdx  = idx % gS1o / s1OuterSize;           // GQA 组内 head 块
s1oIdx = idx % s1OuterSize;                  // query 128 行块
// n2GS1o = n2Outer*gOuter*s1Outer（tiling coreParams 派生）；GQA：qHead = n2oIdx*gSize + goIdx
```

---

### 4. QK 相位：IterateBmm1 + A/B 装载

```cpp
// SetOrgShape 仅在 s1RealSize 变化或 sparse 时重设（省 API 开销；lastVec1S1RealSize 记忆）
if (s1RealSize != lastVec1S1RealSize || sparseType > 0) {
    bmm1.SetOrgShape(s1RealSize, mm1Kb, mm1Ka, mm1Kb, s2RealSize);   // 5 参：M,N,K,K2,N2
    lastVec1S1RealSize = s1RealSize;
}
Bmm1SetTensorA(extraInfo);      // A=Q：三布局偏移（BNSD: b*n2GS1D + n2o*gS1D + go*s1D + s1o*s1BaseD）
SetBmm1TensorB(extraInfo);      // B=K：SetTensorB(keyGm[kOff], true)（★transB 运行时 flag）+ SetTail(M,N,K)
bmm1.template IterateAll<false>(mm1Res[taskIdMod2], false, false, true);   // 异步；L1R 多一个 lastNotPair 参
```

**布局偏移公式表（BNSD 版；BSH/SBH 按同构规则对称推导）**：
- Q(BNSD)：`bOffset=b·n2S1D…`——实际成员名 n2GS1D/gS1D/s1D/s1BaseD（=H*S1*D 家族，ComputeConstexpr 算好）
- K(BNSD)：`b·n2S2D + n2o·s2D + s2StartIdx·dSize`
- V(BNSD)：同 K 公式（s2StartIdx 起）——causal 时 s2StartIdx>0（跳块）

---

### 5. softmax 相位：ProcessVec1（外层 splitN 循环）+ SoftMaxCompute

#### 5.1 ProcessVec1 每 loopIdx（realSplitN = ⌈s2RealSize/列预算⌉ 次）的顺序（★顺序即正确性）

```
① vecS1TailSize 尾块修正（loopIdx==realSplitN-1 → s1RealSize - loopIdx*vecS1BaseSize）
② loopIdx>0: WaitFlag(V_MTE2, evB)                          // 上一轮 P copyout 排空
③ hasPse: PseCopyIn/PseSlopeCopyIn（外mul内add 两种 pseType 分支）
④ GetBmm1Result：DataCopy(mm1Res → UB)（ND 直拷 / NZ 走 NzToNd）+ PipeBarrier<PIPE_V>
   + SetFlag/WaitFlag(MTE2_V)                                // MTE2→V 屏障
⑤ Muls(stage1Ping, S, scaleValue, vecS1TailSize*s2RealSizeAlign64)   // scale
   （PSE_OUTER_ADD_MUL_TYPE 时 scale 延后到 mask 后——保持 add-then-scale 语义）
⑥ CopyInAttenMask(-1)（hasAtten：mask GM→UB）
⑦ ComputeAttenMask（score += mask·negFloat；BAND 二段/PREFIX AND 合并两趟）
⑧ loopIdx<realSplitN-1 且 !hasSink: SetFlag(V_MTE2, evB)     // ★sink 时此 Set 移到 SoftMaxCompute 后
⑨ SoftMaxCompute（5.2）
⑩ loopIdx<realSplitN-1: SetFlag(V_MTE2, evA)                 // bmm1Res UB 复用前放行
⑪ loopIdx>0: WaitFlag(MTE3_V) + PipeBarrier<PIPE_V>
⑫ P 落 GM：T≠INPUT_T（fp16 输出）→ Cast(ROUND) + DataCopy(stage1Res[taskIdMod2][loopIdx*s1BaseTimesS2Align16])
    （2D DataCopyParams：blockCount=vecS1TailSize, blockLen=s2Align16/16, srcStride=(s2Align64-s2Align16)/16）
    NZ 路：Cast → NdToNz（nd2nzParams：dstNzC0Stride=CeilDiv(row,16)*16）
⑬ loopIdx<realSplitN-1: SetFlag(MTE3_V)
```

#### 5.2 SoftMaxCompute（sink 正确用法在此）

```cpp
// ① 尾列填 -inf（hasAtten==false 或非压缩 mask 或 s2RealSize<64 时）：
//    Duplicate(srcTensor[s2RealSizeFloorAlign8], negFloatScalar, mask, vecS1TailSize, 1, s2Align64*elems/blockBytes)
// ② srcTensor.SetShapeInfo(ShapeInfo(2, {vecS1TailSize, s2Align64}, 2, origin, ND)) + SetSize
// ③ sumUb/maxUb 取 softmaxSumBuf/MaxBuf[ (loopIdx % softmaxCopyOutLimit)*vecS1BaseSize*fp32BaseSize ]
//    （8K buf 攒 softmaxCopyOutLimit 轮再统一写 GM——S1>256 时循环复用）
// ④ ★sink 播种（hasSink）：
float inMax = sinkGm.GetValue(n2oIdx * gSize + goIdx);      // sink=[H] fp32，按 head 索引（scale 后尺度）
float inSum = 1.0;
SetFlag<HardEvent::S_V>(ev); WaitFlag<HardEvent::S_V>(ev);   // 标量读回屏障
Duplicate(sumUb, (T)inSum, vecS1TailSize * fp32BaseSize);    // l₀=1（★指数域：等价 l₀=exp(sink-m₀)·e^{m₀} 记法）
Duplicate(maxUb, (T)inMax, vecS1TailSize * fp32BaseSize);    // m₀=sink
// ⑤ SoftMaxTiling = SoftMaxFlashV2TilingFuncImpl(vecS1TailSize, s2Align64, sizeof(T), sizeof(T),
//        tmpSize, /*isUpdate=*/false, /*isBasicBlock=*/true)
// ⑥ SoftmaxFlashV2 调用（★第 2 模板参 isUpdate 有无 sink 不同）：
if (hasSink) {
    expUb = maskTBufPing;                                    // ★sink 时 exp 输出换 buffer（避 state 冲突）
    SoftmaxFlashV2<T, /*isUpdate=*/true,  true, true, false, SOFTMAX_DEFAULT_CFG>(
        srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer, newTiling);
} else {
    SoftmaxFlashV2<T, /*isUpdate=*/false, true, true, false, SOFTMAX_DEFAULT_CFG>(
        srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer, newTiling);
}
// in-place（dst=src 首参）；expUb 布局不可依赖（配方册 §12.13-C 同款结论——α 不从 expMax 读）
// ⑦ softmax sum/max 攒批写 GM：loopIdx==realSplitN-1 || (loopIdx+1)%softmaxCopyOutLimit==0 时
//    V_MTE3 Set/Wait → DataCopy(softmaxSumGm/MaxGm[gmOffset], buf[0], calculateSize) → MTE3_V Set
//    gmOffset = b·n2GS1·8 + n2o·gS1·8 + go·s1Size·8 + s1o·s1Base·8 + softmaxOutOffset·8
```

**sink 数学核对**（StreamingLLM）：`m_f=max(m,sink)`、`l_f=l·exp(m−m_f)+exp(sink−m_f)`——本配方用
播种+isUpdate 把这一递推交给 SoftmaxFlashV2 内部；`l₀=1,m₀=sink` 的初态在指数域等价于虚拟 sink 列
（exp(sink−m)=…）。**不要在 kernel 里手写 max/add 链复刻**（特性册 §12.6-E 的手写链是 kfc 路径的替代品，
S1 模板路径用 SoftmaxFlashV2 组件形态）。

---

### 6. PV + divide 相位：IterateBmm2 + ProcessVec2

```cpp
// IterateBmm2：A=stage1Res[taskIdMod2]（softmax 的 P，GM）；B=V（SetTensorB 无 transFlag）
if (s1RealSize != lastVec2S1RealSize || sparseType > 0) {
    bmm2.SetOrgShape(s1RealSize, mm2Kb, s2RealSizeAlign16, mm2Kb, dSize);
    lastVec2S1RealSize = s1RealSize;
}
bmm2.SetTensorA(stage1Res[taskIdMod2]);
bmm2.SetTensorB(valueGm[vCoreOffset]);            // TSCM 路：V 装载驻留 L1（bmm2Source==TSCM）
bmm2.SetTail(s1RealSize, dSize, s2RealSize);      // ★通过 tail 限制本次 K 维=s2RealSize
bmm2.template IterateAll<false>(mm2Res[taskIdMod2], false, false, true);

// ProcessVec2：每 vec2S1Idx（vec2S1BaseSize=tiling coreParams.s1Vec2BaseSize，D 小时可 >128）
// ① DataCopy(mm2Res → bmm2ResUb)；dSize%16≠0：NzToNd（nz2NdInfo 五字段）回 ND
// ② DataCopy(softmaxSumGm[sumGmOffsetLoop] → sumUb)         // ★从 GM 读回累计 l（不是 UB 里留）
// ③ Bmm2ResultDiv：Brcb(expSumUb, sumUb) → Div(bmm2Res, bmm2Res, expSumUb)（行广播除）
//    （AA_INVALID_LINE_HIGH_PRECISION 模式先 AdjustSoftMaxRes 修正无效行）
// ④ Bmm2DataCopyOut：Cast(fp16, ROUND) → DataCopy(attentionOutGm[qCoreOffset + vec2S1 偏移])
//    用 commonTBuf 双 ping-pong（[0]/[8192]）——搬运与上一轮 cast 重叠
// ⑤ 循环外：SetFlag/WaitFlag(MTE3_MTE2) 收尾配平
```

---

### 7. sparse/causal 跳块：GetS1LoopRange + GetS2LoopRange

```cpp
// GetS1LoopRange（sparse>0）：per-core 段改读 multiCoreParams.sparseStartIdx[blockIdx] 表
//   （host 预按三角面积均衡切好的 per-core 起点数组；L1R 时步进 2：sparseStartIdx[b] ~ [b+2]）
// GetS2LoopRange（每 task）：
CAUSAL（下三角）: s2StartIdx=0; s2EndIdx = Min((s1oIdx+1)*s1BaseSize, s2Size);
BAND（对角带）:   s2StartIdx = Max(s1oIdx*s1BaseSize - preTokens, 0)（dropmask bit 模式向下 8 对齐）
                  s2EndIdx   = Min((s1oIdx+1)*s1BaseSize + nextTokens, s2Size);
PREFIX/rightDown 类似 + ComputeOffsetForPrefixRectangle 的压缩 mask 索引
// ★causal 的省算力在 s2EndIdx 截断（整块跳过对角以上）；对角内逐位由 attenMask 压缩矩阵处理
//   （mask [2048,2048] 压缩格式，offset=ComputeOffsetForCausal(delta=Skv-Sq)：delta≤0 → 行偏移
//    Min(-delta, s1Base)；delta>0 → 列偏移 delta·attenMaskS2Size）
```

---

### 8. workspace 精确公式（InitInput——per-core 基址）

```cpp
s2SizeAlign16 = CeilDiv(s2Size, 16)*16;
mmNRatioOffset   = CeilDiv(s1BaseSize*s2SizeAlign16, 256)*256*sizeof(T);        // mm1Res 单份
stage1WsOffset   = mmNRatioOffset;                                               // stage1Res 单份
if (INPUT_T==float) stage1WsOffset = mmNRatioOffset*sizeof(INPUT_T)/2;           // fp32 稀疏不复用
bmm2ResultOffset = CeilDiv(s1BaseSize*dSizeAlign16, 256)*256*sizeof(T);          // mm2Res 单份
pseAlibiOffset   = CeilDiv(pseAlibiBaseS1*pseAlibiBaseS2*2, 512)*512;
totalOffset = blockIdx * (mmNRatioOffset*2 + stage1WsOffset + bmm2ResultOffset*2 + pseAlibiOffset);
mm1Res[0/1]     = ws + totalOffset + {0, mmNRatioOffset}
stage1Res[0/1]  = ws + totalOffset + 2*mmNRatioOffset + {0, stage1WsOffset/2}
mm2Res[0/1]     = ws + totalOffset + 2*mmNRatioOffset + stage1WsOffset + {0, bmm2ResultOffset}
pseAlibiGm      = ws + totalOffset + 2*mmNRatioOffset + stage1WsOffset + 2*bmm2ResultOffset
// 另：dropmask bit 模式时 workspace 头部还有 dropmaskParams.shapeTotalSize（512 对齐）——
//     InitInput 里 dropMaskGm 先占 workspace 头，后续偏移要跳过它
```

---

### 9. entry 形态（手搓直调，按 §12.11-B/§12.13-A）

**手搓直调**：单 TU `__global__ __mix__(1,2)` + `__kfc_workspace__` 第 8 参 + userWs 独立参数 +
`REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, &t.bmm1Tiling, op.bmm2, &t.bmm2Tiling)` +
`if ASCEND_IS_AIC return` + Init/Process（§12.11-B 14 参形态 / §12.13-A 11 参形态二选一）。
matmul tiling 按框架计算器 `matmul_tiling::MatmulApiTiling`（§12.13-A-2），**手填 TCubeTiling 挂死**。

---

### 10. tiling 公式（§12.11-D 全量公式即本 kernel 的 host tiling 提取；源头坐标）

`s1BB/s2BB/dBB、nRatioMax 4 分支（NZND 入口条件）、s1Ratio grow/shrink（workspaceLimit=131072）、
s1Base/s1Outer、coreParams/multiCoreParams、SetBmm1/Bmm2TilingInput（SetShape/SetOrgShape/SetFixsplit）、
shareL1Size=1MB/shareL0CSize=256KB` —— 逐公式见 fa-kernel-recipes.md §12.11-D。
S1 模板不开 enableL1Reuse、splitFactor 不 ×2——与 s1s2 的差异判据同 §12.11-A。
tilingKey 路由（fp16/bf16 × S1/S1S2/B 模板 × layout × ND/NZ 组合）仅 aclnn registry 分发用，
手搓直调不需要。

---

### 11. 与 §12.13（kfc chunk 流式）的选型对照

| 维度 | 本 recipe（S1 模板） | §12.13（kfc chunk） |
|---|---|---|
| 序列长 | Skv≤1024（s2 一次扫完） | 任意（chunk=1024 流式） |
| sink | SoftmaxFlashV2 isUpdate 播种（本文件 §5.2） | Duplicate 播种 m/l 状态（§12.13-C） |
| S/P 存储 | GM per-core 双缓冲（3 段 workspace） | GM per-worker 双缓冲（S/P 两段） |
| bmm2 累加 | 每 task 单次 IterateAll（O=mm2Res 再 ÷l） | enAtomic=1 跨 chunk 原子累加 O_acc |
| 实测 | 对标杆 sink 0.80x/0.922x（§12.11-H） | geomean 0.693（§12.13-E） |
| 手搓复杂度 | 相位多（pse/mask/drop 分支全） | 相位少（无 pse/drop，mask 走 pad+NEG_LARGE） |

**选型**：sink+短中序列→本 recipe；任意序列长/复合特性（varlen/q8 叠加）→§12.13 基座更省事。

### 12. 手搓自检（S1 模板版，在 §12.11-I 基础上补结构项）

- [ ] Process 三深流水的 taskId>0/>1 与 notLast/notSecondLast 剪枝逐项对齐 §2（多 2 空 task 是 drain，勿删）
- [ ] enableL1Reuse 开时：blockIdx/2 切分 + 奇偶 continue + limit 再 +2 三件套成组出现（§12.10-F 配对律）
- [ ] sink：播种 max=sink[head]/sum=1 + SoftmaxFlashV2 isUpdate=true + expUb 换 maskTBufPing（§5.2 三件缺一不可）
- [ ] hasSink 时 V_MTE2 evB 的 Set 在 SoftMaxCompute **之后**（§5.1-⑧ 的位置差异）
- [ ] SetOrgShape 仅 s1RealSize 变化时重设；SetTail 逐次（M,N,K 用 real/tail）
- [ ] workspace 布局按 §8 精确公式（256 对齐 + fp32 稀疏特例 + dropmask 头部特例）
- [ ] softmaxSum/Max 攒批写 GM 的 gmOffset 五段式（§5.2-⑦）与 Vec2 读回偏移一致
- [ ] dSize%16≠0 全链路走 NZ（bmm1Nz/bmm2Nz + NdToNz/NzToNd），dSize%16==0 走 ND——勿混

## fa-varlen（TND 变长）配方全文（fa-varlen-handcraft-recipe）
> 以下为 varlen recipe 源文件全量正文：两路对照、差分解码与跨 batch 推进、Process 主循环差异、TND 偏移公式、per-batch sparse 截断、needL1Carry、SameAB、host 契约与自检。

> **用途**：变长 batch（每 batch 实际 qSeqlen/kvSeqlen 不同，TND 布局 `[ΣS_b, N, D]`）的 FA kernel
> 生成依据——生产级 varlen kernel（~1200 行量级）核心差异完整拆解。
> **基座**：本配方 = **s1s2 双层组织**（s2LoopCount 内层）+ S1 kernel 同款三深流水/相位
> （QK/softmax/PV/divide 复用 fa-sink-handcraft-recipe.md §4-§6，
> 本文件只写**差异**）。**特性册 §12.12-A/A2 的 kfc 叠加路径**（前缀掩码行 + K/V 转块格式）是另一条
> 自研路径，§0 有两路对照。
>
> **★varlen 三个关键事实（与直觉/手册路径不同）**：
> 1. **seqlen 传累加和，kernel 内差分**：`actualQseqlen/actualKvseqlen` 是 npu 口径的**累加数组**
>    （`qListGm[b]`），`GetSeqQlenKvlenByBoidx` 差分出 per-batch 原始长度（b=0 直取）。无需 host 预差分。
> 2. **TND 直读，不转块格式**：Q/K/V 按 `[ΣS, N, D]` 连续内存 + `s1SizeAcc/s2SizeAcc` 前缀偏移直读
>    （§3 偏移公式）——**不做** paged 块化/转置。尾块靠 `SetTail(real)` + softmax 尾列填 -inf。
> 3. **batch 边界不用掩码行**：per-batch `s2EndIdx` 截断（sparse 公式 §4）——每个 batch 的 KV 循环
>    上界就是自己的 kvLen，天然不越界读下一 batch（区别于特性册 §12.12-A 的"-3e38 前缀掩码行"法，
>    那是等长 padded 布局的 kfc 路径做法）。
>
---

### 0. 两条 varlen 路径对照（生成前先选路）

| 维度 | 本 recipe（s1s2 双层直读路径） | 特性册 §12.12-A2（kfc + paged 基座） |
|---|---|---|
| 布局 | TND 直读零转换（Q/O 直透传，K/V 也是 TND） | Q/O 零拷贝直透传；**K/V 转块格式 [numBlocks,128,H,D]**（尾块 0 填充） |
| batch 边界 | s2EndIdx 截断（每 batch 循环上界=kvLen_b） | 前缀掩码行（k≥kvLen_b → -3e38）+ per-batch tiling 逐 launch |
| seqlen 契约 | 累加和数组进 GM，kernel 差分 | per-batch 原始长度进各自 tiling |
| 组织 | s1s2 双层（s2LoopCount 内层，多遍 softmax） | S1 单遍 / chunk 流式（单遍 softmax） |
| 适用 | 对齐标杆语义（npu_fusion_attention TND） | 复合特性叠加（q8 等）、块格式 KV 已有场景 |
| 实测 | 生产级路径口径 | varlen 1.12x vs 标杆（§12.12-D） |

---

### 1. 任务切分：差分解码 + 跨 batch 推进（逐段）

```cpp
// GM 输入：qListGm/kvListGm = int64 累加和数组 [B]（actualSeqLengths/actualSeqLengthsKv）
__aicore__ void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &qLen, int64_t &kvLen) {
    if (boIdx == 0) { qLen = qListGm.GetValue(0);  kvLen = kvListGm.GetValue(0); }
    else            { qLen = qListGm.GetValue(boIdx) - qListGm.GetValue(boIdx - 1);
                      kvLen = kvListGm.GetValue(boIdx) - kvListGm.GetValue(boIdx - 1); }
}

// 任务总数（host tiling 同公式）：totalSize = Σ_b CeilDiv(qLen_b, s1BaseSize) * n2G
//   （★按实际长度切，不是 padded 长度——短 batch 少占核）

// 核内段起点初始化 CalS1OuterSize(offset)：
s1OuterSizeAcc = 0; s1SizeAcc = 0; s2SizeAcc = 0; attenB1SSOffset = 0; boIdx = 0;
for (i = 0; i < bSize; i++) {
    GetSeqQlenKvlenByBoidx(i, s1Len, s2Len);
    acc = Σ_{j≤i} CeilDiv(s1Len_j, s1Base)*n2G;
    if (offset >= acc) {   // 本核段起点还在更后面的 batch → 累计四件套前移
        s1OuterSizeAcc = acc; s1SizeAcc += s1Len; s2SizeAcc += s2Len;
        attenB1SSOffset += s1Len*s2Len; boIdx++;
    } else break;
}

// 每 task 的 ComputeAxisIdx（while 跨 batch，非 if！）：
while (idx >= s1OuterSizeAcc + CeilDiv(s1Len_boIdx, s1Base)*n2G) {
    四件套前移（同上）; boIdx++; 重读 s1Len/s2Len;
}
n2oIdx = (idx - s1OuterSizeAcc) / CeilDiv(s1Len, s1Base) / gSize;
goIdx  = (idx - s1OuterSizeAcc) / CeilDiv(s1Len, s1Base) % gSize;
s1oIdx = (idx - s1OuterSizeAcc) % CeilDiv(s1Len, s1Base);
GetSeqQlenKvlenByBoidx(boIdx, this->s1Size, this->s2Size);   // 本 task 的实际 S1/S2
```

**四件套累计量的用途**：`s1SizeAcc/s2SizeAcc` = Q/K/V 的 batch 前缀偏移（§3）；`attenB1SSOffset` =
attenMask 的 batch 前缀偏移（per-batch S1·S2 累计）；`s1OuterSizeAcc` = 任务前缀。**四个必须同步推进**
（漏一个 → mask/数据错位到别的 batch，典型症状：个别 batch 输出全错）。

---

### 2. Process 主循环差异

```cpp
// 与 S1 版（fa-sink recipe §2）的差异：
// ① 多一层 s2 内层循环（s1s2 双层组织）：
int64_t s2LoopLimit = CeilDiv(s2EndIdx - s2StartIdx, s2BaseNratioSize) - 1;
for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
    if (s2LoopCount == 0) softmaxPingPongCnt++;           // 每 task 首块切 ping-pong
    ... 三深流水体（WaitBmm1 → SetExtraInfo(带 s2LoopCount/s2LoopLimit) → IterateBmm1
        → ProcessVec1[(taskId+2)%3] → WaitBmm2 → IterateBmm2 → ProcessVec2[(taskId+1)%3]）
}
// ② extraInfo 三槽跨 s2 内层复用（taskId 连续递增，不分层）
// ③ IterateBmm1 双形态路由：extraInfo.needNz2Nd==1 ? bmm1Nz : bmm1（fp32/bf16+尾块走 NZ）
// ④ varlen 无 enableL1Reuse 奇偶对（用 needL1Carry 替代，见 §5）
```

---

### 3. TND 偏移公式（CalcQCoreOffset/CalcKCoreOffset/CalcVCoreOffset）

```cpp
// TND [ΣS, N, D]：bOffset 用前缀长度累计（非 b*固定 stride！）
Q: qCoreOffset = s1SizeAcc*n2GD + s1oIdx*s1BaseN2GD + n2oIdx*gD + goIdx*dSize
K: kCoreOffset = s2SizeAcc*n2D  + s2StartIdx*n2D + s2LoopCount*s2BaseNratioN2D + n2oIdx*dSize
V: vCoreOffset = s2SizeAcc*n2D2 + s2StartIdx*n2D2 + ... + n2oIdx*dSize
attenMask:     attenB1SSOffset（= Σ_{j<b} s1Len_j*s2Len_j）+ s1oIdx 块内偏移
// n2GD/n2D = N*D 家族常量（ComputeConstexpr）；★Q 与 K 的 bOffset 用不同累计量（s1SizeAcc vs s2SizeAcc）
```

---

### 4. per-batch sparse 截断（GetS2LoopRange——全按 actualS2Len 截断）

```cpp
GetSeqQlenKvlenByBoidx(boIdx, s1Len, s2Len);              // ★每 task 现读
CAUSAL（下三角）:        s2Start=0; s2End=Min((s1oIdx+1)*s1BaseSize, s2Len);
RIGHT_DOWN_CAUSAL:       s2Start=0; s2End=Min((s1oIdx+1)*s1BaseSize + s2Len - s1Len, s2Len);
                          // ★Sq≠Skv 右对齐 causal 的偏移项 actualS2Len-actualS1Len（等价 sparse_mode=3）
BAND:                    s2Start=Max(s1oIdx*s1Base - s1SparseValidSize, 0);
                         s2End=Min((s1oIdx+1)*s1Base + s2SparseValidSize, s2Len);
                         if (s2End - s2Start <= 0) { s2Start=0; s2End=s2Len; }   // 整块无效→退全扫
BAND_COMPRESS:           带 preTokens/nextTokens 的压缩 mask 偏移
无 sparse（纯 varlen）:  s2Start=0; s2End=s2Len;          // ★batch 边界即此——不越界读下一 batch
```

**尾块两处收口**：① s2RealSize<64 或非压缩 mask 时 softmax 前尾列 `Duplicate(-inf)`（fa-sink recipe
§5.2-① 同款）；② bmm1/bmm2 `SetTail(s1RealSize, s2RealSize, dSize)` 逐次设。**不需要块格式 0 填充**
（与特性册 §12.12-A2 的分块路径不同点）。

---

### 5. needL1Carry（KV 驻留 L1 跨 batch 复用——varlen 专属优化路径）

```cpp
// host PostTiling 判定：
needL1Carry = !isSameAB && dtype != fp32 && !hasRope
              && s1BasicBlock <= S1_BASIC_BLOCK_L1CARRY_MAX
              && dSize <= D_SIZE_L1CARRY_MAX && d2Size <= D2_SIZE_L1CARRY_MAX;
if (needL1Carry) { bmm1TilingData.shareL1Size = 0; bmm2TilingData.shareL1Size = 0; }  // L1 让给驻留
// kernel 侧（entry 宏 INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN）：needL1Carry 时走
//   op.UnpackInit(...) + op.ProcessL1Carry()（PairGetS1LoopRange 成对切分 + KV 整 batch 驻留 L1，
//   跨 task/batch 复用装载），否则常规 REGIST_MATMUL_OBJ + Process()
// 手搓直调：首次生成**不开 L1Carry**（复杂度高、收益依赖小 batch 短序列）；照常规路径生成跑通后再对照
//   ProcessL1Carry 路径评估
```

---

### 6. SameAB（自注意力 Q=K=V，`…_sab.h`）

SameAB 变体：Q/K/V 同一 GM tensor
（TND 自注意力），K/V 装载路径共享（一次装载两用），bmm1 的 B 与 bmm2 的 B 同源。事件常量
`Q_EVENT0=EVENT_ID2 / KV_EVENT0=EVENT_ID4 / P_EVENT0=EVENT_ID6`（类内自管）。
**适用**：训练侧 TND 自注意力 varlen（Sq==Skv 同源）。手搓生成先做非 SameAB 版（三输入独立），
SameAB 作为装载共享优化叠加。

---

### 7. host 契约与标杆

- 输入：q/k/v TND `[ΣS, N, D]`（k/v 可 BSH/TND 按 layOutType）；`actualSeqLengths/actualSeqLengthsKv`
  **累加和** int64 数组 [B]（npu 口径直传，勿预差分）；softmaxMax/Sum `[ΣS, N, 8]` fp32。
- tiling：host 逐 batch 累计 `totalSize = Σ CeilDiv(qLen_b, s1Base)*n2G`；coreParams/multiCoreParams
  同 S1（§12.11-D 公式按实际长度集合算，s1Base 取 max batch 适配）。
- 标杆：`torch_npu.npu_fusion_attention(..., input_layout="TND", actual_seq_qlen=累加, actual_seq_kvlen=累加)`
  （特性册 §12.12-A2：BNSD+长度数组需 host 转 TND 喂标杆，语义等价 matched=1.0 已验证）。
- 手搓直调 entry：fa-sink recipe §9 同款（`__mix__` 单 TU + REGIST_MATMUL_OBJ + userWs 独立参），
  入参追加 qList/kvList 两个 GM。

### 8. 手搓自检（varlen 版）

- [ ] seqlen 是累加和数组、kernel 差分（b=0 特判直取）——host 预差分后传差分 = 全部长度错一拍
- [ ] 四件套累计（s1Outer/s1Size/s2Size/attenB1SS）在 CalS1OuterSize 与 ComputeAxisIdx 两处同步推进
- [ ] TND bOffset 用 s1SizeAcc/s2SizeAcc（Q 与 K 不同累计量），无任何 b*固定 stride
- [ ] 纯 varlen 也有 s2End=s2Len 截断（batch 边界）；sparse 版按 §4 公式（右对齐 causal 带 s2Len-s1Len）
- [ ] s1s2 双层：s2LoopCount 进 SetExtraInfo；extraInfo 三槽按 taskId 全局取模（不分层）
- [ ] 尾块收口两处（-inf 尾列 + SetTail），无块格式 0 填充
- [ ] needL1Carry 首次生成固定 false（shareL1Size 正常 1MB）
- [ ] 标杆对齐 TND 累加口径；精度 ≥ 标杆互证 matched=1.0（B>1 多长度混合 case 必测，探 batch 串位）

## MLA 手搓端到端配方全文（fa-mla-handcraft-recipe）
> 以下为 MLA recipe 源文件全量正文：组件选型、__DAV 分流骨架、HardEvent 预置全表、GlobalTensor/tiling、process 解码、prologue+延迟 PV 主循环、GM 槽位、FD rescale 逐行步骤、launch/build、mla_tiling 八步、槽位表与 FA-Paged 对照。

> **用途**：本文件是 `fa-mla-paged.md` §8 的**实现替代**——§8 给的是组件链伪代码，本文件给的是
> **MLAKernel 生产级真实结构端到端拆解**（kernel §1-8 + host tiling §11 +
> 槽位表/host §12），按此写即可一次编译+跑通（已实证 20/20 PASS，gate≈3.19）。
>
> **构成**：通用 MLAKernel + H=128 TP1/AMLA 特化 + mla_tiling + kernel_common 的完整结构；
> host 侧按 §9/§12 自写。
>
> **实测纠错（2026-09，本会话手搓踩坑）**：§8 伪代码缺两个关键事实，导致手搓必死锁：
> 1. **AIC/AIV 分流必须用 `#ifdef __DAV_CUBE__` / `#ifdef __DAV_VEC__` 预处理分支**，**不是**
>    `__mix__(1,2)` + runtime `g_coreType`。`__mix__` 是 kfc 单 TU 路径（§12.13），MLA CrossCoreFlag
>    路径的组件/HardEvent/CrossCoreFlag token 经济按 `__DAV_CUBE__/__DAV_VEC__` 双二进制模型设计。
>    用 `__mix__`+g_coreType 会破坏 token 配平 → hang（实测：`__mix__` 下 AIC 单独 pre-set+QK+drain 即死锁）。
> 2. **主循环必须是 prologue + 延迟 PV 流水**（见 §6），**不是** "QK(n)→softmax(n)→PV(n)→rescale(n) 同块锁步"。
>    锁步的 CrossCoreFlag Set/Wait 顺序与组件内部 HardEvent 配对不匹配 → token 不平衡 → hang。
>
> CrossCoreFlag 用法（§8 §8.5 正确，本会话实证 OK）：Set 用 `Arch::CrossCoreSetFlag<0x2,PIPE_X>(flag)`，
> Wait 用 bare `Arch::CrossCoreWaitFlag(flag)`（默认 mode 0/PIPE_S，与 `<0x2,PIPE>` Set 在 dav-2201 配对）。

> **★2026-09 闭卷验证结果**：纯按本 recipe 手搓 `mla_kernel.cpp`（不参照任何外部文件），一次编译通过 +
> 运行不死锁 + 精度：kv=128/1024/2048/4096（多块 nLoop≤32，kvSplitCores=1）**100% PASS**（max_abs≤6e-5）。
> 证明 §1（`__DAV_CUBE__/__DAV_VEC__` 分流）+ §5/§6（prologue+延迟 PV 主循环）+ §2（HardEvent 预置）为
> 正确可生成的完整配方。**剩余**：kv≥6144 触发 flash-decoding kv-split（kvSplitCores>1）时必须实现 §8 FD
> rescale 归并（多核部分 O 合并），否则漂移（max_abs~0.034）；recipe §8 已载此段必需。

---

### 0. 组件选型（与 §8.10 一致，确认真实签名——全部对 dispatch_policy.hpp 核验）

```cpp
using ElementQ=half; using ElementK=half; using ElementP=half; using ElementO=half;
using ElementS=float; using ElementOTmp=float; using ElementUpdate=float;
using LayoutQ   = Catlass::layout::RowMajor;     // Q/S/P/O RowMajor
using LayoutK   = Catlass::layout::ColumnMajor;  // K/V ColumnMajor（host 不预转置）
using LayoutS   = Catlass::layout::RowMajor;
using LayoutP   = Catlass::layout::RowMajor;
using LayoutV   = Catlass::layout::ColumnMajor;
using LayoutO   = Catlass::layout::RowMajor;
using LayoutOTmp= Catlass::layout::RowMajor;
using LayoutUpdate=Catlass::layout::RowMajor;

using L1TileShape = Catlass::GemmShape<128,128,576>;   // K=embed+embedRope=576，K 常驻 L1
using L0TileShape = L1TileShape;

// ★MmadAtlasA2MLAQK/MLAPV 是普通 struct（非模板，dispatch_policy.hpp:74/78）——
//   与 FAIQK<true,false>（模板）不同，勿手滑加模板参
using BlockMmadQK = Catlass::Gemm::Block::BlockMmad<
    Catlass::Gemm::MmadAtlasA2MLAQK, L1TileShape, L0TileShape,
    GemmType<half,LayoutQ>, GemmType<half,LayoutK>, GemmType<float,LayoutS>>;
using BlockMmadPV = Catlass::Gemm::Block::BlockMmad<
    Catlass::Gemm::MmadAtlasA2MLAPV, L1TileShape, L0TileShape,
    GemmType<half,LayoutP>, GemmType<half,LayoutV>, GemmType<float,LayoutOTmp>>;
using EpilogueMLASoftmax   = Catlass::Epilogue::Block::BlockEpilogue<Catlass::Epilogue::EpilogueAtlasA2MLASoftmax,   GemmType<half,LayoutP>, GemmType<float,LayoutS>>;
using EpilogueMLARescaleO  = Catlass::Epilogue::Block::BlockEpilogue<Catlass::Epilogue::EpilogueAtlasA2MLARescaleO,  GemmType<half,LayoutO>,  GemmType<float,LayoutOTmp>>;
// FDRescaleO 是 uint32 模板（6144=ComputeEleNum）；KV_SPLIT_MAX=64、HEADS_PROCESS_MAX=16 由其内提供
using EpilogueMLAFDRescaleO= Catlass::Epilogue::Block::BlockEpilogue<Catlass::Epilogue::EpilogueAtlasA2MLAFDRescaleO<6144>, GemmType<half,LayoutO>>;
```

**TP1/AMLA 特化组件名（H=128 spec 路径，dispatch_policy.hpp:82-90 + epilogue:85-100，已核验存在）**：
`MmadAtlasA2MLAQKTp1Spec` / `MmadAtlasA2MLAPVTp1Spec` / `MmadAtlasA2AMLAPVTp1Spec` +
`EpilogueAtlasA2MLATP1Softmax` / `EpilogueAtlasA2MLATP1RescaleO` /
`EpilogueAtlasA2AMLATP1Softmax` / `EpilogueAtlasA2AMLATP1RescaleO`。
本 recipe §1-8 只展开**通用 MLAKernel**（H=16/32/64，tilingKey 0/1）；H=128 spec 走
`mla_kernel_tp1_spec.cpp`/`amla_kernel_tp1_spec.cpp`（每 Q token 一 process 的重组版，骨架同构、
组件换上表特化名 + `EpilogueAtlasA2MLATP1*` epilogue）——首次生成 spec 路径时按本 recipe §1-8 相位映射同构展开（组件换上表特化名 + TP1 epilogue）。

---

### 1. kernel 类骨架（`__DAV_CUBE__`/`__DAV_VEC__` 预处理分支）

```cpp
template <class BlockMmadQK, class BlockMmadPV, class EpilogueMLASoftmax,
          class EpilogueMLARescaleO, class EpilogueMLAFDRescaleO>
class MLAKernel {
    // type aliases: ElementQ/K/S/P/O/OTmp/Update, Layout*, ArchTag, L1TileShape ...
    // (从 BlockMmadQK:: / Epilogue*:: 取)
public:
    struct Params { GM_ADDR q,qRope,k,kRope,blockTables,o,s,p,oTmp,oUpdate,oCoreTmp,l,tiling; };

    CATLASS_DEVICE void operator()(Params const& params) {
        // §2 HardEvent pre-set (AIC+AIV 共享，§8.6 全表)
        // §3 GlobalTensor SetGlobalBuffer + tiling 读取 (按 #ifdef 分流)
        // §4 coreIdx (AIC=GetBlockIdx / AIV=GetBlockIdx/GetSubBlockNum) + task 遍历
        // §5 prologue (nIdx=0: QK+softmax，无 PV)
        // §6 main loop (nIdx=1..nLoop: QK(n)+softmax(n) if非末 / 末块下task预取 / PV(n-1)+rescale(n-1))
        // §7 HardEvent drain (§8.6 反预置)
        // §8 FD rescale (仅 maxKvSplitCoreNum!=1)
    }
};
```

★分支形式：`#ifdef __DAV_CUBE__ { AIC-only 代码 } #endif` 与 `#ifdef __DAV_VEC__ { AIV-only 代码 } #endif`
**同处并列**（同一 operator() 内），编译器在 `--npu-arch=dav-2201` mix 下自动让 cube 核走 `__DAV_CUBE__`、
vector 核走 `__DAV_VEC__`。**不要** `__mix__` 属性、**不要** `g_coreType` runtime if-else。

---

### 2. HardEvent 预置（§8.6 全表，AIC+AIV 共享，operator() 入口）

```cpp
#ifdef __DAV_CUBE__
SetFlag<HardEvent::M_MTE1>(EVENT_ID0..7);      // 8
SetFlag<HardEvent::FIX_M>(EVENT_ID0); SetFlag<HardEvent::FIX_M>(EVENT_ID1);  // 2
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0..7);  // 8
SetFlag<HardEvent::FIX_MTE1>(EVENT_ID0..5);   // 6
SetFlag<HardEvent::MTE2_FIX>(EVENT_ID0);      // 1    共 25
#endif
#ifdef __DAV_VEC__
SetFlag<HardEvent::MTE3_V>(EVENT_ID0); SetFlag<HardEvent::MTE3_V>(EVENT_ID2); SetFlag<HardEvent::MTE3_V>(EVENT_ID3);  // 3
SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); for(e=2..7) SetFlag<HardEvent::MTE3_MTE2>(e);  // 7
SetFlag<HardEvent::V_MTE2>(EVENT_ID0); SetFlag<HardEvent::V_MTE2>(EVENT_ID1); SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
SetFlag<HardEvent::V_MTE2>(EVENT_ID2); SetFlag<HardEvent::V_MTE2>(EVENT_ID3);  // 5    共 15
#endif
```
**drain（§7，遍历结束后，反序同表）**：把上表所有 `SetFlag` 换 `WaitFlag`，顺序**逐字照抄**实测顺序
（AIV 的 V_MTE2 drain 顺序是 4,0,1,2,3——非递增，照抄勿自排）。

> 实测：pre-set/drain 与 §8.6 一致，**不是死锁源**。死锁源是 §6 的循环结构 + §1 的分支形式。

---

### 3. GlobalTensor + tiling（按 #ifdef 分流）

```cpp
#ifdef __DAV_CUBE__
GlobalTensor<ElementQ> gQ; gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
gQRope/gK/gKRope 同；GlobalTensor<int32_t> gblockTable; ...SetGlobalBuffer((__gm__ int32_t*)params.blockTables);
BlockMmadQK blockMmadQK(resource); BlockMmadPV blockMmadPV(resource);
// 值：embedRope, maxNumBlocksPerQuery, kvHeads, strideQORope, strideKV, strideKVRope（仅 AIC）
#endif
#ifdef __DAV_VEC__
GlobalTensor<ElementO> gO; gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
gOUpdate/gOCoreTmp/gl 同；GlobalTensor<float> gTilingFp64(...params.tiling);
EpilogueMLASoftmax epilogueMLASoftmax(resource, tor, maxKvSplitCoreNum);
EpilogueMLARescaleO epilogueMLARescaleO(resource, maxKvSplitCoreNum);
float tor = gTilingFp64.GetValue(TILING_TOR);
uint32_t glFlag[2] = {1, 1};   // ★真数组，勿传 nullptr（kvSplit==1 不解引但实测须传 {1,1}）
#endif
// 共享：batch/qHeads/embed/blockSize/tilingHeadSize/tilingParaSize/curQheadSplitSize/curQheadSplitNum/maxKvSplitCoreNum
//        strideQO=qHeads*embed, embedRound=RoundUp<BLOCK_SIZE>(embed)
```

`resource` = `Catlass::Arch::Resource<Arch::AtlasA2>`，在 operator() 顶部构造一次（pre-set 之前）。

---

### 4. coreIdx + task 遍历（AIC/AIV 共享同一 loop——含完整 process 解码算法）

```cpp
uint32_t coreIdx = 0;
#ifdef __DAV_CUBE__  coreIdx = GetBlockIdx(); #endif
#ifdef __DAV_VEC__  coreIdx = GetBlockIdx() / GetSubBlockNum(); #endif     // ★AIV 必除，否则 blockIdx 0 与 20 双跑 task0 → flag 死锁
uint32_t coreNum  = GetBlockNum();
uint32_t tilingProcessNum = gTiling.GetValue(TILING_PROCESSNUM);
uint32_t processNum = (tilingProcessNum != 0) ? tilingProcessNum : batch * curQheadSplitNum * maxKvSplitCoreNum;
bool isForward = true, kerFlag = true;    // back-and-forth 蛇形方向位（仅 ==0 路径用）

for (uint32_t process = coreIdx; process < processNum; process += coreNum) {
    // ---- (a) process → curBatch（两种映射，逐行） ----
    uint32_t curBatch;
    uint32_t taskIdx = 0;
    if (tilingProcessNum != 0) {
        // T+1 连续前缀和映射：tilingHost[CUTASK_START_OFFSET+b] = 前 b 个 batch 的 process 累计
        while (taskIdx < batch && process >= gTiling.GetValue(CUTASK_START_OFFSET + taskIdx + 1)) {
            taskIdx++;
        }
        curBatch = taskIdx;
    } else {
        // back-and-forth 蛇形映射：同一核的相邻 process 落在相邻 batch（提高 KV cache 局部性）
        uint32_t bigProcess = process - (process % coreNum) + (coreNum - 1);
        bigProcess = (bigProcess > processNum - 1) ? processNum - 1 : bigProcess;
        uint32_t realProcess = isForward ? process : (bigProcess - process % coreNum);
        isForward = !isForward;
        curBatch = realProcess / (curQheadSplitNum * maxKvSplitCoreNum);
    }
    uint32_t offsetTiling = tilingHeadSize + tilingParaSize * curBatch;
    uint32_t qSeqlen = gTiling.GetValue(offsetTiling);
    uint32_t kvSeqlen = gTiling.GetValue(offsetTiling + 1);
    uint32_t kvSplitPerCore = gTiling.GetValue(offsetTiling + 15);
    uint32_t kvSplitCoreNum = gTiling.GetValue(offsetTiling + 16);
    if (kvSeqlen == 0) { continue; }

    // ---- (b) process → (qHeadSplitIdx, curNIdx) ----
    uint32_t qHeadSplitIdx, curNIdx;
    if (tilingProcessNum != 0) {
        qHeadSplitIdx = 0;                                          // kv-split 路径 H 不再切
        curNIdx = process - gTiling.GetValue(CUTASK_START_OFFSET + curBatch);
    } else {
        qHeadSplitIdx = (process % (curQheadSplitNum * maxKvSplitCoreNum)) / maxKvSplitCoreNum;
        curNIdx = process % maxKvSplitCoreNum;
        if (kerFlag) { kerFlag = false; }                           // 蛇形：偶次正向、奇次反向取 N 段
        else { kerFlag = true; curNIdx = maxKvSplitCoreNum - curNIdx - 1; }
    }
    uint32_t qHeadSplitSizeActual = (qHeadSplitIdx == (curQheadSplitNum - 1)) ?
        (qHeads - qHeadSplitIdx * curQheadSplitSize) : curQheadSplitSize;
    uint32_t curKVSeqlen = kvSplitPerCore;
    if (curNIdx >= kvSplitCoreNum) { continue; }                    // 对齐产生的空槽
    if (curNIdx == (kvSplitCoreNum - 1)) {
        curKVSeqlen = kvSeqlen - curNIdx * kvSplitPerCore;          // 末核吃尾差
    }
    uint32_t nLoop = (curKVSeqlen + seqTile - 1) / seqTile;         // seqTile = blockSize = 128
    uint32_t rowNum = qHeadSplitSizeActual * qSeqlen;               // H 分块 × token pack 进 M
    uint32_t rowNumRound = RoundUp<16>(rowNum);
    // → 进 §5 prologue / §6 主循环
}
```
**★AIV coreIdx 必 `GetBlockIdx()/GetSubBlockNum()`**：否则双子核 blockIdx 0 与 1 都映射 logical 0
而 blockIdx 0 与 20 也都映射 0 → 两个核 Wait 同一 flag → 死锁。
**解码三态**：kv-split 生效（TILING_PROCESSNUM≠0，q=1 decode）走前缀和 + H 不切；未生效走蛇形
（qHeadSplitIdx × curNIdx 二维 + 蛇形反向）。两态的 flag 配平依赖"每 process 恰好一轮 prologue+主循环"
——curNIdx≥kvSplitCoreNum 的空槽 continue **必须保留**（跳过即不碰 flag，配平不破）。

---

### 5. prologue（isFirstTask，nIdx=0：只 QK+softmax，**无 PV**）

```cpp
if (isFirstTask && nLoop > 0) {
    uint32_t nIdx = 0;
    // 尾块 kSeqTile 修正
#ifdef __DAV_CUBE__
    LayoutQ layoutQ(rowNum, embed); LayoutQ layoutQRope(rowNum, embedRope);
    LayoutK layoutK(embed, kSeqTile); LayoutK layoutKRope(embedRope, kSeqTile);
    LayoutS layoutS(rowNumRound, kSeqTileRound);
    GemmCoord actualBlockShapeQK{rowNum, kSeqTile, embed + embedRope};
    MatrixCoord qShapeSingleNd{qHeadSplitSizeActual, embed};
    uint32_t qkPingPongFlag = pingpongIdx % 2;
    int32_t blockTableId = gblockTable.GetValue(realCurBatch*maxNumBlocksPerQuery + startKV/blockSize + nIdx);
    uint64_t kvOffset = (uint64_t)blockTableId*blockSize*strideKV, kvOffsetRope=...*strideKVRope;
    uint64_t gSOffset = (uint64_t)coreIdx*TMP_SIZE_DECODER + (uint64_t)qkPingPongFlag*TMP_SIZE_DECODER/2;
    blockMmadQK(gQ[gQOffset], gQRope[gQRopeOffset], gK[kvOffset], gKRope[kvOffsetRope],
                gS[gSOffset], layoutQ, layoutQRope, layoutK, layoutKRope, layoutS,
                actualBlockShapeQK, qShapeSingleNd, qHeads, nIdx, pingpongIdx);
    pingpongIdx++;
    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
    isFirstTask = false;
#endif
#ifdef __DAV_VEC__
    uint32_t softmaxPingPongFlag = pingpongIdx % 2;
    Arch::CrossCoreWaitFlag(qkReady);
    LayoutP layoutP(rowNum, kSeqTile, kSeqTileRound); LayoutS layoutS(rowNumRound, kSeqTile, kSeqTileRound);
    GemmCoord actualBlockShapeQK{rowNum, kSeqTile, embedRound};
    uint64_t gmOffsetP = (uint64_t)coreIdx*TMP_SIZE + softmaxPingPongFlag*TMP_SIZE/2;
    uint64_t gmOffsetS = (uint64_t)coreIdx*TMP_SIZE_DECODER + softmaxPingPongFlag*TMP_SIZE_DECODER/2;
    epilogueMLASoftmax(gP[gmOffsetP], gS[gmOffsetS], layoutP, layoutS, actualBlockShapeQK,
                       nIdx, qHeadSplitSizeActual, softmaxPingPongFlag, glFlag, taskPingPongFlag);
    pingpongIdx++;
    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
    isFirstTask = false;
#endif
}
```

---

### 6. 主循环（nIdx=1..nLoop：QK(n)+softmax(n) if非末 / 末块下task预取 / **PV(n-1)+rescale(n-1) 延迟**）

```cpp
for (uint32_t nIdx = 1; nIdx < nLoop + 1; nIdx++, pingpongIdx++) {
    // (a) 非末块：QK(n) + softmax(n)
    if (nIdx != nLoop) {
        // 尾块 kSeqTile 修正
#ifdef __DAV_CUBE__
        // ... layoutQ/K/S, blockTableId, gSOffset（同 prologue，nIdx 变）
        blockMmadQK(..., nIdx, pingpongIdx);
        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_VEC__
        Arch::CrossCoreWaitFlag(qkReady);
        // ... layoutP/S, gmOffsetP/S
        epilogueMLASoftmax(..., nIdx, qHeadSplitSizeActual, softmaxPingPongFlag, glFlag, taskPingPongFlag);
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
    }
    // (b) 末块（nIdx==nLoop）：下一 task 的 prologue 预取（QK(0)+softmax(0)，跨 task 流水）
    if (nIdx == nLoop) {
        // peek nextProcess（coreNum+process 起），算其 (nextBatch, nextQHeadSplitIdx, nextCurNIdx)
        // 跑 nextNIdx=0 的 blockMmadQK + epilogueMLASoftmax（AIC Set qkReady / AIV Wait+Set softmaxReady）
        // break（只预取一个 task）
    }
    // (c) ★延迟 PV(n-1) + rescale(n-1)——每轮都做，PV 用 (pingpongIdx-1)%2 槽
    if (nIdx == nLoop) { vSeqTile = curKVSeqlen - (nIdx-1)*seqTile; vSeqTileRound=RoundUp<16>(vSeqTile); }
#ifdef __DAV_CUBE__
    LayoutP layoutP(rowNum, vSeqTile, vSeqTileRound); LayoutV layoutV(embed, vSeqTile);
    LayoutOTmp layoutOTmp(rowNumRound, embedRound);
    GemmCoord actualBlockShapePV{rowNum, embed, vSeqTile};
    uint32_t pvPingPongFlag = (pingpongIdx - 1) % 2;
    uint64_t gPOffset  = (uint64_t)coreIdx*TMP_SIZE + pvPingPongFlag*TMP_SIZE/2;
    uint64_t gOTmpOffset = (uint64_t)coreIdx*TMP_SIZE*2 + pvPingPongFlag*TMP_SIZE;
    blockMmadPV(gP[gPOffset], gOTmp[gOTmpOffset], layoutP, layoutV, layoutOTmp,
                actualBlockShapePV, nIdx, pingpongIdx, softmaxReady, locPingPongIdx);
    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_VEC__
    Arch::CrossCoreWaitFlag(pvReady);
    LayoutO layoutO(tokenNumPerHead, strideQO); LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
    LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
    GemmCoord actualBlockShapePV{rowNum, embed, vSeqTile};
    uint32_t rescaleOPingPongFlag = (pingpongIdx - 1) % 2;
    uint64_t gmOffsetOTmp = (uint64_t)(coreIdx*TMP_SIZE*2 + rescaleOPingPongFlag*TMP_SIZE);
    uint64_t gmOffsetUpdate = (uint64_t)(coreIdx*TMP_SIZE);
    uint32_t isLastNTile = (nIdx == nLoop) ? 1 : 0;
    epilogueMLARescaleO(gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gO[gmOffsetO], gOCoreTmp[oFdOffset],
                        gl[lOffset], layoutOTmp, layoutO, layoutUpdate, actualBlockShapePV,
                        nIdx, isLastNTile, qHeadSplitSizeActual, rescaleOPingPongFlag, glFlag, taskPingPongFlag);
#endif
}
#ifdef __DAV_VEC__ taskPingPongFlag = 1 - taskPingPongFlag; #endif
```

**★为何延迟 PV**：blockMmadPV 内部 `CrossCoreWaitFlag(softmaxReady)` 等 AIV 的 P；AIV softmax 写 P 到
`softmaxPingPongFlag` 槽，PV 用 `(pingpongIdx-1)%2` 读**上一块**的 P——形成 QK(n) 与 PV(n-1) 交错流水，
CrossCoreFlag token（qkReady/softmaxReady/pvReady）按此交错配平。同块锁步会破坏配平→死锁（实测）。

---

### 7. GM 槽位公式（§8.11，照搬）

```
S(gSOffset)    = coreIdx*TMP_SIZE_DECODER + (pp%2)*TMP_SIZE_DECODER/2     // QK 写
P(gmOffsetP)   = coreIdx*TMP_SIZE        + (pp%2)*TMP_SIZE/2              // softmax 写；PV 读用 (pp-1)%2
OTmp(gOTmpOffset) = coreIdx*TMP_SIZE*2   + ((pp-1)%2)*TMP_SIZE            // PV 写；rescale 读同偏移
OUpdate(gmOffsetUpdate) = coreIdx*TMP_SIZE                                // rescale 写
```
TMP_SIZE=65536, TMP_SIZE_DECODER=32768（kernel_common.hpp）。
per-batch Q/QRope/O/l/oFd 的 64 位 GM 地址由 host 拆 hi/lo 32 位写进 tiling 段
（offsetTiling+4/5=q hi/lo, +6/7=qRope, +11/12=l, +13/14=oFd），kernel 拼回（见 §3）。

---

### 8. FD rescale（flash-decoding kv-split 归并，仅 maxKvSplitCoreNum != 1，drain 之后——逐行步骤）

```cpp
#ifdef __DAV_VEC__
    // drain（§7 反预置）之后：
    if (maxKvSplitCoreNum != 1) {
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();       // ★全核汇聚（等所有核写完 oFd/l）

        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        EpilogueMLAFDRescaleO epilogueMLAFDRescaleO(resource, maxKvSplitCoreNum);
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();   // ★此处不除（物理 aiv）
        uint32_t aivId = AscendC::GetBlockIdx();

        uint32_t headsProcess = (COMPUTE_ELE_NUM / embed) > HEADS_PROCESS_MAX ?
                                HEADS_PROCESS_MAX : (COMPUTE_ELE_NUM / embed);
        uint32_t loopsPerBatch = (qHeads + headsProcess - 1) / headsProcess;
        uint32_t loopsTotal = batch * loopsPerBatch;

        for (uint32_t loopIdx = aivId; loopIdx < loopsTotal; loopIdx += aivNum) {
            uint32_t batchIdx = loopIdx / loopsPerBatch;
            uint32_t loopIdxInBatch = loopIdx % loopsPerBatch;
            uint32_t offsetTiling = tilingHeadSize + tilingParaSize * batchIdx;
            uint32_t kvSeqlen = gTiling.GetValue(offsetTiling + 1);
            uint32_t qSeqlen  = gTiling.GetValue(offsetTiling);
            uint32_t kvSplitCoreNum = gTiling.GetValue(offsetTiling + 16);
            if (kvSeqlen == 0) { continue; }

            // per-batch 地址：O/l/oFd 均 hi/lo 拆 32 位存 tiling 槽（+4/5 复用 Q 偏移=O 偏移，
            // 因 decode 时 Q 与 O 同形 [tokens,H,512]；l=+11/12、oFd=+13/14）
            uint64_t oAddr   = ((uint64_t)gTiling.GetValue(offsetTiling+4)  << 32) | gTiling.GetValue(offsetTiling+5);
            uint64_t lOffset = ((uint64_t)gTiling.GetValue(offsetTiling+11) << 32) | gTiling.GetValue(offsetTiling+12);
            uint64_t oFdOffset = ((uint64_t)gTiling.GetValue(offsetTiling+13) << 32) | gTiling.GetValue(offsetTiling+14);

            uint32_t actualHeads = headsProcess;
            if (loopIdxInBatch == loopsPerBatch - 1) {
                actualHeads = qHeads - loopIdxInBatch * headsProcess;    // 尾批头数
            }
            for (uint32_t qSeqIdx = 0; qSeqIdx < qSeqlen; qSeqIdx++) {
                epilogueMLAFDRescaleO(
                    gO[oAddr + qSeqIdx * qHeads * embed + loopIdxInBatch * headsProcess * embed],
                    gOCoreTmp[oFdOffset * maxKvSplitCoreNum + qSeqIdx * qHeads * embed * maxKvSplitCoreNum
                              + loopIdxInBatch * headsProcess * maxKvSplitCoreNum * embed],
                    gl[lOffset + qSeqIdx * qHeads * maxKvSplitCoreNum
                                + loopIdxInBatch * headsProcess * maxKvSplitCoreNum],
                    actualHeads, headsProcess, embed, kvSplitCoreNum);
            }
        }
    }
#endif
```

- **语义**：`O[b,t,h,:] = Σ_n O_coreTmp[n]·exp(m_n−m_max)·l_n / Σ_n l_n·exp(m_n−m_max)`（FDRescaleO 组件
  内部完成多核部分 O/l/m 的在线归并）。调用前所有核必须已写完各自 oFd/l 分片 → `CrossCoreBarrier`。
- **KV_SPLIT_MAX / HEADS_PROCESS_MAX / COMPUTE_ELE_NUM** 来自 `EpilogueAtlasA2MLAFDRescaleO<6144>` 模板
  （6144=ComputeEleNum，实测取值——手搓直接 `EpilogueAtlasA2MLAFDRescaleO<6144>`）。
- kv<6144 或 batch>blockDim·0.8 或 maxQseqlen>1 时 tiling 不 split（§11 决策），本段自动跳过。
- ★`aivId = GetBlockIdx()`（不除 subBlockNum）+ `aivNum = BlockNum*SubBlockNum`：FD 段按**物理** aiv
  分发 loop（与主循环的 logical coreIdx 不同——勿复制主循环的除法）。

---

### 9. launch + build（host 契约）

- launch：`MLA<half><<<aicCoreNum, nullptr, stream>>>(hwSync, dQ,dQRope,dK,dKRope,dBT,dO,dS,dP,dOTmp,dGlobalO,dOCoreTmp,dL,dTiling)` 14 参。
- tilingKey 分发：0/1=MLA(half/bf16)、4/5=AMLATp1Spec、7/8=MLATp1Spec（H=128 用 TP1）。
  key = (specStra?1:0)<<2 | dTypeKey；若 specStra && numTokens%blockDim<=10 && batch<=40 → key=dType?8:7。
- tiling：`MLATiling::GetMLATilingParam(MLAInfo, blockDim, uint32_t* tilingHost)`（host TU）。
- workspace：wsDbl = blockDim*WORKSPACE_BLOCK_SIZE_DB(65536)；dS=wsDbl*float*2, dP=wsDbl*elem*2,
  dOTmp=wsDbl*float*2, dGlobalO=wsDbl*float, oFdSize=embed*numHeads*numTokens*maxKvSplitCoreNum*float,
  lSize=numTokens*numHeads*maxKvSplitCoreNum*float。
- build：`CATLASS_ARCH=2201` + `--npu-arch=dav-2201`（让 `__DAV_CUBE__`/`__DAV_VEC__` 宏生效）；
  host .asc `#include "mla_kernel.cpp"` 单 TU；mla_tiling.cpp 单独 CXX/ASC TU；
  link ascendc_runtime ascendcl runtime profapi mmpa c_sec error_manager graph_base tiling_api register platform unified_dlog dl m。

---

### 11. mla_tiling.cpp（host 侧 tiling，独立 host TU——MLATiling 命名空间逐步）

> MLA tiling 与 FA-Paged 不同：**fp32 数组寻址**（非结构体 memcpy）。头部常量（mla_tiling.h）：
> `TILING_HEAD_SIZE=400`（头区起点）、`TILING_PARA_SIZE=17`（每 batch 段元素数）、
> `CUTASK_START_OFFSET=25`（前缀和表起点）、`CUTASK_MAX_LENGTH=130`、
> `QN_TILE_LIST={128,64,32,16,8,1}`（按 ⌈log2(qSeqlen)⌉ 取，封顶 idx 5）、
> `SPLITKV_RATION=0.8`、`KV_SEQLEN_SLICE=128`。

**入口 `GetMLATilingParam(MLAInfo, blockDim, tilingHost)` 八步**（blockDim = aicCoreNum = 20）：

1. **校验**：tilingHost/qSeqLen/kvSeqLen 非空；`blockSize==128`（否则 -1）。
2. **per-batch 地址偏移表**（Q/QRope/mask 三套累加）：
   `qSeqOffset_b = Σ_{j<b} numHeads·embeddingSize·qSeqLen_j`（QRope 用 embeddingSizeRope、
   mask 用 maxKvSeqlen·qSeqLen_j）。
3. **排序**：`quickSortIndices(sortedIndices, kvSeqLen, qSeqLen, 0, batch-1)`——按 kv 长降序
   （长序列优先切 split，负载均衡）；`realPreTaskNums[b][token]` = 全局 token 前缀（TP1 用）。
4. **约束检查**：qSeqLen>4 时告警（MLA 契约 q∈[1,4]）；
   `Σ kvSeqLen > numBlocks·blockSize` 报错返回（paged cache 容量）。
5. **tor**：`1/sqrt(embeddingSize+embeddingSizeRope)` = 1/√576，按位 reinterpret 成 uint32 写槽
   （AIV 从 `gTilingFp64.GetValue(TILING_TOR)` 位还原读回——勿走 fp64 舍入）。
   `specStrategyFlag = (numHeads==128) ? 1 : 0`（H=128 → TP1 特化路径）。
6. **写头区 + 每 batch 段**（spec ? GetMLATilingSpec : GetMLATilingCommon）：
   头区 `TILING_HEADSIZE..` = batch/TILING_HEADSIZE(=400)/TILING_PARASIZE(17)/NUMHEADS/HEADDIM(512)/
   NUMBLOKS/BLOCKSIZE(128)/MAXBLOCKS(=⌈maxKvSeqlen/128⌉)/TOR/KVHEADS/HEAD_SPLIT_SIZE(=GetQNBlockTile)/
   HEAD_SPLIT_NUM(=⌈H/切⌉)/MASKTYPE/HEADDIM_ROPE(64)/TOTAL_QTOKENS。
   每 batch 段（`offset = TILING_HEAD_SIZE + TILING_PARA_SIZE*b`）见 §12 槽位表。
7. **KV split 决策 + 贪心切分**（GetKVSplitParam，仅非 spec；spec 用 GetKVSplitParamSpec 同思路）：
   ```
   isKVSplit = (maxKvSeqlen >= blockDim*128*2) && (batch <= blockDim*0.8) && (maxQseqlen == 1)
   // 不 split：每 batch kvSplitPerCore=kvSeqlen, kvSplitCoreNum=1, TILING_PROCESSNUM=0, KVCORENUM=1
   // split（T+1 均衡贪心，逐行）：
   MIN_TOKENS_PER_SPLIT = blockSize*4 = 512；MAX_SPLIT_PER_TASK = 8
   alignedKvLens[b] = RoundUp(kvSeqLen, 128)；batchSplitNum 全 1；totalAllocated = batch
   while (totalAllocated < blockDim):
       找 currentLoad=alignedKvLens[i]/batchSplitNum[i] 最大的 i
       （约束 batchSplitNum[i]<8 且 nextLoad≥512）
       batchSplitNum[i]++；totalAllocated++
       （无候选即 break——防过切）
   逐 batch：kvBlockPerCore=⌈(RoundUp(kv,128)/128)/batchSplitNum[b]⌉，kvSplitPerCore=kvBlockPerCore*128，
             kvSplitCoreNum=⌈kvSeqlen/kvSplitPerCore⌉
   前缀和：CUTASK_START_OFFSET[0]=0，[b+1]=Σ_{j≤b} kvSplitCoreNum[j] → TILING_PROCESSNUM=总 process
   TILING_KVCORENUM = max(batchSplitNum) = maxKvSplitCoreNum
   ```
8. **l/oFd 偏移**：`lOffset_b = Σ_{j<b} numHeads·qSeqlen_j·MAX_KV_SPLIT_NUM`（★乘全局最大 split 数）、
   `oFdOffset_b = Σ_{j<b} numHeads·qSeqlen_j·embeddingSize`——hi/lo 拆写 +11/12、+13/14 槽。

**host-kernel 一致性铁律**（同 FA-Paged Step 10）：§4 的两种 process 解码（前缀和/蛇形）必须与
本步 7 的 TILING_PROCESSNUM/KVCORENUM/CUTASK 表严格配套——host 写 split 表而 kernel 走蛇形
（或反之）→ process 解码错乱 → flag 配对破坏 → 死锁。

### 12. tiling 槽位表 + kernel_common 常量 + host 步骤

**每 batch 段槽位**（`offsetTiling = TILING_HEAD_SIZE + TILING_PARA_SIZE*batchIdx`，fp32 元素）：

| 槽 | 含义 | 写方（host tiling） | 读方（kernel） |
|---|---|---|---|
| +0 | qSeqlen（原始非累加） | GetMLATilingCommon | AIC/AIV 任务切分 |
| +1 | kvSeqlen | 同上 | 同上 + FD |
| +2 | sortSeqIdx（排序后原 batch 号） | 同上 | AIC 取 per-batch Q/K 基址 |
| +3 | blockSize(128) | 同上 | — |
| +4/+5 | Q 地址偏移 hi/lo（★decode 时 Q 与 O 同形，FD 段复用为 O 偏移） | GetAddrOffsetMLA | AIC QK；AIV FD |
| +6/+7 | QRope 地址偏移 hi/lo | 同上 | AIC QK |
| +8/+9 | mask 偏移 hi/lo | 同上 | mask 版用 |
| +11/+12 | l 偏移 hi/lo | GetKVSplitParam 步 8 | AIV RescaleO / FD |
| +13/+14 | oFd 偏移 hi/lo | 同上 | AIV FD |
| +15 | kvSplitPerCore（128 倍数） | GetKVSplitParam | 任务切分 curKVSeqlen |
| +16 | kvSplitCoreNum | 同上 | 任务切分 / FD |

**kernel_common.hpp MLA 常量**（共用头，recipe §7 已列 TMP_SIZE=65536 / TMP_SIZE_DECODER=32768，
另需）：`QK_READY_ID=1 / SOFTMAX_READY_ID=2 / PV_READY_ID=3 / BLOCK_SIZE=16 / WORKSPACE_BLOCK_SIZE_DB=65536`
（★MLA 的 WS_DB 是 65536，FA-Paged 是 131072，勿混）+ TILING_* 头区索引常量表（mla_tiling.h 逐值）：
`BATCH=0, NUMHEADS=1, HEADDIM=2, NUMBLOKS=3, BLOCKSIZE=4, MAXBLOCKS=5, TOR=6, KVHEADS=7, HEADSIZE=8,
PARASIZE=9, HEAD_SPLIT_SIZE=10, HEAD_SPLIT_NUM=11, MASKTYPE=12, HEADDIM_ROPE=13, MAX_KVSEQLEN=14,
KVSPLIT=15, KVCORENUM=16, MAX_QSEQLEN=17, TOTAL_QTOKENS=18, FORMERTASKNUM=19, TAILTASKNUM=20,
PROCESSNUM=21`（头区 0..21）；`TILING_HEAD_SIZE=400 / CUTASK_START_OFFSET=25 / CUTASK_MAX_LENGTH=130 /
TILING_PARA_SIZE=17`（CUTASK 前缀和表在头区内 25 起连续 batch+1 项）。

**host 步骤**（mla.cpp，同 FA-Paged Step 11 管线，差异点）：
1. 输入 6 张量：q[tokens,H,512] / qRope[tokens,H,64] / k,kRope（paged 块格式 [numBlocks,128,kvHeads,D]）/
   blockTable / o[tokens,H,512]。
2. tilingHost 按**元素数** malloc：`TILING_HEAD_SIZE + TILING_PARA_SIZE*batch` 个 uint32（fp32 数组非结构体）。
3. workspace（§9 公式）：`wsDbl = blockDim*65536`；dS=wsDbl·4B·2、dP=wsDbl·elem·2、dOTmp=wsDbl·4B·2、
   dGlobalO(=oUpdate)=wsDbl·4B、oFdSize=512·numHeads·numTokens·maxKvSplitCoreNum·4B、
   lSize=numTokens·numHeads·maxKvSplitCoreNum·4B（maxKvSplitCoreNum 从 tiling 回读后再分配——先跑
   GetMLATilingParam 再 malloc oFd/l）。
4. launch 14 参：`MLA<half><<<aicCoreNum,nullptr,stream>>>(hwSync,q,qRope,k,kRope,bt,o,s,p,oTmp,oGlobal,oCoreTmp,l,tiling)`；
   中间槽 nullptr；tilingKey 分发见 §9。

### 10. FA-Paged 近似配方但分流不同

`fai_kernel.cpp` 的 `FAInferKernel` 与 MLA 同范式（CrossCoreFlag `<0x2,PIPE_X>` Set + bare Wait、
§8.6 风格 HardEvent 预置/drain、prologue+延迟 PV 流水、GM 小槽轮转），**但 AIC/AIV 分流机制不同**：
- MLA 用 `#ifdef __DAV_CUBE__`/`#ifdef __DAV_VEC__` **预处理分支**（本 recipe §1）
- FA 用 `operator()<AscendC::AIC>()` / `operator()<AscendC::AIV>()` **模板特化**（见
  fa-paged-handcraft-recipe.md Step 2）

其余组件差异：FA 用 `MmadAtlasA2FAIQK/FAIPV` + `EpilogueAtlasA2OnlineSoftmax/RescaleO`（含 `FAIQKTail/PVTail`
尾块组件），L1TileShape K=128；launch 14 参 `(hwSync,q,k,v,mask,bt,o,qSeq,kvSeq,s,p,oTmp,oUpdate,tiling)`；
tiling 用 `FAInferTiling::GetFATilingParam(FAInfo,blockDim,FATilingData)`。
手搓 FA 时按 fa-kernel-handcraft.md §1-9（FAInferKernel `operator()<AIC>/<AIV>` 特化）写，
CrossCoreFlag/主循环/HardEvent 套用本 recipe §2/§6/§7 的 MLA 同款事实。

## MLA Paged 实现契约全文（fa-mla-paged）
> 以下为 MLA 契约源文件全量正文：实现策略与实测纠错、launch 14 参、tiling、workspace 分配表、shape/语义约束、精度验证、性能基线与吞吐模型、§8 实现架构（数学映射、kv-split、五大性能支柱、组件链、主循环、tiling 决策）。

> **导航**：设计路由见本册「设计路由全文」（MLA 小节）。本文聚焦实现期规则：算子语义、内核架构、host 集成契约、shape 约束、精度与性能基线。
> **实现基点**：MLA 与 FA 同范式（分页 KV + MQA latent 压缩），本 skill 提供整经验证的 MLA 内核（通用 H=16/32/64 + H=128 特化 + AMLA 特化），host 只做封装。
> **来源**：MLA paged 工程 / MLA 手搓工程 工程实测（Ascend 910B3 / dav-2201，catlass gitcode HEAD），18+20 case 精度全 PASS，与 ATB 融合算子标杆三方互证一致。

---

### 0. 实现策略：手搓 MLA kernel（★最高优先级，配方见 fa-mla-handcraft-recipe.md）

**MLA（Multi-head Latent Attention，分页 KV + MQA latent 压缩）与 FA 同范式：手搓路径已实证可一次跑通。**

**kernel 按 fa-mla-handcraft-recipe.md 手搓**（MLAKernel
逐行配方：组件全在 catlass include 层 + CrossCoreFlag/HardEvent 真实用法 + prologue+延迟 PV 主循环
+ GM 槽位公式 + launch/build 契约）。§8 的组件链伪代码已被 recipe 替代——**以 recipe 为准**。

> **★2026-09 实测纠错（本会话手搓踩坑，必读）**：§8 伪代码缺两个事实，导致手搓必死锁，
> recipe 已补正：
> 1. **AIC/AIV 分流必须 `#ifdef __DAV_CUBE__`/`#ifdef __DAV_VEC__` 预处理分支**，**不是**
>    `__mix__(1,2)`+runtime `g_coreType`（那是 kfc 单 TU 路径 §12.13，MLA CrossCoreFlag 路径不用）。
>    实测 `__mix__` 下 AIC 单独 pre-set+QK+drain 即死锁。
> 2. **主循环必须 prologue(QK0+softmax0 无PV) + loop[QK(n)+softmax(n) if非末 / 末块下task预取 /
>    PV(n-1)+rescale(n-1) 延迟一块]**，**不是** QK+PV 同块锁步。锁步的 CrossCoreFlag Set/Wait 顺序
>    与组件内部 HardEvent 配对不匹配→token 不平衡→死锁。
> 3. CrossCoreFlag 用法（§8.5 正确，实证 OK）：Set `Arch::CrossCoreSetFlag<0x2,PIPE_X>(flag)`，
>    Wait bare `Arch::CrossCoreWaitFlag(flag)`（dav-2201 上与 `<0x2,PIPE>` 配对）。
> 4. `glFlag[2]={1,1}` 真数组（勿 nullptr）；AIV `coreIdx=GetBlockIdx()/GetSubBlockNum()`（勿用裸 blockIdx）。

内核形态（覆盖矩阵）：

| 形态 | 适用 |
|---|---|
| 通用 MLA（H=16/32/64，fp16/bf16） | 模型侧 TP8/4/2 场景 |
| H=128 特化（TP1 规格，K 常驻 L1） | 模型侧 TP1 场景 |
| AMLA 特化（H=128） | AMLA 变体 |

组件 dispatch 固化为 `MmadAtlasA2MLAQK` / `MmadAtlasA2MLAPV`，host 不重选。

> **★路由修正（2026-08-28 定稿）**：MLA **手搓是唯一路径**，且必须按 §8 的**达标架构**（§8.9 五大性能支柱）写——单 kernel 锁步 + S/P 小槽轮转 + L1TileShape K=576 K 常驻 + MLAQK/MLAPV 专用 dispatch + flash-decoding split（组件全在 catlass include 层）。历史教训：朴素架构（双 launch + P 全量 GM 物化 + 通用 FAQK）实测仅 0.13~0.36，达标架构 1.16+——**慢的不是手搓，是走错架构**；两架构差异见 §8.9 对照表。

---

### 1. launch 契约：14 参，`aicCoreNum` 并度

```cpp
MLA<T><<<aicCoreNum, nullptr, stream>>>(
    hwSync,          // aclrtGetHardwareSyncAddr 取
    q, qRope,        // [numTokens, H, Dc+Dr 拆两段]
    k, kRope,        // paged KV cache: [numBlocks, 128, Dc] / [numBlocks, 128, Dr]
    blockTable,      // [batch, maxBlocksPerBatch] int32
    o,               // [numTokens, H, Dc] 输出
    s, p,            // workspace（见 §3）
    oTmp, globalo, oCoreTmp,  // workspace（见 §3）
    l,               // [numTokens, H] 归一化因子 workspace
    tiling);         // tiling buf 指针（GM）
```

**q_seqlen / kv_seqlen 只进 tiling，不是 kernel 入参**——与 FA 内核的 per-batch seqlen 契约不同，别照搬。

### 2. tiling：`GetMLATilingParam`

```cpp
uint32_t buf[(400 + numTokens * 17)];   // H==128 用 numTokens，否则 batch
MLAInfo info{...};                       // 含 batch/numTokens/qSeq/kvSeq/H/dtype
MLATiling::GetMLATilingParam(info, blockDim, buf);
// tilingKey 规则（launch 前设置）：
//   H != 128          → 0/1（MLA 通用，fp16/bf16）
//   H == 128 且 AMLA  → 4/5；当 numTokens%20<=10 且 batch<=40 时改走 7/8（MLATp1Spec）
```

**oCoreTmp / l 的尺寸依赖 tiling 结果**：先跑 `GetMLATilingParam`，从 `tiling[TILING_KVCORENUM=16]` 槽位读回 kvCoreNum 再分配这两个 buffer——不能在 tiling 之前按 aicCoreNum 猜。

### 3. workspace 分配表（实测验证的最小尺寸）

| buffer | dtype | 尺寸 |
|---|---|---|
| `s` | f32 | `aicCoreNum × 65536 × 2` |
| `p` | f16 | `aicCoreNum × 65536 × 2` |
| `oTmp` | f32 | `aicCoreNum × 65536 × 2` |
| `globalo` | f16 | `aicCoreNum × 65536 × 1` |
| `oCoreTmp` / `l` | - | 按 tiling 读回的 kvCoreNum 分配（§2） |

### 4. shape / 语义约束（host 侧必须前置校验）

- **算子语义**：`O = softmax(scale·(Q·c_kv + q_rope·k_rope))·V_latent`，`scale=1/√576`（576 = Dc+Dr，不是 √Dc）
- `q_seqlen ∈ [1,4]`（decode/MTP；官方 ATB 算子只支持 q=1，本内核支持 MTP q>1——这是对标杆标杆的反超点之一）
- `kv_seqlen ≥ 128`，`blockSize = 128`（尾块非对齐 OK）。★官方 gen_data 声明范围 [128,16384] **是验证范围非硬约束**——实测本内核与 ATB 标杆均可正确运行至 kv=204800（20万，精度同带 PASS，2026-08-24 长序列矩阵验证）；超 16384 后官方未验证，交付前须自测精度
- `Dc = 512`、`Dr = 64`
- **V 输入是 latent（c_kv 压缩域），不是解压后的 V**——语义写错 golden 直接错
- H ∈ {16, 32, 64, 128}，H=128 走 TP1 spec 路径（tilingKey 见 §2）

### 5. 精度验证

- verify 脚本按本 skill 模板编写（判定逻辑必须从模板复制），golden 用 numpy 纯 fp32 分解链
- 实测 18/18 语义矩阵 PASS（error_ratio=0，max_abs 1.3e-5 ~ 3.1e-4 vs 纯 fp32）；泛化 20 case 同带
- **三方互证**：本内核 vs 标杆 `npu_multi_head_latent_attention`（ATB）vs numpy golden——两两差异同数值量级（1.5e-5~1.2e-4）即互证通过

### 6. 性能基线与吞吐模型（910B3 实测，device 口径）

| 口径 | 结果 |
|---|---|
| vs torch_npu 分解链（gather+einsum+softmax+einsum） | 几何 33.5×（**墙钟口径，含 host 间隙，报告必须明示"融合 vs 分解"**） |
| vs 标杆 ATB `npu_multi_head_latent_attention` | **序列长扫描 gate 3.047**（同设备复测版；B2 固定、kv 128~204800：短 kv≤4096 持平 ~1.0，kv≥6144 ATB 线性退化 ~23μs/1k token、本侧 kv-split 平坦 ~3μs/1k → ratio 升至 **7.16**@kv20万，ATB 复测三次差 <1%）；batch 放大口径 gate 1.161（B2-10 反超 1.16~4.2×；B12-16 波谷 0.66-0.68；B22+ 持平） |
| 长 kv（单 batch 大 kv） | 本内核反快：kv8192 快 3.7×、kv16384 快 5×（ATB 长序列调度退化） |
| decode kernel 绝对时长 | 12.9 ~ 80.7 us（语义矩阵） |
| ★kvSplitCores 台阶 | kv≤4096 = 1（串行扫块）→ kv≥6144 = 8（并行切分）：kv=6144 比 kv=4096 快是 tiling 策略跳档的真实台阶，非测量误差（runner tiling 行实证） |

**吞吐定标模型**（构造/估时 case 用，见 fa-perf-gate-evaluation.md）：
`U = Σ T·H·kv_b`（score 单元数），饱和吞吐 H128≈64-72、H64≈33-42、H32≈20 units/ns。
- H32 慢 3×：通用内核无 K 常驻 L1，与 TP1 spec 差距的主因
- batch 增大吞吐升（B8→B40，H128 54→72 units/ns）——多 batch 是免费性能，评测矩阵应覆盖
- 时间与 B 非线性（kv-split 波次效应），定标需实测 2 轮迭代收敛

### 7. 已知问题

| 现象 | 根因 | 对策 |
|---|---|---|
| 官方 ATB 标杆 event 计时 4061us，device 实际 76us | ATB 每次 launch 重建 executor，host 开销巨大 | 标杆一律 profiler op_summary device 口径（见评测 doc 陷阱 2） |
| ATB 标杆跑 MTP（q>1）报错 | 官方算子仅支持 q=1 | 该 case 用 golden-only 验证；性能对比注明不可比 |
| NNAL/ATB 不可用（`libatb.so` load 失败） | 环境未装 NNAL | mindspore 镜像下 run 包 + `yes \|` 过 EULA 安装；装不了则报告明示"标杆对比不可做" |

---

### 8. MLA kernel 实现架构（手搓/复刻指南——从零写出 MLAKernel 的完整依据）

> **★权威源升级**：本节是设计依据（组件链/5 支柱/主循环伪代码/tiling 决策）。**实际生成代码以
> fa-mla-handcraft-recipe.md 为准**（MLAKernel 逐行配方，
> 含 §8 本节缺的两个死锁关键：`__DAV_CUBE__/__DAV_VEC__` 预处理分支、prologue+延迟 PV 流水）。
> 本节 §8.10 组件链/§8.11 主循环伪代码与 recipe 一致时以 recipe 为准；不一致时**以 recipe 为准**
> （recipe 已实测纠错）。
>
> 目的：按本节设计依据 + recipe 的真实代码结构，从零写出性能达标的 `MLAKernel`（gate 1.16+）。
> host 契约见 §1-§4，FA 通用手搓细节见 fa-kernel-handcraft.md §1-9。

#### 8.1 MLA 数学 → kernel 结构映射

```
O = softmax(scale · (Q·c_kv + q_rope·k_rope)) · V_latent
  c_kv:  latent 压缩 KV（key=value 共用），Dc=512
  q_rope/k_rope:  RoPE 拆分部分，Dr=64
  scale = 1/√576（Dc+Dr，不是 √Dc）
  → 关键：QK 是「两段点积相加」，不是单一 Q·K^T；V 就是 c_kv 本身
```

#### 8.2 Params（14 参，与 launch 一一对应）

`q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling`（13 数据 + tiling 槽）。注意比 FA 多 **qRope/kRope**（RoPE 拆分是 MLA 特有）和 **oCoreTmp/l/oUpdate**（flash-decoding 归并用）。GM buffer 声明：`GlobalTensor<ElementS/P/OTmp/uint32_t>`（s/p/oTmp）+ `ElementQ/K/.../int32_t`（数据）+ `ElementO/OTmp`（AIV 输出）。

#### 8.3 任务切分（AIC/AIV 共用同一遍历，flash-decoding 三重索引）

```
for (process = coreIdx; process < processNum; process += coreNum):
    curBatch     = process / (qHeadSplitNum × maxKvSplitCoreNum)
    qHeadSplitIdx = process % (qHeadSplitNum × maxKvSplitCoreNum) / maxKvSplitCoreNum
    curNIdx      = process % maxKvSplitCoreNum        // ← kv-split 段索引
    curKVSeqlen  = kvSplitPerCore（末段=剩余 kvSeqlen）
    rowNum = qHeadSplitSizeActual × qSeqlen
    nLoop  = (curKVSeqlen + blockSize-1) / blockSize  // 每核扫的 KV 块数
```
- 任务数 = batch × qHeadSplitNum × kvSplitCoreNum，`processNum` 由 tiling 给
- 每核独立扫自己那段 KV，算 partial O/l/m
- 行间 pingpong 交替调度：`isForward = !isForward`（把连续核的任务交错，均衡 L1 复用）

#### 8.4 kv-split（flash-decoding）并行——MLA 长序列性能的关键

- tiling 算 `kvSplitPerCore`（每核 KV 段长）、`kvSplitCoreNum`（切几段，≤8）
- AIC 每核算自己段的 `S_j = QK_j`，AIV 做该段的 **online softmax 但保留 running max/sum 到 partial**（不跨段归一）
- **flash-decoding 归并**（`epilogueMLAFDRescaleO`）：所有核算完后，AIV 扫全部 `kvSplitCoreNum` 段的 partial，按 `w = exp(m_j − m_final)` 缩放各段 O 再累加，最后 `÷l`——这就是 kv-split 正确性的核心，**漏了它就错数**
- 归并读 `gOCoreTmp[oFdOffset × kvSplitCoreNum + ...]`（per-split 槽）+ `gl[lOffset × ...]`（per-split l）

#### 8.5 AIC/AIV 跨核握手（3 个 CrossCoreFlag，modeId=0x2）

```
AIC 算 QK → SetFlag(qkReady, PIPE_FIX)   → AIV WaitFlag(qkReady) → 算 softmax
AIV 算 P   → SetFlag(softmaxReady, PIPE_MTE3) → AIC WaitFlag → 算 PV
AIC 算 PV  → SetFlag(pvReady, PIPE_FIX)   → AIV WaitFlag → 累加 O
```
- flag 对象：`qkReady/softmaxReady/pvReady`（id 常量 QK_READY_ID/SOFTMAX_READY_ID/PV_READY_ID）
- **modeId 必须 0x2**（双子核都要发/收，见 fa-handcraft-pitfalls.md H2）；`SetFlag` 指定 PIPE（FIX/MTE3）与 flag 事件要配对

#### 8.6 HardEvent 预置（AIC/AIV 各自入口，逐事件 ID 写全）

```cpp
// AIC（25 个）：入口按此顺序 SetFlag，出口逆序 WaitFlag
M_MTE1:   EVENT_ID0..7            (8 个)
FIX_M:    EVENT_ID0, EVENT_ID1    (2 个)
MTE1_MTE2: EVENT_ID0..7           (8 个)
FIX_MTE1: EVENT_ID0..5            (6 个)
MTE2_FIX: EVENT_ID0               (1 个)
// AIV（15 个）：
MTE3_V:   EVENT_ID0, EVENT_ID2, EVENT_ID3      (3 个)
MTE3_MTE2: EVENT_ID0, EVENT_ID2..7             (7 个)
V_MTE2:   EVENT_ID0, EVENT_ID1, EVENT_ID4, EVENT_ID2, EVENT_ID3  (5 个)
```
- 漏一个或顺序错 → 流水死锁（与 FA kernel 同规则）；kernel 出口逐一 WaitFlag 配平。

#### 8.7 双特化 + 模板链（写 kernel 时的骨架）

```cpp
template <class BlockMmadQK, class BlockMmadPV, class EpilogueMLASoftmax,
          class EpilogueMLARescaleO, class EpilogueMLAFDRescaleO>
class MLAKernel {
    using ElementS=BlockMmadQK::ElementC; using LayoutS=...;  // 派生所有类型
    struct Params {...};                    // 14 参
    CATLASS_DEVICE void operator()(Params&); // #ifdef __DAV_CUBE__/__DAV_VEC__ 双分支
    Arch::CrossCoreFlag qkReady/softmaxReady/pvReady;
};
// 外层 MLA<T> 模板实例化: dispatch MmadAtlasA2MLAQK/PV + MLASoftmax/RescaleO/FDRescaleO
```
- dispatch 固化 `MmadAtlasA2MLAQK`（QK 两段：Q·c_kv 与 q_rope·k_rope）/ `MmadAtlasA2MLAPV`（P·V）
- H=128 走 TP1 spec（`mla_kernel_tp1_spec.cpp`，4-block 基块、无 K 常驻）；AMLA 变体用 `amla_kernel_tp1_spec.cpp`

#### 8.8 手搓自查（写完后逐条过）

- [ ] 14 参 launch 与 Params 字段一一对应
- [ ] 任务切分：process 遍历 + curBatch/qHeadSplitIdx/curNIdx 三重索引正确
- [ ] kv-split：kvSplitPerCore/kvSplitCoreNum 从 tiling 读、末段剩余长度处理
- [ ] flash-decoding 归并：per-split O/l 槽 + 跨段 rescale 累加（漏了错数）
- [ ] 3 个 CrossCoreFlag modeId=0x2、PIPE 配对、双子核都发
- [ ] HardEvent 入口 SetFlag 数量/顺序与出口 WaitFlag 配对
- [ ] q_rope/k_rope 正确参与 QK（两段点积相加）
- [ ] Dc=512/Dr=64/scale=1/√576/V=latent 语义正确

> 完成 §8 后，可直接按 fa-perf-gate-evaluation.md 评测。

#### 8.13 依赖清单与闭卷实证（纯 md 生成的可行性证据）

- **依赖仅两样**：catlass 库头（`catlass/include`，§8.10 组件链 include 的全部 `catlass/*` 头）+ ASC 工具链（`kernel_operator.h` 自动提供）。无其它本地依赖，**不读任何外部样例**。
- **闭卷冒烟实证（2026-08-25，本节架构）**：全新工程只按契约+组件链从零组装：一次编译通过；B2/H16 kv=128/4096/6144/16384 四点精度 PASS（err_ratio=0，max_abs 1.5e-5~8.9e-5）；`kvSplitCores` 台阶正确（kv≤4096=1、kv≥6144=8）。
- **精度与性能全量证据**：语义矩阵 18/18 + 序列长分桶 20/20 PASS（含 MTP/尾块非对齐/bf16/varlen）；kv=204800（20万）精度同带 PASS（官方声明范围仅 16384，实测远超）；vs 标杆 ATB 序列长 gate 3.116（kv20万 时 7.44×）、batch 放大 1.161。

#### 8.14 ★tiling 来源坑（M7 级元坑：验证的是链接的文件，不是目录里的文件）

实测案例：某验证工程 CMake `MLA_KERNEL_SOURCES` 实际链接的是**外部仓的 tiling**——工程根目录同名文件是历史手搓期修改版、**从未被链接使用**；据此打包的资产行为完全不同（kvSplit 恒 20、小 kv 更慢）。**教训：任何"以工程为事实源"的移植/打包前，必须核对 build 实际链接的文件**（CMake 源列表指向 + quoted-include 解析顺序），不能只看工程目录里有什么。

#### 8.9 ★为什么"朴素手搓"慢一个数量级——快内核的 5 个性能支柱（对照表）

实测对照：朴素手搓（FAQK 组件 + 双 launch + GM 物化）gate **0.13~0.36**；本节架构的内核 **1.16+**。差距全在架构，不在微调：

| # | 性能支柱（快内核） | 朴素手搓的对应错误 | 量级影响 |
|---|---|---|---|
| 1 | **单 kernel AIC/AIV 锁步**（一次 launch，3 个 CrossCoreFlag 逐块握手） | 双 launch（A 算 QK+softmax、B 算 PV+归并），launch 间隙+两遍 GM 往返 | ~2× |
| 2 | **S/P 只在 per-core GM 小槽轮转**（TMP_SIZE=64K/半区 32K，双缓冲），P 不物化全量 | P_unnorm 全量 [rows×kv] 写 GM 再读回 | 内存带宽型，kv 越大越差 |
| 3 | **L1TileShape K=576=embed+rope**：K 的 L1 装载覆盖整个 embedding，Q 块复用 | 通用 FAQK 分块（K=128），K 反复重装 | cube 利用率 |
| 4 | **专用 dispatch**：`MmadAtlasA2MLAQK`（Q·c_kv 与 q_rope·k_rope 两段在组件内一次完成）+ `MmadAtlasA2MLAPV` | 用通用 QK 组件+手工拼两段点积（多一次 GM 中转） | ~1.3× |
| 5 | **flash-decoding kv-split**（长 KV 切多核 + FDRescaleO 归并） | 单核串行扫全 KV | 长序列 3-7× |

**结论：MLA 手搓必须照搬这 5 条**（组件全在 catlass include 层），否则必慢一个数量级。

#### 8.10 ★组件选型与实例化链（逐行照写；全部 catlass include 层组件）

```cpp
// dtype: half / bfloat16_t；S/OTmp/Update = float；Q/K/V/P/O = Dtype；Layout 全 RowMajor(K=ColumnMajor)
using L1TileShape = GemmShape<128, 128, 576>;          // ★K 必须 = embed+embedRope(512+64)=576（K 常驻 L1 的实现方式）
using L0TileShape = L1TileShape;
using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQK;        // QK 两段点积一次完成
using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;
using EpilogueMLASoftmax = Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLASoftmax, PType, SType, MaskType>;
using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPV;
using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;
using EpilogueMLARescaleO = Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLARescaleO, OType, OUpdateType, OTmpType>;
constexpr uint32_t ComputeEleNum = 6144;                // FD 归并的每次处理元素数
using EpilogueMLAFDRescaleO = Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum>, OType, lType>;
```
- H=128 特化：`mla_kernel_tp1_spec.cpp`（4-block 基块、无 K 常驻路径）；AMLA 变体同目录。手搓按本节通用路径，H=128 时参考特化的基块差异。

#### 8.11 ★主循环逐步（性能关键：三处流水预取 + 全局 pingpongIdx）

常量：`TMP_SIZE=65536`（P/Update 槽）、`TMP_SIZE_DECODER=32768`（S 槽）、`BLOCK_SIZE=16`、flag id 1/2/3。

```
task 遍历（process = coreIdx; process < processNum; process += coreNum）：
  ├─ 正逆交替：realProcess = isForward ? process : (组内最大 process − process%coreNum)；isForward 翻转
  │   （把相邻任务的 KV 块顺序倒过来 → 同批 KV 的 L1 驻留被下一任务复用）
  ├─ kerFlag 翻转：curNIdx = maxKvSplitCoreNum−1−curNIdx（每两任务镜像 split 段，均衡尾段负载）
  ├─ 三重索引 → curBatch/qHeadSplitIdx/curNIdx；rowNum = qHeadSplitSizeActual × qSeqlen
  │
  ├─ [首任务 prologue] 发起块 0 的 QK → Set(qkReady)；isFirstTask=false
  │
  └─ for nIdx = 1 .. nLoop：                    ★1 块深流水
      ├─ (nIdx<nLoop) 发 QK(nIdx)：S → 槽[pp%2] → Set(qkReady)
      │    block_table 查表：blockId = table[batch*maxBlocks + startKV/128 + nIdx]
      │    kvOffset = blockId × 128 × strideKV（K/KRope 各一）
      ├─ (nIdx==nLoop 且有下个 process) ★任务级预取：向前扫描 coreNum 步找到下个有效 process，
      │    发它的块 0 QK（流水跨任务不断流）
      └─ 发 PV(nIdx−1)：P 槽[(pp−1)%2] → OTmp → Set(pvReady)
  AIV 侧对称：Wait(qkReady) → epilogueMLASoftmax(S→P) → Set(softmaxReady)
              Wait(pvReady) → epilogueMLARescaleO(O 累加/÷l；isLastNTile 时写 FD partial 槽)
（PV 的 blockMmadPV 调用把 softmaxReady 作为入参传入——组件内部 Wait 它保证 P 就绪后再 MMAD；
  locPingPongIdx 为 AIC 侧本地乒乓，与全局 pp 独立维护）
```

**GM 槽位公式（pp = 全局 pingpongIdx，跨任务连续递增）**：
```
S 槽    = coreIdx × TMP_SIZE_DECODER + (pp%2) × TMP_SIZE_DECODER/2
P 槽    = coreIdx × TMP_SIZE        + (pp%2) × TMP_SIZE/2
OTmp 槽 = coreIdx × TMP_SIZE×2      + ((pp−1)%2) × TMP_SIZE
```
- Q 指针不走 params：host 把**每 batch 的 Q/QRope/O/l/oFd 64 位地址拆 hi/lo 32 位写进 tiling 槽**（kernel `GetValue` 高低位拼回），params 只给各 tensor 基址——per-batch 地址差异全由 tiling 承载。
- blockMmadQK 调用形态：`blockMmadQK(gQ[gQOffset], gQRope[gQRopeOffset], gK[kvOffset], gKRope[kvOffsetRope], gS[gSOffset], layoutQ, layoutQRope, layoutK, layoutKRope, layoutS, GemmCoord{rowNum, kSeqTile, embed+embedRope}, MatrixCoord{qHeadSplitSizeActual, embed}, qHeads, nIdx, pingpongIdx)`——**5 张量 + 5 layout + coord 三件**（QK 组件吃双 Q/K 对）。

**flash-decoding 归并（任务循环后、AIV 侧收尾）**：
```
headsProcess = min(ComputeEleNum/embed, HEADS_PROCESS_MAX)；loopsPerBatch = ⌈qHeads/headsProcess⌉
逐 loop 调 EpilogueMLAFDRescaleO(gO[...], gOCoreTmp[oFd 槽], gl[l 槽], actualHeads, headsProcess, embed, kvSplitCoreNum)
  —— w = exp(m_j − m_final) 跨 split 段缩放累加 + ÷l 一次完成
```

#### 8.12 ★tiling 槽位表与 kv-split 决策（host 逐槽填）

头部槽（uint32 下标）：`0 batch, 1 numHeads, 2 headDim(512), 3 numBlocks, 4 blockSize(128), 5 maxBlocks, 6 tor(float 重解释), 7 kvHeads, 8 TILING_HEAD_SIZE(=头部长度), 9 TILING_PARASIZE(=per-batch 段长), 10 headSplitSize, 11 headSplitNum, 13 headDimRope(64), 14 maxKvSeqlen, 15 kvSplit, 16 kvCoreNum, 18 totalQTokens, 19 processNum`。
per-batch 段（基址 = 槽 8 的值 + 槽 9 的值 × b）：`+0 qSeqlen, +1 kvSeqlen, +2 realBatch(排序后原 batch 号), +4/5 qAddr hi/lo, +6/7 qRopeAddr hi/lo, +11/12 lAddr hi/lo, +13/14 oFdAddr hi/lo, +15 kvSplitPerCore, +16 kvSplitCoreNum`。

**kv-split 决策（T+1 均衡法，逐句可写）**：
```
isKVSplit = (maxKvSeqlen ≥ blockDim×2×128) && (batch ≤ blockDim×0.8) && (maxQSeqlen == 1)   // KV_SEQLEN_SLICE=128, SPLITKV_RATION=0.8
不切：每 batch kvSplitCoreNum=1、kvSplitPerCore=kvSeqlen，processNum=batch
切：MIN_TOKENS_PER_SPLIT = 4×blockSize；MAX_SPLIT_PER_TASK = 8
    batchSplitNum 全 1，totalAllocated=batch
    while totalAllocated < blockDim:
        选 currentLoad=alignedKv[i]/split[i] 最大且 (split[i]<8 且 nextLoad≥MIN_TOKENS) 的 i → split[i]++
    kvSplitPerCore = ⌈⌈kv/128⌉/split⌉ × 128；kvSplitCoreNum = ⌈kv/kvSplitPerCore⌉
    processNum = Σ kvSplitCoreNum；槽 16 填 MAX(split)
```
（batch 按 kvSeqLen 降序排序后填段——长 KV 先分；realBatch 槽携带原 batch 号供 block_table 寻址。）

## FFA / FIA 双 KV 算子配方全文（ffa-handcraft-recipe）
> 以下为 FFA recipe 源文件全量正文：37 参入口契约、双 KV 语义、与 paged recipe 的差异清单、主循环骨架、tilingKey 路由、推荐路径对照与标杆对比。

> **用途**：FFA / FIA（fused_infer_attention_score 类算子，DeepSeek 推理形态）的**算子语义与 kernel
> 组件级结构**——双 KV（TensorList + 共享前缀）、sink 进 epilogue、FD、LSE 输出、nZ·zN KV、量化
> antiquant 入参族的完整契约。
> **生成路径选型（先读这段）**：
> - **手搓生成 FFA（双 KV 对角形态）→ 手册 §12.5.14 终态配方（实证几何 2.19×，
>   精度双口径 PASS）**——FAQK 流式配方，已闭卷验证的生成路径。
> - 本文件的价值：该算子族的**组件级结构事实**（sink-in-epilogue / FD 内建 / LSE 输出 / nZ·zN KV /
>   量化 antiquant 入参族）——生成时对齐算子语义、排查对照、或按组件路径复刻时用。
> - 该 kernel 与 fa-paged-handcraft-recipe.md **同架构**（catlass
>   BlockMmad + CrossCoreFlag `<0x2,PIPE_FIX>` + OnlineSoftmax/RescaleO epilogue + 512 大块堆叠 +
>   PRE_LAUNCH 延迟 PV）——先掌握 paged recipe 再读本文件差异清单即可。

---

### 1. 算子语义（37 参入口契约）

```
O = softmax(scale·(Q[+qRope]·[KSharedPrefix; KTensorList][+kRope]ᵀ + pse + mask)[+sink]·
            [VSharedPrefix; VTensorList]) → attentionOut + softmaxLse(可选)
```

| 入参族 | 参数 | 说明 |
|---|---|---|
| 主体 | query, key(TensorList), value(TensorList) | ★KV 是 **TensorList**（多 tensor 列表）——双 KV 形态 |
| 共享前缀 | keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen | 全 batch 共享的公共前缀 KV（int4 输入时 host 仅做 INT4 视图转换（dtype/view 改写，无数据拷贝）） |
| rope | queryRope, keyRope, keyRopeAntiquantScale | rope 旁路（与 MLA 同思路） |
| sink | learnableSink | ★sink 在 epilogue 内建（`Epilogue::SinkMode`/`ElementSink` 模板参） |
| 量化 | deq/quantScale1/2, quantOffset2, antiquantScale/Offset, key/valueAntiquantScale/Offset | W8A8/反量化入参族（kernel 内 on-load dequant 路径） |
| 变长 | actualSeqLengths(Q)/KV, queryPaddingSize, kvPaddingSize | 累加和口径 + padding 标记 |
| 分页 | blocktable | 跨 tensor 池的物理块号表 |
| 输出 | attentionOut, softmaxLse | LSE 可选（`*_LSEOUT_*` tiling key 位） |

**双 KV 语义**：sharedPrefix（公共前缀，所有 batch 共享一段）+ TensorList local（per-batch 各自池）；
kernel 侧统一按 **paged blockTable 连续寻址**（tiling 的 kvSeqlen = sharedLen + localLen_b 逻辑拼接），
不分两次 launch、不做 host 拼接拷贝。

### 2. kernel 结构（与 fa-paged recipe 的差异清单）

| 维度 | fa-paged（标准 FAInfer 形态） | FIA regular 形态 |
|---|---|---|
| 分流 | `operator()<AIC>/<AIV>` 模板特化 | `#ifdef __DAV_C220_CUBE__/__DAV_C220_VEC__` 同文件并列（MLA 式） |
| 任务型 | `KERNEL_TASK_TYPE` 未显式 | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` |
| 组件 | FAIQK/FAIPV + Tail | 同组件族 + `resetBlockStart(kvStart, pagedBlockSize)`（每 task 重置块内游标） |
| KV 布局 | ND | ND 或 **nZ（K）/zN（V）** 分支（`PAGED_CACHE_FLAG && is_same_v<LayoutK,nZ>`：`kRow=blockStack*strideK` 重排 layout） |
| 大块 | blockStackNum=4（512 列） | `MAX_KV_STACK_LEN` 常量（等价 512）+ `preKVNum=PRE_LAUNCH` 延迟 PV 同款 |
| sink | 无 | `gSink[gmOffsetSink]` 直传 epilogueOnlineSoftmax（8 处调用点全部带 sink 参） |
| mask | 压缩 [1024,1024] | `COMP_TRIU_MASK_DIM_LEN` 压缩上三角（同思路，尺寸常量名不同）+ `LayoutQ layOutFullMask(pseQ,pseKv)` |
| FD | 无 | `IS_FD` 模板位：`kvStart=stS2IdxNow / kvEnd=enS2IdxNow`（flash-decoding 每核只扫自己的 s2 段，MLA FDRescaleO 的组件版对应物） |
| 空任务 | — | `kvSLoopNumTotal<=0 || startIdx>=...` 时 VEC 侧 `epilogueInitOut`（直接初始化输出，防读未写） |
| LSE | 无 | `gLse` + `LayoutLse(totalQTokens, qHeads)` |
| sparse 跳块 | noSkipKvS 公式 | 同公式 + `delStartRow/delEndRow`（band 边界行修正） |
| workspace 槽 | `coreIdx*WS_DB*(PRE_LAUNCH+1)+slot*WS_DB` | 同公式（WORKSPACE_BLOCK_SIZE_DB 同源） |

### 3. 主循环骨架（与 paged recipe Step 3f/4e 同构）

```cpp
blockMmadQK.resetBlockStart(kvStart, pagedBlockSize);
blockMmadPV.resetBlockStart(kvStart, pagedBlockSize);
blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeads);
for (uint32_t kvSIdx = kvStart; kvSIdx < kvEnd + preKVNum; kvSIdx++) {
    if (kvSIdx < kvEnd) {
        stackSeqTile = (尾块) ? noSkipKvS - kvSIdx*MAX_KV_STACK_LEN : MAX_KV_STACK_LEN;
        isLastStackTile = (kvSIdx + 1) >= kvSLoopNumTotal;
        uint32_t slot = stackSeqCount % (PRE_LAUNCH + 1);
        gmOffsetS = coreIdx*WS_DB*(PRE_LAUNCH+1) + slot*WS_DB;
        GemmCoord shapeQK{rowNum, stackSeqTile, embed};
        LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
        CUBE: blockMmadQK(gQ, gK, gS, gBlockTable[...], layouts, shapeQK,
                          kvSIdx, kvSLoopNumTotal, pagedBlockSize, strideK, keyBnStride);
              CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
        VEC:   (MASK_CAUSAL 分支) epilogueOnlineSoftmax(gP, gS, gSink, gMask, layOutP, layOutS,
                  layOutMask, layOutFullMask, shapeQK, (stackSeqCount==0), qSBlockSize, qNBlockSize, slot, ...);
              CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
    }
    if (kvSIdx >= PRE_LAUNCH) { /* 延迟 PV：blockMmadPV(..., softmaxReady) + Set(pvReady) → VEC rescaleO */ }
    stackSeqCount++;
}
```

**与 paged recipe 的 3 条修复仍适用**（LayoutP stride / isLast / L1 偏移）；mask 版 softmax 调用带
sink/fullMask 扩参（paged recipe Step 8e 的 superset）。

### 4. tilingKey 路由（aclnn registry 分发用）

`QF16_KVF16_OUTF16_{NOLSEOUT,LSEOUT}_TND_{NOCACHE,PAGEDCACHE}[_KVNZ]_{NOMASK,CAUSALMASK}_SPLITFUSE_TILING`
——dtype × LSE × paged × KVNZ × mask 五维组合；decode 形态另有 `FAInferDecoding`（`IS_FD` 段切分版）。
**手搓直调不需要 tilingKey**（仅 registry 分发用）。

### 5. 手搓生成 FFA 的推荐路径（对照表）

| 需求形态 | 路径 |
|---|---|
| 双 KV 对角 FFA（`S=Q·K1ᵀ+Q·K2ᵀ对角`） | **配方册 §12.5.14 终态配方**（FAQK 流式 + K 块流式在线 softmax，几何 2.19×，实证） |
| sharedPrefix + local（DeepSeek 推理形态） | fa-paged recipe 基座 + 本文件 §1 语义（blockTable 跨池编址 + kvSeqlen=shared+local）+ 特性册 §12.14-B paged 查表 |
| sink 叠加 | §12.13-C（kfc 播种）或 fa-sink recipe §5.2（SoftmaxFlashV2 isUpdate） |
| W8A8 量化 KV | antiquant 入参族 + §12.12-B dequant 相位（DataCopy→Cast 间 PIPE_ALL，scale=1/s） |
| FD 长序列 decode | MLA recipe §8（FDRescaleO 组件版）或 §12.13 原子累加版 |

### 6. 标杆对比

- Python 标杆：`torch_npu.npu_fused_infer_attention_score*`（注意 executor 每次 launch 重建，计时口径
  见 fa-perf-gate-evaluation.md：必须 profiler op_summary device 口径）。
- 精度互证：smax/ssum 应与标杆位级一致（配方册 §12.5.14 验证闭环同款）。

## 流水排布、同步关系与数值精度

- CrossCoreFlag 编排、HardEvent 预置与 modeId 配对规则与标准 FA 相同（见主 concept）。
- **HardEvent Set/Wait 严格配对**：每个事件 id 每 tile 恰好 1 Set + 1 Wait；Set 过剩 → aicore trap
  （timeout），Wait 过剩 → 死锁。改预取/驻留复用逻辑后必须逐 id 推演账目平衡。[^fa-compound]
- AIV→AIC 方向 flag 双子核都必须发（2Set:1Wait 配平）；反向 AIC→AIV 单 Set 可满足双 Wait。[^fa-handcraft-manual]
- 数值精度：softmax 状态链分槽（step s 读 (s−1)%N 槽、写 s%N 槽）；`P` 送 MMAD 前转输入 dtype；
  器件级 API 墙——`Brcb` 行广播输出布局与头文件注释不符（3 种用法全数值错）、`ReduceSum` 整行
  count=128 静默错（单 repeat 上限 64）、可用替代=整行 `ReduceMax(count=128)` 与 `src1RepStride=0`
  行广播；AIV 向量 API 调用风暴是假「标量瓶颈」真凶，每 chunk 向量调用次数是第一指标。[^fa-compound]

# 约束

- **死锁三前提**（`BlockMmad + Epilogue + CrossCoreFlag` 从零拼装可行的必要条件，缺一即 token
  不平衡死锁）：①分流用 `#ifdef __DAV_CUBE__/__DAV_VEC__`（MLA）或 `operator()<AIC>/<AIV>` 特化（FA）；
  ②主循环 prologue+延迟 PV（非同块锁步）；③HardEvent 预置/drain 按全表逐一配平。[^fa-handcraft-manual]
- S1 路径不开 L1Reuse、splitFactor 不 ×2（aicRatio=1）；D∈[64,128] 的 s1s2 路径则必须 L1Reuse。[^fa-recipe-book]
- 轻量纯向量 merge/归一化 kernel 是单上下文单子核（`GetSubBlockIdx()` 恒 0），按双子核写会 rows 8-15
  全零；双子核共享同核 UB，写缓冲按 `subIdx × SEG` 显式分区。[^fa-handcraft-manual]
- ASC 编译器不吃 device lambda（写私有成员函数）；设备 printf/int→float 隐式转换不可用（GM dump 槽
  + host 读回调试）。[^fa-handcraft-manual]
- 归约 buffer（ubMax/ubSum）一律 512B 对齐起址；循环内 V↔MTE 复用小缓冲必须两段式（全部计算完→
  屏障→全部搬运），`PIPE_ALL` 都挡不住交替复用竞态。[^fa-handcraft-manual]
- MLA host 侧 shape/语义约束前置校验（H=16/32/64 通用 + H=128 TP1 特化 + AMLA 特化）；手搓版性能
  是标杆 kernel 的 1/3~1/9，只作正确性/兜底与组件无覆盖形态。[^fa-mla-paged]
- 性能不过精度的优化不可用（kv-split 曾 53.7× 但多块 FAIL 回退）；kernel 源码迭代禁 sed 批量覆盖
  （不可回滚）。[^fa-compound]

# 失败表现

架构级坑（挂死或错数级，全部实测）：[^fa-handcraft-manual]

| # | 坑 | 现象 | 对策 |
|---|---|---|---|
| H1 | FAQK 双实例不可共存 | 同进程 new 第二个 FAQK 仅构造即破坏 HardEvent 配平 → 出口挂死 | QK 单实例；多任务实例内分趟 |
| H2 | AIV→AIC Set 双子核都必须发 | 单子核发 → AIC WaitFlag 永阻塞 | 2Set:1Wait |
| H4 | 双子核共享同核 UB | 互相覆盖 | 按 subIdx×SEG 分区 |
| H5 | ubMax/ubSum 未 512B 对齐 | 越界覆盖相邻行 | 512B 对齐起址 |
| H6 | 循环内 V↔MTE 复用竞态 | 数据 +1 行移位 | 两段式 + 缓冲物理分开 |
| H7 | CMake 不追踪 .asc 内嵌 include | stale binary 假象（改了没效果） | touch 主 .asc 或 rm -rf build 全量；断言 exe mtime > 源 mtime |

观测元坑：stale dump 假验证（kernel 挂死后 verify 读上一次 dump → 逐位相同的假 PASS——verify 前必删
dump）；memset 排队污染计时（争用卡 25ms/launch，事件对分开报 MEMSET_US/KERNEL_US）；对标参数未
生效（精度/性能分别起进程忘带同一 env 旋钮——旋钮写入 tiling 并回显）；链接文件≠目录文件（打包/移植
前核对 CMake 实际链接的源）。[^fa-compound]

精度随步数劣化按序排查：×8 冗余索引 → 状态链分槽 → MM2 双 set surplus（下下个 PV 提前通过覆盖还在
被读的 oUb，rows 32-63 损坏）。单块对多块错按「跨块 rescale → P 行 padded 步长 → oTmp 单槽竞态」
三条序查。[^fa-handcraft-manual]

# 验证方法

- 通用双口径（内部 golden + `aclnnFlashAttentionScore`/`npu_fusion_attention` 标杆）与 mixed
  tolerance 阈值同主 concept。
- **paged**：恒等 block_table（`table[t]=t`）结果须与非 paged 位级一致；golden 用稠密 gather 后的
  K/V 计算。[^fa-compound]
- **causal**：golden 按行限列计算；`matched=1.0` 且 causal 恰比 full 省一半时间（上三角整块跳过）。[^fa-compound]
- **varlen**：标杆 varlen 只有 TND 形态（BNSD+长度数组需 host 转 TND，语义等价已验证 matched=1.0）；
  seqlen 契约两边核对（recipe 传累加、kernel 差分）。[^fa-varlen-recipe]
- **sink/q8 复合**：≥30/30 PASS 含超长序列；对标杆 gate ≥0.6（实测 varlen 1.12x / q8 1.50x / sink
  0.81x，Sq 至 196608）。[^fa-compound]
- **MLA**：精度逐 case 自测（kv>16384 属标杆未验证区）；性能以 device 口径 gate（封装工程实测
  3.047，长段最高 7.16×；手搓对照组 0.107~0.364 属预期）。[^fa-mla-paged]
- **FFA**：双口径 PASS + 终态配方实证几何 2.19×。[^ffa-recipe]
- 评测矩阵强制 B>1 且各 batch 序列长不等（多 batch bug 探测器）。[^fa-compound]

[^fa-handcraft-manual]: 固定提交中的 FA kernel 手搓手册：四文件结构、任务切分（N pack 进 M、GQA 双平面索引）、AIC/AIV 双入口主循环、跨核同步编排、HardEvent 预置全表、BlockMmad 流水、softmax/rescale epilogue 内部、host 直调模板与变体映射法。
[^fa-recipe-book]: 固定提交中的 FA 算子配方库：FFA 手搓实证与 §12.5.14 终态配方、s1s2 算法原理（L1Reuse/S1 无 aicRatio×2）、S1 sink 流程与 kfc 单 TU 直调配方及 workspace 公式。
[^fa-compound]: 固定提交中的 FA 复合特性叠加册：sink/varlen/q8/causal/paged 手搓落点与步骤、器件级 API 墙（Brcb/ReduceSum/ReduceRepeat）、AIV 调用风暴、HardEvent 配对、测量与泛化坑、S1/kfc 检查清单。
[^fa-paged-recipe]: 固定提交中的 FA-Paged 手搓完整配方：四文件端到端逐步骤（Step 1-11）、3 条关键修复（LayoutP 3 参/isLast/L1_QK_SIZE 偏移）与 mask 双循环路径。
[^fa-sink-recipe]: 固定提交中的 fa-sink/标准 FA S1 模板配方：单核融合三深流水、成员与 UB 布局、sink 播种（SoftmaxFlashV2 isUpdate）、sparse/causal 跳块公式与 workspace 精确公式。
[^fa-varlen-recipe]: 固定提交中的 fa-varlen TND 手搓配方：差分解码、TND 直读零转换、批界 s2End 截断、needL1Carry、SameAB 与两条路径对照表。
[^ffa-recipe]: 固定提交中的 FFA/FIA 算子配方：37 参入口契约、双 KV 语义、与 paged recipe 的结构差异清单、主循环骨架与标杆对比。
[^fa-mla-recipe]: 固定提交中的 MLA 手搓端到端配方：__DAV 分流、HardEvent 全表预置、prologue+延迟 PV 主循环、GM 槽位公式、FD rescale、launch/build 与 tiling 槽位表。
[^fa-mla-paged]: 固定提交中的 MLA 实现契约：launch 14 参、tiling、workspace 分配表、shape/语义约束、性能基线与五大性能支柱、tiling 来源坑。
[^fa-dev-skill-route]: catlass-op-develop SKILL.md 中的 FA 族形态全覆盖检索路由原文（L83/86）：任何 FA 族形态一律按手搓手册路由 + FA 族性能评测 gate 路由。
