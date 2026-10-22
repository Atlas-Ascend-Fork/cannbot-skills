---
type: "operator"
title: "FA-SageAttention Ascend 950 量化注意力实现模式"
description: "SageAttention（K 平滑 + INT8 QK + online softmax + FP16 PV）在 dav-3510 上的双 kernel 架构、量化路径与 comp 补偿代数、CrossCore mode4 协议、8 条不变量 I1-I8、精度取证方法与平台坑排障路由。"
tags: ["catlass-cpp", "flash-attention", "sageattention", "sage", "quantized-attention", "int8", "ascend950", "dav-3510"]
status: "stable"
generated: {"by": "process:catlass-cpp-fa-migration-curation", "at": "2026-10-22T00:00:00Z"}
verified: [{"by": "process:fa-source-cross-check", "at": "2026-10-22T00:00:00Z"}]
sources:
  - {"id": "sage-architecture", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-sageattention-architecture.md", "title": "FA-SageAttention 完整架构规格：硬件前提、GM 布局、双 kernel、flag 表与主循环", "kind": "repository"}
  - {"id": "sage-recipe", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-sageattention-recipe.md", "title": "fa-sageattention 开发流程七步配方：量化路径、工作流、排障路由与验收", "kind": "repository"}
  - {"id": "sage-rules", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-sageattention-rules.md", "title": "FA-SageAttention 硬规则：8 条不变量 I1-I8 与运行纪律 Never/Always", "kind": "repository"}
  - {"id": "sage-precision", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-sageattention-precision.md", "title": "精度/挂死取证方法论：numpy 精确仿真、GM dump、逐层判定与假设拟合", "kind": "repository"}
  - {"id": "sage-troubleshooting", "resource": "git:cann/cannbot-skills@287b4fd5461bbe1c0e1ae26c9da34dc7596aae41:ops/catlass-op-develop/references/patterns/fa-sageattention-troubleshooting.md", "title": "Ascend950 实测平台坑全集（编译/API/L0/同步/精度/新版 catlass 适配）", "kind": "repository"}
operator_families: ["flash-attention"]
architectures: ["ascend950"]
---

# 接口与概念

SageAttention 是 FA 族的**量化注意力**变体，落在 **Ascend 950（dav-3510）**：本 concept 的
`architectures` 与 family 内另两个 concept（atlas_a2_a3）不同，查询时必须指定 `--arch ascend950`。

输入：`Q ∈ R^{B×H×Sq×D}`、`K/V ∈ R^{B×Hkv×Skv×D}`（fp16/bf16，HND）；输出：
`O = softmax(QKᵀ·sm_scale + causal_mask)·V ∈ R^{B×H×Sq×D}`（fp16，kernel 直写 HND）。[^sage-recipe]

数据布局契约（GM）：[^sage-architecture]

| 张量 | 布局 | 说明 |
|------|------|------|
| Q̂ | BSND `[B·sqPad, H·D]` int8 | 量化后 Q，pad 行零 |
| K̂ | BSND `[B·kvBlocks·128, Hkv·D]` int8 | 平滑后量化 K |
| μ | `[B·Hkv·kvBlocks, D·8]` fp32 | fp32 原值 **×8 冗余**（写 4KB 可靠、读 512B 即可） |
| SQ/SK scale | `[blocks, 8]` fp32 | ×8 冗余；FA 读时下标必须 ×8 |
| V | BSND `[B·skPad, Hkv·D]` fp16 | host 从 HND 转 |
| po（splitK partial O） | `[totalWork][128][128] fp32` | 未除 l 的 oacc |
| pml（partial m/l） | `[totalWork][512] fp32` | {sub0: m[64]\|l[64]\|pad, sub1: 同} |

## 算子算法

量化路径 v1（quantScheme=0）：[^sage-recipe]

```text
1) K 平滑:  每 128 行 KV block, μ = mean(K[blk], axis=row) ∈ R^D
            K_s = K − 1·μᵀ; smax = max(amax|K_s|, amax|μ|); sK = 127/smax
            K̂ = round(K_s·sK) → INT8
2) Q 量化:  sQ = 127/amax(|Q[blk]|);  Q̂ = round(Q·sQ) → INT8
3) QK^T:    S_int = (Q̂·K̂ᵀ)_{INT8×INT8→INT32}
            S = S_int·(sm/(sQ·sK_j)) + comp_j + mask
            comp_j[r] = dot(Q̂[r,:], μ_j)·(sm/sQ)   ← K 平滑补偿, sK 代数消去
4) softmax: online（行 max 减法 + exp + 缩放），P ∈ [0,1] → FP16
5) PV:      O_acc^{FP32} += P·V（FP16 Cube）
```

**关键代数（comp 补偿项勿丢）**：`QK_sᵀ = QKᵀ − (Q·μ)·1ᵀ`，故
`S_true = S_int·dScale + dot(Q[r,:],μ)·(sm/sQ)`。μ 以 fp32 原值存 GM（×8 冗余），comp 在 AIV 侧
fp32 向量 dot（整数 dot ≤2M 在 fp32 精确）。[^sage-recipe]

## 完整架构规格全文（fa-sageattention-architecture）
> 以下为架构规格源文件全量正文：硬件前提、GM 数据布局契约、Kernel A 量化步骤、Kernel B 组件选型（K_static=256 垫槽）、UB/L1 布局（到字节）、CrossCore flag 表、4 相位主循环与游标语义、SM/PV/RS/QK 相位、SageMergeKernel、host tiling、SageDivScr helper、性能现状与新版 catlass 适配注记。

> 本文档含复刻该算子所需的全部精确细节：公式、内存布局（到字节）、
> 同步协议（到 flag id）、循环结构（到游标语义）。

### 0. 硬件前提

| 资源 | 容量 | 备注 |
|------|------|------|
| AIC L0A/L0B | 各 64KB | 两 block 槽位从 0 布局, 共享事件 id, **槽距必须相等** |
| AIC L0C | 256KB | 由构造函数 cursor 顺序分区 |
| L1 | 512KB | pL1(P staging) + QK L1A/L1B + PV L1B |
| AIV UB | 248KB/核 | 每 AIV 子核管 64 行, 各自独立布局 |
| 平台探测 | aicNum≈28, aivNum≈56 (PlatformAscendCManager) | 数量随时可能变, 需打印 |

```
__mix__(1, 2) 组成: 1 AIC + 2 AIV 子核; 每 AIV 子核管 64 行 (SH=64)
AIC 的 blockIdx = GetBlockIdx(); AIV 的 blockIdx = GetBlockIdx() >> 1
AIV 子核区分: GetSubBlockIdx() ∈ {0,1} (仅 mix kernel 有效!)
```

### 1. 数据布局契约（GM）

| 张量 | 布局 | 说明 |
|------|------|------|
| Q̂ (qhat) | BSND [B·sqPad, H·D] int8 | 量化后 Q, pad 行零 |
| K̂ (khat) | BSND [B·kvBlocks·128, Hkv·D] int8 | 平滑后量化 K |
| μ (muOut) | [B·Hkv·kvBlocks, D·8] fp32 | **fp32 原值 ×8 冗余** (写 4KB 可靠, 读 512B 即可) |
| SQ/SK scale | [blocks, 8] fp32 | ×8 冗余; **FA 读时下标必须 ×8** |
| V | BSND [B·skPad, Hkv·D] fp16 | host 从 HND 转 |
| O | HND [B,H,Sq,D] fp16 | kernel 直写 |
| po (splitK partial O) | [totalWork][128][128] fp32 | 未除 l 的 oacc, 行距 128 有效列 D |
| pml (partial m/l) | [totalWork][512] fp32 | {sub0: m[64]\|l[64]\|pad, sub1: 同} 每 sub 1KB |

### 2. Kernel A：SageQuantKernel（`__global__ __vector__` AIV-only）

工作单元 = K 块 (b,kvh,j) 共 B·Hkv·kvBlocks 个，或 Q 块 (b,h,i) 共 B·H·sqBlocks 个；
`for (u = GetBlockIdx(); u < total; u += GetBlockNum())` 分摊。

#### K 单元
```
1) GM fp16 → UB half (DataCopyPad 2D, QKB×D) → Cast fp32 (两级: half→fp32 用 count-Cast)
2) μ = 列均值: 树归约 (Add 折半, ua/ub 乒乓, n 从 (QKB/2)·D 折到 D)
   → Muls(um, src, 1/nValid)   # nValid = min(128, sk−j·128)
   # 注意: 树归约会破坏 K 原值, 需从 uh 重新 Cast 恢复
3) smax = max(amax|K−μ| (寄存器行 max + 乒乓), amax|μ|)
4) sK = SageDivScr(127, smax)  # device 标量除法 helper, 见 §6
5) 量化: (K−μ)·sK → Cast RINT → fp16 → int8 (count-Cast 两级)
6) 写 GM: 128 行 2D strided; μ 复制 8 份 (Muls×8) 写 muOut 4KB
```

#### Q 单元
同上去掉 μ/sK 路径；sQ = 127/amax(|Q|)。

### 3. Kernel B：SageFaKernel（`__mix__(1, 2)`）

#### 3.1 catlass 组件选型（精确 Shape）

```cpp
using ArchTag = Arch::Ascend950;
// QK: FAIQK 读 Shape=(M, N, K) —— 注意与 FAIPV 相反!
using L1TSQK = tla::Shape<Int<128>, Int<128>, Int<256>>;  // K=256 是垫的!
//   A 槽 = M·K_static·1 = 32KB, B 槽 = K_static·N·1 = 32KB
using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, int8_t, layout::RowMajor,
    int8_t, layout::ColumnMajor, int32_t, layout::RowMajor, void,
    Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;   // SPLIT_M: fixpipe 双 AIV 各得半行
using BlockMmadQK = Gemm::Block::BlockMmadTla<Gemm::MmadFAIQK<ArchTag,false>,
    L1TSQK, L1TSQK, int8_t, int8_t, int32_t, void, TileCopyQK, TileMmadQK>;

// PV: FAIPV 读 Shape=(M, K, N)
using L1TSPV = tla::Shape<Int<128>, Int<128>, Int<128>>;
//   A 槽 = M·K·2 = 32KB ✓ 与 QK 对齐 (不变量 I1)
using TileCopyPV = ... <half, layout::zN, half, layout::RowMajor, float, RowMajor, SPLIT_M>;
using BlockMmadPV = ... MmadFAIPV<ArchTag,false>, L1TSPV, half, half, float ...>;

// L0C 分区 (cursor 顺序): QK [0,128KB) (64KB×2 stage), PV [128,256KB) — 恰好占满
```

**为什么 K_static=256**：L0A/L0B 槽距 = 静态 Shape 算出的字节数。QK 是 int8（128×128=16KB），
PV 是 half（128×128×2=32KB）。槽距不等 + 共享事件 id {0..3} → 事件计数错配 → 竞态。
垫 K=256 使 QK A 槽 = 128×256×1 = 32KB = PV 的 32KB。运行时 blockK=d=128 不受影响。

#### 3.2 UB 布局（每 AIV 子核, 字节）

```
UB_S_OFF    = 0                    S int32 [64,128] ×2 slot (m2)     2×32KB
UB_O_OFF    = +64KB                O fp32  [64,128] ×2 slot (m2)     2×32KB
UB_OACC_OFF = +128KB               oacc fp32 [64,128]                 32KB
UB_O16_OFF  = +160KB               O fp16 staging                     16KB
UB_P_OFF    = +176KB               P fp16 [64,128] ×2 slot (m2)      2×16KB
UB_QF_OFF   = +208KB               Q̂ 切片: int8 8KB + half 8KB + fp32 16KB (chunk 式)
UB_MU_OFF   = +232KB               μ fp32 [128]
UB_ST_OFF   = +233KB               状态: 12 槽×[64] fp32 (lastMax/sumL/alpha 各 4 槽)
UB_SCR_OFF  = +236KB               nowMax + expSum 各 [64]
UB_DIV_OFF  = ...                  SageDivScr scratch 3×8 fp32
UB_COMP/THR/CIDX = ...             comp[64] / mask 阈值[64] / 列索引[128]
合计 ≈ 240KB ≤ 248KB
```

#### 3.3 L1 布局

```
pL1[m3] ×4 slot: [128,128] half P staging (zN)    4×32KB = 128KB
QK L1A: 128×256 int8 ×2 stage                     64KB   (cursor: 128→192KB)
QK L1B: 128×256 int8 ×2 stage                     64KB   (192→256KB... 实际 N×K 布局)
PV L1B: [128,128] half ×2 stage                   64KB   (256→320KB... )
```

#### 3.4 CrossCore flag 表（mode 4）

```
C1_V1    [2] = {0, 1}     AIC QK done → AIV SM 开始
V1_C2    [4] = {2,3,4,11} AIV P 已进 L1 → AIC PV 开始 (m3 = step%4)
C2_V2    [2] = {5, 6}     AIC PV done → AIV RS 开始 (m2 = step%2)
MM2      [2] = {7, 8}     AIV 消费完 oUb → AIC 下次 PV fixpipe 可写
MM1      [2] = {9, 10}    AIV 消费完 sUb → AIC 下次 QK fixpipe 可写
PV_CONSUMED[4] = {12..15} AIC PV 用完 pL1 → AIV SM 可覆盖 (pL1 槽复用保护)
```
调用规则（I6）：AIC set `id` 和 `16+id`；AIV set 单 `id`；AIC wait `id` 和 `16+id`。
kernel Init 时 AIV 预置 set MM1/MM2 各 id（bootstrap 首次使用）。

#### 3.5 主循环（4 相位流水 + split-KV）

```
for (it = 0; it < P + 3; ++it) {
  AIC: QK(it)      it < P && cursor.Next(curQK)
       PV(it-2)    it≥2 && it-2 < P && cursor.Next(curPV)
  AIV: SM(it-1)    it≥1 && it-1 < P && cursor.Next(curSM)
       RS(it-3)    it≥3 && it-3 < P && cursor.Next(curRS)
}
P = 本核所有 work 的步数和; 每 work = (task, shard)
```

**游标（精确语义, 曾出 bug）**：
```cpp
struct SageCursor {
  int64_t k = 0, j = -1;   // 内部: j=-1 表示新 work 待重算起始
  int64_t jCur, jStart, gCur, jEnd, bh, iq, kvhG;
};
// work g → task = g/SK, shard = g%SK
// jEndF = SageJEnd(task)  (causal 时 = (iq·128+127+sk-sq)/128+1, 否则 kvBlocks)
// jStart = jEndF·shard/SK;  jEnd = jEndF·(shard+1)/SK   (floor 均分, 差≤1 无空片*)
// 每次 Next: jCur = j (本步); j++; j≥jEnd → k++, j=-1
// ★ 调用方一律用 c.jCur (返回值是前进后的内部 j!)
// ★ isFirst 判据 = (jCur == jStart)  不是 ==0  (split-K 时 shard 起步非 0!)
```

**SM 相位（AIV, 每 step）**：
```
wait C1_V1[m2]
qk8: Q̂ 本子核切片 int8 GM→UB (2D strided, srcStride=(H-1)·D)
μ_j: GM→UB 512B (下标 (kvhG·kvBlocks+jCur)·D·8)
cScale = sm/sQ (每 work 缓存, jCur==jStart 时重算)
chunk×2: Cast int8→half→fp32 (32行/块) → SageVfCompDot:
    comp[r] = ReduceSum(qff[r,:] ⊙ μ)·cScale    (fp32, 整数 dot ≤2M 精确)
needMask 时: thrUb.SetValue(r, lim-1) 逐行阈值 (causal: lim=col_row+1 上界)
SageVfDeqMaskMax: S = S_int·dScale + comp + mask; 行 max → nowMax
!isFirst: UpdMax(nowMax, lastMax[(m3+3)%4])     ← I5 链读槽
SageVfExpP: P = exp(S−nowMax), 行和 → expSum
WaitFlag<MTE3_V>(m3Prev)  ← 上次 CopyUb2L1 读 pUb 完成保护
Cast P fp32→fp16 (count-Cast)
!isFirst: UpdL(nm, lmPrev→lmW 本槽, alpha[m3], sumPrev→sum[m3])   ← I5 分槽
else:      CopyF(lastMax[m3], nowMax); CopyF(sumL[m3], expSum)
WaitFlag PV_CONSUMED[m3]  ← pL1 覆盖保护 (I8)
CopyUb2L1Tla: pUb[m2][64,128] fp16 RowMajor → pL1[m3] zN (本子核 64 行片)
set MM1[m2]; set V1_C2[m3]
```

**PV 相位（AIC）**：
```
wait V1_C2[m3] (id 和 16+id)
mmadPV(pL1[m3] zN [128,128], V tile [128,D], oUb[m2], shape=(128,D,128), taskId=m2)
     单次调用 n=D (FAIPV 内部 nLoop 处理 D=64 尾块)
set C2_V2[m2] (id 和 16+id); set PV_CONSUMED[m3]
```

**RS 相位（AIV, 每 step）**：
```
wait C2_V2[m2]
SageVfRescale: oacc = oacc·alpha[m3] + oUb[m2]   (行距 = D, 与 PV fixpipe 紧凑布局一致!)
isLast:
  splitK>1: 写 partial — po[gCur][row0..][0..D) fp32 2D strided (dstStride=(128-D)·4)
            + pml[gCur][sub·256] = {m=lastMax[m3], l=sumL[m3]} 各64 fp32 + pad → 1KB
            + PipeBarrier<PIPE_ALL>()  ← kernel 退出不保证 MTE3 排空!
  else:     DivL(oacc /= l) → Cast fp16 → GM 1D (validRows·D·2)
set MM2[m2]   ← 只 set 本步槽! (I3)
```

#### 3.6 QK 相位（AIC）

```
tensorQ = GetTile(qhat, (b·sqPad + iq·128, h·D), (128, D))
tensorK = GetTile(khat 列主视图, (kvh·D, (kvhG/hkv)·kvBlocks·128 + jCur·128), (D, 128))
mmadQK(tensorQ, tensorK, sUb[m2], ..., taskId=m2,
      qkFirst=(jCur==jStart), qkLast=(jCur==jEnd-1))
  // qkFirst: GM→L1A 装 Q̂ (borrow: 任务内只装一次)
  // qkLast:  轮转 L1A stage
set C1_V1[m2] (id 和 16+id)
```

### 4. SageMergeKernel（`__global__ __vector__` AIV-only, split-KV 归并）

```
映射 (★ __vector__ 无 sub-block 概念, GetSubBlockIdx 恒 0):
  sub = GetBlockIdx() & 1;  task t = GetBlockIdx() >> 1;  步长 = GetBlockNum() >> 1

per task:
  1) 一次性整块载入本 task 全部分片 m/l (works 连续: pml[t·SK·512, +SK·512))
     → PipeBarrier (★勿用连续小拷贝+复用 event id, 实测不可靠)
  2) 紧凑化: 有效分片 (jStart<jEnd) 前移到前 nValid 槽
     (★ nValid==k 时只计数不拷贝, 连 continue 会跳过计数 → 首分片被覆盖)
  3) m* = max_k m_k; w_k = exp(m_k−m*) 原位; l* = Σ w_k·l_k
  4) oacc = Σ w_k·po_k[r,:]   (★ acc += w·po 用 Mul+Add; MulDstAdd 是 a·dst+b!)
  5) O = oacc/l* → fp16 → GM
```

### 5. Host（.asc 单 TU）

```
tiling: b/h/hkv/sq/sk/sqPad/skPad/d/sqBlocks/kvBlocks/totalTasks/
        splitK/totalWork/coreNum/groupSize/smScale/isCausal/quantScheme
splitK = ceil(aicNum / totalTasks), cap 到 kvBlocks, ≥1   (ceil 避免空闲核)
coreNum = min(aicNum, totalWork)
launch: SageQuantKernel<<<aivNum>>> → SageFaKernel<<<coreNum>>> → (splitK>1) SageMergeKernel<<<aivNum>>>
缓冲: qhat/khat/mu/sq/sk/o + po[totalWork·128·128·4] + pml[totalWork·512·4]
★ kernel 内 td 是逐字段从 GM 拷贝的 — 新增 tiling 字段必须加进拷贝列表!
```

### 6. 关键 helper

```cpp
// device 标量除法 (ccec 无法 lower 标量 fp div)
SageDivScr(a, b, scrA, scrB, scrDst):
  Duplicate(scrA, a, 8); Duplicate(scrB, b==0?1:b, 8);
  Div(scrDst, scrA, scrB, 8);          // count≥8, 禁原地
  PipeBarrier<PIPE_ALL>();             // Div(向量)→GetValue(标量) 需同步
  return scrDst.GetValue(0);
```

### 7. 性能现状与优化路线（已实测）

| 项 | 数据 |
|----|------|
| 每流水迭代 | ~10-11us/核 (满负载), 其中 ~6 次跨核 flag 往返 × 1.5-2us 未被流水隐藏 |
| 已做 | 去 V→V PipeBarrier(+9%), Q̂ L1A borrow, cScale 缓存 |
| split-KV | k1024: 81→43.5us（标杆 FA 25.2）— 短序列 ratio 0.67 |
| 路线① | KV tile 加宽到 256: QK 单 mmad 两块, 迭代数减半; L0C 需 QK 单缓冲 128K+PV 128K=256K 恰好放下 |
| 路线② | 加深 sUb/oUb 槽 (2→4) 让 MM1/MM2 等待预先满足 (UB +160KB 需先砍 QF 区) |
| 路线③ | S fixpipe 走 DEQF16 (int32→fp16 带 scale, catlass 支持) 省 32KB/步流量 |

### 8. 新版 catlass 适配注记（2026-08 重建上板，详见 fa-sageattention-troubleshooting.md §F）

本规格描述的是**算法/数据流设计**（数学公式、内存布局、flag 协议、不变量 I1-I8），与 catlass 版本无关。但**新版 catlass 的 `BlockMmadTla` 实现行为变了**，重建实现时须按以下适配（原旧版 catlass 的对应行为不同）：

- **MM1/MM2 wait 改由 catlass 内部完成**：新版 `BlockMmadTla`（FAIQK/FAIPV）内部已 `CrossCoreWaitFlag(MM1_RES_INTRA_EVENT[taskId])`/`MM2_RES_INTRA_EVENT[taskId]`（flag ID 与本算子 §3.4 的 MM1={9,10}/MM2={7,8} 全同）。故 §3.5 PV 相位、§3.6 QK 相位中**不再有 kernel 手动的 MM1/MM2 wait**——AIC 侧只 set C1_V1/C2_V2/PV_CONSUMED，AIV 侧 set MM1/MM2（catlass 等）。双重 wait 会死锁（pitfalls P31）。
- **PV_CONSUMED 用 step 守卫非 bootstrap**：循环前在 `PIPE_FIX` 上预置的 set 不传播（冷管道），改 §3.5 的 `if (it > SAGE_TASK_NUM3)` 守卫（首次填充跳过、复用才 wait），单 buffer P-scale 与 4-slot pL1 时序天然错开（pitfalls P32）。
- **merge 紧凑化 copy 512**：§4 SageMergeKernel 紧凑化 copy 整个 512-float slot（sub0+sub1 m|l），非 128（pitfalls P33）。
- **bf16 输入 host 转 fp16**：host 读入后 bf16→fp16(RNE) 再喂 kernel（pitfalls P34）。
- **构建**：CMakeLists 设 `CATLASS_ARCH=3510`；`aclrtMalloc`→`uint8_t*`+cast 喂 `GM_ADDR`（pitfalls P29/P30）。

> 验收：重建版 22/22 PASS（cos 0.9996~0.9999），见 fa-sageattention-recipe.md §6。FP8 PV（quantScheme=1）集成见 pitfalls P36（撞 catlass FP8 MX API，须先探针实测组件真实数据流，暂缓）。

# 用法

## 分核策略与基本块切分

硬件前提（dav-3510）：L0A/L0B 各 64KB（两 block 槽位从 0 布局、共享事件 id，**槽距必须相等**）；
L0C 256KB；L1 512KB（pL1 P staging + QK/PV L1A/L1B）；AIV UB 248KB/核；平台探测 aic≈28/aiv≈56
（数量随时可能变，需打印）。`__mix__(1,2)` = 1 AIC + 2 AIV 子核，每 AIV 子核管 64 行（SH=64）；
AIC `blockIdx=GetBlockIdx()`，AIV `blockIdx=GetBlockIdx()>>1`，子核 `GetSubBlockIdx()∈{0,1}`
（仅 mix kernel 有效）。[^sage-architecture]

三 kernel 分工：[^sage-architecture]

- **Kernel A `SageQuantKernel`**（`__global__ __vector__` AIV-only）：工作单元 = K 块 `(b,kvh,j)` 或
  Q 块；`for (u=GetBlockIdx(); u<total; u+=GetBlockNum())` 分摊。K 单元：GM fp16→UB→Cast fp32→
  树归约列均值（Add 折半、乒乓，注意树归约破坏 K 原值需重新 Cast 恢复）→ smax → `SageDivScr` 标量
  除法 → `(K−μ)·sK → RINT → int8` 两级 count-Cast → 写 GM（128 行 2D strided）；μ 复制 8 份写 4KB。
- **Kernel B `SageFaKernel`**（`__mix__(1,2)`）：4 相位流水 + split-KV，
  `for (it=0; it<P+3; ++it)`：AIC `QK(it)`+`PV(it−2)`；AIV `SM(it−1)`+`RS(it−3)`；`P`=本核所有
  work（task, shard）步数和。
- **SageMergeKernel**（AIV-only）：split-KV 归并（按 partial m/l 归并 po）。

开发工作流七步：环境先验（CANN 9.x + dav-3510，全部以本知识为准、不查外部样例）→ 工程搭建
（.asc 单 TU；CMakeLists 把头文件加进 add_executable sources 否则头改动不触发重编；必须设
`CATLASS_ARCH=3510`）→ Kernel A 用 `SAGE_SKIP_FA=1` + numpy 全量比对 → Kernel B 分相位门控冒烟
（QK only → +SM → +PV → 全开）→ 精度取证（numpy 仿真对拍逐层定位）→ 功能全覆盖 → 性能与封装。[^sage-recipe]

## 数据路径与存储层级

FA 读 GM 下标规则：SQ/SK/mu 的 GM 元素下标必须 ×8（mu 是 ·D·8）；违反则 j>0 块全读块 0 的 scale。[^sage-rules]

L1 布局：pL1（P staging，4-slot）+ QK L1A/L1B + PV L1B；L0C 由构造函数 cursor 顺序分区。[^sage-architecture]

## 实现顺序

1. 先背 I1-I8 不变量与 Never/Always 纪律再写代码。
2. Kernel A 独立验证（qhat/khat/μ/scale 全量数值 vs numpy）。
3. Kernel B 分相位门控，每步安全运行器验证。
4. 精度取证逐层定位（S_int → comp → 状态链 → O），cos ≥ 0.99 再扩覆盖。

# 代码模式

**CrossCore mode4 flag 表**（Kernel B）：[^sage-architecture]

```text
C1_V1     [2]={0,1}      AIC QK done → AIV SM 开始
V1_C2     [4]={2,3,4,11} AIV P 已进 L1 → AIC PV 开始（m3 = step%4）
C2_V2     [2]={5,6}      AIC PV done → AIV RS 开始（m2 = step%2）
MM2       [2]={7,8}      AIV 消费完 oUb → AIC 下次 PV fixpipe 可写
MM1       [2]={9,10}     AIV 消费完 sUb → AIC 下次 QK fixpipe 可写
PV_CONSUMED[4]={12..15}  AIC PV 用完 pL1 → AIV SM 可覆盖（pL1 槽复用保护）
```

调用规则（I6）：AIC set `id` 和 `16+id`（双 AIV 各一份）；AIV set 单 `id`（硬件按 sub-core 路由）；
AIC wait `id` 和 `16+id`。[^sage-rules]

**新版 catlass 适配（重建必读，6 处）**：[^sage-recipe]

1. CMakeLists 设 `CATLASS_ARCH=3510`（未设则 catlass 3510 头被 `#if` 跳过，报 ScaleGranularity 未声明）。
2. `aclrtMalloc` 的 `void*` 须经 `uint8_t*` + `(void**)&` cast 喂 `GM_ADDR`。
3. **★双重 wait 死锁**：新版 `BlockMmadTla`（FAIQK/FAIPV）内部已 wait MM1/MM2（flag ID 与上表全同
   {9,10}/{7,8}）→ kernel 侧勿再手动 wait MM1/MM2。
4. PV_CONSUMED bootstrap 勿用循环前 `PIPE_FIX` 预置（冷 set 不传播，首 SM 永卡）；改 step 守卫。
5. merge 紧凑化 copy 整个 512-float slot（sub0+sub1 的 m|l）；只 copy 128 漏 sub1 → inf。
6. bf16 输入 host 先 bf16→fp16(RNE) 再喂 kernel。

## 开发流程与逐步骤配方全文（fa-sageattention-recipe）
> 以下为开发配方源文件全量正文：算子定义与量化路径、七步开发工作流、不变量要点、CrossCore mode4 规划、排障路由、验证与验收标准、新版 catlass 适配六处与参考文档导航。

> **定位**：Ascend950（dav-3510）量化注意力 FA-SageAttention（K 平滑 + INT8 QK + online softmax + FP16 PV）
> 的完整开发配方：数学语义、双 kernel 架构、CrossCore mode4 协议、精度取证与排障，全部在本文件与四个
> 同目录参考（architecture / rules /
> troubleshooting / precision）内自包含。
> 与 FA 族其它算子配方同层（覆盖路由见 fa-kernel-handcraft.md §12.14-C）。



> 本 skill 是在 Ascend950PR 上从零开发 FA-SageAttention 算子的完整方法论，
> 由一次完整开发实践（约 20+ 个 bug 的全量取证与修复、22 case 功能全 PASS、
> 精度 cos 0.9996~0.9999、序列长至 200K）沉淀而来。v1（INT8 QK + FP16 PV）已达标。
> **全部开发经验固化在 四个 fa-sageattention-* 参考文件**（架构 / 8 不变量 / 坑全集 / 精度取证），自包含。

### 第 1 步 —— 算子定义与量化路径

```
输入: Q ∈ R^{B×H×S_q×D}, K ∈ R^{B×H_kv×S_kv×D}, V ∈ R^{B×H_kv×S_kv×D}  (fp16/bf16, HND)
输出: O = softmax(QKᵀ·sm_scale + causal_mask)·V ∈ R^{B×H×S_q×D}  (fp16)

SageAttention 量化路径 (v1, quantScheme=0):
  1) K 平滑:    每 128 行 KV block, μ = mean(K[blk], axis=row) ∈ R^D
                K_s = K − 1·μᵀ;  smax = max(amax|K_s|, amax|μ|);  sK = 127/smax
                K̂ = round(K_s·sK) → INT8
  2) Q 量化:    sQ = 127/amax(|Q[blk]|);  Q̂ = round(Q·sQ) → INT8
  3) QK^T:      S_int = (Q̂·K̂ᵀ)_{INT8×INT8→INT32}
                S = S_int·(sm/(sQ·sK_j)) + comp_j + mask
                其中 comp_j[r] = dot(Q̂[r,:], μ_j)·(sm/sQ)   ← K 平滑补偿, sK 代数消去
  4) softmax:   online (行 max 减法 + exp + 缩放), P ∈ [0,1] → FP16
  5) PV:        O_acc^{FP32} += P·V  (FP16 Cube)
```

**关键代数**（comp 补偿项，勿丢）：`QK_sᵀ = QKᵀ − (Q·μ)·1ᵀ`，故 `S_true = S_int·dScale + dot(Q[r,:],μ)·(sm/sQ)`。
μ 以 fp32 原值存 GM（×8 冗余），comp 在 AIV 侧 fp32 向量 dot（整数 dot ≤2M 在 fp32 精确），与旧"n=144 GEMM 增广列"逐位等价。完整公式/内存布局见 `fa-sageattention-architecture.md`。

### 第 2 步 —— 开发工作流（七步）

```
Step 1  环境与先验: CANN 9.x + dav-3510; 克隆 catlass(gitcode.com/cann/catlass);
        读 catlass/docs/（库定位与组件约束）; CrossCore flag 协议/组件选型/内存布局
        全部以本 skill references/ 为准（自包含，不查任何外部样例/算子源码）
Step 2  工程搭建: .asc 单 TU (量化kernel + FA kernel + host main);
        CMakeLists 把头文件加进 add_executable sources (否则头改动不触发重编!)
        ★ 新版 catlass: 必须设 CATLASS_ARCH=3510 (见 troubleshooting §F P29)
Step 3  Kernel A (量化, AIV-only): architecture.md §2
        → SAGE_SKIP_FA=1 + numpy 比对 qhat/khat/μ/scale 全量数值验证
Step 4  Kernel B (主算子, __mix__(1,2)): architecture.md §3
        → 分相位门控冒烟 (QK only → +SM → +PV → 全开), 每步 srun.sh 验证
Step 5  精度取证: precision-verification.md 的 numpy 仿真对拍,
        逐层定位 (S_int → comp → 状态链 → O), 直到 cos ≥ 0.99
Step 6  功能全覆盖: causal/GQA/MQA/D=64/B>1/bf16/尾块/长序列(200K)
Step 7  性能与封装: split-KV → KV tile 加宽 → (ACLNN 封装 Owner 已跳过)
```

### 第 3 步 —— 写代码前先背 8 条不变量 + 运行纪律

违反任意一条不变量 = 挂死或错数。完整 I1-I8 + Never/Always 见 `fa-sageattention-rules.md`。要点：
- **Always**：用 `srun.sh`（rc+新鲜度检查）；同卡挂 2 次换卡；`--bench` 测时；新增 tiling 字段同步加拷贝列表。
- **Never**：直跑二进制后解读输出；timeout 后解读任何文件；AIC 用 `PipeBarrier<PIPE_ALL>`；**新版 catlass 下 kernel 手动 wait MM1/MM2**（catlass 内部已 wait，双重 wait 死锁）；**PV_CONSUMED 用循环前 PIPE_FIX 预置**（冷 set 不传播）。

### 第 4 步 —— 规划 cube/vec PIPE sync（CrossCore mode4）

AIC 与 AIV 子核间的 4 相位流水（QK→SM→PV→RS）用 CrossCore mode4 flag 同步。完整 flag 表（id 分配、set/wait 双份语义、PV_CONSUMED handshake）见 `fa-sageattention-architecture.md` §3.4。要点：
- AIC set `id` 和 `16+id`（双 AIV 各一份）；AIV set 单 id（硬件按 sub-core 路由）；AIC wait `id` 和 `16+id`。
- pL1 槽复用：AIC PV 后 set PV_CONSUMED(id 12-15)，AIV SM 覆盖 pL1[m3] 前 wait（step 守卫：首次填充跳过、复用才 wait，勿用循环前预置）。

### 第 5 步 —— 排障路由

| 症状 | 去处（`fa-sageattention-troubleshooting.md`）|
|------|------|
| 单块对、多块错/挂 | §L0 槽位类 (P2/P3) + 状态链 (I4/I5) |
| 挂死无输出 | precision-verification.md §挂死取证 (SAGE_WATCH 分相位标记) |
| 精度随步数劣化 (0.9999→0.99→0.88) | I2(×8索引) → I5(状态链) → I3(MM2 surplus) 顺序排查 |
| 特定行段损坏 (rows 32-63) | AIV 子核半槽竞态 / MulDstAdd 误用 / merge sub 映射 (P33) |
| case0 splitK sub1 全 inf | merge 紧凑化只 copy sub0 (P33) → copy 512 全 slot |
| D=64 崩溃 507015 | RS 相位行距硬编码 128, 须 D-aware |
| bf16 cos≈0.28 | host 未转 bf16→fp16 (P34) |
| 编译错 ScaleGranularity 未声明 | CMakeLists 没设 CATLASS_ARCH=3510 (P29) |
| kernel 挂死、看门狗 ~27s | AIC 重复 wait MM1/MM2 (P31) 或 PV_CONSUMED 冷预置 (P32) |
| 单次 srun 7~27s 疑卡顿 | 实为每进程 setDevice/Reset 开销 (P35)，用 --bench |

### 第 6 步 —— 验证与验收

精度取证方法论（numpy 精确仿真作真值 + kernel 中间量 GM dump + 假设拟合定位）见 `fa-sageattention-precision.md`。验收标准（v1 已达标，2026-08 重建上板复现）：
- [x] **kernel 级精度: vs FP32 golden cosine ≥ 0.99** — 22/22 case PASS，cos 0.999592~0.999909
- [x] **覆盖: causal/GQA(8:1,4:1,2:1)/MQA/D=64/B>1/bf16/尾块/长序列 32K~200K 不爆内存**
- [x] **性能基线**（`--bench`，28 AIC+56 AIV）：128²≈31μs / 1k²≈73μs / 4k causal≈505μs / 200k≈719ms
- [ ] quantScheme=1 (FP8 PV) — 撞 catlass FP8 MX API 墙（P36），须先用探针 kernel 实测 catlass FP8 MX 组件真实数据流再复刻，非刚需、暂缓
- [ ] 规格补全: lse / varlen(`sageattn_varlen`) / NHD 布局 / D=96（真功能缺口，优先于 FP8）

### 第 7 步 —— 新版 catlass 适配（重建必读）

原开发用旧版 catlass；重建环境装的是 gitcode 最新 catlass，`BlockMmadTla` 行为变了。重建→22/22 PASS 修了 6 处（详见 `fa-sageattention-troubleshooting.md` §F P29-P35）：
1. **构建胶水**：CMakeLists 设 `CATLASS_ARCH=3510`；`aclrtMalloc` 的 `void*` 须经 `uint8_t*` + `(void**)&` cast 喂 `GM_ADDR`。
2. **★双重 wait 死锁**：新版 `BlockMmadTla` 内部已 `CrossCoreWaitFlag(MM1_RES/MM2_RES_INTRA_EVENT[taskId])`（FAIQK:223 / FAIPV:197，ID 与本算子全同 {9,10}/{7,8}）→ kernel 侧勿再手动 wait MM1/MM2。
3. **PV_CONSUMED bootstrap**：勿用循环前 `PIPE_FIX` 预置；改 step 守卫。
4. **merge 紧凑化**：copy 512 全 slot（sub0+sub1 的 m|l），只 copy 128 漏 sub1 → inf。
5. **bf16 输入**：host 须先 bf16→fp16(RNE) 再喂 kernel。
6. **计时纪律**：单次 srun 的 7~27s 是每进程 `aclrtSetDevice`+`Reset` 开销，非 kernel 卡顿；用 `--bench`。

### References

| 文档 | 何时读 |
|------|--------|
| `fa-sageattention-architecture.md` | 完整架构：数学公式、双 kernel、内存布局（UB/L1/L0 精确字节）、flag 表、8 条不变量、新版 catlass 适配注记 §8 |
| `fa-sageattention-rules.md` | 写代码前：8 条不变量 I1-I8 + 运行纪律 Never/Always |
| `fa-sageattention-troubleshooting.md` | 排障/重建时：平台坑全集 A~F 类（含新版 catlass 适配坑 P29-P36）：症状→根因→解法 |
| `fa-sageattention-precision.md` | 精度取证：numpy 精确仿真、GM dump 布局、假设拟合定位、挂死取证 |

## 流水排布、同步关系与数值精度

- **8 条不变量（违反即挂死/错数）**：[^sage-rules]

| # | 不变量 | 违反后果 |
|---|--------|---------|
| I1 | FAIQK 读 Shape=(M,N,K)、FAIPV 读 (M,K,N)；两 block 的 L0A/L0B 槽都从 0 布局且共享事件 id {0..3}，槽距必须完全相等（都=32KB：QK 垫 K_static=256） | P≥2 竞态挂死/错数 |
| I2 | SQ/SK/mu 的 GM 元素下标 ×8 | j>0 块全读块 0 的 scale |
| I3 | MM1/MM2 释放只 set 本步槽；双 set surplus → 下下个 PV 提前通过覆盖还在被读的 oUb | O 的 rows 32-63 损坏 |
| I4 | 游标 jCur：SageCursorNext 返回前已 j++，调用方一律用 c.jCur | 步序错乱 |
| I5 | 状态链分槽：step s 读 lastMax/sumL 的 (s−1)%N 槽、写 s%N 槽 | 多步精度劣化 |
| I6 | CrossCore mode4 双份 set 语义（见上） | 挂死或单 AIV 数据缺失 |
| I7 | AIC 上禁 `PipeBarrier<PIPE_ALL>`（AIC 管集合无 V/MTE3） | 必挂死 |
| I8 | pL1 槽复用 handshake：AIC PV 后 set PV_CONSUMED，AIV SM 覆盖 pL1[m3] 前 wait；P-Cast 前 `WaitFlag<MTE3_V>(m3Prev)` | P>槽数时挂死 |

- 数值精度：comp 补偿与 softmax 状态链全程 fp32；P 送 Cube 前 FP16；splitK partial 按 fp32 m/l 归并。

# 约束

运行纪律（血泪教训）：[^sage-rules]

- **Always**：一律用安全运行器（内置 rc + o.bin 新鲜度检查，防 stale 欺骗）；同卡挂 2 次立即换卡
  （npu-smi Health 可能仍显示 OK 但核已污染）；`--bench` 测真实 kernel 用时（单次运行 7~27s 是每进程
  setDevice/Reset 开销，非 kernel 卡顿）；新增 tiling 字段必须同步加进 kernel 内逐字段拷贝列表
  （漏 → 设备读栈垃圾 → P=0 → 输出全零）；改被 include 的头后 touch 引用它的 .asc TU 再 make
  （CMake 不跟踪 include 依赖）。
- **Never**：绝不直接跑二进制后解读输出（kernel 挂死被 timeout 杀后 o.bin 不更新，读到旧文件假结果）；
  timeout 后绝不能解读任何输出文件；AIC 绝不用 `PipeBarrier<PIPE_ALL>`；新版 catlass 下 kernel 勿
  手动 wait MM1/MM2；PV_CONSUMED 勿用循环前 PIPE_FIX 预置。

规格边界（v1 已达标项之外的缺口）：quantScheme=1（FP8 PV）撞 catlass FP8 MX API 墙，须先用探针
kernel 实测组件真实数据流再复刻，非刚需暂缓；lse/varlen/NHD 布局/D=96 为真功能缺口，优先于 FP8。[^sage-recipe]

## 硬规则全文（fa-sageattention-rules）

> 以下为硬规则源文件全量正文：8 条不变量 I1-I8 与运行纪律（Always/Never 清单），与上文归纳互为对照、以本表为准。

> 违反任意一条 = 挂死或错数。写代码前通读，写完逐条对照。

### A. 8 条不变量（I1-I8，违反即挂死/错数）

| # | 不变量 | 违反后果 |
|---|--------|---------|
| I1 | **L0A/L0B 槽距对齐**: FAIQK 读 Shape=(M,N,K)、FAIPV 读 (M,K,N); 两 block 的 L0A/L0B 槽都从 0 布局且共享事件 id {0..3}, 槽距必须完全相等(都=32KB: QK 垫 K_static=256) | P≥2 竞态挂死/错数 |
| I2 | **×8 冗余张量索引**: SQ/SK/mu 的 GM 元素下标必须 ×8 (mu 是 ·d·8) | j>0 块全读块 0 的 scale |
| I3 | **MM1/MM2 释放只 set 本步槽**: 双 set 产生 surplus → 下下个 PV 提前通过、覆盖还在被读的 oUb | O 的 rows 32-63 损坏 |
| I4 | **游标 jCur**: SageCursorNext 返回前已 j++, 调用方一律用 c.jCur | 步序 [1,2,0] 错乱 |
| I5 | **状态链分槽**: step s 读 lastMax/sumL 的 (s-1)%N 槽, 写 s%N 槽 | 多步精度劣化 |
| I6 | **CrossCore mode4**: AIC set id 和 16+id (双 AIV 各一份); AIV set 单 id (硬件按 sub-core 路由); AIC wait id 和 16+id | 挂死或单 AIV 数据缺失 |
| I7 | **AIC 上禁 `PipeBarrier<PIPE_ALL>`** (AIC 管集合无 V/MTE3) | 必挂死 |
| I8 | **pL1 槽复用 handshake**: AIC PV 后 set PV_CONSUMED(id 12-15), AIV SM 覆盖 pL1[m3] 前 wait; P-Cast 前要 `WaitFlag<MTE3_V>(m3Prev)` | P>槽数时挂死 |

### B. 运行纪律（血泪教训，Never/Always）

#### Always
- 一律用安全运行器 `SAGE_DEV=N bash scripts/srun.sh <dir>`（内置 rc + o.bin 新鲜度检查，防 stale 欺骗）
- 同卡挂 2 次立即换卡；npu-smi Health 可能仍显示 OK 但核已污染
- 用 `--bench` 测真实 kernel 用时（单次 srun 的 7~27s 是每进程 `aclrtSetDevice`/`Reset` 开销，非 kernel 卡顿）
- 新增 tiling 字段必须同步加进 kernel 内 `td` 逐字段拷贝列表（漏→设备读栈垃圾→P=0→输出全零）
- 改 .hpp/被 `#include` 的文件后，必须 touch 引用它的 .asc TU 源文件再 make（CMake 不跟踪 include 依赖，旧 exe 假验证），并确认 exe mtime > 源 mtime

#### Never
- 绝不直接跑 `./build/sage_fa` 后解读输出 —— kernel 挂死被 timeout 杀掉后 o.bin 不更新，读到的是旧文件的假结果（一次事故污染全部 8 张卡）
- timeout 后绝不能解读任何输出文件
- AIC 上绝不用 `PipeBarrier<PIPE_ALL>`（见 I7）
- 新版 catlass 下：kernel 侧勿再手动 `CrossCoreWaitFlag MM1/MM2`（catlass `BlockMmadTla` 内部已 wait，双重 wait 永久阻塞，见 troubleshooting §F P31）
- PV_CONSUMED 勿用循环前 `PIPE_FIX` 预置（冷 set 不传播，首 SM 永卡）；改 step 守卫（见 troubleshooting §F P32）

# 失败表现

排障路由（症状 → 根因方向）：[^sage-recipe]

| 症状 | 排查方向 |
|------|---------|
| 单块对、多块错/挂 | L0 槽位类（FAIQK/FAIPV Shape 语义相反）+ 状态链 I4/I5 |
| 挂死无输出 | 分相位标记取证；AIC 重复 wait MM1/MM2 或 PV_CONSUMED 冷预置 |
| 精度随步数劣化（0.9999→0.99→0.88） | 按 I2(×8 索引) → I5(状态链) → I3(MM2 surplus) 顺序 |
| 特定行段损坏（rows 32-63） | AIV 子核半槽竞态 / MulDstAdd 误用 / merge sub 映射 |
| case0 splitK sub1 全 inf | merge 紧凑化只 copy sub0 → copy 512 全 slot |
| D=64 崩溃 507015 | RS 相位行距硬编码 128，须 D-aware |
| bf16 cos≈0.28 | host 未转 bf16→fp16 |
| 编译错 ScaleGranularity 未声明 | CMakeLists 没设 CATLASS_ARCH=3510 |

平台坑补充（编译器/API 层，实测全集见溯源文件）：AIV 上标量 GM store 静默失败；count-API Muls/Adds
的 dst 寻址坑；StoreUnAlign 需左值；device 标量 fp 除法无法 lower（用 `SageDivScr` helper）；
`__vector__` kernel 无 sub-block；MulDstAdd 语义与直觉相反；小 DataCopyPad 写不可靠；`__kfc_workspace__`
路径的 stale o.bin 欺骗是最危险元坑。[^sage-troubleshooting]

## 精度/挂死取证方法论全文（fa-sageattention-precision）
> 以下为精度取证源文件全量正文：三层真值体系、numpy 精确仿真、GM dump 布局、逐层判定流程、假设拟合、挂死取证、安全运行器 srun.sh 与 22 case 回归套件结构。

> 核心思想：**numpy 精确仿真作真值 + kernel 中间量 GM dump + 假设拟合定位**。
> 该方法在本次开发中定位了全部 20 个 bug，无一例外。

### 1. 三层真值体系

```
golden  = FP32 全精度 attention (numpy, 从 q.bin/k.bin/v.bin 直接算)
emu     = numpy 精确仿真"本算子设计语义" (含量化/补偿/online-softmax/split-K)
kernel  = 设备实际输出 o.bin

判定: emu vs golden ≈ 0.9999 → 设计正确
      kernel vs emu 偏差 → kernel 实现错误 (而非算法问题)
```

### 2. numpy 精确仿真（emu.py 核心，直接可用）

```python
# 每 step j:
S = (qhat @ khat[j].T) * (sm/(sQ*sK_j)) + comp_j[:,None]
#   comp_j = (qhat @ mu[j]) * (sm/sQ)      ← sK 代数消去!
nm = S.max(-1, keepdims=True)
if j > 0: nm = maximum(nm, prev_nm)         # online merge
P = exp(S - nm); es = P.sum(-1)
if j == 0: O = P@V_j; l = es
else:      a = exp(prev_nm - nm); l = a*l + es; O = a*O + P@V_j
prev_nm = nm
result = O / l
```
量化输入 qhat/khat/mu/sK 从 GM dump 的 bin 文件读（kernel A 的产物），
逐位复现 kernel 语义。**先修好 emu 再修 kernel** —— emu 对 golden 0.9999
证明数学设计正确，之后 kernel 与 emu 的差异全是实现 bug。

### 3. GM dump 布局（SAGE_DUMP=1 → qdbg.bin 512KB）

| 偏移 | 内容 | 写入时机 |
|------|------|---------|
| [0,192K) / [192K,384K) | sub0/sub1 的 S_int（SM 入口 dump sUb[m2]）| 每 SM 步（后步覆盖）|
| +128K | P 取证（fp16，两 sub 竞写）| 每 SM 步 |
| +144K (+336K) | O 取证（RS 入口 dump oUb[m2]）| 每 RS 步 |
| [400K,416K) | 每 step 状态快照（12 槽×64 fp32，step×4K）| 每 SM 步 |
| [416K,424K) | 每 step comp 快照（step×2K）| 每 SM 步 |
| +250K | AIC pr[0..3] 进度（标量写，AIC 可靠）| 每 AIC 相位 |
| [452K,460K) | AIV SageMark 槽（alive/smDone/rsWait/rsDone）| 各相位 |
| [464K,496K) | merge 内部（载入后 m/l、算完后 w）| merge |

host 侧落盘：`qhat/khat/sk_scale/mu/o/qdbg/pml/po → <dir>/*.bin`
（mu 是 ×8 冗余布局；pml/po 仅 splitK>1 时）。

### 4. 逐层判定流程（哪层开始错 → 直接指向根因）

```
emu 对不对 (vs golden)
  └→ 状态链对不对 (stchk: lm/sum/alpha 逐 step 逐槽 vs numpy)
       └→ comp 对不对 (compchk: 逐行 vs numpy)
            └→ S_int 对不对 (vs qhat@khat.T 精确整数比较)
                 └→ O 输出对不对 (ochk: vs P_j@V_j)
```

行/列模式特征（直接指认根因类）：
- **cols 0-63 坏、64-127 好** → 第二次调用操作数覆盖在飞 mmad（L0A 槽溢出）
- **rows 32-63 坏** → AIV 子核半槽竞态 / MM2 surplus 提前覆盖 / merge sub 映射错
- **特定行散布错**（max 未更新）→ 状态链槽位
- **误差随步数 ~3×/步增长** → per-block scale 错（×8 索引）/ comp 错
- **整个 work 输出 = 线性组合但系数错** → merge 权重（拟合系数 vs 期望）

### 5. 假设拟合（当直接比较不够时）

**dScale 组合拟合**（定位 scale 用错哪块）：
```python
for ds_i in range(nb):
    S1 = E1 * (sm/(sQ*skv[ds_i*8])) + comp1
    nm = maximum(S1.max(-1), m0)
    print(f'ds_{ds_i}: lm match={allclose(lmk, nm)}')  # 全部用块 i 的 scale?
```

**逐行系数拟合**（定位 merge/softmax 权重错）：
```python
A = stack([po[k][r] for k in range(SK)], 1)   # [D, SK]
c = lstsq(A, o[r])                             # 实际用的系数
expected = w_[:,r] / lstar[r]                  # 期望系数
# 系数交换 → 输入序错; 比率错 → m* 错; 全零 → 该行未写
```

### 6. 挂死取证

1. **分相位标记**（AIC 用标量 GM 写 pr[]，AIV 用 SageMark=Duplicate+DataCopyPad）：
   alive/smDone/rsWait/rsDone + it 号 → 看最后停在哪相位哪迭代
2. **AIC 与 AIV 分别看**：AIC 到 it=P+2 而 AIV 卡中段 → flag 等待分析
3. **flag 计数平衡表**：按 I6 语义逐对核算 set/wait（本会话靠此排除了一整类嫌疑）
4. **单发 vs 双发**（NOWARMUP）：单发对双发错 → 跨 launch flag 残留
   （尾部 set 无人消费时 surplus 留到下一 launch）
5. **dump 探针改变时序**——dump 模式与 no-dump 的挂点可能不同，结论必须交叉验证

### 7. 安全运行器（srun.sh，防 stale 欺骗）

```bash
#!/bin/bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
DEV=${SAGE_DEV:?必须显式 SAGE_DEV=N 指定卡}
DIR=$1; shift || true
export ASCEND_RT_VISIBLE_DEVICES=$DEV
rm -f "$DIR/o.bin"
START=$(date +%s)
timeout 30 $ROOT/build/sage_fa "$DIR" --device 0 "$@" > /tmp/srun.log 2>&1
RC=$?; END=$(date +%s)
[ $RC -ne 0 ] && { echo "RUN-HANG(rc=$RC, $((END-START))s) — 结果无效"; exit 2; }
[ ! -f "$DIR/o.bin" ] && { echo "RUN-NOOUT — 结果无效"; exit 3; }
AGE=$(( $(stat -c %Y "$DIR/o.bin") - START ))
[ $AGE -lt 0 ] || [ $AGE -gt $((END-START+1)) ] && { echo "RUN-STALE — 结果无效"; exit 4; }
echo "RUN-OK($((END-START))s)"
```

### 8. 精度回归套件结构（22 case 全维度）

```bash
run_case tag b h hkv sq sk d causal dtype   # gen_data → srun → verify
# 维度: 基础形状/ q·kv 尾块 / causal×3 / GQA(8:1,4:1,2:1) / D=64×3
#       / B>1×2 / bf16×2 / 长序列 32K→200K (--max-planes 采样校验防 golden OOM)
```
验收：kernel vs FP32 golden cosine ≥ 0.99（实测全部 0.99989~1.0）。

## 平台坑全集全文（fa-sageattention-troubleshooting）
> 以下为排障源文件全量正文：A 编译器/API（P1-P10）、B L0/catlass 组件（P11-P14）、C 流水/同步（P15-P19）、D 数据/精度（P20）、E 其他（P21-P28）、F 新版 catlass 适配（P29-P36）。

> 全部来自 FA-SageAttention 开发实践，按类别组织。
> 每条：症状 → 根因 → 解法。写代码前通读一遍可省数天。

### A. 编译器 / API 层

#### P1 AIV 上标量 GM store 静默失败
- **症状**: `(__gm__ uint32_t*)pr[4] = x` 在 AIC 上正常落盘，在 AIV 上**完全不落盘**（无报错）
- **解法**: AIV 侧 GM 进度标记必须 `Duplicate(UB 值) + DataCopyPad ≥1KB`

#### P2 count-API Muls/Adds 的 dst 寻址
- **症状**: `Muls(tensor + off, ...)` 编译错或解析到错误 overload
- **解法**: 一律 `Muls(tensor[off], ...)`（operator[]）

#### P3 StoreUnAlign 需左值
- **症状**: `StoreUnAlign(compAddr + rowOff, ...)` 报 no matching function
- **解法**: 先存局部 `__ubuf__ float* cAddr = compAddr + rowOff;`

#### P4 RegTensor 赋值
- **症状**: `t0 = f0`（RegTensor 拷贝）编译失败
- **解法**: 分支写法避免（if/else 各自完整路径）

#### P5 device 标量 fp 除法无法 lower
- **症状**: `error in backend: Do not know how to split...`
- **解法**: SageDivScr（8 元素 UB scratch 走向量 Div，count≥8、禁原地）

#### P6 `__simd_vf__` 参数限制
- **症状**: 参数必须 `__ubuf__` 地址空间；不能传 `__gm__*`
- **解法**: 需 GM 参数的 helper 用普通 `CATLASS_DEVICE` 函数

#### P7 `__vector__` kernel 无 sub-block
- **症状**: merge kernel 中 `GetSubBlockIdx()` 恒返回 0 → 行 64-127 无人算、输出半零
- **解法**: `sub = GetBlockIdx() & 1; task = GetBlockIdx() >> 1; 步长 = GetBlockNum() >> 1`

#### P8 MulDstAdd 语义
- **症状**: `MulDstAdd(dst, a, b)` 以为是 `dst += a·b`，实际是 **`dst = a·dst + b`**（乘的是累加器！）
- **解法**: 累加语义用 `Mul(t, wk, p); Add(acc, acc, t);`

#### P9 事件方向（HardEvent 方向性）
- **症状**: `SetFlag<V_MTE3>; WaitFlag<V_MTE3>` 想等 MTE3 拷贝落盘——方向反了（V 置位、MTE3 等，不等拷贝完成）
- **解法**: 等 MTE3 完成用 `PipeBarrier<PIPE_ALL>`（或 SetFlag<MTE3_V>+异地 Wait）。
  **kernel 退出不保证 MTE3 排空**，partial 写后必须 fence

#### P10 小 DataCopyPad 写不可靠
- **症状**: <1KB 的 UB→GM 写偶发不落盘；32B 必失败
- **解法**: ≥1KB；scale/μ 用 ×8 冗余凑尺寸。512B GM→UB 读可靠

### B. L0 / catlass 组件层

#### P11 FAIQK 与 FAIPV 的 Shape 语义相反
- **症状**: `BlockMmadTla` 的 L1TileShape 在两个 block 里读法不同——FAIQK 读 (M,N,K)，FAIPV 读 (M,K,N)
- **根因**: 传错导致静态槽距 ≠ 运行时实际 → L0A 槽溢出踩踏在飞 mmad
- **症状实例**: PV `Shape<128,64,128>` 使 K_static=64 而运行时 blockK=128 → L0A 槽距 16KB < 实需 32KB → 第二次 A-load 覆盖第一个在飞 mmad → 输出 cols 0-63 恒坏 ≈V 列均值
- **解法**: 见不变量 I1——垫 QK K_static=256 使两 block 槽距都是 32KB

#### P12 L0A/L0B 槽距错位竞态
- **症状**: P=1 对、P≥2 挂死或错数
- **根因**: 两 block 的 L0A/L0B 槽都从 0 布局且共享事件 id {0..3}，槽距不等时事件计数错配
- **解法**: I1；L0C 由 cursor 天然分区不冲突

#### P13 kernel 内 tiling 逐字段拷贝
- **症状**: 新增 tiling 字段后部分 case 输出全零（设备读到栈垃圾 → P=0）
- **根因**: kernel 里 `td` 是从 GM 逐字段拷贝的，漏拷新字段
- **解法**: 加字段必须同步加拷贝行

#### P14 CMake 头文件追踪
- **症状**: 改了 .asc include 的头，重编后行为不变（跑的旧二进制）
- **解法**: 头文件加入 `add_executable` 的 sources

### C. 流水 / 同步层

#### P15 AIC 上 PipeBarrier<PIPE_ALL>
- **症状**: 必挂死（AIC 管集合不含 V/MTE3）
- **解法**: 删；AIC 只用事件

#### P16 游标 post-increment
- **症状**: 步序错乱 [1,2,0]；t0 单块因 wrap 恰好回 0 而"假通过"
- **解法**: c.jCur（I4）；split-K 下 isFirst 判据用 jStart 非 0

#### P17 状态链槽位
- **症状**: 2 块 0.75、3 块 0.99、4 块 0.89 的渐进劣化
- **根因**: step s 读了同槽（未更新）的 lastMax/sumL
- **解法**: I5——读 (s-1)%N 槽写 s%N 槽（lmR/lmW/sumR/sumW 分离）

#### P18 MM2 双 set surplus
- **症状**: 4 块起 O 的 rows 32-63 损坏
- **根因**: RS 同时 set MM2{7,8} 两 id → surplus 让下下个 PV 提前通过等待，fixpipe 覆盖还在被读的 oUb
- **解法**: I3——只 set 本步槽 `MM2_INTRA_EVT[m2]`

#### P19 pL1 槽复用无 handshake
- **症状**: 步数 > 槽数时挂死
- **根因**: SM(s+N) 覆盖 pL1[m3] 时 AIC PV(s) 仍在读 L1
- **解法**: I8——PV_CONSUMED flag (id 12-15) + P-Cast 前 WaitFlag<MTE3_V>(m3Prev)

### D. 数据 / 精度层

#### P20 ×8 冗余 scale 索引漏乘
- **症状**: 1 块 0.9999、2 块 0.999、4 块 0.885 渐进劣化（拟合显示全部用了块 0 的 scale）
- **根因**: `gmSK.GetValue(idx)` 读的是 float 下标，×8 冗余布局必须 `idx*8`
- **解法**: I2；取证法见 fa-sageattention-precision.md §dScale 拟合

### E. 其他

#### P21 D=64 行距硬编码
- **症状**: D=64 崩溃 507015
- **根因**: RS 相位 `r*128` 硬编码，而 D=64 时 PV fixpipe 写 oUb 是紧凑 [64,64]（行距 64）；
  且 2D GM 写 (128B blockLen + srcStride) 组合触发设备错误
- **解法**: Rescale/DivL 加 D 参数（行距=D），Cast 只转 SH·D，GM 写 1D 连续

#### P22 merge 连续小拷贝 + 复用 event id
- **症状**: merge 读到的 m/l 部分为零/垃圾
- **解法**: works 连续 → 一次性整块拷贝 + PipeBarrier

#### P23 bf16 生成器
- **症状**: gen_data 声称 bf16 实际生成 fp16 数据
- **解法**: fp32 >> 16 截断路径

#### P24 GQA Q̂ 加载 stride 反了
- **症状**: GQA 下崩溃或错数
- **解法**: srcStride=(H-1)·D（GM 侧跳行），dstStride=0（UB 紧凑）

#### P25 因果 mask 反向
- **症状**: causal 输出错
- **根因**: `Sub(t0, c0, thr)` 方向反了（有效位置被 mask）；golden 的 causal 约定还有 (sk-sq) 偏移
- **解法**: `Sub(t0, thr, c0)` + 阈值 lim-1

#### P26 host JSON bool 解析
- **症状**: `atoll("true")` = 0 → causal 永远关
- **解法**: 前瞻匹配 "true"/"false"

#### P27 stale o.bin 欺骗（元坑，最危险）
- **症状**: kernel 挂死被 timeout 杀掉后 o.bin 不更新 → 验证脚本读旧文件 → 每次"结果"一模一样的假象 → 不知情连续发射挂死 kernel → 污染全部 8 张卡
- **解法**: **每次运行必查 rc + o.bin mtime**（srun.sh 固化）；timeout 后绝不解读任何输出

#### P28 python 批量编辑脚本静默失败
- **症状**: replace 字符串不匹配时不报错不写入；assert 中断导致文件整体未写、实验代码残留
- **解法**: 每次编辑后 grep 验证标记；**编辑后验括号平衡**（一次失衡导致 string.h 假错 + stale binary 连锁）

---

### F. 新版 catlass 适配坑（2026-08 重建上板时踩到，全部已修）

> 背景：源码丢失后按本 skill 重建，但重建环境装的是 **gitcode 最新 catlass**（比原开发时的旧版新）。
> 新版 catlass 的 `BlockMmadTla` 行为变了，重建代码沿用旧设计会直接挂死/编译错。以下 6 条是重建→22/22 PASS 修的全部。

#### P29 CATLASS_ARCH 未设 → catlass ascend950 头被 #if 跳过
- **症状**: 一上来连锁编译错（`ScaleGranularity` 未声明、`PackedTileCopyTla` 模板缺省不匹配）
- **根因**: catlass 头按 `CATLASS_ARCH==3510` 门控 ascend950 专用实现（含 `ScaleGranularity` 枚举）；CMakeLists 没设，catlass 默认 2201(AtlasA2) → ascend950 分支全跳过
- **解法**: CMakeLists 加 `target_compile_definitions(sage_fa PRIVATE CATLASS_ARCH=3510)`（dav-3510↔3510 映射）

#### P30 aclrtMalloc 的 void* 不能转 kernel 形参 GM_ADDR
- **症状**: `cannot initialize a parameter of type '__gm__ uint8_t *' with an lvalue of type 'void *'`
- **根因**: `aclrtMalloc` 给 `void*`，kernel 形参 `GM_ADDR`=`__gm__ uint8_t*`；ASC 不做 `void*→__gm__ uint8_t*` 隐式转换（cann-samples 用 `uint8_t* dX` 裸指针）
- **解法**: host 的 `DevMem.p` 用 `uint8_t*`（非 `void*`）+ `aclrtMalloc((void**)&p, ...)` 显式 cast

#### P31 AIC 重复 wait MM1/MM2（新版 catlass 内部已 wait）★最隐蔽
- **症状**: kernel 挂死（卡5/卡2 均超时，看门狗 ~27s 释放）
- **根因**: 新版 catlass `BlockMmadTla`（FAIQK:223 / FAIPV:197）**内部自己 `CrossCoreWaitFlag(MM1_RES_INTRA_EVENT[taskId])` / `MM2_RES_INTRA_EVENT[taskId]`**，flag ID 与本算子全同（{9,10}/{7,8}）；重建代码 AIC 又手动 wait 同一组 → 双重 wait：第一次消费 flag、catlass 内部第二次 wait 永久阻塞
- **解法**: 删 AIC 侧手动 `CrossCoreWaitFlag MM1/MM2`（保留 AIV 侧 set，catlass 等的就是它）
- **判别**: grep 本仓 catlass include 的 FAIQK/FAIPV 实现源，确认内部已含 `CrossCoreWaitFlag(MM*_RES_INTRA_EVENT[...])`（组件库源可直接核验）

#### P32 PV_CONSUMED 用循环前预置（冷 set 不传播）
- **症状**: 首次 SM 等 PV_CONSUMED 永卡 → 看门狗
- **根因**: 循环前在 AIC Init 上 `CrossCoreSetFlag<PIPE_FIX>` 预置 PV_CONSUMED，但此时 FIX 管道无活动，**预置的 set 不传播**（与 MM1/MM2 在 PIPE_V 上的预置不同——后者 V 管道同样冷却生效，故 MM1/MM2 bootstrap 可用，PV_CONSUMED 在 FIX 上不可）
- **解法**: 改 **step 守卫**：pL1 槽首次填充(step<槽数)跳过 wait、仅复用(step≥槽数)才 wait（仿 MTE3_V 的 `if it>=3` 守卫），无需 bootstrap
- **注意**: 单 buffer P-scale 与 4-slot pL1 不冲突——SM(s) it=s+1 写、PV(s) it=s+2 读、SM(s+4) it=s+5 才覆盖，PV 读在覆盖前 3 步，且已有 PV_CONSUMED 守卫同 SM 步保护

#### P33 merge 紧凑化只 copy sub0（sub1 漏合 → inf）
- **症状**: case0(causal 1024, splitK=4) qblock0 的 **sub1(rows 64-127) 全 inf**，其余 sub1 finite 但少合 shard
- **根因**: `Muls(wSlab[nValid*512], wSlab[k*512], 1.0f, 128)` 只 copy 128 floats（sub0 的 m|l@offset0），**漏 sub1 的 m|l@offset+256**；SageMergeW 对 sub1 读未紧凑化的 slot（含空 shard 的 garbage po）→ qblock0 仅 shard3 非空、sub1 读到空 shard → l*=0 → DivL inf
- **取证**: 看 o.bin 行模式——sub0 全 finite、sub1 全 inf → 直指 merge 的 sub 分支
- **解法**: copy **512 floats 全 slot**（sub0 m|l + pad + sub1 m|l + pad 都搬）

#### P34 bf16 输入被当 fp16 读（host 未转）
- **症状**: bf16 case cos≈0.28（非 inf，是错值）
- **根因**: host 把 bf16 输入的位直接喂 kernel（kernel 量化只认 `half`），bf16 位模式被误读成 fp16 → 量化全错
- **解法**: host 读入后若 bf16，**先 bf16→fp16(RNE)** 再喂 kernel（同 numpy `astype(float16)`）；emu/verify 的 bf16→fp32 路径不变

#### P35 "看门狗慢" 误判（实为每进程设备开销，非卡顿 flag）
- **症状**: 单次 srun 7~27s，疑似卡顿 flag/看门狗
- **根因**: 实测 host 逐阶段计时——`aclrtSetDevice`≈1.8s + `aclrtResetDevice`/`aclFinalize`≈4.6s 的**每进程设备初始化/销毁开销**（驱动+固件加载），kernel 本身仅 ~4ms；之前 27s 是卡被多次挂死后 setDevice/reset 变慢
- **解法**: 用 `--bench`（进程内预热+多 iter 均摊）测真实 kernel 用时；持久进程（ACLNN/torch_npu）后该开销消失
- **教训**: 计时前先分阶段定位，勿把设备开销当 kernel 卡顿

#### P36 FP8 MX PV 集成墙（path B 未完成，留作后续）
- **症状**: 把 PV 换 `MmadFAIPVMx` 编译错（`is_one_of<half, __fp8e4m3...>` 断言失败）
- **根因**: catlass FP8 cube 要求 **L1 内 P/V 就是 fp8(E4M3)**，不是 half；`CopyGmToL1B` 不自动 half→fp8 转换（DataCopy no match）。即"pL1 仍 half、catlass load 时转"的假设错
- **正解**: 用 ≤100 行探针 kernel 实测 catlass FP8 MX 组件的**真实 fp8 数据流**（`EpilogueAscend950FASoftmax` 怎么产 fp8 P+pScale、V 怎么 half→fp8 load、确切类型），照实测复刻，**勿靠 build error 盲猜类型**
- **教训**: catlass MX FP8 API 复杂，build-error 猜类型低效；先通读能跑的参考实现再写

# 验证方法

**精度取证方法论**：三层真值体系（FP32 golden → numpy 精确仿真 → kernel GM dump）；numpy 仿真直接
可用（每 step `comp_j = (qhat @ mu[j]) · (sm/sQ)`，sK 代数消去）；`SAGE_DUMP=1` dump qdbg 布局固定
512KB；逐层判定流程（哪层开始错 → 直接指向根因：系数交换→输入序错、比率错→m* 错、全零→该行未写）；
直接比较不够时做假设拟合；挂死用分相位标记取证。[^sage-precision]

**验收标准**（v1 已达标并重建复现）：[^sage-recipe]

- kernel 级精度：vs FP32 golden cosine ≥ 0.99 —— 22/22 case PASS，cos 0.999592~0.999909。
- 覆盖：causal / GQA(8:1,4:1,2:1) / MQA / D=64 / B>1 / bf16 / 尾块 / 长序列 32K~200K 不爆内存
  （超长序列用 max-planes 采样校验防 golden OOM）。
- 性能基线（`--bench`，28 AIC+56 AIV）：128²≈31μs / 1k²≈73μs / 4k causal≈505μs / 200k≈719ms。

[^sage-architecture]: 固定提交中的 FA-SageAttention 架构规格：硬件前提、GM 布局契约（×8 冗余）、Kernel A/B/Merge 结构、CrossCore mode4 flag 表、4 相位主循环与新版 catlass 适配注记。
[^sage-recipe]: 固定提交中的 fa-sageattention 开发配方：量化路径与 comp 代数、七步工作流、不变量要点、排障路由、验收标准与重建 6 处适配。
[^sage-rules]: 固定提交中的硬规则：8 条不变量 I1-I8 与运行纪律（Always/Never 清单）。
[^sage-precision]: 固定提交中的精度取证方法论：三层真值、numpy 精确仿真、GM dump 布局、逐层判定、假设拟合、挂死取证与 22 case 回归套件结构。
[^sage-troubleshooting]: 固定提交中的 dav-3510 平台坑全集：编译器/API、L0/catlass 组件、流水同步、数据精度与新版 catlass 适配坑（P1-P36）。
