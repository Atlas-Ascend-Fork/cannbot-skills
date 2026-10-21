---
type: "operator"
title: "FAG dense 反向 Ascend 950 实现模式"
description: "dense flash attention backward 的 ABI、反向公式、Tile 级 C++ 组装、五次 MMAD 数据链、L0C/UB/L1 预算、raw-NZ 存序、FagPostKernel、同步协议、失败表现与验收矩阵。"
tags: ["catlass-cpp", "flash-attention-grad", "fag", "flash-attn-grad", "ascend950", "tile-level", "raw-nz"]
status: "stable"
generated: {"by": "process:catlass-cpp-fag-as-built-curation", "at": "2026-10-20T00:00:00Z"}
verified: [{"by": "process:fag-as-built-code-cross-check", "at": "2026-10-20T00:00:00Z"}]
sources:
  - {"id": "fag-interface", "resource": "git:cann/ops-transformer@97e06c5ebb3f6a8d7c7661c40adfe063fd166e45:attention/flash_attn_grad/README.md", "title": "FlashAttnGrad 算法与接口说明", "kind": "repository"}
  - {"id": "fag-torchapi", "resource": "git:cann/ops-transformer@97e06c5ebb3f6a8d7c7661c40adfe063fd166e45:attention/flash_attn_grad/docs/torchapi_flash_attn_grad.md", "title": "flash_attn_grad torch 接口文档：产品支持范围、公式、参数与调用示例", "kind": "repository"}
  - {"id": "catlass-arch-api", "resource": "git:cann/catlass@2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52:include/catlass/arch/arch.hpp", "title": "Ascend950 ArchTag 与片上存储容量定义", "kind": "repository"}
  - {"id": "catlass-tile-copy", "resource": "git:cann/catlass@2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52:include/catlass/gemm/tile/tile_copy.hpp", "title": "PackedTileCopyTla 与 GM/L1/L0 拷贝组件", "kind": "repository"}
  - {"id": "catlass-tile-mmad", "resource": "git:cann/catlass@2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52:include/catlass/gemm/tile/tile_mmad.hpp", "title": "TileMmadTla 单 tile MMAD 接口", "kind": "repository"}
  - {"id": "catlass-l0c-copy", "resource": "git:cann/catlass@2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52:include/catlass/gemm/tile/ascend950/copy_l0c_to_gm.hpp", "title": "Ascend950 CopyL0CToGmTla 与 ScaleGranularity", "kind": "repository"}
  - {"id": "asc-fixpipe-api", "resource": "git:cann/asc-devkit@bc50e8825e59807694f0a525095adbead5a40d37:include/basic_api/kernel_struct_fixpipe.h", "title": "FixpipeConfig、CO2Layout、FixpipeParamsC310 与 dualDstCtl", "kind": "repository"}
operator_families: ["flash-attention-grad"]
architectures: ["ascend950"]
---

# 接口与概念

FAG 是 dense flash-attention backward：由 `Q/K/V/O`（`O` 即 vendor 的 `attention_in`）、上游梯度
`dO` 和前向 `LSE` 一次算出 `dQ/dK/dV`。本族不接收 block mask，也不依赖反向 metadata 分核表；
任务空间是 `(batch, qHead, qBlock)` 的直角坐标，KV 轴在单个任务内部顺序推进。[^fag-interface]

冻结的直调 ABI（与 `torch_npu.npu_fusion_attention_grad` 语义一致，但按直调入口固定）：

- 输入顺序：`q, k, v, dout, attn_out, softmax_lse, actual_seq_qlen, actual_seq_kvlen, scale, layout`。
- 输出顺序：`dq, dk, dv`，shape、dtype、layout 分别与 `q/k/v` 一致。
- dtype：`q/k/v/dout/attn_out` 为 `float16` 或 `bfloat16`，同一调用必须同 dtype；
  `softmax_lse` 为 `float32`，`actual_seq_*` 为 `int32`，`scale` 为 `float32` 标量，
  `layout` 为 `int32` 标量。
- layout 是显式 ABI，禁止按 shape 推断：
  - `0` = BNSD：`q/o/do` 为 `[B,N1,S1,D]`，`k/v` 为 `[B,N2,S2,D]`，LSE 为 `[B,N1,S1]`；
  - `1` = BSND：`q/o/do` 为 `[B,S1,N1,D]`，`k/v` 为 `[B,S2,N2,D]`，LSE 为 `[B,N1,S1]`；
  - `2` = TND：`q/o/do` 为 `[T1,N1,D]`，`k/v` 为 `[T2,N2,D]`，LSE 为 `[T1,N1]`。
- TND 的 `actual_seq_*` 是长度为 `B` 的累计结束位置，段 `b` 拥有 `[cum[b-1], cum[b])` 行；
  不允许把它传成逐段长度。BNSD/BSND 下它们是信息性输入，必须分别等于 `S1`、`S2` 的逐 batch
  长度数组。
- GQA/MQA：`N1 = N2 * group`，query head `h` 使用 KV head `h / group`；同一 KV head 的
  `dK/dV` 必须对所有 query head 求和。[^fag-interface]

## 算子算法

```text
S   = scale * Q @ K^T
P   = exp(S - LSE)
D   = rowsum(dO * O)
dP  = dO @ V^T
dS  = P * (dP - D)
dQ  = scale * dS @ K
dK  = scale * dS^T @ Q
dV  = P^T @ dO
```

`LSE` 由前向给出，反向不需要重算 running max/sum，也不需要恢复历史 `O` 的分块状态。`P` 是唯一
的 softmax 反向相关中间量，`D` 是逐 query 行的标量。[^fag-torchapi]

`scale` 的位置是 FAG 最常见的精度错误来源：

- `S` 阶段乘一次 `scale`；
- `dS`、`dP`、`dV` 不乘 `scale`；
- `dQ/dK` 在出口再乘一次 `scale`；
- `dV` 出口绝不乘 `scale`，即使与其他梯度共用同一个 Post kernel。

内部 MMAD 累加使用 fp32。`P/dS` 在送入下一次 MMAD 前转成输入 dtype；`LSE`、`D`、`S`、`dP` 的
行归约与减法保持 fp32。把 `LSE` 或 `S` 提前降到 fp16/bf16 会放大近零区域误差。

输出语义：

- `dQ` 的每个 `(batch, query head, query block)` 只属于一个任务，任务内跨 KV 扫掠累加；
- `dK/dV` 的每个 KV 行可能被多个 query head 或多个 query block 贡献；GQA/MQA 必须做 FP32 归约；
- 尾块行或列不参与有效输出；输入平面中补零行不能污染 `P/dS` 或输出。

# 用法

## 分核策略与基本块切分

采用 Q-outer 的 Tile 级模式，不采用 CATLASS 上游 `BlockMmadFag` 或 example 105 的 Block 级
组件链。一个 AIC 与两个 AIV 组成一个核组，`FagMainKernel` 使用
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`。

- host 生成 `nTask = sum_b(N1 * qBlk[b])`，其中 `qBlk[b] = ceil(qLen[b] / kBM)`。
- task 索引展开为 `(b, qHead, qBlock)`；组内 AIC 与两个 AIV 必须走完全相同的 task 顺序。
- AIC 的组编号使用 `GetBlockIdx()`；AIV 的组编号使用 `GetBlockIdx() / GetTaskRatio()`，
  子块编号使用 `GetSubBlockIdx()`。不要直接拿 AIV 原始 block index 当 task index。
- 每个 task 内 KV 轴按 `kBN` 顺序扫掠，Q/dO 在整个 task 内常驻，K/V 按 KV step 双缓冲。
- `kBN = 128` 固定；`kBM = 128` 当 `D <= 128`，`kBM = 64` 当 `D = 192`。
- host 只实例化 6 个 TilingKey 组合：`{float16, bfloat16} × {D=64, D=128, D=192}`；D 的派发阈值是
  `D <= 64`、`D <= 128`、否则 `D=192`，不支持 D=256。
- `layout` 是运行期字段，不进入 TilingKey；`qRound = AlignUp(qLen, kBM)`，
  `kvRound = AlignUp(kvLen, kBN)`，任务数按 padded `qBlk` 计算。
- 尾块允许 `S1/S2` 不是基本块整数倍；有效行、有效列必须分别由 `rowMax`、`colMax` 限制。
- TND 每个 batch segment 独立切块；不能把多个 segment 合并成一个连续 Q/KV 任务。

必须显式处理的边界：

- `S1 < kBM` 或 `S2 < kBN`；
- `S1 != S2`；
- TND 段长不等，例如 `[32,64,96]`；
- GQA/MQA 的 `group` 不固定，不能只支持 2 或 4；
- `D = 192` 时 `kBM` 减半，所有依赖 `kBM` 的 L1/L0C/UB 偏移必须随 `Geo<kD>` 重算。

## 数据路径与存储层级

### 输入与地址

- `Q/K/V/O/dO/LSE` 从 GM 读取。支持 layout 0/1/2 时，直接使用用户 tensor 的
  `base + headStride + row*rowStride` 形式寻址，不再生成 packed 输入平面。
- 每个 tile 的行数必须按 batch segment clamp，防止尾块越过用户 tensor 边界。
- `FagCopyKernel` 只保留为 canonical packed plane 的兼容路径；当前支持的 layout 全部走
  plane alias，host 不应无条件启动 pack/unpack。
- LSE 始终使用 host 构造的 canonical `[N, paddedQ]` fp32 平面；它很小，不值得 alias。

workspace 的通用布局是：packed 输入平面（Q/K/V/O/dO/LSE）→ fp32 `dqWs/dkWs/dvWs` →
canonical 输入 dtype 输出平面（`dqOut/dkOut/dvOut`），每段起点按 512 B 对齐。plane alias 模式下
输入/输出直接指向用户 tensor，但这些 workspace 段和 fp32 梯度区仍然存在；不要把
`ws + offDqWs` 当成 workspace 的绝对起点，dK/dV 的 plane offset 必须相对该指针重定位。

### L1 / L0 / UB 几何

所有偏移由 `Geo<kD>` 编译期常量给出。以 `kBN=128`、`DCHUNK=64` 为固定值，按当前 as-built 公式
计算出的实际占用如下（单位 KiB）：

| D | kBM | L1 | L0C | L0A 单槽 | L0B 单槽 | L0A/L0B 双槽 | UB/AIV |
|---:|---:|---:|---:|---:|---:|---|---:|
| 64 | 128 | 224 | 160 | 32 | 16 | 是 | 232.25 |
| 128 | 128 | 320 | 192 | 32 | 32 | 是 | 232.25 |
| 192 | 64 | 304 | 240 | 16 | 48 | 否 | 116.125 |

这些数字比旧的“L1 256 KiB、L0C 176 KiB”表更准确；不要用旧表覆盖源码中的
`static_assert`。L1 预算由以下部分组成：

```text
L1 = 2 * (kBM*kD*2)              # Q, dO 各一份，task resident
   + 4 * (kBN*kD*2)              # K/V 各两个 ping-pong 槽
   + 4 * (kBM*kBN*2)             # dS/P 各两个 ping-pong 槽
```

L0C 使用三个物理区域，不用四个：

```text
l0cTmp = max(kBM*kBN, kBN*kD) * 4    # S -> dK
l0cAux = 同一大小                     # dP -> dV
l0cQ   = kBM*kD*4                    # dQ，跨 KV sweep 常驻
L0C    = l0cTmp + l0cAux + l0cQ
```

禁止增加第 4 个 L0C 槽。D=192 时 `l0cTmp` 与 `l0cAux` 合计 192 KiB，连同 `l0cQ` 共 240 KiB；
再加一块会越过 256 KiB，运行时通常表现为 `507015`。

L0A/L0B 采用双槽来重叠 `L1->L0` 装载与上一组 MMAD；D=192 时 L0B 单槽已经为 48 KiB，双槽会超过
64 KiB，因此 `kL0Dual=false`，槽 1 必须退化为槽 0 的别名。不要只按“D≤128 双槽”复制代码而忘记
D=192 的断言和槽选择。`l0Slot` 按 MMAD 组递增，不按 KV round，也不在 task 边界重置；槽奇偶
必须让 AIC 的 prologue、steady、drain 和相邻 task 连续衔接。

### GEMM 类型映射

五次 MMAD 固定使用下面五个 `FagTileGemm` 实例：

| GEMM | A 侧 layout | B 侧 layout | 输出 | L0C 区域 |
|---|---|---|---|---|
| `GQK` | `RowMajor` | `ColumnMajor` | `S` | `l0cTmp` |
| `GPV` | `RowMajor` | `ColumnMajor` | `dP` | `l0cAux` |
| `GDQ` | `RowMajor` | `RowMajor` | `dQ` | `l0cQ` |
| `GDK` | `ColumnMajor` | `RowMajor` | `dK` | `l0cTmp` |
| `GDV` | `ColumnMajor` | `RowMajor` | `dV` | `l0cAux` |

`GDK/GDV` 的 A 侧不是复制出来的转置矩阵：AIV 写入 L1 的 `zN(kBM,kBN)` slab 与
`nZ(kBN,kBM)` 的字节序相同，因此直接把同一 slab 声明成 `ColumnMajor`/`nZ` 视图即可。D=192 的
`kBM=64, kBN=128` 必须使用正确的 `nZ(kBN,kBM)` 视图；误写成 `nZ(kBM,kBN)` 会让 dK/dV 读到错误
或越界字节。

## 实现顺序

按下面的顺序落地，避免先写完整 kernel 再回头改资源几何：

1. 固定 ABI、layout 枚举、dtype 分支和逐 batch 的 `qLen/kvLen`。
2. 实现 `Geo<kD>`：`kBM/kBN`、L1/L0C/L0A/L0B/UB 偏移和静态断言。
3. 实现 `FagTileGemm` 的 `LoadA/LoadB/Mmad/Store`，先用单 tile 验证五种 layout 组合。
4. 实现 Q-outer task 循环和 Q/dO 常驻、K/V 双缓冲。
5. 接 S/dP raw-NZ Fixpipe 和 AIV 的 D/P/dS 链。
6. 接 dS/P 的 zN L1 store，再跑 mm3/mm4/mm5。
7. 接 dQ 常驻 L0C 出口，以及 direct/workspace 两条出口。
8. 接 `FagPostKernel` 的 GQA 归约、scale 和 cast。
9. 最后接 pack/unpack 兼容路径；支持的 layout 不应依赖它。

# 代码模式

## 一 MMAD 的 `FagTileGemm`

不要用 CATLASS 的 `BlockMmadFag` 直接替换本模式。我们采用的是 Tile 级 helper，形状如下：

```cpp
template <class ArchTag, class EA, class LA, class EB, class LB, class EC, class LC>
struct FagTileGemm {
    using TileCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, EA, LA, EB, LB, EC, LC>;
    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, LA, typename TileCopy::LayoutTagL1A>;

    static void LoadA(l1A, gmA, m, k, evFree);   // GM -> L1, MTE2/MTE1 握手
    static void LoadB(l1B, gmB, k, n, evFree);   // GM -> L1, MTE2/MTE1 握手
    static void Mmad(tL1A, tL1B, l0A, l0B, l0C, m, n, k, initC,
                     evA, evB, evL0C, unitFlag);
    static void Store(tL0C, gmC, evL0C);         // NO_QUANT 的通用出口
};
```

实现细节：

- `LoadA/LoadB` 在拷贝前等待空闲 credit，拷贝后设置 `MTE2_MTE1`，再等待该事件完成；它们不是
  异步 fire-and-forget，调用者不要误以为返回时 L1 尚未就绪。
- `Mmad` 先做 `L1->L0A/L0B`，设置 `MTE1_M`，等待两个操作数都到位后再调用 `TileMmadTla`。
- `unitFlag != 0` 时使用硬件 unit flag，不再软件等待/设置 `FIX_M`；`unitFlag == 0` 时才走
  软件 `FIX_M` 路径。两种路径不能在同一区域混用而不成对。
- 通用 `Store` 使用 `CopyL0CToGmTla<..., ScaleGranularity::NO_QUANT, false>`。需要随路 scale
  的 direct 出口必须显式实例化 `PER_TENSOR`；不要指望通用 `Store` 自动乘 scale。
- dQ/dK/dV 的 direct 出口和 workspace 出口共享 MMAD 结果，但输出地址、scale 粒度和 atomic
  语义不同，不能只改一个 bool 而漏掉另一条路径。
- unit flag 链默认开启：`kUnitFlagChain=true`、`kMmadUF=0b11`、`kMmadUFMid=0b10`、`kFixUF=0b11`。
  只有 dQ 累加链（mm3）的中间轮使用 `0b10`，其最后一轮使用 `0b11`；mm1/mm2/mm4/mm5 每轮以及
  出口 Fixpipe 都使用 `0b11`。不要把 mm3 中间轮也设成 `0b11`，否则会留下无人释放的 FIX credit。

## 流水排布、同步关系与数值精度

### 主流水

#### AIC 侧

每个 KV step 执行 A 阶段，随后执行上一 step 的 C 阶段：

```text
A(step):
  Load K/V(step) -> L1
  mm1: S  = Q @ K^T        -> l0cTmp
  raw-NZ Fixpipe S         -> AIV UB
  mm2: dP = dO @ V^T       -> l0cAux
  raw-NZ Fixpipe dP        -> AIV UB

C(step-1):
  wait dS/P in L1
  mm3: dQ += dS @ K        -> l0cQ       (跨 KV sweep 常驻)
  mm4: dK  = dS^T @ Q      -> l0cTmp
  mm5: dV  = P^T @ dO      -> l0cAux
```

关键顺序：

- step 循环必须跑 `nKv + 1` 次：`step 0` 只做 A，`step nKv` 只做 C；少跑一轮会丢掉最后一组
  dQ/dK/dV。mm3 只在首个 C 轮（`step == 1`）以 `initC=true` 初始化，后续 C 轮累加；mm4/mm5
  每轮都以 `initC=true` 重新计算本 KV 块。
- A(step) 必须在 C(step-1) 之前发射，使 mm1/mm2 与 AIV 的上一轮 B 阶段重叠。
- dQ 只在 task 结束时写一次；不要在每轮 KV step 对 `l0cQ` 做 Fixpipe。
- dK 复用 `l0cTmp`、dV 复用 `l0cAux` 的旋转关系必须保持；这是不增加第 4 个 L0C 槽的前提。
- `K/V` 的最后一个消费者是 C 阶段；C 把 L1 读入 L0 后才能归还 K/V 的 MTE2 credit。

host 的 launch 顺序是：需要时先 pack（兼容路径）→ `FagMainKernel` → 若 `dirAll==0` 再
`FagPostKernel` → 需要时 unpack（兼容路径）。支持的 layout 走 alias 时不启动两次 copy launch；
direct 路径下不启动 Post。不能用“Post 是否存在”判断 kernel 是否完整。

#### AIV 侧

每个 task 先算 `D`，再按 KV step 做 P/dS：

每个 AIV 子块负责 `row0 = qb*kBM + sub*kHalf` 起的 `kHalf` 行；`rowMax` 由
`qLen - row0` clamp 到 `[0, kHalf]`，`colMax` 由 `kvLen - step*kBN` clamp 到 `[0, kBN]`。负的
`rowMax` 绝不能被直接用于 UB 尾偏移。

```text
task 级:
  load O/dO/LSE 的分块
  D = rowsum(dO * O)                  // fp32
  build LSE 和 D 的 panel-order broadcast

每 KV step:
  wait FL_S
  S = scale * S
  P = exp(S - LSE)                    // fp32 原地
  wait FL_DP
  dP = dP - D
  dS = dP * P                         // 不加 scale
  cast P/dS -> dtype
  mask 越界行/列
  StoreZNToL1(P), StoreZNToL1(dS)
  发 FL_DS
```

`LSE` 和 `D` 的行广播面板在 task 内只生成一次，KV 扫掠期间复用。`P/dS` 的 UB tile 是单个
`kHalf x kBN` 面板序存储；全块用一次 flat Cast，尾块必须按面板清零再对有效列做 multi-repeat
Cast。不能把尾块当成 full block 直接 cast，否则 padding 列会读到 K 的零填充或旧数据。

### raw-NZ Fixpipe

S/dP 从 L0C 到 AIV UB 使用：

```cpp
constexpr AscendC::FixpipeConfig kCfgNzUb = {AscendC::CO2Layout::NZ, true};
```

`isToUB=true` 是必需的；`CFG_NZ` 的 `isToUB=false` 表示 L0C→GM 语义，不能用于 L0C→UB。

`FixpipeParamsC310<CO2Layout::NZ>` 的关键字段：

- `nSize = kBN`；
- `mSize = kBM`；
- `srcStride = kBM`，单位是 L0C panel pitch 的 64B；
- `dstStride = kHalf * 16`，单位是 UB 中的 fp32 元素；
- `dualDstCtl = 1`，将 M 对半写入两个 AIV 的 UB；
- `quantPre = NoQuant`，`reluEn = false`，`unitFlag = kFixUF`。

`dualDstCtl=1` 要求 `mSize` 为偶数，因此 D=192 的 `kBM=64` 仍然合法。一次 Fixpipe 同时服务两个
AIV 子块，不能每个子块各自再发一次，否则会重复写入或打乱 FL_S/FL_DP 的完成顺序。

L0C fp32 的 `[n/16][m][16]` 元素序与 fp16 的 `zN(kBM,kBN)` 相同，flat Cast 只改元素大小，不改
面板顺序；因此 AIV 侧不需要 `PermuteToZN`。`UB_PP`/`UB_DSP` 是两个历史暂存区，当前模式不再需要。

面板序的行广播和 L1 写回固定如下：

- `RowBrcBlkZN`：用两次 `Brcb`，参数为 `BrcbRepeatParams(2,16)`；第二次目的地址加 8 个 block，
  将每个 LSE/D 行标量展开成每行 16 份。
- `SubRowBrcZN`：每个 16 列面板一次 `Sub`，`mask=64`（4 行 × 16 fp32）、
  `repeat=kHalf/4`、`BinaryRepeatParams(1,1,1,8,8,8)`；共 `kBN/16` 次。
- `StoreZNToL1`：输入已经是面板序，使用一次 `DataCopy`，`blockCount=kBN/C0`、
  `blockLen=kHalf*C0`、`srcStride=0`、`dstStride=列块 pitch - chunk`；不要再做行主序到 zN 的
  `PermuteToZN`。
- 行归约的 `DRSE=1`：`ReduceRepeat` 的 per-repeat 结果在 UB 中按 1 个 float 连续排列，不是按 8
  个 float 的 32B block 排列。按 8-float stride 读会取到第 0、8、16… 行，D 全错并进一步污染
  dQ/dK。

### `FagPostKernel`

只在 `dirAll == 0` 时由 host 启动，`blockDim` 使用 AIV 核数。它读取 fp32 `dqWs/dkWs/dvWs`，
按 64 行一块处理：

- dQ：按 query head 和行块读取，乘 `scale`，cast 到输入 dtype，写用户布局；
- dK：对同一 KV head 的 `group` 个 query head 求和，乘 `scale`，cast，写用户布局；
- dV：同样求和，但不乘 `scale`，直接 cast，写用户布局。

GQA 归约的 `g==0` 用 `Adds(..., 0.0f, ...)` 初始化，后续 group 用 `Add` 累加；不要把
`MTE3_V` 的等待放进 group 循环，否则 G>1 时 credit 会被提前耗尽并挂死。每个输出行块的
`V_MTE2`/`MTE3_V` 等待与设置必须成对，且一个输出 buffer 的一对事件只在整个行块结束时结算一次。

### 同步协议

跨核 flag 使用 `SYNC_MODE=4`，固定保留 8/9/10 给 CATLASS barrier helper；本族使用：

| flag | 方向 | 语义 |
|---|---|---|
| `FL_S=11` | AIC→AIV | S 已进入 AIV UB，可开始 `scale*S`/`exp` |
| `FL_DP=12` | AIC→AIV | dP 已进入 AIV UB，可开始 dS 链 |
| `FL_DS=13` | AIV→AIC | 本轮 P/dS 已写入 L1，可开始 mm3/mm4/mm5 |
| `FL_ACK=14` | AIC→AIV | AIC 已消费上一轮 L1 slab，可回写同奇偶槽 |
| `FL_FREE=15` | AIV→AIC | AIV 已消费 UB 的 S/dP 槽，可被下一次 Fixpipe 覆盖 |

`FL_S` 与 `FL_DP` 必须分开：S 的 exp 链只依赖 mm1，dS 链才依赖 mm2。合并两者会让整个 AIV
链等待较晚的 Fixpipe。

`gRound` 是跨 task 连续递增的全局轮次，所有 ping-pong 槽都按 `gRound & 1` 选择。task 之间不能
重置轮次，否则边界会同时复用两个槽。AIV 侧对 `FL_FREE` 支付两个 prologue credit；AIC 侧对
`FL_ACK` 支付一个 prologue credit。`FL_DS` 是饱和的 mode-4 counter，必须由 `FL_ACK` 形成反向
背压，不能只靠 `FL_DS` 自增。

### direct 与 workspace 出口

host 计算 `dirAll`，语义是**全 batch 全有或全无**：

```text
dirAll = (group == 1)
         && 每个 batch 的 qLen % kBM == 0
         && 每个 batch 的 kvLen % kBN == 0
         && max(qBlk, kvBlk) <= directAccumCap
```

`directAccumCap` 按 dtype 分开：fp16 为 32，bf16 为 8。bf16 不要沿用 fp16 的 32；GQA、
尾块或长累加链一律走 fp32 workspace。

workspace 和 direct 的输出平面都必须在首次累加前清零。dQ 由唯一 task 拥有，主 kernel 直接覆盖写；
dK/dV 会被多个 query block 累加，workspace 出口用 fp32 atomic，direct 出口用输入 dtype atomic。
不要把 dQ 也改成 atomic，也不要漏掉 dK/dV 的预清零。

- `dirAll == 1`：主 kernel 直接把 dQ/dK/dV 写用户输出平面；dQ/dK 用 `PER_TENSOR` scale，dV 用
  `NO_QUANT`。dK/dV 仍需按 query block 原子累加，使用 `SetAtomicAdd<Element>()`，并在每个
  输出块结束后 `SetAtomicNone()`。
- `dirAll == 0`：主 kernel 把 dQ 覆盖写、dK/dV 原子累加到 fp32 workspace，随后
  `FagPostKernel` 做 GQA 归约、scale 和 cast。
- AIV 代码中可能保留一个 `kUseDirectExit=false` 的旧局部常量；它不是 host 的出口开关。真正
  开关是 tiling 中的 `dirAll`，不要根据局部常量判断是否启动 Post。

# 约束

- 组件边界：五次矩阵计算必须走 CATLASS Tile 组件；本模式使用
  `PackedTileCopyTla`、`TileMmadTla`、`CopyL0CToGmTla`，不用 `DeviceGemm`，不手写矩阵乘。
- 不要从 CATLASS `BlockMmadFag` 或上游 FMA/FMHA 示例直接移植 Block 级流水。它们可以作为 API
  语义参考，但本模式的 task 映射、L0C 旋转、raw-NZ UB 路径和 Post 出口必须按本文实现。
- 不要使用 `PackedTileCopyTlaToUB<SPLIT_M>` + RowMajor + `PermuteToZN` 的旧路径。
- 不要使用 `CFG_NZ`（`isToUB=false`）做 L0C→UB。
- 不要增加第 4 个 L0C 物理槽；不要让 dK/dV 共享同一个相位槽。
- 不要把 `kBM=128` 硬编码到所有 D；D=192 必须使用 `kBM=64`、`kHalf=32` 和正确的 nZ 视图。
- 不要无条件启动 pack/unpack。layout 0/1/2 使用 plane alias；copy kernel 只是兼容回退。
- 不要在每次 KV step 写 dQ。dQ 在 `l0cQ` 跨 KV sweep 累加，task 结束写一次。
- 不要用 packed 输入平面掩盖 TND/尾块错误；先验证 alias 寻址和 segment clamp。
- 不要放宽冻结的逐例 tolerance 或 `matched_ratio=1.0`；不要修改 golden、shape 或 layout 枚举。
- raw-NZ 依赖当前 CANN/PTO 的 nz2nz stride 语义；升级 CANN 或编译器后必须重新跑覆盖矩阵。

# 失败表现

| 现象 | 优先检查 |
|---|---|
| `507046` 挂起/超时 | `FL_S/FL_DP` 是否分开；`FL_DS`/`FL_ACK` 是否成对；`gRound` 是否跨 task 连续；unitFlag 链是否在 FIX 阶段正确收尾；启用 copy fallback 时 UB scratch 的 MTE3→MTE2/MTE3→V credit 是否跨 job 保持 |
| `507015` AICORE_EXCEPTION | L0C 是否超 256 KiB；D=192 是否误用 `kBM=128`；`rowMax` 是否为负或 UB 尾偏移越界 |
| `507014` 挂死 | 是否在 G==1 且 KV 尾块不整除时走了低精度 direct 出口；应回退 fp32 workspace |
| dK/dV 整片错，dQ 正常 | AIV 写 L1 的 zN slab 是否被错误声明为 `nZ(kBM,kBN)`；mm4/mm5 的 ColumnMajor A 视图是否正确 |
| dQ/dK 成比例偏大/偏小，dV 正常 | `scale` 在 S 和出口重复/遗漏；Post 是否只给 dQ/dK 乘 scale |
| 只有 GQA/MQA 的 dK/dV 错 | `kvHead = qHead / group`；Post 的 group 循环是否从 g=0 初始化并避免过早结算 `MTE3_V` |
| TND 第二段开始整体错位 | `actual_seq_*` 是否误传成逐段长度；segment base/row stride 是否串段 |
| 尾块随机坏行/坏列 | `PlaneRows`、`rowMax`、`colMax`；全块 Cast 是否覆盖了未清零的 padding |
| S/dP 对但 P/dS 错 | raw-NZ Fixpipe 是否用了 `isToUB=false`；`dualDstCtl` 是否为 1；面板序是否被二次 permute |
| dK/dV 重复累加或跨进程不一致 | 精度复跑是否 fresh workdir、单次 launch；dK/dV 是 atomic 累加，不能拿重复 launch 的输出做精度 |

# 验证方法

## 精度

- 使用纯 torch oracle，CPU 端 fp32/fp64 累加，逐元素比较 `dQ/dK/dV`。
- 容差取冻结用例的 `max_atol/max_rtol`，`required_matched_ratio` 必须为 1.0；不得用统一容差替代。
- 每个 case 使用全新的工作目录且只 launch 一次。dK/dV 是原子累加，重复 launch 会产生假失败。
- 至少单独检查：`S`、重算 `P`、`dS`、`D`，以区分输入 LSE 误差、softmax 反向误差和后三次 MMAD
  误差。

## 覆盖矩阵

至少覆盖：

- layout 0/1/2；
- fp16 与 bf16；
- MHA、GQA、MQA；
- D=64/128/192；
- B=1/2/3；
- `S1 != S2`；
- 尾块，例如 `S=257,K=129`、`S=128,K=64`、`S<kBN`；
- TND 多段且段长不等，例如 `[32,64,96]`；
- 大 shape：1024、2048、4096；
- direct 可用与 direct 被禁止的两类 batch 组合。

## 性能

性能口径固定为 `FagMainKernel + FagPostKernel` 的 msprof `Task Duration`；剔除首个 warm-up，
取至少 4 个正式样本的中位数。vendor 比较对象是 `torch_npu.npu_fusion_attention_grad` 的 device
kernel，不使用 host wall time。

参考实现观测（Ascend 950PR、CANN 9.2.0-beta.2、`CATLASS_ARCH=3510`，仅用于校准，不是普遍承诺）：

| shape | ours total µs | vendor µs | ratio |
|---|---:|---:|---:|
| bnsd-mha-d128-s1024 | 56.34 | 57.59 | 1.022 |
| bnsd-mha-d128-s2048 | 329.99 | 324.45 | 0.983 |
| bnsd-mha-d128-s4096 | 1293.96 | 1189.41 | 0.919 |

同一参考实现的全量 19 例观测为 15/19 ≥0.9x、13/19 ≥1.0x、中位 1.043x；剩余主要缺口是
`D=192`、大 shape 的 bf16/GQA 以及 TND bf16 的 Post 成本。这些数字是回归参考，不是 knowledge
对任何工程验收的声明。

## 实机检查清单

1. 编译日志确认 `--npu-arch=dav-3510`、`-DCATLASS_ARCH=3510` 和 catlass include 都进入真实编译行。
2. 运行前确认源码/二进制 md5，避免测到旧对象。
3. 先用 D=64、MHA、无尾块、fp16 的小 case 验证 tile 和出口，再上 GQA/TND/tail。
4. 第一次失败先查 ABI/layout、`rowMax/colMax`、`dirAll`，不要直接调 tile 形状。
5. 出现挂起先查 flag/event 配对和 prologue credit；出现整片数值错先查 zN/nZ 视图；出现比例错先查
   `scale`。
6. 性能采集前确认 `FagPostKernel` 是否应该存在；direct 路径下它的缺失是正确行为，不是丢 kernel。

[^fag-interface]: 固定提交中的 FAG 反向 ABI、输入输出顺序、layout、dtype、GQA/MQA 语义与目录职责说明。
[^fag-torchapi]: 固定提交中的产品支持范围、前向/反向公式、`scale` 语义、参数说明与调用示例。
[^catlass-arch-api]: 固定提交中的 `Arch::Ascend950`、L1/L0A/L0B/L0C/UB 容量与架构标签定义。
[^catlass-tile-copy]: 固定提交中的 `PackedTileCopyTla`、GM→L1、L1→L0 拷贝接口与 layout tag 选择。
[^catlass-tile-mmad]: 固定提交中的 `TileMmadTla` 单 tile MMAD 调用形式、initC 与 unitFlag 语义。
[^catlass-l0c-copy]: 固定提交中的 `CopyL0CToGmTla`、`ScaleGranularity::NO_QUANT/PER_TENSOR` 与输出 layout 处理。
[^asc-fixpipe-api]: 固定提交中的 `FixpipeConfig`、`CO2Layout`、`FixpipeParamsC310`、`dualDstCtl` 与 `isToUB` 定义。
