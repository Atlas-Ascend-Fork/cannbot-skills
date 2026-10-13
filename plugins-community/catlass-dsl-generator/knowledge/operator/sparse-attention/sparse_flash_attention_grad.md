---
type: CATLASS DSL Operator Example
title: Sparse Flash Attention Grad Arc35
description: topk 稀疏注意力反向（SFAG）：逐行 gather 选中 KV、五路加两路 rope MMAD、一轮延迟的 AIC/AIV 流水、位置序 scratch 加 token 行原子散写。
tags: [catlass-dsl, operator, sparse-attention, sparse-flash-attention-grad, sfag, backward, topk-gather, rope, mixed, atomic-add, ascend-950]
status: stable
generated: {by: human:codex, at: '2026-09-28T01:00:00Z'}
verified:
  - {by: human:codex, at: '2026-09-28T01:00:00Z'}
sources:
  - id: guide
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/sparse_flash_attention_grad/README.md
    title: SparseFlashAttentionGrad 接口、topkIndices gather 语义与阶段公式
  - id: entry
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/sparse_flash_attention_grad/op_kernel/arch35/sparse_flash_attention_grad_entry_regbase.h
    title: MIX 1:2 任务类型与 pre/主循环/post 三段调用链
  - id: pipeline
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/sparse_flash_attention_grad/op_kernel/arch35/sparse_flash_attention_grad_kernel.h
    title: ProcessUnDeter/ProcessDeter 的一轮与两轮延迟流水、task 枚举与槽位
  - id: flags
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/sparse_flash_attention_grad/op_kernel/arch35/sparse_flash_attention_grad_common.h
    title: AIC/AIV 各阶段交接的 cross flag 编号
  - id: cube
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/sparse_flash_attention_grad/op_kernel/arch35/sparse_flash_attention_grad_block_cube.h
    title: 五路 MMAD 与 Q/K/dY/L0C 的缓冲 selector
  - id: vec
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/sparse_flash_attention_grad/op_kernel/arch35/sparse_flash_attention_grad_block_vec.h
    title: GatherKV、ProcessVec1-4 与 ScatterAdd 的向量侧实现
  - id: pre
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/sparse_flash_attention_grad/op_kernel/arch35/sparse_flash_attention_grad_pre_regbase.h
    title: 主循环前 dq/dk/dv workspace 清零与全核同步
  - id: post
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/sparse_flash_attention_grad/op_kernel/arch35/sparse_flash_attention_grad_post_regbase.h
    title: FP32 workspace 的缩放、cast 与输出回写
operator_families: [sparse-attention, attention]
arch: [c310, ascend-950]
---

# 接口与概念

## 算子算法

输入为 `query/key/value/sparse_indices/d_out/out/softmax_max/softmax_sum/query_rope/key_rope`
（TND 另需 `actual_seq_qlen/actual_seq_kvlen`），输出 `d_query/d_key/d_value/d_query_rope/d_key_rope`。
`sparse_indices` 是逐 `(t1, n2)` 行的 topk 位置索引：先按 `selectedBlockSize` 对 key/value 做
gather 重排得到 selectedKey/selectedValue，再在这份重排结果上算注意力的反向。[^guide]

每个 `(t1, n2)` 的有效 KV 数是变长的：vendor 在每轮用 `GetActualSelCount` 取该组实际选中的
block 数，收尾与 task 边界也随之变化。[^pipeline] 它等价于逐行
`nuse = klen - qlen + (i - qb) + 1` 并截断 TOPK；`nuse <= 0` 的行 `d_query` 为 0 且不贡献
dK/dV，chunk 内 `pos >= nuse` 的位置既不 gather 也不散写。

一个 128 位置的 chunk 上的计算链（`D = rowsum(dO*O)` 在 AIV 侧先算）：

```text
S      = Q @ K^T + Q_rope @ K_rope^T
P      = exp(scale * S - softmax_max) / softmax_sum
dP     = dY @ K^T
dS     = P * (dP - D)
dV     = P^T @ dY
dQ     = dS @ K            dQ_rope = dS @ K_rope
dK     = dS^T @ Q          dK_rope = dS^T @ Q_rope
```

路径选择：BSND 与 TND 复用同一 kernel，fp16 与 bf16 复用同一 kernel，`G = N1 / N2` 个 q head
在 MMAD 的 M 维一次算完；`sparse_block_size != 1`、`sparse_mode != 3`、`attention_mode != 2`、
`D_noPE != D_v`、`N1 % N2 != 0` 一律报错。[^guide]

# 用法

## 分核策略与基本块切分

kernel 是 MIX 类型（1 个 AIC 配 2 个 AIV 组成一个 block），按 28 个 block 交付。[^entry]

task 是“一个 query 行 × 一个 KV head”，task 内部沿位置序 chunk 前进（每轮 S2C=128 个位置，
最多 `TOPK / S2C` 轮），轮数由该行的有效 KV 数决定。vendor 侧的枚举是
`t1Index × n2Index × blkCntOffset`（按核 grid-stride，`blkCntOffset` 以
`selectedCountOffset` 为步长），与这里的“行 × head × chunk”是同一条链。[^pipeline]

输出所有权按“谁是这个位置的唯一写者”划分：

- `d_query/d_query_rope` 是一个 task 的私有结果（该 task 已覆盖这一行的全部 KV），直接写出，
  不需要原子累加；
- `d_key/d_key_rope/d_value` 要跨 query 行汇聚，先由 cube 写成**每 block 私有**的位置序
  scratch，再由 AIV 逐 token 行原子累加进 FP32 累加器，最后在收尾阶段缩放、cast 回写。
  vendor 侧同样是 selectedK / mm4Res / mm5Res 工作区加 `ScatterAdd` 的两段式写回。[^vec][^flags]

**dQ 累加器的半区所有权**：两个 AIV sub-block 各占 dQ 累加器的一半
（`DQH = G*(D+DR)/2`），清零发生在各自**下一个 task 的 prologue**。两个 sub-block 在那一点上
没有别的手握手，所以每个 sub-block 只能读、缩放、发布**自己那一半**；`G == 2` 时切分按 head
对齐，`G == 1` 时切点落在单行 `(dQ ++ dQ_rope)` 内部。

vendor 侧同一轮内的五路 MMAD（dP、S、dK、dQ、dV）由 cube block 用 Q/K/dY/L0C 的缓冲 selector
组织，和每轮重载 L1 Q/dY 槽的做法同构。[^cube]

# 代码模式

## 数据路径与存储层级

```text
GM(Q,Q_rope,dY) -> L1           每 task 一次
GM(K,K_rope)   -> AIV gather -> gk/gkr 每 block 私有 scratch (SLOT=2)
                 -> L1 K 槽     每 chunk 一次
L1 -> L0A/L0B -> MMAD -> L0C     S/dP/dQ/dQ_rope/dK/dK_rope/dV
L0C -> (fixpipe) GM 位置序 scratch   dkpos 每 block 私有, 行 stride ACCW=D+DR+D
L0C -> UB -> L1                  p/dS 双槽, 交回 cube 的后三路
dkdv_ws (FP32) -> UB -> d_key/d_key_rope/d_value
```

位置序 scratch 与 gather scratch 都是每 block 私有的定长分区，寻址 stride 直接决定相邻 block
是否踩踏：

- gather scratch 是槽位化的（`SLOT=2` 双槽），block stride **必须含 SLOT 因子**：
  `GKB = SLOT*S2C*D`、`GKRB = SLOT*S2C*DR`；漏乘时 block b 的槽 1 与 block b+1 的槽 0 是同一段
  内存，而单 chunk（`nuse <= 128`）只用槽 0，别名不可见。
- `dkpos/dqws` 用行 stride `ACCW = D + DR + D`，同一行内依次放 dK（D）、dK_rope（DR）、dV（D）。
- dK/dK_rope/dV 的 L0C→GM 是**整 tile 带 stride 写出**（mm4 的 NDD 段、mm4r、mm5 的 NDD 段），
  不是逐行小 DMA。

vendor 侧的落盘同样分两段：selectedK 工作区承接 gather 结果，mm4/mm5 结果先落到每 block 的
工作区再由 `ScatterAdd` 散写回 token 序。[^vec][^flags]

## 流水排布、同步关系与数值精度

主循环是**一轮延迟**的 AIC/AIV 流水，槽数 `SLOT=2`：

```text
stage 0 (AIV)  gather 下一个 chunk  -> gk/gkr
stage 1 (AIC)  本轮 S / dP
stage 2 (AIV)  本轮 p / dS
stage 3 (AIC)  上一轮的 p / dS -> dV / dQ / dK
stage 4 (AIV)  上一轮的位置序 scratch -> token 行原子散写
```

cube 第 r 轮产 S/dP 的同时消费第 r-1 轮的 p/dS，所以 stage 3 的等待通常已经满足，cube 不会
空等 vector。vendor 侧同一结构表现为“Gather(t) + Mm12(t)”与“Softmax(t-1) + Mm345(t-1)”
成对执行，并按 flag 资源决定 Scatter 滞后一轮还是两轮。[^pipeline]

cross flag 与本地 flag 的分工：`f_gather`（gather 已发布）、`f_sdp`、`f_pds`、`f_task`、
`f_pos/f_pos_free`（位置序 scratch 的读写许可）负责跨核交接；`bar1/bar2` 是 `mode=0` 的全 AIV
会合，只用在两个阶段边界（收尾前的散写累加器清零发布、主循环到收尾的发布）；核内
`r1/f1`（MTE2↔MTE1）、`r2/f2`（MTE1↔CUBE）、`r3/f3`（CUBE↔FIX）串本地流水，`fld0/fld1` 与
`fst0/fst1` 让散写 slab 的填充与写出 ping-pong。vendor 的 flag 表是同一套交接的编号表，
槽位 second-copy 用 `16 + id`。[^flags]

vendor 侧同族约束：一轮延迟流水的 credit 距离等于槽数，收尾前先排空最后的 fixpipe 再发布。[^pipeline]

数值精度：`p/dS` 在 FP32 形成后按输入 dtype 舍入再进后三路 MMAD；dQ/dK 侧只应用一次
`scale`，dV 不乘；`D` 取自 `out` 输入而不是重算；FP32 散写累加器必须先清零并以全 AIV 会合
发布，收尾阶段才能回读。vendor 同样把清零放在主循环之前、把缩放与 cast 放在主循环之后。[^pre][^post]

# 约束

- 槽位化 scratch 的 block stride 必须含 SLOT 因子（`GKB`、`GKRB`），否则相邻 block 相邻槽互相覆盖。
- dQ 半区只能读、缩放、发布自己那一半；不许整行读回。
- dK/dK_rope/dV 的 L0C→GM 必须整 tile 带行 stride 写出，退回逐行会带来数量级的小 DMA。
- 单一 `@tla.kernel`，每次调用 1 次 launch；host wrapper 只读 metadata 并做纯分配。
- `mode=0` 的全 AIV 会合只用于阶段边界，不代替逐槽 cross flag。
- task 内 L1 K 双槽、p/dS 双槽、dQ 每 block 私有；跨 block 只共享 FP32 散写累加器（原子）与
  按 block 隔离的 gather/位置序 scratch。

# 失败表现

- 单 chunk（`nuse <= 128`）正确、多 chunk 错：gather scratch 的 block stride 漏乘 SLOT，相邻
  AIV block 的槽别名。单 chunk 时只用槽 0，别名不可见。
- 整行 `d_query` 偶发为 0 或半累加混合值：某个 sub-block 读了兄弟的 dQ 半区，与兄弟的清零竞态。
- 正确但每个 chunk 有上千次 512 B 级小 DMA、fixpipe 占比高：dK/dV 退回逐行写。
- 末 chunk 梯度缺失或重复计入：stage 3/4 的“上一轮”判定与该行有效 KV 数的边界不一致。
- 偶发读到上一 task 的 gather 结果：`f_gather` 的 set/wait 没有按 task 重置。
- 收尾读到半更新的累加器：阶段边界会合之后仍有人写散写累加器，或清零没有被会合发布。

# 验证方法

- 单 case 对 CPU reference 逐个输出比对，9 个 workload 批量跑；边界覆盖 BSND/TND、
  fp16/bf16、`G=1/2`、`S1 > TOPK` 与 `nuse = 0` 的空行。
- 竞态判定用连跑自差：同一 workload 连跑两遍，两遍结果必须逐元素一致（单 chunk 与多 chunk 各覆盖）。
- 计时按官方 bench 口径：`step_trace_time.Computing`，固定设备与输入数据，warmup 1 / trials 3，
  输出目录由 `--output` 指定。
- profiler 关注 `kernel_details.csv` 的 block 数与 mix block 数，以及 AIC 的 fixpipe 占比与
  AIV 的 MTE2/MTE3/scalar 占比。
- 本文只给机制、适用条件与失败指纹；本实现的实测幅度、生效范围与原始证据在
  [项目实测知识](../../learned/index.md) 的 2026-09-24 四条 sparse-attention 结论里。

[^guide]: 固定提交中 SparseFlashAttentionGrad 的接口、gather 语义与阶段公式。
[^entry]: 固定提交中 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 与 pre/主循环/post 三段调用链。
[^pipeline]: 固定提交中 `ProcessUnDeter` 的 task 枚举、`GetActualSelCount` 与 one/two-round 延迟分支。
[^flags]: 固定提交中 AIC/AIV 交接用的 cross flag 编号表。
[^cube]: 固定提交中 cube block 的五路 MMAD 与 Q/K/dY/L0C 缓冲 selector。
[^vec]: 固定提交中 `GatherKV`、`ProcessVec1-4` 与 `ScatterAdd` 的实现。
[^pre]: 固定提交中主循环前对 dq/dk/dv workspace 的清零与全核同步。
[^post]: 固定提交中对 FP32 workspace 做 `Muls(scaleValue)` + `Cast` 并回写输出的后处理。
