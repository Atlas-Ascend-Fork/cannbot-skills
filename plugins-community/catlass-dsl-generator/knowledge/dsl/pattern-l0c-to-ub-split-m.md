---
type: CATLASS DSL Programming Concept
title: L0C 经 SPLIT_M 直通 UB
description: 累加结束后用 CopyL0C2DstParams(SPLIT_M) 一次搬进两个 AIV 的 UB，中间结果不落 GM。
tags: [catlass-dsl, l0c, fixpipe, split-m, unit-flag, nd2nz]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-10-23T15:45:00+08:00'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-10-23T15:45:00+08:00'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: CopyL0C2DstParams 与 L0C2UBMode
dsl_version: ascend-catlass-dsl 2.1.0
arch: [c310]
---

# 接口与概念

矩阵结果在 L0C 累加完成后，可以经 FixPipe 直接写进 AIV 的 UB；用 `SPLIT_M` 模式时，
一次调用会把 `[BM, BN]` 按行对半切开，分别落到两个 AIV 各自的 UB 位置。这样中间
结果**从不落地到 GM**，省掉一次全量写回与再读。

# 用法

```python
tla.copy(
    ub_dst, l0c,
    tla.params.CopyL0C2DstParams(
        unit_flag=3,
        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
    ),
)
```

两个 AIV 传入**同一个 UB 指针**，硬件各自取自己那半行。

# 代码模式

- 搬运前必须有"L0C 已就绪"的 flag（CUBE→FIX），搬运后置"UB 可用"（FIX→VECTOR）。
- MMAD 的最后一个 K 块要带 `unit_flag=3`，让累加链在此收口；否则 FixPipe 可能读到
  未完成累加的部分结果。
- K 方向尾块由 ND2NZ 补零，因此 `[BM, BN]` 的 full tile 语义在 mmad 与 fixpipe 两侧一致。

# 约束

- `unit_flag` 必须在"最后一个 K 子块"上置位，与 fixpipe 的 tile 语义对齐；中间块保持 0。
- BM 必须是偶数（按行对半切），且两个 AIV 的 UB 视图起始偏移一致。
- 一个 L0C 缓冲被下一次 MMAD 覆盖前，必须确认两个 AIV 都已消费（双 flag 或双向交棒）。

# 失败表现

- unit_flag 未在最后一块置位：尾块数据缺失或残留上一轮结果，表现为边缘行/列错。
- 只等了一个 AIV 就复用 L0C：另一路读到被覆盖的数据，且错误随轮次漂移。

# 验证方法

把两个 AIV 的 UB 半区分别 dump 出来与 host 侧 `[0:half]`、`[half:]` 对拍；
再用非 2 的幂的 M/N 验证尾块。

[^api]: 固定提交的 CopyL0C2DstParams 实现。
