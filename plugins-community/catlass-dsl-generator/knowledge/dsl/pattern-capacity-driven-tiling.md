---
type: CATLASS DSL Programming Concept
title: 容量反推 tiling
description: 先枚举 L1/L0A/L0B/L0C/UB 的可用字节，再反推 tile 形状，而不是先定 tile 再撞容量。
tags: [catlass-dsl, tiling, capacity, l0b, ub]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-10-23T12:00:00Z'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-10-23T12:00:00Z'}
sources:
  - id: measure
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: Ascend950PR 片上容量
dsl_version: ascend-catlass-dsl 2.1.0
arch: [c310]
---

# 接口与概念

tile 形状应当由容量预算**反推**，而不是凭经验先选一个再试。先取片上可用字节：

```python
tla.arch.get_capacity_in_bytes()   # A5/3510 实测：L1=512KB, L0A=64KB, L0B=64KB, L0C=256KB, UB=256KB
```

再按"每块数据要驻留几次"反推。**分层思路**（优先于"单级塞满"）：

- **L1 装完整 tile**：一个 L1 半区装 A[BM, BK] + B[BK, BN]。例：BM=128、BK=256、
  BN=192 时 bf16 合计 224KB，两个半区共 448KB，落在 512KB 内，双缓冲成立。
- **L0A/L0B 装 K 方向切片**：MMA 的 K 步取 SK（如 32），则 L0B 只需 `2*BN*SK` 字节
  （例 2×192×32×2B = 24KB），可双缓冲，远小于 64KB 上限。
- **L0C 装完整输出 tile**：BM×BN fp32（例 128×192×4B = 96KB）≤ 256KB。
- **UB 装消费侧中间量**：半区行数 × BN fp32 加统计量。

原则：**不要为了让某一级"刚好填满"而牺牲流水双缓冲**；分层后每级都有余量，
比单级塞满更能掩盖搬运延迟。

# 用法

1. 列出每个片上缓冲的持有者与生命周期（L1 ping-pong 份数、L0C 是否双缓冲、UB 里同时在飞的
   中间量）。
2. 用 `元素数 × dtype 字节 × 副本数` 求和，与容量对比；瓶颈项决定 tile 的上限。
3. tile 形状取"刚好填满最紧张那级缓冲"的值，而不是整数好看的值。
4. 容量结论必须回写设计文档的契约约束清单，避免实现期重新试错。

# 约束

- L1 可用容量与实际分配存在差异：超限不会编译失败，而是在上板时触发 aicore OOB，必须实测确认。
- 本条目数值为 A5/3510 实测，其它架构按 `tla.arch.get_capacity_in_bytes` 重新取。

接口行为以固定提交的实现为准。[^measure]

[^measure]: 固定提交的源码与文档中对应实现。

# 代码模式

```python
caps = tla.arch.get_capacity_in_bytes()
need_l0b = BLOCK_H * BLOCK_V * 2          # bf16
need_l0c = BLOCK_M * BLOCK_V * 4          # fp32
need_ub  = (slots * BLOCK_M * BLOCK_V * 4) + extra
assert need_l0b <= 64 * 1024 and need_l0c <= 256 * 1024
```

# 失败表现

- 编译通过但上板触发 aicore OOB，或运行结果局部为脏数据；
- 为规避超限把 tile 缩到很小，性能骤降。

# 验证方法

打印 `get_capacity_in_bytes` 与各项需求，逐项核对；先用最小 shape 跑通，
再放大到目标 shape。容量结论必须写进设计文档的约束清单。
