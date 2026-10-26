---
type: CATLASS DSL Programming Concept
title: 单个 cross flag 的双向互斥用法
description: 用一个 cross flag 在 CUBE 与 VECTOR 之间交替传递所有权，代替 ready/free 两个单向 flag。
tags: [catlass-dsl, cross-flag, synchronization, mixed-kernel]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-10-23T15:40:00+08:00'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-10-23T15:40:00+08:00'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: cross_flag 的方向与 aiv_id 语义
dsl_version: ascend-catlass-dsl 2.1.0
arch: [c310]
---

# 接口与概念

缓冲在两核之间来回传递时，常见写法是一对单向 flag（ready / free）。另一种更省资源
的写法是**只用一个 cross flag，让所有权在两侧交替**：

    VECTOR 初始化 -> set
    FIX    等待 VECTOR 交接 -> 使用 -> set 交回
    VECTOR 等待 FIX 交接 -> 使用 -> set 交回

每次 set 都是一次"交棒"，wait 是"接棒"，因此不需要为两个方向各建一个 flag。

# 用法

```python
exchange = tla.cross_flag("exchange")
with tla.vector():
    tla.cross_core_set_flag(exchange, tla.arch.VECTOR)      # 开场放行

tla.cross_core_wait_flag(exchange, tla.arch.FIX)            # 生产侧接棒
tla.copy(ub_dst, l0c, tla.params.CopyL0C2DstParams(
    unit_flag=3, l0c2ub_mode=tla.arch.L0C2UBMode.SPLIT_M))
tla.cross_core_set_flag(exchange, tla.arch.FIX)             # 交棒

tla.cross_core_wait_flag(exchange, tla.arch.VECTOR)         # 消费侧接棒
tla.cross_core_set_flag(exchange, tla.arch.VECTOR)          # 用完交回
```

# 代码模式

见上文：开场由消费侧 set 一次，之后 set/wait 严格交替，任何一侧都不得连续两次 set。

# 约束

- set 与 wait 必须严格配对交替：多一次 set 会让对侧永不阻塞，少一次会死锁。
- 两个执行引擎都必须各有一次 wait 与一次 set。
- 与按 aiv_id 定向的 flag 区别：本用法适用于"两个引擎独占同一缓冲"；若多个 AIV
  需要各自独立信号，仍应使用定向 flag。

# 失败表现

- 多 set / 少 wait：对侧提前读到未写完的数据，数值错乱但不报错。
- 少 set：对侧永久阻塞，表现为挂死或超时。

# 验证方法

把 set/wait 序列画成两列（FIX / VECTOR）逐行核对，确保每行只有一个动作且交替；
最小 shape 连跑两轮，确认第二轮不挂。

[^api]: 固定提交的 cross_flag 实现与方向语义。
