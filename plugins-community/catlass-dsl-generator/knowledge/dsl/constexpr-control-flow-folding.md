---
type: CATLASS DSL Programming Concept
title: Constexpr 折叠与动态 if 初始化约束
description: tla.Constexpr 参数、dataclass 属性访问在 tla.range 循环体内的静态折叠边界、动态 if 的变量初始化检查，以及向量循环内动态 if 的性能代价。
tags: [catlass-dsl, control-flow, constexpr, folding, dynamic-if, preinitialize]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-09-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-09-10T00:00:00Z'}
sources:
  - id: frontend
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/base_dsl/ast_preprocessor.py
    title: Python 前端的静态控制流判定与动态 if 初始化检查
  - id: kernel
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/block_sparse_attention_grad/op_kernel/block_sparse_attention_grad.cpp
    title: 以 `if tilingData.tndMode:` 属性访问 + 预初始化的实际用法
operator_families: [sparse-attention, flash-attention]
arch: [c310]
---

# 接口与概念

前端用 `_is_static_python_if_test` 判定 `if` 是否可在 Python 侧静态折叠，只识别三种测试：
`ast.Constant` 的 bool 字面量、模块级（非 local）bool 的 `ast.Name`、以及 `tla.const_expr(...)`
调用。`tla.Constexpr` 参数、以及 dataclass 字段的属性访问（如 `tilingData.tndMode`）都不被
识别为静态，因而在 `tla.range` 循环体内会 lower 成动态 `if`。[^frontend]

动态 `if` 受 `_reject_unsupported_dynamic_if_new_defs` 约束：then/else 分支内「首次定义」且
「if 之后被 load」的变量，必须在该 `if` 之前先初始化，否则抛
`dynamic Tla if requires variables used after the branch to be initialized before the if`。[^frontend]

# 用法

## 属性访问在循环内是动态 if，须预初始化

`tilingData.X`（Attribute 访问）在 `tla.range` 循环体内不会被折叠成 Python 分支，按动态 if
处理。分支内首定义、分支后使用的变量要在 if 前给默认值；mask 寄存器用恒假比较构造全假默认值：

```python
for task_id in tla.range(0, task_num, 1):
    kv_offset = 0
    q_offset = 0
    last_q_total_seqlen = 0
    q_cur_seqlen = 0
    if tilingData.tndMode:                     # 动态 if
        q_cur_seqlen = q_cu_seqlen[batch_idx, 0].to(tla.Int32)
        last_q_total_seqlen = tla.as_numeric(0)
        for bi_q in tla.range(0, batch_idx, 1):
            last_q_total_seqlen = last_q_total_seqlen + q_cu_seqlen[bi_q, 0].to(tla.Int32)
    else:
        q_cur_seqlen = Q_SEQ_LEN
    # q_cur_seqlen 在分支后使用，须在上面先初始化为 0
```

mask 寄存器（`tla.cmp`/`tla.full`）的预初始化需要类型匹配的「全假/全零」值，
用 `tla.cmp(reg_lane, 0, "lt", mask=pregFull)`（`reg_lane` 是 `tla.arange`，恒 `reg_lane < 0`
为假），不能用 Python 字面量 `tla.cmp(0, 0, ...)`（`lhs` 需要 vector_ssa）。[^kernel]

# 代码模式

## 编译期开关：属性访问 vs 模块全局 bool vs const_expr

`tla.range` 循环体内的 `if` 若要让条件在 Python 侧折叠，只能走 `tla.const_expr(...)`；把
`tilingData.X` 重命名为局部变量（`LOCAL = tilingData.X; if LOCAL:`）也不能恢复静态折叠，
因为该名字已进入 local scope，`_is_static_python_if_test` 对 local `ast.Name` 不折叠。[^frontend]

## 向量循环内动态 if 的性能代价

在 `tla.vec.func` 的按行 `for` 中写「运行时才能确定」的 `if`，即使条件几乎总是 False，
该段向量循环仍可能慢约 3 倍（项目实测观测，见 optimization 概念）。需要编译期分支时，
把条件换成可静态折叠的 `tla.const_expr(...)` 或模块级 bool，避免在向量循环体里留动态 if。

# 约束

- `tla.Constexpr` 参数与 dataclass 属性访问在循环体内都是动态 if，不享受 Python 展开的零成本。
- 动态 if 分支内新定义且分支后使用的名字必须预初始化；错设为 Python 字面量会因类型不匹配
  （`expected vector_ssa, got int`）在 lowering 阶段失败。
- 预初始化的目的是满足静态检查；运行时未走该分支时值不被使用，但仍需类型正确。

# 失败表现

- `SyntaxError: dynamic Tla if requires variables ... initialized before the if: <name>`：分支内
  首定义变量未预初始化。
- `TlaCoreAPIError: tla.cmp: invalid argument 'lhs' (position 0): expected vector_ssa, got int`：
  用 Python 字面量构造 mask 默认值。
- 3× 量级的向量循环慢：向量循环体内保留了运行时动态 if。

# 验证方法

静态检查：对 decorated kernel 检查 `if` 测试是否可折叠、分支名字是否预初始化。运行时验证：
编译通过、正确性通过，并用 profiler 对比「动态 if」vs「const_expr 折叠」的向量段耗时。
[^frontend][^kernel]

[^frontend]: 固定提交的 ast_preprocessor 静态控制流判定与动态 if 初始化检查实现。
[^kernel]: 固定提交的 block_sparse_attention_grad 中属性访问 + 预初始化/mask 恒假默认值的实际用法。