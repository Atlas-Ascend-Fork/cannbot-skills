---
type: CATLASS DSL Programming Concept
title: 动态控制流中的 Tensor Binding
description: 处理 dynamic-if、tla.range 和 outlined region 中 Tensor descriptor、数值状态与 loop-carried binding 的合法构造方式。
tags: [catlass-dsl, dsl, dynamic-if, tensor-descriptor, tensor-binding, loop-carried, lowering]
status: stable
generated: {by: human:maintainer, at: '2026-09-20T00:00:00-07:00'}
verified:
  - {by: human:maintainer, at: '2026-09-20T00:00:00-07:00'}
sources:
  - id: syntax
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/docs/en/core_concepts/syntax_guide.md
    title: DSL Python Syntax Constraints
  - id: tests
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/tests/test_control_flow_analysis.py
    title: Control-flow analysis tests
operator_families: [elementwise, matmul, sparse-attention]
arch: [c310]
---

# 接口与概念

TLA 动态条件会 lower 为 IR region。分支或循环后仍存活的值必须在所有路径上定义，并能形成
类型兼容的 SSA/loop-carried 状态。Tensor descriptor 同时包含 pointer、layout 和 coordinate，
比标量更难合法合并。[^syntax][^tests]

# 用法

运行时变化的普通标量先用 `tla.as_numeric` 建立动态数值 binding。Tensor view 尽量在动态
region 外预构造；动态分支只选择执行哪次 copy/compute，不把不同 descriptor 赋给同一变量。

# 代码模式

避免：

```python
if dynamic_condition:
    source = tla.make_tensor(ptr_a, layout_a)
else:
    source = tla.make_tensor(ptr_b, layout_b)
tla.copy(dst, source)
```

优先：

```python
source_a = tla.make_tensor(ptr_a, layout_a)
source_b = tla.make_tensor(ptr_b, layout_b)
if dynamic_condition:
    tla.copy(dst, source_a)
else:
    tla.copy(dst, source_b)
```

动态循环需要使用的标量在循环前初始化；循环 induction variable 不在循环后复用，所需结果
通过显式 loop-carried 标量输出。

# 约束

- 不在仅部分动态路径中首次定义后续要用的变量。
- 不让同名 binding 在 Tensor、标量和 Python 对象之间改变类别。
- 不把 `tla.as_numeric` 当作 Tensor descriptor 的强制合并工具。
- `range_constexpr` 只适合编译期展开，不能承载运行时边界。
- outlined DSL region 内外的 live value 必须符合前端允许的携带类型。

# 失败表现

- uninitialized variable：存在未定义路径。
- incompatible dynamic-if carried value：分支结果类别、dtype 或 descriptor 不兼容。
- dynamic for-loop carried tensor descriptor：Tensor view 被错误带过循环边界。
- induction variable reused after loop：循环变量逃逸。

# 验证方法

先运行 Python/AST 预处理和 lowering，不上板；用零次、一次和两次循环覆盖 carried state。
再 dump IR，确认分支内执行 copy，而不是试图合并两个 Tensor descriptor。

[^syntax]: 固定提交语法指南对动态控制流、循环和禁止语句的约束。
[^tests]: 固定提交控制流测试对分支合并、初始化和 loop-carried 状态的覆盖。
