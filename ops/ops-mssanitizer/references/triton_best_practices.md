# Triton 算子 mssanitizer 最佳实践

本文档提供 Triton 算子内存检测的最佳实践、常见错误模式和实战经验。

---

## Triton 检测特点

### 与 AscendC 的关键差异

| 特性 | Triton 算子 | AscendC 算子 |
|------|-----------|-------------|
| **开发语言** | Python + JIT 编译 | C++ 编译 |
| **测试载体** | Python 脚本或 pytest | C++ 可执行文件或 Python 脚本 |
| **环境变量** | 需关闭内存池、强制重编译 | 无特殊要求 |
| **kernel-name** | 推荐（避免框架算子干扰） | 可选 |
| **崩溃影响** | 结果不可信 | 结果可信（除非程序异常） |

---

## 必需环境变量（Top 1 关键）

### 为什么必须设置这些变量？

**PyTorch 框架的内存池会干扰检测结果**

```bash
# ❌ 错误：未关闭内存池
mssanitizer -t memcheck -- python3 test_triton.py
# 检测结果可能全是误报

# ✅ 正确：关闭内存池
PYTORCH_NO_NPU_MEMORY_CACHING=1 \
mssanitizer -t memcheck -- python3 test_triton.py
```

### 三大必需变量

```bash
export PYTORCH_NO_NPU_MEMORY_CACHING=1  # 关闭 PyTorch 内存池（必须）
export TRITON_ALWAYS_COMPILE=1          # 强制 JIT 重编译（推荐）
export TRITON_ENABLE_SANITIZER=1        # 启用全量检测能力（必须）
```

**设置原因**：

| 变量 | 设置原因 |
|------|---------|
| `PYTORCH_NO_NPU_MEMORY_CACHING=1` | PyTorch 框架默认使用内存池管理 GM 内存，会干扰内存检测 |
| `TRITON_ALWAYS_COMPILE=1` | 防止未重新编译的算子缓存影响检测结果 |
| `TRITON_ENABLE_SANITIZER=1` | 启用 Triton 编译器的全量检测能力 |

---

## 测试用例选择（Top 2 关键）

### 方式一：使用已有 pytest 测试（强烈推荐）

```bash
PYTORCH_NO_NPU_MEMORY_CACHING=1 TRITON_ALWAYS_COMPILE=1 TRITON_ENABLE_SANITIZER=1 \
    mssanitizer -t memcheck --kernel-name=<kernel_name> \
    -- pytest -sv tests/test_xxx.py::test_case_name
```

**推荐原因**：
- ✅ 已有测试经过验证，输入数据正确
- ✅ pytest 可精确指定单个测试用例
- ✅ 避免自行构造用例引入额外问题

### 方式二：自行构造测试脚本（谨慎）

**必须严格参照已有测试的输入生成逻辑**：

```python
# ❌ 错误：随意构造小规模用例
B = 1
T = 128
x = torch.randn(B, T, 512, device='npu')
# 可能触发 kernel 边界问题导致崩溃

# ✅ 正确：复用已有测试的参数范围
# 参考已有测试中的参数：
# - B (batch size)：如 1-16
# - T (sequence length)：如 128-2048
# - H (head num)：如 8-32
# 参考已有测试的数据初始化：
x = F.normalize(torch.randn(B, T, 512, device='npu'), dim=-1)
```

**⚠️ 实战经验**：自行构造的小规模用例（如 B=1, T=128）可能触发 kernel 的边界问题导致崩溃，检测结果**不完整且不可信**。

---

## 使用 --kernel-name（Top 3 关键）

### 为什么推荐使用？

**避免 PyTorch 框架算子的干扰**

```bash
# ❌ 错误：不指定 kernel-name
mssanitizer -t memcheck -- python3 test_triton.py
# 会检测所有 kernel，包括 PyTorch 框架内部的算子

# ✅ 正确：指定 Triton kernel 名称
mssanitizer -t memcheck --kernel-name=triton_add -- python3 test_triton.py
# 只检测指定的 Triton 算子
```

### 如何确定 kernel-name？

**kernel-name 对应 Triton kernel 函数名**

```python
@triton.jit
def triton_add_kernel(  # ← 这是 kernel-name
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # kernel 内部逻辑

# ❌ 错误：使用 kernel 内部的子函数名
@triton.jit
def helper_function(...):  # 这不是有效的 kernel-name
    ...

# ✅ 正确：使用具有 host 侧 launch 调用点的 kernel 入口函数
def triton_add(x, y):
    output = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    triton_add_kernel[grid](  # ← host 侧 launch 调用点
        x, y, output,
        x.numel(),
        BLOCK_SIZE=1024,
    )
    return output
```

**注意**：`--kernel-name` 必须是具有 host 侧 launch 调用点的 kernel 入口函数，而非 kernel 内部被 `@triton.jit` 装饰的封装子函数。

---

## 常见错误模式

### 1. tl.load() 越界读取（Top 1 Triton 错误）

#### 错误症状

```
====== ERROR: illegal read of size 4
======    at 0x12c000000010 on GM
```

#### 错误原因

**mask 参数缺失或计算错误**

```python
# ❌ 错误：缺少 mask，可能越界访问
offsets = tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offsets)

# ✅ 正确：添加 mask 保护
offsets = tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
```

#### 修复步骤

1. 检查所有 tl.load() 调用
2. 确认每个 load 都有 mask 参数
3. 检查 mask 计算逻辑是否正确

---

### 2. tl.store() 越界写入（Top 2 Triton 错误）

#### 错误症状

```
====== ERROR: illegal write of size 4
======    at 0x12c000000010 on GM
```

#### 错误原因

**store 操作越界**

```python
# ❌ 错误：缺少 mask
tl.store(output_ptr + offsets, result)

# ✅ 正确：添加 mask 保护
mask = offsets < n_elements
tl.store(output_ptr + offsets, result, mask=mask)
```

---

### 3. 多核数据竞争（Top 3 Triton 错误）

#### 错误症状

```
====== ERROR: Race detected
======    Two different blocks accessed the same address
```

#### 错误原因

**多个 program 同时写入同一内存位置**

```python
# ❌ 错误：多核并行写入同一位置
tl.store(output_ptr + offsets, value, mask=mask)

# ✅ 正确：使用 atomic 操作避免竞争
tl.atomic_add(output_ptr + offsets, value, mask=mask)
```

#### 修复步骤

1. 检查所有 tl.store() 是否可能写入同一位置
2. 使用 atomic 操作替代（tl.atomic_add, tl.atomic_max 等）
3. 重新设计并行策略避免冲突

---

## 特有注意事项

### 1. 程序崩溃时结果不可信（Top 1 Triton 特有）

**问题**：检测过程中程序崩溃（EZ9999、CCU error、aicore error）

**影响**：mssanitizer 只能采集到崩溃前的部分信息，检测结果**不完整且可能不准确**

**排查步骤**：
1. 确认是否为用例构造问题（参数过小导致 kernel 边界异常）
2. 确认是否为 kernel 本身存在严重 bug（如非法地址访问）
3. 修复后重新执行检测

**汇总报告会标注**：执行状态（正常完成 / 异常退出）

---

### 2. initcheck 对 boundary_check 场景的大量报错

**问题**：使用 `tl.make_block_ptr` + `boundary_check` 产生大量 uninitialized read 报告

**原因**：boundary_check 触发时越界区域用未初始化值填充

**判断方法**：结合代码逻辑判断这些未初始化值是否参与了后续有效计算

```python
# 使用 boundary_check 的场景
x_block = tl.make_block_ptr(...)
x = tl.load(x_block, boundary_check=[0])  # 可能触发 uninitialized read

# 判断方法：检查这些值是否参与有效计算
# 如果只是 padding 区域，不影响计算结果，可忽略
```

---

### 3. 检测可能影响数值精度

**问题**：开启检测后，部分算子精度断言失败

**原因**：检测工具注入对计算过程的副作用

**处理方法**：
- 如遇到精度断言失败但无内存错误报告
- 可忽略精度差异，关注检测工具报告的内存问题

---

### 4. 日志行号信息可能缺失

**问题**：错误定位显示为 `code in :0`，无源码行号

**解决方法**：
- Triton JIT 编译自动处理调试信息
- 通过 serialNo 指令序列号辅助定位
- 查看编译生成的中间代码

---

### 5. 日志目录权限要求

**问题**：mssanitizer 要求日志目录权限不高于 0750

**解决方法**：
```bash
chmod 750 <log_directory>
```

---

## 检测流程最佳实践

### 方式一：pytest（推荐）

```bash
# 步骤 1：确认测试用例
pytest --collect-only tests/test_xxx.py  # 查看所有测试用例

# 步骤 2：执行检测
SKILL_DIR="<skill_install_path>"

bash "${SKILL_DIR}/scripts/triton/run_mssanitizer_triton.sh" \
    tests/test_xxx.py \
    <cann_root> \
    --kernel-name <kernel_name>

# 步骤 3：查看报告
cat tests/mssanitizer_logs/mssanitizer_summary_*.md
```

### 方式二：自定义脚本

```bash
# 步骤 1：严格参照已有测试构造脚本

# 步骤 2：执行检测
bash run_unified_mssanitizer.sh test_triton_custom.py <cann_root>

# 步骤 3：查看报告
cat mssanitizer_logs/mssanitizer_summary_*.md
```

---

## 常见问题排查

### Q: 检测结果全是误报？

**A: 检查是否关闭了内存池**
```bash
# 确认已设置
export PYTORCH_NO_NPU_MEMORY_CACHING=1
```

### Q: 程序执行崩溃怎么办？

**A: 检测结果不可信，需先修复崩溃问题**
1. 检查用例参数是否合理（参考已有测试）
2. 检查是否存在严重的非法访问
3. 修复后重新检测

### Q: 如何只检测特定 kernel？

**A: 使用 --kernel-name 参数**
```bash
mssanitizer -t memcheck --kernel-name=triton_add -- python3 test.py
```

### Q: 检测很慢怎么办？

**优化方法**：
- 使用 `--kernel-name` 指定特定 kernel
- 使用 `--block-id` 只检测特定 block
- 减小测试数据的 shape

### Q: boundary_check 场景大量报错？

**A: 结合代码逻辑判断**
- boundary_check 触发时越界区域用未初始化值填充
- 如果只是 padding 区域，不影响计算结果，可忽略

---

## 实战经验总结

### ✅ 必须做

1. **关闭内存池**：`PYTORCH_NO_NPU_MEMORY_CACHING=1`
2. **启用全量检测**：`TRITON_ENABLE_SANITIZER=1`
3. **使用已有 pytest 测试**：避免自行构造用例
4. **使用 --kernel-name**：避免框架算子干扰

### ❌ 避免做

1. **随意构造小规模用例**：可能触发边界问题导致崩溃
2. **忽略程序崩溃**：崩溃后检测结果不可信
3. **不设置必需环境变量**：检测结果全是误报

---

## 参考资源

- [错误类型详解](error_types.md)
- [mssanitizer 基础案例 — Triton 算子检测场景](https://www.hiascend.com/document/detail/zh/mindstudio/2600/msOT/Operatordevelopmenttools/docs/zh/best_practices/mssanitizer_basic_cases.md)
- [AscendNPU IR 调试调测指南](https://triton-ascend.readthedocs.io/)