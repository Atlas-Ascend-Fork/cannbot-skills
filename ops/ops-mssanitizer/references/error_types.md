# mssanitizer 错误类型详解

本文档详细说明 mssanitizer 检测到的各类错误类型、原因分析和修复建议。

---

## 错误类型分类

### 🔴 严重错误（立即修复）

#### 1. illegal_write（非法写入）

**定义**：向非法内存地址写入数据

**日志示例**：
```
====== ERROR: illegal write of size 8
======    at 0x12c000000010 on GM
======    in block 0 on device 0
======    serialNo:22
```

**常见原因**：
- DataCopy 参数错误（AscendC）
- tl.store() 越界写入（Triton）
- 指针计算错误
- 缓冲区大小不足

**修复步骤**：
1. 根据 serialNo 定位代码位置（编译时添加 `-g`）
2. 检查地址计算逻辑
3. 检查数据搬运参数（元素数量 vs 字节数）
4. 检查缓冲区大小分配

#### 2. illegal_read（非法读取）

**定义**：从非法内存地址读取数据

**日志示例**：
```
====== ERROR: illegal read of size 4
======    at 0x12c000000008 on GM
======    in block 0 on device 0
======    serialNo:15
```

**常见原因**：
- tl.load() 越界读取（Triton）
- mask 参数缺失或错误（Triton）
- DataCopy 源地址错误（AscendC）

**修复步骤**：
1. 添加边界检查和 mask 保护
2. 检查 offset 计算
3. 确认数据范围

#### 3. illegal_free（非法释放）

**定义**：释放未分配或已释放的内存

**日志示例**：
```
====== ERROR: illegal free
======    at address 0x12c000000010
======    Address was not allocated
```

**常见原因**：
- AllocTensor/FreeTensor 不配对（AscendC）
- 重复释放同一内存
- 越界写入破坏内存管理结构（关联 illegal_write）

**修复步骤**：
1. 检查 AllocTensor/FreeTensor 配对
2. 先修复关联的 illegal_write
3. 检查内存生命周期管理

---

### 🟡 中等错误（尽快修复）

#### 4. memory_leak（内存泄漏）

**定义**：分配的内存未正确释放

**日志示例**：
```
====== ERROR: memory leak of 1024 bytes
======    at address 0x12c000000100
======    Block was allocated at serialNo:10
```

**分阶段检测策略**：

**步骤 1：检测 Device 接口泄漏**
```bash
mssanitizer -t memcheck --check-device-heap=yes --leak-check=yes -- python3 test.py
```

- 无泄漏输出 → 泄漏发生在 Device 侧应用
- 有泄漏输出 → 继续步骤 2

**步骤 2：检测 AscendCL 接口泄漏**
```bash
mssanitizer -t memcheck --check-cann-heap=yes --leak-check=yes -- python3 test.py
```

**常见原因**：
- AllocTensor 后忘记 FreeTensor（AscendC）
- Python 对象未释放（Triton）
- 框架缓存未清理

**修复步骤**：
1. 检查所有内存分配是否有对应释放
2. 检查异常路径的内存释放
3. 检查全局缓存的生命周期

---

### 🟢 轻微错误（建议修复）

#### 5. uninitialized_read（未初始化内存读取）

**定义**：读取未初始化的内存值

**日志示例**：
```
====== ERROR: use of uninitialized value
======    at address 0x12c000000010
======    in block 0 on device 0
```

**常见原因**：
- 使用未初始化的局部变量
- buffer 未正确初始化
- boundary_check 场景（Triton 特有）

**修复步骤**：
1. 初始化所有局部变量
2. 使用 Duplicate 初始化 buffer（AscendC）
3. Triton boundary_check 场景需结合代码逻辑判断是否为真实问题

#### 6. race_condition（数据竞争）

**定义**：多个 block 同时访问同一内存位置

**日志示例**：
```
====== ERROR: Race detected
======    Two different blocks accessed the same address
======    Block 0 wrote at serialNo:15
======    Block 1 wrote at serialNo:16
```

**常见原因**：
- 多核并行写入同一位置（Triton）
- 缺少 atomic 操作（Triton）
- 同步机制缺失（AscendC）

**修复步骤**：
1. 使用 atomic 操作（Triton: tl.atomic_add）
2. 添加同步机制（AscendC: sync_all）
3. 重新设计并行策略避免冲突

---

## 错误关联分析

### illegal_free + illegal_write

**典型场景**：越界写入破坏内存管理结构

**修复顺序**：
1. 先修复 illegal_write（根本原因）
2. 再修复 illegal_free（次生问题）

### memory_leak + illegal_access

**典型场景**：内存泄漏导致地址空间耗尽，后续访问越界

**修复顺序**：
1. 先修复 memory_leak
2. 再检查 illegal_access 是否消失

---

## 日志字段详解

| 字段 | 说明 | 用途 |
|------|------|------|
| `at 0x... on GM` | 全局内存地址 | 分析内存布局 |
| `in block N on device M` | Block 和 Device ID | 定位并行问题 |
| `serialNo:N` | 指令序列号 | 定位代码位置（需 `-g` 编译） |
| `size N` | 访问字节数 | 判断数据类型 |
| `Address was not allocated` | 内存分配状态 | illegal_free 定位 |
| `Block was allocated at` | 分配点信息 | memory_leak 定位 |

---

## 问题定位技巧

### 1. 使用 serialNo 定位源码

**编译选项**：
```bash
# AscendC 算子编译
cmake -DCMAKE_CXX_FLAGS="-g -O0" ..

# Triton 算子（JIT 自动处理）
# 无需手动设置
```

**定位步骤**：
1. 记录日志中的 serialNo（如 serialNo:22）
2. 查看编译生成的汇编或中间代码
3. 对应到源码位置

### 2. 地址分析

**GM 地址范围**：
- AscendC: 通常为 0x12c000000000 ~ 0x12c0ffffffff
- Triton: 依框架分配而定

**分析步骤**：
1. 记录错误地址（如 0x12c000000010）
2. 计算与预期地址的偏差
3. 判断越界方向和范围

### 3. 大小分析

| size 值 | 可能数据类型 |
|---------|------------|
| 1 | uint8, int8 |
| 2 | half, float16 |
| 4 | float, float32, int32 |
| 8 | double, float64, int64 |

**分析用途**：
- 判断访问的数据类型
- 检查是否为批量访问（size * 元素数量）

---

## 修复验证流程

### 修复后重新检测

```bash
# AscendC 算子
bash run_unified_mssanitizer.sh <project> <cann_root>

# Triton 算子
bash run_unified_mssanitizer.sh <script.py> <cann_root>
```

### 验证标准

| 检测项 | 验证条件 | 说明 |
|--------|---------|------|
| memcheck | 退出码为0且结构化错误条目为0 | 无非法访问和泄漏 |
| racecheck | 退出码为0且结构化错误条目为0 | 无数据竞争 |
| initcheck | 退出码为0且无真实错误条目 | boundary_check报告仍需结合代码判断 |
| synccheck | 退出码为0且结构化错误条目为0 | 无同步问题 |

非零退出码表示检测未完成。即使日志为空，也不能将该检测项标记为通过。

### 汇总报告评级

| 评级 | 条件 | 建议 |
|------|------|------|
| 🟢 **良好 (GOOD)** | 所有检测项无错误 | 算子内存安全性良好 |
| 🟡 **中等 (MODERATE)** | 存在少量内存泄漏 | 建议尽快修复泄漏 |
| 🔴 **严重 (CRITICAL)** | 存在非法访问、越界写入 | 必须立即修复 |

---

## 参考资源

- [AscendC 最佳实践](ascendc_best_practices.md)
- [Triton 最佳实践](triton_best_practices.md)
- [CANN mssanitizer 官方文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/devtoolquickstart/atlasopdev_16_0006.html)
