# AscendC 算子 mssanitizer 最佳实践

本文档提供 AscendC 算子内存检测的最佳实践、常见错误模式和实战经验。

---

## 检测模式选择

### 工程类型判断

```
算子工程目录
├── 存在 op_graph/ 目录？ ─── 是 ──→ ops 算子仓 → C++ 模式 (推荐)
├── 存在 examples/test_geir_*.cpp？ ── 是 ──→ ops 算子仓 → C++ 模式 (GE IR)
├── 存在 examples/test_aclnn_*.cpp？ ── 是 ──→ ops 算子仓 → C++ 模式 (aclnn)
├── 存在 op_host/（无 op_graph/）？ ── 是 ──→ 自定义算子仓 → Python 模式
└── 以上均无 ──→ 需手动指定模式
```

### 模式对比

| 特性 | Python 模式 | C++ 模式 |
|------|------------|---------|
| **适用场景** | 自定义算子仓 | ops 算子仓 |
| **优势** | 覆盖多种 shape/dtype | 直接调用 kernel，更准确 |
| **劣势** | 自动生成的基础用例只适合简单签名 | 需编译 C++ 可执行文件 |
| **推荐指数** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**实战建议**：ops 算子仓优先使用 C++ 模式。

---

## 常见错误模式

### 1. DataCopy 参数错误（Top 1 常见错误）

#### 错误症状

```
====== ERROR: illegal write of size 4
======    at 0x12c000000010 on GM
======    serialNo:22
```

#### 错误原因

**误解：DataCopy 的第三个参数是字节数**

```cpp
// ❌ 错误：乘以 sizeof(T)
DataCopy(xLocal, xGm, totalLength * sizeof(T));

// ❌ 错误：乘以其他系数
DataCopy(xLocal, xGm, totalLength * 2);

// ✅ 正确：第三个参数是元素数量
DataCopy(xLocal, xGm, totalLength);
```

#### 修复步骤

1. 检查所有 DataCopy 调用
2. 确认第三个参数是元素数量，不是字节数
3. 检查源/目标地址是否正确

#### 验证方法

```cpp
// 添加调试输出
printf("DataCopy: src=%p, dst=%p, count=%d\n",
       xGm.GetPtr(), xLocal.GetPtr(), totalLength);
```

---

### 2. AllocTensor/FreeTensor 不配对（Top 2 常见错误）

#### 错误症状

```
====== ERROR: memory leak of 2048 bytes
======    at address 0x12c000000100
======    Block was allocated at serialNo:10
```

#### 错误原因

**忘记释放 Tensor 或 EnQue/DeQue 不配对**

```cpp
// ❌ 错误：忘记 FreeTensor
LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
DataCopy(xLocal, xGm, tileLength);
inQueueX.EnQue(xLocal);
// ... 计算过程
// 缺少 inQueueX.FreeTensor(xLocal);

// ✅ 正确：配对使用
LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
DataCopy(xLocal, xGm, tileLength);
inQueueX.EnQue(xLocal);
// ... 计算过程
LocalTensor<half> xDeQue = inQueueX.DeQue<half>();
// ... 使用 xDeQue
inQueueX.FreeTensor(xDeQue);
```

#### 修复步骤

1. 检查每个 AllocTensor 是否有对应的 FreeTensor
2. 检查 EnQue/DeQue 是否配对
3. 使用代码审查工具检查内存生命周期

#### 最佳实践

```cpp
// 推荐：使用 RAII 模式管理 Tensor
class TensorGuard {
public:
    TensorGuard(LocalTensor<T>& tensor, TQue& queue)
        : tensor_(tensor), queue_(queue) {}
    ~TensorGuard() { queue_.FreeTensor(tensor_); }
private:
    LocalTensor<T>& tensor_;
    TQue& queue_;
};
```

---

### 3. UB 缓冲区大小不足（Top 3 常见错误）

#### 错误症状

```
====== ERROR: UB address out of bounds
======    at address 0xffff0000
======    size 32
```

#### 错误原因

**pipe.InitBuffer 分配的空间不足**

```cpp
// ❌ 错误：未考虑临时空间需求
pipe.InitBuffer(tmpBuffer, tileLength * sizeof(T));

// ✅ 正确：根据算子需求分配足够空间
pipe.InitBuffer(tmpBuffer, (tileLength + EXTRA_SPACE) * sizeof(T));
```

#### 修复步骤

1. 检查所有 pipe.InitBuffer 调用
2. 确认缓冲区大小计算是否正确
3. 检查是否需要 32 字节对齐

#### 计算公式

```cpp
// buffer 大小 = 数据大小 + 临时空间 + 对齐 padding
int64_t bufferSize = tileLength * sizeof(T);
int64_t alignedBufferSize = ((bufferSize + 31) / 32) * 32;  // 32 字节对齐
pipe.InitBuffer(tmpBuffer, alignedBufferSize);
```

---

### 4. FP16 升精度流程错误

#### 错误症状

```
====== ERROR: use of uninitialized value
======    或精度异常
```

#### 错误原因

**FP16 数据直接参与计算（未升精度到 FP32）**

```cpp
// ❌ 错误：FP16 直接计算
LocalTensor<half> xLocal = inQueueX.DeQue<half>();
LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
Add(yLocal, xLocal, weightLocal, tileLength);  // FP16 直接计算，精度不足

// ✅ 正确：升精度到 FP32 计算
LocalTensor<half> xLocalFp16 = inQueueX.DeQue<half>();
LocalTensor<float> xLocalFp32 = tmpBuffer.Get<float>();
Cast(xLocalFp32, xLocalFp16, RoundMode::CAST_NONE, tileLength);
// 在 FP32 上计算
Add(yLocalFp32, xLocalFp32, weightLocalFp32, tileLength);
// 降精度回 FP16
Cast(yLocalFp16, yLocalFp32, RoundMode::CAST_RINT, tileLength);
```

#### 修复步骤

1. 检查所有 FP16/BF16 输入的计算逻辑
2. 确认已 Cast 到 FP32 计算
3. 确认已 Cast 回 FP16 输出

---

## 关键约束（实战经验）

### 1. --check-device-heap 与 --check-cann-heap 互斥

**错误信息**：
```
CANNOT enable both --check-device-heap and --check-cann-heap
```

**解决方法**：分两次运行 memcheck

```bash
# 第一次：检测 Device 接口泄漏
mssanitizer -t memcheck --check-device-heap=yes --leak-check=yes -- ./test_geir_op

# 第二次：检测 AscendCL 接口泄漏
mssanitizer -t memcheck --check-cann-heap=yes --leak-check=yes -- ./test_geir_op
```

**脚本已自动处理**：run_mssanitizer_geir.sh 会自动分两次执行。

---

### 2. ReduceSum/ReduceMax 可能修改源 Tensor

**错误症状**：
```
====== ERROR: illegal read of size 4
======    在归约操作后
```

**原因**：归约操作可能修改源 Tensor

```cpp
// ❌ 错误：归约后继续使用源 Tensor
ReduceSum(resultLocal, srcLocal, tmpBuffer, dimLen);
Add(dstLocal, srcLocal, otherLocal, tileLength);  // srcLocal 可能已被修改

// ✅ 正确：归约前备份源 Tensor
Adds(backupLocal, srcLocal, 0.0f, tileLength);  // 备份
ReduceSum(resultLocal, backupLocal, tmpBuffer, dimLen);
Add(dstLocal, srcLocal, otherLocal, tileLength);  // 安全使用
```

---

### 3. CANN 版本锁定（容器环境）

**问题**：容器中可能存在多套 CANN（如 cann-8.5 和 cann-9.0）

**解决方法**：显式传入 CANN 路径

```bash
# ❌ 错误：依赖环境变量（可能指向错误版本）
bash run_unified_mssanitizer.sh <project>

# ✅ 正确：显式指定 CANN 路径
bash run_unified_mssanitizer.sh <project> <cann_root>
```

---

## 编译选项配置

### 添加调试信息

```cmake
# CMakeLists.txt
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
```

**效果**：mssanitizer 日志可显示源码行号，而非 `<unknown>:0`

### 禁用优化

```cmake
# CMakeLists.txt
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
```

**效果**：避免优化导致 serialNo 定位困难

---

## 检测流程最佳实践

### 1. ops 算子仓检测流程

```bash
# 步骤 1：确认工程结构
ls -la <project>/examples/test_geir_*.cpp  # GE IR 子模式
ls -la <project>/examples/test_aclnn_*.cpp  # aclnn 子模式

# 步骤 2：一键检测
bash run_unified_mssanitizer.sh <project> <cann_root>

# 步骤 3：查看报告
cat <project>/mssanitizer_logs/geir_mssanitizer_summary_*.md
```

### 2. 自定义算子仓检测流程

```bash
# 步骤 1：生成测试脚本
python3 scripts/ascendc/gen_test_script.py \
    --operator my_custom_op \
    --dtypes float16 float32 \
    --output <project>/mssanitizer_test/my_custom_op_test.py

# 步骤 2：执行检测
bash run_unified_mssanitizer.sh <project>/mssanitizer_test/my_custom_op_test.py <cann_root>

# 步骤 3：查看报告
cat <project>/mssanitizer_logs/mssanitizer_summary_*.md
```

---

## 常见问题排查

### Q: 日志文件大小为 0，是否正常？

**A: 仅在工具退出码为0且测试实际命中目标算子时正常。** 工具执行失败也可能留下空日志；此时必须标记为未完成，不能判定通过。

### Q: 如何判断 GE IR 还是 aclnn 子模式？

**A: 脚本自动判断。** run_mssanitizer_geir.sh 会优先查找 `test_geir_*.cpp`，找不到时自动回退到 `test_aclnn_*.cpp`。

### Q: C++ 模式构建失败怎么办？

**排查项**：
1. 确认 examples/ 目录下存在测试源文件
2. 确认 op_graph/ 目录存在（GE IR 模式）
3. 确认 CANN 环境已激活（ASCEND_HOME_PATH）
4. 检查 CMake 输出的编译错误

---

## 参考资源

- [错误类型详解](error_types.md)
- [CANN mssanitizer 官方文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/devtoolquickstart/atlasopdev_16_0006.html)
- [AscendC 编程指南](https://www.hiascend.com/document)
