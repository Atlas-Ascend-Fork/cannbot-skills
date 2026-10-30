<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram bin index 计算

> 语言中立。定义元素→桶映射的两种算法、`min==max` 特判、边界钳制与越界丢弃。

## 1 等宽线性映射

适用 EqualWidth 形态。预计算 `invWidth = nbins / (hi - lo)`（或直接 `(val - lo) * nbins / range`），则：

```
idx = floor((val - lo) * nbins / (hi - lo))
```

实现要点：

- 用实数乘法 + 向零取整（对非负数等价于 `floor`），避免额外的取整调用。
- `lo/hi/range` 预取为局部标量，减少重复的全局内存读取；`nbins` 转为实数参与乘法。

## 2 min==max 特判

当 `hi == lo` 时 `range == 0`，公式除零。语义约定：所有落入区间的元素都归到中间桶：

```
if (hi == lo) idx = nbins / 2;
```

这是参考实现（`tf.histogram_fixed_width` 等）的一致行为，需保持。

## 3 边界钳制与越界丢弃

- **上界钳制**：等宽映射下，`val == hi` 时 `(val - lo) * nbins / range == nbins`，须钳制到 `nbins - 1`（`max` 精确值归入最后一个桶）：

  ```
  if (idx == nbins) idx = nbins - 1;
  ```

- **越界丢弃**：`val < lo` 或 `val > hi` 的元素不计入任何桶（直接丢弃）。注意在整数化/浮点误差下，判定边界应以 `` `lo ≤ val ≤ hi` `` 为准，钳制只处理排除了下溢的等值上界。

### 3.1 边界丢弃 vs 计入边界（按目标算子确认）

上界等值 `val == hi`（自定义边界为 `val == edges[-1]`）的归属在参考实现中不统一，**必须按实际开发的算子语义确认**，两种语义不可混用：

| 语义 | 区间模型 | `val == hi` 处理 | 参考语义 |
|------|---------|------------------|---------|
| **计入边界**（上界钳制，§3 默认） | 上界闭 `[lo, hi]` | 钳制到 `nbins - 1`，归最后桶 | `tf.histogram_fixed_width`；`numpy.histogram` 末桶 `[edges[-2], edges[-1]]` 右闭 |
| **边界丢弃**（右开） | 右开 `[lo, hi)` | `idx == nbins`，直接丢弃 | 统一左闭右开 `[bin_edges[j], bin_edges[j+1])` 的实现 |

- 差异**只落在端点等值处**：`lo < val < hi` 的中间区间两种语义结果一致，映射公式、`min==max` 特判无区别。
- **越界丢弃（`val < lo` / `val > hi`）在两种语义下都成立**，与「上界等值计入还是丢弃」是相互独立的两件事：先判定是否落在有效区间内，再决定上界等值计入还是丢弃。不要把「上界右开丢弃」与「越界丢弃」混为一谈。
- 下界 `val == lo` 通常计入首桶（左闭），歧义集中在上界；若目标算子为左开 `` `(lo, hi]` ``，则 `lo` 侧同样需丢弃，一并确认。
- 自定义边界（§4）同理：`val == edges[-1]` 是归 `nbins - 1`（末桶右闭，`numpy.histogram` 行为）还是丢弃，按目标算子确认。
- 落地时把选定语义固化进钳制/丢弃分支，并在 UT/ST 显式覆盖 `val == lo`、`val == hi`、`val == edges[-1]` 端值用例，防止两种语义被误混。

## 4 自定义边界二分定位

适用 CustomEdges 形态。对递增边界 `bin_edges[0..nbins]`，元素落在 `[edges[j], edges[j+1])` 时 `idx = j`：

```
idx = 二分查找第一个 > val 的边界下标 - 1
// 或等价：二分查找最后一个 ≤ val 的边界下标
```

- 右边界闭/开约定需与目标算子一致（`numpy.histogram` 最后桶为闭区间 `[edges[-2], edges[-1]]`）。
- `val` 落在 `edges` 之外时丢弃；落在最后一个闭桶时归 `nbins-1`。
- SIMT 落地中二分实现是逐线程标量查找（无向量化二分原语），见 [../adapters/ascendc/kernel-skeleton.md](../adapters/ascendc/kernel-skeleton.md) §bin index 分支。

## 5 向量化下的一致性

按 `` `VECTOR_LOAD_WIDTH` `` 向量化读入后，对向量的每个分量各自独立执行 §1~§3 的映射与钳制，语义与标量逐元素完全一致（参考样例逐分量展开）。向量化只改变访存，不改变映射公式与特判。

## 6 二维 / 多维 bin index 与展平

D 维直方图把「单特征 → 单 bin index」推广为「D 特征 → D 维 bin index → 展平 index」，分两步：

### 6.1 每轴独立 bin index

对每个特征轴 `d ∈ [0, D)`，独立套用 §1~§5 的映射规则（等宽公式 §1 / `min==max` 特判 §2 / 边界钳制与越界丢弃 §3 / 自定义边界二分 §4），得到该轴桶下标 `i_d ∈ [0, nbins_axis[d])`：

```
i_d = BinIndex_d(val_d, bin_edges_d / lo_d / hi_d)     // 各轴规则与其 bins 形态一致
```

各轴的 bins 形态、range、`nbins_axis[d]` 可不同（见 [bin-model.md](bin-model.md) §5.1），映射规则逐轴选择。

### 6.2 离散类标轴特例

若某轴是**离散类标轴**（取值即类标，见 [scene.md](scene.md) §2），其 bin index 就是类标本身：

```
i_d = class_d                                   // 无 floor / 二分，bin 边界是整数格点
if (class_d < 0 || class_d >= nbins_axis[d]) 丢弃   // 越界类标丢弃
```

### 6.3 行优先展平

D 个每轴 bin index 组合成 D 维下标后，按 [bin-model.md](bin-model.md) §5.3 行优先展平（`` `FLAT_INDEX_CALC` ``，Horner 递推）：

```
idx = ((…((i0 × nbins_axis1 + i1) × nbins_axis2 + i2) × …) × nbins_axis_{D-1} + i_{D-1})
D=2：  idx = i0 × nbins_axis1 + i1
```

展平后 `idx ∈ [0, ∏ nbins_axis)`，与 1D 的 `idx ∈ [0, nbins)` 同构，后续累加与归并完全复用 1D 路径。
