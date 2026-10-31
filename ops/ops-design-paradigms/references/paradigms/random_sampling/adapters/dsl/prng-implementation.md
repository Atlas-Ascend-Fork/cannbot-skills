<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling PRNG 的 DSL 实现

> **层级**：CANNBotDSL 语言适配层。Philox4x32-10 流模型与常数见 [../../generic/prng-model.md](../../generic/prng-model.md)；分布变换见 [../../generic/distribution-transform.md](../../generic/distribution-transform.md)。
>
> DSL 无公共底座（对照：AscendC 侧 `random_common` 的 `PhiloxRandom<10>`/`PhiloxRandomSimt`），本文给出轮函数与 u 构造的 **DSL API 级表达清单**；一切实现须对流基准对拍（消费规则 4）。

## 1 Philox4x32-10 单轮的算子映射（SIMD）

Philox 轮 = `mulhilo32(单乘) + Weyl 常数加 + key 提升`，DSL reg 算子逐项对应：

| Philox 步骤（prng-model §1） | SIMD 算子（`vf(mode="simd")` 内） | 备注 |
|------|------|------|
| `mulhilo32(a, M)`：32×32→64 高低位 | `vmull(a, m, dtype=dtypes.uint32, mask=...) -> (hi, lo)` | 返回 (high, low) 对——**高低序以实测对拍锁死** |
| Weyl 序列加（counter + W32_A/B） | `vadd`；需进位链时 `vaddc(src0, src1, carry_src, mask) -> (carry, sum)` | uint32 域回绕 |
| key 提升 `k ^ W` | `vbitwise_xor` | |
| 高低混合（位移拼接） | `vshl` / `vshr` + `vbitwise_or` | |
| 位级重解释（u ↔ bits） | `vreinterpret(src, dtype)` | u 构造核心（§3） |
| dtype 域转换 | `vcast`（配 `RoundingMode`）；标量操作数版 `vadds/vmuls` | |

- 掩码纪律：所有带 `mask` 参数的接口，mask 必须先经掩码寄存器操作（`full_mask`/`update_mask`/`mask_counter`）**预先赋值**；未赋值掩码内容不确定
- 常数经 `vdups` 广播到全 lane；常数池见 [capability-mapping.md](capability-mapping.md) §2
- 10 轮展开为直线代码或 `cbd.range_constexpr(10)`（编译期循环）；**轮数 10 显式登记**，不留隐式默认

## 2 SIMT 轮函数（纯线程版）

```python
with cb.vf(mode="simt"):
    # 每线程本地 key[2]/counter[4]（Python 局部变量承载，寄存器分配）
    # 32×32→64：cb.simt 标量整数乘 + 移位拆高低（或 fma/短向量，实测择优）
    hi = (a * M_) >> ...        # 具体高低拆分口径对拍锁死
    lo = ...
    counter = cb.simt.bitcast(...) / 位运算组合
```

- `cb.simt` 标量数学类目：`cast`、`bitcast`、`fma`、位运算（接口分类见 compute.md「SIMT 模式」）
- uint64 乘优化（AscendC `uint2` 先例）在 DSL 无对应物——按标量整数语义直写，产物质量以实测为准
- 每线程状态体积受寄存器档位约束（线程数 × 状态量，[pitfalls.md](pitfalls.md) §2.3）

## 3 u 构造（两口径的 DSL 落地）

**R7 硬规则**：口径选择以**流基准同路径**为准，同流不同值——登记所选口径并全程一致。

| 口径 | 值域 | SIMD 落地 | SIMT 落地 |
|------|------|-----------|-----------|
| 位重解释 | [0,1)（含 0） | `vbitwise_and(u, 0x7fffff)` → `vbitwise_or(…, 127<<23)` → `vreinterpret(→float32)` → `vadds(…, -1.0)` | `cb.simt.bitcast(位拼接结果, dtype=float32)` 后减 1 |
| 乘法+半步 | (0,1)（不含 0/1） | `vcast(u→float32)` → `vmuls(RAND_2POW32_INV)` → `vadds(+RAND_2POW32_INV_HALF)` | 标量 cast + 乘加 |

- half/bf16 输出：fp32 中间域计算后 `vcast(→目标, rounding=RoundingMode.RN)`；或搬出时 `vstore_pack`（B32→B16）——AscendC 的 `SetCtrlSpr<60,60>` 半精度旁路在 DSL 无对应开关，半精度舍入行为**待验证**（对拍锁死）
- 常量池：`RAND_2POW32_INV = 2.3283064e-10`、`RAND_2POW32_INV_HALF`（乘法+半步口径，capability-mapping §2）

## 4 快除（线程映射 / 反线性化共用）

host 构图期预计算（纯 Python，同 AscendC `GetUintDivMagicAndShift` 算法）：

```python
magic, shift = get_uint_div_magic_and_shift(divisor)   # host 侧
# 传入：Constexpr（编译期常量）或 params Tensor（运行期值）
```

kernel 侧复算：SIMD `vmul` + `vshr`；SIMT 整数乘移位。全值域与 `/` 对拍后方可启用（待验证项）。

## 5 状态与流推进（对照表）

| AscendC 底座概念 | DSL 对等物 | 出处 |
|------|------|------|
| `Skip(offset)`（流起点对齐） | 每线程 counter 初值换算 `⌈offset/4⌉ + 区间映射`（SIMT 自算）或 host 算好进 params | state-management §3 |
| `GenRandomSIMD` 批量产出 | SIMD VF 内 `vmull` 轮链 + `vstore` 进 UB | 本文 §1 |
| counter 消耗 n/k 推进 | 循环边界算术（`cbd.range` 计数） | prng-model §2 |
| offset 回写 `SetValue + DataCacheCleanAndInvalid` | SIMT 唯一线程 GM 直写 + `cb.simt.threadfence`；或拆独立顺序 launch | simt-path §4 |
| 块协议 `CalcSplitBlocks` | host Python 多段划分 + 每段独立 launch（offset 累进） | simt-path §5 |

## 6 首例对拍清单（验证义务落地）

1. 单核小批量：k=4 步产出与 torch/TF 对标序列**逐位**一致（key/counter 布局按接口族，prng-model §1.1）
2. u 构造边界：所选口径断言值域端点（含 0 / 不含 0 用例）
3. 轮数敏感性：改轮数为 9 时对拍**必须**失败（防"轮数常量漂移仍自洽"）
4. 跨核序列拼接：双核各半区间，拼接后与单核全区间逐位一致（流区间隔离）
5. offset 递进：两次调用第二次输出 = 跳过第一次消耗量后的序列
