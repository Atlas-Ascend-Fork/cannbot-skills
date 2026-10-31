<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# random_uniform_v2_simd_draft —— DRAFT 参考实现说明

> **状态：DRAFT（未编译、未上板、未对拍）**。RandomSampling 范式 A/C 族 SIMD 路线的结构参考实现，**与 AscendC 侧同算子对齐**（[../../../ascendc/example/random_uniform_v2_package/](../../../ascendc/example/random_uniform_v2_package/)，语义契约一致，实现载体按语言）；cannbot-dsl 仓当前没有随机算子样例，本文件填补"从文档到首例"之间的结构空白。

## 结构展示点

| 文件位置 | 展示内容 | 依据文档 |
|---------|---------|---------|
| `random_uniform_v2`（@host） | seed/seed2 **属性**折叠 Constexpr key 字 + chunk_loops 推导 | 接口契约（seed/seed2 为属性而非张量） |
| `offset_out[0] = offset_in[0] + out.shape[0]` | offset 递进协议（元素计数口径，单写者：单 block kernel） | state-management.md §3 |
| `base_words = dtypes.int64(offset_in[0]) // 4` | counter 起点换算（⌈offset/4⌉ 字，设备侧运行期） | prng-model.md §1.1 |
| `_word_to_uniform`（@jit helper） | u 构造位重解释口径链 | prng-implementation.md §3（R7 口径一致性） |
| 轮函数循环（`range_constexpr(10)`） | Philox 单轮 → vmull/vbitwise_xor/vadd 映射 + Weyl key bump | prng-implementation.md §1 |
| chunk 循环 + VF 作用域 + 搬出 | Hardware Loop 纪律 + VF 外 mem_copy + tile_slice chunk 视图 | simd-path.md §2/§4 |

## 待验证点（首例接入时逐项闭环）

| # | 待验证项 | 说明 |
|---|---------|------|
| A1 | `vmull` 返回 (high, low) 序 | 高低位序反了不报错、序列完全错误——对拍锁死 |
| A2 | `vdups` 接受运行期标量（`base_words`） | key 字已改构图期常量消除该依赖；counter 起点仍需运行期广播，不可用时改 UB 中转 + vload_broadcast |
| A3 | 尾块掩码 | 本草稿按整 chunk 写出；真实接入按 simd-path.md §2 用 update_mask/mask_counter 处理非整除尾块 |
| A4 | counter/key 装配与 offset 换算 | 草稿只示意（c0 = j*VL + base + lane）；按 prng-model §1.1 接口族布局与 ⌈offset/4⌉ 口径逐位核对 |
| A5 | 半精度输出 | dtype 属性 fp16/bf16 需 fp32 中间域 + 出口 vcast（RoundingMode.RN），舍入口径对拍 |
| A6 | offset 读取/递进的标量流水路径 | `offset_in[0]` 标量读与 int64 `//` 在 SIMD kernel 内的可用性（compute.md「标量计算」警告） |

## 首例对拍义务（prng-implementation.md §6，五项全做）

1. 单核小批量与 torch/TF 对标序列**逐位**一致
2. u 构造边界断言（[0,1) 含 0 端点用例）
3. 轮数改 9 必须对拍失败
4. 跨核序列拼接 = 单核全区间
5. offset 递进：第二次调用输出跳过第一次消耗量（`offset_out` 回填闭环）
