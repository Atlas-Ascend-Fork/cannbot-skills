---
name: ge-inductor-autofuse-analysis
description: 融合性能分析。分析Profiling和Dump图，识别可融合范围；支持融合前后对比找出融合劣化片段和可融但未融合片段；支持Inductor+Triton/DVM/AutoFuse(AscendC)及GE+AF多种融合模式两两对比，找出融合范围差异和相同融合范围的算子性能差异，定位AutoFuse融合范围和融合算子优化点。触发场景：融合性能分析、融合算子分析、AutoFuse分析、DVM分析、Triton融合分析、融合对比、融合前后对比、融合劣化、可融未融、融合范围差异、AscendC融合、自动融合瓶颈、自动算子融合优化、融合劣化片段、可融合范围识别。需要分析昇腾NPU上自动融合算子性能、对比不同自动融合后端效果、定位融合劣化或可融未融片段。
tags: [融合, AutoFuse, DVM, Triton, AscendC, GE, Profiling, Dump图, 融合对比, 算子融合]
---
# 融合性能分析 Skill

## 核心理念

本 Skill 以 **SKILL.md 为分析方法论主源**，脚本 `autofuse_analysis.py` 仅为数据提取工具。分析人员应遵循本 SKILL.md 的分析步骤、判定规则、可视化规范来解读脚本输出并补充人工分析。

## 场景列表

| # | 场景         | 触发条件                               | 必选输入                      | 可选输入          | 脚本子命令                         |
| - | ------------ | -------------------------------------- | ----------------------------- | ----------------- | ---------------------------------- |
| 1 | 单模式分析   | 用户提供一组 Profiling，需分析融合效果 | Profiling 目录                | Dump 图文件       | `profile` + `graph` 或 `all` |
| 2 | 融合前后对比 | 用户提供同一模式融合前后两组 Profiling | 融合前/后 Profiling 目录      | 融合前/后 Dump 图 | `compare --mode before-after`    |
| 3 | 双模式对比   | 用户提供两种融合模式各自 Profiling     | 模式A/B Profiling 目录 + 名称 | 模式A/B Dump 图   | `compare --mode cross-mode`      |

**场景说明**：

- 场景1 覆盖所有融合模式（GE+AF / Inductor+AscendC / Inductor+Triton / Inductor+DVM），脚本自动识别模式。
- 场景2 融合前数据可为未开启融合的 baseline，也可为旧融合配置；融合后为新配置。
- 场景3 典型场景：模型从 DVM/Triton 切换到 AutoFuse 后性能下降，需定位原因。

## 融合模式识别

| 融合模式                   | op_statistic 前缀                                                     | 其他特征                                                       |
| -------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------- |
| GE+AF                      | `autofuse_` / `autofused_`                                        | 有 ModelExecute                                                |
| Inductor+AutoFuse(AscendC) | `autofuse_` / `autofused_`                                        | 无 ModelExecute，dump 目录含`asc_kernel.py`/`asc_graph.py` |
| Inductor+Triton            | `triton_poi_fused_` / `triton_per_fused_` / `triton_unk_fused_` | 无 ModelExecute                                                |
| Inductor+DVM               | `dvm_fused_` / `dvm_`                                             | 无 ModelExecute                                                |

> **关键约束**：执行模式判断**仅依据 `op_statistic` 融合算子前缀**，禁止从 dump 图文件中读取环境变量作为判断依据。

## 核心 WorkFlow

> `op_statistic 找热点类型 → op_summary 看占比/shape/Block Dim → output_code.py 找执行顺序 → asc_graph.py/DVM docstring 看融合范围 → fx_graph_readable.py 找前端差异 → 输出结论`

### 分析前准备

记录工作目录：网络名、场景(forward/backward/train step/inference)、AF/DVM/Triton 中间件路径、prof 路径、主分析 rank、是否跨 rank 汇总。

### 通用分析步骤（四步法）

以下四步为所有场景共享的分析骨架，各场景按需裁剪。

#### 第 0 步：迭代次数与算子耗时口径确定

所有"每迭代"指标都依赖迭代次数，必须先确定口径：

| 指标                                | 口径                                                                                                                          | 优先级                                                        |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **迭代次数**                  | `step_trace_*.csv` 行数（不含表头）                                                                                         | 优先（GE 通常有此文件，最准确）                               |
| **迭代次数（无 step_trace）** | `detect_iter_count()`：Synchronize API count 为候选，用 `op_statistic` min Count 校验，再用 `op_summary` 行数整除性验证 | 回退（Inductor 无 step_trace 时）                             |
| **每迭代算子耗时**            | `sum(op_statistic Total Time(us))` / 迭代次数                                                                               | 始终用 op_statistic，**不**用 step_trace Iteration Time |
| **每迭代平均耗时**            | `avg(step_trace Iteration Time(us))`                                                                                        | 仅 GE 有 step_trace 时可用，补充指标                          |
| **每迭代算子数**              | `sum(op_statistic Count)` / 迭代次数                                                                                         | 始终用 op_statistic Count 总和                                |

> 算子耗时和 ≤ 每迭代平均耗时(step_trace)，差额来自 Host 调度/同步/H2D。两者都看才能区分"算子慢"与"调度/同步拖累"。

#### 第 1 步：融合范围识别与差距分析

1. **从 profiling 找 fused kernel 名称**：看 `op_statistic_*.csv`，筛选 `autofused` / `dvm_fused` / `triton_*_fused` 前缀
2. **从 AF `asc_graph.py` 抽融合范围**：重点看文件头部注释
   - `Original ATen`：粗粒度范围
   - `Graph fragment`：精确看输入、输出、实际节点和中间变量
   - `MemoryDep` / `load offset`：对 slice 类 kernel 特别重要
3. **从 DVM `output_code.py` 抽融合范围**：看 docstring 中的 `torch.ops.aten.*`（高层范围）+ `def dvm_fused_*_build(k)` 内部（实际算子表达）
4. **用 `output_code.py` 对齐调用顺序**：不要只看 kernel 名字，必须看执行序列，按语义链路配对而非按名称配对

输出融合范围差异表：

| 语义 | AF kernel | AF 范围 | DVM/Triton kernel | DVM/Triton 范围 | 差异判断 |
| ---- | --------- | ------- | ----------------- | --------------- | -------- |

#### 第 2 步：相同融合范围下的性能对比

1. **判断是否"相同范围"**：只有满足以下条件才适合直接比较单 kernel 性能
   - 输入输出语义相同（不只是 shape 相同，还要确认 slice start/end、是否含额外 cast/mul）
   - ATen 范围相同或高度相似
   - 调用次数一致或可归一化
   - shape 接近
2. **从 `op_statistic` 做总量比较**：哪类 fused kernel 总耗时高
3. **从 `op_summary` 做单次执行比较**：看 `Task Duration` / `Task Wait Time` / `Block Dim`；模板选错证据按算子类型看占比——Vector 类看 `aiv_*_ratio`，CV/CC 融合类看 `aic_*_ratio` + `cube_utilization(%)`。**占比高者即为该维度的瓶颈**，具体判定表见[第 3 步 A. 模板选错]。

> **关键原则**（详见[常见误区]）：
>
> - 模板选错证据用占比 `*_ratio`，不用绝对耗时 `*_time`
> - Vector 类看 AIV 占比，CV/CC 类看 AIC 占比，二者证据源不同
> - 核数选错不能用 `Task Wait Time`/`Task Duration` 判定，必须对比融合前后 `Block Dim` + shape

#### 第 3 步：融合劣化原因定位

融合劣化（融合后耗时 > 融合前各算子耗时和）的可能原因分两大类：

> **代码仓分析要求**：本地 `asc_graph.py`/`asc_kernel.py`/`output_code.py` 是融合后实际生成的算子代码（融合范围、模板实现），但无法定位"为什么选了这个融合范围/模板"。如 [常见代码路径] 中的 [TorchAir代码](https://gitcode.com/Ascend/torchair/tree/master/experimental/_inductor_npu_ext/python/inductor_npu_ext)）和 [Inductor代码仓](https://gitcode.com/Ascend/pytorch/tree/master)）能访问，**必须联网获取相关源码并分析**，用于劣化原因定位和融合范围分析。重点查看：`npu.py`（`can_fuse_vertical/horizontal` 融合规则、codegen 入口）、`codegen/_asc_codegen.py`（tiling func 与 schedule 模板生成）、`config.py`（开关与 debug option）、`lowering/`（NPU 特殊 lowering）、`torch_npu/_inductor/dvm/`（DVM 融合规则）。脚本会在报告"需联网获取的代码路径提示"章节输出建议查看的代码路径和仓库链接。

**A. 模板选错**（kernel 内部实现模板与计算/访存模式不匹配）

判定流程：先按融合算子类型，用 `op_summary` 对应维度**占比（ratio）** 确认瓶颈维度（**证据**），再用 `asc_graph.py`（看 load/store 次数）/ `asc_kernel.py`（看模板实现）在代码层次进一步定位。

**Vector 类**（纯 vector pointwise kernel）：

| 占比字段（证据）        | 含义        | 可能原因                                                     | 代码定位                    |
| ----------------------- | ----------- | ------------------------------------------------------------ | --------------------------- |
| `aiv_vec_ratio` 高    | Vector 计算 | 计算表达式复杂，模板未选高效实现（exp/sigmoid/pow 软件展开） | `asc_kernel.py` 模板代码  |
| `aiv_scalar_ratio` 高 | scalar      | 标量计算多、index 计算多，未向量化                           | `asc_kernel.py` 标量循环  |
| `aiv_mte2_ratio` 高   | 读搬运      | 读数据量大、访问不连续、融合后重复读                         | `asc_graph.py` load 次数  |
| `aiv_mte3_ratio` 高   | 写回        | 输出大、写出不连续、多输出未合并                             | `asc_graph.py` store 次数 |

**CV/CC 融合类**（含 Cube 计算，如 matmul+eltwise 融合）：

| 占比字段（证据）                 | 含义           | 可能原因                                     | 代码定位                         |
| -------------------------------- | -------------- | -------------------------------------------- | -------------------------------- |
| `aic_mac_ratio` 低(<30%)       | MAC 计算占比低 | 计算非瓶颈，shape 太小或 cube 模板未充分利用 | `asc_kernel.py` + Shapes       |
| `aic_scalar_ratio` 高          | cube scalar    | cube 标量计算开销大                          | `asc_kernel.py` 标量逻辑       |
| `aic_mte1_ratio` 高            | L1 读搬运      | L1 访问效率低，tiling/读模板选错             | `asc_kernel.py` tiling/读模板  |
| `aic_mte2_ratio` 高            | cube MTE2      | cube 读数据多/访问不连续                     | `asc_kernel.py` 读模板         |
| `aic_mte3_ratio` 高            | cube MTE3      | cube 写回开销大                              | `asc_kernel.py` 写模板         |
| `aic_fixpipe_ratio` 高         | fixpipe        | fixpipe 处理开销大                           | `asc_kernel.py` fixpipe 逻辑   |
| `cube_utilization(%)` 低(<20%) | Cube 利用率低  | shape 太小，cube 算力未充分利用              | 结合`Input/Output Shapes` 评估 |

> **不可用绝对耗时 `*_time` 替代占比 `*_ratio`**（绝对耗时高不代表该维度是瓶颈，可能整体都高）。

**B. 核数选错**（调度核数与计算量/数据量不匹配）

> **判定原则（重要）**：核数选错**不能单凭 `Task Wait Time` 或 `Task Duration` 大来判定**（Wait 高也可能由流依赖/同步/上游慢引起；Duration 高也可能由模板选错引起）。必须同时满足：①**融合前后 Block Dim 对比**（观察核数变化方向）；②**结合 `Input/Output Shapes` 评估数据量**（判断 Block Dim 相对数据量是否过大/过小）。

| 现象（融合前后 Block Dim + shape 评估）                       | 可能原因                                |
| ------------------------------------------------------------- | --------------------------------------- |
| 融合后 Block Dim < 融合前各算子 Block Dim 之和，且 shape 较大 | 并行度下降，核数被压缩，单核负载过重    |
| 融合后 Block Dim 过大，但 shape 较小                          | 核数浪费，launch/调度开销 > 计算收益    |
| 融合前后 Block Dim 未变，但融合后计算量明显增加               | 核数未随计算量同步增加，并行度不足      |
| 多 launch 串行且每个 kernel Block Dim 偏小                    | 每个融合 kernel 核数少，launch 开销累积 |

> 双模式对比额外原因（AF 不如 DVM/Triton）：

| 原因                   | 典型现象                                                   | 证据源                                                              |
| ---------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------- |
| producer 被 inline     | AF 一个 producer 的 pointwise 被多个 fused kernel 重复计算 | `ir_post_fusion.txt` users, AF `asc_graph.py` 重复 fragment     |
| decomposition 不同     | AF 展开成多个基础 op，DVM 保留原 op                        | `fx_graph_readable.py`                                            |
| 共享子图丢失           | AF 每层生成一份，DVM 全图共享                              | FX 图节点出现次数,`output_code.py` 调用次数                       |
| fusion 边界过激进/保守 | AF 范围更大但含重计算 / AF 没融进去 DVM 融了               | AF`asc_graph.py` vs DVM docstring, `output_code.py` fallback op |
| lowering 差异          | 同语义 lowering 到不同 ATen 序列                           | `fx_graph_transformed.py`, `output_code.py`                     |

**判断"AF 范围更大但更差"**：找 AF 多出来的 fragment → 搜索是否已在前一个 fused kernel 中出现 → 看中间结果是否已物化 → 对比 DVM 是否读物化 buffer 而非重新 inline → 若是，结论写"AF 范围更大但包含重计算"。

#### 第 3.5 步：优化建议生成

脚本输出数据（劣化原因分类、耗时分解、Block Dim、可融未融片段），Agent 必须基于这些数据和[融合规则知识库]为每类问题生成优化建议，追加到报告对应章节末尾。

**可融未融片段优化建议**（针对每个 Top 片段，按**算子类别组合**判定，不绑定具体算子名）：

> 判定流程：识别算子类别 → 查[垂直融合规则]/[水平融合规则]判定类别组合是否可融（Y/Y*）→ 验证拓扑连通性 → 按类别组合给出建议。**禁止硬编码具体算子名作为判定条件**。无 Dump 图时标注"拓扑连通性未验证"。

| 判定条件（算子类别组合）                      | 建议内容                                                                                        |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 同类 Elemwise 连续（任意 Elemwise→Elemwise） | 满足垂直融合规则（Elemwise→Elemwise=Y），建议扩展 AutoFuse 融合边界将其纳入                    |
| Elemwise ∥ Elemwise 共享下游消费者           | 满足水平融合规则，建议将并行 Elemwise 水平+垂直融合为单个 kernel                                |
| Split/Slice + Elemwise                        | 满足 Split/Slice→Elemwise=Y，建议将 Split/Slice 与后续 Elemwise 融合                           |
| Gather + Elemwise（A5+ 专属）                 | 满足 Gather→Elemwise=Y，建议将 Gather 与后续 Elemwise 融合                                     |
| Transpose + Elemwise/Matmul                   | 满足 Transpose→Elemwise=Y / Transpose→Matmul=Y*4，建议融合                                    |
| Reduce + Elemwise（相同轴约束）               | 满足 Reduce→Elemwise=Y / Reduce→Reduce=Y*1，建议融合                                          |
| 含约束的 Y* 组合                              | 需额外校验约束条件（如 Reduce 相同轴、Concat 相同拼接轴、Transpose 仅末两轴），满足方可建议融合 |
| 其他组合（不在规则表中为 N 的）               | 不可融合，不做融合建议；如需优化从其他方向（如 tiling、核数）入手                               |
| 单迭代占比 > 10%                              | 标注"优化优先级高"                                                                              |

> 必须同时验证拓扑连通性（图边存在性），无 Dump 图时标注"拓扑连通性未验证"。

**融合劣化片段优化建议**（针对每个 Top 劣化片段，按劣化原因分类给出建议；证据源见[第 3 步 A/B 表格]）：

**A. 模板选错**（对应[第 3 步 A]各占比字段）：

| 瓶颈维度                                       | 建议内容                                                                                        |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `aiv_mte2_ratio` 高                          | 检查`asc_graph.py` load 次数 vs 输入 tensor 数，优化读模板（减少重复读/改善访问连续性）       |
| `aiv_mte3_ratio` 高                          | 检查`asc_graph.py` store 次数 vs 输出 tensor 数，合并写回或优化写模板                         |
| `aiv_vec_ratio` 高                           | 检查`asc_kernel.py` 模板代码，确认 exp/sigmoid/pow 是否用高效实现，考虑拆分复杂计算或更换模板 |
| `aiv_scalar_ratio` 高                        | 检查`asc_kernel.py` 标量循环逻辑，考虑将标量计算向量化或减少 index 计算                       |
| `aic_mac_ratio` 低                           | 结合`Input/Output Shapes` 评估 shape 是否过小，考虑增大 BS 或优化 cube 模板                   |
| `aic_scalar/mte1/mte2/mte3/fixpipe_ratio` 高 | 检查`asc_kernel.py` 对应 cube 模板逻辑（标量/tiling/读/写/fixpipe）                           |
| `cube_utilization(%)` 低                     | shape 太小，cube 算力未充分利用，考虑增大 BS                                                    |

**B. 核数选错**（对应[第 3 步 B]各现象）：

| 现象                        | 建议内容                                                                   |
| --------------------------- | -------------------------------------------------------------------------- |
| Block Dim 过大 + shape 小   | 核数浪费，建议减少 Block Dim                                               |
| Block Dim 被压缩 + shape 大 | 并行度被压缩，建议调整核数分配策略或增大 Block Dim                         |
| 非计算开销 占比高           | launch/调度开销累积，建议增大 kernel 粒度（融合更多算子）或减少串行 launch |
| 无明显主导因素              | 需结合 op_summary 占比指标和`asc_graph.py` 源码进一步定位                |

> 建议必须引用具体数据（如"`aiv_mte2_ratio` 占 41%，Block Dim 从 4→1"），禁止泛泛而谈。
> **如 TorchAir/Inductor 代码仓能访问，必须联网获取源码并分析**（见[第 3 步]代码仓分析要求），结合代码给出劣化原因和优化建议；代码路径见[常见代码路径]和[参考链接]。

#### 第 4 步：前端 FX 图处理差异

1. 打开 AF 和 DVM 的 `fx_graph_readable.py`
2. 搜索关键源码行（如 `npu_fa.py:14`、`rotary_pos_emb`、`rms_norm`、`silu`）
3. 统计关键 op 出现次数（如 `triu`、`sub`、`ge`、`where`、`npu_fusion_attention`）
4. 看 DVM 是否有共享节点：多个 attention 是否引用同一个 mask tensor
5. 打开 `fx_graph_transformed.py`，确认经过 decomposition/lowering 后差异是否扩大

输出 FX 图差异表：

| 主题 | AF | DVM/Triton | 性能影响 |
| ---- | -- | ---------- | -------- |

### 场景1：单模式分析

1. **识别融合模式**：从 `op_statistic` 融合算子前缀判断（GE+AF / Inductor+AscendC / Inductor+Triton / Inductor+DVM），禁止从 dump 图读环境变量。输出融合模式标签 + 证据链
2. **Profiling 分析**：解析 op_summary / op_statistic / api_statistic，Top 重复结构（算子数 > 1）及单迭代耗时占比
3. **图结构分析**（有 Dump 图时）：解析 Dump 图，构建算子拓扑 DAG，识别融合节点，分析重复子图与可融但未融合片段
   - **可融未融报告**：子图结构（`-->` 箭头可视化）、重复次数、单迭代耗时占比，按[第 3.5 步]给优化建议
   - **融合劣化报告**：融合算子耗时 > 融合前各算子耗时和即为劣化；劣化影响 = 重复次数 × (融合算子耗时 - 融合前耗时和)；劣化原因按[第 3 步]分类；按[第 3.5 步]给优化建议
4. **汇总报告**：合并 Profiling 与图结构分析为 Markdown，必须完成[快速检查清单]中所有任务项

### 场景2：融合前后对比

1. 分别对融合前/后 Profiling 进行模式识别与热点提取
2. 对比算子级耗时变化：热点概览（分"融合后劣化"/"未融合"/"融合后改善"三类）；融合劣化片段（融合后 avg 比融合前高 1% 或高 2us 以上）；劣化原因按[第 3 步]分类
3. 结合[融合规则知识库]分析可融但未融合子图结构（`-->` 箭头可视化、预期收益、占比），按[第 3.5 步]给优化建议
4. 若有 Dump 图，分析前后图结构变化（节点数/融合节点数对比、新增融合范围）
5. 汇总报告，必须完成[快速检查清单]中所有任务项

### 场景3：双模式对比

1. 分别对模式 A（AutoFuse）/ 模式 B（Triton/DVM）进行模式识别
2. **热点概览对比**：单迭代耗时、迭代算子数、融合算子数对比
3. **可融未融合分析**：
   - **能力对比**：AutoFuse 未融合但 Triton/DVM 已融合的算子，含迭代影响(us/iter)和占比，`-->` 箭头可视化
   - **AutoFuse 自身能力**：从 dump 图分析重复结构，结合[融合规则知识库]给出可融未融子图结构、重复次数、单迭代耗时影响；无 dump 图时从 op_summary 执行序列提取
4. **相同融合范围劣化分析**：两模式都融合了相同语义但 AutoFuse kernel 更慢，劣化原因按[第 3 步]分类，按[第 3.5 步]给优化建议
5. **总结**：可融未融 + 相同范围劣化对迭代的合计影响
6. 若有 Dump 图，分别分析图结构
7. 汇总报告，必须完成[快速检查清单]中所有任务项

### 建议输出表

最终建议至少输出 5 张表：

| 表                  | 内容                                                |
| ------------------- | --------------------------------------------------- |
| 热点 OP Type 表     | 从`op_statistic` 汇总各模式的 fused kernel 总耗时 |
| fused kernel 清单表 | 每个 fused kernel 的次数、fullname、语义            |
| 融合范围配对表      | 按语义配对各模式的 graph fragment                   |
| 同范围性能表        | 只比较融合范围相同或高度相似的 kernel               |
| 前端 FX 图差异表    | 节点是否被分解、是否共享、是否重复生成              |

## 输入文件格式

### Profiling 文件（MindStudio Profiler 输出目录）

| 数据类型     | 文件名模式                                        | 关键字段                                                                                                                                                                                                      |
| ------------ | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 算子耗时明细 | `*op_summary_*.csv` 或 `kernel_details.csv`   | OP Type, Op Name, Task Duration(us), Task Wait Time(us), Block Dim, Input/Output Shapes, AIV 占比(aiv_vec/scalar/mte2/mte3_ratio), AIC 占比(aic_mac/scalar/mte1/mte2/mte3/fixpipe_ratio), cube_utilization(%) |
| 算子汇总     | `*op_statistic_*.csv`                           | OP Type, Count, Total Time(us), Avg Time(us), Ratio(%), Core Type                                                                                                                                             |
| Host API     | `*api_statistic_*.csv` 或 `api_statistic.csv` | API Name, Count, Time(us)                                                                                                                                                                                     |
| 迭代耗时     | `step_trace_*.csv` 或 `step_trace_time.csv`   | Device_id, Iteration ID, FP Start(us), BP End(us), Iteration End(us),**Iteration Time(us)**, FP to BP Time(us)                                                                                          |
| trace数据    | `msprof_*.json` 或 `trace_view.json`          | —                                                                                                                                                                                                            |

**op_summary 关键字段解读**：

| 字段类别                                      | 字段                                                                                                                       | 含义                                                             |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 基础                                          | `OP Type`, `Op Name`                                                                                                   | 算子类型/fused kernel 类型，单次执行 task 名称                   |
| 耗时                                          | `Task Duration(us)`, `Task Wait Time(us)`                                                                              | device task 执行耗时/排队等待（Wait 不可单凭判定核数选错）       |
| 核数                                          | `Block Dim` / `Block Num`                                                                                              | 算子实际使用核数（核数选错判定：对比融合前后 Block Dim + shape） |
| Shape                                         | `Input Shapes` / `Output Shapes`                                                                                       | 输入输出 shape（辅助配对 + 核数选错时评估数据量）                |
| **Vector 类（AIV 占比，模板选错证据）** | `aiv_vec_ratio`, `aiv_scalar_ratio`, `aiv_mte2_ratio`, `aiv_mte3_ratio`                                            | Vector 计算/scalar/读搬运/写回 占比                              |
| **CV/CC 类（AIC 占比，模板选错证据）**  | `aic_mac_ratio`, `aic_scalar_ratio`, `aic_mte1_ratio`, `aic_mte2_ratio`, `aic_mte3_ratio`, `aic_fixpipe_ratio` | Cube MAC/scalar/L1读/MTE2/MTE3/fixpipe 占比                      |
| CV/CC 补充                                    | `cube_utilization(%)`                                                                                                    | Cube 利用率（低<20%→shape 太小）                                |

> - 模板选错证据用 `*_ratio` 占比字段，不用 `*_time` 绝对耗时（详见[第 3 步 A]）
> - AF 自定义 fused kernel 在 `op_summary` 中常显示 `Input Shapes=N/A`，需回 `asc_graph.py` 查真实 shape

### Dump 图文件

| 图源                  | 文件格式                                                | 解析方法                                       |
| --------------------- | ------------------------------------------------------- | ---------------------------------------------- |
| GE Build图（编译前）  | `*_AfterInfershape*.pbtxt`                            | 纯文本解析，提取 node 信息                     |
| GE Build图（编译后）  | `*_Build*.pbtxt`                                      | 纯文本解析，提取 node 信息                     |
| PyTorch fx图 (编译后) | `*output_code*.py`                                    | AST 解析，提取 fused kernel 调用和 fallback op |
| PyTorch fx图 (编译前) | `*fx_graph_runnable*.py` / `*fx_graph_readable*.py` | AST 解析，提取 torch.ops 调用                  |

> **fx图文件选择约束**：若目录下同时存在 `*output_code*.py` 和 `*fx_graph_runnable*.py`，**必须使用 `*output_code*.py`**（编译后实际生成代码，反映融合后真实执行结构）。若仅有 `fx_graph_runnable.py` 则可使用，但需在报告中注明"基于编译前 FX 图分析，融合结构可能不完整"。
> **GE图文件选择约束**: 如果 _Build.pbtxt 文件有多个，则多个文件需要都分析；编译后的GE图，必须使用`*_Build*.pbtxt` 分析

### AF/DVM/Triton 融合范围中间件

| 模式         | 文件                                                        | 提取内容                                                                                   |
| ------------ | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| AF (AscendC) | `autofused_*/asc_graph.py`                                | `Original ATen`（粗粒度范围）、`Graph fragment`（精确节点）、`MemoryDep`（切片范围） |
| AF (AscendC) | `autofused_*/asc_kernel.py`                               | AscendC 实现代码                                                                           |
| DVM          | `output_code.py` docstring + `def dvm_fused_*_build(k)` | 高层`torch.ops.aten.*` 融合范围 + 实际算子表达                                           |
| Triton       | `output_code.py` 中 triton kernel 定义                    | 类似 DVM                                                                                   |

## 脚本使用方式

统一脚本 `scripts/autofuse_analysis.py`，仅 Python 标准库依赖。

### profile — 单组 Profiling 分析

```bash
python autofuse_analysis.py profile <profiling_dir> [-o report.md]
```

输出：融合模式识别、核心指标、融合算子清单、热点算子（按 Ratio 降序累计 >= 80%）、Top 重复结构、可融但未融合算子。

### graph — Dump 图结构分析

```bash
python autofuse_analysis.py graph <graph_file> [--profiling-dir DIR] [-o report.md]
```

输出：图基本信息、重复子图分析、可融但未融合片段、融合劣化片段（含 `-->` 箭头可视化）。Agent 须按[第 3.5 步]追加优化建议。

### compare — 融合对比分析

```bash
# 融合前后对比
python autofuse_analysis.py compare --mode before-after \
  --before-prof <dir> --after-prof <dir> [-o report.md]

# 双模式对比（以 AutoFuse 为基线）
python autofuse_analysis.py compare --mode cross-mode \
  --mode-a-prof <dir> --mode-a-name "AutoFuse" \
  --mode-b-prof <dir> --mode-b-name "Triton" [-o report.md]
```

### all — 单模式综合分析（profile + graph）

```bash
python autofuse_analysis.py all <profiling_dir> [--graph-file FILE] [--graph-dir DIR] [-o report.md]
```

自动发现图文件（若未指定 `--graph-file`，从 Profiling 目录上级搜索最大的 `.pbtxt` 或 `output_code.py`）。

## 可视化规范（--> 箭头 + 详情表格）

**不使用 Mermaid**。所有子图/执行片段统一使用文本箭头格式：

1. **箭头行**：`OP_Type --> OP_Type --> OP_Type`，仅显示 OP Type，融合算子标注 `[fused]`
2. **详情表格**：紧跟箭头行下方，列出每个算子的 op name, op type, input shape, output shape, dtype, fused

```
**实例1**: `SoftmaxV2 --> autofused_to_copy_slice[fused] --> Add`

| # | Op Name | OP Type | Input Shape | Output Shape | dtype | Fused |
|---|---------|---------|-------------|--------------|-------|:-----:|
| 1 | softmax_0 | SoftmaxV2 | [1,12,128,128] | [1,12,128,128] | float16 | 否 |
| 2 | autofused_xxx | autofused_to_copy_slice | [1,12,128,128] | [1,12,128,128] | float16 | 是 |
| 3 | add_0 | Add | [1,12,128,128] | [1,12,128,128] | float16 | 否 |
```

- **图分析子图**：从 dump 图节点真实边关系绘制；**对比分析子图**：从 `op_summary` 执行序列提取（context_before=1, context_after=1）
- 每个可视化对象最多展示 3 个实例；有 Profiling 时在标题标注总耗时
- **边必须基于真实数据依赖**，禁止虚构线性链

## 融合规则知识库

### 融合必要条件与判定流程

两个算子可融合，**必须同时满足**：

1. **拓扑连通性**：参与融合的算子间必须存在真实数据依赖边（直接连边，或穿过 trivial 节点如 cast/reshape/view 的传递边）。垂直融合需前驱→后继 DAG 边；水平融合需共享同一消费者。**禁止虚构连边**——不可仅凭执行序列时间相邻或 OP Type 相同就假定可融合。
2. **融合规则**：算子类别组合满足[垂直融合规则]或[水平融合规则]中的 Y/Y* 判定。含约束的 Y* 需额外校验（如 Reduce→Reduce 相同轴、Transpose→Matmul 仅末两轴）。Cube 算子不可融入 vector pointwise kernel。

> **判定流程**：先验证拓扑连通性 → 再查融合规则 → 两者均通过方可判定"可融合"。

**已融合算子再融合**：autofuse 融合算子参与再融合时，先按名字推断类别（含 mm/matmul→Matmul；sum/mean/min/max→Reduce；embedding/gather→Gather；cat→Concat；slice/split→Slice/Split，不区分大小写），再套用规则，且必须有拓扑连边。

**可融但未融合判定**：①链式模式——拓扑序中相邻的未融合 vector 算子，须通过真实图边连通；②并行模式——共享同一下游消费者的同类型未融合算子组。排除 cube 算子和已融合算子。

### 算子分类

| 类别      | 包含算子（ATen/GE）                                                    | 融合角色                             |
| --------- | ---------------------------------------------------------------------- | ------------------------------------ |
| Elemwise  | add/sub/mul/div/exp/sqrt/relu/sigmoid/silu/clamp/neg/abs/where/cast 等 | 垂直+水平全打通                      |
| Broadcast | 二元/三元算子 shape 不同时                                             | 同 Elemwise                          |
| Reduce    | sum/mean/max/min/amax/amin/var_mean                                    | 垂直受限，水平不可                   |
| Concat    | cat                                                                    | 垂直受限，水平不可                   |
| Slice     | slice/select                                                           | 垂直全打通（新规则）                 |
| Split     | split/split_with_sizes                                                 | 垂直受限                             |
| Gather    | gather/embedding/index_select                                          | 垂直受限，A5+ 专属                   |
| Transpose | transpose/t/permute                                                    | 垂直全打通（新规则）                 |
| Matmul    | mm/addmm/bmm/baddbmm                                                   | cube 算子，不可融入 vector pointwise |

### 垂直融合规则（前→后串联）

行=前节点，列=后节点，Y=可融合，N=不可，Y*=有约束

| 前＼后    | Elemwise | Broadcast | Reduce | Concat | Slice | Split | Gather | Transpose | Matmul |
| --------- | :------: | :-------: | :----: | :----: | :---: | :---: | :----: | :-------: | :----: |
| Elemwise  |    Y    |     Y     |   Y   |   Y   |   Y   |   Y   |   Y   |     Y     |   N   |
| Broadcast |    Y    |     Y     |   Y   |   Y   |   Y   |   Y   |   Y   |     Y     |   N   |
| Reduce    |    Y    |     Y     |  Y*1  |   N   |   Y   |   N   |   N   |     Y     |   N   |
| Concat    |    Y    |     Y     |   N   |  Y*2  |   Y   |   N   |   N   |     Y     |   N   |
| Slice     |    Y    |     Y     |   Y   |   Y   |   Y   |   Y   |   Y   |     Y     |   N   |
| Split     |    Y    |     Y     |   N   |   Y   |   Y   |   N   |   N   |     Y     |   N   |
| Gather    |    Y    |     Y     |   Y   |   N   |   Y   |   N   |   N   |     Y     |   N   |
| Transpose |    Y    |     Y     |   Y   |   Y   |   Y   |   Y   |   Y   |     Y     |  Y*4  |
| Matmul    |    Y    |    Y*3    |   N   |   N   |   N   |   N   |   N   |     N     |   N   |

约束：

- *1 Reduce→Reduce：相同原始轴
- *2 Concat→Concat：相同拼接轴
- *3 Matmul→Broadcast：B轴≠Batch轴，且Broadcast不在Matmul输出链路
- *4 Transpose→Matmul：仅限Transpose交换最后两轴

### 水平融合规则（左∥右并行）

- 仅 Elemwise∥Elemwise、Elemwise∥Broadcast、Broadcast∥Broadcast 可水平融合
- Matmul不支持水平融合；Reduce不支持水平融合；多个Slice/Split可与Concat融合，多个Gather可与Concat融合
- 水平融合后算子，依然需要满足[垂直融合规则]要求

### Cube 算子排除

"可融但未融合"判定中**排除 cube 算子**（MatMulV3/BatchMatMulV3/MatMul/Conv2D 等），因为 cube 算子不能融入 vector pointwise kernel。

### Matmul Epilogue 融合

- Matmul→Elemwise Y：纯 Elemwise 可作为 epilogue
- Matmul→Broadcast Y*3：有约束的 Broadcast
- Transpose→Matmul Y*4：仅最后两轴交换的 Transpose 可前向融合
- Epilogue 黑名单：origin chain 含 slice/split/gather 时阻止融合

## 优化建议校验规则

### 规则1：禁止凭记忆编写实施参数

优化建议中的环境变量、配置参数必须通过联网查询昇腾官方文档确认。若无法验证，不得给出具体参数，改为引导用户查阅文档。

### 规则2：区分执行模式

| 执行模式         | 判断依据                                       | 适用措施                            | 不适用措施      |
| ---------------- | ---------------------------------------------- | ----------------------------------- | --------------- |
| GE+AF            | `autofuse_`/`autofused_` + 有 ModelExecute | GE 图编译 config、AutoFuse Pass     | Eager 专用接口  |
| Inductor+AscendC | `autofuse_`/`autofused_` + 无 ModelExecute | Inductor 环境变量、AscendC 后端配置 | GE 专用环境变量 |
| Inductor+Triton  | `triton_*_fused_` 前缀                       | Inductor 环境变量、Triton 后端配置  | GE 专用环境变量 |
| Inductor+DVM     | `dvm_` 前缀                                  | Inductor 环境变量、DVM 后端配置     | GE 专用环境变量 |

### 规则3：融合对比可比性判断

见[第 2 步]"判断是否相同范围"——不可直接比较：语义不同/ATen 范围不同/调用次数不可归一化；可直接比较：ATen 范围相同或高度相似/调用次数一致或可归一化/shape 接近。

### 规则4：实施示例必须标注来源

```
来源类型：
  - [昇腾官方文档]：URL 链接
  - [CANN 代码仓]：gitcode 链接
  - [Skill 内置知识]：本 SKILL.md
  - [AF_DVM分析指南]：AF_DVM_performance_analysis_guide.md
```

## 注意事项与常见误区

| 误区                                              | 正确做法                                                                                            |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 只按 OP Type 名称/shape 配对                      | 按语义链路和`output_code.py` 执行顺序配对；slice 必须看 start/end 或 AF `MemoryDep/load offset` |
| AF 融合范围大就认为更优                           | 检查多出来的 fragment 是否已在别处计算（重计算风险）                                                |
| 只看 rank1                                        | 训练场景必须看跨 rank 差异                                                                          |
| 只看`op_statistic` 或只看源码                   | `op_statistic` 找热点，`op_summary` 和源码解释原因，前端 FX 图可能已决定共享/分解，三者缺一不可 |
| 用`*_time` 绝对耗时判模板选错                   | 证据用`*_ratio` 占比字段，占比高者才是瓶颈维度                                                    |
| 用`Task Wait Time`/`Task Duration` 判核数选错 | 必须对比融合前后`Block Dim` + `Input/Output Shapes` 评估数据量                                  |
| Vector 类和 CV/CC 类用同一套指标                  | Vector 类看 AIV 占比(`aiv_*_ratio`)，CV/CC 类看 AIC 占比(`aic_*_ratio`) + `cube_utilization`  |

> 补充：AF OP Type 常带 hash，同一语义不同子图可能 hash 不同；profile 里的 hash 不一定在第一个子图里（如 qwen3-4B 中 profile 出现的是 `model__1_forward_5.2` 的 hash）。

## 快速检查清单

分析完成前逐项确认：

- [ ] 已定位各模式 profiling 和中间件路径，确认 profile 中 hash 对应哪个 `model__*` 子图
- [ ] 已用 `asc_graph.py`（AF）/ `output_code.py` docstring（DVM）抽取融合范围
- [ ] 已按语义配对（非仅按名字/shape），区分"融合范围差异"和"相同范围性能差异"
- [ ] 模板选错已按 Vector(AIV)/CV-CC(AIC) 分类查看占比，核数选错已对比融合前后 Block Dim + shape
- [ ] 已检查 FX 图 decomposition 和共享差异
- [ ] 已给出 AF 不如 DVM 的原因和证据文件路径；如 TorchAir/Inductor 代码仓能访问，已联网获取源码做劣化和融合范围分析
- [ ] 已删除所有临时文件

## 常见代码路径

AF 侧常看：

| 位置                                                     | 用途                                                    |
| -------------------------------------------------------- | ------------------------------------------------------- |
| [TorchAir代码]/`config.py`           | 开关、debug option                                      |
| [TorchAir代码]/`npu.py`              | `NPUScheduling.can_fuse_vertical/horizontal`（融合规则）、codegen 入口 |
| [TorchAir代码]/`codegen/_asc_codegen.py` | tiling func 与 schedule 模板生成                        |
| [TorchAir代码]/`common/asc_graph.py` | AF graph 输出                                           |
| [TorchAir代码]/`lowering/`           | NPU 特殊 lowering                                       |

DVM 侧常看：

| 位置                                                       | 用途                         |
| ---------------------------------------------------------- | ---------------------------- |
| [Inductor代码]/`torch_npu/_inductor/dvm/decomp.py`       | DVM 排除或修改 decomposition |
| [Inductor代码]/`torch_npu/_inductor/dvm/mlir_fusion.py`  | DVM fusion patch 入口        |
| [Inductor代码]/`torch_npu/_inductor/dvm/graph_fusion.py` | DVM graph fusion 规则        |

Triton 侧常看：

| 位置                                                       | 用途                         |
| ---------------------------------------------------------- | ---------------------------- |
| [Inductor代码]/`torch_npu/_inductor/triton_experimental/`       | Triton后端代码 |

## 参考链接

| 名称                        | 链接                                                                                                                                                              |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [AutoFuse]                  | https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/graphdevg/autofuse_1_0004.html                                                        |
| [TORCHINDUCTOR_NPU_BACKEND] | https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/ENV/docs/zh/environment_variable_reference/TORCHINDUCTOR_NPU_BACKEND.md                         |
| [环境变量总览]              | https://www.hiascend.com/document/detail/zh/canncommercial/latest/maintenref/envvar/envref_07_0001.html                                                           |
| [图Pass开发]                | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/graphdevg/docs/zh/user_guides/graph_dev/custom_pass_development/introduction.md |
| [GE options]                | https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendgraphapi/atlasgeapi_07_0150.html                                                      |
| [TorchAir代码]              | https://gitcode.com/Ascend/torchair/tree/master/experimental/_inductor_npu_ext/python/inductor_npu_ext                                                            |
| [Inductor代码]              | https://gitcode.com/Ascend/pytorch/tree/master/                                                                                                                   |
| [CANN代码仓]                | https://gitcode.com/cann                                                                                                                                          |
| [CANN社区]                  | https://www.hiascend.com/cann                                                                                                                                     |
