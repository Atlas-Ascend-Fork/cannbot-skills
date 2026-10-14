# 超越函数 + ALU 混合流 Overlap 标定（Ascend950 / DAV_3510）

测量日期：2026-06-08，设备 `bench-device-02`，单核 `block_dim=1`，`iterations=20000`，每 case 重复 3 次。
基准：vector core 1.65 GHz，sys-counter 1.0 GHz。指标 `vec-cyc/group = measured_cycles × 1.65 / (iterations×16)`。
复现：`python3 scripts/overlap_analysis.py`（原始数据 `data/overlap_raw.jsonl`）。

## 背景问题

理论评估器 `evaluate_a5_vector_roofline.py` 计算 vector 时间时，对**同一执行单元（vector_alu）上的指令时间求和**。
超越函数（Exp/Log/Sqrt/Div，4 拍吞吐）与廉价 ALU（Add/Sub/Mul，1 拍）都挂在 vector_alu 上，于是
被相加 ⇒ 对"超越函数+算术"混合的 vector-bound 算子（softmax / gelu / normalize）系统性高估。

## 测量结果

| case | 结构(每group) | 实测 vec-cyc/group | sum 模型 | max 资源模型 | 误差(vs sum) | 误差(vs max) |
|---|---|---:|---:|---:|---:|---:|
| add_fp32 | 1 add | 0.50 | 0.50 | 0.50 | +0.5% | +0.5% |
| exp_fp32 | 1 exp | 2.00 | 2.00 | 2.00 | +0.2% | +0.2% |
| sqrt_fp32 | 1 sqrt | 2.00 | 2.00 | 2.00 | +0.1% | +0.1% |
| exp_then_sub | 1 exp + 1 sub (dep) | 2.00 | 2.50 | 2.00 | **-19.9%** | +0.1% |
| exp_add2 | 1 exp + 2 add (dep) | 2.36 | 3.00 | 2.00 | -21.2% | +18.1% |
| exp_add4 | 1 exp + 4 add (dep) | 3.88 | 4.00 | 2.50 | -3.1% | +55.0% |
| exp_add8 | 1 exp + 8 add (dep) | 7.19 | 6.00 | 4.50 | +19.8% | +59.7% |
| exp_mul_add | 1 exp + mul + add (dep) | 2.50 | 3.00 | 2.00 | -16.6% | +25.2% |
| sqrt_add4 | 1 sqrt + 4 add (dep) | 4.00 | 4.00 | 2.50 | +0.1% | +60.1% |
| **exp_add4_indep** | 1 exp ∥ 4 add (独立) | **2.63** | 4.00 | 2.50 | -34.3% | **+5.1%** |
| **exp_add8_indep** | 1 exp ∥ 8 add (独立) | **4.78** | 6.00 | 4.50 | -20.3% | **+6.3%** |

## 结论

1. **单指令 class 全部复验通过**：纯 exp/sqrt = 2.0、纯 add = 0.5 vec-cyc，与模型一致（误差 <1%）。

2. **吞吐受限的理论下界是 `max` 而非 `sum`。** ILP 充足时（独立 ALU 变体），混合流贴合
   `max(发射口, 超越函数子流水, ALU 子流水)`，误差仅 +5~6%；而现 `sum` 模型高估 20~34%。
   - 发射口 = (超越函数数 + ALU 数) / 2（双发射，每指令占 1 个发射槽）
   - 超越函数子流水 = 超越函数数 × 2 vec-cyc（4 拍 / 2 lane）
   - ALU 子流水 = ALU 数 × 0.5 vec-cyc

3. **依赖链 + ILP 不足会抬高真实时间，甚至超过 sum。** 依赖版（4 lane harness）随 ALU 密度上升从 max 漂向
   sum，exp_add8 实测 7.19 已超 sum(6.0)。这是 4 条独立链喂不满流水导致的**延迟受限假象**——
   真实算子元素多、ILP 充足，不会落到这个区间（对照 exp_add4: dep 3.88 vs indep 2.63）。

4. **真实 softmax/gelu 片段（exp_mul_add）实测 2.50**：sum 高估 +20%，max 低估 -25%，真值居中。
   说明对带依赖的真实算子，正确做法是 **以 max 为下界，再叠加一个可标定的依赖/ILP 修正**，
   而不是默认 sum。

## 对评估器的建议改动

在 vector_alu 单元内把"发射口"与"超越函数子流水"建成两个独立资源，单元时间取
`max(issue_port_time, transcendental_pipe_time, simple_alu_pipe_time)`，替代当前的逐指令时间求和。
依赖链修正（`dependency`）应作用在该 max 下界之上，而非把不同子流水的时间无脑相加。

> 量级：对 exp+4add 这类流，下界从 sum 的 4.0 降到 2.5 cyc/group（理论时间 -38%）；
> exp_then_sub 从 2.5 降到 2.0（-20%）。

---

# 端到端验证：真实 softmax_v2 算子（fp32, ar_full_load, 950PR_9589）

测量 2026-06-08，TTK `kernel -b=release`（CANN 自带 ascend950 二进制），msprof per-pipe。
评估器 case：`regbase-microbench/docs/softmax_cases.jsonl`。每元素向量流 Max+Sub+Exp+Add+Div（Exp/Div 为超越函数）。

| shape | 实测 aiv_time | 评估器 total | bound(评估器) | aiv_mte2(实测) | aiv_vec(实测) | 评估器 vector(新) | 评估器 vector(旧sum) |
|---|---:|---:|---|---:|---:|---:|---:|
| 4096×2048 | 26.8us | 20.97us | mte2 ✓ | 22.81us (85%) | 11.29us (42%) | 4.97us | 6.87us |
| 8192×4096 | 138.8us | 83.89us | mte2 ✓ | 134.2us (97%) | 42.0us (30%) | 19.86us | 27.4us |

**验证结论**：
1. **bound 判定正确**：评估器判 mte2-bound，profiling aiv_mte2_ratio 0.85 / 0.97 印证；softmax fp32 是 memory-bound（向量管线只占 30-42%）。
2. **向量 overlap 修正安全**：新向量时间(4.97/19.86) 远低于 mte(21/84)，softmax total 不受影响——修正只在 vector-bound 算子上改变结果，不扰动 memory-bound 预测。新 vs 旧向量下界仍是 73%（Exp+Div 双超越函数主导）。
3. **新挖出两个标定 gap（下一轮目标，与向量 overlap 无关）**：
   - **GM 读带宽随规模 derate**：实测 aiv_mte2 有效读带宽 1.47TB/s(小)→1.0TB/s(大)，均与 json 的 1.6TB/s peak 不符；评估器 mte2(21/84) 对小 shape 准(差8%)、对大 shape 低估38%。需给 GM 带宽模型加 sustained/规模 derate。
   - **真实 kernel 向量管线 ≈ 吞吐下界的 2.3×**：实测 aiv_vec 11.29us vs 模型下界 4.97us。差距来自 scalar/mask 更新(scalar_ratio 0.1)、归约、Duplicate/Copy 等下界未计的开销——这正是 `target_efficiency` 要兜的部分。模型给的是理想吞吐下界，符合 skill「理论上界」定位。向量 overlap 模型本身的精确校验由微指令 bench（指令级，±6%）完成，softmax（memory-bound + scalar 重）不适合用来仲裁 overlap 模型。

---

# vector-bound 算子尺度验证：评估器 vs 实测（微指令流 kernel）

softmax fp32 是 memory-bound（GM 8B/元素），无法在整算子尺度检验向量模型。最干净的 vector-bound
验证对象是微指令 bench 本身——它就是真实跑在 950 上、纯 UB 无 GM 的 vector-bound kernel。把评估器
喂这些 exact 流，对照实测 cyc/group：

| flow（每 group） | 实测 | 评估器(新 max) | 评估器(旧 sum) | err 新 | err 旧 |
|---|---:|---:|---:|---:|---:|
| Exp + Sub (dep) | 2.00 | 2.00 | 2.50 | 0% | +25% |
| Exp + Mul + Add (dep, softmax 片段) | 2.50 | 2.00 | 3.00 | -20% | +20% |
| Exp ∥ 4·Add (indep, ILP 足) | 2.63 | 2.50 | 4.00 | **-4.9%** | +52% |
| Exp ∥ 8·Add (indep, ILP 足) | 4.78 | 4.50 | 6.00 | **-5.9%** | +25.5% |

**结论**：在真实 vector-bound kernel 上，改后评估器的 vector_time 对 ILP 充足的流是**紧致下界**（-6%~0%），
旧 sum 模型高估 **20%~52%**。依赖且 ILP 受限的流（exp_mul_add）实测落在 max 下界之上 20%，符合
「模型给吞吐下界、依赖/ILP 损耗在其上」的定位。这是对整套 flow 聚合（issue port + 子流水 max + 单发射）
在算子尺度的端到端验证。

> 真实库算子里 vector-bound 的例子（如 rms_norm_dynamic_mx_quant，fp16→fp4 量化，GM~2.5B/元素 + 重位运算）
> 结构上是 vector-bound，但经 TTK 驱动有 golden 依赖/binary 匹配/tiling 约束等工程问题，未在本轮取到干净的整算子数；
> 上面的微指令流验证已在算子尺度证明模型成立。

---

# 自写 vector-bound 直调算子端到端验证（决定性）

为在**真实整算子尺度**直接对照新模型，自写直调算子 `vecbound-op`（已被 skill 内 `tests/onboard/kernels/vchain` 取代）：
`y = HeavyAct(x)`，每元素 K 次 `a = Exp(a); a = a*0.3 + 0.1`（收敛到 ~0.7，无 inf；每次 1 超越函数 + 2 ALU）。
流式 GM 进出，64 核，M=1048576 fp32，GM 仅 8MB → 计算远超搬运 → **vector-bound**（评估器判 bound=vector_bound，
mte 仅 2.6us）。host 多次 launch 测 min 时延；扫 K=32/128 取稳态斜率（去掉固定开销）= (T₁₂₈−T₃₂)/96 us/K。

ILP（独立 reg lane 数）扫描：

| 并行 lane 数 | 实测稳态 us/K | vs 新模型 floor(0.310) | vs 旧 sum(0.465) |
|---|---:|---:|---:|
| 4 | 0.901 | 2.91× | 1.94×（latency-bound，ILP 不足）|
| 8 | 0.476 | 1.54× | 1.02×（恰好≈旧模型，是延迟受限假吞吐）|
| **16（ILP 足）** | **0.3025** | **0.98×（误差 2.5%）** | 0.65× |

评估器预测（K=128，M=1048576，64 核）：
- **新 max 模型**：bound=vector_bound，vector_time=39.72us = **0.3103 us/K**（被 `vector_transcendental` 子流水主导，Muls/Adds 在 ALU 子流水免费重叠）。
- **旧 sum 模型**：59.58us = 0.4655 us/K。

**结论（整算子尺度，决定性）**：
1. 随 ILP 4→8→16 lane，实测稳态吞吐单调收敛到 **0.302 us/K**，与评估器**新 max 模型 floor 0.310 吻合到 2.5%**。
2. **旧 sum 模型 0.465 高估真实可达下界 54%**。
3. 陷阱印证：8 lane（中等 ILP）实测 0.476 恰好≈旧 sum——看着"旧模型也对"，实为延迟受限的假吞吐；ILP 拉满后真下界 0.302 仅新模型预测正确。这就是为什么校验必须在 ILP 充足、真正吞吐受限的条件下做。

> 注：vector_time 是吞吐下界，需 ILP 充足（此处 16 lane）才达到；低 ILP（4 lane）的 latency 损耗甚至超过旧 sum 模型，由 `target_efficiency` / `dependency` 兜。

---

# GM 带宽标定（task #7，仪器见 skill 内 `tests/onboard/kernels/membw`）

纯搬运 bench（只 DataCopyPad，无计算），64 核 fp32，扫传输量，测 read/write/copy 有效带宽（详见 membw-bench/README.md）。

有效带宽 (TB/s) vs 单张量传输量：

| 单张量 | read | write | copy(gm=2×) |
|---:|---:|---:|---:|
| 2MB | 0.31 | 0.31 | 0.55 |
| 16MB | 1.53 | 1.50 | 2.59 |
| 32MB | 2.36 | 2.42 | 3.67 |
| 64MB | **3.26** | 3.04 | 2.26 |
| 256MB | **1.51** | 1.50 | 1.41 |

**结论（推翻"json 带宽偏低"的原始假设）**：
1. **真实 HBM 持续带宽 ≈ 1.4–1.5 TB/s**（>256MB 超 L2）。json 的 `gm_bandwidth` 1.4–1.6e12 ≈ 此值，**并不偏低**。
2. **L2 加持峰值 ≈ 3.3 TB/s**（read，工作集 ≲L2 112MB）；中等 size 远超 HBM 持续率。
3. 小 size（<8MB）固定开销受限（~6.5us），非带宽。
4. **读写在 HBM 争抢**：copy(各32MB)=18.28us vs max(read,write)=14.19us ⇒ 合并 ≈ `max(mte2,mte3)+0.3·min`。`max_pipe` 假设读写完全并行，过于乐观——**这才是 softmax 低估的真因，不是峰值带宽低**。

**评估器改动**：已测值写入 `ascend950pr.json` 的 `common.gm_bandwidth_measured`；新增 opt-in `gm_rw_contention`（默认关，回归不变；`true`→用 α=0.3，或给数字）。验证 softmax fp32：

| shape | 关(旧) | 开(rw争抢) | 实测 |
|---|---:|---:|---:|
| 4096×2048 | 20.97（-18%）| **27.26（+6.5%）** | 25.6 |
| 8192×4096 | 83.89（-39%）| 109.05（-21%）| 138.6 |

小 shape 基本修正；大 shape 仍低估，因其 strided/计算交织访问连 HBM 持续率都达不到（实测 mte2 仅 1.0TB/s），残差属访问效率（target_efficiency），带宽模型到此为止。L2 加持的 size-aware 带宽暂记为数据、未自动建模（访问模式相关，风险高）。
