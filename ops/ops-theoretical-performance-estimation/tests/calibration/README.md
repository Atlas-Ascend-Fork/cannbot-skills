# 标定层（calibration）

把引擎/JSON 里的**标定常量**绑定到**上板 bench 实测**，让「不许为迎合算子凑常量」从口头铁律变成门禁。

## 分工（务必分清）
- **bench = 常量的来源/真值**：`bench_results/*.csv` 是上板微基准实测的**汇总**；产出它的**仪器源码 + 原始板上数据**全部随 skill 版本化（见下「数据来源溯源」），不依赖工作区外部目录——保证透明、可回归、可验证。
- **算子 onboard = 独立验证集**：`tests/onboard/compare.py` 的 add/rms_norm/vchain，用来验证标定后的模型，**不参与**标定。
- **本层 = 一致性门禁**：`check_calibration.py` 只验「JSON 常量 ⟷ bench 实测」一致，不碰算子。

> 三者不能混：常量从 bench 推出 → 算子独立验证 → 一致性检查锁死「常量必须有 bench 背书」。

微基准实测报告由 `instruments/regbase-microbench/scripts/analyze_results.py` 根据设备采集记录生成，记录本工具的测量结果。

## 文件
- `bench_results/*.csv` — 标定常量的实测汇总（`check_calibration.py` 直接读）。
- `instruments/regbase-microbench/` — **单指令吞吐 + overlap 标定仪器源头**：仪器源码（`op_host`/`op_kernel`）、manifest（`config/calibration_cases.json`）、方法/findings 文档（`docs/`）、入库板上原始数据（`data/task5_raw.jsonl` 30 指令合并原始 + `data/overlap_raw.jsonl`）。逐 case 的 `output/raw_*.jsonl` + `report_*.json` 由 `run.sh` 上板重跑生成，属本地临时 scratch（不入库）；`bench_results/insn_throughput.csv` 即由该重跑的 `report_*.json` 摘出、并可对 `data/task5_raw.jsonl` 复核。
- `check_calibration.py` — 离线一致性检查（读 bench CSV + 跑引擎，无需板子，已纳入 `tests/run.py`）。

## 数据来源溯源（每条 csv ↔ 仪器 + 原始数据，全部在 skill 内）
| 标定 csv | 产出仪器（skill 内） | 原始数据 |
|---|---|---|
| `bench_results/insn_throughput.csv` | `instruments/regbase-microbench/`（单核 `GetSystemCycle`、unroll、无依赖） | `instruments/regbase-microbench/data/task5_raw.jsonl`（30 指令合并原始；逐 case `raw_*/report_*` 由 `run.sh` 重跑，不入库） |
| `bench_results/membw_bandwidth.csv` | `tests/onboard/kernels/membw/`（`<<<>>>` 直调，read/write/copy） | csv 即记录的板上实测 |
| `bench_results/membw_copy_sweep.csv` | `tests/onboard/kernels/rms_norm/ --variant copy`（纯拷贝变体） | csv 即记录的板上实测 |
| overlap 模型（sum→max） | `instruments/regbase-microbench/`（exp + k·add 扫描） | `instruments/regbase-microbench/data/overlap_raw.jsonl` + `docs/overlap_calibration_findings.md` |

## check_calibration.py 验什么
| 检查 | bench | 常量 | 判据 |
|---|---|---|---|
| l2_aware GM 曲线 | membw copy 扫描(≥128MB) | `gm_bandwidth_l2_aware.curve` | 引擎(l2_aware 拷贝) ≈ 实测 copy_us，≤15% |
| 单指令吞吐 | regbase-microbench(单核) | `vector_instruction_types.json` 各 op `execution_model` | 引擎(单核 count×elements) ≈ microbench_us，比值∈[0.9,1.1] |
| HBM 带宽 | membw read/write/copy(@512MB) | `peak.gm_bandwidth` / l2_aware 尾点 | 引擎(纯访存) ≈ 实测，≤10% |

单指令吞吐容差 [0.9,1.1]：用正确口径(`measured_cycles`)引擎与 microbench 近乎一致(t4 超越函数 0.999、single_issue reduce 0.957)，故容差收紧到 ±10%，能锁死 throughput_cycles——t4→0.999 过、误标 t3.25→0.81 挂、t8→2.0 挂。⚠ `microbench_us` 必须用 `measured_cycles`(稳态吞吐真值)；曾用 `body_cycles`(减空循环基线)虚高 23%，把 t4 误判 0.616→t3.25。

## ⚠️ 吞吐只能用单核无依赖 microbench 测，不能用依赖链算子
单指令**吞吐**(throughput_cycles)的真值来自 `regbase-microbench`：单核、unroll、**无数据依赖**、cycle 计数。
**不要**用 `t=op(t)` 依赖链 kernel 去标吞吐——那测的是**延迟受限**(受寄存器/ILP 限),会把吞吐地板拟合错
(本仓曾用 vchain 依赖链版把 fp32 Sqrt 误标 t8;regbase-microbench 微基准实测报告记录单核测量值 0.4996 issue/vec-cyc=t4)。
算子(vchain/rms_norm)只做**独立验证**:vchain 现已重写为**吞吐最优** `y=Σ_k sqrt(x+k)`(无依赖链),理论≈实测。

## 常量来源现状

指令峰值依据 `references/vector_instruction_types.json` 中记录的 asc-devkit 公开性能表和单双发表；硬件 API 合法类型、文档覆盖的性能类型、实测校验三者分别记录。

| 常量 | 来源 | 验证依据 |
|---|---|---|
| 公开表覆盖的指令并行度与单双发 | asc-devkit 公开文档，版本见 `source_documents` | `tests/offline/test_public_instruction_rates.py` 的 300 个文档组合 |
| GM `l2_aware` 曲线 / HBM read/write/copy | 微基准实测 | membw 扫描 |
| fp32 Sqrt / Exp / Log / ReduceSum | asc-devkit 公开值 + 微基准实测交叉验证 | `bench_results/insn_throughput.csv` |
| transcendental 与 ALU 的混合流重叠 | 微基准标定的模型 | `data/overlap_raw.jsonl` 与 `docs/overlap_calibration_findings.md`（均位于 `instruments/regbase-microbench/`） |
| fp32 Log2 / Log10 | 微基准实测；该公开性能汇总未单独列出 | `instruments/regbase-microbench/data/task5_raw.jsonl` |
| Rsqrt / Reciprocal、未列出类型和特殊模式 | JSON 中显式标注的模型近似，仍需独立校准 | 不作为 asc-devkit 单指令性能保证 |

`muladddst_fp32` 的旧微基准是 `dependent_dst_update` 流，原始计数保留为该依赖链的实测记录。它的吞吐范围不代表独立指令峰值；浮点 MulAddDst/MulDstAdd 的估算按公开表换算为 1 拍，文档列出的整数 MulAddDst 仍为 2 拍。

## 重新标定一条指令吞吐（流程）
1. 在 `instruments/regbase-microbench/` 的 manifest(`config/calibration_cases.json`)+kernel 加该指令的单核 case（无依赖、unroll），用其 `run.sh` 上板跑出 `output/raw_<op>.jsonl` + `report_<op>.json`。
2. 据 `measured_cycles`(**不是** body_cycles，见上方告警) 调 `vector_instruction_types.json` 该 op `execution_model` 的 throughput_cycles，使引擎单核 ≈ microbench_us。优先用微基准实测报告的 `issue_per_vector_cycle` + `expected_class`。
3. 往 `bench_results/insn_throughput.csv` 追加该 op 一行 → `check_calibration.py` 自动纳入门禁。
