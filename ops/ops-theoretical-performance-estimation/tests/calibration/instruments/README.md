# 标定仪器源头（instruments）

放置标定常量的**数据来源源头**——产出 `../bench_results/*.csv` 的微基准**仪器源码 + 原始板上数据 + 方法文档**，随 skill 一起版本化，保证透明、可回归、可验证、**不依赖工作区外部目录**。

## 目录
- `regbase-microbench/` — **单指令吞吐 + 超越函数↔ALU overlap 标定仪器**（A5 RegBase，单核 `GetSystemCycle`、unroll、无依赖 cycle 计数）。
  - `op_host/` `op_kernel/` `CMakeLists.txt` `run.sh` — 仪器源码与上板入口。
  - `config/calibration_cases.json` — 指令 case manifest。
  - `data/` — **入库的板上原始数据**（无 in-repo 一键重跑法，故版本化保留）：`task5_raw.jsonl`（30 指令合并原始 cycle 计数，指令库 `source` 引用、复核 `../bench_results/insn_throughput.csv` 用）、`overlap_raw.jsonl`（overlap 扫描原始，`scripts/overlap_analysis.py` 输入）。
  - `output/` — `run.sh <case>` 上板重跑生成的逐 case `raw_*.jsonl`（原始计数）+ `report_*.json`（派生指标 `issue_per_vector_cycle`/`expected_class`/`class_match`），属本地临时 scratch（整目录被顶层 `**/output/` 忽略，不入库）；`../bench_results/insn_throughput.csv` 即由重跑的 `report_*.json` 摘出。
  - `docs/` — `regbase_microbench_method.md`（方法）、`overlap_calibration_findings.md`（overlap sum→max 标定结论，引擎 `evaluate_a5_vector_roofline.py` 与 SKILL.md 引用）、`softmax_cases.jsonl`（overlap findings 配套的评估器输入示例）。

## 其余标定常量的源头（不在本目录，但都在 skill 内）
- GM/L2 带宽（`bench_results/membw_*.csv`）→ 仪器 `tests/onboard/kernels/membw/`（read/write/copy 直调）+ `tests/onboard/kernels/rms_norm/ --variant copy`（纯拷贝扫描）。
- 完整 csv ↔ 仪器 ↔ 原始数据 映射见 `../README.md`「数据来源溯源」。

## 重跑/扩标
`regbase-microbench/` 是自包含 CMake 工程：改 `config` manifest + kernel → 用 `run.sh` 上板 → 产 `output/raw_*.jsonl`+`report_*.json` → 更新 `../bench_results/insn_throughput.csv` → `check_calibration.py` 自动纳入门禁。详见 `../README.md`「重新标定一条指令吞吐」。

> 注：`build/` 等构建产物被 `.gitignore` 挡掉，不入库。
