# Kirin 平台 Simulator 使用指导

## 适用范围

Kirin 系列端侧 SoC（Kirin9030、KirinX90 等以 Kirin 开头的平台）。**Kirin 平台不支持 `npusim` / `cannsim` 命令**（`npusim` 仅支持 Ascend 950/950DT）。Kirin 的仿真入口是算子工程内的一键脚本 `run.sh`，本文件指导其使用。

> 仿真命令在**你的算子工程目录**下执行；工程来自 `ascendc-direct-invoke-template` skill 的 `references/kirin_add_template/` 模板（复制改造步骤见该 skill 的 `references/kirin_vector_guide.md`）。ops-kirin-simulator skill 本身不提供 run.sh。

## 环境前置

1. **CANN**：需安装 mobile-station 版本（含 `simulator/<plat>/lib/` 仿真库，含 `libruntime_camodel.so` 等）。source `set_env.sh` 后 `ASCEND_HOME_PATH` 指向 CANN 安装根。
2. **Simulator 库**：确认 `$ASCEND_HOME_PATH/x86_64-linux/simulator/Kirin9030/lib/`（KirinX90 同理）存在。`ascendc-env-check` skill 的 `check_env.sh` 会自动探测该目录。
3. **GLIBCXX 兼容**：Kirin simulator 依赖 GLIBCXX_3.4.32，系统常自带 3.4.30。绕行：下载 `libstdc++.so.6.0.34` 放到工程 `ext_lib/`，`run.sh` 通过 `LD_PRELOAD` 加载（系统已有 GLIBCXX ≥ 3.4.32 时可跳过，详见 `kirin_vector_guide.md` 步骤 0）。

## 仿真命令

```bash
bash run.sh -r sim -v Kirin9030     # 或 -v KirinX90
```

| 参数 | 含义 | 取值 |
|------|------|------|
| `-r` | 运行模式 | `sim`（NPU 仿真，无需硬件）/ `cpu` / `npu`（需真实 NPU）；**Kirin 无硬件时用 `sim`，不支持 `cpu`，`npu` 需真机** |
| `-v` | SoC 版本 | `Kirin9030` / `KirinX90`（大小写敏感，仅这两个合法值） |
| `-i` | CANN 安装路径 | 可选，缺省时 run.sh 自动检测 `ASCEND_TOOLKIT_HOME` / `ASCEND_HOME_PATH` |

`run.sh` 自动完成：参数解析 → 校验 SoC 与模式 → sim 模式配置 simulator 库路径（`LD_LIBRARY_PATH` 指向 `simulator/<plat>/lib`、设 `CAMODEL_LOG_PATH=./sim_log`）→ ccec 编译（CMake 按 SoC 版本自动注入编译选项，产物 `./<app>_sim`）→ 重生成输入（`scripts/gen_data.py`，先清 `input/*.bin output/*.bin`）→ `LD_PRELOAD` 加载 `ext_lib/libstdc++.so.6.0.34` → 运行 `./<app>_sim` → 精度验证（`scripts/verify_result.py` 比对 `output/*.bin` 与 golden，输出 error ratio / test pass）→ 清理 sim 日志与 `*.vcd`（需保留时改 run.sh 末行）。

## 输出与数据口径

| 维度 | 来源 | 说明 |
|------|------|------|
| 精度 | `output/*.bin` | run.sh 末尾 `verify_result.py` 与 golden 比对，输出 error ratio |
| 性能 | sim 运行 stdout（tick/cycle）+ `sim_log` | 从运行日志解析；确定性仿真，多轮结果一致（MAD≈0） |
| **不产生** | `summary.json` / `trace_core*.json` | npusim 的报告产物 Kirin **没有**；`ops-simulator` skill 的"性能分析（summary.json 快速诊断 / trace 空泡分析）"章节对 Kirin 不适用 |

## 与 npusim（Ascend 950）路径差异对照

| 项 | Ascend 950 | Kirin 系列 |
|----|-----------|-----------|
| 入口命令 | `npusim record/report` | 工程内 `bash run.sh -r sim -v Kirinxxx` |
| 报告产物 | `summary.json` + `trace_core*.json` | 无（精度看 `output/*.bin`，性能看 stdout/sim_log） |
| 性能分析 | `ops-simulator` skill"性能分析"章节适用 | 不适用，自行解析 sim 日志 |
| 多核 | 支持（`-n` 指定 core） | Kirin 端侧单核 |

## 常见问题

| 问题 | 解决 |
|------|------|
| `npusim: command not found` 或报不支持该 SoC | 正常，Kirin 不用 npusim，改用 `bash run.sh -r sim -v Kirinxxx` |
| simulator 库缺失 | 确认装了 mobile-station 版 CANN，`simulator/Kirin9030/lib/` 存在 |
| `GLIBCXX_3.4.32 not found` | 放 `ext_lib/libstdc++.so.6.0.34`，run.sh 会 `LD_PRELOAD` 加载 |
| SoC 名拼写 | 仅 `Kirin9030` / `KirinX90` 合法，大小写敏感 |
