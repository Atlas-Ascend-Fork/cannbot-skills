---
name: ops-kirin-simulator
description: Kirin 平台 NPU 仿真器技能。提供 CANN Simulator 的使用指导，包括精度仿真、性能仿真。当 Kirin 平台无 NPU 硬件需要仿真验证算子功能，或误用 npusim 报不支持该 SoC 时使用。
---

# Kirin 平台 Simulator 使用指南

## 平台适用性（先读）

Kirin 系列端侧 SoC（Kirin9030、KirinX90 等以 Kirin 开头的平台）**不支持 `npusim` / `cannsim` 命令**（`npusim` 仅支持 Ascend 950/950DT）。Kirin 的仿真入口是算子工程内的一键脚本 `run.sh`：

```bash
bash run.sh -r sim -v Kirin9030     # 或 -v KirinX90
```

仿真命令在**你的算子工程目录**下执行；工程来自 `ascendc-direct-invoke-template` skill 的 `references/kirin_add_template/` 模板。本 skill 本身不提供 run.sh。

昇腾 NPU（Ascend 950/950DT）的 `npusim` 精度/性能仿真与 `summary.json` 性能分析归 `ops-simulator` skill，与本 skill 无关。

## 使用约束

| 约束项 | 说明 |
|--------|------|
| 入口命令 | 不支持 `npusim` / `cannsim`，使用工程内 `bash run.sh -r sim -v Kirinxxx` |
| SoC 版本 | 仅 `Kirin9030` / `KirinX90` 合法，大小写敏感 |
| 运行模式 | 无硬件用 `sim`，不支持 `cpu`；`npu` 需真机 |
| 环境前置 | 需 mobile-station 版 CANN（含 `simulator/<plat>/lib/` 仿真库），服务器版 CANN 不含 Kirin Simulator |

## 输出与数据口径

| 维度 | 来源 | 说明 |
|------|------|------|
| 精度 | `output/*.bin` | run.sh 末尾 `verify_result.py` 与 golden 比对，输出 error ratio |
| 性能 | sim 运行 stdout（tick/cycle）+ `sim_log` | 从运行日志解析；确定性仿真，多轮结果一致（MAD≈0） |
| **不产生** | `summary.json` / `trace_core*.json` | npusim 的报告产物 Kirin **没有**，相应性能分析章节不适用 |

## 常见问题

| 问题 | 解决 |
|------|------|
| `npusim: command not found` 或报不支持该 SoC | 正常，Kirin 不用 npusim，改用 `bash run.sh -r sim -v Kirinxxx` |
| simulator 库缺失 | 确认装了 mobile-station 版 CANN，`simulator/Kirin9030/lib/` 存在 |
| `GLIBCXX_3.4.32 not found` | 放 `ext_lib/libstdc++.so.6.0.34`，run.sh 会 `LD_PRELOAD` 加载 |
| SoC 名拼写 | 仅 `Kirin9030` / `KirinX90` 合法，大小写敏感 |

## 参考资料

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| `references/kirin-simulator-guide.md` | Kirin 平台 Simulator（run.sh -r sim）使用指导 | 需要环境前置细节、run.sh 参数与自动流程、与 npusim 差异对照时 |
