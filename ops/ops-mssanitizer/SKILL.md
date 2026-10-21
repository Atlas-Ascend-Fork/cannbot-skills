---
name: ops-mssanitizer
description: 使用mssanitizer对Ascend C和Triton算子执行内存访问、泄漏、未初始化、竞争与同步检查，自动识别工程类型，运行可复现用例并生成结构化报告。适用于开发完成后的主动内存安全验证和已知内存异常复核；不用于一般崩溃、性能或精度问题诊断。
---

# 算子 mssanitizer 内存检测

对目标算子执行可复现的内存安全检查，并区分“检测通过”“检测到问题”和“工具未完成”。不要把工具执行失败、测试未命中目标算子或空日志误判为通过。

## 执行原则

- 优先复用目标算子已有且能够稳定运行的UT/ST或pytest用例；确认用例实际调用目标kernel，不得用其他实现fallback后宣称目标算子通过。
- CANN路径来自用户参数或当前环境的`ASCEND_HOME_PATH`。不要写死用户名、conda环境、安装目录或版本。
- 只有mssanitizer退出码为0且未发现结构化错误条目时，检测项才可标记为通过。
- 非零退出码表示检测未完成；即使日志为空，也必须标记为失败或未完成并保留退出码。
- 一次只检测用户指定的工程或脚本。生成日志、构建目录或测试脚本前，说明输出位置并避免覆盖已有文件。

## 参考资料路由

- Ascend C工程识别、测试载体和常见错误模式：阅读[Ascend C最佳实践](references/ascendc_best_practices.md)。
- Triton环境变量、pytest选择和kernel过滤：阅读[Triton最佳实践](references/triton_best_practices.md)。
- 解释具体日志字段、严重程度和修复顺序：阅读[错误类型详解](references/error_types.md)。

只读取当前任务需要的分支资料，不要把两套完整流程同时加载。

## 工作流

### 1. 确认目标和检测边界

收集并验证：

- 算子工程目录或可运行测试脚本；
- 目标CANN根目录；
- NPU设备可用性；
- 希望运行的工具，默认是`memcheck`、`racecheck`、`initcheck`和`synccheck`；
- Triton场景下可选的`kernel-name`。

用户只要求分析已有日志时，不启动新检测；直接使用日志解析器并在报告中注明未重新执行工具。

### 2. 识别算子类型

统一入口按以下信号路由：

| 信号 | 类型 | 执行方式 |
|---|---|---|
| 工程含`op_graph/`或`examples/test_geir_*.cpp` | Ascend C ops工程 | C++/GE IR |
| 工程含`examples/test_aclnn_*.cpp` | Ascend C ops工程 | C++/aclnn |
| Python脚本调用`torch.ops.customize` | Ascend C自定义算子 | Python |
| Python脚本含`@triton.jit` | Triton算子 | Python/pytest |

多个信号冲突或均未命中时，停止自动执行并请用户指定类型；不要猜测。

### 3. 选择能够命中目标kernel的测试

1. 优先使用仓库已有测试，并先在不启用mssanitizer时证明用例能够运行。
2. Triton优先精确到单个pytest用例，并尽量提供`--kernel-name`，避免框架kernel污染结果。
3. Ascend C自定义算子没有可用测试时，可用生成器创建基础用例：

```bash
python3 scripts/ascendc/gen_test_script.py \
  --operator <operator_name> \
  --dtypes float16 float32 \
  --output <new_test_script.py>
```

生成器仅适用于单输入、单输出且可由随机Tensor覆盖的基础算子。复杂签名应扩展现有测试，不能通过fallback替代目标算子。

### 4. 加载用户指定的CANN环境

```bash
source <cann_root>/set_env.sh
```

若安装包使用其他环境脚本，以实际安装目录为准。随后确认：

```bash
test -n "${ASCEND_HOME_PATH:-}"
test -x "<cann_root>/tools/mssanitizer/bin/mssanitizer"
npu-smi info
```

### 5. 执行检测

优先使用统一入口：

```bash
bash scripts/run_unified_mssanitizer.sh <project_or_test_script> <cann_root>
```

需要Triton kernel过滤时，直接使用对应脚本：

```bash
bash scripts/triton/run_mssanitizer_triton.sh \
  <test_script.py> <cann_root> --kernel-name <kernel_name>
```

工具脚本职责：

| 脚本 | 职责 |
|---|---|
| `scripts/run_unified_mssanitizer.sh` | 识别工程类型并路由 |
| `scripts/ascendc/run_mssanitizer.sh` | Ascend C Python测试 |
| `scripts/ascendc/run_mssanitizer_geir.sh` | Ascend C C++测试构建与检测 |
| `scripts/triton/run_mssanitizer_triton.sh` | Triton测试与kernel过滤 |
| `scripts/common/parse_mssanitizer_log.py` | 解析日志并生成Markdown |

### 6. 判定结果

每个检测项必须同时记录：

- mssanitizer退出码；
- 结构化错误条目数，而不是包含`ERROR`文本的行数；
- 原始日志路径；
- 用例是否实际命中目标算子或kernel。

状态判定：

| 条件 | 状态 |
|---|---|
| 退出码为0，错误条目为0 | 通过 |
| 退出码为0，错误条目大于0 | 发现问题 |
| 退出码非0 | 未完成/失败，结果不完整 |
| 未证明命中目标kernel | 无效检测 |

空日志只有在退出码为0、测试确实执行且命中目标kernel时，才能作为“未发现问题”的证据。

### 7. 生成报告并复核

报告至少包含：

- 算子类型、测试入口、CANN路径和目标kernel；
- 每个工具的退出码、错误条目数和状态；
- 错误类型、地址、block/device、serialNo及访问大小；
- 根因分析、修复建议和复测命令；
- 未完成、未命中或环境缺失等限制。

修复后使用相同用例、CANN版本和检测参数复测。不能用“日志为空”替代退出码、命中证据和结构化错误计数。

## 单独解析已有日志

```bash
python3 scripts/common/parse_mssanitizer_log.py \
  <mssanitizer.log> --output <analysis_report.md>
```

## 输出位置

执行脚本默认在工程或测试脚本所在目录创建`mssanitizer_logs/`，保存原始日志和一份Markdown汇总报告。脚本返回非零表示至少一个检测项未正常完成，调用方不得将其忽略。

## 权威资料

- [CANN官方mssanitizer使用指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/devtoolquickstart/atlasopdev_16_0006.html)
