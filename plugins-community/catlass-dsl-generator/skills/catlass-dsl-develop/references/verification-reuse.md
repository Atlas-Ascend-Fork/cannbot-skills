# 测试与最终验收证据复用

默认尝试复用，缺少结构化执行证据时继续原流程。状态版本保持 workflow.v4、
optimization.v5；不将旧记录中的 passed 自动升级为可复用证据。

## 验证上下文

在执行环境采集实际的环境快照 JSON，包含 device、arch、catlass、cann、torch、python、
seed、warmup、trials 和 anti_hack。版本信息必须来自执行环境，不能填写占位值以触发复用；
anti_hack 为 off 或 single-fused。每次评审和 finish submission 的
`verification_environment` 都应提供当前快照，不能直接复制历史记录冒充当前环境。
缺少快照时不复用。

```bash
python3 skills/catlass-dsl-develop/scripts/develop_state.py export-verification-context \
  --state <develop-run>/state.json --environment <current-environment.json> \
  --output <workspace/operator>/verification-context.json
```

上下文包括批准配置摘要、reference/definition/workload 的路径和摘要、批准命令、非性能
required cases、环境和采样参数。脚本还绑定项目内命令直接引用的 Python/Shell 文件、
env.sh、pytest.ini、pyproject.toml；pytest 命令同时绑定 tests/ 下的 Python 文件和项目根
test_*.py。主实现文件单独通过 kernel SHA-256 绑定，不固化在上下文中。
对象摘要使用 verification.py 的 digest()：排序键、紧凑 JSON、UTF-8。

## focused_test 的逐 case 报告

design 在 config.focused_cases 中列出 focused case IDs，并在 commands 中批准实际
focused 命令；没有专用命令时执行批准的 full_test。验收范围仍是全部非性能 required cases。

执行器在测试结束后输出实际命令、退出码及逐 case 结果。下面是执行报告结构，只能从
真实测试结果生成，不能根据 required_cases 列表补写 passed：

```json
{
  "schema": "catlass.dsl.test-report.v1",
  "kernel_sha256": "<当前主实现文件的 SHA-256>",
  "context_sha256": "<verification-context 对象的 digest>",
  "command_id": "full-test",
  "argv": ["python3", "-m", "pytest"],
  "exit_code": 0,
  "status": "passed",
  "cases": [{"case_id": "correctness-main", "status": "passed"}]
}
```

focused_test/full_test submission 可增加以下字段（下列环境值仅为结构示意）：

```json
{
  "test_reports": ["<项目相对路径>/traces/iter-NNN-focused_test/test-report.json"],
  "verification_environment": {
    "device": "npu:0", "arch": "<实际架构>", "catlass": "<实际版本>",
    "cann": "<实际版本>", "torch": "<实际版本>", "python": "<实际版本>",
    "seed": 1, "warmup": 1, "trials": 2, "anti_hack": "off"
  }
}
```

仅当报告的实际 argv 与对应 required case 的批准命令一致、退出码为 0、全部 case 明确
passed、源码及上下文一致时，才能作为完整测试证据。部分覆盖、skipped、not_run、重复
case、任一失败、缺失报告或 submission 自行声明的覆盖均不能跳过 full_test。报告可分散
在多个命令文件中，完整报告归档到该轮 validation/，避免原始日志清理后失效。

final_review 未改代码且完整证据仍有效时，直接进入 benchmark。测试脚本、输入或环境
变化会使复用失效；没有可复用证据时仍执行 full_test。

## optimize 最终验收移交

初始化 optimize 时增加 `--verification-context <上述导出的 JSON>`，其余批准参数不变。
finalize 的 command_results[command_id].evidence 指向上述执行器 JSON 报告，而不是
仅包含 passed 的文字日志。finalize 保持完整 correctness、benchmark 和 profiling；
验证成功后在 final/result.json 增加 acceptance，在 final/acceptance/ 保留报告、完整
benchmark 和 profiling CSV。严格模式额外保留 manifest 和逐 iteration 审计 CSV。
移交记录必须来自最终 best 的 finalize，不能使用 baseline 或普通迭代。

develop 的 optimize submission 增加 `optimize_result`（项目内 final/result.json 的路径）
和当前 verification_environment。控制器检查 optimize run 已 complete、状态与最终结果
一致，复核源码、输入、命令、环境、性能目标及全部证据摘要后，直接进入 final_review。
评审未修改代码时进入 finish，不再重复 full_test 或 benchmark。

缺少上下文、旧式日志、普通轮次结果或任一不匹配均记录原因并继续原验证流程。
finish 重新检查来源文件和全部证据，归档后再被修改、删除或环境改变也必须补跑门禁。
状态和 ITERATIONS.md 保存真实来源与复用原因，不虚构新的测试执行阶段。
