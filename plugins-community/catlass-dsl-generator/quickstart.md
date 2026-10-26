# CANNBot · Catlass DSL 算子开发快速入门

## 概述

`catlass-dsl-generator` 为 Ascend NPU CATLASS DSL 算子开发提供完整的工具链，覆盖设计、编码、基准测试、性能优化与知识管理五大阶段。内置**离线 OKF 知识库**，支持 API 查询、样例检索、问题排查与性能调优，无需网络连接。

---

## 一、安装

> 将 `<plugin-source>` 替换为本仓库的本地绝对路径、`owner/repo` 或 Git URL。安装前请确认来源可信。

### 方式 A：init.sh 脚本安装（推荐）

适用于 OpenCode、Claude Code、Trae（IDE/插件版/CLI）、Cursor、Codex、Copilot、CodeArts，
通过软链把 5 个 Skill 和内置 OKF 知识库注册到工具的项目级/全局发现路径，并生成
自动加载的工作流配置（`AGENTS.md` / `CLAUDE.md`）——新会话不依赖对话历史即可恢复工作流。

```bash
# 项目级安装（在目标项目根目录执行，tool 换成你在用的工具）
bash <plugin-source>/init.sh project trae

# 全局安装（Trae 仅支持项目级）
bash <plugin-source>/init.sh global claude

# 卸载（仅移除本插件安装的白名单内容；参数与安装时一致）
bash <plugin-source>/init.sh --uninstall project trae
```

各工具的安装路径与差异：

| Tool | Skills/知识库路径 | 配置文件 | 级别 |
|:-----|:-----------------|:---------|:-----|
| OpenCode | `.opencode/{skills,knowledge}/` | `AGENTS.md` | 项目/全局 |
| Claude Code | `.claude/{skills,knowledge}/` | `CLAUDE.md` | 项目/全局 |
| Trae | `.trae/`（IDE，自动检测 `.marscode/`、`.traecli/` 变体） | `AGENTS.md` | 仅项目级 |
| Cursor | `.cursor/{skills,knowledge}/` | `AGENTS.md` | 项目/全局 |
| Codex | `.agents/{skills,knowledge}/` | `AGENTS.md` | 项目/全局 |
| Copilot | `.github/{skills,knowledge}/` | `AGENTS.md` | 项目/全局 |
| CodeArts | `.codeartsdoer/{skills,knowledge}/` | `AGENTS.md` | 项目/全局 |

说明：
- 全部为软链安装，插件仓库仍是唯一源，`git pull` 更新后自动生效。
- 同一项目安装多个工具时，配置文件以最后一次安装为准。
- 卸载只删除本插件的白名单内容，不影响项目里其他 skills。

### 方式 B：工具原生插件市场

#### Codex

```bash
codex plugin marketplace add <plugin-source>
codex plugin add catlass-dsl-generator@catlass-dsl-generator-dev
```

运行 `codex plugin list` 确认插件存在，然后新建会话以加载 Skill。

#### Claude Code

```bash
claude plugin marketplace add <plugin-source>
claude plugin install catlass-dsl-generator@catlass-dsl-generator-dev
```

重启 Claude Code，或在会话中运行 `/reload-plugins`。本地临时试用：

```bash
claude --plugin-dir <local-plugin-path>
```

#### Cursor

将完整仓库放到 Cursor 的本地插件目录：

```bash
mkdir -p ~/.cursor/plugins/local
git clone <git-repository-url> ~/.cursor/plugins/local/catlass-dsl-generator
```

已有本地 checkout 时也可直接链接到 `~/.cursor/plugins/local/catlass-dsl-generator`。随后执行 `Developer: Reload Window`，在 `Settings > Plugins` 中确认插件已加载。

> GitHub 仓库也可在 Cursor Agent 中使用 `/add-plugin <git-repository-url>` 导入。

#### OpenCode

在目标项目的 `opencode.json` 中添加：

```json
{
  "plugin": ["catlass-dsl-generator@git+<git-repository-url>"]
}
```

用户级安装则修改 `~/.config/opencode/opencode.json`。重启 OpenCode 后使用原生 `skill` 工具确认五个 Skill 均可发现。更多说明见 [`.opencode/INSTALL.md`](.opencode/INSTALL.md)。

---

## 二、快速上手

开发新算子只需两步：

> **第一步** — 使用 `catlass-dsl-design` 生成算子规格、初步设计、Torch
> `reference.py` 精度标杆、`definition.json` 和 `workload.jsonl` 用例：

```text
使用 catlass-dsl-design 设计一个 <算子名称> 算子。
输入、输出、dtype、shape、数值语义和性能目标是：...
```

> **第二步** — 检查生成的可读 `DESIGN.md`，确认接口、语义、允许路径和测试方法后批准。
> 严格的命令 argv 和状态机字段直接冻结在 develop run 根目录的 `state.json`：

```text
我批准 DESIGN.md，请使用 catlass-dsl-develop 按设计完成开发。
```

`develop` 会读取与 `DESIGN.md` 摘要绑定的 state config，自动完成**任务拆分 → 实现 → 调试 → Review → 测试 → Benchmark/Profiling**全流程。最终交付始终包含当前实现的性能现状和 candidate profiler 原始数据；仅当设计声明了性能门槛且初始实现未达标时，才会进入 optimize 阶段。

---

## 三、可用技能

| Skill | 用途 | 示例 |
|:------|:-----|:-----|
| [`catlass-dsl-design`](skills/catlass-dsl-design/SKILL.md) | 生成规格、Torch 精度标杆和 workload 用例 | `设计一个 float16 masked softmax 算子` |
| [`catlass-dsl-develop`](skills/catlass-dsl-develop/SKILL.md) | 按已批准的设计完成开发 | `按照 DESIGN.md 实现并完成测试` |
| [`catlass-dsl-bench`](skills/catlass-dsl-bench/SKILL.md) | 独立验证正确性与测量性能 | `用 solution、workload 和 definition 执行 benchmark` |
| [`catlass-dsl-optimize`](skills/catlass-dsl-optimize/SKILL.md) | 独立优化已有算子 | `优化 matmul，目标 mean_ms < 0.5` |
| [`catlass-dsl-knowledge`](skills/catlass-dsl-knowledge/SKILL.md) | 查询或维护本地 OKF 知识 | `查询 layout 不一致的排查方法` |

> **提示：** `debug` 和 `review` 已内置在 `develop` 流程中，无需单独调用。批准后的 `DESIGN.md` 不应直接修改；需求变化时请建立新的 develop run 并重新批准 Resolved Plan。

---

## 四、独立使用

### Benchmark

三个输入文件的接口定义见 [`catlass-dsl-bench/SKILL.md`](skills/catlass-dsl-bench/SKILL.md)。
默认 `fusion.mode=performance`，design 以整体性能选择融合范围，允许多个 DSL kernel、
executor 和 GM 中间张量。benchmark 使用 `--anti-hack off`，关闭全部 anti-hack，
结果显式记录 `anti_hack.status=disabled`；正确性和 profiling 门禁仍保留。
只有用户明确要求单个融合 kernel 时采用 `single_kernel`，benchmark/optimize 使用
`--anti-hack single-fused`：检查唯一源码声明、每次单 launch 及审计证据。

```bash
python3 skills/catlass-dsl-bench/scripts/bench.py \
  --solution skills/catlass-dsl-bench/templates/solution.json \
  --workload skills/catlass-dsl-bench/templates/workload.jsonl \
  --definition skills/catlass-dsl-bench/templates/definition.json \
  --output .catlass-dsl/runs/example/benchmark
```

### Optimize

调用 optimize 时需提供算子路径、测试命令、benchmark 指标、性能目标和迭代上限：

```text
使用 catlass-dsl-optimize 优化 src/matmul.py。
完整测试命令为 ...，benchmark 指标为 performance.candidate.mean_ms，
目标小于 0.5 ms，最多迭代 8 轮。
```

### Knowledge

首次使用先初始化，之后可按关键词、标签或算子族查询：

```bash
# 初始化知识库
python3 skills/catlass-dsl-knowledge/scripts/record_knowledge.py initialize \
  --project-root <project>

# 按关键词查询
python3 skills/catlass-dsl-knowledge/scripts/record_knowledge.py query \
  --project-root <project> --text "layout"

# 按知识类别 + 算子族查询
python3 skills/catlass-dsl-knowledge/scripts/record_knowledge.py query \
  --project-root <project> --category optimization --operator-family matmul
```

> 知识涵盖 DSL API、算子样例、调试方法、优化技巧及项目实测经验。查询仅读取本地文件；来源链接仅用于归因和可选审计。

---

## 五、使用要求

- 直接在当前 **Git workspace** 中运行，并明确允许修改的路径
- 测试、benchmark 和 profiling 命令应能从目标项目直接执行
- 融合范围由 design 按整体性能确定，NPU 结果包含
  profiling CSV 证据；严格模式额外要求 `anti_hack.status=passed`
- 缺少 CANN、NPU 或 CATLASS DSL 环境时，相关结果为 `not_run`，不会被视作通过

---

## 六、插件自检

```bash
python3 -m pytest
python3 -m compileall -q skills
```


## 完整生成与后续调试

develop 默认从 implement 开始，一次生成主 Python 文件内的全部 kernel、host 编排及
完整 `solution.json`，不拆任务、不逐任务上板。生成只检查语法和源码包一致性，不要求
本地安装 NPU/CATLASS；生成完成不代表验证通过。

随后进入 focused_test；编译、运行、精度或评审失败时先查知识库、读取采用条目，再
增量修改并重建 `solution.json`，回到验证。环境缺失记录 blocked/not_run，保留代码及
待执行阶段，环境就绪后直接继续。连接与同步由使用者现有方式处理。

开发状态使用 `catlass.dsl.workflow.v4`，优化状态使用 `catlass.dsl.optimization.v5`，
benchmark 结果为 version 4。旧 run 不自动迁移，需要重新创建设计配置。

真实 NPU 冒烟通过 `CATLASS_DSL_BENCH_SOLUTION`、`CATLASS_DSL_BENCH_WORKLOAD`、
`CATLASS_DSL_BENCH_DEFINITION` 提供绝对路径，默认方案可包含多个 kernel。
另设 `CATLASS_DSL_BENCH_SINGLE_SOLUTION` 指向同一算子的单融合方案，即可同时覆盖
严格模式；无对应配置时各自标记 skipped，不伪造硬件验证结果。

## 调试验收去重

- focused_test 的执行器报告覆盖全部 required cases 和批准命令时，评审后直接 benchmark，
  不重复 full_test；部分覆盖仍补跑全测。
- optimize 初始化可传 `--verification-context`。其 finalize 保留完整验收记录后，develop
  通过 optimize submission 的 `optimize_result` 接收，只补 final_review；条件不变则 finish。
- 复用绑定源码、规格、命令与脚本、设备环境和测量参数；当前快照、报告或摘要不完整时
  继续原验证流程，finish 前再次检查。

环境快照、逐 case 报告、上下文导出及 submission 格式见
[证据复用协议](skills/catlass-dsl-develop/references/verification-reuse.md)。旧 run 的状态版本
保持不变，缺少结构化证据时不自动跳过门禁。
