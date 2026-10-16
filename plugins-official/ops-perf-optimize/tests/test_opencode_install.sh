#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_PROJECT="$(mktemp -d /tmp/ops-perf-opencode-install.XXXXXX)"

cleanup() {
    rm -rf "$TEST_PROJECT"
}
trap cleanup EXIT

mkdir -p "$TEST_PROJECT/asc-devkit" "$TEST_PROJECT/cann-samples"
printf '%s\n' '# user instructions' '' 'preserve me' > "$TEST_PROJECT/AGENTS.md"

run_install() {
    (cd "$TEST_PROJECT" && bash "$PLUGIN_ROOT/init.sh" project opencode >/dev/null)
}

run_install
python3 "$SCRIPT_DIR/legacy_project_install.py" seed "$TEST_PROJECT" "$PLUGIN_ROOT" opencode
run_install
python3 "$SCRIPT_DIR/legacy_project_install.py" verify "$TEST_PROJECT" "$PLUGIN_ROOT" opencode

python3 - "$TEST_PROJECT" "$PLUGIN_ROOT" <<'PY'
from pathlib import Path
import sys

project = Path(sys.argv[1])
plugin = Path(sys.argv[2])
opencode = project / ".opencode"
trigger = (project / "AGENTS.md").read_text(encoding="utf-8")

assert trigger.startswith("# user instructions\n\npreserve me\n")
assert trigger.count("<!-- CANNBOT:ops-perf-optimize:begin -->") == 1
assert trigger.count("<!-- CANNBOT:ops-perf-optimize:end -->") == 1
assert "必须调用 `ops-perf-optimize-workflow` Skill" in trigger
assert "仅当用户明确要求对某个具体 AscendC 算子或代码目录" in trigger
assert "只提到 cases.csv、msprof、profiling、加速比" in trigger
assert "不加载 Skill、不创建运行目录、不派发专家" in trigger
assert str(opencode / "skills" / "ops-perf-optimize-workflow" / "SKILL.md") in trigger
assert str(opencode / "agents" / "perf-optimize.md") not in trigger
assert str(opencode / "workflows" / "task-prompts.md") not in trigger
assert str(opencode / "workflows" / "resume-and-lessons.md") not in trigger
for role_template in ("baseline", "preparation", "generation", "analysis", "implementation", "integration"):
    assert (opencode / "workflows" / "prompts" / f"{role_template}.md").is_file()
assert "项目入口不复制或重构工作流细节" in trigger
assert "precision-case-selection.json" not in trigger
assert "门禁 G" not in trigger
assert (opencode / "agents" / "perf-optimize.md").resolve() == (plugin / "AGENTS.md").resolve()
assert (opencode / "skills" / "ops-perf-optimize-workflow" / "SKILL.md").resolve() == (plugin / "skills" / "ops-perf-optimize-workflow" / "SKILL.md").resolve()
impl = (opencode / "skills" / "ops-perf-implementation" / "SKILL.md").read_text(encoding="utf-8") + (opencode / "skills/ops-perf-implementation/references/implementation-execution.md").read_text(encoding="utf-8")
assert "不派发子 Agent" in impl or "不再派发子 Agent" in impl
assert not (plugin / "project-AGENTS.md").exists()

print("OpenCode generated trigger contract: PASS")
PY
