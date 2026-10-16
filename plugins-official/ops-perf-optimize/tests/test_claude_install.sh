#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_ROOT="$(mktemp -d /tmp/ops-perf-claude-install.XXXXXX)"
TEST_HOME="$TEST_ROOT/home"
TEST_PROJECT="$TEST_ROOT/project"

cleanup() {
    rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

# Keep the installer offline and verify that unrelated Claude settings and
# discovery entries survive repeated manifest-based installs.
mkdir -p \
    "$TEST_HOME/.claude/skills" \
    "$TEST_PROJECT/asc-devkit" \
    "$TEST_PROJECT/cann-samples" \
    "$TEST_PROJECT/.claude/agents" \
    "$TEST_PROJECT/.claude/skills/user-skill"

printf '%s\n' '{"theme":"dark","permissions":{"allow":["Read"]}}' \
    > "$TEST_PROJECT/.claude/settings.json"
printf '%s\n' '---' 'name: user-skill' 'description: preserved test skill' '---' \
    > "$TEST_PROJECT/.claude/skills/user-skill/SKILL.md"
printf '%s\n' '---' 'name: user-agent' 'description: preserved test agent' '---' \
    > "$TEST_PROJECT/.claude/agents/user-agent.md"

run_install() {
    (cd "$TEST_PROJECT" && HOME="$TEST_HOME" bash "$PLUGIN_ROOT/init.sh" project claude >/dev/null)
}

run_install
python3 "$SCRIPT_DIR/legacy_project_install.py" seed "$TEST_PROJECT" "$PLUGIN_ROOT" claude
run_install
python3 "$SCRIPT_DIR/legacy_project_install.py" verify "$TEST_PROJECT" "$PLUGIN_ROOT" claude

python3 - "$TEST_PROJECT" "$PLUGIN_ROOT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

project = Path(sys.argv[1])
plugin = Path(sys.argv[2])
claude = project / ".claude"

plugin_manifest = json.loads((plugin / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
assert "ops-direct-invoke-skills" in plugin_manifest["dependencies"]
assert "./AGENTS.md" in plugin_manifest["agents"]
assert not (plugin / "project-AGENTS.md").exists()

expected_skills = {
    "ops-perf-baseline-evaluate",
    "ops-perf-preparation",
    "ops-perf-analysis",
    "ascendc-performance-review",
    "ops-perf-implementation",
    "ops-perf-postprocess",
    "ops-perf-acceptance-criteria",
    "ops-perf-profile-feature-extractor",

    "ops-perf-optimize-workflow",
    "ops-simulator",
    "ops-profiling",
    "ascendc-perf-optimize",
    "ascendc-performance-best-practices",
    "ascendc-blaze-best-practice",
    "ascendc-blaze-migration",
    "ascendc-regbase-best-practice",
    "ascendc-docs-search",
    "ascendc-env-check",
    "npu-arch",
    "ascendc-st-design",
    "ascendc-precision-debug",
    "ascendc-sync-audit",
    "ascendc-mc2-best-practice",
    "ops-precision-standard",
    "ascendc-registry-invoke-template",
    "ops-theoretical-performance-estimation",
}
# Stages dispatch to the generic subagent; only the primary agent is installed.
expected_agents = {
    "perf-optimize.md",
}

assert not (claude / "CLAUDE.md").exists()
assert (claude / "workflows").resolve() == (plugin / "workflows").resolve()
assert (claude / "cannbot-ops-perf-optimize-tools").resolve() == (plugin / "scripts").resolve()
assert (claude / "cannbot-ops-perf-optimize-tools" / "workflow_state.py").is_file()
assert (claude / "cannbot-ops-perf-optimize-tools" / "prepare_case_sets.py").is_file()
assert (claude / "cannbot-ops-perf-optimize-tools" / "validate_implementation_conformance.py").is_file()
assert (claude / "cannbot-ops-perf-optimize-tools" / "validate_integration_safety.py").is_file()
assert (claude / "cannbot-ops-perf-optimize-tools" / "validate_source_isolation.py").is_file()
for name in expected_agents:
    assert (claude / "agents" / name).resolve().is_file(), name
assert (claude / "agents" / "perf-optimize.md").resolve() == (plugin / "AGENTS.md").resolve()
assert (claude / "agents" / "user-agent.md").is_file()
assert (claude / "skills" / "user-skill" / "SKILL.md").is_file()

for name in expected_skills:
    assert (claude / "skills" / name / "SKILL.md").is_file(), name

settings = json.loads((claude / "settings.json").read_text(encoding="utf-8"))
assert settings["theme"] == "dark"
assert settings["permissions"] == {"allow": ["Read"]}
assert "agent" not in settings

manifest = json.loads((claude / "cannbot-manifest.json").read_text(encoding="utf-8"))
assert Path(manifest["implementation_conformance_tool"]).name == "validate_implementation_conformance.py"
assert Path(manifest["integration_safety_tool"]).name == "validate_integration_safety.py"
assert Path(manifest["case_set_tool"]).name == "prepare_case_sets.py"
assert manifest["tool"] == "claude"
assert manifest["managed_default_agent"] is False
assert set(manifest["installed_skills"]) == expected_skills
assert set(manifest["installed_agents"]) == expected_agents

wrapper = (claude / "agents" / "perf-optimize.md").read_text(encoding="utf-8")
entry_skill = (claude / "skills" / "ops-perf-optimize-workflow" / "SKILL.md").read_text(encoding="utf-8")
primary = (claude / "skills" / "ops-perf-optimize-workflow" / "references" / "workflow-contract.md").read_text(encoding="utf-8")
assert re.search(r"^name:\s*perf-optimize\s*$", wrapper, flags=re.M)
assert "skills/ops-perf-optimize-workflow/SKILL.md" in wrapper
assert "references/workflow-contract.md" in entry_skill
assert "唯一工作流真源" in primary
freezing = (claude / "skills" / "ops-perf-baseline-evaluate/references/input-freezing.md").read_text(encoding="utf-8")
generation = (claude / "skills" / "ops-perf-baseline-evaluate/references/case-generation.md").read_text(encoding="utf-8")
selection = (claude / "skills" / "ops-perf-implementation/references/selection-and-final.md").read_text(encoding="utf-8")
implementation = (claude / "skills" / "ops-perf-implementation/references/implementation-execution.md").read_text(encoding="utf-8")
prompts = (claude / "workflows" / "task-prompts.md").read_text(encoding="utf-8")
assert "必须保留的稳定派发入口" in prompts
assert "唯一工作流真源" in prompts
for role_template in ("baseline", "preparation", "generation", "analysis", "implementation", "integration"):
    assert (claude / "workflows" / "prompts" / f"{role_template}.md").is_file()
assert not (claude / "workflows" / "resume-and-lessons.md").exists()
assert "禁止默认取前 N" in freezing
assert "coverage-v1" in freezing
assert "performance-full-cases.csv" in freezing
assert "performance-tuning-cases.csv" in freezing
assert "functional-acceptance-cases.csv" in freezing
for mode in ("reuse_performance", "generate", "existing"):
    assert mode in freezing
assert "eligible_for_performance" in implementation
assert "身份与覆盖完全相同时复用已有证据" in primary
assert "final-functional-regression.md" in selection
assert "final-performance-comparison.md" in selection
assert "known_baseline_failure" in primary
assert "精度诊断没有自动停止预算" in primary
assert "仅当用户明确要求" in entry_skill
assert "一般咨询、cases.csv 解读" in entry_skill
hook = (plugin / "hooks" / "session-start-ops-perf-optimize").read_text(encoding="utf-8")
assert "only when the user explicitly asks" in hook
assert "merely mention cases.csv" in hook
impl = (claude / "skills" / "ops-perf-implementation" / "SKILL.md").read_text(encoding="utf-8") + (claude / "skills/ops-perf-implementation/references/implementation-execution.md").read_text(encoding="utf-8")
assert "functional-acceptance-cases.csv" in impl
assert "通过功能/序列门禁后采性能" in impl
project_skill = (claude / "skills" / "ops-perf-postprocess" / "SKILL.md").read_text(encoding="utf-8")
assert "disable-model-invocation:" not in project_skill
integrate = (claude / "skills" / "ops-perf-postprocess" / "references" / "writeback.md").read_text(encoding="utf-8")
assert "两套全集性能 gate" in integrate

print("Claude static install contract: PASS")
PY

# A v2 reinstall removes only the default-agent setting previously owned by
# this plugin. v3 must not make perf-optimize the default again.
LEGACY_ROOT="$TEST_ROOT/legacy-default"
mkdir -p "$LEGACY_ROOT/asc-devkit" "$LEGACY_ROOT/cann-samples" "$LEGACY_ROOT/.claude"
printf '%s\n' '{"theme":"dark","agent":"perf-optimize"}' > "$LEGACY_ROOT/.claude/settings.json"
printf '%s\n' \
    '{' \
    '  "tool": "claude",' \
    '  "managed_default_agent": true,' \
    '  "installed_skills": [],' \
    '  "installed_agents": []' \
    '}' > "$LEGACY_ROOT/.claude/cannbot-manifest.json"
(cd "$LEGACY_ROOT" && HOME="$TEST_HOME" bash "$PLUGIN_ROOT/init.sh" project claude >/dev/null)
python3 - "$LEGACY_ROOT" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1]) / ".claude"
settings = json.loads((root / "settings.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "cannbot-manifest.json").read_text(encoding="utf-8"))
assert settings == {"theme": "dark"}
assert manifest["managed_default_agent"] is False
print("Claude legacy default-agent cleanup: PASS")
PY

# A different personal skill with the same name must fail loudly because Claude
# gives it precedence over the project-installed skill.
CONFLICT_ROOT="$TEST_ROOT/conflict"
mkdir -p \
    "$CONFLICT_ROOT/asc-devkit" \
    "$CONFLICT_ROOT/cann-samples" \
    "$TEST_HOME/.claude/skills/ascendc-st-design"
printf '%s\n' '---' 'name: ascendc-st-design' 'description: conflicting test skill' '---' \
    > "$TEST_HOME/.claude/skills/ascendc-st-design/SKILL.md"
if (cd "$CONFLICT_ROOT" && HOME="$TEST_HOME" bash "$PLUGIN_ROOT/init.sh" project claude >/dev/null 2>&1); then
    echo "Expected personal skill precedence conflict to fail" >&2
    exit 1
fi

echo "Claude precedence guard: PASS"
