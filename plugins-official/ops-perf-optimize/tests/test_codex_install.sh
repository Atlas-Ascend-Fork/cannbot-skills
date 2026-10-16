#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_PROJECT="$(mktemp -d /tmp/ops-perf-codex-install.XXXXXX)"

cleanup() {
    rm -rf "$TEST_PROJECT"
}
trap cleanup EXIT

# Keep the installer offline and exercise preservation of unrelated project
# configuration in Codex's shared discovery directories.
mkdir -p \
    "$TEST_PROJECT/asc-devkit" \
    "$TEST_PROJECT/cann-samples" \
    "$TEST_PROJECT/.agents/skills/user-skill" \
    "$TEST_PROJECT/.codex/agents"

printf '%s\n' \
    '# user instructions' \
    '' \
    '# 工作流加载规则' \
    '' \
    '旧版步骤：Step 0 → Step 1 → Step 2' \
    '' \
    '详见 `.opencode/agents/perf-optimize.md`，本文件仅为触发入口。' \
    > "$TEST_PROJECT/AGENTS.md"
printf '%s\n' \
    'model = "user-model-sentinel"' \
    '' \
    '[user_settings]' \
    'preserve = true' \
    '' \
    '[agents]' \
    'enabled = false' \
    'max_concurrent_threads_per_session = 2' \
    > "$TEST_PROJECT/.codex/config.toml"
printf '%s\n' '---' 'name: user-skill' 'description: preserved test skill' '---' > "$TEST_PROJECT/.agents/skills/user-skill/SKILL.md"
printf '%s\n' \
    'name = "user-agent"' \
    'description = "preserved test agent"' \
    'developer_instructions = "Preserve this unrelated agent."' \
    > "$TEST_PROJECT/.codex/agents/user-agent.toml"

run_install() {
    (cd "$TEST_PROJECT" && bash "$PLUGIN_ROOT/init.sh" project codex >/dev/null)
}

# Run twice to cover manifest-based replacement and managed AGENTS.md merging.
run_install
python3 "$SCRIPT_DIR/legacy_project_install.py" seed "$TEST_PROJECT" "$PLUGIN_ROOT" codex
run_install
python3 "$SCRIPT_DIR/legacy_project_install.py" verify "$TEST_PROJECT" "$PLUGIN_ROOT" codex

python3 - "$TEST_PROJECT" "$PLUGIN_ROOT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import tomllib

project = Path(sys.argv[1])
plugin = Path(sys.argv[2])
codex = project / ".codex"
skill_root = project / ".agents" / "skills"

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
# Stages dispatch to the generic subagent; no per-stage agents are installed.
expected_agents = set()

# Codex runtime files must be generated from the common cross-client sources;
# the repository must not grow a second hand-maintained role/template tree.
assert not list((plugin / "agents" / "codex").glob("*.toml"))
assert not (plugin / "project-AGENTS-codex.md").exists()
assert not (plugin / "project-AGENTS.md").exists()

for role in [plugin / "AGENTS.md", *sorted((plugin / "agents").glob("ascendc-perf-*.md"))]:
    frontmatter = role.read_text(encoding="utf-8").split("---", 2)[1]
    match = re.search(r"^skills:\n((?:  - .+\n)+)", frontmatter, flags=re.M)
    if match:
        role_skills = {line.removeprefix("  - ") for line in match.group(1).splitlines()}
        assert role_skills <= expected_skills, (role, role_skills - expected_skills)
manifest = json.loads((codex / "cannbot-manifest.json").read_text(encoding="utf-8"))
assert manifest["tool"] == "codex"
assert Path(manifest["main_workflow"]) == skill_root / "ops-perf-optimize-workflow" / "SKILL.md"
assert Path(manifest["workflow_contract"]) == skill_root / "ops-perf-optimize-workflow" / "references" / "workflow-contract.md"
assert Path(manifest["case_set_tool"]).name == "prepare_case_sets.py"
assert Path(manifest["implementation_conformance_tool"]).name == "validate_implementation_conformance.py"
assert Path(manifest["integration_safety_tool"]).name == "validate_integration_safety.py"
assert Path(manifest["source_isolation_tool"]).is_file()
assert Path(manifest["performance_effects_tool"]).name == "summarize_performance_effects.py"
assert set(manifest["installed_skills"]) == expected_skills
assert {Path(name).stem for name in manifest["installed_agents"]} == expected_agents
assert "user-skill" not in manifest["installed_skills"]
assert "user-agent.toml" not in manifest["installed_agents"]

for name in expected_skills:
    assert (skill_root / name / "SKILL.md").is_file(), name
for name in ("ops-perf-optimize-workflow", "ops-perf-postprocess"):
    path = skill_root / name
    assert not path.is_symlink(), name
    assert (path / ".cannbot-ops-perf-optimize").is_file(), name
assert (skill_root / "user-skill" / "SKILL.md").is_file()

config = tomllib.loads((codex / "config.toml").read_text(encoding="utf-8"))
assert config["model"] == "user-model-sentinel"
assert config["user_settings"]["preserve"] is True
assert config["agents"]["enabled"] is True
assert config["agents"]["max_concurrent_threads_per_session"] == 4

assert (codex / "agents" / "user-agent.toml").is_file()
assert not (codex / "agents" / "general_purpose.toml").exists()

agents_md = (project / "AGENTS.md").read_text(encoding="utf-8")
assert agents_md.startswith("# user instructions\n")
assert "旧版步骤" not in agents_md
assert agents_md.count("# 工作流加载规则") == 1
assert "项目入口不复制或重构工作流细节" in agents_md
assert agents_md.count("<!-- CANNBOT:ops-perf-optimize:begin -->") == 1
assert agents_md.count("<!-- CANNBOT:ops-perf-optimize:end -->") == 1
assert "必须调用 `ops-perf-optimize-workflow` Skill" in agents_md
assert "仅当用户明确要求对某个具体 AscendC 算子或代码目录" in agents_md
assert "只提到 cases.csv、msprof、profiling、加速比" in agents_md
assert "不加载 Skill、不创建运行目录、不派发专家" in agents_md
assert str(skill_root / "ops-perf-optimize-workflow" / "SKILL.md") in agents_md
assert str(codex / "perf-optimize.md") not in agents_md
assert str(codex / "workflows" / "task-prompts.md") not in agents_md
assert str(codex / "workflows" / "resume-and-lessons.md") not in agents_md
assert str(skill_root) in agents_md
assert "precision-case-selection.json" not in agents_md
assert "门禁 G" not in agents_md

assert (codex / "perf-optimize.md").resolve() == (plugin / "AGENTS.md").resolve()
assert (codex / "workflows").resolve() == (plugin / "workflows").resolve()
assert (codex / "cannbot-ops-perf-optimize-tools").resolve() == (plugin / "scripts").resolve()
assert (codex / "cannbot-ops-perf-optimize-tools" / "workflow_state.py").is_file()
assert (codex / "cannbot-ops-perf-optimize-tools" / "prepare_case_sets.py").is_file()
assert (codex / "cannbot-ops-perf-optimize-tools" / "validate_implementation_conformance.py").is_file()
assert (codex / "cannbot-ops-perf-optimize-tools" / "validate_integration_safety.py").is_file()
assert (codex / "cannbot-ops-perf-optimize-tools" / "summarize_performance_effects.py").is_file()
assert (
    skill_root
    / "ops-perf-acceptance-criteria"
    / "references"
    / "performance-jitter-policy.md"
).is_file()
wrapper = (codex / "perf-optimize.md").read_text(encoding="utf-8")
entry_skill = (skill_root / "ops-perf-optimize-workflow" / "SKILL.md").read_text(encoding="utf-8")
primary = (skill_root / "ops-perf-optimize-workflow" / "references" / "workflow-contract.md").read_text(encoding="utf-8")
assert "skills/ops-perf-optimize-workflow/SKILL.md" in wrapper
assert "references/workflow-contract.md" in entry_skill
assert "唯一工作流真源" in primary
freezing = (skill_root / "ops-perf-baseline-evaluate/references/input-freezing.md").read_text(encoding="utf-8")
generation = (skill_root / "ops-perf-baseline-evaluate/references/case-generation.md").read_text(encoding="utf-8")
selection = (skill_root / "ops-perf-implementation/references/selection-and-final.md").read_text(encoding="utf-8")
implementation = (skill_root / "ops-perf-implementation/references/implementation-execution.md").read_text(encoding="utf-8")
assert "仅当用户明确要求" in entry_skill
assert "一般咨询、cases.csv 解读" in entry_skill
prompts = (codex / "workflows" / "task-prompts.md").read_text(encoding="utf-8")
assert "必须保留的稳定派发入口" in prompts
assert "唯一工作流真源" in prompts
for role_template in ("baseline", "preparation", "generation", "analysis", "implementation", "integration"):
    assert (codex / "workflows" / "prompts" / f"{role_template}.md").is_file()
assert not (codex / "workflows" / "resume-and-lessons.md").exists()
assert "统一派发给 Codex 通用子 agent" in agents_md
assert "`general-purpose` → Codex 内置 agent `worker`" not in agents_md
assert 'fork_turns="none"' in agents_md
assert "full-history fork" in agents_md
assert "禁止默认取前 N" in freezing
assert "coverage-v1" in freezing
assert "performance-full-cases.csv" in freezing
assert "performance-tuning-cases.csv" in freezing
assert "functional-acceptance-cases.csv" in freezing
for mode in ("reuse_performance", "generate", "existing"):
    assert mode in freezing
assert "最多 40" in generation
assert "eligible_for_performance" in implementation
assert "身份与覆盖完全相同时复用已有证据" in primary
assert "原始基线和最终实现分别在性能全集上最多采一次" in selection
assert "workflow-state.json" in primary
assert "dead-code-audit.md" in implementation
assert "roundN/analysis.md" in primary
assert "final-functional-regression.md" in selection
assert "final-performance-comparison.md" in selection
assert "known_baseline_failure" in primary
assert "精度诊断没有自动停止预算" in primary
assert "不另建进度 JSON、双份多轮汇总或逐轮同名报告" in primary

case_designer_source = (plugin / "skills/ops-perf-baseline-evaluate/references/case-generation.md").read_text(encoding="utf-8")
assert "不编译、不运行 NPU" in case_designer_source
assert "最多 40 条" in case_designer_source
assert "只处理 standard 的 `FUNCTIONAL_MODE=generate`，或 generalization 的泛化用例生成" in case_designer_source

print("Codex static install contract: PASS")
PY

if command -v codex >/dev/null 2>&1; then
    (
        cd "$TEST_PROJECT"
        codex debug prompt-input "只列出已发现的技能，不执行任务。" > codex-prompt-input.json
    )
    python3 - "$TEST_PROJECT" <<'PY'
from pathlib import Path
import sys

project = Path(sys.argv[1])
prompt = (project / "codex-prompt-input.json").read_text(encoding="utf-8")
for name in (
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
    "ops-precision-standard",
    "ascendc-registry-invoke-template",
    "ops-theoretical-performance-estimation",
):
    assert f"- {name}:" in prompt, name
assert "ops-perf-optimize-workflow:ops-perf-postprocess" not in prompt
assert str(project / ".agents" / "skills" / "ops-perf-optimize-workflow" / "SKILL.md") in prompt
print("Codex runtime discovery contract: PASS")
PY
else
    echo "Codex runtime discovery contract: SKIP (codex executable not found)"
fi
