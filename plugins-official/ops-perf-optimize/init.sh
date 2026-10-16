#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -e

# --- Color & output helpers ---
if [ -t 1 ]; then
  GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

ok()   { echo -e "  ${DIM}${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC}${DIM} $*${NC}"; }
info() { echo -e "  ${DIM}${CYAN}→${NC}${DIM} $*${NC}"; }
step() { echo -e "${DIM}$*${NC}"; }

optional_git_pull() {
  local repo_dir="$1"
  if command -v timeout >/dev/null 2>&1; then
    timeout 15s git -C "$repo_dir" pull --ff-only --quiet 2>/dev/null
  else
    git -C "$repo_dir" pull --ff-only --quiet 2>/dev/null
  fi
}

BRAND="cannbot"
VERSION="4.1.0"

# --- Team-specific filters ---
# Stages are dispatched to the client's generic subagent (no named expert agents);
# the stage table in workflow-contract.md maps ROLE -> stage skill.
# Union of all skills needed by the primary workflow Skill, the five stages
# (dispatched to generic subagents; separate extraction and writeback roles), the
# generalization-mode competitor benchmark, plus the explicit workflow entry skill
# (skills/ops-perf-optimize-workflow, invoked as /ops-perf-optimize-workflow). Resolved LOCAL-first: in-plugin
# skills (skills/ops-perf-optimize-workflow, skills/ops-perf-acceptance-criteria,
# skills/ascendc-performance-review)
# ship with the plugin and take priority; the rest fall back to the shared ../../ops tree.
# ascendc-blaze-best-practice and ascendc-blaze-migration serve the matmul-family Blaze
# route referenced by ascendc-performance-best-practices (loaded on demand via
# CONDITIONAL_SKILLS, not as role base skills).
# Keep this explicit list aligned with every role and prompt reference so init.sh
# installations are self-contained across clients.
INCLUDED_SKILLS="ops-perf-baseline-evaluate ops-perf-preparation ops-perf-analysis ascendc-performance-review ops-perf-implementation ops-perf-postprocess ops-perf-acceptance-criteria ops-perf-profile-feature-extractor ops-theoretical-performance-estimation ops-perf-optimize-workflow ops-simulator ops-profiling ascendc-perf-optimize ascendc-performance-best-practices ascendc-blaze-best-practice ascendc-blaze-migration ascendc-regbase-best-practice ascendc-docs-search ascendc-env-check npu-arch ascendc-st-design ascendc-precision-debug ascendc-sync-audit ascendc-mc2-best-practice ops-precision-standard ascendc-registry-invoke-template"
# OpenCode and Claude register Markdown agents by filename under agents/; the
# primary agent wrapper (AGENTS.md, mode: primary) is linked there as perf-optimize.md.
# Claude also selects it through settings.json when that does not overwrite a
# user-selected default. Codex uses project AGENTS.md for the primary agent.
# Stages dispatch to the client's generic subagent; no per-stage agent files exist.
PRIMARY_AGENT_NAME="perf-optimize"
CODEX_MANAGED_SKILL_MARKER=".cannbot-ops-perf-optimize"
CLAUDE_DEFAULT_AGENT_MANAGED=false

show_banner() {
  echo ""
  echo -e "${CYAN}"
  cat << 'BANNER'
   ____    _    _   _ _   _ ____        _
  / ___|  / \  | \ | | \ | | __ )  ___ | |_
 | |     / _ \ |  \| |  \| |  _ \ / _ \| __|
 | |___ / ___ \| |\  | |\  | |_) | (_) | |_
  \____/_/   \_\_| \_|_| \_|____/ \___/ \__|
BANNER
  echo -e "${NC}"
  echo -e "  ${BOLD}AscendC Operator Perf Optimization Team${NC}"
  echo ""
}

show_help() {
    cat << EOF
CANNBot - AscendC Operator Performance Optimization Environment Installer

Usage: init.sh [level] [tool]

Arguments:
  level   - Installation level: "project" (default) or "global"
  tool    - Target tool: "opencode" (default), "claude", "codex", or "codearts"

Options:
  --help  - Show this help message

Examples:
  cd /path/to/your/project
  bash /path/to/ops-perf-optimize/init.sh                    # Project-level, OpenCode
  bash /path/to/ops-perf-optimize/init.sh project claude     # Project-level, Claude Code
  bash /path/to/ops-perf-optimize/init.sh global claude      # Global-level, Claude Code
  bash /path/to/ops-perf-optimize/init.sh project codex      # Project-level, Codex
  bash /path/to/ops-perf-optimize/init.sh global codex       # Global-level, Codex
  bash /path/to/ops-perf-optimize/init.sh project codearts   # Project-level, CodeArts

Installation paths (CANNBot brand):
  Project-level: $PWD/.claude/, $PWD/.opencode/, or $PWD/.codex/ + $PWD/.agents/skills/
  Global-level:  $HOME/.claude/, $HOME/.config/opencode/, or $HOME/.codex/ + $HOME/.agents/skills/

After installation, launch directly:
  OpenCode: opencode
  Claude:   claude
  Codex:    codex
EOF
}

LEVEL="project"
TOOL="opencode"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$PWD"
PLUGIN_SRC="$SCRIPT_DIR"
# In-plugin skills (bundled with this plugin, e.g. ops-perf-optimize-workflow / ops-perf-acceptance-criteria)
# take priority; shared skills fall back to ../../ops. Skill resolution is LOCAL-first.
LOCAL_SKILL_ROOT="$PLUGIN_SRC/skills"
if [ -d "$SCRIPT_DIR/../../ops" ]; then
    ASCEND_AGENT_ROOT="$(cd "$SCRIPT_DIR/../../ops" && pwd)"
else
    ASCEND_AGENT_ROOT=""
fi

for arg in "$@"; do
    case "$arg" in
        --help)            show_help; exit 0 ;;
        global|project)    LEVEL="$arg" ;;
        opencode|claude|codex|codearts)   TOOL="$arg" ;;
        *)  echo "Error: Unknown argument '$arg'. Valid: global, project, opencode, claude, codex, codearts, --help."
            exit 1 ;;
    esac
done

# Determine config root directory
if [ "$LEVEL" = "global" ]; then
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$HOME/.config/opencode"
    elif [ "$TOOL" = "codearts" ]; then
        CONFIG_ROOT="$HOME/.codeartsdoer"
    elif [ "$TOOL" = "codex" ]; then
        CONFIG_ROOT="$HOME/.codex"
    else
        CONFIG_ROOT="$HOME/.claude"
    fi
else
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$PROJECT_ROOT/.opencode"
    elif [ "$TOOL" = "codearts" ]; then
        CONFIG_ROOT="$PROJECT_ROOT/.codeartsdoer"
    elif [ "$TOOL" = "codex" ]; then
        CONFIG_ROOT="$PROJECT_ROOT/.codex"
    else
        CONFIG_ROOT="$PROJECT_ROOT/.claude"
    fi
fi

CANNBOT_DIR="$CONFIG_ROOT"

# Codex discovers skills from .agents/skills, while custom agents and workflow
# configuration live below .codex. Other clients keep skills below CONFIG_ROOT.
SKILL_DISCOVERY_ROOT="$CANNBOT_DIR/skills"
if [ "$TOOL" = "codex" ]; then
    if [ "$LEVEL" = "global" ]; then
        SKILL_DISCOVERY_ROOT="$HOME/.agents/skills"
    else
        SKILL_DISCOVERY_ROOT="$PROJECT_ROOT/.agents/skills"
    fi
fi

# Remove only this plugin's managed block (or its legacy OpenCode trigger) from
# a project AGENTS.md. Preserve unrelated user content and delete the file only
# when no non-whitespace content remains.
remove_project_trigger() {
    local target="$1"
    [ -f "$target" ] || return 0
    python3 - "$target" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
begin = "<!-- CANNBOT:ops-perf-optimize:begin -->"
end = "<!-- CANNBOT:ops-perf-optimize:end -->"
updated = re.sub(
    re.escape(begin) + r".*?" + re.escape(end) + r"\n?",
    "",
    text,
    flags=re.S,
)
updated = re.sub(
    r"^# 工作流加载规则\n.*?^详见 `\.opencode/agents/perf-optimize\.md`，本文件仅为触发入口。\s*",
    "",
    updated,
    count=1,
    flags=re.S | re.M,
)
if updated == text:
    raise SystemExit(0)
if updated.strip():
    path.write_text(updated.rstrip() + "\n", encoding="utf-8")
else:
    path.unlink()
PY
}

# Generate and merge a minimal project trigger. The trigger loads the installed
# ops-perf-optimize-workflow Skill; it never duplicates Stage details or acceptance gates.
install_project_trigger() {
    local target="$1"
    local skill_file="$2"
    local include_codex_adapter="$3"
    python3 - "$target" "$skill_file" "$SKILL_DISCOVERY_ROOT" "$include_codex_adapter" <<'PY'
from pathlib import Path
import re
import sys

target = Path(sys.argv[1])
skill_file, skill_root, include_codex_adapter = sys.argv[2:5]
common = f"""# 工作流加载规则

仅当用户明确要求对某个具体 AscendC 算子或代码目录执行性能优化/性能调优时：
1. 必须调用 `ops-perf-optimize-workflow` Skill，并先完整读取 `{skill_file}`。
2. 按该 Skill 指向的统一工作流契约执行。
3. Skill 加载完成前不得编译、上板或派发任何 Stage；项目入口不复制或重构工作流细节。

安装/使用咨询、报告或日志分析、性能概念解释、精度排障、普通代码检视，以及只提到 cases.csv、msprof、profiling、加速比但没有明确优化动作和目标算子的请求，都不得触发本工作流。此时按普通请求直接回答，不加载 Skill、不创建运行目录、不派发专家。"""
adapter = ""
if include_codex_adapter == "true":
    adapter = f"""

## Codex 运行适配（由 init.sh 自动生成）

- 当前 Codex 主线程就是主 agent；只负责编排、门禁复核、设备调度和汇总，不参与性能分析、策略制定或方案代码实施。
- 五个阶段（baseline / preparation / analysis / implementation / integration）统一派发给 Codex 通用子 agent，派发键值块中声明 ROLE 并要求加载对应阶段 Skill；阶段职责与验收以各 Skill 为唯一真源。

派发使用统一工作流契约中的派发键值块。派发自定义 agent 时显式设置 `fork_turns="none"`；full-history fork 不能同时选择自定义 agent。主 agent 必须等待当前阶段所需的全部子 agent 返回并校验落盘产物。

工作流所需 Skills 安装在 `{skill_root}`。主 agent 或子 agent 被任务、角色点名某个 Skill 时，必须先从 Codex 可用技能列表中找到它，完整读取其 `SKILL.md`，再按该 Skill 执行。
""".rstrip()
begin = "<!-- CANNBOT:ops-perf-optimize:begin -->"
end = "<!-- CANNBOT:ops-perf-optimize:end -->"
body = common + ("\n" + adapter if adapter else "")
block = f"{begin}\n{body}\n{end}"

existing = target.read_text(encoding="utf-8") if target.exists() else ""
existing = re.sub(
    re.escape(begin) + r".*?" + re.escape(end) + r"\n?",
    "",
    existing,
    flags=re.S,
).rstrip()
# A project may already contain this plugin's legacy OpenCode trigger. Replace
# that known section with the generated Codex block while preserving unrelated
# repository instructions.
opencode_pattern = re.compile(
    r"^# 工作流加载规则\n.*?^详见 `\.opencode/agents/perf-optimize\.md`，本文件仅为触发入口。\s*",
    flags=re.S | re.M,
)
if opencode_pattern.search(existing):
    existing = opencode_pattern.sub("", existing, count=1).rstrip()
merged = (existing + "\n\n" if existing else "") + block.rstrip() + "\n"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(merged, encoding="utf-8")
PY
}

# Enable multi-agent support in Codex while preserving unrelated TOML keys.
# The workflow can fan out to at most three implementation agents, so four
# concurrent child threads cover all built-in stages without serial fallback.
ensure_codex_agents_config() {
    local config="$CONFIG_ROOT/config.toml"
    python3 - "$config" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
section_start = None
section_end = len(lines)
for index, line in enumerate(lines):
    if re.match(r"^\s*\[agents\]\s*(?:#.*)?$", line):
        section_start = index
        for idx in range(index + 1, len(lines)):
            if re.match(r"^\s*\[\[?.+\]\]?\s*(?:#.*)?$", lines[idx]):
                section_end = idx
                break
        break

required = {
    "enabled": "enabled = true",
    "max_concurrent_threads_per_session": "max_concurrent_threads_per_session = 4",
}
if section_start is None:
    if lines and lines[-1].strip():
        lines.append("")
    lines.extend([
        "# Managed requirements for CANNBot ops-perf-optimize.",
        "[agents]",
        required["enabled"],
        required["max_concurrent_threads_per_session"],
    ])
else:
    found = set()
    for index in range(section_start + 1, section_end):
        for key, rendered in required.items():
            if re.match(rf"^\s*{re.escape(key)}\s*=", lines[index]):
                lines[index] = rendered
                found.add(key)
    missing = [required[key] for key in required if key not in found]
    if missing:
        lines[section_end:section_end] = missing

path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
PY
}

# A personal Claude skill takes precedence over a project skill. Refuse a
# project install when a same-name personal skill resolves to different content;
# otherwise Claude would silently preload the wrong implementation.
validate_claude_skill_precedence() {
    [ "$TOOL" = "claude" ] && [ "$LEVEL" = "project" ] || return 0

    local personal_root="$HOME/.claude/skills"
    local skill_name expected_source personal_target
    for skill_name in $INCLUDED_SKILLS; do
        expected_source="$LOCAL_SKILL_ROOT/$skill_name"
        [ -d "$expected_source" ] || expected_source="$ASCEND_AGENT_ROOT/$skill_name"
        [ -d "$expected_source" ] || continue
        personal_target="$personal_root/$skill_name"
        [ -e "$personal_target" ] || [ -L "$personal_target" ] || continue
        if [ ! -e "$personal_target" ]; then
            err "Personal Claude skill is a broken link and shadows the project skill: $personal_target"
            exit 1
        fi
        if [ "$(realpath "$personal_target")" != "$(realpath "$expected_source")" ]; then
            err "Personal Claude skill shadows this plugin with different content: $personal_target"
            info "Expected source: $(realpath "$expected_source")"
            info "Remove or realign the personal skill before reinstalling."
            exit 1
        fi
    done
}

# Reinstalls remove the default-agent key created by versions before v3. The
# current installer never sets a default agent: unrelated requests stay with
# the client's normal agent and only the narrow trigger routes optimization.
remove_managed_claude_default_agent() {
    local manifest="$1"
    local settings="$CONFIG_ROOT/settings.json"
    [ "$TOOL" = "claude" ] && [ -f "$settings" ] || return 0
    python3 - "$manifest" "$settings" "$PRIMARY_AGENT_NAME" <<'PY'
from pathlib import Path
import json
import sys

manifest_path, settings_path = map(Path, sys.argv[1:3])
agent = sys.argv[3]
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
if not manifest.get("managed_default_agent", False):
    raise SystemExit(0)
try:
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
if settings.get("agent") != agent:
    raise SystemExit(0)
del settings["agent"]
settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

# --- Uninstall previous installation (manifest-based, safe) ---
uninstall_previous() {
    local manifest="$CONFIG_ROOT/cannbot-manifest.json"
    [ -f "$manifest" ] || return 0

    remove_managed_claude_default_agent "$manifest"

    # Remove only skills owned by this installer. Bundled Codex skills are
    # copied (rather than linked) and carry an explicit ownership marker.
    local old_skills
    old_skills=$(python3 -c "
import json
try:
    with open('$manifest') as f:
        data = json.load(f)
    print(' '.join(data.get('installed_skills', [])))
except: pass
" 2>/dev/null)
    for skill in $old_skills; do
        local target="$SKILL_DISCOVERY_ROOT/$skill"
        if [ -L "$target" ]; then
            # Retired skills may already be absent from the updated plugin.
            # Match the recorded name's exact managed source even for dangling
            # links; preserve links the user redirected outside those sources.
            if python3 - "$target" "$LOCAL_SKILL_ROOT/$skill" "$ASCEND_AGENT_ROOT/$skill" <<'PY'
from pathlib import Path
import sys

target, *sources = map(Path, sys.argv[1:])
raise SystemExit(0 if target.resolve() in {source.resolve() for source in sources} else 1)
PY
            then
                rm -f "$target"
            else
                warn "Skill symlink changed since installation; preserving it: $target"
            fi
        elif [ -d "$target" ] && [ -f "$target/$CODEX_MANAGED_SKILL_MARKER" ]; then
            rm -rf "$target"
        fi
    done

    # Remove previously installed agents. Codex adapters are managed regular
    # TOML files; other clients use symlinks.
    local old_agents
    old_agents=$(python3 -c "
import json
try:
    with open('$manifest') as f:
        data = json.load(f)
    print(' '.join(data.get('installed_agents', [])))
except: pass
" 2>/dev/null)
    for agent in $old_agents; do
        local target="$CANNBOT_DIR/agents/$agent"
        if [ -L "$target" ]; then
            rm -f "$target"
        elif [ "$TOOL" = "codex" ] && [ -f "$target" ] && grep -q "Managed by CANNBot ops-perf-optimize" "$target" 2>/dev/null; then
            rm -f "$target"
        fi
    done

    # Remove config symlinks (only if they are symlinks, never real files)
    if [ "$TOOL" = "opencode" ] || [ "$TOOL" = "codearts" ]; then
        [ -L "$CONFIG_ROOT/AGENTS.md" ] && rm -f "$CONFIG_ROOT/AGENTS.md"
    elif [ "$TOOL" = "claude" ]; then
        [ -L "$CONFIG_ROOT/CLAUDE.md" ] && rm -f "$CONFIG_ROOT/CLAUDE.md"
    fi
    [ -L "$CONFIG_ROOT/workflows" ] && rm -f "$CONFIG_ROOT/workflows"
    [ -L "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools" ] && rm -f "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools"
    [ -L "$CONFIG_ROOT/perf-optimize.md" ] && rm -f "$CONFIG_ROOT/perf-optimize.md"

    # Remove only this plugin's project trigger; preserve unrelated guidance.
    if [ "$TOOL" = "opencode" ] && [ "$LEVEL" = "project" ]; then
        remove_project_trigger "$PROJECT_ROOT/AGENTS.md"
    fi
    if [ "$TOOL" = "codex" ]; then
        if [ "$LEVEL" = "project" ]; then
            remove_project_trigger "$PROJECT_ROOT/AGENTS.md"
        else
            remove_project_trigger "$CONFIG_ROOT/AGENTS.md"
        fi
    fi

    rm -f "$manifest"
}

validate_claude_skill_precedence
uninstall_previous

# Clean up legacy cannbot subdirectory from previous installations
if [ -e "$CONFIG_ROOT/$BRAND" ] || [ -L "$CONFIG_ROOT/$BRAND" ]; then
    rm -rf "$CONFIG_ROOT/$BRAND"
fi
# OpenCode: also clean legacy teams link
if [ "$TOOL" = "opencode" ] && [ -L "$CONFIG_ROOT/teams" ]; then
    rm -f "$CONFIG_ROOT/teams"
fi

show_banner
echo "  Tool:      $TOOL"
echo "  Level:     $LEVEL"
echo "  Path:      $CONFIG_ROOT"
echo ""

# --- Step 1: Create directory symlinks ---
step "[1/5] Setting up CANNBot directory..."
mkdir -p "$CANNBOT_DIR"

step1_summary=""
step1_warns=""
if [ "$TOOL" = "opencode" ]; then
    # OpenCode: per-item symlinks for skills (LOCAL in-plugin first, then shared ../../ops)
    mkdir -p "$CANNBOT_DIR/skills"
    skill_count=0
    for skill_name in $INCLUDED_SKILLS; do
        skill_dir="$LOCAL_SKILL_ROOT/$skill_name"
        [ -d "$skill_dir" ] || skill_dir="$ASCEND_AGENT_ROOT/$skill_name"
        [ -d "$skill_dir" ] || continue
        ln -sfn "$(realpath "$skill_dir")" "$CANNBOT_DIR/skills/$skill_name"
        skill_count=$((skill_count + 1))
    done
    step1_summary="skills(${skill_count}) "

    # OpenCode: stages dispatch to the generic subagent; only the primary
    # agent wrapper is linked under agents/ (Step 2).
    mkdir -p "$CANNBOT_DIR/agents"
    ok "Linked: $step1_summary"
elif [ "$TOOL" = "codex" ]; then
    mkdir -p "$SKILL_DISCOVERY_ROOT" "$CANNBOT_DIR/agents"

    skill_count=0
    for skill_name in $INCLUDED_SKILLS; do
        skill_dir="$LOCAL_SKILL_ROOT/$skill_name"
        bundled_skill=true
        if [ ! -d "$skill_dir" ]; then
            skill_dir="$ASCEND_AGENT_ROOT/$skill_name"
            bundled_skill=false
        fi
        [ -d "$skill_dir" ] || continue
        skill_target="$SKILL_DISCOVERY_ROOT/$skill_name"
        if [ -e "$skill_target" ] || [ -L "$skill_target" ]; then
            if [ -L "$skill_target" ]; then
                if [ "$(realpath "$skill_target")" != "$(realpath "$skill_dir")" ]; then
                    err "Skill symlink points to a different source: $skill_target"
                    exit 1
                fi
                rm -f "$skill_target"
            elif [ -d "$skill_target" ] && [ -f "$skill_target/$CODEX_MANAGED_SKILL_MARKER" ]; then
                rm -rf "$skill_target"
            else
                err "Skill target already exists and is not managed by CANNBot: $skill_target"
                exit 1
            fi
        fi
        if [ "$bundled_skill" = true ]; then
            # Codex derives plugin namespaces through a directory symlink's
            # canonical target. A project-local copy preserves the bare skill
            # name used by the cross-client workflow contract.
            cp -a "$skill_dir" "$skill_target"
            # The copy breaks the symlink trail back to the plugin, so record the
            # plugin root in the marker; skills/ops-perf-optimize-workflow/scripts/resolve-plugin-root.sh
            # reads it to find the canonical Skill contract and workflow tools.
            printf '%s\n' "$(realpath "$PLUGIN_SRC")" > "$skill_target/$CODEX_MANAGED_SKILL_MARKER"
        else
            ln -s "$(realpath "$skill_dir")" "$skill_target"
        fi
        skill_count=$((skill_count + 1))
    done

    # Remove stale files from earlier versions (including general_purpose.toml)
    # only when they carry this installer's ownership marker.
    for agent_target in "$CANNBOT_DIR/agents"/*.toml; do
        [ -f "$agent_target" ] || continue
        if grep -q "Managed by CANNBot ops-perf-optimize" "$agent_target" 2>/dev/null; then
            rm -f "$agent_target"
        fi
    done

    ok "Linked skills(${skill_count}); stages dispatch to the generic subagent (no per-stage agent files)"
else
    # Claude: create directories (per-item symlinks handled in Step 3)
    mkdir -p "$CONFIG_ROOT/skills" "$CONFIG_ROOT/agents"
    ok "Prepared: skills/, agents/"
fi
[ -n "$step1_warns" ] && echo -e "$step1_warns"
echo ""

# --- Step 2: Install primary agent / workflow configuration ---
step "[2/5] Installing configuration..."
mkdir -p "$CONFIG_ROOT"

if [ "$TOOL" = "opencode" ]; then
    # OpenCode scans agents/*.md to register subagents (filename = agent name).
    # The plugin's AGENTS.md (mode: primary, Skill-dispatch wrapper) is linked
    # under agents/ as perf-optimize.md so it can be dispatched as a subagent.
    mkdir -p "$CANNBOT_DIR/agents"
    config_target="$CANNBOT_DIR/agents/$PRIMARY_AGENT_NAME.md"
elif [ "$TOOL" = "codearts" ]; then
    config_target="$CONFIG_ROOT/AGENTS.md"
elif [ "$TOOL" = "codex" ]; then
    # The current Codex thread is the primary orchestrator. Keep AGENTS.md as a
    # small wrapper and let the managed project trigger load the Skill on demand.
    ensure_codex_agents_config
    config_target="$CONFIG_ROOT/perf-optimize.md"
else
    # Claude custom agents are first-class definitions under agents/. Treating
    # this file as CLAUDE.md would load a large workflow as generic memory and
    # would not register the primary agent's frontmatter.
    config_target="$CONFIG_ROOT/agents/$PRIMARY_AGENT_NAME.md"
fi

if [ "$TOOL" = "codex" ] && [ -e "$config_target" ] && [ ! -L "$config_target" ]; then
    err "Codex workflow target already exists and is not managed by CANNBot: $config_target"
    exit 1
elif [ -e "$config_target" ] && [ ! -L "$config_target" ]; then
    warn "$(basename "$config_target") already exists (not a symlink), skipping — remove manually if you want to overwrite"
elif [ -e "$config_target" ] || [ -L "$config_target" ]; then
    rm -f "$config_target"
    ln -sf "$PLUGIN_SRC/AGENTS.md" "$config_target"
    ok "$(basename "$config_target") (replaced)"
else
    ln -sf "$PLUGIN_SRC/AGENTS.md" "$config_target"
    ok "$(basename "$config_target")"
fi

# Link workflows directory
if [ -d "$PLUGIN_SRC/workflows" ]; then
    workflow_target="$CONFIG_ROOT/workflows"
    if [ "$TOOL" = "codex" ] && [ -e "$workflow_target" ] && [ ! -L "$workflow_target" ]; then
        err "Codex workflow directory already exists and is not managed by CANNBot: $workflow_target"
        exit 1
    elif [ -e "$workflow_target" ] && [ ! -L "$workflow_target" ]; then
        warn "workflows/ already exists (not a symlink), skipping — remove manually if you want to overwrite"
    else
        [ -e "$workflow_target" ] || [ -L "$workflow_target" ] && rm -rf "$workflow_target"
        ln -sfn "$(realpath "$PLUGIN_SRC/workflows")" "$workflow_target"
    fi
else
    warn "workflows/ not found, skipping"
fi

# Link workflow helper scripts. The directory symlink keeps checkpoint and
# journal fixes live across clients without requiring a reinstall.
if [ -d "$PLUGIN_SRC/scripts" ]; then
    scripts_target="$CONFIG_ROOT/cannbot-ops-perf-optimize-tools"
    if [ -e "$scripts_target" ] && [ ! -L "$scripts_target" ]; then
        err "CANNBot workflow tools target already exists and is not managed by CANNBot: $scripts_target"
        exit 1
    else
        [ -e "$scripts_target" ] || [ -L "$scripts_target" ] && rm -rf "$scripts_target"
        ln -sfn "$(realpath "$PLUGIN_SRC/scripts")" "$scripts_target"
    fi
else
    warn "scripts/ not found, skipping"
fi

# Install a generated project-root trigger (OpenCode only). Workflow details
# remain behind the ops-perf-optimize-workflow Skill entry.
if [ "$TOOL" = "opencode" ] && [ "$LEVEL" = "project" ]; then
    trigger_target="$PROJECT_ROOT/AGENTS.md"
    [ -L "$trigger_target" ] && rm -f "$trigger_target"
    install_project_trigger "$trigger_target" "$SKILL_DISCOVERY_ROOT/ops-perf-optimize-workflow/SKILL.md" false
    ok "AGENTS.md (generated project trigger) installed"
fi

# Codex uses AGENTS.md for the primary agent. Merge a managed trigger block so
# existing project/global guidance remains intact.
if [ "$TOOL" = "codex" ]; then
    if [ "$LEVEL" = "project" ]; then
        trigger_target="$PROJECT_ROOT/AGENTS.md"
    else
        trigger_target="$CONFIG_ROOT/AGENTS.md"
    fi
    install_project_trigger "$trigger_target" "$SKILL_DISCOVERY_ROOT/ops-perf-optimize-workflow/SKILL.md" true
    ok "AGENTS.md (generated Codex primary-agent trigger) installed"
fi
echo ""

# --- Step 3: Configure tool discovery ---
step "[3/5] Configuring tool discovery..."

if [ "$TOOL" = "opencode" ]; then
    # OpenCode: skills/agents already at auto-scan paths, no extra discovery needed
    ok "Auto-scan: skills/, agents/"
elif [ "$TOOL" = "codex" ]; then
    # Codex scans these two distinct project/global discovery roots.
    ok "Auto-scan: $SKILL_DISCOVERY_ROOT and $CANNBOT_DIR/agents/*.toml"
else
    # Claude: create per-skill discovery symlinks (LOCAL in-plugin first, then shared ../../ops)
    DISCOVERY="$CONFIG_ROOT/skills"

    link_count=0
    for skill_name in $INCLUDED_SKILLS; do
        skill_dir="$LOCAL_SKILL_ROOT/$skill_name"
        [ -d "$skill_dir" ] || skill_dir="$ASCEND_AGENT_ROOT/$skill_name"
        [ -d "$skill_dir" ] || continue
        ln -sfn "$(realpath "$skill_dir")" "$DISCOVERY/$skill_name"
        link_count=$((link_count + 1))
    done

    # Clean broken symlinks
    for link in "$DISCOVERY"/*/; do
        link="${link%/}"
        [ -L "$link" ] && [ ! -e "$link" ] && rm "$link"
    done

    ok "Skills: $link_count discovery symlinks"

    # Claude: stages dispatch to the generic subagent; no per-stage agent
    # symlinks. Clean dangling agent links left by earlier installs.
    AGENT_DISCOVERY="$CONFIG_ROOT/agents"
    for link in "$AGENT_DISCOVERY"/*.md; do
        [ -L "$link" ] && [ ! -e "$link" ] && rm "$link"
    done
fi
echo ""

# --- Step 4: Setup reference repositories (optional) ---
step "[4/5] Setting up reference repositories..."
ASC_DEVKIT_DIR="$PROJECT_ROOT/asc-devkit"
CANN_SAMPLES_DIR="$PROJECT_ROOT/cann-samples"
ASC_DEVKIT_SAFE_TO_CLEAN=false

# Setup asc-devkit
if [ -d "$ASC_DEVKIT_DIR" ]; then
    if git -C "$ASC_DEVKIT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        if [ -n "$(git -C "$ASC_DEVKIT_DIR" status --porcelain 2>/dev/null)" ]; then
            warn "asc-devkit has local changes; preserving them and skipping update"
        else
            if optional_git_pull "$ASC_DEVKIT_DIR"; then
                ok "asc-devkit updated"
            else
                warn "asc-devkit git pull failed or timed out, using existing version"
            fi
            ASC_DEVKIT_SAFE_TO_CLEAN=true
        fi
    else
        warn "asc-devkit is not a git worktree, using existing directory"
    fi
else
    git clone --quiet https://gitcode.com/cann/asc-devkit.git "$ASC_DEVKIT_DIR" 2>/dev/null || warn "git clone failed, skipping asc-devkit (optional)"
    if [ -d "$ASC_DEVKIT_DIR" ]; then
        ok "asc-devkit cloned"
        ASC_DEVKIT_SAFE_TO_CLEAN=true
    fi
fi

# Setup cann-samples
if [ -d "$CANN_SAMPLES_DIR" ]; then
    if git -C "$CANN_SAMPLES_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        if [ -n "$(git -C "$CANN_SAMPLES_DIR" status --porcelain 2>/dev/null)" ]; then
            warn "cann-samples has local changes; preserving them and skipping update"
        else
            if optional_git_pull "$CANN_SAMPLES_DIR"; then
                ok "cann-samples updated"
            else
                warn "cann-samples git pull failed or timed out, using existing version"
            fi
        fi
    else
        warn "cann-samples is not a git worktree, using existing directory"
    fi
else
    git clone --quiet https://gitcode.com/cann/cann-samples.git "$CANN_SAMPLES_DIR" 2>/dev/null || warn "git clone failed, skipping cann-samples (optional)"
    [ -d "$CANN_SAMPLES_DIR" ] && ok "cann-samples cloned"
fi

if [ "$ASC_DEVKIT_SAFE_TO_CLEAN" = true ] && [ -n "$ASCEND_AGENT_ROOT" ] && [ -d "$ASC_DEVKIT_DIR" ] && [ -f "$ASCEND_AGENT_ROOT/ascendc-docs-search/scripts/clean_markdown.py" ]; then
    python3 "$ASCEND_AGENT_ROOT/ascendc-docs-search/scripts/clean_markdown.py" --dir "$ASC_DEVKIT_DIR" --no-backup --quiet > /dev/null 2>&1 || warn "markdown cleanup failed"
fi

# --- Step 5: Generate brand manifest (silent) ---
MANIFEST="$CONFIG_ROOT/cannbot-manifest.json"

# Collect only skills owned by this installer. The Codex discovery directory
# can also contain unrelated project skills.
SKILLS_JSON="[]"
if [ -d "$SKILL_DISCOVERY_ROOT" ]; then
  SKILLS_JSON=$(for skill_name in $INCLUDED_SKILLS; do
    [ -d "$SKILL_DISCOVERY_ROOT/$skill_name" ] && echo "$skill_name"
  done | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
fi

# Collect only this installer's agents for the same reason.
AGENTS_JSON="[]"
if [ -d "$CANNBOT_DIR/agents" ]; then
  if [ "$TOOL" = "codex" ]; then
    AGENTS_JSON=$(for agent_file in "$CANNBOT_DIR/agents"/*.toml; do
      [ -f "$agent_file" ] || continue
      grep -q "Managed by CANNBot ops-perf-optimize" "$agent_file" 2>/dev/null || continue
      basename "$agent_file"
    done | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
  else
    AGENTS_JSON=$({
      if { [ "$TOOL" = "opencode" ] || [ "$TOOL" = "claude" ]; } && [ -e "$CANNBOT_DIR/agents/$PRIMARY_AGENT_NAME.md" ]; then
        echo "$PRIMARY_AGENT_NAME.md"
      fi
    } | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
  fi
fi

cat > "$MANIFEST" << MANIFEST_EOF
{
  "brand": "CANNBot",
  "version": "$VERSION",
  "team": "$(basename "$SCRIPT_DIR")",
  "level": "$LEVEL",
  "tool": "$TOOL",
  "installed_skills": $SKILLS_JSON,
  "installed_agents": $AGENTS_JSON,
  "managed_default_agent": $CLAUDE_DEFAULT_AGENT_MANAGED,
  "project_agents_trigger": "$([ "$LEVEL" = "project" ] && { [ "$TOOL" = "opencode" ] || [ "$TOOL" = "codex" ]; } && echo "$PROJECT_ROOT/AGENTS.md" || echo "")",
  "primary_agent_instructions": "$([ "$TOOL" = "codex" ] && { [ "$LEVEL" = "project" ] && echo "$PROJECT_ROOT/AGENTS.md" || echo "$CONFIG_ROOT/AGENTS.md"; } || echo "")",
  "skill_discovery_dir": "$SKILL_DISCOVERY_ROOT",
  "main_workflow": "$([ "$TOOL" = "codex" ] && echo "$SKILL_DISCOVERY_ROOT/ops-perf-optimize-workflow/SKILL.md" || echo "")",
  "workflow_contract": "$SKILL_DISCOVERY_ROOT/ops-perf-optimize-workflow/references/workflow-contract.md",
  "case_set_tool": "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools/prepare_case_sets.py",
  "workflow_state_tool": "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools/workflow_state.py",
  "implementation_conformance_tool": "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools/validate_implementation_conformance.py",
  "integration_safety_tool": "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools/validate_integration_safety.py",
  "source_isolation_tool": "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools/validate_source_isolation.py",
  "performance_effects_tool": "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools/summarize_performance_effects.py",
  "devkit_dir": "$ASC_DEVKIT_DIR",
  "samples_dir": "$CANN_SAMPLES_DIR",
  "brand_dir": "$CONFIG_ROOT",
  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST_EOF


# --- Step 5: Health check ---
echo ""
step "[5/5] Running health check..."
health_ok=true
health_errors=""

# Check discovery directories
HEALTH_SUBS="skills agents"
for sub in $HEALTH_SUBS; do
  if [ "$sub" = "skills" ]; then
    target="$SKILL_DISCOVERY_ROOT"
  else
    target="$CANNBOT_DIR/agents"
  fi
  if [ -d "$target" ]; then
    if [ "$sub" = "agents" ]; then
      if [ "$TOOL" = "codex" ]; then
        count=$(ls "$target"/*.toml 2>/dev/null | wc -l)
      else
        count=$(ls "$target"/*.md 2>/dev/null | wc -l)
      fi
    else
      count=$(ls -d "$target"/*/ 2>/dev/null | wc -l)
    fi
    [ "$count" -eq 0 ] && { health_errors="${health_errors}\n  ${YELLOW}⚠${NC} $sub/ is empty"; }
  else
    health_errors="${health_errors}\n  ${RED}✗${NC} $sub/ missing"
    health_ok=false
  fi
done

# Check skill count consistency (Claude only)
if [ "$TOOL" = "claude" ] && [ -d "$DISCOVERY" ]; then
  expected=$(echo $INCLUDED_SKILLS | wc -w)
  actual=$link_count
  if [ "$actual" -ne "$expected" ]; then
    health_errors="${health_errors}\n  ${YELLOW}⚠${NC} Skill discovery mismatch: $actual/$expected"
  fi
fi

# Codex-specific static validation: all skills expose SKILL.md, all custom
# agent files are valid TOML with no unresolved adapter placeholders, and all
# main-agent workflow entry points resolve.
if [ "$TOOL" = "codex" ]; then
  expected=$(echo $INCLUDED_SKILLS | wc -w)
  actual=$(for skill_name in $INCLUDED_SKILLS; do
    [ -d "$SKILL_DISCOVERY_ROOT/$skill_name" ] && echo "$skill_name"
  done | wc -l)
  if [ "$actual" -ne "$expected" ]; then
    health_errors="${health_errors}\n  ${RED}✗${NC} Skill discovery mismatch: $actual/$expected"
    health_ok=false
  fi
  for skill_name in $INCLUDED_SKILLS; do
    [ -f "$SKILL_DISCOVERY_ROOT/$skill_name/SKILL.md" ] || {
      health_errors="${health_errors}\n  ${RED}✗${NC} Skill missing SKILL.md: $skill_name"
      health_ok=false
    }
    if [ -d "$LOCAL_SKILL_ROOT/$skill_name" ]; then
      if [ -L "$SKILL_DISCOVERY_ROOT/$skill_name" ] || [ ! -f "$SKILL_DISCOVERY_ROOT/$skill_name/$CODEX_MANAGED_SKILL_MARKER" ]; then
        health_errors="${health_errors}\n  ${RED}✗${NC} Bundled Codex skill must be a managed local copy: $skill_name"
        health_ok=false
      fi
    fi
  done
  if ! python3 - "$CONFIG_ROOT/config.toml" "$CONFIG_ROOT/agents" "$PLUGIN_SRC/agents" <<'PY'
from pathlib import Path
import sys
import tomllib
import re

config = Path(sys.argv[1])
agent_dir = Path(sys.argv[2])
source_dir = Path(sys.argv[3])
tomllib.loads(config.read_text(encoding="utf-8"))
agents = [
    path
    for path in sorted(agent_dir.glob("*.toml"))
    if "Managed by CANNBot ops-perf-optimize" in path.read_text(encoding="utf-8")
]
sources = sorted(source_dir.glob("ascendc-perf-*.md"))
if len(agents) != len(sources):
    raise SystemExit(f"Codex agent count mismatch: {len(agents)}/{len(sources)}")
required = {"name", "description", "developer_instructions"}
for source, path in zip(sources, agents):
    text = path.read_text(encoding="utf-8")
    if "__CANNBOT_" in text:
        raise SystemExit(f"unresolved placeholder in {path}")
    data = tomllib.loads(text)
    missing = required - data.keys()
    if missing:
        raise SystemExit(f"{path} missing fields: {sorted(missing)}")
    if not re.fullmatch(r"[a-z0-9_]+", data["name"]):
        raise SystemExit(f"{path} has a Codex-incompatible agent name: {data['name']}")
    source_text = source.read_text(encoding="utf-8")
    name_match = re.search(r"^name:\s*(\S+)\s*$", source_text, flags=re.M)
    if not name_match or data["name"] != name_match.group(1).replace("-", "_"):
        raise SystemExit(f"{path} name does not match common source {source}")
    source_path = str(source.resolve())
    instructions = data["developer_instructions"]
    if source_path not in instructions:
        raise SystemExit(f"{path} does not dynamically reference common source {source_path}")
    if source_text.rstrip() in instructions:
        raise SystemExit(f"{path} embeds a stale copy of common source {source}")
    if "read its SKILL.md completely" not in instructions:
        raise SystemExit(f"{path} does not enforce runtime Skill loading")
PY
  then
    health_errors="${health_errors}\n  ${RED}✗${NC} Codex config or custom-agent TOML validation failed"
    health_ok=false
  fi
  [ -f "$CONFIG_ROOT/perf-optimize.md" ] || {
    health_errors="${health_errors}\n  ${RED}✗${NC} Codex main workflow missing"
    health_ok=false
  }
  [ -f "$SKILL_DISCOVERY_ROOT/ops-perf-optimize-workflow/references/workflow-contract.md" ] || {
    health_errors="${health_errors}\n  ${RED}✗${NC} Codex workflow contract missing"
    health_ok=false
  }
fi

[ -f "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools/workflow_state.py" ] || {
  health_errors="${health_errors}\n  ${RED}✗${NC} Workflow state tool missing"
  health_ok=false
}
[ -f "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools/prepare_case_sets.py" ] || {
  health_errors="${health_errors}\n  ${RED}✗${NC} Case-set preparation tool missing"
  health_ok=false
}
[ -f "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools/validate_implementation_conformance.py" ] || {
  health_errors="${health_errors}\n  ${RED}✗${NC} Implementation conformance tool missing"
  health_ok=false
}
[ -f "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools/validate_integration_safety.py" ] || {
  health_errors="${health_errors}\n  ${RED}✗${NC} Integration safety tool missing"
  health_ok=false
}
[ -f "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools/validate_source_isolation.py" ] || {
  health_errors="${health_errors}\n  ${RED}✗${NC} Source-isolation tool missing"
  health_ok=false
}
[ -f "$CONFIG_ROOT/cannbot-ops-perf-optimize-tools/summarize_performance_effects.py" ] || {
  health_errors="${health_errors}\n  ${RED}✗${NC} Performance effects tool missing"
  health_ok=false
}

# Check reference repositories
if [ ! -d "$ASC_DEVKIT_DIR" ]; then
  health_errors="${health_errors}\n  ${YELLOW}⚠${NC} asc-devkit not available (optional)"
fi
if [ ! -d "$CANN_SAMPLES_DIR" ]; then
  health_errors="${health_errors}\n  ${YELLOW}⚠${NC} cann-samples not available (optional)"
fi

# Check primary agent / config file
if [ "$TOOL" = "opencode" ]; then
  [ -f "$CANNBOT_DIR/agents/$PRIMARY_AGENT_NAME.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} $PRIMARY_AGENT_NAME primary agent missing"; health_ok=false; }
  # Check project-root AGENTS.md trigger (project-level only)
  if [ "$LEVEL" = "project" ]; then
    [ -f "$PROJECT_ROOT/AGENTS.md" ] || { health_errors="${health_errors}\n  ${YELLOW}⚠${NC} Project AGENTS.md trigger not installed"; }
  fi
elif [ "$TOOL" = "codearts" ]; then
  [ -f "$CONFIG_ROOT/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing"; health_ok=false; }
elif [ "$TOOL" = "codex" ]; then
  if [ "$LEVEL" = "project" ]; then
    primary_trigger="$PROJECT_ROOT/AGENTS.md"
  else
    primary_trigger="$CONFIG_ROOT/AGENTS.md"
  fi
  if [ ! -f "$primary_trigger" ] || ! grep -q "CANNBOT:ops-perf-optimize:begin" "$primary_trigger" 2>/dev/null; then
    health_errors="${health_errors}\n  ${RED}✗${NC} Codex primary-agent AGENTS.md trigger missing"
    health_ok=false
  fi
else
  [ -f "$CONFIG_ROOT/agents/$PRIMARY_AGENT_NAME.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} Claude primary agent missing: $PRIMARY_AGENT_NAME"; health_ok=false; }
fi

# Check manifest
[ -f "$MANIFEST" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} Manifest generation failed"; health_ok=false; }

if [ "$health_ok" = true ] && [ -z "$health_errors" ]; then
  ok "All checks passed"
else
  echo -e "$health_errors"
  [ "$health_ok" = true ] && warn "Some warnings, see above" || err "Some checks failed, see above"
fi

if { [ "$TOOL" = "codex" ] || [ "$TOOL" = "claude" ]; } && [ "$health_ok" != true ]; then
  err "$TOOL installation is incomplete; refusing to report success"
  exit 1
fi

# --- Summary & Quick Start ---
echo ""
echo -e "  ${GREEN}${BOLD}✓ CANNBot installed successfully!${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
if [ "$TOOL" = "codearts" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 CodeArts CLI / IDE 启动"
elif [ "$TOOL" = "opencode" ]; then
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}opencode${NC}"
elif [ "$TOOL" = "codex" ]; then
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}codex${NC}"
else
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}claude${NC}（若保留了其他默认 Agent：${GREEN}claude --agent $PRIMARY_AGENT_NAME${NC}）"
fi
echo -e "  ${CYAN}2.${NC} 选择使用方式（将 /path/to/... 和目标 SoC 替换为实际配置）："
echo ""
echo -e "  ${BOLD}方式一：自动化调优${NC}"
cat <<'EOF'
  请对 /path/to/operator/ 执行自动化性能调优，目标 SoC 为 Ascend950。
  性能用例：/path/to/cases.csv；功能用例复用性能用例。
  性能对标：同环境原始实现，逐 case 目标比例 1.0；固定优化 3 轮。
  Python：/path/to/venv/bin/python；CANN：source /path/to/Ascend/cann/bin/setenv.bash。
  中间结果输出到 /path/to/output/。
EOF
echo ""
echo -e "  ${BOLD}方式二：交互式调优${NC}"
echo -e "  ${CYAN}Step 1：性能分析与方案生成${NC}"
cat <<'EOF'
  对 /path/to/operator/，先只采集性能并分析瓶颈、输出方案，完成后暂停，等我选择再实施。
  使用 ops-profiling 采集，结合 ascendc-perf-optimize 和 ascendc-performance-best-practices 分析。
  目标 SoC：Ascend950；用例：/path/to/cases.csv；Python：/path/to/venv/bin/python。
  CANN：source /path/to/Ascend/cann/bin/setenv.bash；输出到 /path/to/output/。
EOF
echo "  Step 1 完成后，可自主审视方案，并与 Agent 对话修改；确认方案后再进入 Step 2。"
echo ""
echo -e "  ${CYAN}Step 2：按照生成的方案进行优化${NC}"
cat <<'EOF'
  请对 /path/to/operator/ 做性能优化，采用审阅修改后确认的 <方案编号>。
  沿用已确认的环境、用例、约束和输出目录，实施方案并验证精度与性能。
  输出候选的逐 case 性能对比和精度结果，完成后暂停，等我确认再做全量验收与回集成。
EOF
echo ""
echo -e "  ${CYAN}Step 3：全量验收与原工程回集成${NC}"
cat <<'EOF'
  使用上一步保留的可信实现，完成全量精度与性能验收，并回集成到 /path/to/operator/。
  验证原工程调用方式，输出最终逐 case 性能对比、代码位置、验证报告和复现命令。
EOF
echo ""
echo -e "  完整说明与多轮对话示例：${CYAN}$SCRIPT_DIR/quickstart.md${NC}"
echo ""
