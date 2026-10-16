# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Seed/verify a managed pre-unification install in an installer test workspace."""

import json
from pathlib import Path
import re
import sys

from stage_skill_contract import check_skill_layout


def main() -> None:
    action, project_arg, plugin_arg, client = sys.argv[1:]
    project, plugin = Path(project_arg), Path(plugin_arg)
    config = project / f".{client}"
    skills = project / ".agents/skills" if client == "codex" else config / "skills"
    manifest_path = config / "cannbot-manifest.json"
    legacy_skills = (
        "ascendc-perf-prepare",
        "ascendc-perf-integrate",
        "ascendc-perf-project",
    )
    legacy_agents = [
        f"{name.replace('-', '_')}.toml" if client == "codex" else f"{name}.md"
        for name in (
            "ascendc-perf-project-preparer",
            "ascendc-perf-precision-tester",
            "ascendc-perf-project-expert",
            "ascendc-perf-case-designer",
        )
    ]
    redirected_name = "retired-user-redirect"
    redirected = skills / redirected_name
    user_source = project / "user-skill-source"

    if action == "seed":
        manifest = json.loads(manifest_path.read_text())
        if client == "codex":
            # Bundled skills are copies on Codex. A refresh must remove documents
            # retired since installation, not merely overwrite surviving files.
            for name in ("report-and-verify.md", "worked-example-xlog1py.md"):
                (skills / "ops-perf-postprocess/references" / name).write_text(
                    "retired managed reference\n"
                )
            (
                skills / "ops-perf-optimize-workflow/references/analysis-execution.md"
            ).write_text("retired stage reference\n")
        for name in legacy_skills:
            path = skills / name
            if client == "codex":
                path.mkdir()
                (path / ".cannbot-ops-perf-optimize").write_text(str(plugin))
                (path / "SKILL.md").write_text("legacy managed skill\n")
            else:
                # Simulate an upgrade whose source no longer contains the old skill.
                source = plugin / "skills" / name
                assert not source.exists()
                path.symlink_to(source, target_is_directory=True)
            manifest["installed_skills"].append(name)
        for name in legacy_agents:
            agent = config / "agents" / name
            if client == "codex":
                agent.write_text("# Managed by CANNBot ops-perf-optimize\n")
            else:
                agent.symlink_to(plugin / "agents" / name)
            manifest["installed_agents"].append(name)
        user_source.mkdir()
        (user_source / "SKILL.md").write_text("user-owned sentinel\n")
        redirected.symlink_to(user_source, target_is_directory=True)
        manifest["installed_skills"].append(redirected_name)
        manifest_path.write_text(json.dumps(manifest))
    elif action == "verify":
        check_skill_layout(skills, config / "workflows")
        for path in [
            *(skills / name for name in legacy_skills),
            *(config / "agents" / name for name in legacy_agents),
        ]:
            assert not path.exists() and not path.is_symlink(), path
        assert redirected.is_symlink() and redirected.resolve() == user_source
        assert (redirected / "SKILL.md").read_text() == "user-owned sentinel\n"
        manifest = json.loads(manifest_path.read_text())
        assert not set(legacy_skills) & set(manifest["installed_skills"])
        assert not set(legacy_agents) & set(manifest["installed_agents"])
        assert redirected_name not in manifest["installed_skills"]
        routes = (config / "workflows/task-prompts.md").read_text()
        for role, skill in (
            ("preparation", "ops-perf-preparation"),
            ("integration", "ops-perf-postprocess"),
            ("baseline", "ops-perf-baseline-evaluate"),
        ):
            row = next(
                line for line in routes.splitlines() if line.startswith(f"| `{role}` |")
            )
            assert row.split("|")[-2].strip() == f"通用 subagent + `{skill}`", row
        skill = skills / "ops-perf-postprocess"
        source_skill = plugin / "skills/ops-perf-postprocess"
        installed_docs = {path.relative_to(skill) for path in skill.rglob("*.md")}
        source_docs = {
            path.relative_to(source_skill)
            for path in source_skill.rglob("*.md")
        }
        assert installed_docs == source_docs, (installed_docs, source_docs)
        pending, reachable = [Path("SKILL.md")], set()
        while pending:
            relative = pending.pop()
            if relative in reachable:
                continue
            path = skill / relative
            assert path.is_file(), path
            reachable.add(relative)
            for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                if "://" in link or link.startswith("#"):
                    continue
                target = (path.parent / link.split("#")[0]).resolve()
                pending.append(target.relative_to(skill.resolve()))
        assert source_docs == reachable, (
            f"Orphan project references: {source_docs - reachable}"
        )
        print(
            f"{client}: unified project role upgrade and user-link preservation: PASS"
        )
    else:
        raise ValueError(action)


if __name__ == "__main__":
    main()
