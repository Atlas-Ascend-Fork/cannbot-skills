# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate stage loading boundaries in both source and installed layouts."""

from pathlib import Path
import re

STAGE_SKILLS = {
    "baseline": "ops-perf-baseline-evaluate",
    "preparation": "ops-perf-preparation",
    "analysis": "ops-perf-analysis",
    "implementation": "ops-perf-implementation",
    "integration": "ops-perf-postprocess",
}


def check_skill_layout(skills: Path, workflows: Path) -> None:
    main = skills / "ops-perf-optimize-workflow"
    assert {p.name for p in (main / "references").iterdir()} == {
        "workflow-contract.md",
        "request-intake.md",
        "migration.md",
    }
    assert (workflows / "references/state-reporting.md").is_file()
    for name in (
        *STAGE_SKILLS.values(),
        "ops-perf-acceptance-criteria",
        "ascendc-performance-review",
    ):
        root = (skills / name).resolve()
        entry = root / "SKILL.md"
        assert entry.is_file(), entry
        pending, reachable = [entry], set()
        while pending:
            path = pending.pop()
            if path in reachable:
                continue
            assert path.is_file(), path
            reachable.add(path)
            prose = re.sub(r"```[^\n]*\n.*?```", "", path.read_text(), flags=re.S)
            for link in re.findall(r"\]\(([^)]+)\)", prose):
                if "://" in link or link.startswith("#") or link.startswith("skill:"):
                    continue
                target = (path.parent / link.split("#")[0]).resolve()
                assert target.exists(), (path, link)
                assert main.resolve() not in target.parents, (path, link)
                if target.suffix == ".md" and root in target.parents:
                    pending.append(target)
        # The intentionally unlinked knowledge template is an authoring asset.
        docs = set(root.rglob("*.md")) - {
            root / "references/perf-review-knowledge/_template.md"
        }
        assert docs <= reachable, (name, docs - reachable)
        if name in STAGE_SKILLS.values():
            assert "阶段内不加载主编排 Skill" in entry.read_text()
            assert "ops-perf-acceptance-criteria" in entry.read_text()
