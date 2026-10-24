# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Read CATLASS cards from the installed cannbot-knowledge checkout.

This CLI preserves the workflow's query/get entry points. Knowledge is held by
cannbot-knowledge; this Skill does not install or copy a private bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


CATLASS_ROUTE = Path("knowledge/ops/catlass")
ATTENTION = Path("operators/direct/transformer/attention")
FAMILY_PREFIXES = {
    "linear-attention": "linear_attention_",
    "sparse-attention": "sparse_attention_",
    "sparse-flash-mla": "sparse_flash_mla_",
    "block-sparse-attention-grad": "block_sparse_attention_grad_",
    "flash-attention-grad": "flash_attention_grad_",
    "flash-attention": "flash_attention_",
}
VOCABULARY = json.loads(
    (Path(__file__).resolve().parents[1] / "references/query-vocabulary.yaml").read_text(encoding="utf-8")
)
FAMILY_ALIASES = {
    alias.lower().replace("-", "_"): family
    for family, definition in VOCABULARY["operator_families"].items()
    for alias in (family, *definition["aliases"])
}
PROFILE_ALIASES = {
    "catlass": "apis",
    "workflow": "guides",
    "operator": "operators",
    "apis": "apis",
    "guides": "guides",
    "operators": "operators",
}
OKF_TYPES = {"apis": "API", "guides": "Guide", "operators": "Operator"}
SAFE_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")


def _output(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _knowledge_root(project_root: Path, explicit: str | None) -> Path:
    configured = explicit or os.environ.get("CANNBOT_KNOWLEDGE_ROOT")
    if configured:
        root = Path(configured).resolve()
    else:
        root = (
            project_root.resolve()
            / ".cannbot/dependencies/catlass-op-generator/cannbot-knowledge"
        )
    catlass = root / CATLASS_ROUTE
    if not catlass.is_dir():
        raise ValueError(
            f"CATLASS knowledge route is unavailable: {catlass}; "
            "install catlass-op-generator and its cannbot-knowledge dependency"
        )
    return catlass


def _scalar(value: str) -> str:
    return value.strip().strip("'\"")


def _card(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    if not match:
        raise ValueError(f"missing frontmatter: {path}")
    header = match.group(1)
    metadata: dict[str, Any] = {}
    current_list: str | None = None
    for line in header.splitlines():
        top = re.match(r"^([a-z_]+):(?:\s*(.*))?$", line)
        if top:
            key, raw = top.groups()
            current_list = key if not raw else None
            if raw:
                metadata[key] = (
                    [_scalar(value) for value in raw[1:-1].split(",") if value.strip()]
                    if key in {"tags", "platforms"} and raw.startswith("[") and raw.endswith("]")
                    else _scalar(raw)
                )
            elif key in {"tags", "platforms"}:
                metadata[key] = []
            continue
        item = re.match(r"^-\s+(.+)$", line)
        if item and current_list in {"tags", "platforms"}:
            metadata[current_list].append(_scalar(item.group(1)))
    return metadata, text[match.end():]


def _family(value: str | None) -> str | None:
    if not value:
        return None
    key = value.lower().replace("-", "_")
    name = FAMILY_ALIASES.get(key, value.lower().replace("_", "-"))
    if name not in FAMILY_PREFIXES:
        raise ValueError(f"unsupported CATLASS family: {value}")
    return name


def _paths(catlass: Path, profile: str | None, family: str | None) -> list[Path]:
    paths: list[Path] = []
    for current in ("apis", "guides", "operators"):
        if profile and profile != current:
            continue
        root = catlass / current
        if not root.is_dir():
            continue
        for path in root.rglob("*.md"):
            if path.name == "index.md":
                continue
            if family:
                matched_family = next(
                    (
                        name for name, prefix in sorted(
                            FAMILY_PREFIXES.items(), key=lambda item: len(item[1]), reverse=True
                        )
                        if path.name.startswith(prefix)
                    ),
                    None,
                )
                if current != "operators" or matched_family != family:
                    continue
            paths.append(path)
    return sorted(paths)


def initialize(catlass: Path) -> dict[str, Any]:
    """Confirm the installed knowledge route; do not duplicate its content."""
    count = len(_paths(catlass, None, None))
    if count == 0:
        raise ValueError(f"CATLASS knowledge route has no cards: {catlass}")
    return {"status": "passed", "count": count, "knowledge_root": str(catlass)}


def query(
    catlass: Path,
    profile: str | None,
    tags: list[str],
    family: str | None,
    arch: str | None,
    text: str | None,
    compact: bool,
    include_draft: bool,
) -> dict[str, Any]:
    wanted_profile = PROFILE_ALIASES.get(profile) if profile else None
    if profile and not wanted_profile:
        raise ValueError(f"unsupported profile: {profile}")
    wanted_family = _family(family)
    terms = [term.lower() for term in (text or "").split() if term]
    wanted_arch = arch.lower().replace("_", "-") if arch else None
    results = []
    for path in _paths(catlass, wanted_profile, wanted_family):
        meta, body = _card(path)
        profile_name = path.relative_to(catlass).parts[0]
        if meta.get("status") not in {"stable", "draft"}:
            continue
        if meta.get("status") != "stable" and not wanted_family and not include_draft:
            continue
        if tags and any(tag not in meta.get("tags", []) for tag in tags):
            continue
        platforms = [value.lower() for value in meta.get("platforms", [])]
        if wanted_arch:
            required = (
                "950" if "950" in wanted_arch
                else "a2" if any(v in wanted_arch for v in ("a2", "2201", "910b"))
                else wanted_arch
            )
            if required not in platforms:
                continue
        haystack = " ".join([path.name, str(meta.get("title", "")), str(meta.get("description", "")), body]).lower()
        if terms and not all(term in haystack for term in terms):
            continue
        item = {
            "path": path.relative_to(catlass).as_posix(),
            "type": OKF_TYPES[profile_name],
            "title": meta.get("title", ""),
            "description": meta.get("description", ""),
            "status": meta.get("status", ""),
        }
        if not compact:
            item["tags"] = meta.get("tags", [])
            item["platforms"] = meta.get("platforms", [])
        results.append(item)
    return {"status": "passed", "count": len(results), "results": results}


def get(catlass: Path, value: str) -> dict[str, Any]:
    if not value or not SAFE_PATH.fullmatch(value):
        raise ValueError("concept path must be a safe relative path")
    relative = Path(value)
    if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
        raise ValueError("concept path must be a safe relative path")
    target = catlass / relative
    if target.is_symlink() or not target.is_file() or target.suffix != ".md" or target.name == "index.md":
        raise ValueError("concept path must resolve to a CATLASS knowledge card")
    if not target.resolve().is_relative_to(catlass.resolve()):
        raise ValueError("concept path escapes CATLASS knowledge route")
    return {"status": "passed", "path": relative.as_posix(), "content": target.read_text(encoding="utf-8")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("initialize", "query", "get", "validate"):
        item = subparsers.add_parser(command)
        item.add_argument("--project-root", required=True, type=Path)
        item.add_argument("--knowledge-root")
        if command == "query":
            item.add_argument("--type")
            item.add_argument("--tag", action="append", default=[])
            item.add_argument("--family")
            item.add_argument("--arch")
            item.add_argument("--text")
            item.add_argument("--compact", action="store_true")
            item.add_argument("--include-draft", action="store_true")
        if command == "get":
            item.add_argument("--path", required=True)
    args = parser.parse_args()
    try:
        catlass = _knowledge_root(args.project_root, args.knowledge_root)
        if args.command in {"initialize", "validate"}:
            result = initialize(catlass)
        elif args.command == "query":
            result = query(
                catlass, args.type, args.tag, args.family, args.arch, args.text, args.compact, args.include_draft
            )
        else:
            result = get(catlass, args.path)
    except (OSError, ValueError) as exc:
        _output({"status": "failed", "errors": [str(exc)]})
        return 1
    _output(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
