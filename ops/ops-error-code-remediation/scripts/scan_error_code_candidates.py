#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------
"""Scan C/C++ sources for deprecated CANN codes and E*9*** reporting candidates."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from error_code_triggers import (
    CALL_RE,
    CODE_RE,
    CUBE_ENTRIES,
    DEFAULT_EXCLUDES,
    DEPRECATED_MACROS,
    DEPRECATED_RE,
    E9_RE,
    ERRNO_REPORT_NAMES,
    GRAPH_PARSER_ENTRIES,
    INTERNAL_REPORT_NAMES,
    LOG_ONLY_NAMES,
    MIXED_FORMAL_REPORT_NAMES,
    NULL_ENTRIES,
    RETURN_CHECK_ENTRIES,
    SOURCE_SUFFIXES,
    TILING_EXEC_ENTRIES,
    VECTOR_ENTRIES,
    bare_name,
    errno_code,
    first_argument,
    first_trigger,
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    entry: str
    code: str
    confidence: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Repository or source directory to scan")
    parser.add_argument("--format", choices=("json", "csv", "markdown"), default="json")
    parser.add_argument("--output", type=Path, help="Write output to this file instead of stdout")
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Additional directory name to exclude; repeat as needed (for example experimental)",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        default=[],
        help="Additional source suffix such as .cu; repeat as needed",
    )
    parser.add_argument(
        "--include-definitions",
        action="store_true",
        help="Also report matches inside preprocessor macro definitions",
    )
    return parser.parse_args()


def iter_sources(root: Path, excludes: set[str], suffixes: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            relative_parts = path.relative_to(root).parts[:-1]
        except ValueError:
            continue
        if any(part in excludes for part in relative_parts):
            continue
        yield path


def classify_statement(relative: str, line_no: int, raw: str) -> Finding | None:
    text = raw.strip()
    if not text or text.startswith("//"):
        return None
    if re.search(r'"(?:ErrCode|errCode)"\s*:', text):
        return None

    deprecated_code = DEPRECATED_RE.search(text)
    deprecated_entry = first_trigger(text, DEPRECATED_MACROS)
    if deprecated_code or deprecated_entry:
        code = deprecated_code.group(0) if deprecated_code else DEPRECATED_MACROS[deprecated_entry]
        return Finding(relative, line_no, "deprecated", deprecated_entry or "direct-code", code, "high", text)

    call = CALL_RE.search(text)
    entry = call.group(1) if call else ""
    bare_entry = bare_name(entry) if entry else ""
    e9_code = E9_RE.search(text)
    if e9_code:
        return Finding(relative, line_no, "e9-direct", entry or "direct-code", e9_code.group(0), "high", text)

    if bare_entry in NULL_ENTRIES | VECTOR_ENTRIES | CUBE_ENTRIES | GRAPH_PARSER_ENTRIES:
        if bare_entry in NULL_ENTRIES:
            kind = "e9-implicit-null"
        elif bare_entry in VECTOR_ENTRIES:
            kind = "e9-implicit-vector"
        elif bare_entry in CUBE_ENTRIES:
            kind = "e9-implicit-cube"
        else:
            kind = "e9-implicit-wrapper"
        return Finding(relative, line_no, kind, entry, "", "medium", text)

    if bare_entry in TILING_EXEC_ENTRIES:
        return Finding(relative, line_no, "e9-implicit-tiling-exec", entry, "", "medium", text)

    if bare_entry in RETURN_CHECK_ENTRIES:
        if "uniqueExecutor.get() != nullptr" in text:
            return Finding(
                relative,
                line_no,
                "unreachable-executor-null-check",
                entry,
                "ACLNN_ERR_INNER_NULLPTR",
                "high",
                text,
            )
        errno_match = re.search(r"\b(ACLNN_ERR_[A-Z0-9_]+|ACL_ERROR_[A-Z0-9_]+|[135]\d{5})\b", text)
        mapped = errno_code(errno_match.group(1)) if errno_match else ""
        if mapped and E9_RE.fullmatch(mapped):
            return Finding(
                relative,
                line_no,
                "e9-errno-inner-return",
                entry,
                mapped,
                "high",
                text,
            )
        return Finding(relative, line_no, "return-check-dynamic", entry, mapped, "medium", text)

    if bare_entry == "OP_LOGE" or bare_entry in ERRNO_REPORT_NAMES:
        first_arg = first_argument(text, entry)
        mapped = errno_code(first_arg)
        if mapped == "EZ1001":
            return None
        if mapped == "EZ9903":
            kind, confidence = "e9-errno-runtime", "high"
        elif mapped == "EZ9999":
            kind, confidence = "e9-errno-inner", "high"
        else:
            kind = "e9-implicit-op-loge"
            confidence = "medium"
        return Finding(relative, line_no, kind, entry, mapped or first_arg, confidence, text)

    if bare_entry in INTERNAL_REPORT_NAMES:
        return Finding(relative, line_no, "e9-internal-report-dynamic", entry, "", "medium", text)

    if bare_entry in MIXED_FORMAL_REPORT_NAMES:
        if CODE_RE.search(text):
            return None
        return Finding(relative, line_no, "registered-report-dynamic", entry, "", "medium", text)

    if bare_entry in LOG_ONLY_NAMES:
        return None

    return None


def logical_statements(text: str) -> Iterable[tuple[int, str, bool]]:
    """Yield multiline reporting statements and direct code references once."""
    lines = text.splitlines()
    index = 0
    in_directive = False
    while index < len(lines):
        raw = lines[index]
        is_directive = in_directive or raw.lstrip().startswith("#")
        in_directive = is_directive and raw.rstrip().endswith("\\")
        if not (CALL_RE.search(raw) or DEPRECATED_RE.search(raw) or E9_RE.search(raw)):
            index += 1
            continue
        start = index + 1
        chunk = raw.strip()
        balance = chunk.count("(") - chunk.count(")")
        while index + 1 < len(lines) and balance > 0:
            index += 1
            addition = lines[index].strip()
            chunk += " " + addition
            balance += addition.count("(") - addition.count(")")
            if index + 1 - start > 30:
                break
        yield start, re.sub(r"\s+", " ", chunk).strip(), is_directive
        index += 1


def scan(root: Path, excludes: set[str], suffixes: set[str], include_definitions: bool) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(iter_sources(root, excludes, suffixes)):
        relative = path.relative_to(root).as_posix()
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        for line_no, statement, is_definition in logical_statements(content):
            if is_definition and not include_definitions:
                continue
            finding = classify_statement(relative, line_no, statement)
            if finding:
                findings.append(finding)
    return findings


def json_text(root: Path, findings: list[Finding]) -> str:
    return json.dumps(
        {
            "root": str(root.resolve()),
            "summary": dict(sorted(Counter(item.kind for item in findings).items())),
            "total": len(findings),
            "findings": [asdict(item) for item in findings],
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def markdown_text(root: Path, findings: list[Finding]) -> str:
    counts = Counter(item.kind for item in findings)
    lines = ["# CANN error-code candidates", "", f"Root: `{root.resolve()}`", "", "## Summary", ""]
    lines.extend(f"- `{kind}`: {count}" for kind, count in sorted(counts.items()))
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "| File | Line | Kind | Entry | Code/arg | Confidence |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for item in findings:
        lines.append(
            f"| `{item.path}` | {item.line} | {item.kind} | `{item.entry}` | "
            f"`{item.code}` | {item.confidence} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(output, findings: list[Finding]) -> None:
    writer = csv.DictWriter(output, fieldnames=Finding.__dataclass_fields__.keys())
    writer.writeheader()
    writer.writerows(asdict(item) for item in findings)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"error: --root is not a directory: {root}", file=sys.stderr)
        return 2

    excludes = DEFAULT_EXCLUDES | set(args.exclude_dir)
    suffixes = SOURCE_SUFFIXES | {item if item.startswith(".") else f".{item}" for item in args.suffix}
    findings = scan(root, excludes, suffixes, args.include_definitions)

    if args.format == "csv":
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
                write_csv(handle, findings)
        else:
            write_csv(sys.stdout, findings)
    else:
        rendered = json_text(root, findings) if args.format == "json" else markdown_text(root, findings)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
