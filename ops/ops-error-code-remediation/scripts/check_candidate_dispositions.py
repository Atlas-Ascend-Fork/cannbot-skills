#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Freeze remediation candidates and fail closed on missing or inconsistent dispositions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

from scan_error_code_candidates import (
    DEFAULT_EXCLUDES,
    SOURCE_SUFFIXES,
    iter_sources,
    scan,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def snapshot(root, scope):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(
            iter_sources(root, set(scope["excludes"]), set(scope["suffixes"]))
        )
    }


def discover(root, scope):
    before = snapshot(root, scope)
    rows = [
        asdict(row)
        for row in scan(
            root,
            set(scope["excludes"]),
            set(scope["suffixes"]),
            scope["include_definitions"],
        )
    ]
    if snapshot(root, scope) != before:
        raise ValueError("source changed during scan; rerun on a stable snapshot")
    return before, rows


def tool_hashes():
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("scan_error_code_candidates.py", "error_code_triggers.py")
    }


def initialize(root, excludes=(), suffixes=(), include_definitions=False):
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("scan root must be a directory")
    scope = {
        "excludes": sorted(DEFAULT_EXCLUDES | set(excludes)),
        "suffixes": sorted(
            SOURCE_SUFFIXES | {s if s.startswith(".") else "." + s for s in suffixes}
        ),
        "include_definitions": include_definitions,
    }
    hashes, findings = discover(root, scope)
    baseline = {
        "root": str(root),
        "scope": scope,
        "source_sha256": hashes,
        "tool_sha256": tool_hashes(),
        "findings": findings,
    }
    return {
        "schema_version": 1,
        "baseline": baseline,
        "baseline_sha256": digest(baseline),
        "decisions": [
            {
                "id": digest(row),
                "status": "PENDING",
                "root_cause": "",
                "reason": "",
                "evidence": {},
                "after": None,
            }
            for row in findings
        ],
    }


def location(row):
    return row["path"], row["line"], row["entry"]


def has_text(value):
    return isinstance(value, str) and bool(value.strip())


def check(ledger):
    baseline = ledger["baseline"]
    if ledger["schema_version"] != 1 or digest(baseline) != ledger["baseline_sha256"]:
        raise ValueError(
            "invalid or edited baseline; preserve the pre-remediation snapshot"
        )
    if baseline["tool_sha256"] != tool_hashes():
        raise ValueError(
            "scanner changed since baseline; reconcile with the original snapshot, do not silently reset it"
        )
    root = Path(baseline["root"]).resolve(strict=True)
    hashes, current = discover(root, baseline["scope"])
    original = {digest(row): row for row in baseline["findings"]}
    if len(original) != len(baseline["findings"]):
        raise ValueError("duplicate baseline candidates")
    decisions = ledger["decisions"]
    ids = [row["id"] for row in decisions]
    if len(ids) != len(set(ids)) or set(ids) != set(original):
        raise ValueError("decisions must cover every baseline candidate exactly once")
    remaining = {location(row): row for row in current}
    covered = set()
    problems = []
    for decision in decisions:
        old = original[decision["id"]]
        label = f"{old['path']}:{old['line']} {old['entry']}"
        status = decision["status"]
        if status not in {"FIXED", "RETAINED", "REMOVED", "BLOCKED"}:
            problems.append(f"{label}: missing disposition ({status})")
            continue
        if status == "BLOCKED":
            problems.append(f"{label}: blocked: {decision.get('reason', '')}")
            continue
        if not all(has_text(decision.get(key)) for key in ("root_cause", "reason")):
            problems.append(f"{label}: root cause and reason required")
        evidence = decision.get("evidence", {})
        required = {"macro_definition", "call_chain"}
        if status == "FIXED":
            required |= {"registration", "build", "negative_test", "first_error"}
        elif status == "RETAINED":
            required |= {"retention_basis", "negative_test"}
        else:
            required |= {"removal_basis"}
        if not isinstance(evidence, dict) or any(
            not has_text(evidence.get(key)) for key in required
        ):
            problems.append(
                f"{label}: missing evidence fields: {', '.join(sorted(required))}"
            )
        changed = baseline["source_sha256"].get(old["path"]) != hashes.get(old["path"])
        if status in {"FIXED", "REMOVED"} and not changed:
            problems.append(f"{label}: marked {status} but source is unchanged")
        after = decision.get("after")
        if status == "REMOVED":
            if after is not None:
                problems.append(
                    f"{label}: REMOVED must not declare a live after location"
                )
            continue
        if (
            not isinstance(after, dict)
            or not has_text(after.get("path"))
            or type(after.get("line")) is not int
            or after["line"] < 1
            or not has_text(after.get("entry"))
        ):
            problems.append(
                f"{label}: valid after path/line/entry required; remap shifted lines explicitly"
            )
            continue
        path = (root / after["path"]).resolve(strict=True)
        path.relative_to(root)
        relative = path.relative_to(root).as_posix()
        if relative != after["path"] or relative not in hashes:
            raise ValueError(f"invalid or out-of-scope after source: {after['path']}")
        lines = path.read_text(encoding="utf-8").splitlines()
        if after["line"] > len(lines) or not re.search(
            r"\b" + re.escape(after["entry"]) + r"\b", lines[after["line"] - 1]
        ):
            problems.append(f"{label}: after location does not match actual source")
        key = location(after)
        if status == "RETAINED":
            if old["kind"] == "deprecated":
                problems.append(
                    f"{label}: deprecated code cannot be closed as RETAINED"
                )
            if key not in remaining:
                problems.append(
                    f"{label}: retained candidate missing at declared after location"
                )
            if key in covered:
                problems.append(f"{label}: after candidate assigned more than once")
            covered.add(key)
        elif key in remaining:
            problems.append(f"{label}: marked FIXED but still a governance candidate")
    for key in remaining.keys() - covered:
        problems.append(
            f"{key[0]}:{key[1]} {key[2]}: residual/new candidate without justified retention"
        )
    if snapshot(root, baseline["scope"]) != hashes:
        raise ValueError("source changed during validation")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--root", required=True, type=Path)
    init.add_argument("--output", required=True, type=Path)
    init.add_argument("--exclude-dir", action="append", default=[])
    init.add_argument("--suffix", action="append", default=[])
    init.add_argument("--include-definitions", action="store_true")
    verify = sub.add_parser("check")
    verify.add_argument("--ledger", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "init":
            ledger = initialize(
                args.root, args.exclude_dir, args.suffix, args.include_definitions
            )
            # Never overwrite the only pre-remediation baseline.
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(ledger, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(
                f"Frozen {len(ledger['decisions'])} candidates; all dispositions PENDING"
            )
            return 0
        problems = check(json.loads(args.ledger.read_text(encoding="utf-8")))
        for problem in problems:
            print(f"BLOCKED: {problem}")
        print(
            "Candidate disposition reconciliation: "
            + ("FAIL" if problems else "PASS (semantic evidence still requires review)")
        )
        return 1 if problems else 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: disposition check incomplete (blocked): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
