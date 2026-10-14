#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Analyze RegBase microbenchmark JSONL records."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path | None) -> list[dict[str, Any]]:
    text = sys.stdin.read() if path is None else path.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSONL at line {lineno}: {exc}") from exc
        records.append(record)
    if not records:
        raise SystemExit("no records found")
    return records


def case_by_name(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(case["case_name"]): case for case in manifest.get("implemented_cases", [])
    }


def analyze_record(
    record: dict[str, Any],
    manifest_case: dict[str, Any] | None,
    manifest: dict[str, Any],
    cycle_mode: str,
) -> dict[str, Any]:
    cycle_field = "measured_cycles" if cycle_mode == "measured" else "body_cycles"
    elapsed_cycles = float(record.get(cycle_field, 0))
    sys_counter_hz = float(
        manifest.get(
            "sys_counter_frequency_hz",
            manifest.get("cycle_frequency_hz", 1_000_000_000),
        )
    )
    vector_core_hz = float(manifest.get("vector_core_frequency_hz", sys_counter_hz))
    vector_to_sys_ratio = vector_core_hz / sys_counter_hz if sys_counter_hz > 0 else 1.0
    iterations = float(record.get("iterations", 0))
    unroll = float(record.get("unroll", 0))
    logical_ops = float(record.get("logical_ops_per_unroll", 1))
    elements_per_inst = float(record.get("elements_per_inst", 0))
    instruction_count = iterations * unroll * logical_ops
    issue_per_sys_cnt_tick = (
        instruction_count / elapsed_cycles if elapsed_cycles > 0 else 0.0
    )
    issue_per_vector_cycle = (
        issue_per_sys_cnt_tick / vector_to_sys_ratio if vector_to_sys_ratio > 0 else 0.0
    )
    elems_per_sys_cnt_tick = issue_per_sys_cnt_tick * elements_per_inst
    elems_per_vector_cycle = issue_per_vector_cycle * elements_per_inst

    status = int(record.get("status", -1))
    expected_min = (
        manifest_case.get("expected_issue_per_cycle_min") if manifest_case else None
    )
    expected_max = (
        manifest_case.get("expected_issue_per_cycle_max") if manifest_case else None
    )
    class_match = None
    if status == 0 and expected_min is not None and expected_max is not None:
        class_match = (
            float(expected_min) <= issue_per_vector_cycle <= float(expected_max)
        )

    result = {
        **record,
        "cycle_mode": cycle_mode,
        "cycles_used_field": cycle_field,
        "cycles_used": elapsed_cycles,
        "sys_counter_frequency_hz": sys_counter_hz,
        "vector_core_frequency_hz": vector_core_hz,
        "vector_to_sys_counter_ratio": vector_to_sys_ratio,
        "instruction_count": instruction_count,
        "issue_per_sys_cnt_tick": issue_per_sys_cnt_tick,
        "issue_per_vector_cycle": issue_per_vector_cycle,
        "issue_per_cycle": issue_per_vector_cycle,
        "elements_per_sys_cnt_tick_per_core": elems_per_sys_cnt_tick,
        "elements_per_vector_cycle_per_core": elems_per_vector_cycle,
        "elements_per_cycle_per_core": elems_per_vector_cycle,
        "time_us_950pr": elapsed_cycles / 1000.0,
        "class_match": class_match,
    }
    if manifest_case:
        result["expected_class"] = manifest_case.get("expected_class")
        result["expected_issue_per_cycle_min"] = expected_min
        result["expected_issue_per_cycle_max"] = expected_max
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=root / "config/calibration_cases.json"
    )
    parser.add_argument("--input", type=Path, help="JSONL records. Defaults to stdin.")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cycle-mode",
        choices=("measured", "body"),
        default="measured",
        help="measured keeps load/store time; body subtracts empty_cycles.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    records = load_records(args.input)
    by_name = case_by_name(manifest)
    results = [
        analyze_record(
            r, by_name.get(str(r.get("case_name"))), manifest, args.cycle_mode
        )
        for r in records
    ]

    issues_sys = [r["issue_per_sys_cnt_tick"] for r in results]
    issues_vec = [r["issue_per_vector_cycle"] for r in results]
    elems_vec = [r["elements_per_vector_cycle_per_core"] for r in results]
    summary = {
        "schema_version": 1,
        "record_count": len(results),
        "case_name": results[0].get("case_name"),
        "cycle_mode": args.cycle_mode,
        "all_status_ok": all(int(r.get("status", -1)) == 0 for r in results),
        "all_class_match": all(r.get("class_match") is not False for r in results),
        "issue_per_sys_cnt_tick_mean": statistics.fmean(issues_sys),
        "issue_per_vector_cycle_mean": statistics.fmean(issues_vec),
        "issue_per_cycle_mean": statistics.fmean(issues_vec),
        "issue_per_cycle_min": min(issues_vec),
        "issue_per_cycle_max": max(issues_vec),
        "elements_per_vector_cycle_per_core_mean": statistics.fmean(elems_vec),
        "elements_per_cycle_per_core_mean": statistics.fmean(elems_vec),
        "results": results,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
