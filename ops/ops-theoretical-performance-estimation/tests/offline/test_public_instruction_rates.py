#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check documented instruction throughput against independent asc-devkit table data.

The fixture records single-instruction elements/cycle and a separate issue count.
It deliberately excludes unlisted dtypes, composite APIs and special modes.
"""

from __future__ import annotations

import csv
import importlib
import math
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKILL_DIR / "scripts"))
engine = importlib.import_module("evaluate_a5_vector_roofline")
DB = engine.load_json(SKILL_DIR / "references/vector_instruction_types.json")
FIXTURE = Path(__file__).parent / "fixtures/asc_devkit_instruction_rates.csv"
ALIASES = {
    "fp16": ("fp16", "float16", "half"),
    "bf16": ("bf16", "bfloat16", "bfloat16_t"),
    "fp32": ("fp32", "float32", "float"),
}
CONVERSIONS = {"Cast", "Truncate", "Pack", "UnPack"}


def evaluate(op, src, dst):
    inst = {"op": op, "dtype": src, "elements": 1024}
    if op in CONVERSIONS:
        inst.update(src_dtype=src, dst_dtype=dst)
    return engine.evaluate_instruction(inst, {}, 1, 1e9, DB)


def main():
    failures = []
    checked = 0
    with FIXTURE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    for row in rows:
        op, src, dst = row["op"], row["src_dtype"], row["dst_dtype"]
        expected_slots = float(row["issue_slots"])
        expected_parallel = (
            float(row["elements_per_cycle_single_instruction"]) * expected_slots
        )
        # Expand spelling aliases for the corrected rows so dtype aliases cannot
        # silently fall back to a different execution model.
        srcs = (
            ALIASES.get(src, (src,))
            if op in {"Log", "MulDstAdd", "MulAddDst"}
            else (src,)
        )
        for spelling in srcs:
            checked += 1
            actual = evaluate(op, spelling, dst)
            if not (
                actual["issue_slots"] == expected_slots
                and math.isclose(
                    actual["elements_per_cycle_per_core"], expected_parallel
                )
                and math.isclose(actual["time_us"], 1024 / expected_parallel / 1000)
            ):
                failures.append(
                    f"{op}/{spelling}->{dst}: expected {expected_parallel:g} elements/cycle/core, got {actual}"
                )

    # Compatibility names follow their documented target's modeled throughput.
    for alias, target in (("Mula", "MulAddDst"), ("CompareScalar", "Compares")):
        for dtype in ("fp16", "bf16", "fp32", "int16", "int32"):
            checked += 1
            actual = evaluate(alias, dtype, dtype)
            expected = evaluate(target, dtype, dtype)
            if actual["time_us"] != expected["time_us"]:
                failures.append(
                    f"{alias}/{dtype} differs from its model target {target}"
                )
    if failures:
        print("FAIL: asc-devkit instruction rates")
        for failure in failures:
            print("  -", failure)
        return 1
    print(
        f"PASS  ({len(rows)} asc-devkit rows; {checked} documented-rate/alias checks)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
