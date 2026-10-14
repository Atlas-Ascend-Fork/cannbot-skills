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

"""离线 ST：超越函数+ALU overlap 标定回归。

Verifies that evaluate_case aggregates mixed vector flows with the
max(issue_port, sub-pipe) rule calibrated on Ascend950 microbench
(tests/calibration/instruments/regbase-microbench/docs/overlap_calibration_findings.md), NOT the old
sum-all-instruction-time rule. Run:

    python3 tests/offline/test_overlap_model.py
"""

from __future__ import annotations
import importlib
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKILL_DIR / "scripts"))
ev = importlib.import_module("evaluate_a5_vector_roofline")

HW = ev.load_json(SKILL_DIR / "references/hardware/ascend950pr.json")
DB = ev.load_json(SKILL_DIR / "references/vector_instruction_types.json")

# fp32: bytes_per_instruction 256 / width 4 = 64 elements per vector instruction.
# elements=64, count=k  =>  exactly k vector instructions, so cyc/group math is direct.
ELEMS = 64


def vec_time(instr_list):
    case = {
        "case_id": "t",
        "active_vector_cores": 1,
        "memory": {},
        "vector": {"instructions": instr_list},
    }
    return ev.evaluate_case(case, HW, DB)["vector_time_us"]


def add(count):
    return {"op": "Add", "dtype": "fp32", "elements": ELEMS, "count": count}


def exp(count):
    return {"op": "Exp", "dtype": "fp32", "elements": ELEMS, "count": count}


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


def main() -> int:
    one_exp = vec_time([exp(1)])
    one_add = vec_time([add(1)])
    fails = []

    # Pure-op anchors: 1 exp = 2 cyc, 1 add = 0.5 cyc  =>  exp/add = 4.0.
    if not approx(one_exp / one_add, 4.0):
        fails.append(f"exp/add ratio {one_exp / one_add:.3f} != 4.0")

    checks = [
        # (label, instructions, expected multiple of one_exp, old sum multiple)
        ("exp+1add", [exp(1), add(1)], 1.0, 1.25),  # max(2, .5, issue 1.0)=2.0
        ("exp+4add", [exp(1), add(4)], 1.25, 2.0),  # max(2, 2, issue 2.5)=2.5
        ("exp+8add", [exp(1), add(8)], 2.25, 3.0),  # max(2, 4, issue 4.5)=4.5
    ]
    for label, instrs, exp_mult, sum_mult in checks:
        got = vec_time(instrs)
        want_max = exp_mult * one_exp
        want_sum = sum_mult * one_exp
        if not approx(got, want_max):
            fails.append(f"{label}: vector_time {got:.4g} != max-model {want_max:.4g}")
        # Guard: the new model must be strictly below the old sum model whenever
        # they differ (i.e. real overlap is captured).
        if sum_mult > exp_mult and not (got < want_sum - 1e-9):
            fails.append(f"{label}: {got:.4g} not below old sum {want_sum:.4g}")

    # Pure ALU (no transcendental) stays issue/alu bound: 4 adds == 4 * one add.
    if not approx(vec_time([add(4)]), 4 * one_add):
        fails.append("4*add != 4 x one add")

    if fails:
        print("FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print(
        "PASS  (one_exp=%.4g us, one_add=%.4g us; exp+4add=%.4g us vs old-sum %.4g us)"
        % (one_exp, one_add, vec_time([exp(1), add(4)]), 2.0 * one_exp)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
