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

"""Compare measured mixed transcendental+ALU flows against three overlap models.

Reads raw JSONL (measured_cycles in 1GHz sys-counter ticks) plus the manifest
overlap_calibration block (transcendental/alu counts per group), then reports
measured vector-cycles-per-group vs:

  sum_model       = T_exp + T_alu        (current evaluator: sums per-unit times)
  max_resource    = max(issue, exp_pipe, alu_pipe)
  issue_port      = total_instr / 2

Vector-cycle basis: vector_core 1.65 GHz, sys-counter 1.0 GHz.
A "group" = one transcendental op plus its accompanying ALU ops; there are
iterations * 16 groups per run (16 group-slots per VF iteration).
"""

from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path

VEC_PER_SYS = 1.65  # vector cycles per sys-counter tick
GROUPS_PER_ITER = 16  # group-slots per VF iteration
EXP_PIPE_CYC = 2.0  # measured pure-exp/sqrt: 2.0 vec-cyc per transcendental
ALU_CYC = 0.5  # measured pure-add: 0.5 vec-cyc per ALU op (dual-issue)
ISSUE_SLOTS = 2.0  # VALU issue ports


def models(t, a):
    """t transcendental ops/group, a ALU ops/group -> dict of model cyc/group."""
    issue = (t + a) / ISSUE_SLOTS
    exp_pipe = t * EXP_PIPE_CYC
    alu_pipe = a * ALU_CYC
    return {
        "sum_model": exp_pipe + alu_pipe,
        "max_resource": max(issue, exp_pipe, alu_pipe),
        "issue_port": issue,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=root / "data/overlap_raw.jsonl")
    ap.add_argument(
        "--manifest", type=Path, default=root / "config/calibration_cases.json"
    )
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cal = {
        c["case_name"]: c.get("overlap_calibration")
        for c in manifest["implemented_cases"]
    }

    by_case: dict[str, list[float]] = {}
    meta: dict[str, dict] = {}
    for line in args.raw.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        groups = r["iterations"] * GROUPS_PER_ITER
        vec_per_group = r["measured_cycles"] * VEC_PER_SYS / groups
        by_case.setdefault(r["case_name"], []).append(vec_per_group)
        meta[r["case_name"]] = r

    # transcendental/alu counts per group; for pure-op anchors infer directly.
    anchor = {
        "add_fp32_independent": (0, 1),
        "exp_fp32": (1, 0),
        "sqrt_fp32": (1, 0),
        "exp_then_sub_fp32": (1, 1),
    }

    hdr = f"{'case':24} {'meas':>6} {'sum':>6} {'maxR':>6} {'issue':>6}  {'err_sum':>8} {'err_max':>8}  dep"
    print(hdr)
    print("-" * len(hdr))
    for case, vals in by_case.items():
        meas = statistics.fmean(vals)
        cinfo = cal.get(case)
        if cinfo:
            t = cinfo["transcendental_count_per_group"]
            a = cinfo["alu_count_per_group"]
        else:
            t, a = anchor.get(case, (0, 0))
        m = models(t, a)
        err_sum = (
            (meas - m["sum_model"]) / m["sum_model"] * 100 if m["sum_model"] else 0.0
        )
        err_max = (
            (meas - m["max_resource"]) / m["max_resource"] * 100
            if m["max_resource"]
            else 0.0
        )
        dep = "indep" if "indep" in case else ("dep" if (t and a) else "-")
        print(
            f"{case:24} {meas:6.2f} {m['sum_model']:6.2f} {m['max_resource']:6.2f} "
            f"{m['issue_port']:6.2f}  {err_sum:+7.1f}% {err_max:+7.1f}%  {dep}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
