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

"""离线 ST：核间尾块负载均衡（core_balance）回归。

锁住「core_balance 由分核单位 core_split_units 决定，而不是核内 UB tile_count」这条修正——
尤其是 row_resident 算子走自动派生路径、且行数不被核数整除时：
  - eta 必须 = (rows/cores)/ceil(rows/cores)（按 rows，不是 tile_count）
  - 行数整除核数 → eta==1.0（无核间罚）
  - 行数非整除 → eta<1.0，且明确 ≠ 用 tile_count 会算出的值（防回退到旧 bug）

背景：早期 core_balance 误用 UB tile_count 当分核粒度，对均分整除的算子（如 vchain
8388608/64）误罚出假 BELOW_FLOOR。见 references/perf_case_schema.md「核间尾块负载均衡」。

用法: python3 tests/offline/test_core_balance.py
"""

from __future__ import annotations

import math
import importlib
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKILL_DIR / "scripts"))
bridge = importlib.import_module("_engine_bridge")
flow_runtime = importlib.import_module("flow_runtime")

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "core_balance_resident_cases.json"
)

fails: list[str] = []


def ck(label: str, cond: bool, detail: str = "") -> None:
    if not cond:
        fails.append(f"{label}{(' :: ' + detail) if detail else ''}")


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def eta_of(units: float, cores: float) -> float:
    return (units / cores) / math.ceil(units / cores)


def main() -> int:
    def build(case, attrs, ctx):
        return flow_runtime.build("rms_norm", case, attrs, ctx)

    _, rows = bridge.evaluate_spec(build, FIXTURE)

    n = 0
    for row in rows:
        cd, r = row["case_dict"], row["result"]
        cid = r["case_id"]
        cores = float(r["active_vector_cores"])
        csu = (cd.get("tiling") or {}).get("core_split_units")
        n_rows = int(cid.split("rows")[1].split("_")[0])

        # 1) 自动派生路径下 core_split_units 必须 = 行数（分核单位），不是 UB tile_count
        ck(
            f"{cid} core_split_units==rows",
            csu == n_rows,
            f"core_split_units={csu} 应={n_rows}（resident 按行分核）",
        )

        # 2) core_balance 必须按 core_split_units 算
        exp = eta_of(n_rows, cores)
        ck(
            f"{cid} core_balance==eta(rows)",
            approx(r["core_balance"], exp),
            f"core_balance={r['core_balance']:.4f} 应={exp:.4f}",
        )

        # 3) 整除/非整除语义
        if n_rows % cores == 0:
            ck(
                f"{cid} 整除→eta==1",
                approx(r["core_balance"], 1.0),
                f"core_balance={r['core_balance']:.4f}",
            )
        else:
            ck(
                f"{cid} 非整除→eta<1",
                r["core_balance"] < 1.0 - 1e-9,
                f"core_balance={r['core_balance']:.4f} 应<1",
            )
            # 4) 防回退：明确 ≠ 用 tile_count 会算出的旧值
            tc_eta = eta_of(float(r["tile_count"]), cores)
            if abs(exp - tc_eta) > 1e-3:
                ck(
                    f"{cid} 没回退到 tile_count 口径",
                    not approx(r["core_balance"], tc_eta),
                    f"core_balance={r['core_balance']:.4f} 等于了 tile_count 口径 {tc_eta:.4f}（旧 bug）",
                )
        n += 1

    if fails:
        print(f"FAIL  ({len(fails)} 处 / {n} cases)")
        for f in fails:
            print("  -", f)
        return 1
    print(
        f"PASS  ({n} cases：resident 按行分核 core_balance；整除 eta=1、非整除 eta<1 且不回退 tile_count 口径)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
