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

"""离线 ST（无需板子）：跑通每个算子的计算流，断言推导出的 case 与引擎结果自洽。

自动发现 references/examples/<name>_cases.json，按约定配 references/examples/<name>_flow.json，
经通用解释器 flow_runtime 路由+解释，对每个 case：build → 引擎 evaluate，检查：
  - 不抛异常（引擎已强校验 op/dtype 合法、UB 可行）
  - memory 字节 ≥0 且至少一项 >0；vector.instructions 非空
  - bound ∈ {mte2_bound, mte3_bound, vector_bound}
  - theoretical_time_us > 0，cp4_target ≥ theoretical，0<eff≤1
这是 skill 的门禁：改 flow/解释器/引擎后一条命令即知是否跑挂。

用法: python3 tests/offline/test_builders.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2]
SCRIPTS = SKILL_DIR / "scripts"
EXAMPLES = SKILL_DIR / "references/examples"
sys.path.insert(0, str(SCRIPTS))
bridge = importlib.import_module("_engine_bridge")
flow_runtime = importlib.import_module("flow_runtime")


def load_builder(name):
    """约定 <name>_cases.json ↔ <name>_flow.json，经通用解释器路由+解释（无 per-op .py）。"""
    flow_path = EXAMPLES / f"{name}_flow.json"
    if not flow_path.exists():
        raise FileNotFoundError(
            f"缺少计算流: {flow_path}（约定 <name>_cases.json ↔ <name>_flow.json）"
        )
    return lambda case, attrs, ctx: flow_runtime.build(name, case, attrs, ctx)


def check_result(name, r):
    cid = r["case_id"]
    errs = []
    if not (r["theoretical_time_us"] > 0):
        errs.append("theoretical_time_us 应 >0")
    if r["bound"] not in ("mte2_bound", "mte3_bound", "vector_bound"):
        errs.append(f"bound 非法: {r['bound']}")
    for k in ("mte2_time_us", "mte3_time_us", "vector_time_us"):
        if r[k] < 0:
            errs.append(f"{k} <0")
    if not (r["cp4_target_time_us"] >= r["theoretical_time_us"] - 1e-9):
        errs.append("cp4_target 应 ≥ theoretical")
    if not (0 < r["target_efficiency"] <= 1):
        errs.append("target_efficiency 应 ∈(0,1]")
    return [f"[{name}/{cid}] {e}" for e in errs]


def check_case_dict(name, cid, cd):
    errs = []
    mem = cd.get("memory") or {}
    byts = [
        mem.get("gm_read_bytes", 0),
        mem.get("gm_write_bytes", 0),
        mem.get("l2_read_bytes", 0),
        mem.get("l2_write_bytes", 0),
    ]
    byts = [b for b in byts if isinstance(b, (int, float))]
    if any(b < 0 for b in byts):
        errs.append("memory 字节出现负值")
    if not any(b > 0 for b in byts):
        errs.append("memory 字节全为 0（计算流没搬数据？）")
    if not (cd.get("vector", {}).get("instructions")):
        errs.append("vector.instructions 为空")
    return [f"[{name}/{cid}] {e}" for e in errs]


def main():
    case_files = sorted(EXAMPLES.glob("*_cases.json"))
    if not case_files:
        print("FAIL: 未发现任何 *_cases.json")
        return 1
    fails, n_cases = [], 0
    for cf in case_files:
        name = cf.name[: -len("_cases.json")]
        try:
            build = load_builder(name)
            spec, rows = bridge.evaluate_spec(build, cf)
        except Exception as exc:  # build/校验/引擎 任一报错都算失败
            fails.append(f"[{name}] 跑挂: {type(exc).__name__}: {exc}")
            continue
        for row in rows:
            n_cases += 1
            fails += check_case_dict(name, row["case"]["case_id"], row["case_dict"])
            fails += check_result(name, row["result"])

    ops = [cf.name[: -len("_cases.json")] for cf in case_files]
    if fails:
        print(f"FAIL  ({len(fails)} 处问题, {n_cases} cases, 算子: {', '.join(ops)})")
        for f in fails:
            print("  -", f)
        return 1
    print(f"PASS  ({n_cases} cases / {len(case_files)} 算子: {', '.join(ops)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
