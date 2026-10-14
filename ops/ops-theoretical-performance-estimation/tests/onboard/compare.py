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

"""上板对照工具：理论下界 vs 950 实测，按验收口径出误差表 + 判定。

读一份实测 CSV（含注释头声明 operator/cases），跑对应 builder 得理论值，
按 case_id join，逐 case 判定：

  measured < theoretical·(1-TOL)  → BELOW_FLOOR  下界被跑穿 = 模型低估硬件能力(bug，最该抓)
  eff = theoretical/measured ≥ target_eff → OK            （健康：实测落在 [理论, cp4] 内）
  否则                                   → LOW_EFF        实测远高于下界：kernel 未优化 或 builder 漏建模

实测 CSV 是 point-in-time 板上快照，**不随 skill 提交**——按 collect_onboard.md 在 950
采数、存到 skill 外任意位置，再用 --file 对照。存在 BELOW_FLOOR 时退出码 1。

CSV 格式（# 注释头 + 列）:
  # operator: softmax
  # cases: softmax_cases.json          (可省，默认 <operator>_cases.json)
  # source: 2026-06-09 950_9589 TTK/profiling Task Duration(us)
  case_id,measured_us,note
  softmax_m_4096x1024,18.5,

用法:
  python3 tests/onboard/compare.py --file <your_measured.csv>     # 对照一份 CSV
  python3 tests/onboard/compare.py --dir  <你的实测目录>          # 对照目录下所有 *.csv（跳过 _ 开头）
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

TOL = 0.03  # 跑穿下界的容噪带：实测低于理论 3% 以上才判 BELOW_FLOOR

_VERDICT_NOTE = {
    "BELOW_FLOOR": "下界被跑穿→模型低估硬件(bug)",
    "OK": "",
    "LOW_EFF": "实测远高于下界(kernel未优化/漏建模)",
}


def classify(measured, theoretical, target_eff, tol=TOL):
    """单 case 判定（纯函数，便于自测）：
    measured < 理论·(1-tol)        → BELOW_FLOOR  下界被跑穿(模型 bug)
    eff = 理论/measured ≥ target_eff → OK
    否则                            → LOW_EFF
    """
    if measured < theoretical * (1 - tol):
        return "BELOW_FLOOR"
    eff = theoretical / measured if measured > 0 else 0.0
    return "OK" if eff >= target_eff - 1e-9 else "LOW_EFF"


SKILL_DIR = Path(__file__).resolve().parents[2]
SCRIPTS = SKILL_DIR / "scripts"
EXAMPLES = SKILL_DIR / "references/examples"
sys.path.insert(0, str(SCRIPTS))
bridge = importlib.import_module("_engine_bridge")
flow_runtime = importlib.import_module("flow_runtime")


def load_builder(op):
    """通用解释器入口（无 per-op .py）：按 <op>_flow.json 路由模版+解释。"""
    return lambda case, attrs, ctx: flow_runtime.build(op, case, attrs, ctx)


def parse_csv(path):
    meta, rows = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if ":" in s:
                k, v = s[1:].split(":", 1)
                meta[k.strip()] = v.strip()
            continue
        parts = [p.strip() for p in s.split(",")]
        if parts[0] == "case_id":
            continue
        rows.append((parts[0], float(parts[1]), parts[2] if len(parts) > 2 else ""))
    return meta, rows


def compare_file(path):
    meta, rows = parse_csv(path)
    op = meta.get("operator")
    if not op:
        raise ValueError(f"{path}: CSV 缺 `# operator:` 头")
    cases_file = EXAMPLES / meta.get("cases", f"{op}_cases.json")
    _, evald = bridge.evaluate_spec(load_builder(op), cases_file)
    by_id = {r["result"]["case_id"]: r["result"] for r in evald}

    is_example = "example" in (meta.get("source", "").lower())
    title = f"算子: {op}    来源: {meta.get('source', '?')}"
    if is_example:
        title += "   ⚠ 示例数据（非真实实测，仅演示对照工具）"
    print("=" * 100)
    print(title)
    print("=" * 100)
    hdr = f"{'case_id':26} {'measured_us':>11} {'理论us':>9} {'cp4_us':>9} {'eff':>6}  judge"
    print(hdr)
    print("-" * len(hdr))

    below, lows, oks, missing = [], [], [], []
    for cid, measured, _note in rows:
        r = by_id.get(cid)
        if r is None:
            missing.append(cid)
            print(
                f"{cid:26} {measured:11.3f}      —        —      —   NO_CASE(builder 无此 case_id)"
            )
            continue
        theo = r["theoretical_time_us"]
        cp4 = r["cp4_target_time_us"]
        eff = theo / measured if measured > 0 else 0.0
        verdict = classify(measured, theo, r["target_efficiency"])
        {"BELOW_FLOOR": below, "OK": oks, "LOW_EFF": lows}[verdict].append(cid)
        print(
            f"{cid:26} {measured:11.3f} {theo:9.3f} {cp4:9.3f} {eff:6.2f}  {verdict}  {_VERDICT_NOTE[verdict]}"
        )

    print("-" * len(hdr))
    print(
        f"小结: OK={len(oks)}  BELOW_FLOOR={len(below)}  LOW_EFF={len(lows)}  NO_CASE={len(missing)}"
    )
    if below:
        print(f"  ⚠ 跑穿下界(必修): {', '.join(below)}")
    print()
    return 1 if below else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", type=Path, help="对照一份实测 CSV")
    ap.add_argument("--dir", type=Path, help="对照目录下所有 *.csv（跳过 _ 开头）")
    args = ap.parse_args()

    if args.file:
        files = [args.file]
    elif args.dir:
        files = sorted(p for p in args.dir.glob("*.csv") if not p.name.startswith("_"))
    else:
        ap.error(
            "需 --file <csv> 或 --dir <目录>；实测 CSV 不随 skill 提交，按 collect_onboard.md 在 950 采数后指定路径"
        )
    if not files:
        print(f"目录无实测 CSV：{args.dir}")
        return 0

    rc = 0
    for f in files:
        rc |= compare_file(f)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
