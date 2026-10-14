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

"""离线 ST：设计符合性 —— 验证「算子计算流 → 评估脚本」翻译对不对（多模版版）。

计算流在 references/examples/<op>_flow.json：算子 = 多模版，每个模版一条完整 flow
（copy_in / op / copy_out）。builder 经 flow_runtime 按 case 路由到某模版并解释。

测试对每个 case：跑 builder 得 case dict（含所选模版名 `_template`），再**独立**按该模版的
flow 展开期望（不调 flow_runtime），逐项比对：
  1. 指令序列（含低精度首尾 Cast）== 模版 op 步骤
  2. 每步元素数 full=N / reduced=行数
  3. gm_read = Σ copy_in 张量字节（同名出现两次=重读两次）、gm_write = Σ copy_out 字节
独立展开(测试) vs 解释器(flow_runtime) 互为实现 → 抓 flow 翻译错误 + 防漂移。缺 flow.json 判失败。

用法: python3 tests/offline/test_design_conformance.py
"""

from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2]
SCRIPTS = SKILL_DIR / "scripts"
EXAMPLES = SKILL_DIR / "references/examples"
sys.path.insert(0, str(SCRIPTS))
bridge = importlib.import_module("_engine_bridge")
tc = importlib.import_module("perf_testcase")
flow_runtime = importlib.import_module("flow_runtime")


def load_builder(name):
    """通用解释器入口（无 per-op .py）：被测对象就是 flow_runtime 对 <name>_flow.json 的解释。"""
    return lambda case, attrs, ctx: flow_runtime.build(name, case, attrs, ctx)


def find_template(flow, tname):
    return next((t for t in flow["templates"] if t["name"] == tname), None)


def expand_template(template, flow, case, attrs, dtype_bytes, cols):
    """独立展开选中模版（VF 结构）：→ (ops, ons, gm_read, gm_write, ubpu_auto)。不调 flow_runtime。

    ubpu_auto = 各 VF 的 UB footprint 取 max（无显式 tiling 时与 builder 对照）。
    """
    main = case["inputs"][0]
    dtype = main["dtype"]
    lowp = dtype != "fp32" and flow.get("compute_dtype") == "fp32_for_lowp"
    present = {i.get("name") for i in case["inputs"]}
    outs = case.get("outputs") or []

    def wok(w):
        if w == "lowp":
            return lowp
        if w and w.startswith("has:"):
            return w[4:] in present
        return True

    def ub_size(b):
        dt = b.get("dtype", "@in")
        dt = dtype if dt == "@in" else dt
        return (
            cols * dtype_bytes[dt]
            if b.get("extent", "row") == "row"
            else dtype_bytes[dt]
        )

    ops, ons, gm_read, gm_write, footprints = [], [], 0.0, 0.0, []
    for s in template["flow"]:
        if not wok(s.get("when")):
            continue
        if "copy_in" in s:
            nm = s["copy_in"]
            if nm == "*":
                gm_read += sum(
                    tc.tensor_bytes(x, dtype_bytes) for x in tc.expand_inputs(case)
                )
            else:
                t = tc.find_input(case, nm)
                if t is not None:
                    gm_read += tc.tensor_bytes(t, dtype_bytes)
        elif "copy_out" in s:
            out = next(
                (output for output in outs if output.get("name") == s["copy_out"]), None
            )
            if out is not None:
                gm_write += tc.tensor_bytes(out, dtype_bytes)
            else:
                gm_write += tc.tensor_bytes(
                    tc.find_input(case, s.get("like")) or main, dtype_bytes
                )
        elif "vf" in s:
            footprints.append(
                sum(
                    ub_size(b) * (2 if b.get("db", True) else 1)
                    for b in s.get("live", [])
                    if wok(b.get("when"))
                )
            )
            for op_s in s.get("ops", []):
                if wok(op_s.get("when")):
                    ops.append(op_s["op"])
                    ons.append(op_s.get("on", "full"))
    ubpu_auto = max(footprints) if footprints else None
    return ops, ons, gm_read, gm_write, ubpu_auto


def rows_cols(flow, case, attrs):
    main = case["inputs"][0]
    n = tc.elems(main["shape"])
    axis_attr = flow.get("reduce_axis_attr")
    if not axis_attr:
        return n, None
    axis = int(attrs.get(axis_attr, -1)) % len(main["shape"])
    return n, n // int(main["shape"][axis])


def main():
    _, _, dtype_bytes = bridge.load_dbs()
    case_names = {p.name[: -len("_cases.json")] for p in EXAMPLES.glob("*_cases.json")}
    flow_names = {p.name[: -len("_flow.json")] for p in EXAMPLES.glob("*_flow.json")}

    fails, n_cases = [], 0
    missing = sorted(case_names - flow_names)
    if missing:
        fails.append(f"缺计算流定义 <op>_flow.json: {missing}")

    for name in sorted(case_names & flow_names):
        flow = json.loads((EXAMPLES / f"{name}_flow.json").read_text(encoding="utf-8"))
        try:
            spec, rows = bridge.evaluate_spec(
                load_builder(name), EXAMPLES / f"{name}_cases.json"
            )
        except Exception as exc:
            fails.append(f"[{name}] 跑挂: {type(exc).__name__}: {exc}")
            continue

        for row in rows:
            n_cases += 1
            cid, cd, case = row["case"]["case_id"], row["case_dict"], row["case"]
            attrs = tc.merged_attrs(spec, case)
            tname = cd.get("_template")
            template = find_template(flow, tname)
            if template is None:
                fails.append(f"[{name}/{cid}] 选中模版 {tname!r} 不在 flow.templates")
                continue

            n, r = rows_cols(flow, case, attrs)
            cols = n // r if r else n
            exp_ops, exp_ons, exp_read, exp_write, exp_ubpu = expand_template(
                template, flow, case, attrs, dtype_bytes, cols
            )
            instrs = cd["vector"]["instructions"]
            act_ops = [i["op"] for i in instrs]
            if act_ops != exp_ops:
                fails.append(
                    f"[{name}/{cid}|{tname}] 指令流不符\n      期望 {exp_ops}\n      实得 {act_ops}"
                )
            else:
                for op, on, inst in zip(exp_ops, exp_ons, instrs):
                    want = n if on == "full" else r
                    got = float(inst.get("elements"))
                    if want is None or abs(got - want) > 0.5:
                        fails.append(
                            f"[{name}/{cid}|{tname}] {op} elements 不符: builder={got:.0f} 期望({on})={want}"
                        )

            for key, exp in (
                ("gm_read_bytes", exp_read),
                ("gm_write_bytes", exp_write),
            ):
                got = float(cd.get("memory", {}).get(key, 0))
                if abs(got - exp) > 0.5:
                    fails.append(
                        f"[{name}/{cid}|{tname}] {key} 不符: builder={got:.0f} 期望={exp:.0f}"
                    )

            # 无显式 tiling 的模版：核对自动 footprint 与 builder 一致
            if not template.get("tiling") and exp_ubpu is not None:
                got = float(cd.get("tiling", {}).get("ub_bytes_per_unit", 0))
                if abs(got - exp_ubpu) > 0.5:
                    fails.append(
                        f"[{name}/{cid}|{tname}] ub_bytes_per_unit(自动) 不符: "
                        f"builder={got:.0f} 期望={exp_ubpu:.0f}"
                    )

    ops = sorted(case_names & flow_names)
    if fails:
        print(f"FAIL  ({len(fails)} 处, {n_cases} cases, 算子: {', '.join(ops)})")
        for f in fails:
            print("  -", f)
        return 1
    print(f"PASS  ({n_cases} cases / {len(ops)} 算子符合 flow: {', '.join(ops)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
