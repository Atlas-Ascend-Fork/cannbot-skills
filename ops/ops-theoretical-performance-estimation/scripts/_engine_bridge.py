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

"""算子 builder 与共享引擎之间的桥：加载引擎/数据、跑 case、打印。

把「load 引擎 + load case json + 逐 case build → evaluate → 打印」的样板抽出来，
每个算子 builder 只需提供一个 build_case(case, attrs, ctx) 函数。引擎本身不被触碰。
evaluate_spec() 不打印、返回结构化结果，供 tests/ 与 onboard/compare.py 复用。

路径相对本文件解析（scripts/ 同级有引擎，../references 有数据），因此与分支检出
位置、当前工作目录无关。
"""

from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
engine = importlib.import_module("evaluate_a5_vector_roofline")
tc = importlib.import_module("perf_testcase")

HARDWARE_JSON = SKILL_DIR / "references/hardware/ascend950pr.json"
INSTRUCTION_JSON = SKILL_DIR / "references/vector_instruction_types.json"

_DBS = None


def load_dbs():
    """加载并缓存 (hardware_db, instruction_db, dtype_bytes)。"""
    global _DBS
    if _DBS is None:
        hardware_db = engine.load_json(HARDWARE_JSON)
        instruction_db = engine.load_json(INSTRUCTION_JSON)
        _DBS = (hardware_db, instruction_db, instruction_db["dtype_bytes"])
    return _DBS


class Ctx:
    """传给 build_case 的上下文：dtype 字节表 + UB 容量 + 通用辅助（透传 perf_testcase）。"""

    def __init__(self, dtype_bytes, ub_bytes):
        self.dtype_bytes = dtype_bytes
        self.ub_bytes = ub_bytes  # 模版路由判 UB 可行性用
        self.elems = tc.elems
        self.tensor_bytes = lambda t: tc.tensor_bytes(t, dtype_bytes)
        self.find_input = tc.find_input
        self.expand_inputs = tc.expand_inputs


def evaluate_spec(build_case, cases_path):
    """加载+校验 case json，逐 case 跑 build_case→引擎。返回结构化结果列表，不打印。

    每项: {"case", "case_dict", "result"}。异常（dtype 非法 / UB 不可行等）原样抛出。
    """
    hardware_db, instruction_db, dtype_bytes = load_dbs()
    ub_bytes = float(hardware_db["common"]["memory_capacity_bytes_per_core"]["ub"])
    ctx = Ctx(dtype_bytes, ub_bytes)
    spec = tc.load_spec(cases_path, dtype_bytes)
    variant = spec.get("hardware_variant")

    out = []
    for case in spec["cases"]:
        attrs = tc.merged_attrs(spec, case)
        cd = build_case(case, attrs, ctx)
        cd["case_id"] = case["case_id"]
        if variant and "hardware_variant" not in cd:
            cd["hardware_variant"] = variant
        tc.apply_tuning(cd, case)
        result = engine.evaluate_case(cd, hardware_db, instruction_db)
        out.append({"case": case, "case_dict": cd, "result": result})
    return spec, out


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


def run_build(build_case, cases_path, show_case=False, emit_jsonl=None):
    """跑一份 cases.json：build→引擎→打印表格。供通用 CLI(perf.py)与日后任何入口复用。

    show_case 打印推导出的标准 case dict（人眼审字节/指令/tiling）；
    emit_jsonl 把中间产物落盘成 JSONL（审计 / 直接喂引擎 CLI 复跑）。表格含路由到的 template 列。
    """
    spec, rows = evaluate_spec(build_case, cases_path)
    variant = spec.get("hardware_variant")
    derived = [r["case_dict"] for r in rows]
    results = [r["result"] for r in rows]

    if emit_jsonl:
        Path(emit_jsonl).write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in derived) + "\n",
            encoding="utf-8",
        )

    if show_case:
        print("=" * 78)
        print("推导出的标准 case dict（人眼审：字节数 / 指令流 / tiling 是否正确）")
        print("=" * 78)
        for c in derived:
            print(json.dumps(c, ensure_ascii=False))
        print()

    print("=" * 96)
    print(
        f"算子: {spec['operator']}    variant: {variant}    引擎: evaluate_a5_vector_roofline (共享/标定)"
    )
    print("=" * 96)
    hdr = (
        f"{'case_id':26} {'template':20} {'bound':12} {'mte2_us':>9} {'vec_us':>9} {'mte3_us':>9} "
        f"{'tiles':>9} {'理论us':>9} {'CP4目标us':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        r = row["result"]
        tmpl = str(row["case_dict"].get("_template", "-"))
        print(
            f"{r['case_id']:26} {tmpl:20} {r['bound']:12} "
            f"{_fmt(r['mte2_time_us']):>9} {_fmt(r['vector_time_us']):>9} {_fmt(r['mte3_time_us']):>9} "
            f"{_fmt(r['tile_count']):>9} {_fmt(r['theoretical_time_us']):>9} {_fmt(r['cp4_target_time_us']):>10}"
        )
    print()
    print(
        "提示：把同一份 cases.json 喂给 TTK/profiling 跑上板，实测 Task Duration 与 CP4目标us 对照判断达标。"
    )
    return results
