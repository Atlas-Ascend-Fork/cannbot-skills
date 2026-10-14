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

"""计算流解释器：把一个算子的 <op>_flow.json（多模版）按 case 路由 + 解释成引擎 case dict。

算子 = 多个模版（templates），每个模版是一条**完整无条件**的计算流（copy_in / op / copy_out）。
- copy_in <张量>  → MTE2，字节计入 gm_read（同一张量出现两次 = 重读两次，如非全载多 pass）
- op <指令> on=full|reduced → Vector，elements = n(全量) 或 rows(逐行)
- copy_out <张量> → MTE3，字节计入 gm_write

路由：case 可用 "template":"<name>" 强制；否则按模版顺序，选第一个「select 谓词命中 且 tiling UB 可行」的。
全载/非全载因此自动落出：全载模版列前面，放不下 UB 时 fall 到行切分模版（其 flow 里显式重读）。
谓词：{"attr":"<名>","eq":<值>} / {"attr":"<名>","in":[...]} / {"dtype":[...]}；select=[] 表示无条件候选。

低精度策略 compute_dtype="fp32_for_lowp"：低精度输入时在 op 序列首尾自动补 Cast，中间按 fp32 算。
"""

from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
SKILL_DIR = SCRIPTS.parent
EXAMPLES = SKILL_DIR / "references/examples"
sys.path.insert(0, str(SCRIPTS))
engine = importlib.import_module("evaluate_a5_vector_roofline")

_FLOW_CACHE: dict[str, dict] = {}


class FlowError(ValueError):
    pass


def load_flow(name: str) -> dict:
    if name not in _FLOW_CACHE:
        _FLOW_CACHE[name] = json.loads(
            (EXAMPLES / f"{name}_flow.json").read_text(encoding="utf-8")
        )
    return _FLOW_CACHE[name]


def make_vars(flow, case, attrs, ctx):
    main = case["inputs"][0]
    n = ctx.elems(main["shape"])
    axis_attr = flow.get("reduce_axis_attr")
    if axis_attr:
        axis = int(attrs.get(axis_attr, -1)) % len(main["shape"])
        cols = int(main["shape"][axis])
        rows = n // cols
    else:
        cols, rows = n, 1
    v = {
        "n": n,
        "rows": rows,
        "cols": cols,
        "H": cols,
        "db": ctx.dtype_bytes[main["dtype"]],
        "ub": ctx.ub_bytes,
        "nin": len(ctx.expand_inputs(case)),
    }
    # 数值型 attr 也可在表达式(如 op count="K")里引用；不覆盖上面的内置变量。
    for k, val in attrs.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool) and k not in v:
            v[k] = val
    return v


def _pred_ok(pred, attrs, main):
    if "attr" in pred:
        v = attrs.get(pred["attr"])
        if "eq" in pred:
            return v == pred["eq"]
        if "in" in pred:
            return v in pred["in"]
        return False
    if "dtype" in pred:
        return main["dtype"] in pred["dtype"]
    return False


def _footprint(template, flow, case, ctx, variables):
    """模版的 UB footprint = 各 VF 的 live 张量字节取 max（无 VF 则 None）。

    逐 buffer 双缓冲：每个 live 张量 `db`（默认 true）决定该 buffer 计 ×2 还是 ×1。
    一般只对流式进出的 buffer(x/y) 开 DB；加载一次的(gamma/beta/weight)与临时量标 db:false。
    DB 已计入 footprint，所以自动 tiling 的 double_buffer 取 1。
    """
    main = case["inputs"][0]
    dtype = main["dtype"]
    lowp = dtype != "fp32" and flow.get("compute_dtype") == "fp32_for_lowp"
    present = {i.get("name") for i in case["inputs"]}
    fps = []
    for s in template["flow"]:
        if "vf" not in s or not _when_ok(s.get("when"), present, lowp):
            continue
        fps.append(
            sum(
                _ub_size(b, variables, ctx, dtype) * (2 if b.get("db", True) else 1)
                for b in s.get("live", [])
                if _when_ok(b.get("when"), present, lowp)
            )
        )
    return max(fps) if fps else None


def _feasible(template, flow, case, ctx, variables):
    """tiling 显式 → 按其 ub_bytes_per_unit×double_buffer；否则按 VF footprint(已含逐 buffer DB)。≤ UB。"""
    til = template.get("tiling")
    if til:
        bpu = engine.safe_eval(til["ub_bytes_per_unit"], variables)
        dbf = engine.safe_eval(til.get("double_buffer", 2), variables)
    else:
        bpu = _footprint(template, flow, case, ctx, variables)
        if bpu is None:
            return True
        dbf = 1
    return bpu * dbf <= variables["ub"]


def select_template(flow, case, attrs, ctx, variables):
    main = case["inputs"][0]
    forced = case.get("template")
    for t in flow["templates"]:
        if forced is not None:
            if t["name"] == forced:
                return t
            continue
        if all(_pred_ok(p, attrs, main) for p in t.get("select", [])) and _feasible(
            t, flow, case, ctx, variables
        ):
            return t
    if forced is not None:
        raise FlowError(f"{case.get('case_id')}: forced template {forced!r} not found")
    raise FlowError(
        f"{case.get('case_id')}: no eligible/feasible template (check UB feasibility / select 谓词)"
    )


def _when_ok(w, present, lowp):
    """步骤条件：has:<输入名>（输入存在）或 lowp（主输入为低精度）。"""
    if not w:
        return True
    if w.startswith("has:"):
        return w[4:] in present
    if w == "lowp":
        return lowp
    raise FlowError(f"unknown step when: {w!r}")


def _ub_size(b, variables, ctx, main_dtype):
    """一个 UB 存活张量的 per-tile-unit 字节：row=cols*dtype，scalar≈1 elem。"""
    dt = b.get("dtype", "@in")
    dt = main_dtype if dt == "@in" else dt
    db = ctx.dtype_bytes[dt]
    return variables["cols"] * db if b.get("extent", "row") == "row" else db


def interpret(template, flow, case, attrs, ctx, variables):
    main = case["inputs"][0]
    dtype = main["dtype"]
    # lowp 仅决定 op 步骤的计算精度(cdt)；Cast 是 flow 里显式 when:"lowp" 步骤。
    lowp = dtype != "fp32" and flow.get("compute_dtype") == "fp32_for_lowp"
    cdt = "fp32" if lowp else dtype
    present = {i.get("name") for i in case["inputs"]}
    outs = case.get("outputs") or []

    def emit_op(s, instrs):
        on = s.get("on", "full")
        if on not in (
            "full",
            "reduced",
        ):  # 否则 typo(如 "Full"/"reduce") 静默按 reduced 改元素数
            raise FlowError(f"op {s.get('op')!r} 的 on={on!r} 非法，须 full|reduced")
        inst = {
            "op": s["op"],
            "elements": variables["n"] if on == "full" else variables["rows"],
        }
        cnt = engine.safe_eval(s.get("count", 1), variables)
        if cnt != 1:
            inst["count"] = cnt
        if s["op"] == "Cast":
            # @in = 主输入 dtype；无 src/dst 视为 cast 算子本身（src=主输入, dst=输出 dtype）。
            src, dst = s.get("src_dtype"), s.get("dst_dtype")
            inst["src_dtype"] = dtype if src in (None, "@in") else src
            if dst is None:
                inst["dst_dtype"] = outs[0]["dtype"] if outs else attrs.get("to")
            else:
                inst["dst_dtype"] = dtype if dst == "@in" else dst
        else:
            inst["dtype"] = cdt
        instrs.append(inst)

    instrs, gm_read, gm_write, gm_reread = [], 0.0, 0.0, 0.0
    seen_in: set = set()  # 已 copy_in 的张量名：再次出现=重读(L2 复用)
    for s in template["flow"]:
        if not _when_ok(s.get("when"), present, lowp):
            continue
        if "copy_in" in s:  # GM→UB
            nm = s["copy_in"]
            if nm == "*":
                for x in ctx.expand_inputs(case):
                    b = ctx.tensor_bytes(x)
                    gm_read += b
                    xn = x.get("name")
                    if xn in seen_in:
                        gm_reread += b
                    else:
                        seen_in.add(xn)
            else:
                t = ctx.find_input(case, nm)
                if (
                    t is None
                ):  # when 已命中却缺必需输入 → 否则静默漏算 gm_read 还照常 emit 计算
                    raise FlowError(
                        f'copy_in 张量 {nm!r} 不在 case inputs；必需输入缺失（可选输入须用 when:"has:{nm}" 守卫）'
                    )
                b = ctx.tensor_bytes(t)
                gm_read += b
                if nm in seen_in:  # 同名第二次 copy_in = 非全载重读 x
                    gm_reread += b
                else:
                    seen_in.add(nm)
        elif "copy_out" in s:  # UB→GM
            out = next(
                (output for output in outs if output.get("name") == s["copy_out"]), None
            )
            gm_write += (
                ctx.tensor_bytes(out)
                if out is not None
                else ctx.tensor_bytes(ctx.find_input(case, s.get("like")) or main)
            )
        elif "vf" in s:  # 一段寄存器驻留的矢量计算
            for op_s in s.get("ops", []):
                if _when_ok(op_s.get("when"), present, lowp):
                    emit_op(op_s, instrs)
        else:
            raise FlowError(f"unknown flow step: {s}")

    # tiling：显式覆盖（如行切分 split 模版）优先；否则由 VF 的 UB footprint 自动推。
    # core_split_units = 分核单位数（核间均衡用，elementwise=n / row 类=rows），与
    # total_tile_units（核内 UB 切分）解耦；显式块转发其声明值，自动路径取 total（即分核单位本身）。
    if template.get("tiling"):
        til = template["tiling"]
        csu_expr = til.get("core_split_units")
        tiling = {
            "core_split_units": engine.safe_eval(csu_expr, variables)
            if csu_expr is not None
            else None,
            "total_tile_units": engine.safe_eval(til["total_tile_units"], variables),
            "ub_bytes_per_unit": engine.safe_eval(til["ub_bytes_per_unit"], variables),
            "double_buffer": engine.safe_eval(til.get("double_buffer", 2), variables),
        }
    else:
        total = variables["rows"] if flow.get("reduce_axis_attr") else variables["n"]
        ubpu = _footprint(template, flow, case, ctx, variables)
        if ubpu is None:
            ubpu = (variables["nin"] + 1) * variables["db"]
        # 逐 buffer DB 已计入 ubpu，故 double_buffer=1（不再统一 ×2）。
        # total 即分核单位（rows 或 n）→ 直接作 core_split_units。
        tiling = {
            "core_split_units": total,
            "total_tile_units": total,
            "ub_bytes_per_unit": ubpu,
            "double_buffer": 1,
        }

    mem = {
        "gm_read_bytes": gm_read,
        "gm_write_bytes": gm_write,
        "gm_reread_bytes": gm_reread,
    }
    # L2 容量模型(gm_bandwidth_model=l2_capacity)用:pass 内同时占 L2 的大流数(reread 读 + 写出并存挤 L2),
    # 模版可标 l2_concurrent_streams（行级双 pass 重读+写 = 2）→ 引擎 C_eff = C/份数。默认 1(独占)。
    cs = template.get("l2_concurrent_streams")
    if cs is not None:
        mem["l2_concurrent_streams"] = engine.safe_eval(cs, variables)

    out_case = {
        "gm_rw_contention": flow.get("gm_rw_contention", True),
        "launch_floor_model": flow.get("launch_floor_model", "none"),
        "tiling": tiling,
        "memory": mem,
        "vector": {"instructions": instrs},
        "_template": template["name"],
    }
    # gm_bandwidth_model 可模版级覆盖 flow 级:resident 流式(1R:1W,读写争用)用 l2_aware 曲线;
    # 行级双 pass(reread 受 L2 容量限)用 l2_capacity。不同模版访存模式不同,带宽模型也不同。
    gbm = template.get("gm_bandwidth_model") or flow.get("gm_bandwidth_model")
    if gbm:
        out_case["gm_bandwidth_model"] = (
            gbm  # l2_aware 时，引擎用标定曲线，忽略上面的 contention
        )
    return out_case


def build(name, case, attrs, ctx):
    """算子 builder 的统一实现：加载 <name>_flow.json，路由到模版，解释成 case dict。"""
    flow = load_flow(name)
    variables = make_vars(flow, case, attrs, ctx)
    template = select_template(flow, case, attrs, ctx, variables)
    return interpret(template, flow, case, attrs, ctx, variables)
