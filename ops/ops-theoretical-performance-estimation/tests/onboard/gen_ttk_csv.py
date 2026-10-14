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

"""从 <op>_cases.json 生成 TTK kernel 测试 CSV（上板实测用），case 名与 builder 一致。

本脚本只做 cases.json → TTK kernel CSV 的转换；生成 CSV 无需 NPU。
采集时在当前 Ascend950 环境按 collect_onboard.md 的 TTK 流程执行。CSV 列布局对齐 ops-test-kit
examples/case_store/kernel/*.csv。

每个算子一个 adapter（op_name + 如何由 case 的 inputs/outputs/attrs 拼 TTK 行）。
新算子先确认其 kernel 在 950 可被 `ttk kernel -b=release` 测到，再加 adapter。

用法: python3 gen_ttk_csv.py --op add --cases ../../references/examples/add_cases.json --out add_ttk.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADER = (
    "testcase_name,network_name,op_name,input_shapes,input_dtypes,input_formats,"
    "output_shapes,output_dtypes,output_formats,input_ori_shapes,input_ori_formats,"
    "output_ori_shapes,output_ori_formats,attributes,input_data_ranges,precision_tolerances,"
    "absolute_precision,output_inplace_indexes,output_shape_unknown_indexes,is_enabled,"
    "remark,soc_series,priority,dump_file_prefix,manual_input_binaries,manual_golden_binaries"
)

# TTK 用 Python-literal dtype 名
_DT = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16", "int32": "int32"}


def _tup(shapes):
    inner = ", ".join(
        "(" + ", ".join(str(d) for d in s) + ("," if len(s) == 1 else "") + ")"
        for s in shapes
    )
    return "(" + inner + ("," if len(shapes) == 1 else "") + ")"  # 单元素外层也要尾逗号


def _row(name, op_name, in_shapes, in_dtype, out_shapes, rng):
    dt = f"('{in_dtype}',)"
    ins, outs = _tup(in_shapes), _tup(out_shapes)
    return ",".join(
        [
            name,
            "UNKNOWN",
            op_name,
            f'"{ins}"',
            f'"{dt}"',
            "\"('ND',)\"",
            f'"{outs}"',
            f'"{dt}"',
            "\"('ND',)\"",
            f'"{ins}"',
            "\"('ND',)\"",
            f'"{outs}"',
            "\"('ND',)\"",
            "{}",
            f'"{rng}"',
            "",
            "1e-08",
            "()",
            "()",
            "True",
            "",
            "",
            "0",
            "",
            "()",
            "()",
        ]
    )


# ---- adapters: op → 把一个 case 转成若干 TTK 行（返回 [csv_row, ...]）----
def _adapter_add(case, dtype_bytes=None):
    """Elementwise Add：两个等形输入；仅处理 2 路 fp32/fp16（addn 多路跳过）。"""
    if any(t.get("repeat", 1) > 1 for t in case["inputs"]) or len(case["inputs"]) > 2:
        return []  # 动态多路 AddN 不是标准 2 输入 add，跳过
    x = case["inputs"][0]
    dt = _DT.get(x["dtype"])
    if dt is None:
        return []
    shp = tuple(x["shape"])
    return [_row(case["case_id"], "add", [shp, shp], dt, [shp], "((-1, 1), (-1, 1))")]


ADAPTERS = {
    "add": _adapter_add,
    # TODO softmax/layer_norm/rms_norm：需先确认其 kernel 在 950 可 ttk 测，
    #   再补 adapter（op_name 如 SoftmaxV2/LayerNorm/RmsNorm，含 axis/epsilon attr 与 gamma/beta 多输入）。
    #   更优做法见 collect_onboard.md：为 ST 自写 <<<>>> 直调 kernel，实测值直接对应 flow。
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", required=True, help="算子名（需在 ADAPTERS 内）")
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.op not in ADAPTERS:
        raise SystemExit(
            f"无 adapter: {args.op}（已支持: {sorted(ADAPTERS)}）。"
            f" 先确认其 kernel 在 950 可 ttk 测，再加 adapter。"
        )
    spec = json.loads(args.cases.read_text(encoding="utf-8"))
    rows = []
    for case in spec["cases"]:
        if "template" in case:  # 强制模版的对照 case 不上板
            continue
        rows += ADAPTERS[args.op](case)
    if not rows:
        raise SystemExit(
            f"{args.op}: 无可生成的 TTK 行（dtype/路数不被该 adapter 支持）"
        )
    out = HEADER + "\n" + "\n".join(rows) + "\n"
    if args.out:
        args.out.write_text(out, encoding="utf-8")
        print(f"written {args.out} ({len(rows)} cases)")
    else:
        print(out)


if __name__ == "__main__":
    raise SystemExit(main())
