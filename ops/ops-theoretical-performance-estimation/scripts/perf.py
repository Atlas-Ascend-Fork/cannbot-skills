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

"""通用算子性能评估 CLI：单一入口，按 --op 路由到对应 references/examples/<op>_flow.json。

算子无关——建模在共享引擎，算子语义全在 <op>_flow.json（多模版）。本文件不含任何算子专属
代码：新算子只需写 <op>_flow.json + <op>_cases.json，不再新增任何 .py（旧的 per-op
builder 已并入此处，单一代码路径更利于回溯/定位）。

用法:
  python3 scripts/perf.py --op softmax \
      --cases references/examples/softmax_cases.json [--show-case] [--emit-jsonl out.jsonl]
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
bridge = importlib.import_module("_engine_bridge")
flow_runtime = importlib.import_module("flow_runtime")


def build_case_for(op):
    """返回该算子的 build_case 闭包：委托 flow_runtime 按 <op>_flow.json 路由模版+解释。"""

    def build_case(case, attrs, ctx):
        return flow_runtime.build(op, case, attrs, ctx)

    return build_case


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--op", required=True, help="算子名（对应 references/examples/<op>_flow.json）"
    )
    ap.add_argument("--cases", type=Path, required=True, help="性能测试 case json")
    ap.add_argument(
        "--emit-jsonl",
        type=Path,
        help="把推导出的标准 case 写成 JSONL（审计 / 喂引擎 CLI）",
    )
    ap.add_argument(
        "--show-case", action="store_true", help="打印每个推导出的 case dict"
    )
    args = ap.parse_args(argv)
    bridge.run_build(
        build_case_for(args.op),
        args.cases,
        show_case=args.show_case,
        emit_jsonl=args.emit_jsonl,
    )


if __name__ == "__main__":
    main()
