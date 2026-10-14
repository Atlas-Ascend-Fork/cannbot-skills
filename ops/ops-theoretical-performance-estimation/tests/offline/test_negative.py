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

"""离线 ST：负向 / 守卫测试 —— 断言各「门禁确实会咬」。

正向三关只验「正常输入能过」；本测试反过来喂坏输入，断言对应层**确实报错**
（或路由正确）。防止某次改动让某个门禁变成永远 PASS（失去拦截力）而无人发现。

覆盖：perf_testcase 校验 / 引擎强校验 / flow_runtime 模版路由（含 attr 谓词、错误路径）。
用法: python3 tests/offline/test_negative.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2]
SCRIPTS = SKILL_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS))
engine = importlib.import_module("evaluate_a5_vector_roofline")
tc = importlib.import_module("perf_testcase")
fr = importlib.import_module("flow_runtime")
bridge = importlib.import_module("_engine_bridge")

HW, IDB, DTB = bridge.load_dbs()
CTX = bridge.Ctx(DTB, float(HW["common"]["memory_capacity_bytes_per_core"]["ub"]))

fails = []


def expect_raise(label, exc, fn):
    try:
        fn()
        fails.append(f"[应报错未报错] {label}")
    except exc:
        pass
    except Exception as e:  # 报了，但类型不对
        fails.append(
            f"[报错类型不符] {label}: 期望 {exc.__name__} 得 {type(e).__name__}: {e}"
        )


def expect_ok(label, fn, check=None):
    try:
        r = fn()
        if check and not check(r):
            fails.append(f"[结果不符] {label}")
    except Exception as e:
        fails.append(f"[不应报错却报错] {label}: {type(e).__name__}: {e}")


# ---------- 1) perf_testcase 输入校验 ----------
def _spec(case):
    return {"operator": "o", "cases": [case]}


expect_raise(
    "未知 dtype",
    tc.TestcaseError,
    lambda: tc.validate(
        _spec({"case_id": "c", "inputs": [{"shape": [4], "dtype": "fp99"}]}), DTB
    ),
)
expect_raise(
    "符号维 shape",
    tc.TestcaseError,
    lambda: tc.validate(
        _spec({"case_id": "c", "inputs": [{"shape": ["N"], "dtype": "fp32"}]}), DTB
    ),
)
expect_raise(
    "非白名单 tuning（防改标定）",
    tc.TestcaseError,
    lambda: tc.validate(
        _spec(
            {
                "case_id": "c",
                "inputs": [{"shape": [4], "dtype": "fp32"}],
                "tuning": {"issue_slots": 9},
            }
        ),
        DTB,
    ),
)
expect_raise(
    "case_id 重复",
    tc.TestcaseError,
    lambda: tc.validate(
        {
            "operator": "o",
            "cases": [
                {"case_id": "d", "inputs": [{"shape": [4], "dtype": "fp32"}]},
                {"case_id": "d", "inputs": [{"shape": [4], "dtype": "fp32"}]},
            ],
        },
        DTB,
    ),
)
expect_raise(
    "未知顶层字段",
    tc.TestcaseError,
    lambda: tc.validate(
        {
            "operator": "o",
            "junk": 1,
            "cases": [{"case_id": "c", "inputs": [{"shape": [4], "dtype": "fp32"}]}],
        },
        DTB,
    ),
)


# ---------- 2) 引擎强校验 ----------
def _ev(cd):
    cd.setdefault("case_id", "c")
    cd.setdefault("memory", {})
    return engine.evaluate_case(cd, HW, IDB)


expect_raise(
    "未知 RegBase 指令",
    engine.EvalError,
    lambda: _ev(
        {"vector": {"instructions": [{"op": "Foo", "dtype": "fp32", "elements": 64}]}}
    ),
)
expect_raise(
    "不支持的 op/dtype: ReduceSum bf16",
    engine.EvalError,
    lambda: _ev(
        {
            "vector": {
                "instructions": [{"op": "ReduceSum", "dtype": "bf16", "elements": 64}]
            }
        }
    ),
)
expect_raise(
    "不支持的 Cast 转换对",
    engine.EvalError,
    lambda: _ev(
        {
            "vector": {
                "instructions": [
                    {
                        "op": "Cast",
                        "src_dtype": "fp32",
                        "dst_dtype": "bool",
                        "elements": 64,
                    }
                ]
            }
        }
    ),
)
expect_raise(
    "elements 缺失",
    engine.EvalError,
    lambda: _ev({"vector": {"instructions": [{"op": "Add", "dtype": "fp32"}]}}),
)
expect_raise(
    "UB 不可行(一单元超 UB)",
    engine.EvalError,
    lambda: _ev(
        {
            "tiling": {
                "total_tile_units": 1,
                "ub_bytes_per_unit": 1000000000,
                "double_buffer": 2,
            },
            "vector": {
                "instructions": [{"op": "Add", "dtype": "fp32", "elements": 64}]
            },
        }
    ),
)


# 未提供的自动修正必须报错，包括原先会忽略爬坡字段的 L2 带宽分支。
_VALID_VECTOR = {"instructions": [{"op": "Add", "dtype": "fp32", "elements": 64}]}
for bandwidth_model in (
    "chip_peak",
    "vector_core_conservative",
    "l2_aware",
    "l2_capacity",
):
    expect_raise(
        f"不支持预置爬坡/{bandwidth_model}",
        engine.EvalError,
        lambda model=bandwidth_model: _ev(
            {
                "gm_bandwidth_model": model,
                "gm_rampup_model": "hbm_rampup",
                "vector": _VALID_VECTOR,
            }
        ),
    )
for explicit in ({}, {"fixed_overhead_us": 2.5}):
    expect_raise(
        f"不支持预置固定开销/{explicit}",
        engine.EvalError,
        lambda values=explicit: _ev(
            {
                "fixed_overhead_model": "hardware_default",
                "vector": _VALID_VECTOR,
                **values,
            }
        ),
    )


# ---------- 3) flow_runtime 模版路由 ----------
# attr 驱动的两模版：mode=fast / mode=safe
ATTR_FLOW = {
    "templates": [
        {
            "name": "fast",
            "select": [{"attr": "mode", "eq": "fast"}],
            "tiling": {
                "total_tile_units": "n",
                "ub_bytes_per_unit": "db",
                "double_buffer": 2,
            },
            "flow": [
                {"copy_in": "a"},
                {"vf": "v", "ops": [{"op": "Add", "on": "full"}]},
                {"copy_out": "y", "like": "a"},
            ],
        },
        {
            "name": "safe",
            "select": [{"attr": "mode", "eq": "safe"}],
            "tiling": {
                "total_tile_units": "n",
                "ub_bytes_per_unit": "db",
                "double_buffer": 2,
            },
            "flow": [
                {"copy_in": "a"},
                {"vf": "v", "ops": [{"op": "Mul", "on": "full"}]},
                {"copy_out": "y", "like": "a"},
            ],
        },
    ]
}


def _route(case, flow=ATTR_FLOW):
    attrs = tc.merged_attrs({}, case)
    v = fr.make_vars(flow, case, attrs, CTX)
    return fr.select_template(flow, case, attrs, CTX, v)


_in = [{"name": "a", "shape": [64], "dtype": "fp32"}]
expect_ok(
    "attr 路由: mode=fast → fast 模版",
    lambda: _route({"case_id": "c", "attrs": {"mode": "fast"}, "inputs": _in}),
    lambda t: t["name"] == "fast",
)
expect_ok(
    "attr 路由: mode=safe → safe 模版",
    lambda: _route({"case_id": "c", "attrs": {"mode": "safe"}, "inputs": _in}),
    lambda t: t["name"] == "safe",
)
expect_raise(
    "无命中模版(attr 都不满足)",
    fr.FlowError,
    lambda: _route({"case_id": "c", "attrs": {"mode": "other"}, "inputs": _in}),
)
expect_raise(
    "强制不存在的模版",
    fr.FlowError,
    lambda: _route(
        {"case_id": "c", "template": "nope", "attrs": {"mode": "fast"}, "inputs": _in}
    ),
)

# ---------- 4) flow_runtime 解释期强校验（缺必需 copy_in / 非法 on） ----------
# rms_norm 必需 weight；只给 x → copy_in weight 缺失必须报错（否则静默漏算 gm_read 还照常 emit 计算）。
expect_raise(
    "缺必需输入 weight(copy_in 找不到)",
    fr.FlowError,
    lambda: fr.build(
        "rms_norm",
        {
            "case_id": "c",
            "inputs": [{"name": "x", "shape": [128, 4096], "dtype": "fp32"}],
        },
        {"axis": -1, "epsilon": 1e-6},
        CTX,
    ),
)
# vf op 的 on 只能 full|reduced；typo 必须报错（否则静默按 reduced 改元素数）。
_BAD_ON_FLOW = {
    "operator": "t",
    "attrs": [],
    "templates": [
        {
            "name": "t",
            "select": [],
            "tiling": {
                "core_split_units": "n",
                "total_tile_units": "n",
                "ub_bytes_per_unit": "db*2",
                "double_buffer": 2,
            },
            "flow": [
                {"copy_in": "a"},
                {"vf": "v", "ops": [{"op": "Add", "on": "REDUCE"}]},
                {"copy_out": "y", "like": "a"},
            ],
        }
    ],
}


def _bad_on():
    case = {"case_id": "c", "inputs": [{"name": "a", "shape": [64], "dtype": "fp32"}]}
    attrs = tc.merged_attrs({}, case)
    v = fr.make_vars(_BAD_ON_FLOW, case, attrs, CTX)
    return fr.interpret(_BAD_ON_FLOW["templates"][0], _BAD_ON_FLOW, case, attrs, CTX, v)


expect_raise("非法 on 值(REDUCE 非 full|reduced)", fr.FlowError, _bad_on)


def main():
    n = 22  # 包含循环展开的 expect_* 总数
    if fails:
        print(f"FAIL  ({len(fails)} 处 / {n} 项负向用例)")
        for f in fails:
            print("  -", f)
        return 1
    print(f"PASS  ({n} 项负向用例：校验/强校验/路由 门禁均按预期拦截)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
