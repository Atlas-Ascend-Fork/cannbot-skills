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

"""性能测试 case json 的加载与校验（零依赖）+ 算子 builder 通用辅助。

这是两层架构里「前端契约」的共享部分：所有算子专属 builder 都用它来
加载/校验输入 json、展开动态输入、算字节数，避免每个 builder 各写一遍。
本文件的 validate() 即 case json 的唯一真契约（零依赖手写校验）。引擎(evaluate_a5_vector_roofline)不被触碰。
"""

from __future__ import annotations

import json
from pathlib import Path

# 与引擎有语义的旋钮一致；builder 注入 tuning 时只允许这些 key
TUNING_WHITELIST = {
    "target_efficiency",
    "gm_bandwidth_model",
    "gm_rw_contention",
    "gm_rampup_model",
    "overlap_model",
    "fill_drain",
    "fixed_overhead_us",
    "fixed_overhead_model",
    "active_vector_cores",
}


class TestcaseError(ValueError):
    pass


def _require(cond, msg):
    if not cond:
        raise TestcaseError(msg)


def _check_tensor(t, where, dtype_bytes):
    _require(isinstance(t, dict), f"{where}: tensor 必须是对象")
    extra = set(t) - {"name", "shape", "dtype", "optional", "repeat", "comment"}
    _require(not extra, f"{where}: 未知字段 {sorted(extra)}")
    _require("shape" in t and "dtype" in t, f"{where}: 缺 shape/dtype")
    shape = t["shape"]
    _require(
        isinstance(shape, list)
        and shape
        and all(isinstance(d, int) and d >= 1 for d in shape),
        f"{where}: shape 必须是正整数数组（性能评估需具体 shape，不支持符号维）",
    )
    _require(
        t["dtype"] in dtype_bytes,
        f"{where}: 未知 dtype {t['dtype']!r}（须在指令库 dtype_bytes 内）",
    )
    rep = t.get("repeat", 1)
    _require(isinstance(rep, int) and rep >= 1, f"{where}: repeat 必须是 >=1 的整数")


def validate(spec, dtype_bytes):
    """对照 schema 做结构与语义校验（手写，零依赖）。"""
    _require(isinstance(spec, dict), "顶层必须是对象")
    extra = set(spec) - {
        "operator",
        "hardware_variant",
        "comment",
        "default_attrs",
        "cases",
    }
    _require(not extra, f"顶层未知字段 {sorted(extra)}")
    _require(isinstance(spec.get("operator"), str) and spec["operator"], "缺 operator")
    _require(
        "default_attrs" not in spec or isinstance(spec["default_attrs"], dict),
        "default_attrs 必须是对象",
    )
    cases = spec.get("cases")
    _require(isinstance(cases, list) and cases, "cases 必须是非空数组")

    seen = set()
    for i, case in enumerate(cases):
        w = f"cases[{i}]"
        _require(isinstance(case, dict), f"{w}: 必须是对象")
        extra = set(case) - {
            "case_id",
            "inputs",
            "outputs",
            "attrs",
            "tuning",
            "tuning_reason",
            "active_vector_cores",
            "comment",
            "template",
        }
        _require(not extra, f"{w}: 未知字段 {sorted(extra)}")
        cid = case.get("case_id")
        _require(isinstance(cid, str) and cid, f"{w}: 缺 case_id")
        _require(cid not in seen, f"{w}: case_id 重复 {cid!r}")
        seen.add(cid)

        inputs = case.get("inputs")
        _require(
            isinstance(inputs, list) and inputs, f"{w}({cid}): inputs 必须是非空数组"
        )
        for j, t in enumerate(inputs):
            _check_tensor(t, f"{w}({cid}).inputs[{j}]", dtype_bytes)
        for j, t in enumerate(case.get("outputs", []) or []):
            _check_tensor(t, f"{w}({cid}).outputs[{j}]", dtype_bytes)
        _require(
            "attrs" not in case or isinstance(case["attrs"], dict),
            f"{w}({cid}): attrs 必须是对象",
        )

        tuning = case.get("tuning") or {}
        _require(isinstance(tuning, dict), f"{w}({cid}): tuning 必须是对象")
        bad = set(tuning) - TUNING_WHITELIST
        _require(
            not bad,
            f"{w}({cid}): tuning 含非白名单 key {sorted(bad)}（防止改到引擎标定）",
        )
    return spec


def load_spec(path, dtype_bytes):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate(spec, dtype_bytes)


# ---------------- builder 通用辅助 ----------------
def merged_attrs(spec, case):
    """default_attrs 浅合并 case.attrs。"""
    out = dict(spec.get("default_attrs") or {})
    out.update(case.get("attrs") or {})
    return out


def expand_inputs(case):
    """把 repeat>1 的张量展开成多条，得到真实输入序列（动态输入个数）。"""
    out = []
    for t in case["inputs"]:
        out.extend([t] * int(t.get("repeat", 1)))
    return out


def elems(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def tensor_bytes(t, dtype_bytes):
    return elems(t["shape"]) * dtype_bytes[t["dtype"]]


def find_input(case, name, default=None):
    """按名取输入张量；动态/可选输入时方便 builder 拿主输入或权重。"""
    for t in case["inputs"]:
        if t.get("name") == name:
            return t
    return default


def apply_tuning(case_dict, case):
    """把 case 的 tuning（白名单已校验）注入引擎 case dict。"""
    for k, v in (case.get("tuning") or {}).items():
        case_dict[k] = v
    if "active_vector_cores" in case:
        case_dict["active_vector_cores"] = case["active_vector_cores"]
    return case_dict
