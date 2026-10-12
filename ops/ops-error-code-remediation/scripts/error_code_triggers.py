#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------
"""Shared trigger definitions for candidate scanning and final auditing."""

from __future__ import annotations

import re
from typing import Iterable


SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".hxx", ".cuh", ".inc", ".inl"
}
DEFAULT_EXCLUDES = {
    ".git", ".svn", "build", "out", "output", "dist", "install", "third_party", "node_modules"
}
CODE_RE = re.compile(r"\bE[A-Z0-9][0-9]{4}\b")
E9_RE = re.compile(r"\bE[A-Z0-9]9[0-9]{3}\b")
DEPRECATED_RE = re.compile(r"\bEZ000[123567]\b")
DEPRECATED = {"EZ0001", "EZ0002", "EZ0003", "EZ0005", "EZ0006", "EZ0007"}
DEPRECATED_MACROS = {
    "OP_LOGE_WITH_INVALID_INPUT_SHAPE": "EZ0001",
    "OP_LOGE_WITH_INVALID_ATTR": "EZ0002",
    "OP_LOGE_WITH_INVALID_ATTR_SIZE": "EZ0003",
    "OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE": "EZ0005",
    "OP_LOGE_WITH_INVALID_INPUT_FORMAT": "EZ0006",
    "OP_LOGE_WITH_INVALID_INPUT_DTYPE": "EZ0007",
}

NULL_ENTRIES = {
    "OP_CHECK_NULL_WITH_CONTEXT",
    "OPS_CHECK_NULL_WITH_CONTEXT",
    "OPS_CHECK_NULL_WITH_CONTEXT_RET",
    "OPS_CHECK_NOT_NULL_WITH_CONTEXT",
    "MOE_OPS_CHECK_NULL_WITH_CONTEXT_RET",
}
VECTOR_ENTRIES = {
    "VECTOR_INNER_ERR_REPORT_TILIING",
    "OPS_REPORT_VECTOR_INNER_ERR",
    "MOE_VECTOR_INNER_ERR_REPORT_TILIING",
}
CUBE_ENTRIES = {
    "OPS_REPORT_CUBE_INNER_ERR",
    "CUBE_INNER_ERR_REPORT",
    "CUBE_INNER_ERR_REPORT_PLUGIN",
}
GRAPH_PARSER_ENTRIES = {
    "ONNX_PLUGIN_LOGE",
    "GE_OP_LOGE_WITH_REPORT",
    "INFER_AXIS_TYPE_ERR_REPORT",
}
TILING_EXEC_ENTRIES = {
    "TR_EXECUTE",
    "REGISTER_OP_TILING_V3_WITH_VECTOR",
    "REGISTER_OP_TILING_V3_CUSTOM",
    "REGISTER_OP_TILING_V4_WITH_VECTOR",
    "REGISTER_OP_TILING_V4_CUSTOM",
    "REGISTER_OP_TILING_V4_CUBE_PATTERN",
}
RETURN_CHECK_ENTRIES = {"CHECK_RET", "CHECK_COND"}
MIXED_FORMAL_REPORT_NAMES = {"ReportErrMessage"}
MIXED_INTERNAL_REPORT_NAMES = {"ReportInterErrMessage"}
ERRNO_REPORT_NAMES = {"ReportErrorMessage"}
FORMAL_REPORT_NAMES = {
    "REPORT_PREDEFINED_ERR_MSG",
    "REPORT_CALL_ERROR",
    "REPORT_ERROR",
    "REPORT_ERROR_MESSAGE",
} | MIXED_FORMAL_REPORT_NAMES
INTERNAL_REPORT_NAMES = {
    "REPORT_INNER_ERR_MSG",
    "REPORT_INNER_ERROR",
    "REPORT_INTER_ERR_MESSAGE",
} | MIXED_INTERNAL_REPORT_NAMES
LOG_ONLY_NAMES = {
    "OP_LOGE_WITHOUT_REPORT",
    "OPS_LOG_E",
    "D_OP_LOGE",
    "GELOGE",
    "HCCL_ERROR",
}

KNOWN_ERRNO_CODES = {
    "ACLNN_ERR_PARAM_NULLPTR": "EZ1001",
    "ACLNN_ERR_PARAM_INVALID": "EZ1001",
    "ACL_ERROR_INVALID_PARAM": "EZ1001",
    "ACLNN_ERR_RUNTIME_ERROR": "EZ9903",
    "ACL_ERROR_FAILURE": "EZ9999",
    "ACLNN_ERR_INNER": "EZ9999",
    "ACLNN_ERR_INNER_INFERSHAPE_ERROR": "EZ9999",
    "ACLNN_ERR_INNER_TILING_ERROR": "EZ9999",
    "ACLNN_ERR_INNER_FIND_KERNEL_ERROR": "EZ9999",
    "ACLNN_ERR_INNER_CREATE_EXECUTOR": "EZ9999",
    "ACLNN_ERR_INNER_NULLPTR": "EZ9999",
    "ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID": "EZ9999",
}

EXACT_TRIGGER_NAMES = (
    set(DEPRECATED_MACROS)
    | NULL_ENTRIES
    | VECTOR_ENTRIES
    | CUBE_ENTRIES
    | GRAPH_PARSER_ENTRIES
    | TILING_EXEC_ENTRIES
    | RETURN_CHECK_ENTRIES
    | FORMAL_REPORT_NAMES
    | INTERNAL_REPORT_NAMES
    | ERRNO_REPORT_NAMES
    | LOG_ONLY_NAMES
)
_EXACT_PATTERN = "|".join(re.escape(name) for name in sorted(EXACT_TRIGGER_NAMES, key=len, reverse=True))
CALL_RE = re.compile(
    r"\b((?:[A-Za-z_]\w*::)*(?:"
    rf"{_EXACT_PATTERN}|"
    r"REPORT_[A-Z0-9_]+|"
    r"OP_LOGE(?:_[A-Z0-9_]+)?|"
    r"OPS_[A-Z0-9_]*(?:LOGE|REPORT)[A-Z0-9_]*|"
    r"[A-Z][A-Z0-9_]*(?:INNER_ERR_REPORT|ERR_REPORT|LOGE)[A-Z0-9_]*"
    r"))\s*\("
)


def bare_name(name: str) -> str:
    """Strip namespace qualifiers from a call name."""
    return name.split("::")[-1]


def first_trigger(statement: str, names: Iterable[str]) -> str:
    """Return the first named trigger present in a statement."""
    for name in sorted(names, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\s*\(", statement):
            return name
    return ""


def first_argument(statement: str, entry: str) -> str:
    """Extract a call's first argument while respecting nested parentheses and strings."""
    bare = bare_name(entry)
    match = re.search(rf"\b(?:[A-Za-z_]\w*::)*{re.escape(bare)}\s*\(", statement)
    if not match:
        return ""
    start = match.end()
    depth = 0
    quote = ""
    escaped = False
    for index in range(start, len(statement)):
        char = statement[index]
        if escaped:
            escaped = False
            continue
        if quote and char == "\\":
            escaped = True
            continue
        if char in {'"', "'"}:
            if not quote:
                quote = char
            elif quote == char:
                quote = ""
            continue
        if quote:
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                return statement[start:index].strip()
            depth -= 1
        elif char == "," and depth == 0:
            return statement[start:index].strip()
    return ""


def errno_code(expression: str) -> str:
    """Map a known or literal nnopbase errno to its ErrorManager code family."""
    normalized = re.sub(r"\b(?:static_cast|reinterpret_cast)\s*<[^>]+>\s*", "", expression).strip()
    normalized = normalized.strip("() ")
    if normalized in KNOWN_ERRNO_CODES:
        return KNOWN_ERRNO_CODES[normalized]
    if normalized.startswith("ACLNN_ERR_PARAM"):
        return "EZ1001"
    if normalized == "ACLNN_ERR_RUNTIME_ERROR":
        return "EZ9903"
    if normalized.startswith("ACLNN_ERR_INNER") or normalized == "ACL_ERROR_FAILURE":
        return "EZ9999"
    numeric = re.fullmatch(r"[+-]?(\d+)(?:[uUlL]*)", normalized)
    if not numeric:
        return ""
    leading = numeric.group(1)[0]
    return {"1": "EZ1001", "3": "EZ9903", "5": "EZ9999"}.get(leading, "")
