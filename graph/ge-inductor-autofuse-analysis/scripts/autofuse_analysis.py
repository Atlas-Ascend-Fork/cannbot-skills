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
"""
AutoFuse 融合性能分析统一脚本（脚本合一）

将三类分析合并为单一入口，通过子命令调度：
  profile   — 单组 Profiling 分析（融合模式识别 / 热点 / 重复结构 / 可融未融 / 问题算子）
  graph     — Dump 图结构分析（pbtxt / output_code.py，重复子图 / 可融未融 / 融合劣化，--> 箭头可视化）
  compare   — 融合对比（before-after 融合前后 / cross-mode 双模式，以 AutoFuse 为基线）
  all       — 单模式综合分析（profile + graph，自动发现图文件）

可视化约定: 子图/执行片段统一使用 `OP_Type --> OP_Type` 文本箭头格式，
           仅显示 OP Type（融合算子标注 [fused]），详情（算子名/shape/dtype）置于下方表格。

用法:
  python autofuse_analysis.py profile <profiling_dir> [-o report.md]
  python autofuse_analysis.py graph   <graph_file> [--profiling-dir DIR] [-o report.md]
  python autofuse_analysis.py compare --mode before-after --before-prof <dir> --after-prof <dir> [-o report.md]
  python autofuse_analysis.py compare --mode cross-mode --mode-a-prof <dir> --mode-a-name "AutoFuse" \
                                      --mode-b-prof <dir> --mode-b-name "Triton" [-o report.md]
  python autofuse_analysis.py all     <profiling_dir> [--graph-dir DIR] [--graph-file FILE] [-o report.md]

依赖: 仅 Python 标准库
参考: AF_DVM_performance_analysis_guide.md（AF/DVM 性能差异分析方法手册）
"""

import os
import sys
import re
import ast
import csv
import glob
import logging
import argparse
from collections import defaultdict, deque, Counter, OrderedDict
from typing import Optional, List, Dict, Any, Tuple, Set

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
#  通用 CSV / 数值工具
# ============================================================


def read_csv_safe(filepath: str) -> Tuple[List[str], List[Dict[str, str]]]:
    if not filepath or not os.path.exists(filepath):
        return [], []
    for encoding in ["utf-8-sig", "utf-8", "gbk", "latin-1"]:
        try:
            with open(filepath, "r", encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return [], []
                hl = len(header)
                rows = []
                for row in reader:
                    if not row or all(c.strip() == "" for c in row):
                        continue
                    if len(row) < hl:
                        row.extend([""] * (hl - len(row)))
                    elif len(row) > hl:
                        row = row[:hl]
                    rows.append(dict(zip(header, row)))
                return header, rows
        except (UnicodeDecodeError, csv.Error, OSError):
            continue
    return [], []


def safe_float(val) -> float:
    if val is None:
        return 0.0
    val = str(val).strip()
    if val in ("", "N/A", "n/a", "NA", "None", "null"):
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def safe_int(val) -> int:
    return int(safe_float(val))


# ============================================================
#  融合前缀 / 模式识别常量
# ============================================================

FUSION_PREFIXES = {
    "autofuse": ["autofuse_", "autofused_"],
    "triton": ["triton_poi_fused_", "triton_per_fused_", "triton_unk_fused_"],
    "dvm": ["dvm_fused_", "dvm_"],
}

GE_AF_FUSED_TYPES = {"FusedAscBackend", "AscBackend"}

FUSED_PREFIXES = (
    "autofused",
    "autofuse_",
    "triton_poi_fused",
    "triton_per_fused",
    "triton_unk_fused",
    "dvm_fused",
    "dvm_",
)

FUSED_PREFIX_LEN = len("fused_")


def is_fused_op_type(op_type: str) -> bool:
    for prefixes in FUSION_PREFIXES.values():
        if any(op_type.startswith(p) for p in prefixes):
            return True
    return op_type in GE_AF_FUSED_TYPES


def is_fused_op(op_type: str) -> bool:
    return (
        any(op_type.startswith(p) for p in FUSED_PREFIXES)
        or op_type in GE_AF_FUSED_TYPES
    )


def is_fused_node(name: str, op_type: str) -> bool:
    return is_fused_op(op_type) or is_fused_op(name)


def get_fusion_prefix(op_type: str) -> str:
    for mode, prefixes in FUSION_PREFIXES.items():
        for p in prefixes:
            if op_type.startswith(p):
                return mode
    if op_type in GE_AF_FUSED_TYPES:
        return "autofuse"
    return ""


def infer_semantic(op_type: str) -> str:
    """从融合算子名推断语义（去掉前缀和 hash）"""
    for prefixes in FUSION_PREFIXES.values():
        for p in prefixes:
            if op_type.startswith(p):
                prefix_len = len(p)
                name = op_type[prefix_len:]
                m = re.search(r"_([0-9a-f]{32})$", name)
                if m:
                    name = name[: m.start()]
                return name.replace("_", " ").strip()
    if op_type == "FusedAscBackend":
        return "GE fused kernel"
    if op_type == "AscBackend":
        return "GE autofuse backend"
    return op_type


infer_fusion_semantics = infer_semantic


def sanitize_mermaid_label(text: str) -> str:
    """转义 Mermaid 节点标签中的特殊字符"""
    return text.replace("/", "_").replace("#", "_").replace('"', "'")


# ============================================================
#  融合规则知识库（来自 SKILL.md / subgraph-generalization）
# ============================================================

CUBE_OP_TYPES = {
    "MatMulV3",
    "BatchMatMulV3",
    "MatMul",
    "BatchMatMul",
    "Conv2D",
    "Conv2DV3",
    "Convolution",
    "aten.mm",
    "aten.bmm",
    "aten.addmm",
    "aten.baddbmm",
    "aten.matmul",
}

FUSION_CATEGORY = {
    "Elemwise": {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "DivNoNan",
        "RealDiv",
        "Abs",
        "Exp",
        "Log",
        "Log1p",
        "Sqrt",
        "Rsqrt",
        "Pow",
        "Neg",
        "Reciprocal",
        "Square",
        "Sigmoid",
        "Relu",
        "Tanh",
        "Gelu",
        "GeluV2",
        "Cast",
        "ClipByValue",
        "ClipByValueV2",
        "Maximum",
        "Minimum",
        "Floor",
        "Ceil",
        "Round",
        "Sign",
        "FloorDiv",
        "FloorMod",
        "Erf",
        "Erfc",
        "Where",
        "Equal",
        "Greater",
        "Less",
        "NotEqual",
        "LogicalAnd",
        "LogicalOr",
        "FusedMulAdd",
        "Softplus",
        "Softsign",
        "HardSwish",
        "HardSigmoid",
        "aten.add",
        "aten.sub",
        "aten.mul",
        "aten.div",
        "aten.abs",
        "aten.exp",
        "aten.sqrt",
        "aten.rsqrt",
        "aten.neg",
        "aten.reciprocal",
        "aten.sigmoid",
        "aten.relu",
        "aten.tanh",
        "aten.gelu",
        "aten.clamp",
        "aten.where",
        "aten.pow",
        "aten.maximum",
        "aten.minimum",
    },
    "Broadcast": {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "DivNoNan",
        "Maximum",
        "Minimum",
        "aten.add",
        "aten.sub",
        "aten.mul",
        "aten.div",
    },
    "Reduce": {
        "ReduceMax",
        "ReduceMin",
        "ReduceSum",
        "ReduceMean",
        "ReduceAll",
        "ReduceAny",
        "VarMean",
        "aten.sum",
        "aten.mean",
        "aten.amax",
        "aten.amin",
    },
    "Concat": {"Concat", "ConcatD", "Pack", "aten.cat", "aten.stack"},
    "Slice": {
        "Slice",
        "StridedSlice",
        "Select",
        "SelectV2",
        "aten.slice",
        "aten.select",
    },
    "Split": {
        "Split",
        "SplitD",
        "SplitV",
        "Unpack",
        "aten.split",
        "aten.split_with_sizes",
    },
    "Gather": {
        "Gather",
        "GatherV2",
        "Embedding",
        "IndexSelect",
        "aten.embedding",
        "aten.gather",
        "aten.index_select",
    },
    "Transpose": {
        "Transpose",
        "TransposeD",
        "Permute",
        "aten.t",
        "aten.permute",
        "aten.transpose",
    },
    "Matmul": {
        "MatMul",
        "MatMulV3",
        "BatchMatMul",
        "BatchMatMulV3",
        "aten.mm",
        "aten.bmm",
        "aten.addmm",
        "aten.matmul",
    },
}

VERTICAL_FUSE_TABLE = {
    ("Elemwise", "Elemwise"): True,
    ("Elemwise", "Broadcast"): True,
    ("Elemwise", "Reduce"): True,
    ("Elemwise", "Concat"): True,
    ("Elemwise", "Slice"): True,
    ("Elemwise", "Split"): True,
    ("Elemwise", "Gather"): True,
    ("Elemwise", "Transpose"): True,
    ("Elemwise", "Matmul"): False,
    ("Broadcast", "Elemwise"): True,
    ("Broadcast", "Broadcast"): True,
    ("Broadcast", "Reduce"): True,
    ("Broadcast", "Concat"): True,
    ("Broadcast", "Slice"): True,
    ("Broadcast", "Split"): True,
    ("Broadcast", "Gather"): True,
    ("Broadcast", "Transpose"): True,
    ("Broadcast", "Matmul"): False,
    ("Reduce", "Elemwise"): True,
    ("Reduce", "Broadcast"): True,
    ("Reduce", "Reduce"): True,
    ("Reduce", "Concat"): False,
    ("Reduce", "Slice"): True,
    ("Reduce", "Split"): False,
    ("Reduce", "Gather"): False,
    ("Reduce", "Transpose"): True,
    ("Reduce", "Matmul"): False,
    ("Concat", "Elemwise"): True,
    ("Concat", "Broadcast"): True,
    ("Concat", "Reduce"): False,
    ("Concat", "Concat"): True,
    ("Concat", "Slice"): True,
    ("Concat", "Split"): False,
    ("Concat", "Gather"): False,
    ("Concat", "Transpose"): True,
    ("Concat", "Matmul"): False,
    ("Slice", "Elemwise"): True,
    ("Slice", "Broadcast"): True,
    ("Slice", "Reduce"): True,
    ("Slice", "Concat"): True,
    ("Slice", "Slice"): True,
    ("Slice", "Split"): True,
    ("Slice", "Gather"): True,
    ("Slice", "Transpose"): True,
    ("Slice", "Matmul"): False,
    ("Split", "Elemwise"): True,
    ("Split", "Broadcast"): True,
    ("Split", "Reduce"): False,
    ("Split", "Concat"): True,
    ("Split", "Slice"): True,
    ("Split", "Split"): False,
    ("Split", "Gather"): False,
    ("Split", "Transpose"): True,
    ("Split", "Matmul"): False,
    ("Gather", "Elemwise"): True,
    ("Gather", "Broadcast"): True,
    ("Gather", "Reduce"): True,
    ("Gather", "Concat"): False,
    ("Gather", "Slice"): True,
    ("Gather", "Split"): False,
    ("Gather", "Gather"): False,
    ("Gather", "Transpose"): True,
    ("Gather", "Matmul"): False,
    ("Transpose", "Elemwise"): True,
    ("Transpose", "Broadcast"): True,
    ("Transpose", "Reduce"): True,
    ("Transpose", "Concat"): True,
    ("Transpose", "Slice"): True,
    ("Transpose", "Split"): True,
    ("Transpose", "Gather"): True,
    ("Transpose", "Transpose"): True,
    ("Transpose", "Matmul"): True,
    ("Matmul", "Elemwise"): True,
    ("Matmul", "Broadcast"): True,
    ("Matmul", "Reduce"): False,
    ("Matmul", "Concat"): False,
    ("Matmul", "Slice"): False,
    ("Matmul", "Split"): False,
    ("Matmul", "Gather"): False,
    ("Matmul", "Transpose"): False,
    ("Matmul", "Matmul"): False,
}

HORIZONTAL_FUSE_PAIRS = {
    ("Elemwise", "Elemwise"),
    ("Elemwise", "Broadcast"),
    ("Broadcast", "Broadcast"),
}


def classify_op_for_fusion(op_type: str, node_name: str = "") -> str:
    """将算子分类为融合规则类别。对 autofuse 融合算子，通过名字关键词推断类别（不区分大小写）。"""
    key = op_type.lower()
    for cat, ops in FUSION_CATEGORY.items():
        if key in {op.lower() for op in ops}:
            return cat
    if node_name:
        low = (node_name + "_" + op_type).lower()
        if any(k in low for k in ["mm", "matmul", "bmm", "addmm"]):
            return "Matmul"
        if any(
            k in low
            for k in [
                "sum",
                "mean",
                "reduce_max",
                "reduce_min",
                "reducemax",
                "reducemin",
                "reducesum",
                "reducemean",
                "reduce_all",
                "reduceall",
                "var_mean",
            ]
        ):
            return "Reduce"
        if any(
            k in low for k in ["embedding", "gather", "index_select", "indexselect"]
        ):
            return "Gather"
        if any(k in low for k in ["cat", "concat", "pack", "stack"]):
            return "Concat"
        if any(k in low for k in ["slice", "strided_slice", "select"]):
            return "Slice"
        if any(k in low for k in ["split", "unpack", "split_with_sizes"]):
            return "Split"
        if any(k in low for k in ["transpose", "permute"]):
            return "Transpose"
        if any(
            k in low
            for k in [
                "cast",
                "clip",
                "add",
                "sub",
                "mul",
                "div",
                "exp",
                "sqrt",
                "rsqrt",
                "pow",
                "neg",
                "abs",
                "sigmoid",
                "relu",
                "tanh",
                "gelu",
                "where",
                "equal",
                "greater",
                "less",
                "maximum",
                "minimum",
                "floor",
                "ceil",
                "round",
                "sign",
                "erf",
                "log",
                "log1p",
                "reciprocal",
                "square",
                "floordiv",
                "floormod",
                "not_equal",
                "logical",
                "softplus",
                "hardswish",
                "hardsigmoid",
                "fusedmuladd",
            ]
        ):
            return "Elemwise"
    return "Unknown"


def can_vertically_fuse(cat_before: str, cat_after: str) -> bool:
    return VERTICAL_FUSE_TABLE.get((cat_before, cat_after), False)


def can_horizontally_fuse(cat_a: str, cat_b: str) -> bool:
    return (cat_a, cat_b) in HORIZONTAL_FUSE_PAIRS or (
        cat_b,
        cat_a,
    ) in HORIZONTAL_FUSE_PAIRS


# ============================================================
#  Profiling 文件发现
# ============================================================


class ProfilingFileLocator:
    FILE_PATTERNS = {
        "op_summary": ["*op_summary*.csv", "kernel_details.csv"],
        "op_statistic": ["*op_statistic*.csv", "op_statistic.csv"],
        "api_statistic": ["api_statistic_*.csv", "api_statistic.csv"],
        "step_trace": ["step_trace_*.csv", "step_trace_time.csv"],
        "msprof_json": ["msprof_*.json", "trace_view.json"],
        "task_time": ["task_time_*.csv", "task_time.csv"],
        "fusion_op": ["fusion_op_*.csv", "fusion_op.csv"],
    }

    def __init__(self, profiling_dir: str):
        self.profiling_dir = profiling_dir
        self.files: Dict[str, Optional[str]] = {}

    def discover(self) -> Dict[str, Optional[str]]:
        for data_type, patterns in self.FILE_PATTERNS.items():
            self.files[data_type] = self._find(patterns)
        return self.files

    def get(self, data_type: str) -> Optional[str]:
        return self.files.get(data_type)

    def _find(self, patterns: List[str]) -> Optional[str]:
        for pattern in patterns:
            matches = glob.glob(
                os.path.join(self.profiling_dir, "**", pattern), recursive=True
            )
            if matches:
                if "op_summary" in pattern and len(matches) > 1:
                    return self._pick_best_op_summary(matches)
                return matches[0]
        return None

    @staticmethod
    def _pick_best_op_summary(matches: List[str]) -> str:
        best, best_score = matches[0], -200
        for path in matches:
            score = 0
            base = os.path.basename(path).lower()
            if "_output_" in base:
                score -= 100
            if "_no_op_name" in base:
                score -= 50
            for enc in ["utf-8-sig", "utf-8", "gbk"]:
                try:
                    with open(path, "r", encoding=enc) as f:
                        header = next(csv.reader(f), None)
                        if header and "Op Name" in header:
                            score += 10
                            for row in csv.reader(f):
                                if any(c.strip() for c in row):
                                    score += 5
                                    break
                        break
                except Exception:
                    continue
            if score > best_score:
                best, best_score = path, score
        return best


# ============================================================
#  融合模式识别
# ============================================================


def detect_fusion_mode(op_statistic_rows, api_statistic_rows, fusion_op_rows=None):
    """从 op_statistic 融合算子前缀识别融合模式（禁止从 dump 图读环境变量）"""
    result = {
        "mode": "unknown",
        "mode_label": "无法判断",
        "evidence": [],
        "fusion_prefix": "",
        "has_model_execute": False,
    }
    has_autofuse = has_triton = has_dvm = has_ge_fused = False
    fusion_types = []
    for row in op_statistic_rows:
        op_type = str(row.get("OP Type", "")).strip()
        if any(op_type.startswith(p) for p in FUSION_PREFIXES["autofuse"]):
            has_autofuse = True
            fusion_types.append(op_type)
        elif any(op_type.startswith(p) for p in FUSION_PREFIXES["triton"]):
            has_triton = True
            fusion_types.append(op_type)
        elif any(op_type.startswith(p) for p in FUSION_PREFIXES["dvm"]):
            has_dvm = True
            fusion_types.append(op_type)
        elif op_type in GE_AF_FUSED_TYPES:
            has_ge_fused = True
            fusion_types.append(op_type)
    has_autofuse_in_fusion_op = False
    if fusion_op_rows:
        for row in fusion_op_rows:
            if str(row.get("Fusion Op", "")).strip().startswith("autofuse_"):
                has_autofuse_in_fusion_op = True
                break
    has_model_execute = False
    for row in api_statistic_rows:
        if "ModelExecute" in str(row.get("API Name", "")):
            if safe_int(row.get("Count", 0)) > 1:
                has_model_execute = True
                result["has_model_execute"] = True
                break
    if has_triton:
        result.update(
            mode="inductor_triton",
            mode_label="Inductor + Triton",
            fusion_prefix="triton",
        )
        result["evidence"].append(
            f"op_statistic: 含 triton_*_fused_ 前缀融合算子 ({len(fusion_types)}种)"
        )
    elif has_dvm:
        result.update(
            mode="inductor_dvm", mode_label="Inductor + DVM", fusion_prefix="dvm"
        )
        result["evidence"].append(
            f"op_statistic: 含 dvm_ 前缀融合算子 ({len(fusion_types)}种)"
        )
    elif has_autofuse and has_model_execute:
        result.update(
            mode="ge_autofuse", mode_label="GE + AutoFuse", fusion_prefix="autofuse"
        )
        result["evidence"].append(
            f"op_statistic: 含 autofuse_/autofused_ 前缀融合算子 ({len(fusion_types)}种)"
        )
        result["evidence"].append("api_statistic: ModelExecute")
    elif has_ge_fused and has_model_execute:
        result.update(
            mode="ge_autofuse", mode_label="GE + AutoFuse", fusion_prefix="autofuse"
        )
        result["evidence"].append(
            f"op_statistic: 含 FusedAscBackend/AscBackend 融合算子 ("
            f"{len([t for t in fusion_types if t in GE_AF_FUSED_TYPES])}种)"
        )
        result["evidence"].append("api_statistic: ModelExecute")
        if has_autofuse_in_fusion_op:
            result["evidence"].append("fusion_op.csv: 含 autofuse_ 前缀融合算子")
    elif has_autofuse:
        result.update(
            mode="inductor_ascendc",
            mode_label="Inductor + AutoFuse(AscendC)",
            fusion_prefix="autofuse",
        )
        result["evidence"].append(
            f"op_statistic: 含 autofuse_/autofused_ 前缀融合算子 ({len(fusion_types)}种)，无 ModelExecute"
        )
    else:
        result["evidence"].append("未检测到融合算子前缀")
    result["fusion_op_types"] = fusion_types
    return result


# ============================================================
#  GraphNode + 图格式识别
# ============================================================


class GraphNode:
    def __init__(self):
        self.name: str = ""
        self.op_type: str = ""
        self.raw_op_type: str = ""
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.input_shapes: List[str] = []
        self.input_dtypes: List[str] = []
        self.output_shapes: List[str] = []
        self.output_dtypes: List[str] = []
        self.is_subgraph: bool = False
        self.subgraph_nodes: List["GraphNode"] = []
        self.source_line: int = 0
        self.raw_call: str = ""
        self.is_fused: bool = False
        self.shape_str: str = ""
        self.dtype_str: str = ""

    def get_shape_dtype_label(self) -> str:
        if self.shape_str and self.dtype_str:
            return f"{self.dtype_str} {self.shape_str}"
        if self.shape_str:
            return self.shape_str
        if self.dtype_str:
            return self.dtype_str
        return ""


def detect_graph_type(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pbtxt":
        return "pbtxt"
    elif ext == ".py":
        return "fxgraph"
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(2000)
        if "ir_version:" in head or "graph {" in head:
            return "pbtxt"
        if "class Repro" in head or "torch.ops" in head:
            return "fxgraph"
        raise ValueError(f"无法识别文件格式: {filepath}")


def get_trivial_types(graph_type: str) -> Set[str]:
    if graph_type == "pbtxt":
        return {
            "Data",
            "Const",
            "Constant",
            "Variable",
            "Placeholder",
            "NetOutput",
            "subgraph",
            "Send",
            "Recv",
            "NoOp",
            "Identity",
        }
    else:
        return {
            "getitem",
            "item",
            "reinterpret_tensor",
            "empty_strided",
            "empty_strided_cpu",
            "empty_strided_cpu_pinned",
            "alloc_from_pool",
        }


def is_compute_op(op_type: str, trivial_types: Set[str]) -> bool:
    return op_type not in trivial_types and op_type != ""


# ============================================================
#  Pbtxt 解析器
# ============================================================


class PbtxtParser:
    def __init__(self):
        self.ir_version = ""
        self.producer_name = ""

    def parse_file(self, filepath: str) -> List[GraphNode]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return self._parse_lines(lines)

    def _parse_lines(self, lines):
        re_input = re.compile(r'^\s*input:\s*"?(.*?)"?\s*$')
        re_output = re.compile(r'^\s*output:\s*"?(.*?)"?\s*$')
        re_name = re.compile(r'^\s*name:\s*"?(.*?)"?\s*$')
        re_op_type = re.compile(r'^\s*op_type:\s*"?(.*?)"?\s*$')
        regexes = {
            "input": re_input,
            "output": re_output,
            "name": re_name,
            "op_type": re_op_type,
        }
        i, n = 0, len(lines)
        top_nodes = []
        while i < n:
            line = lines[i].strip()
            if line.startswith("ir_version:"):
                self.ir_version = line.split(":", 1)[1].strip()
            elif line == "graph {":
                i += 1
                break
            i += 1
        while i < n:
            line = lines[i].strip()
            if line == "}" or line == "},":
                break
            if line.startswith("node {") or line == "node {":
                node, i = self._parse_node(lines, i, regexes)
                if node:
                    node.is_fused = is_fused_node(node.name, node.op_type)
                    top_nodes.append(node)
            else:
                i += 1
        return top_nodes

    def _parse_node(self, lines, start_idx, regexes):
        node = GraphNode()
        i = start_idx + 1
        n = len(lines)
        brace_depth = 1
        attr_name = ""
        attr_s_val = ""
        attr_ints_vals = []
        while i < n and brace_depth > 0:
            line = lines[i]
            stripped = line.strip()
            if stripped.endswith("{") or stripped == "{":
                if (
                    brace_depth == 1
                    and stripped.startswith("attribute {")
                    or stripped == "attribute {"
                ):
                    attr_name = ""
                    attr_s_val = ""
                    attr_ints_vals = []
                brace_depth += 1
                i += 1
                continue
            if stripped == "}":
                brace_depth -= 1
                if brace_depth == 1 and attr_name:
                    self._apply_attr(node, attr_name, attr_s_val, attr_ints_vals)
                    attr_name = ""
                i += 1
                if brace_depth == 0:
                    break
                continue
            m = regexes["input"].match(line)
            if m:
                val = m.group(1).strip().strip('"')
                if val:
                    node.inputs.append(val)
                i += 1
                continue
            m = regexes["output"].match(line)
            if m:
                val = m.group(1).strip().strip('"')
                if val:
                    node.outputs.append(val)
                i += 1
                continue
            if brace_depth == 1:
                m = regexes["name"].match(line)
                if m:
                    node.name = m.group(1).strip().strip('"')
                    i += 1
                    continue
                m = regexes["op_type"].match(line)
                if m:
                    raw = m.group(1).strip().strip('"')
                    node.raw_op_type = raw
                    node.op_type = raw[3:] if raw.startswith("ge:") else raw
                    i += 1
                    continue
            if brace_depth == 2:
                m = re.match(r'^\s*name:\s*"?(.*?)"?\s*$', line)
                if m:
                    attr_name = m.group(1).strip().strip('"')
                    i += 1
                    continue
                m = re.match(r'^\s*s:\s*"?(.*?)"?\s*$', line)
                if m:
                    attr_s_val = m.group(1).strip().strip('"')
                    i += 1
                    continue
                m = re.match(r"^\s*ints:\s*(\d+)", line)
                if m:
                    attr_ints_vals.append(int(m.group(1)))
                    i += 1
                    continue
            i += 1
        return node, i

    @staticmethod
    def _apply_attr(node, attr_name, s_val, ints_vals):
        if attr_name.startswith("output_desc_dtype:"):
            dtype = s_val.replace("DT_", "").lower() if s_val else ""
            if dtype:
                node.output_dtypes.append(dtype)
        elif attr_name.startswith("output_desc_shape:"):
            shape = "x".join(str(v) for v in ints_vals) if ints_vals else ""
            if shape:
                node.output_shapes.append(shape)
        elif attr_name.startswith("input_desc_dtype:"):
            dtype = s_val.replace("DT_", "").lower() if s_val else ""
            if dtype:
                node.input_dtypes.append(dtype)
        elif attr_name.startswith("input_desc_shape:"):
            shape = "x".join(str(v) for v in ints_vals) if ints_vals else ""
            if shape:
                node.input_shapes.append(shape)


# ============================================================
#  FX Graph AST 解析器（output_code.py / fx_graph_runnable.py）
# ============================================================


class FXGraphParser:
    _METHOD_NAMES = ("forward", "call", "run")
    _CLASS_NAMES = ("Repro", "Runner", "Callable")

    def __init__(self):
        self.source_lines: List[str] = []
        self._alias_map: Dict[str, str] = {}

    def parse_file(self, filepath: str) -> List[GraphNode]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        self.source_lines = source.splitlines()
        tree = ast.parse(source, filename=filepath)
        return self._parse_ast(tree)

    def _parse_ast(self, tree):
        nodes = []
        target_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name in self._CLASS_NAMES:
                    target_class = node
                    break
                if target_class is None:
                    target_class = node
        if target_class is not None:
            entry = self._find_entry_method(target_class.body)
            if entry:
                for stmt in entry.body:
                    self._parse_stmt(stmt, nodes)
                return nodes
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in self._METHOD_NAMES
            ):
                for stmt in node.body:
                    self._parse_stmt(stmt, nodes)
                return nodes
        return nodes

    @staticmethod
    def _find_entry_method(class_body):
        for item in class_body:
            if (
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name in FXGraphParser._METHOD_NAMES
            ):
                return item
        for item in class_body:
            if (
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name != "__init__"
            ):
                return item
        return None

    def _parse_stmt(self, stmt, nodes):
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                self._register_with_alias(item)
            for child in stmt.body:
                self._parse_stmt(child, nodes)
        elif isinstance(stmt, ast.If):
            for child in stmt.body:
                self._parse_stmt(child, nodes)
            for child in stmt.orelse:
                self._parse_stmt(child, nodes)
        elif isinstance(stmt, (ast.For, ast.While)):
            for child in stmt.body:
                self._parse_stmt(child, nodes)
            for child in stmt.orelse:
                self._parse_stmt(child, nodes)
        elif isinstance(stmt, ast.Assign):
            targets = []
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
                elif isinstance(t, ast.Tuple):
                    for elt in t.elts:
                        if isinstance(elt, ast.Name):
                            targets.append(elt.id)
            if not targets:
                return
            target_name = targets[0]
            val = stmt.value
            if isinstance(val, ast.Name):
                self._alias_map[target_name] = self._resolve_alias(val.id)
                return
            if isinstance(val, ast.Call):
                node = self._parse_call(val, target_name, stmt.lineno)
                if node:
                    nodes.append(node)
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name):
                target_name = stmt.target.id
                val = stmt.value
                if val is None:
                    return
                if isinstance(val, ast.Name):
                    self._alias_map[target_name] = self._resolve_alias(val.id)
                    return
                if isinstance(val, ast.Call):
                    node = self._parse_call(val, target_name, stmt.lineno)
                    if node:
                        nodes.append(node)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            node = self._parse_call(stmt.value, f"__expr_{stmt.lineno}", stmt.lineno)
            if node:
                nodes.append(node)

    def _register_with_alias(self, item):
        if item.optional_vars is None:
            return
        if isinstance(item.optional_vars, ast.Name):
            if isinstance(item.context_expr, ast.Call):
                func_name = self._extract_func_name(item.context_expr.func)
                self._alias_map[item.optional_vars.id] = func_name

    def _resolve_alias(self, name):
        seen = set()
        while name in self._alias_map and name not in seen:
            seen.add(name)
            name = self._alias_map[name]
        return name

    @staticmethod
    def _extract_attr_chain(node):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        return parts

    @staticmethod
    def _extract_name(expr):
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Subscript):
            return FXGraphParser._extract_name(expr.value)
        if isinstance(expr, ast.Attribute):
            return FXGraphParser._extract_name(expr.value)
        return ""

    @staticmethod
    def _extract_func_name(func):
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = FXGraphParser._extract_attr_chain(func)
            return ".".join(parts) if parts else ""
        return ""

    @staticmethod
    def _resolve_arg(arg):
        if isinstance(arg, ast.Name):
            return arg.id
        if isinstance(arg, ast.Attribute):
            return ".".join(FXGraphParser._extract_attr_chain(arg))
        if isinstance(arg, ast.Constant):
            return str(arg.value)
        if isinstance(arg, ast.List):
            return "[...]"
        if isinstance(arg, ast.Starred) and isinstance(arg.value, ast.Name):
            return f"*{arg.value.id}"
        return None

    def _parse_call(self, expr, target_name, lineno):
        op_info = self._extract_torch_op(expr.func)
        raw_inputs = []
        for a in expr.args:
            raw_inputs.append(self._resolve_arg(a) or f"<{type(a).__name__}>")
        resolved_inputs = [
            self._resolve_alias(inp)
            if not inp.startswith("'")
            and not inp.startswith("[")
            and not inp.startswith("<")
            else inp
            for inp in raw_inputs
        ]
        if op_info:
            full = (
                f"{op_info[0]}.{op_info[1]}.{op_info[2]}"
                if op_info[2]
                else f"{op_info[0]}.{op_info[1]}"
            )
            node = GraphNode()
            node.name = target_name
            node.raw_op_type = full
            node.op_type = f"{op_info[0]}.{op_info[1]}"
            node.outputs = [target_name]
            node.source_line = lineno
            node.inputs = resolved_inputs
            node.is_fused = is_fused_node(node.name, node.op_type)
            if 0 < lineno <= len(self.source_lines):
                node.raw_call = self.source_lines[lineno - 1].strip()
            return node
        func_name = self._extract_func_name(expr.func)
        if func_name and is_fused_op(func_name):
            node = GraphNode()
            node.name = target_name
            node.raw_op_type = func_name
            node.op_type = func_name
            node.outputs = [target_name]
            node.source_line = lineno
            node.inputs = resolved_inputs
            node.is_fused = True
            if 0 < lineno <= len(self.source_lines):
                node.raw_call = self.source_lines[lineno - 1].strip()
            return node
        if func_name in (
            "reinterpret_tensor",
            "empty_strided",
            "empty_strided_cpu",
            "empty_strided_cpu_pinned",
            "alloc_from_pool",
        ):
            node = GraphNode()
            node.name = target_name
            node.raw_op_type = func_name
            node.op_type = func_name
            node.outputs = [target_name]
            node.source_line = lineno
            node.inputs = resolved_inputs
            if 0 < lineno <= len(self.source_lines):
                node.raw_call = self.source_lines[lineno - 1].strip()
            return node
        return None

    @staticmethod
    def _extract_torch_op(func):
        parts = FXGraphParser._extract_attr_chain(func)
        if len(parts) >= 4 and parts[0] == "torch" and parts[1] == "ops":
            return (parts[2], parts[3], parts[4] if len(parts) > 4 else "")
        return None


# ============================================================
#  Profiling 数据加载器（图分析用，提供耗时/shape 查询）
# ============================================================


class ProfilingDataLoader:
    def __init__(self, profiling_dir: str):
        self.profiling_dir = profiling_dir
        self.op_duration_map: Dict[str, float] = {}
        self.op_type_avg_map: Dict[str, float] = {}
        self.op_shape_map: Dict[str, str] = {}
        self.op_dtype_map: Dict[str, str] = {}
        self.op_timing_map: Dict[str, Dict[str, float]] = {}
        self.op_type_blockdim_map: Dict[str, int] = {}
        self.total_op_time_us: float = 0.0
        self.loaded = False

    def load(self):
        f = self._find_file(["*op_summary*.csv", "kernel_details.csv"])
        if f:
            self._load_op_summary(f)
        f = self._find_file(["*op_statistic*.csv", "op_statistic.csv"])
        if f:
            self._load_op_statistic(f)
        self.loaded = bool(self.op_duration_map or self.op_type_avg_map)

    def get_duration(self, node_name, op_type=""):
        if node_name in self.op_duration_map:
            return self.op_duration_map[node_name]
        for prof_name, dur in self.op_duration_map.items():
            if (
                len(node_name) >= 3
                and len(prof_name) >= 3
                and (node_name in prof_name or prof_name in node_name)
            ):
                return dur
        aclnn_key = (
            f"aclnn{node_name}" if not node_name.startswith("aclnn") else node_name
        )
        if aclnn_key in self.op_duration_map:
            return self.op_duration_map[aclnn_key]
        if op_type and op_type in self.op_type_avg_map:
            return self.op_type_avg_map[op_type]
        return 0.0

    def get_shape_dtype(self, node_name, op_type=""):
        for key in [node_name, op_type]:
            if not key:
                continue
            if key in self.op_shape_map or key in self.op_dtype_map:
                shape = self.op_shape_map.get(key, "")
                dtype = self.op_dtype_map.get(key, "")
                if shape or dtype:
                    return shape, dtype
        for prof_name in self.op_shape_map:
            if len(node_name) >= 3 and (
                node_name in prof_name or prof_name in node_name
            ):
                return self.op_shape_map[prof_name], self.op_dtype_map.get(
                    prof_name, ""
                )
        return "", ""

    def get_timing_breakdown(self, node_name, op_type=""):
        if node_name in self.op_timing_map:
            return self.op_timing_map[node_name]
        for prof_name in self.op_timing_map:
            if len(node_name) >= 3 and (
                node_name in prof_name or prof_name in node_name
            ):
                return self.op_timing_map[prof_name]
        return None

    def _find_file(self, patterns):
        for pattern in patterns:
            matches = glob.glob(
                os.path.join(self.profiling_dir, "**", pattern), recursive=True
            )
            if matches:
                return matches[0]
        return None

    def _load_op_summary(self, filepath):
        header, rows = read_csv_safe(filepath)
        if not rows:
            return
        is_kd = "Name" in (header or []) and "Accelerator Core" in (header or [])
        name_col = "Name" if is_kd else "Op Name"
        type_col = "Type" if is_kd else "OP Type"
        dur_col = "Duration(us)" if is_kd else "Task Duration(us)"
        out_shape_col = "Output Shapes"
        out_dtype_col = "Output Data Types"
        type_durs = defaultdict(list)
        for row in rows:
            op_name = str(row.get(name_col, "")).strip()
            op_type = str(row.get(type_col, "")).strip()
            dur = safe_float(row.get(dur_col, 0))
            if op_name and dur > 0:
                if (
                    op_name not in self.op_duration_map
                    or dur > self.op_duration_map[op_name]
                ):
                    self.op_duration_map[op_name] = dur
                self.total_op_time_us += dur
            if op_type and dur > 0:
                type_durs[op_type].append(dur)
            out_shape = str(row.get(out_shape_col, "")).strip().strip('"')
            out_dtype = str(row.get(out_dtype_col, "")).strip().strip('"')
            if op_name and out_shape and out_shape not in ("N/A", "NA"):
                self.op_shape_map[op_name] = out_shape
            if op_name and out_dtype and out_dtype not in ("N/A", "NA"):
                self.op_dtype_map[op_name] = out_dtype
            if op_name and dur > 0:
                wait_t = safe_float(row.get("Task Wait Time(us)", 0))
                aiv_vec = safe_float(row.get("aiv_vec_time(us)", 0))
                aiv_scalar = safe_float(row.get("aiv_scalar_time(us)", 0))
                aiv_mte2 = safe_float(row.get("aiv_mte2_time(us)", 0))
                aiv_mte3 = safe_float(row.get("aiv_mte3_time(us)", 0))
                aiv_total = safe_float(row.get("aiv_time(us)", 0))
                block_dim = safe_int(row.get("Block Dim", row.get("block_dim", 0)))
                in_shape = str(row.get("Input Shapes", "")).strip().strip('"')
                self.op_timing_map[op_name] = {
                    "dur": dur,
                    "wait": wait_t,
                    "aiv_vec": aiv_vec,
                    "aiv_scalar": aiv_scalar,
                    "aiv_mte2": aiv_mte2,
                    "aiv_mte3": aiv_mte3,
                    "aiv_total": aiv_total,
                    "block_dim": block_dim,
                    "in_shape": in_shape,
                    "out_shape": out_shape,
                    "aiv_vec_ratio": safe_float(row.get("aiv_vec_ratio", 0)),
                    "aiv_scalar_ratio": safe_float(row.get("aiv_scalar_ratio", 0)),
                    "aiv_mte2_ratio": safe_float(row.get("aiv_mte2_ratio", 0)),
                    "aiv_mte3_ratio": safe_float(row.get("aiv_mte3_ratio", 0)),
                    "aic_mac_ratio": safe_float(row.get("aic_mac_ratio", 0)),
                    "aic_scalar_ratio": safe_float(row.get("aic_scalar_ratio", 0)),
                    "aic_mte1_ratio": safe_float(row.get("aic_mte1_ratio", 0)),
                    "aic_mte2_ratio": safe_float(row.get("aic_mte2_ratio", 0)),
                    "aic_mte3_ratio": safe_float(row.get("aic_mte3_ratio", 0)),
                    "aic_fixpipe_ratio": safe_float(row.get("aic_fixpipe_ratio", 0)),
                    "cube_utilization": safe_float(row.get("cube_utilization(%)", 0)),
                }
        for t, ds in type_durs.items():
            self.op_type_avg_map[t] = sum(ds) / len(ds)
        type_blockdims = defaultdict(list)
        for row in rows:
            op_type = str(row.get(type_col, "")).strip()
            if op_type:
                bd = safe_int(row.get("Block Dim", row.get("block_dim", 0)))
                if bd > 0:
                    type_blockdims[op_type].append(bd)
        for t, bds in type_blockdims.items():
            self.op_type_blockdim_map[t] = max(bds) if bds else 0

    def _load_op_statistic(self, filepath):
        _, rows = read_csv_safe(filepath)
        for row in rows:
            op_type = str(row.get("OP Type", "")).strip()
            avg = safe_float(row.get("Avg Time(us)", row.get("Avg(us)", 0)))
            if op_type and avg > 0:
                self.op_type_avg_map[op_type] = avg


# ============================================================
#  子图可视化：--> 箭头结构 + 详情表格（Mermaid 仅作可选建议）
#  图中只保留 OP Type（带 [fused] 标记），不带 op name/dtype/shape；
#  子图下方用表格列出 Op Name / OP Type / Input Shape / Output Shape / dtype / Fused。
# ============================================================


def _node_input_shape_str(node: "GraphNode") -> str:
    if node.input_shapes:
        return ", ".join(node.input_shapes)
    return ""


# ============================================================
#  AF 融合范围中间件解析（asc_graph.py / asc_kernel.py）
# ============================================================


class AscGraphParser:
    """解析 AF 融合算子的 asc_graph.py，提取融合范围与 load/store 次数。

    依据 SKILL：Original ATen（粗粒度范围）、Graph fragment（精确节点）、
    MemoryDep（切片范围）、load/store 次数（用于 mte2/mte3 ratio 高的代码层次定位）。
    """

    @staticmethod
    def find_asc_graph_dir(dump_root, fused_name):
        """根据融合算子名（如 autofused_abs_xxxhash）在 dump 目录下定位对应子目录。"""
        if not dump_root or not fused_name:
            return None
        base_name = fused_name
        for prefix in (
            "triton_poi_fused_",
            "triton_per_fused_",
            "triton_unk_fused_",
            "dvm_fused_",
        ):
            if base_name.startswith(prefix):
                return None
        for cand in (base_name, base_name.split("_", 1)[1] if "_" in base_name else ""):
            if not cand:
                continue
            for root, dirs, _ in os.walk(dump_root):
                if cand in dirs or any(d == cand for d in dirs):
                    full = os.path.join(root, cand)
                    if os.path.isfile(os.path.join(full, "asc_graph.py")):
                        return full
        hash_match = re.search(r"([0-9a-f]{16,})", fused_name)
        if hash_match:
            h = hash_match.group(1)
            for root, dirs, _ in os.walk(dump_root):
                for d in dirs:
                    if h in d and os.path.isfile(os.path.join(root, d, "asc_graph.py")):
                        return os.path.join(root, d)
        return None

    @staticmethod
    def parse_asc_graph(filepath):
        """解析单个 asc_graph.py，返回融合范围 + load/store 次数。"""
        if not filepath or not os.path.isfile(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return None
        result = {
            "file": filepath,
            "original_aten": [],
            "graph_fragments": [],
            "source_nodes": [],
            "load_count": 0,
            "store_count": 0,
            "ops_calls": [],
            "memory_deps": [],
            "input_tensors": set(),
            "output_tensors": set(),
        }
        for m in re.finditer(r"Original ATen:\s*\[([^\]]*)\]", content):
            for aten in re.findall(r"aten\.\w+", m.group(1)):
                if aten not in result["original_aten"]:
                    result["original_aten"].append(aten)
        for m in re.finditer(
            r"Topologically Sorted Source Nodes:\s*\[([^\]]*)\]", content
        ):
            nodes = [n.strip() for n in m.group(1).split(",") if n.strip()]
            result["source_nodes"].extend(nodes)
        frag_blocks = re.findall(r"# Graph fragment:\n((?:#.*\n)+)", content)
        for fb in frag_blocks:
            frag_lines = []
            for ln in fb.split("\n"):
                if ln.startswith("#"):
                    frag_lines.append(ln.lstrip("# ").strip())
            result["graph_fragments"].append(frag_lines)
            for ln in frag_lines:
                cm = re.search(r"call_function\[target=(torch\.ops\.[^\]]+)\]", ln)
                if cm:
                    result["ops_calls"].append(cm.group(1))
                if "PlaceHolder[target=" in ln:
                    pm = re.search(r"PlaceHolder\[target=(\w+)\]", ln)
                    if pm:
                        result["input_tensors"].add(pm.group(1))
        result["load_count"] = len(re.findall(r"\bops\.load\b", content))
        result["store_count"] = len(re.findall(r"\bops\.store", content))
        for m in re.finditer(r"MemoryDep\('([^']+)'", content):
            result["memory_deps"].append(m.group(1))
        result["input_tensors"] = sorted(result["input_tensors"])
        return result

    @staticmethod
    def parse_asc_kernel(filepath):
        """粗解析 asc_kernel.py，识别模板特征（是否含 exp/sigmoid/pow 软件展开等）。"""
        if not filepath or not os.path.isfile(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return None
        features = []
        code_lines = []
        for ln in content.split("\n"):
            stripped = ln.strip()
            if (
                stripped.startswith("#define")
                or stripped.startswith("//")
                or stripped.startswith("/*")
                or stripped.startswith("*")
            ):
                continue
            code_lines.append(ln)
        code_body = "\n".join(code_lines)
        for kw, label in (
            (r"(?:::[Ee]xp\b|graph_hint/exp)", "exp"),
            (r"(?:::[Ss]igmoid\b|graph_hint/sigmoid)", "sigmoid"),
            (r"(?:::[Pp]ow\b|graph_hint/pow)", "pow"),
            (r"(?:::[Dd]iv\b|graph_hint/div)", "div"),
            (r"(?:::[Rr]sqrt\b|graph_hint/rsqrt)", "rsqrt"),
            (r"(?:::[Ss]qrt\b|graph_hint/sqrt)", "sqrt"),
            (r"(?:::[Ll]og\b|::[Ll]n\b|graph_hint/log|graph_hint/ln)", "log"),
            (r"(?:::[Tt]anh\b|graph_hint/tanh)", "tanh"),
        ):
            if re.search(kw, code_body):
                features.append(label)
        has_loop = bool(re.search(r"\bfor\s+\w+\s+in\s+range", content))
        has_vector_api = bool(re.search(r"AscendC\.| ascendc\b", content, re.I))
        return {
            "file": filepath,
            "compute_features": features,
            "has_scalar_loop": has_loop,
            "has_ascendc_api": has_vector_api,
            "size": len(content),
        }

    @staticmethod
    def collect_for_fused_ops(dump_root, fused_op_names):
        """批量收集融合算子对应的 asc_graph/asc_kernel 解析结果。

        返回 {fused_name: {asc_graph: ..., asc_kernel: ...}}。
        """
        cache = {}
        if not dump_root or not fused_op_names:
            return cache
        seen_dirs = set()
        for name in fused_op_names:
            d = AscGraphParser.find_asc_graph_dir(dump_root, name)
            if not d or d in seen_dirs:
                continue
            seen_dirs.add(d)
            ag = AscGraphParser.parse_asc_graph(os.path.join(d, "asc_graph.py"))
            ak = AscGraphParser.parse_asc_kernel(os.path.join(d, "asc_kernel.py"))
            if ag or ak:
                cache[name] = {"asc_graph": ag, "asc_kernel": ak, "dir": d}
        return cache


def build_subgraph_arrow(analyzer, node_details, adj, reverse_adj, title):
    """生成 --> 箭头结构表示 + 详情表格。
    返回 list[str]。图中只保留 OP Type（已融合算子加 [fused]），不画 op name/dtype/shape。
    """
    lines = [f"**{title}**", ""]
    if not node_details:
        lines.append("(空)")
    inst_name_set = {nd["name"] for nd in node_details}
    name_to_idx = {nd["name"]: j for j, nd in enumerate(node_details)}

    drawn_edges = set()
    for nd in node_details:
        src_name = nd["name"]
        for dst_name in adj.get(src_name, set()):
            if dst_name in inst_name_set and dst_name in name_to_idx:
                edge_key = (name_to_idx[src_name], name_to_idx[dst_name])
                drawn_edges.add(edge_key)

    chain_parts = []
    for nd in node_details:
        marker = "[fused]" if nd.get("is_fused") else ""
        chain_parts.append(f"{nd['op_type']}{marker}")
    lines.append(f"`{' --> '.join(chain_parts)}`")

    if not drawn_edges:
        all_producers = defaultdict(list)
        all_consumers = defaultdict(list)
        for nd in node_details:
            name = nd["name"]
            for producer in reverse_adj.get(name, set()):
                if producer not in inst_name_set and producer in analyzer.node_map:
                    all_producers[producer].append(name)
            for consumer in adj.get(name, set()):
                if consumer not in inst_name_set and consumer in analyzer.node_map:
                    all_consumers[consumer].append(name)
        boundary_notes = []
        for producer, targets in all_producers.items():
            if len(targets) >= 2:
                p_node = analyzer.node_map[producer]
                boundary_notes.append(f"共享生产者: {p_node.op_type} ({producer})")
        for consumer, sources in all_consumers.items():
            if len(sources) >= 2:
                c_node = analyzer.node_map[consumer]
                boundary_notes.append(f"共享消费者: {c_node.op_type} ({consumer})")
        if boundary_notes:
            lines.append("")
            lines.append("; ".join(boundary_notes))
    lines.append("")

    lines.append(
        "| # | Op Name | OP Type | Input Shape | Output Shape | dtype | Fused |"
    )
    lines.append(
        "|---|---------|---------|-------------|--------------|-------|:-----:|"
    )
    for j, nd in enumerate(node_details, 1):
        in_shape = nd.get("input_shape", "")
        out_shape = nd.get("shape", "")
        dtype = nd.get("dtype", "")
        fused = "是" if nd.get("is_fused") else "否"
        lines.append(
            f"| {j} | {nd['name']} | {nd['op_type']} | {in_shape} | {out_shape} | {dtype} | {fused} |"
        )
    return lines


def build_exec_subgraph_arrow(
    exec_seq, target_type, context_before=1, context_after=1, max_instances=3
):
    """从执行序列提取未融合算子及其上下文，生成 --> 箭头结构 + 详情表格。"""
    instances = []
    for i, op in enumerate(exec_seq):
        if op["op_type"] == target_type:
            start = max(0, i - context_before)
            end = min(len(exec_seq), i + context_after + 1)
            instances.append(exec_seq[start:end])
    if not instances:
        return ["(执行序列中未找到该算子)", ""]
    lines = []
    for inst_idx, inst in enumerate(instances[:max_instances], 1):
        chain_parts = []
        for op in inst:
            marker = "[fused]" if op.get("is_fused") else ""
            chain_parts.append(f"{op['op_type']}{marker}")
        lines.append(f"**实例{inst_idx}**: `{' --> '.join(chain_parts)}`")
        lines.append("")
        lines.append(
            "| # | Op Name | OP Type | Input Shape | Output Shape | dtype | Fused |"
        )
        lines.append(
            "|---|---------|---------|-------------|--------------|-------|:-----:|"
        )
        for j, op in enumerate(inst, 1):
            fused = "是" if op.get("is_fused") else "否"
            lines.append(
                f"| {j} | {op['op_name']} | {op['op_type']} | {op.get('input_shape', '')} | "
                f"{op.get('output_shape', '')} | {op.get('dtype', '')} | {fused} |"
            )
        lines.append("")
    return lines


# ============================================================
#  ProfilingParser — profile 子命令
# ============================================================


class ProfilingParser:
    def __init__(self, profiling_dir: str):
        self.locator = ProfilingFileLocator(profiling_dir)
        self.locator.discover()
        self.profiling_dir = profiling_dir

    def parse(self) -> Dict[str, Any]:
        result = OrderedDict()
        result["profiling_dir"] = self.profiling_dir
        result["files"] = self.locator.files
        op_stat_rows = self._load("op_statistic")
        api_stat_rows = self._load("api_statistic")
        op_summary_rows = self._load("op_summary")
        fusion_op_rows = self._load("fusion_op")
        step_trace_rows = self._load("step_trace")

        result["fusion_mode"] = detect_fusion_mode(
            op_stat_rows, api_stat_rows, fusion_op_rows
        )
        result["fusion_ops"] = self._extract_fusion_ops(op_stat_rows)
        result["hotspots"] = self._extract_hotspots(op_stat_rows)
        result["core_metrics"] = self._calc_core_metrics(
            op_stat_rows, api_stat_rows, op_summary_rows, step_trace_rows
        )
        iter_count = result["core_metrics"]["iter_count"]
        result["repeated_structures"] = self._extract_repeated_structures(
            op_summary_rows, iter_count
        )
        result["unfused_ops"] = self._extract_unfused_ops(
            op_stat_rows, result["fusion_mode"]
        )
        return result

    def _load(self, data_type):
        f = self.locator.get(data_type)
        if f:
            _, rows = read_csv_safe(f)
            return rows
        return []

    @staticmethod
    def _extract_fusion_ops(op_stat_rows):
        fusion_ops = []
        for row in op_stat_rows:
            op_type = str(row.get("OP Type", "")).strip()
            if is_fused_op_type(op_type):
                fusion_ops.append(
                    {
                        "op_type": op_type,
                        "semantic": infer_fusion_semantics(op_type),
                        "count": safe_int(row.get("Count", 0)),
                        "total_us": safe_float(row.get("Total Time(us)", 0)),
                        "avg_us": safe_float(
                            row.get("Avg Time(us)", row.get("Avg(us)", 0))
                        ),
                        "ratio_pct": safe_float(row.get("Ratio(%)", 0)),
                        "core_type": str(row.get("Core Type", "")),
                    }
                )
        fusion_ops.sort(key=lambda x: x["total_us"], reverse=True)
        return fusion_ops

    @staticmethod
    def _extract_hotspots(op_stat_rows):
        hotspots = []
        for row in op_stat_rows:
            op_type = str(row.get("OP Type", "")).strip()
            if not op_type:
                continue
            hotspots.append(
                {
                    "op_type": op_type,
                    "count": safe_int(row.get("Count", 0)),
                    "total_us": safe_float(row.get("Total Time(us)", 0)),
                    "avg_us": safe_float(
                        row.get("Avg Time(us)", row.get("Avg(us)", 0))
                    ),
                    "ratio_pct": safe_float(row.get("Ratio(%)", 0)),
                    "core_type": str(row.get("Core Type", "")),
                }
            )
        hotspots.sort(key=lambda x: x["total_us"], reverse=True)
        return hotspots

    @staticmethod
    def _extract_repeated_structures(op_summary_rows, iter_count=1):
        if not op_summary_rows:
            return []
        sample = op_summary_rows[0]
        start_col = (
            "Task Start Time(us)"
            if "Task Start Time(us)" in sample
            else ("Start Time(us)" if "Start Time(us)" in sample else "")
        )
        if not start_col:
            return []
        type_col = (
            "OP Type" if "OP Type" in sample else ("Type" if "Type" in sample else "")
        )
        dur_col = (
            "Task Duration(us)"
            if "Task Duration(us)" in sample
            else ("Duration(us)" if "Duration(us)" in sample else "")
        )
        name_col = (
            "Op Name" if "Op Name" in sample else ("Name" if "Name" in sample else "")
        )
        if not type_col or not dur_col:
            return []
        sorted_rows = sorted(
            op_summary_rows, key=lambda r: safe_float(r.get(start_col, 0))
        )
        total_ops = len(sorted_rows)
        ops_per_iter = total_ops // iter_count if iter_count > 1 else total_ops
        op_types, op_durs, op_names = [], [], []
        for row in sorted_rows[:ops_per_iter]:
            t = str(row.get(type_col, "")).strip()
            d = safe_float(row.get(dur_col, 0))
            n = str(row.get(name_col, "")).strip() if name_col else ""
            if t:
                op_types.append(t)
                op_durs.append(d)
                op_names.append(n)
        if len(op_types) < 2:
            return []
        patterns = defaultdict(list)
        for n in range(2, min(10, len(op_types) + 1)):
            for i in range(len(op_types) - n + 1):
                end = i + n
                patterns[tuple(op_types[i:end])].append(i)
        total_iter_us = sum(op_durs)
        candidates, seen_sigs = [], set()
        for ngram, indices in sorted(
            patterns.items(), key=lambda x: (len(x[0]), len(x[1])), reverse=True
        ):
            if len(indices) < 2:
                continue
            sig = frozenset(Counter(ngram).items())
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            candidates.append((ngram, indices))
        covered, results = set(), []
        for ngram, indices in candidates:
            n = len(ngram)
            non_overlapping = []
            for start in indices:
                positions = set(range(start, start + n))
                if not positions & covered:
                    non_overlapping.append(start)
                    covered |= positions
            repeat_count = len(non_overlapping)
            if repeat_count < 2:
                continue
            per_iter_us = (
                sum(
                    op_durs[i]
                    for i in range(non_overlapping[0], non_overlapping[0] + n)
                )
                * repeat_count
            )
            ratio_pct = per_iter_us / total_iter_us * 100 if total_iter_us > 0 else 0
            start = non_overlapping[0]
            end = start + n
            first_names = op_names[start:end]
            results.append(
                {
                    "pattern": " --> ".join(ngram),
                    "pattern_len": n,
                    "repeat_count": repeat_count,
                    "per_iter_us": round(per_iter_us, 1),
                    "ratio_pct": round(ratio_pct, 1),
                    "first_instance_names": first_names[:4],
                }
            )
        results.sort(key=lambda x: x["per_iter_us"], reverse=True)
        return results[:20]

    @staticmethod
    def _extract_problem_ops(op_summary_rows):
        problems = []
        for row in op_summary_rows:
            op_name = str(row.get("Op Name", "")).strip()
            op_type = str(row.get("OP Type", "")).strip()
            task_dur = safe_float(row.get("Task Duration(us)", 0))
            aiv_scalar = safe_float(row.get("aiv_scalar_ratio", 0))
            aic_scalar = safe_float(row.get("aic_scalar_ratio", 0))
            cube_util = safe_float(row.get("cube_utilization(%)", 0))
            aic_mac = safe_float(row.get("aic_mac_ratio", 0))
            issues = []
            scalar = max(aiv_scalar, aic_scalar)
            if scalar >= 0.30:
                issues.append(f"scalar_ratio>=30%({scalar:.1%})")
            if cube_util > 0 and cube_util < 20.0:
                issues.append(f"cube_util<20%({cube_util:.1f}%)")
            if aic_mac > 0 and aic_mac < 0.30:
                issues.append(f"mac_ratio<30%({aic_mac:.1%})")
            if issues:
                problems.append(
                    {
                        "op_name": op_name,
                        "op_type": op_type,
                        "task_type": str(row.get("Task Type", "")),
                        "task_dur_us": task_dur,
                        "issues": issues,
                        "input_shapes": str(row.get("Input Shapes", ""))[:60],
                        "output_shapes": str(row.get("Output Shapes", ""))[:60],
                        "aiv_vec": safe_float(row.get("aiv_vec_time(us)", 0)),
                        "aiv_mte2": safe_float(row.get("aiv_mte2_time(us)", 0)),
                        "aiv_mte3": safe_float(row.get("aiv_mte3_time(us)", 0)),
                    }
                )
        problems.sort(key=lambda x: x["task_dur_us"], reverse=True)
        return problems

    @staticmethod
    def _extract_unfused_ops(op_stat_rows, fusion_mode):
        unfused = []
        vector_types = {
            "Embedding",
            "Transpose",
            "ViewCopy",
            "ClipByValueV2",
            "Slice",
            "Concat",
            "Split",
            "Gather",
            "Cast",
            "Reshape",
            "BroadcastTo",
        }
        for row in op_stat_rows:
            op_type = str(row.get("OP Type", "")).strip()
            if is_fused_op_type(op_type):
                continue
            if op_type in vector_types or "Copy" in op_type or "Slice" in op_type:
                count = safe_int(row.get("Count", 0))
                avg = safe_float(row.get("Avg Time(us)", row.get("Avg(us)", 0)))
                if count >= 2 and avg > 0:
                    unfused.append(
                        {
                            "op_type": op_type,
                            "count": count,
                            "total_us": safe_float(row.get("Total Time(us)", 0)),
                            "avg_us": avg,
                            "core_type": str(row.get("Core Type", "")),
                        }
                    )
        unfused.sort(key=lambda x: x["total_us"], reverse=True)
        return unfused

    @staticmethod
    def _calc_core_metrics(
        op_stat_rows, api_stat_rows, op_summary_rows=None, step_trace_rows=None
    ):
        total_op_us = sum(safe_float(r.get("Total Time(us)", 0)) for r in op_stat_rows)
        total_op_count = sum(safe_int(r.get("Count", 0)) for r in op_stat_rows)
        iter_count = detect_iter_count(
            op_stat_rows, op_summary_rows, api_stat_rows, step_trace_rows
        )
        iter_wall_us = detect_iter_wall_us(step_trace_rows)
        h2d_us = 0.0
        for row in api_stat_rows:
            name = str(row.get("API Name", "")).strip()
            if "InputCopy" in name or "Memcpy" in name:
                h2d_us += safe_float(row.get("Time(us)", 0))
        h2d_per_iter = h2d_us / iter_count if iter_count > 0 else 0
        op_per_iter = total_op_us / iter_count if iter_count > 0 else 0
        op_count_per_iter = total_op_count / iter_count if iter_count > 0 else 0
        total_iter = h2d_per_iter + op_per_iter
        return {
            "total_op_us": total_op_us,
            "iter_count": iter_count,
            "h2d_per_iter_us": h2d_per_iter,
            "op_per_iter_us": op_per_iter,
            "total_iter_us": total_iter,
            "op_total_ratio": op_per_iter / total_iter * 100 if total_iter > 0 else 0,
            "iter_wall_us": iter_wall_us,
            "iter_count_source": "step_trace" if step_trace_rows else "inferred",
            "op_count_per_iter": round(op_count_per_iter, 1),
        }

    @staticmethod
    def _extract_host_overhead(api_stat_rows):
        overhead = []
        for row in api_stat_rows:
            name = str(row.get("API Name", "")).strip()
            total = safe_float(row.get("Time(us)", 0))
            count = safe_int(row.get("Count", 0))
            if total > 0 and count > 0:
                overhead.append(
                    {
                        "api_name": name,
                        "level": str(row.get("Level", "")),
                        "total_us": total,
                        "count": count,
                        "avg_us": total / count,
                    }
                )
        overhead.sort(key=lambda x: x["total_us"], reverse=True)
        return overhead[:15]


# ============================================================
#  GraphAnalyzer — graph 子命令
# ============================================================


def _shape_is_small_global(shape_str):
    if not shape_str or shape_str in ("N/A", "NA"):
        return False
    nums = re.findall(r"\d+", shape_str)
    total = 1
    for n in nums:
        v = int(n)
        if v > 0:
            total *= v
        if total > 100000:
            return False
    return total < 50000


class GraphAnalyzer:
    def __init__(self, nodes, graph_type, profiling_dir=""):
        self.graph_type = graph_type
        self.all_nodes = nodes
        self.flat_nodes = list(nodes)
        self.node_map = {n.name: n for n in self.flat_nodes}
        self._cache = {}
        self.profiling_dir = profiling_dir or ""
        self.profiling_loader = None
        if profiling_dir and os.path.isdir(profiling_dir):
            self.profiling_loader = ProfilingDataLoader(profiling_dir)
            self.profiling_loader.load()
        self.trivial_types = get_trivial_types(graph_type)
        self._populate_shape_dtype()

    def _populate_shape_dtype(self):
        for node in self.flat_nodes:
            if node.output_shapes:
                node.shape_str = node.output_shapes[0] if node.output_shapes else ""
            if node.output_dtypes:
                node.dtype_str = node.output_dtypes[0] if node.output_dtypes else ""
            if (
                not node.shape_str
                and not node.dtype_str
                and self.profiling_loader
                and self.profiling_loader.loaded
            ):
                shape, dtype = self.profiling_loader.get_shape_dtype(
                    node.name, node.op_type
                )
                if shape:
                    node.shape_str = shape
                if dtype:
                    node.dtype_str = dtype

    def analyze_op_distribution(self):
        counter = defaultdict(int)
        for node in self.flat_nodes:
            counter[node.op_type] += 1
        return dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))

    def analyze_topology(self):
        children, in_degree = self._build_adj()
        root_nodes = [n.name for n in self.flat_nodes if in_degree.get(n.name, 0) == 0]
        leaf_nodes = [n.name for n in self.flat_nodes if not children.get(n.name)]
        max_depth, max_width, processed = self._bfs_depth_width(children, in_degree)
        avg_branch = round(
            sum(len(v) for v in children.values()) / max(len(children), 1), 2
        )
        return {
            "total_nodes": len(self.flat_nodes),
            "max_depth": max_depth,
            "max_width": max_width,
            "avg_branching_factor": avg_branch,
            "root_count": len(root_nodes),
            "leaf_count": len(leaf_nodes),
            "has_cycle": processed < len(self.flat_nodes),
        }

    def find_repeated_subgraphs(self, min_len=2, max_len=20, min_repeat=3):
        topo = self._topological_sort()
        compute_entries = [
            (name, self.node_map[name].op_type)
            for name in topo
            if name in self.node_map
            and is_compute_op(self.node_map[name].op_type, self.trivial_types)
        ]
        compute_names = [e[0] for e in compute_entries]
        op_seq = [e[1] for e in compute_entries]
        all_patterns = defaultdict(list)
        actual_max = min(max_len, len(op_seq))
        for n in range(min_len, actual_max + 1):
            for i in range(len(op_seq) - n + 1):
                end = i + n
                ngram = tuple(op_seq[i:end])
                window = compute_names[i:end]
                if n == 1 or self._is_connected(window):
                    all_patterns[ngram].append((i, i + n))
        candidates = {k: v for k, v in all_patterns.items() if len(v) >= min_repeat}
        maximal = self._select_maximal(candidates)
        has_prof = bool(self.profiling_loader and self.profiling_loader.loaded)
        results = []
        for ngram, occs in maximal.items():
            results.append(self._build_result(ngram, occs, compute_names, has_prof))
        results.sort(key=lambda x: x["sort_score"], reverse=True)
        return results

    def find_unfused_segments(self):
        topo = self._topological_sort()
        compute = [
            (name, self.node_map[name])
            for name in topo
            if name in self.node_map
            and is_compute_op(self.node_map[name].op_type, self.trivial_types)
            and self.node_map[name].op_type not in CUBE_OP_TYPES
        ]
        adj = self._build_adjacency()
        reverse_adj = defaultdict(set)
        for src, dsts in adj.items():
            for dst in dsts:
                reverse_adj[dst].add(src)
        segments = []
        used = set()

        i = 0
        while i < len(compute):
            if compute[i][0] not in used:
                j = i
                seg_names = [compute[i][0]]
                seg_cats = [
                    classify_op_for_fusion(compute[i][1].op_type, compute[i][1].name)
                ]
                while j + 1 < len(compute) and compute[j + 1][0] not in used:
                    candidate = compute[j + 1]
                    cand_name, cand_node = candidate[0], candidate[1]
                    cand_cat = classify_op_for_fusion(cand_node.op_type, cand_node.name)
                    connected = any(
                        cand_name in adj.get(sn, set())
                        or sn in adj.get(cand_name, set())
                        for sn in seg_names
                    )
                    if not connected:
                        break
                    can_fuse = any(
                        can_vertically_fuse(seg_cats[k], cand_cat)
                        or can_vertically_fuse(cand_cat, seg_cats[k])
                        for k in range(len(seg_cats))
                    )
                    if not can_fuse:
                        break
                    seg_names.append(cand_name)
                    seg_cats.append(cand_cat)
                    j += 1
                if len(seg_names) >= 2:
                    has_unfused = any(
                        not self.node_map[sn].is_fused for sn in seg_names
                    )
                    if has_unfused:
                        for sn in seg_names:
                            used.add(sn)
                        self._append_segment(segments, seg_names)
                i = j + 1
            else:
                i += 1

        consumer_groups = defaultdict(list)
        for name, node in compute:
            if name not in used:
                for dst in adj.get(name, set()):
                    consumer_groups[dst].append(name)
        for consumer, producers in consumer_groups.items():
            available = [p for p in producers if p not in used]
            if len(available) >= 2:
                cats = [
                    classify_op_for_fusion(
                        self.node_map[p].op_type, self.node_map[p].name
                    )
                    for p in available
                ]
                first_cat = cats[0]
                if all(can_horizontally_fuse(first_cat, c) for c in cats[1:]):
                    for p in available:
                        used.add(p)
                    self._append_segment(segments, available)

        segments.sort(
            key=lambda x: x["ratio_pct"] if x["ratio_pct"] > 0 else x["total_us"],
            reverse=True,
        )
        return segments

    def _append_segment(self, segments, seg_names):
        seg_nodes = [(name, self.node_map[name]) for name in seg_names]
        has_prof = bool(self.profiling_loader and self.profiling_loader.loaded)
        total_us = 0.0
        if has_prof:
            for name, node in seg_nodes:
                total_us += self.profiling_loader.get_duration(name, node.op_type)
        iter_us = (
            self.profiling_loader.total_op_time_us
            if has_prof and self.profiling_loader.total_op_time_us > 0
            else 0
        )
        ratio_pct = (total_us / iter_us * 100) if iter_us > 0 and total_us > 0 else 0
        fused_flags = [node.is_fused for _, node in seg_nodes]
        has_fused = any(fused_flags)
        has_unfused = not all(fused_flags)
        if has_fused and has_unfused:
            seg_type = "再融合(混合)"
        elif has_fused:
            seg_type = "再融合(已融合间)"
        else:
            seg_type = "可融未融"
        segments.append(
            {
                "nodes": [name for name, _ in seg_nodes],
                "op_types": [node.op_type for _, node in seg_nodes],
                "shapes": [node.shape_str for _, node in seg_nodes],
                "dtypes": [node.dtype_str for _, node in seg_nodes],
                "input_shapes": [_node_input_shape_str(node) for _, node in seg_nodes],
                "is_fused_flags": fused_flags,
                "seg_type": seg_type,
                "length": len(seg_names),
                "total_us": round(total_us, 2),
                "ratio_pct": round(ratio_pct, 1),
            }
        )

    def find_fusion_degradation(self):
        if not self.profiling_loader or not self.profiling_loader.loaded:
            return []
        type_avg = self.profiling_loader.op_type_avg_map
        if not type_avg:
            return []
        results = []
        for node in self.flat_nodes:
            if not node.is_fused:
                continue
            fused_dur = self.profiling_loader.get_duration(node.name, node.op_type)
            if fused_dur <= 0:
                continue
            constituents = self._infer_constituents(node.name, node.op_type)
            if not constituents:
                continue
            constituent_total = 0.0
            constituent_details = []
            constituent_blockdims = []
            for ct in constituents:
                avg = type_avg.get(ct, 0.0)
                constituent_total += avg
                c_bd = self.profiling_loader.op_type_blockdim_map.get(ct, 0)
                constituent_blockdims.append(c_bd)
                constituent_details.append(
                    {"op_type": ct, "avg_us": round(avg, 2), "block_dim": c_bd}
                )
            if constituent_total <= 0:
                continue
            if fused_dur > constituent_total:
                repeat_count = self._count_similar_fused(node)
                degradation = (fused_dur - constituent_total) * repeat_count
                iter_us = self.profiling_loader.total_op_time_us
                ratio_pct = degradation / iter_us * 100 if iter_us > 0 else 0
                timing = self.profiling_loader.get_timing_breakdown(
                    node.name, node.op_type
                )
                cause = self._analyze_degradation_cause(
                    fused_dur, timing, constituent_blockdims
                )
                results.append(
                    {
                        "fused_name": node.name,
                        "fused_op_type": node.op_type,
                        "fused_dur_us": round(fused_dur, 2),
                        "constituents": constituent_details,
                        "constituent_total_us": round(constituent_total, 2),
                        "degradation_us": round(fused_dur - constituent_total, 2),
                        "repeat_count": repeat_count,
                        "degradation_impact_us": round(degradation, 2),
                        "ratio_pct": round(ratio_pct, 1),
                        "timing": timing,
                        "cause": cause,
                    }
                )
        results.sort(key=lambda x: x["degradation_impact_us"], reverse=True)
        return results

    def _collect_asc_graph_data(self):
        """收集所有融合算子对应的 asc_graph/asc_kernel 解析结果。

        dump_root 即 profiling_dir（autofused_* 目录为其子目录）。
        FX 图中融合算子名存在 op_type（如 autofused_abs_xxxhash），name 仅为 __expr_N。
        """
        if not self.profiling_dir:
            return {}
        fused_names = []
        for n in self.flat_nodes:
            if n.is_fused:
                cand = (
                    n.op_type
                    if n.op_type and n.op_type.startswith("autofused_")
                    else n.name
                )
                if cand:
                    fused_names.append(cand)
        if not fused_names:
            return {}
        return AscGraphParser.collect_for_fused_ops(self.profiling_dir, fused_names)

    @staticmethod
    def _generate_code_path_hints(degradation):
        """根据劣化主导因素生成需联网获取的代码路径提示（供 Agent webfetch）。"""
        if not degradation:
            return []
        hints = []
        seen_keys = set()
        for d in degradation:
            cause = d.get("cause", {})
            cat = cause.get("category", "")
            dominant = cause.get("dominant", "")
            cause_type = cause.get("cause_type", "")
            key = (cause_type, dominant)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            torpair_url = (
                "https://gitcode.com/Ascend/torchair/tree/master/experimental/"
                "_inductor_npu_ext/python/inductor_npu_ext"
            )
            if cause_type == "block":
                hints.append(
                    {
                        "issue": f"核数选错（{cat}）",
                        "code_paths": [
                            "本地 `asc_kernel.py` → 融合算子实际实现（schedule 模板，由 `_asc_codegen.py` 生成）",
                            "TorchAir `_inductor_npu_ext/codegen/_asc_codegen.py` → tiling func 与 schedule 模板生成",
                            "TorchAir `_inductor_npu_ext/npu.py` → `NPUScheduling.can_fuse_vertical/horizontal`（融合规则）",
                            "TorchAir `_inductor_npu_ext/config.py`（融合开关与 debug option）",
                        ],
                        "repo": torpair_url,
                    }
                )
            elif cause_type == "template":
                if dominant in ("aiv_mte2", "aiv_mte3"):
                    hints.append(
                        {
                            "issue": f"模板选错-Vector读写瓶颈（{dominant}）",
                            "code_paths": [
                                "TorchAir `_inductor_npu_ext/npu.py` → codegen 模板选择逻辑",
                                "TorchAir `_inductor_npu_ext/lowering/` → NPU 特殊 lowering（shape 相关模板）",
                                "TorchAir `_inductor_npu_ext/common/asc_graph.py` → AF graph 输出（核对融合范围）",
                            ],
                            "repo": torpair_url,
                        }
                    )
                elif dominant == "aiv_vec":
                    hints.append(
                        {
                            "issue": f"模板选错-Vector计算瓶颈（{dominant}）",
                            "code_paths": [
                                "TorchAir `_inductor_npu_ext/npu.py` → codegen 计算表达式生成",
                                "TorchAir `_inductor_npu_ext/lowering/` → 算子 lowering（exp/sigmoid/pow 等软件展开）",
                            ],
                            "repo": torpair_url,
                        }
                    )
                elif any(
                    k in cat
                    for k in (
                        "cube_utilization",
                        "aic_mac",
                        "aic_scalar",
                        "aic_mte",
                        "aic_fixpipe",
                    )
                ):
                    hints.append(
                        {
                            "issue": f"模板选错-CV/CC瓶颈（{cat}）",
                            "code_paths": [
                                "TorchAir `_inductor_npu_ext/npu.py` → `can_fuse_vertical/horizontal`（CV/CC 融合判断）",
                                "TorchAir `_inductor_npu_ext/lowering/` → cube 模板 lowering",
                            ],
                            "repo": torpair_url,
                        }
                    )
        return hints

    @staticmethod
    def _infer_constituents(node_name, op_type):
        raw = (
            node_name
            if any(node_name.startswith(p) for p in FUSED_PREFIXES)
            else op_type
        )
        for prefix in FUSED_PREFIXES:
            if raw.startswith(prefix):
                prefix_len = len(prefix)
                raw = raw[prefix_len:]
                break
        if raw.startswith("fused_"):
            raw = raw[FUSED_PREFIX_LEN:]
        hash_match = re.search(r"_([0-9a-f]{32})$", raw)
        if hash_match:
            raw = raw[: hash_match.start()]
        parts = re.split(r"[_/]", raw)
        known_ops = {
            "Cast",
            "ClipByValue",
            "ClipByValueV2",
            "Add",
            "Sub",
            "Mul",
            "Div",
            "DivNoNan",
            "RealDiv",
            "Sigmoid",
            "Relu",
            "Tanh",
            "Softmax",
            "SoftmaxV2",
            "Gelu",
            "GeluV2",
            "ReduceMax",
            "ReduceMin",
            "ReduceSum",
            "ReduceMean",
            "Equal",
            "Greater",
            "Less",
            "Select",
            "SelectV2",
            "Concat",
            "ConcatD",
            "Split",
            "Gather",
            "GatherV2",
            "Transpose",
            "Reshape",
            "Abs",
            "Exp",
            "Log",
            "Log1p",
            "Sqrt",
            "Rsqrt",
            "Pow",
            "Maximum",
            "Minimum",
            "Neg",
            "Reciprocal",
            "Square",
            "Slice",
            "StridedSlice",
            "Pack",
            "Unpack",
            "ZerosLike",
            "OnesLike",
            "Range",
            "Where",
            "NonZero",
            "Erf",
            "Erfc",
            "Floor",
            "Ceil",
            "Round",
            "Sign",
            "FloorDiv",
            "FloorMod",
            "FusedMulAdd",
            "BatchMatMul",
            "MatMul",
            "MatMulV3",
            "SoftmaxGrad",
            "ReduceAll",
            "SqrtGrad",
            "ReluGrad",
            "SigmoidGrad",
            "TanhGrad",
        }
        merged = []
        i = 0
        while i < len(parts):
            for end in range(min(i + 4, len(parts)), i, -1):
                candidate = "".join(parts[i:end])
                if candidate in known_ops:
                    merged.append(candidate)
                    i = end
                    break
            else:
                i += 1
        return merged if merged else []

    def _count_similar_fused(self, node):
        count = 0
        for n in self.flat_nodes:
            if n.is_fused and n.op_type == node.op_type:
                count += 1
        return max(count, 1)

    @staticmethod
    def _analyze_degradation_cause(fused_dur, timing, constituent_blockdims=None):
        if not timing or fused_dur <= 0:
            return {
                "category": "未知",
                "detail": "无 op_summary 耗时分解数据",
                "dominant": "",
            }
        wait = timing.get("wait", 0)
        aiv_total = timing.get("aiv_total", 0)
        block_dim = timing.get("block_dim", 0)
        in_shape = timing.get("in_shape", "")
        out_shape = timing.get("out_shape", "")
        causes = []
        cause_type = "template"
        has_cube_metric = any(
            timing.get(k, 0) > 0
            for k in (
                "aic_mac_ratio",
                "aic_scalar_ratio",
                "aic_mte1_ratio",
                "aic_mte2_ratio",
                "aic_mte3_ratio",
                "aic_fixpipe_ratio",
            )
        )
        if has_cube_metric:
            cube_util = timing.get("cube_utilization", 0)
            aic_mac = timing.get("aic_mac_ratio", 0)
            aic_scalar = timing.get("aic_scalar_ratio", 0)
            aic_mte1 = timing.get("aic_mte1_ratio", 0)
            aic_mte2 = timing.get("aic_mte2_ratio", 0)
            aic_mte3 = timing.get("aic_mte3_ratio", 0)
            aic_fix = timing.get("aic_fixpipe_ratio", 0)
            if cube_util > 0 and cube_util < 20.0:
                causes.append(
                    f"cube_utilization {cube_util:.1f}%（模板选错：shape 太小，cube 算力未充分利用）"
                )
            if (
                aic_mac > 0
                and aic_mac < 0.30
                and not (cube_util > 0 and cube_util < 20.0)
            ):
                causes.append(
                    f"aic_mac_ratio {aic_mac:.1%}（模板选错：计算非瓶颈，shape 小或 cube 模板未充分利用）"
                )
            if aic_scalar > 0.30:
                causes.append(
                    f"aic_scalar_ratio {aic_scalar:.1%}（模板选错：cube 标量开销大）"
                )
            if aic_mte1 > 0.30:
                causes.append(
                    f"aic_mte1_ratio {aic_mte1:.1%}（模板选错：L1 读搬运效率低）"
                )
            if aic_mte2 > 0.30:
                causes.append(
                    f"aic_mte2_ratio {aic_mte2:.1%}（模板选错：cube 读数据多/访问不连续）"
                )
            if aic_mte3 > 0.30:
                causes.append(
                    f"aic_mte3_ratio {aic_mte3:.1%}（模板选错：cube 写回开销大）"
                )
            if aic_fix > 0.30:
                causes.append(
                    f"aic_fixpipe_ratio {aic_fix:.1%}（模板选错：fixpipe 开销大）"
                )
        vec_ratio = timing.get("aiv_vec_ratio", 0)
        scalar_ratio = timing.get("aiv_scalar_ratio", 0)
        mte2_ratio = timing.get("aiv_mte2_ratio", 0)
        mte3_ratio = timing.get("aiv_mte3_ratio", 0)
        if mte2_ratio > 0.25:
            causes.append(
                f"aiv_mte2_ratio {mte2_ratio:.1%}（模板选错：读数据多/访问不连续/融合后重复读）"
            )
        if mte3_ratio > 0.25:
            causes.append(
                f"aiv_mte3_ratio {mte3_ratio:.1%}（模板选错：输出多/写回不连续/多输出未合并）"
            )
        if vec_ratio > 0.30:
            causes.append(f"aiv_vec_ratio {vec_ratio:.1%}（模板选错：计算表达式复杂）")
        if scalar_ratio > 0.40:
            causes.append(
                f"aiv_scalar_ratio {scalar_ratio:.1%}（模板选错：标量计算多/index 计算多）"
            )
        overhead = fused_dur - aiv_total if aiv_total > 0 else 0
        overhead_ratio = overhead / fused_dur if fused_dur > 0 else 0
        const_bd_sum = sum(b for b in (constituent_blockdims or []) if b > 0)
        const_bd_max = max(
            (b for b in (constituent_blockdims or []) if b > 0), default=0
        )
        block_causes = []
        if block_dim > 0:
            shape_str = in_shape or out_shape or "N/A"
            if const_bd_sum > 0 and block_dim < const_bd_sum:
                block_causes.append(
                    f"Block Dim {block_dim} < 融合前各算子 Block Dim 之和 {const_bd_sum}（核数选错：并行度被压缩）"
                )
            if block_dim > 0 and _shape_is_small_global(shape_str) and block_dim >= 8:
                block_causes.append(
                    f"Block Dim {block_dim} 较大但 shape 较小({shape_str})（核数选错：核数浪费，launch 开销 > 计算收益）"
                )
            if (
                const_bd_max > 0
                and block_dim == const_bd_max
                and len(constituent_blockdims or []) > 1
            ):
                block_causes.append(
                    f"融合前后 Block Dim 未变({block_dim})但融合后计算量增加（核数选错：并行度不足）"
                )
        if overhead_ratio > 0.3 and not block_causes:
            block_causes.append(
                f"非计算开销 占执行 {overhead_ratio * 100:.0f}%（launch/调度开销累积，建议增大 kernel 粒度）"
            )
        if block_causes:
            causes.extend(block_causes)
            cause_type = "block"
        if not causes:
            if aiv_total > 0:
                causes.append(
                    f"无明显主导因素（mte2_ratio {mte2_ratio:.1%}，mte3_ratio {mte3_ratio:.1%}，vec_ratio {vec_ratio:.1%}）"
                )
            else:
                causes.append("无 aiv/aic 计算数据")
        category = "、".join(c.split("（")[0] for c in causes)
        detail = "；".join(causes)
        dominant = ""
        if mte2_ratio > 0.25:
            dominant = "aiv_mte2"
        elif mte3_ratio > 0.25:
            dominant = "aiv_mte3"
        elif vec_ratio > 0.30:
            dominant = "aiv_vec"
        return {
            "category": category,
            "detail": detail,
            "dominant": dominant,
            "cause_type": cause_type,
            "wait_us": round(wait, 2),
            "overhead_ratio": round(overhead_ratio, 3),
            "block_dim": block_dim,
            "constituent_blockdim_sum": const_bd_sum,
            "constituent_blockdim_max": const_bd_max,
            "mte2_ratio": round(mte2_ratio, 3),
            "mte3_ratio": round(mte3_ratio, 3),
            "vec_ratio": round(vec_ratio, 3),
        }

    def format_report(self):
        topo = self.analyze_topology()
        subgraphs = self.find_repeated_subgraphs()
        unfused = self.find_unfused_segments()
        has_prof = bool(self.profiling_loader and self.profiling_loader.loaded)
        lines = []
        lines.append("# AutoFuse 图结构分析报告\n")
        lines.append("## 1. 基本信息\n")
        lines.append(f"- **节点总数**: {topo['total_nodes']}")
        if has_prof:
            lines.append("- **Profiling 关联**: 已加载")
            lines.append(
                f"- **算子总耗时**: {self.profiling_loader.total_op_time_us:.1f} us"
            )
        else:
            lines.append("- **Profiling 关联**: 未加载")
        lines.append(
            f"- **拓扑结构**: 最大深度 {topo['max_depth']}, 最大宽度 {topo['max_width']}, "
            f"平均分支因子 {topo['avg_branching_factor']}, 根节点 {topo['root_count']}, 叶节点 {topo['leaf_count']}"
        )
        fused_count = sum(1 for n in self.flat_nodes if n.is_fused)
        lines.append(f"- **融合算子节点**: {fused_count} / {topo['total_nodes']}")
        lines.append("")

        lines.append("## 2. 重复子图分析\n")
        if subgraphs:
            refusable = [sg for sg in subgraphs if sg.get("can_refuse", True)]
            non_refusable = [sg for sg in subgraphs if not sg.get("can_refuse", True)]
            lines.append(
                f"共 {len(subgraphs)} 种模式（可再融合 {len(refusable)}，不可再融合 {len(non_refusable)}）\n"
            )
            if has_prof:
                lines.append(
                    "| # | Pattern | Repeat | Len | 可再融合 | 首个实例 | Avg(us) | Total(us) |"
                )
                lines.append(
                    "|---|---------|--------|-----|----------|----------|---------|-----------|"
                )
            else:
                lines.append("| # | Pattern | Repeat | Len | 可再融合 | 首个实例 |")
                lines.append("|---|---------|--------|-----|----------|----------|")
            for i, sg in enumerate(subgraphs[:30], 1):
                pat = sg["pattern"][:60]
                first = sg["first_instance"][:40]
                refuse_str = "✅" if sg.get("can_refuse", True) else "❌"
                if has_prof:
                    lines.append(
                        f"| {i} | {pat} | {sg['repeat_count']} | {sg['pattern_len']} | "
                        f"{refuse_str} | {first} | {sg['avg_us']:.1f} | {sg['total_us']:.1f} |"
                    )
                else:
                    lines.append(
                        f"| {i} | {pat} | {sg['repeat_count']} | {sg['pattern_len']} | {refuse_str} | {first} |"
                    )
            if non_refusable:
                lines.append(
                    f"\n> ❌ 不可再融合的模式共 {len(non_refusable)} 种，相邻算子类别不满足垂直融合规则，已融合算子之间无法合并。\n"
                )
        else:
            lines.append("(未发现重复子图)")
        lines.append("")

        lines.append("## 3. 子图可视化 (Top 5 可再融合)\n")
        adj = self._build_adjacency()
        reverse_adj = defaultdict(set)
        for src, dsts in adj.items():
            for dst in dsts:
                reverse_adj[dst].add(src)
        vis_count = 0
        for sg in subgraphs:
            if vis_count >= 5:
                break
            if not sg.get("can_refuse", True):
                continue
            vis_count += 1
            lines.append(
                f"### 子图 {vis_count}: `{sg['pattern'][:60]}` (重复 {sg['repeat_count']} 次)\n"
            )
            if sg.get("instances"):
                inst = sg["instances"][0]
                time_str = (
                    f" (耗时 {inst['instance_time_us']:.1f}us)" if has_prof else ""
                )
                arrow_lines = build_subgraph_arrow(
                    self,
                    inst.get("node_details", []),
                    adj,
                    reverse_adj,
                    f"实例1{time_str}",
                )
                lines.extend(arrow_lines)

        lines.append("## 4. 可融但未融合 / 再融合片段（按单迭代耗时占比排序）\n")
        if unfused:
            lines.append(
                "| # | 类型 | 算子序列 | 长度 | Total(us) | 占比(%) | 算子名(融合状态) |"
            )
            lines.append(
                "|---|------|---------|------|-----------|---------|-----------------|"
            )
            for i, seg in enumerate(unfused[:20], 1):
                seq = " → ".join(seg["op_types"][:5])
                if len(seg["op_types"]) > 5:
                    seq += " ..."
                fused_flags = seg.get("is_fused_flags", [False] * len(seg["nodes"]))
                name_flags = [
                    f"{n}{'[已融]' if f else ''}"
                    for n, f in zip(seg["nodes"][:3], fused_flags[:3])
                ]
                names = ", ".join(name_flags)
                if len(seg["nodes"]) > 3:
                    names += " ..."
                t_str = f"{seg['total_us']:.1f}" if seg["total_us"] > 0 else "N/A"
                r_str = f"{seg['ratio_pct']:.1f}" if seg["ratio_pct"] > 0 else "N/A"
                seg_type = seg.get("seg_type", "可融未融")
                lines.append(
                    f"| {i} | {seg_type} | {seq} | {seg['length']} | {t_str} | {r_str} | {names} |"
                )
            lines.append("")
            lines.append("### 可融但未融合 / 再融合片段可视化 (Top 3)\n")
            for i, seg in enumerate(unfused[:3], 1):
                seg_type = seg.get("seg_type", "可融未融")
                lines.append(
                    f"**片段 {i}** [{seg_type}] ({seg['length']} 算子, {seg['total_us']:.1f}us):\n"
                )
                seg_node_details = []
                fused_flags = seg.get("is_fused_flags", [False] * len(seg["nodes"]))
                input_shapes = seg.get("input_shapes", [""] * len(seg["nodes"]))
                for j, (name, op_type) in enumerate(zip(seg["nodes"], seg["op_types"])):
                    shape = (
                        seg.get("shapes", [""] * len(seg["nodes"]))[j]
                        if j < len(seg.get("shapes", []))
                        else ""
                    )
                    dtype = (
                        seg.get("dtypes", [""] * len(seg["nodes"]))[j]
                        if j < len(seg.get("dtypes", []))
                        else ""
                    )
                    seg_node_details.append(
                        {
                            "name": name,
                            "op_type": op_type,
                            "shape": shape,
                            "dtype": dtype,
                            "is_fused": fused_flags[j]
                            if j < len(fused_flags)
                            else False,
                            "input_shape": input_shapes[j]
                            if j < len(input_shapes)
                            else "",
                        }
                    )
                arrow_lines = build_subgraph_arrow(
                    self, seg_node_details, adj, reverse_adj, f"{seg_type}片段{i}"
                )
                lines.extend(arrow_lines)
        else:
            lines.append("(未检测到可融但未融合片段)")
        lines.append("")

        lines.append("## 5. AutoFuse 融合劣化片段\n")
        degradation = self.find_fusion_degradation()
        asc_graph_cache = self._collect_asc_graph_data()
        if degradation:
            lines.append(
                "| # | 融合算子 | 融合后耗时(us) | 融合前算子 | 融合前耗时和(us) | 劣化(us) | 重复次数 | 劣化影响(us) | 占比(%) | 劣化原因 |"
            )
            lines.append(
                "|---|---------|--------------|-----------|----------------|---------|"
                "----------|-------------|---------|---------|"
            )
            for i, d in enumerate(degradation[:20], 1):
                consts = " + ".join(
                    f"{c['op_type']}({c['avg_us']})" for c in d["constituents"]
                )
                cause_cat = d.get("cause", {}).get("category", "未知")
                lines.append(
                    f"| {i} | {d['fused_name'][:40]} | {d['fused_dur_us']} | {consts} | "
                    f"{d['constituent_total_us']} | {d['degradation_us']} | {d['repeat_count']} | "
                    f"{d['degradation_impact_us']} | {d['ratio_pct']} | {cause_cat} |"
                )
            lines.append("")
            lines.append("### 融合劣化片段详情 (Top 3)\n")
            for i, d in enumerate(degradation[:3], 1):
                lines.append(
                    f"#### 劣化片段 {i}: `{d['fused_name'][:50]}` (劣化影响 {d['degradation_impact_us']}us)\n"
                )
                cause = d.get("cause", {})
                timing = d.get("timing")
                lines.append(f"- **劣化原因**: {cause.get('detail', '未知')}\n")
                if timing:
                    dur = d["fused_dur_us"]
                    aiv_total = timing.get("aiv_total", 0)
                    wait = timing.get("wait", 0)
                    bd = timing.get("block_dim", 0)
                    const_bd_sum = d.get("cause", {}).get("constituent_blockdim_sum", 0)
                    lines.append(
                        f"- **耗时分解** (执行 {dur}us, 等待 {wait:.1f}us, Block Dim={bd}):"
                    )
                    if aiv_total > 0:
                        lines.append(
                            f"  - aiv_vec(计算): {timing.get('aiv_vec', 0):.2f} us "
                            f"(aiv_vec_ratio={timing.get('aiv_vec_ratio', 0):.1%})"
                        )
                        lines.append(
                            f"  - aiv_scalar: {timing.get('aiv_scalar', 0):.2f} us "
                            f"(aiv_scalar_ratio={timing.get('aiv_scalar_ratio', 0):.1%})"
                        )
                        lines.append(
                            f"  - aiv_mte2(读搬运): {timing.get('aiv_mte2', 0):.2f} us "
                            f"(aiv_mte2_ratio={timing.get('aiv_mte2_ratio', 0):.1%})"
                        )
                        lines.append(
                            f"  - aiv_mte3(写回): {timing.get('aiv_mte3', 0):.2f} us "
                            f"(aiv_mte3_ratio={timing.get('aiv_mte3_ratio', 0):.1%})"
                        )
                        overhead = dur - aiv_total
                        lines.append(
                            f"  - 非计算开销(launch等): {overhead:.2f} us (占执行 {overhead / dur * 100:.0f}%)"
                        )
                    else:
                        lines.append("  - (无 aiv 计算分解数据)")
                    if any(
                        timing.get(k, 0) > 0
                        for k in (
                            "aic_mac_ratio",
                            "aic_scalar_ratio",
                            "aic_mte1_ratio",
                            "aic_mte2_ratio",
                            "aic_mte3_ratio",
                            "aic_fixpipe_ratio",
                        )
                    ):
                        lines.append(
                            f"  - aic_mac_ratio={timing.get('aic_mac_ratio', 0):.1%}, "
                            f"aic_scalar_ratio={timing.get('aic_scalar_ratio', 0):.1%}, "
                            f"aic_mte1_ratio={timing.get('aic_mte1_ratio', 0):.1%}, "
                            f"aic_mte2_ratio={timing.get('aic_mte2_ratio', 0):.1%}, "
                            f"aic_mte3_ratio={timing.get('aic_mte3_ratio', 0):.1%}, "
                            f"aic_fixpipe_ratio={timing.get('aic_fixpipe_ratio', 0):.1%}, "
                            f"cube_utilization={timing.get('cube_utilization', 0):.1f}%"
                        )
                    if const_bd_sum > 0:
                        lines.append(
                            f"  - 融合前后 Block Dim 对比: 融合后={bd}, 融合前各算子之和={const_bd_sum}"
                        )
                    if wait > dur and wait > 1:
                        lines.append(
                            f"  - **注意**: Wait {wait:.1f}us > 执行 {dur:.1f}us"
                            "（排队/同步，需结合 Block Dim+shape 评估是否核数问题，不可单凭 Wait 判定）"
                        )
                    lines.append("")
                else:
                    lines.append("- **耗时分解**: (无 op_summary 数据)\n")
                ag_entry = asc_graph_cache.get(d["fused_name"])
                if ag_entry and ag_entry.get("asc_graph"):
                    ag = ag_entry["asc_graph"]
                    ak = ag_entry.get("asc_kernel") or {}
                    atens = ", ".join(ag.get("original_aten", [])) or "N/A"
                    n_load = ag.get("load_count", 0)
                    n_store = ag.get("store_count", 0)
                    n_inputs = len(ag.get("input_tensors", []))
                    n_ops = len(ag.get("ops_calls", []))
                    lines.append(
                        f"- **AscGraph 融合范围**: Original ATen=[{atens}], 输入tensor={n_inputs}个, ops调用={n_ops}个"
                    )
                    lines.append(f"  - load 次数={n_load}, store 次数={n_store}")
                    if n_inputs > 0 and n_load > n_inputs:
                        lines.append(
                            f"  - **注意**: load({n_load}) > 输入tensor数({n_inputs})，存在重复读，与 aiv_mte2_ratio 偏高一致"
                        )
                    if n_store > 1:
                        lines.append(
                            f"  - **注意**: store 次数={n_store} > 1，多输出写回，与 aiv_mte3_ratio 偏高一致"
                        )
                    feats = ak.get("compute_features", []) if ak else []
                    if feats:
                        lines.append(
                            f"  - asc_kernel 计算特征: {', '.join(feats)}（软件展开，与 aiv_vec_ratio 偏高相关）"
                        )
                    lines.append("")
                fnode = self.node_map.get(d["fused_name"])
                fused_detail = {
                    "name": d["fused_name"],
                    "op_type": d["fused_op_type"],
                    "shape": fnode.shape_str if fnode else "",
                    "dtype": fnode.dtype_str if fnode else "",
                    "is_fused": True,
                    "input_shape": _node_input_shape_str(fnode) if fnode else "",
                }
                const_details = [
                    {
                        "name": c["op_type"],
                        "op_type": c["op_type"],
                        "shape": "",
                        "dtype": "",
                        "is_fused": False,
                        "input_shape": "",
                    }
                    for c in d["constituents"]
                ]
                arrow_lines = build_subgraph_arrow(
                    self,
                    const_details + [fused_detail],
                    adj,
                    reverse_adj,
                    f"融合劣化片段{i}",
                )
                lines.extend(arrow_lines)
        else:
            lines.append("(未检测到融合劣化片段)")
        lines.append("")

        if asc_graph_cache:
            lines.append("## 6. AscGraph 融合范围分析\n")
            lines.append(
                f"共解析 {len(asc_graph_cache)} 个融合算子的 asc_graph.py / asc_kernel.py（dump 目录: {self.profiling_dir}）\n"
            )
            lines.append(
                "| # | 融合算子 | 融合范围(箭头) | load | store | 输入 | ops | kernel特征 |"
            )
            lines.append(
                "|---|---------|---------------|------|-------|-----|-----|-----------|"
            )
            for i, (fname, entry) in enumerate(asc_graph_cache.items(), 1):
                ag = entry.get("asc_graph") or {}
                ak = entry.get("asc_kernel") or {}
                atens = ag.get("original_aten", [])
                arrow = " --> ".join(atens) + " [fused]" if atens else "N/A"
                feats = ", ".join(ak.get("compute_features", [])) or "-"
                lines.append(
                    f"| {i} | {fname[:40]} | {arrow} | {ag.get('load_count', 0)} | "
                    f"{ag.get('store_count', 0)} | {len(ag.get('input_tensors', []))} | "
                    f"{len(ag.get('ops_calls', []))} | {feats} |"
                )
            lines.append("")
            show_detail = list(asc_graph_cache.items())[:3]
            if show_detail:
                lines.append("### AscGraph 详情 (Top 3)\n")
                for i, (fname, entry) in enumerate(show_detail, 1):
                    ag = entry.get("asc_graph") or {}
                    ak = entry.get("asc_kernel") or {}
                    lines.append(f"#### {i}. `{fname[:50]}`\n")
                    atens = ag.get("original_aten", [])
                    arrow = " --> ".join(atens) + " [fused]" if atens else "N/A"
                    lines.append(f"- **融合范围**: `{arrow}`")
                    lines.append(f"- **Original ATen**: [{', '.join(atens) or 'N/A'}]")
                    lines.append(
                        f"- **load/store**: {ag.get('load_count', 0)} / {ag.get('store_count', 0)}"
                    )
                    lines.append(
                        f"- **输入tensor**: {', '.join(ag.get('input_tensors', [])) or 'N/A'}"
                    )
                    lines.append(
                        f"- **ops调用**: {', '.join(ag.get('ops_calls', [])[:8]) or 'N/A'}"
                    )
                    src_nodes = ag.get("source_nodes", [])
                    if src_nodes:
                        lines.append(f"- **Source Nodes**: {', '.join(src_nodes[:10])}")
                    frags = ag.get("graph_fragments", [])
                    if frags:
                        lines.append("- **Graph fragment**:")
                        lines.append("```")
                        for frag in frags[:2]:
                            for ln in frag[:8]:
                                lines.append(ln)
                        lines.append("```")
                    if ak:
                        feats = ak.get("compute_features", [])
                        lines.append(
                            f"- **asc_kernel**: 计算特征={feats or 'N/A'}, "
                            f"标量loop={'有' if ak.get('has_scalar_loop') else '无'}, "
                            f"AscendC API={'有' if ak.get('has_ascendc_api') else '无'}"
                        )
                    lines.append("")
            lines.append("")

        hints = self._generate_code_path_hints(degradation)
        if hints:
            lines.append("## 7. 需联网获取的代码路径提示\n")
            lines.append(
                "> 以下为基于劣化分析推断的代码路径，Agent 可联网获取最新代码做进一步定位。\n"
            )
            lines.append("| # | 劣化问题 | 建议查看的代码路径 | 代码仓 |")
            lines.append("|---|---------|------------------|-------|")
            for i, h in enumerate(hints, 1):
                paths = "<br>".join(h["code_paths"])
                lines.append(f"| {i} | {h['issue']} | {paths} | [仓库]({h['repo']}) |")
            lines.append("")
            lines.append("**参考链接**:")
            lines.append(
                "- TorchAir 代码仓: https://gitcode.com/Ascend/torchair/tree/master/"
                "experimental/_inductor_npu_ext/python/inductor_npu_ext"
            )
            lines.append(
                "- Inductor 代码仓: https://gitcode.com/Ascend/pytorch/tree/master/torch_npu/_inductor"
            )
            lines.append("")

        return "\n".join(lines)

    def _build_adj(self):
        children = defaultdict(list)
        in_degree = defaultdict(int)
        for node in self.flat_nodes:
            in_degree.setdefault(node.name, 0)
            for inp in node.inputs:
                if not inp:
                    continue
                src = inp.split(":")[0] if self.graph_type == "pbtxt" else inp
                if src in self.node_map:
                    children[src].append(node.name)
                    in_degree[node.name] += 1
        return children, in_degree

    @staticmethod
    def _bfs_depth_width(children, in_degree):
        queue = deque()
        depth = {}
        for name, deg in in_degree.items():
            if deg == 0:
                queue.append(name)
                depth[name] = 0
        max_d = max_w = 0
        width_d = defaultdict(int)
        processed = 0
        while queue:
            cur = queue.popleft()
            d = depth.get(cur, 0)
            max_d = max(max_d, d)
            width_d[d] += 1
            for child in children.get(cur, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    depth[child] = d + 1
                    queue.append(child)
            processed += 1
        for cnt in width_d.values():
            max_w = max(max_w, cnt)
        return max_d, max_w, processed

    def _topological_sort(self):
        children, in_degree = self._build_adj()
        queue = deque(
            [n.name for n in self.flat_nodes if in_degree.get(n.name, 0) == 0]
        )
        result = []
        while queue:
            cur = queue.popleft()
            result.append(cur)
            for child in children.get(cur, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return result

    def _is_connected(self, node_names):
        if len(node_names) <= 1:
            return True
        node_set = set(node_names)
        adj = self._build_adjacency()
        internal_edges = 0
        for nn in node_names:
            internal_edges += len(adj.get(nn, set()) & node_set)
        compute_ops = [
            nn
            for nn in node_names
            if is_compute_op(self.node_map[nn].op_type, self.trivial_types)
        ]
        if internal_edges >= len(compute_ops) - 1:
            return True
        return self._has_shared_consumer(node_names, adj)

    def _has_shared_consumer(self, node_names, adj):
        consumer_sets = []
        for nn in node_names:
            consumers = adj.get(nn, set())
            if consumers:
                consumer_sets.append(consumers)
        if len(consumer_sets) < 2:
            return False
        common = consumer_sets[0]
        for cs in consumer_sets[1:]:
            common = common & cs
        return len(common) >= 1

    def _build_adjacency(self):
        cached = self._cache.get("adjacency")
        if cached is not None:
            return cached
        adj = defaultdict(set)
        for node in self.flat_nodes:
            for inp in node.inputs:
                if not inp:
                    continue
                src = inp.split(":")[0] if self.graph_type == "pbtxt" else inp
                if src in self.node_map:
                    adj[src].add(node.name)
        adj = self._add_transitive_edges(adj)
        cached = dict(adj)
        self._cache["adjacency"] = cached
        return cached

    def _add_transitive_edges(self, adj):
        trivial = self.trivial_types
        compute_names = set()
        for n in self.flat_nodes:
            if is_compute_op(n.op_type, trivial):
                compute_names.add(n.name)
        for src_name in list(compute_names):
            visited = set()
            queue = deque(adj.get(src_name, set()))
            while queue:
                mid = queue.popleft()
                if mid in visited:
                    continue
                visited.add(mid)
                mid_node = self.node_map.get(mid)
                if mid_node and is_compute_op(mid_node.op_type, trivial):
                    adj[src_name].add(mid)
                elif mid_node and mid_node.op_type in trivial:
                    queue.extend(adj.get(mid, set()))
        return adj

    @staticmethod
    def _select_maximal(candidates):
        by_len = defaultdict(list)
        for ngram, occ in candidates.items():
            by_len[len(ngram)].append((ngram, occ))
        deduped = {}
        for length in sorted(by_len.keys(), reverse=True):
            group = sorted(by_len[length], key=lambda x: len(x[1]), reverse=True)
            kept_sigs = []
            for ngram, occ in group:
                sig = frozenset(Counter(ngram).items())
                if not any(sig == ks for ks in kept_sigs):
                    deduped[ngram] = occ
                    kept_sigs.append(sig)
        maximal = {}
        covered = []
        for ngram, occs in sorted(
            deduped.items(), key=lambda x: len(x[0]), reverse=True
        ):
            n = len(ngram)
            uncovered = [
                (s, e)
                for s, e in occs
                if not any(cs <= s and ce >= s + n for cs, ce in covered)
            ]
            if len(uncovered) >= 2:
                maximal[ngram] = uncovered
                for s, _ in uncovered:
                    covered.append((s, s + n))
        return maximal

    def _build_result(self, ngram, occs, compute_names, has_prof):
        instances = []
        times = []
        for s, e in occs:
            node_names = compute_names[s:e]
            t = 0.0
            if has_prof:
                for nn in node_names:
                    node = self.node_map.get(nn)
                    if node:
                        t += self.profiling_loader.get_duration(nn, node.op_type)
            times.append(t)
            details = []
            for nn in node_names:
                node = self.node_map.get(nn)
                if node:
                    details.append(
                        {
                            "name": node.name,
                            "op_type": node.op_type,
                            "is_fused": node.is_fused,
                            "source_line": node.source_line,
                            "raw_call": node.raw_call,
                            "shape": node.shape_str,
                            "dtype": node.dtype_str,
                            "input_shape": _node_input_shape_str(node),
                        }
                    )
            instances.append(
                {
                    "nodes": node_names,
                    "instance_time_us": round(t, 2),
                    "node_details": details,
                }
            )
        rc = len(occs)
        avg_t = sum(times) / rc if times else 0
        total_t = avg_t * rc
        sep = " → "
        cats = [
            classify_op_for_fusion(
                t,
                self.node_map[compute_names[occs[0][0] + k]].name
                if compute_names[occs[0][0] + k] in self.node_map
                else "",
            )
            for k, t in enumerate(ngram)
        ]
        can_refuse = True
        for k in range(len(cats) - 1):
            if not can_vertically_fuse(cats[k], cats[k + 1]):
                can_refuse = False
                break
        has_fused_in_pattern = any(
            self.node_map[compute_names[occs[0][0] + k]].is_fused
            if compute_names[occs[0][0] + k] in self.node_map
            else False
            for k in range(len(ngram))
        )
        return {
            "pattern": sep.join(ngram),
            "op_types": list(ngram),
            "pattern_len": len(ngram),
            "repeat_count": rc,
            "instances": instances[:20],
            "avg_us": round(avg_t, 2),
            "total_us": round(total_t, 2),
            "sort_score": round(total_t if has_prof else len(ngram) * rc, 2),
            "first_instance": instances[0]["nodes"][0] if instances else "",
            "can_refuse": can_refuse,
            "categories": cats,
            "has_fused": has_fused_in_pattern,
        }


# ============================================================
#  迭代次数检测 — 优先 step_trace 行数；否则融合算子名/非aclnn名众数 > Synchronize + op_statistic Count 校验
# ============================================================


def _iter_count_by_name_mode(op_summary_rows):
    """基于 op_summary 算子名众数推断迭代次数（无 step_trace 时使用，比 Synchronize API 更准）。

    依据：GE prof 的非 aclnn 开头 op_name、Inductor 的融合算子 name 基本单迭代唯一，
    故其出现次数 = 迭代次数。取所有此类名字出现次数的中位数众数，避免极值干扰。
    """
    if not op_summary_rows:
        return 0
    is_kd = "Name" in (op_summary_rows[0] or {}) and "Accelerator Core" in (
        op_summary_rows[0] or {}
    )
    name_col = "Name" if is_kd else "Op Name"
    type_col = "Type" if is_kd else "OP Type"
    from collections import Counter

    name_counter = Counter()
    fused_counter = Counter()
    for row in op_summary_rows:
        op_name = str(row.get(name_col, "")).strip()
        op_type = str(row.get(type_col, "")).strip()
        if not op_name:
            continue
        if is_fused_op(op_type) or is_fused_op(op_name):
            fused_counter[op_name] += 1
        elif not op_name.startswith("aclnn"):
            name_counter[op_name] += 1
    candidates = []
    for cnt in (fused_counter, name_counter):
        for op_name, c in cnt.items():
            if c >= 1:
                candidates.append(c)
    if not candidates:
        return 0
    candidates.sort()
    mid = len(candidates) // 2
    mode_val = (
        candidates[mid]
        if len(candidates) % 2 == 1
        else (candidates[mid - 1] + candidates[mid]) // 2
    )
    return max(mode_val, 1)


def detect_iter_count(
    op_statistic_rows,
    op_summary_rows=None,
    api_statistic_rows=None,
    step_trace_rows=None,
) -> int:
    """检测迭代次数。

    优先级 1: step_trace 行数（GE Profiling 通常有此文件，每行对应一次迭代，最准确）。
    优先级 2: op_summary 算子名众数——GE prof 非 aclnn 开头 op_name、Inductor 融合算子 name
    基本单迭代唯一，其出现次数 = 迭代次数；比 Synchronize API 更准。
    优先级 3: Synchronize API count 作为候选，用 op_statistic 中 OP Type Count 最小值校验
    （迭代数不应超过 min Count），再用 op_summary 行数整除性验证。
    """
    if step_trace_rows:
        n = len(step_trace_rows)
        if n > 0:
            return n
    name_mode = _iter_count_by_name_mode(op_summary_rows)
    if name_mode >= 1:
        stat_min = 0
        for row in op_statistic_rows or []:
            c = safe_int(row.get("Count", 0))
            if c > 0 and (stat_min == 0 or c < stat_min):
                stat_min = c
        if stat_min > 0 and name_mode <= stat_min:
            return name_mode
        if not op_statistic_rows:
            return name_mode
    sync_count = 0
    for row in api_statistic_rows or []:
        name = str(row.get("API Name", "")).strip()
        if "Synchronize" in name or name.startswith("aclrtSync"):
            c = safe_int(row.get("Count", 0))
            if c > sync_count:
                sync_count = c
    stat_min = 0
    for row in op_statistic_rows or []:
        c = safe_int(row.get("Count", 0))
        if c > 0 and (stat_min == 0 or c < stat_min):
            stat_min = c
    best = sync_count if sync_count > 1 else (name_mode if name_mode >= 1 else 1)
    if stat_min > 0 and best > stat_min:
        for c in range(stat_min, 0, -1):
            if op_summary_rows and len(op_summary_rows) % c == 0:
                best = c
                break
        else:
            best = stat_min
    if op_summary_rows and len(op_summary_rows) > 0 and best > 0:
        total = len(op_summary_rows)
        if total % best != 0:
            for c in range(min(best, total), 0, -1):
                if total % c == 0 and (stat_min == 0 or c <= stat_min):
                    best = c
                    break
    return max(best, 1)


def detect_iter_wall_us(step_trace_rows) -> Optional[float]:
    """从 step_trace 计算「每迭代平均耗时」平均值（us）。

    仅 GE Profiling 有 step_trace 文件时可用；Inductor 模式返回 None。
    """
    if not step_trace_rows:
        return None
    times = []
    for row in step_trace_rows:
        v = safe_float(row.get("Iteration Time(us)", 0))
        if v > 0:
            times.append(v)
    if not times:
        return None
    return sum(times) / len(times)


# ============================================================
#  FusionDataLoader — compare 子命令单组 Profiling 数据加载
# ============================================================


class FusionDataLoader:
    """加载一个 Profiling 目录的融合相关数据"""

    def __init__(self, profiling_dir: str, label: str = ""):
        self.profiling_dir = profiling_dir
        self.label = label
        self.op_statistic: List[Dict[str, str]] = []
        self.op_summary: List[Dict[str, str]] = []
        self.api_statistic: List[Dict[str, str]] = []
        self.step_trace: List[Dict[str, str]] = []
        self.fusion_ops: List[Dict[str, Any]] = []
        self.non_fusion_ops: List[Dict[str, Any]] = []
        self.total_fusion_us: float = 0.0
        self.total_non_fusion_us: float = 0.0
        self.total_op_count: int = 0
        self.iter_count: int = 1
        self.iter_wall_us: Optional[float] = None
        self.fusion_mode: str = "unknown"
        self.fusion_mode_label: str = "未知"
        self._op_summary_is_kernel_details: bool = False
        self._op_summary_cols: List[str] = []

    def load(self):
        self._load_op_statistic()
        self._load_op_summary()
        self._load_api_statistic()
        self._load_step_trace()
        self._classify_ops()
        self._detect_mode()
        self._infer_iter_count()

    def _find_file(self, patterns):
        for pattern in patterns:
            matches = glob.glob(
                os.path.join(self.profiling_dir, "**", pattern), recursive=True
            )
            if matches:
                return matches[0]
        return None

    def _load_op_statistic(self):
        f = self._find_file(["*op_statistic*.csv", "op_statistic.csv"])
        if f:
            _, self.op_statistic = read_csv_safe(f)

    def _load_op_summary(self):
        f = self._find_file(["*op_summary*.csv", "kernel_details.csv"])
        if f:
            self._op_summary_cols, self.op_summary = read_csv_safe(f)
            self._op_summary_is_kernel_details = "Name" in self._op_summary_cols

    def _load_api_statistic(self):
        f = self._find_file(["api_statistic_*.csv", "api_statistic.csv"])
        if f:
            _, self.api_statistic = read_csv_safe(f)

    def _load_step_trace(self):
        f = self._find_file(["step_trace_*.csv", "step_trace_time.csv"])
        if f:
            _, self.step_trace = read_csv_safe(f)

    def _classify_ops(self):
        for row in self.op_statistic:
            op_type = str(row.get("OP Type", "")).strip()
            count = safe_int(row.get("Count", 0))
            total = safe_float(row.get("Total Time(us)", 0))
            avg = safe_float(row.get("Avg Time(us)", row.get("Avg(us)", 0)))
            ratio = safe_float(row.get("Ratio(%)", 0))
            entry = {
                "op_type": op_type,
                "semantic": infer_semantic(op_type),
                "count": count,
                "total_us": total,
                "avg_us": avg,
                "ratio_pct": ratio,
                "core_type": str(row.get("Core Type", "")),
            }
            if is_fused_op_type(op_type):
                self.fusion_ops.append(entry)
                self.total_fusion_us += total
            else:
                self.non_fusion_ops.append(entry)
                self.total_non_fusion_us += total
            self.total_op_count += count
        self.fusion_ops.sort(key=lambda x: x["total_us"], reverse=True)
        self.non_fusion_ops.sort(key=lambda x: x["total_us"], reverse=True)

    def _detect_mode(self):
        has_autofuse = any(
            get_fusion_prefix(op["op_type"]) == "autofuse" for op in self.fusion_ops
        )
        has_triton = any(
            get_fusion_prefix(op["op_type"]) == "triton" for op in self.fusion_ops
        )
        has_dvm = any(
            get_fusion_prefix(op["op_type"]) == "dvm" for op in self.fusion_ops
        )
        has_me = any(
            "ModelExecute" in str(r.get("API Name", "")) for r in self.api_statistic
        )
        if has_triton:
            self.fusion_mode = "inductor_triton"
            self.fusion_mode_label = "Inductor + Triton"
        elif has_dvm:
            self.fusion_mode = "inductor_dvm"
            self.fusion_mode_label = "Inductor + DVM"
        elif has_autofuse and has_me:
            self.fusion_mode = "ge_autofuse"
            self.fusion_mode_label = "GE + AutoFuse"
        elif has_autofuse:
            self.fusion_mode = "inductor_ascendc"
            self.fusion_mode_label = "Inductor + AutoFuse(AscendC)"

    def _infer_iter_count(self):
        self.iter_count = detect_iter_count(
            self.op_statistic, self.op_summary, self.api_statistic, self.step_trace
        )
        self.iter_wall_us = detect_iter_wall_us(self.step_trace)

    def get_op_summary_detail(self, op_type: str) -> List[Dict[str, Any]]:
        """获取 op_summary 中指定 OP Type 的详细记录（含占比 ratio / Block Dim / shape）"""
        is_kd = self._op_summary_is_kernel_details
        name_col = "Name" if is_kd else "Op Name"
        type_col = "Type" if is_kd else "OP Type"
        dur_col = "Duration(us)" if is_kd else "Task Duration(us)"
        results = []
        for row in self.op_summary:
            if str(row.get(type_col, "")).strip() == op_type:
                results.append(
                    {
                        "op_name": str(row.get(name_col, "")),
                        "task_dur": safe_float(row.get(dur_col, 0)),
                        "aiv_vec": safe_float(row.get("aiv_vec_time(us)", 0)),
                        "aiv_mte2": safe_float(row.get("aiv_mte2_time(us)", 0)),
                        "aiv_mte3": safe_float(row.get("aiv_mte3_time(us)", 0)),
                        "aiv_vec_ratio": safe_float(row.get("aiv_vec_ratio", 0)),
                        "aiv_mte2_ratio": safe_float(row.get("aiv_mte2_ratio", 0)),
                        "aiv_mte3_ratio": safe_float(row.get("aiv_mte3_ratio", 0)),
                        "aic_mac_ratio": safe_float(row.get("aic_mac_ratio", 0)),
                        "cube_utilization": safe_float(
                            row.get("cube_utilization(%)", 0)
                        ),
                        "block_dim": safe_int(
                            row.get("Block Dim", row.get("block_dim", 0))
                        ),
                        "in_shape": str(row.get("Input Shapes", "")).strip().strip('"'),
                        "out_shape": str(row.get("Output Shapes", ""))
                        .strip()
                        .strip('"'),
                    }
                )
        return results

    def get_semantic_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """按语义推断分组融合算子"""
        sem_map = defaultdict(list)
        for op in self.fusion_ops:
            sem_map[op["semantic"]].append(op)
        return dict(sem_map)

    def get_exec_sequence(self, iter_limit: int = 1) -> List[Dict[str, Any]]:
        """获取 op_summary 按时间排序的执行序列（前 N 个迭代）"""
        if not self.op_summary:
            return []
        is_kd = self._op_summary_is_kernel_details
        name_col = "Name" if is_kd else "Op Name"
        type_col = "Type" if is_kd else "OP Type"
        dur_col = "Duration(us)" if is_kd else "Task Duration(us)"
        start_col = "Start Time(us)" if is_kd else "Task Start Time(us)"
        sorted_rows = sorted(
            self.op_summary, key=lambda r: safe_float(r.get(start_col, 0))
        )
        total_ops = len(sorted_rows)
        ops_per_iter = (
            total_ops // max(self.iter_count, 1) if self.iter_count > 1 else total_ops
        )
        limit = ops_per_iter * iter_limit
        result = []
        for row in sorted_rows[:limit]:
            op_type = str(row.get(type_col, "")).strip()
            result.append(
                {
                    "op_name": str(row.get(name_col, "")),
                    "op_type": op_type,
                    "dur_us": safe_float(row.get(dur_col, 0)),
                    "is_fused": is_fused_op_type(op_type),
                    "semantic": infer_semantic(op_type)
                    if is_fused_op_type(op_type)
                    else op_type,
                }
            )
        return result


# ============================================================
#  语义匹配
# ============================================================


def semantic_similarity(sem_a: str, sem_b: str) -> float:
    """计算两个语义字符串的相似度（基于关键词重叠）"""
    words_a = set(sem_a.lower().replace("_", " ").split())
    words_b = set(sem_b.lower().replace("_", " ").split())
    if not words_a or not words_b:
        return 0.0
    overlap = words_a & words_b
    union = words_a | words_b
    return len(overlap) / len(union)


def match_unfused_to_fused(
    unfused_type: str, fused_semantics: List[str]
) -> Optional[str]:
    """将未融合算子类型与融合语义列表匹配（排除 cube 类算子）"""
    if unfused_type in CUBE_OP_TYPES:
        return None
    unfused_lower = unfused_type.lower()
    for sem in fused_semantics:
        sem_lower = sem.lower()
        if unfused_lower in sem_lower or sem_lower in unfused_lower:
            return sem
    keywords = {
        "embedding": ["embedding"],
        "softmax": ["softmax"],
        "pack": ["stack"],
        "concat": ["cat", "concat"],
        "slice": ["slice"],
        "relu": ["relu"],
        "sigmoid": ["sigmoid"],
        "add": ["add"],
    }
    unfused_key = unfused_lower.replace("v2", "").replace("d", "").strip()
    for key, syns in keywords.items():
        if key in unfused_key:
            for sem in fused_semantics:
                for syn in syns:
                    if syn in sem.lower():
                        return sem
    return None


# ============================================================
#  对比分析
# ============================================================


class FusionComparator:
    def __init__(
        self,
        data_a: FusionDataLoader,
        data_b: FusionDataLoader,
        label_a: str,
        label_b: str,
    ):
        self.a = data_a
        self.b = data_b
        self.label_a = label_a
        self.label_b = label_b

    def compare_hotspots(self) -> List[Dict[str, Any]]:
        a_map = {
            str(r.get("OP Type", "")).strip(): r
            for r in self.a.op_statistic
            if str(r.get("OP Type", "")).strip()
        }
        b_map = {
            str(r.get("OP Type", "")).strip(): r
            for r in self.b.op_statistic
            if str(r.get("OP Type", "")).strip()
        }
        all_types = set(a_map.keys()) | set(b_map.keys())
        results = []
        for op_type in all_types:
            a_op = a_map.get(op_type, {})
            b_op = b_map.get(op_type, {})
            a_total = safe_float(a_op.get("Total Time(us)", 0))
            b_total = safe_float(b_op.get("Total Time(us)", 0))
            a_avg = safe_float(a_op.get("Avg Time(us)", a_op.get("Avg(us)", 0)))
            b_avg = safe_float(b_op.get("Avg Time(us)", b_op.get("Avg(us)", 0)))
            a_fused = is_fused_op_type(op_type) and op_type in a_map
            b_fused = is_fused_op_type(op_type) and op_type in b_map
            a_in_b_fused = False
            if not a_fused and op_type in a_map:
                b_fused_semantics = list(self.b.get_semantic_map().keys())
                a_in_b_fused = (
                    match_unfused_to_fused(op_type, b_fused_semantics) is not None
                )
            if a_fused and b_fused and a_avg > b_avg:
                status = "融合劣化"
            elif a_fused and b_fused and a_avg <= b_avg:
                status = "融合正常"
            elif not a_fused and a_in_b_fused:
                status = "未融合(可优化)"
            elif not a_fused and not a_in_b_fused:
                status = "未融合(无对比)"
            elif a_fused and not b_fused:
                status = "仅A融合"
            else:
                status = "其他"
            results.append(
                {
                    "op_type": op_type,
                    "a_count": safe_int(a_op.get("Count", 0)),
                    "a_total": a_total,
                    "a_avg": a_avg,
                    "b_count": safe_int(b_op.get("Count", 0)),
                    "b_total": b_total,
                    "b_avg": b_avg,
                    "diff_total": b_total - a_total,
                    "diff_pct": ((b_total - a_total) / a_total * 100)
                    if a_total > 0
                    else 0,
                    "is_fused": is_fused_op_type(op_type),
                    "a_fused": a_fused,
                    "b_fused": b_fused,
                    "status": status,
                }
            )
        results.sort(key=lambda x: abs(x["diff_total"]), reverse=True)
        return results

    def find_af_unfused_vs_b_fused(self) -> List[Dict[str, Any]]:
        """找出模式A未融合但模式B已融合的算子（可融但未融机会）"""
        a_unfused = {op["op_type"]: op for op in self.a.non_fusion_ops}
        b_fused_semantics = list(self.b.get_semantic_map().keys())
        b_sem_map = self.b.get_semantic_map()
        results = []
        for op_type, a_op in a_unfused.items():
            matched_sem = match_unfused_to_fused(op_type, b_fused_semantics)
            if matched_sem:
                b_ops = b_sem_map.get(matched_sem, [])
                b_total = sum(op["total_us"] for op in b_ops)
                b_count = sum(op["count"] for op in b_ops)
                a_per_iter = a_op["total_us"] / max(self.a.iter_count, 1)
                b_per_iter = b_total / max(self.b.iter_count, 1)
                iter_impact = a_per_iter - b_per_iter
                results.append(
                    {
                        "a_op_type": op_type,
                        "a_count": a_op["count"],
                        "a_total_us": a_op["total_us"],
                        "a_avg_us": a_op["avg_us"],
                        "a_per_iter_us": round(a_per_iter, 1),
                        "b_fused_semantic": matched_sem,
                        "b_fused_types": [op["op_type"][:50] for op in b_ops],
                        "b_count": b_count,
                        "b_total_us": b_total,
                        "b_per_iter_us": round(b_per_iter, 1),
                        "iter_impact_us": round(iter_impact, 1),
                        "iter_impact_pct": round(iter_impact / (a_per_iter) * 100, 1)
                        if a_per_iter > 0
                        else 0,
                    }
                )
        results.sort(key=lambda x: x["iter_impact_us"], reverse=True)
        return results

    def find_same_scope_degradation(self) -> List[Dict[str, Any]]:
        """相同融合范围下的性能对比，找出模式A劣化的kernel"""
        a_sem = self.a.get_semantic_map()
        b_sem = self.b.get_semantic_map()
        common_sem = set(a_sem.keys()) & set(b_sem.keys())
        results = []
        for sem in common_sem:
            a_ops = a_sem[sem]
            b_ops = b_sem[sem]
            for a_op in a_ops:
                for b_op in b_ops:
                    a_details = self.a.get_op_summary_detail(a_op["op_type"])
                    b_details = self.b.get_op_summary_detail(b_op["op_type"])
                    if a_details and b_details:
                        a_avg_dur = sum(d["task_dur"] for d in a_details) / len(
                            a_details
                        )
                        b_avg_dur = sum(d["task_dur"] for d in b_details) / len(
                            b_details
                        )
                        a_avg_vec = sum(
                            d.get("aiv_vec_ratio", 0) for d in a_details
                        ) / len(a_details)
                        b_avg_vec = sum(
                            d.get("aiv_vec_ratio", 0) for d in b_details
                        ) / len(b_details)
                        a_avg_mte2 = sum(
                            d.get("aiv_mte2_ratio", 0) for d in a_details
                        ) / len(a_details)
                        b_avg_mte2 = sum(
                            d.get("aiv_mte2_ratio", 0) for d in b_details
                        ) / len(b_details)
                        a_avg_mte3 = sum(
                            d.get("aiv_mte3_ratio", 0) for d in a_details
                        ) / len(a_details)
                        b_avg_mte3 = sum(
                            d.get("aiv_mte3_ratio", 0) for d in b_details
                        ) / len(b_details)
                        a_avg_bd = sum(d.get("block_dim", 0) for d in a_details) / len(
                            a_details
                        )
                        b_avg_bd = sum(d.get("block_dim", 0) for d in b_details) / len(
                            b_details
                        )
                        dur_diff = a_avg_dur - b_avg_dur
                        dur_pct = (dur_diff / b_avg_dur * 100) if b_avg_dur > 0 else 0
                        a_is_slower = a_avg_dur > b_avg_dur
                        a_per_iter = (
                            a_avg_dur * a_op["count"] / max(self.a.iter_count, 1)
                        )
                        b_per_iter = (
                            b_avg_dur * b_op["count"] / max(self.b.iter_count, 1)
                        )
                        iter_impact = a_per_iter - b_per_iter
                        results.append(
                            {
                                "semantic": sem,
                                "a_type": a_op["op_type"][:50],
                                "b_type": b_op["op_type"][:50],
                                "a_dur": round(a_avg_dur, 2),
                                "b_dur": round(b_avg_dur, 2),
                                "dur_diff": round(dur_diff, 2),
                                "dur_pct": round(dur_pct, 1),
                                "a_is_slower": a_is_slower,
                                "a_vec": round(a_avg_vec, 2),
                                "b_vec": round(b_avg_vec, 2),
                                "a_mte2": round(a_avg_mte2, 2),
                                "b_mte2": round(b_avg_mte2, 2),
                                "a_mte3": round(a_avg_mte3, 2),
                                "b_mte3": round(b_avg_mte3, 2),
                                "a_block_dim": round(a_avg_bd, 1),
                                "b_block_dim": round(b_avg_bd, 1),
                                "a_count": a_op["count"],
                                "b_count": b_op["count"],
                                "a_per_iter_us": round(a_per_iter, 1),
                                "b_per_iter_us": round(b_per_iter, 1),
                                "iter_impact_us": round(iter_impact, 1),
                                "bottleneck": self._identify_bottleneck(
                                    a_avg_vec,
                                    a_avg_mte2,
                                    a_avg_mte3,
                                    b_avg_vec,
                                    b_avg_mte2,
                                    b_avg_mte3,
                                ),
                            }
                        )
        results.sort(key=lambda x: x["iter_impact_us"], reverse=True)
        return results

    @staticmethod
    def _identify_bottleneck(a_vec, a_mte2, a_mte3, b_vec, b_mte2, b_mte3) -> str:
        """瓶颈判定：基于占比 ratio 差值（非绝对耗时），哪个维度差异最大即该维度为瓶颈"""
        vec_diff = a_vec - b_vec
        mte2_diff = a_mte2 - b_mte2
        mte3_diff = a_mte3 - b_mte3
        max_diff = max(vec_diff, mte2_diff, mte3_diff)
        if max_diff == vec_diff and vec_diff > 0.1:
            return "vec计算"
        if max_diff == mte2_diff and mte2_diff > 0.1:
            return "mte2搬运(读)"
        if max_diff == mte3_diff and mte3_diff > 0.1:
            return "mte3搬运(写)"
        return "无明显差异"

    def find_degradation_segments(self) -> List[Dict[str, Any]]:
        if not self.a.op_statistic or not self.b.op_statistic:
            return []
        a_map = {
            str(r.get("OP Type", "")).strip(): r
            for r in self.a.op_statistic
            if str(r.get("OP Type", "")).strip()
        }
        b_map = {
            str(r.get("OP Type", "")).strip(): r
            for r in self.b.op_statistic
            if str(r.get("OP Type", "")).strip()
        }
        degradations = []
        all_types = set(a_map.keys()) | set(b_map.keys())
        for op_type in all_types:
            a_op = a_map.get(op_type, {})
            b_op = b_map.get(op_type, {})
            a_avg = safe_float(a_op.get("Avg Time(us)", a_op.get("Avg(us)", 0)))
            b_avg = safe_float(b_op.get("Avg Time(us)", b_op.get("Avg(us)", 0)))
            if a_avg > 0 and b_avg > 0:
                pct = (b_avg - a_avg) / a_avg * 100
                if pct > 10:
                    degradations.append(
                        {
                            "op_type": op_type,
                            "a_avg": a_avg,
                            "b_avg": b_avg,
                            "diff_pct": round(pct, 1),
                            "is_fused": is_fused_op_type(op_type),
                        }
                    )
        degradations.sort(key=lambda x: x["diff_pct"], reverse=True)
        return degradations

    def find_unfused_opportunities(self) -> List[Dict[str, Any]]:
        results = []
        for loader, label in [(self.a, self.label_a), (self.b, self.label_b)]:
            for op in loader.non_fusion_ops:
                count = op["count"]
                avg = op["avg_us"]
                total = op["total_us"]
                if count >= 2 and avg > 0 and avg < 5.0:
                    results.append(
                        {
                            "source": label,
                            "op_type": op["op_type"],
                            "count": count,
                            "avg_us": avg,
                            "total_us": total,
                        }
                    )
        results.sort(key=lambda x: x["total_us"], reverse=True)
        return results


# ============================================================
#  自身能力分析（从执行序列提取可融未融结构）
# ============================================================


def _extract_self_unfused_structures(
    exec_seq: List[Dict[str, Any]], iter_count: int
) -> List[Dict[str, Any]]:
    """从执行序列提取可融但未融合的重复结构（>=2 算子），按单迭代耗时占比排序。

    仅保留最长的不重叠模式，避免短模式重复计算长模式的算子。
    """
    if not exec_seq or len(exec_seq) < 2:
        return []
    unfused_seq = [
        (e["op_type"], e["dur_us"], e["op_name"])
        for e in exec_seq
        if not e.get("is_fused", False) and e["op_type"] not in CUBE_OP_TYPES
    ]
    if len(unfused_seq) < 2:
        return []
    op_types = [t for t, _, _ in unfused_seq]
    op_durs = [d for _, d, _ in unfused_seq]
    op_names = [n for _, _, n in unfused_seq]
    patterns = defaultdict(list)
    for n in range(2, min(10, len(op_types) + 1)):
        for i in range(len(op_types) - n + 1):
            end = i + n
            ngram = tuple(op_types[i:end])
            patterns[ngram].append(i)
    total_iter_us = sum(op_durs)
    candidates = []
    seen_sigs = set()
    for ngram, indices in sorted(
        patterns.items(), key=lambda x: (len(x[0]), len(x[1])), reverse=True
    ):
        if len(indices) < 2:
            continue
        sig = frozenset(Counter(ngram).items())
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        candidates.append((ngram, indices))
    covered = set()
    results = []
    for ngram, indices in candidates:
        n = len(ngram)
        non_overlapping = []
        for start in indices:
            positions = set(range(start, start + n))
            if not positions & covered:
                non_overlapping.append(start)
                covered |= positions
        repeat_count = len(non_overlapping)
        if repeat_count < 2:
            continue
        per_iter_us = (
            sum(op_durs[i] for i in range(non_overlapping[0], non_overlapping[0] + n))
            * repeat_count
        )
        ratio_pct = per_iter_us / total_iter_us * 100 if total_iter_us > 0 else 0
        start = non_overlapping[0]
        end = start + n
        first_names = op_names[start:end]
        results.append(
            {
                "pattern": " --> ".join(ngram),
                "pattern_len": n,
                "op_types": list(ngram),
                "repeat_count": repeat_count,
                "per_iter_us": round(per_iter_us, 1),
                "ratio_pct": round(ratio_pct, 1),
                "first_names": first_names[:4],
            }
        )
    results.sort(key=lambda x: x["per_iter_us"], reverse=True)
    return results[:15]


# ============================================================
#  报告生成 — profile 子命令
# ============================================================


def generate_profile_report(data: Dict[str, Any]) -> str:
    lines = []
    fm = data["fusion_mode"]
    lines.append("# 融合性能分析报告\n")
    lines.append("## 1. 融合模式识别\n")
    lines.append(f"- **融合模式**: {fm['mode_label']}")
    if fm.get("evidence"):
        for e in fm["evidence"]:
            lines.append(f"  - {e}")
    lines.append(f"- **融合算子种类数**: {len(data.get('fusion_ops', []))}")
    lines.append(f"- **Profiling 目录**: {data.get('profiling_dir', '')}")
    files = data.get("files", {})
    found = {dt: p for dt, p in files.items() if p}
    if found:
        file_str = "; ".join(f"{dt}={os.path.basename(p)}" for dt, p in found.items())
        lines.append(f"- **已发现文件**: {file_str}")
    missing = [dt for dt, p in files.items() if not p]
    if missing:
        lines.append(f"- **未找到文件**: {', '.join(missing)}")
    lines.append("")

    lines.append("## 2. 核心指标汇总\n")
    cm = data["core_metrics"]
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    lines.append(
        f"| 迭代次数 | {cm['iter_count']}（来源: {cm['iter_count_source']}） |"
    )
    lines.append(f"| 算子总耗时 | {cm['total_op_us']:.1f} us |")
    lines.append(f"| 每迭代算子耗时(算子耗时和) | {cm['op_per_iter_us']:.1f} us |")
    lines.append(f"| 每迭代算子数 | {cm.get('op_count_per_iter', 0)} |")
    lines.append(f"| 每迭代H2D耗时 | {cm['h2d_per_iter_us']:.1f} us |")
    lines.append(f"| 每迭代总耗时(算子+H2D) | {cm['total_iter_us']:.1f} us |")
    if cm.get("iter_wall_us") is not None:
        lines.append(f"| 每迭代平均耗时(step_trace) | {cm['iter_wall_us']:.1f} us |")
    lines.append("")

    lines.append("## 3. 融合算子清单\n")
    fusion_ops = data.get("fusion_ops", [])
    if fusion_ops:
        lines.append(
            "| OP Type | 语义推断 | Count | Total(us) | Avg(us) | Ratio(%) | Core Type |"
        )
        lines.append(
            "|---------|----------|-------|-----------|---------|----------|-----------|"
        )
        for op in fusion_ops:
            lines.append(
                f"| {op['op_type'][:50]} | {op['semantic']} | {op['count']} | "
                f"{op['total_us']:.1f} | {op['avg_us']:.1f} | {op['ratio_pct']:.1f} | {op['core_type']} |"
            )
    else:
        lines.append("(未检测到融合算子)")
    lines.append("")

    lines.append("## 4. 热点算子（按 Ratio 降序，累计 >= 80%）\n")
    lines.append("| # | OP Type | Count | Total(us) | Avg(us) | Ratio(%) | 累计(%) |")
    lines.append("|---|---------|-------|-----------|---------|----------|---------|")
    cum_ratio = 0.0
    for i, op in enumerate(data.get("hotspots", []), 1):
        cum_ratio += op["ratio_pct"]
        lines.append(
            f"| {i} | {op['op_type'][:50]} | {op['count']} | "
            f"{op['total_us']:.1f} | {op['avg_us']:.1f} | {op['ratio_pct']:.1f} | {cum_ratio:.1f} |"
        )
        if cum_ratio >= 80.0:
            break
    lines.append("")

    lines.append("## 5. Top 重复结构（>=2 算子，按单迭代耗时占比排序）\n")
    repeated = data.get("repeated_structures", [])
    if repeated:
        lines.append(
            "| # | Pattern | 算子数 | 重复次数 | 单迭代耗时(us) | 占比(%) | 首个实例 |"
        )
        lines.append(
            "|---|---------|--------|----------|---------------|---------|----------|"
        )
        for i, r in enumerate(repeated[:20], 1):
            first = ", ".join(r["first_instance_names"][:2])[:40]
            lines.append(
                f"| {i} | {r['pattern'][:55]} | {r['pattern_len']} | "
                f"{r['repeat_count']} | {r['per_iter_us']:.1f} | {r['ratio_pct']:.1f} | {first} |"
            )
    else:
        lines.append("(未检测到重复结构)")
    lines.append("")

    lines.append("## 6. 可融但未融合算子\n")
    unfused = data.get("unfused_ops", [])
    if unfused:
        lines.append("| OP Type | Count | Total(us) | Avg(us) | Core Type |")
        lines.append("|---------|-------|-----------|---------|-----------|")
        for op in unfused[:20]:
            lines.append(
                f"| {op['op_type']} | {op['count']} | "
                f"{op['total_us']:.1f} | {op['avg_us']:.1f} | {op['core_type']} |"
            )
    else:
        lines.append("(未检测到可融但未融合算子)")
    lines.append("")

    return "\n".join(lines)


# ============================================================
#  报告生成 — compare 子命令（cross-mode）
# ============================================================


def generate_cross_mode_report(comparator: FusionComparator) -> str:
    a, b = comparator.a, comparator.b
    la, lb = comparator.label_a, comparator.label_b
    lines = []
    lines.append(f"# 融合模式对比分析报告（基线: {la}）\n")
    lines.append("## 1. 数据与口径\n")
    lines.append("| 项 | 基线模式 | 对比模式 |")
    lines.append("|---|---------|---------|")
    lines.append(f"| 标签 | {la} | {lb} |")
    lines.append(f"| 融合模式 | {a.fusion_mode_label} | {b.fusion_mode_label} |")
    lines.append(f"| 融合算子种类 | {len(a.fusion_ops)} | {len(b.fusion_ops)} |")
    lines.append(
        f"| 融合算子总耗时 | {a.total_fusion_us:.1f} us | {b.total_fusion_us:.1f} us |"
    )
    lines.append(
        f"| 非融合算子总耗时 | {a.total_non_fusion_us:.1f} us | {b.total_non_fusion_us:.1f} us |"
    )
    a_total_us = a.total_fusion_us + a.total_non_fusion_us
    b_total_us = b.total_fusion_us + b.total_non_fusion_us
    lines.append(f"| 算子总耗时 | {a_total_us:.1f} us | {b_total_us:.1f} us |")
    lines.append(f"| 迭代次数 | {a.iter_count} | {b.iter_count} |")
    a_per_iter = a_total_us / max(a.iter_count, 1)
    b_per_iter = b_total_us / max(b.iter_count, 1)
    lines.append(
        f"| 每迭代算子耗时(算子耗时和) | {a_per_iter:.1f} us | {b_per_iter:.1f} us |"
    )
    a_op_count_per = a.total_op_count / max(a.iter_count, 1)
    b_op_count_per = b.total_op_count / max(b.iter_count, 1)
    lines.append(f"| 每迭代算子数 | {a_op_count_per:.1f} | {b_op_count_per:.1f} |")
    lines.append(f"| 每迭代差异 | {a_per_iter - b_per_iter:.1f} us | - |")
    if a.iter_wall_us is not None or b.iter_wall_us is not None:
        aw = f"{a.iter_wall_us:.1f} us" if a.iter_wall_us is not None else "N/A"
        bw = f"{b.iter_wall_us:.1f} us" if b.iter_wall_us is not None else "N/A"
        lines.append(f"| 每迭代平均耗时(step_trace) | {aw} | {bw} |")
    lines.append("")

    lines.append("## 2. AutoFuse 可融但未融合\n")

    lines.append("### 2.1 能力对比分析（AutoFuse 未融合但对比模式已融合）\n")
    unfused_vs_fused = comparator.find_af_unfused_vs_b_fused()
    if unfused_vs_fused:
        lines.append(
            f"以下算子在 {la} 中未融合，但在 {lb} 中已被融合为 fused kernel，是 AutoFuse 的优化机会：\n"
        )
        lines.append(
            f"| # | 基线未融合算子 | 基线 Count | 基线 Avg(us) | 基线/迭代(us) | "
            f"{lb} 融合范围 | {lb} Count | {lb}/迭代(us) | 迭代影响(us) | 迭代影响(%) |"
        )
        lines.append(
            "|---|---------------|-----------|-------------|-------------|------------|"
            "-----------|------------|-------------|------------|"
        )
        total_iter_impact = 0.0
        for i, u in enumerate(unfused_vs_fused, 1):
            total_iter_impact += u["iter_impact_us"]
            b_fused_short = u["b_fused_semantic"][:30]
            lines.append(
                f"| {i} | {u['a_op_type']} | {u['a_count']} | {u['a_avg_us']:.2f} | "
                f"{u['a_per_iter_us']:.1f} | {b_fused_short} | {u['b_count']} | "
                f"{u['b_per_iter_us']:.1f} | {u['iter_impact_us']:.1f} | {u['iter_impact_pct']:.1f}% |"
            )
        impact_pct = total_iter_impact / a_per_iter * 100 if a_per_iter > 0 else None
        impact_pct_str = f"{impact_pct:.1f}%" if impact_pct is not None else "N/A"
        lines.append(
            f"\n**能力对比分析对迭代总影响**: {total_iter_impact:.1f} us/iter "
            f"（占基线每迭代 {impact_pct_str}）\n"
        )
    else:
        lines.append(f"(未检测到 {la} 可融但未融但 {lb} 已融的算子)\n")

    if unfused_vs_fused:
        lines.append("#### 2.1 子图结构（--> 箭头）\n")
        a_exec = a.get_exec_sequence(iter_limit=1)
        for u in unfused_vs_fused[:8]:
            lines.append(
                f"##### {u['a_op_type']}（基线未融，{lb} 融为 `{u['b_fused_semantic']}`）\n"
            )
            arrow_lines = build_exec_subgraph_arrow(
                a_exec,
                u["a_op_type"],
                context_before=1,
                context_after=1,
                max_instances=3,
            )
            lines.extend(arrow_lines)

    lines.append("### 2.2 AutoFuse 自身能力分析（从执行序列提取可融未融结构）\n")
    a_exec_seq = a.get_exec_sequence(iter_limit=1)
    self_analysis = _extract_self_unfused_structures(a_exec_seq, a.iter_count)
    if self_analysis:
        lines.append(
            f"从 {la} 的 op_summary 执行序列中检测到以下可融但未融合的重复结构：\n"
        )
        lines.append(
            "| # | 算子序列 | 算子数 | 重复次数 | 单迭代耗时(us) | 占比(%) | 首个实例 |"
        )
        lines.append(
            "|---|---------|--------|----------|---------------|---------|----------|"
        )
        for i, r in enumerate(self_analysis[:15], 1):
            first = ", ".join(r["first_names"][:2])[:40]
            lines.append(
                f"| {i} | {r['pattern'][:55]} | {r['pattern_len']} | "
                f"{r['repeat_count']} | {r['per_iter_us']:.1f} | {r['ratio_pct']:.1f} | {first} |"
            )
        lines.append("")
        self_total = sum(r["per_iter_us"] for r in self_analysis)
        self_pct = self_total / a_per_iter * 100 if a_per_iter > 0 else None
        self_pct_str = f"{self_pct:.1f}%" if self_pct is not None else "N/A"
        lines.append(
            f"**AutoFuse 自身可融未融结构单迭代合计**: {self_total:.1f} us "
            f"（占基线每迭代 {self_pct_str}）\n"
        )
    else:
        lines.append(f"(从 {la} 执行序列中未检测到可融但未融合的重复结构)\n")

    lines.append("## 3. AutoFuse 相同融合范围性能劣化分析\n")
    same_scope = comparator.find_same_scope_degradation()
    if same_scope:
        degraded = [s for s in same_scope if s["a_is_slower"]]
        improved = [s for s in same_scope if not s["a_is_slower"]]
        if degraded:
            lines.append(f"### 3.1 基线 {la} 劣化的融合 kernel\n")
            lines.append(
                f"| # | 语义 | {la} Type | {lb} Type | {la} Dur(us) | {lb} Dur(us) | 差异(us) | "
                f"劣化(%) | {la}/迭代(us) | {lb}/迭代(us) | 迭代影响(us) | {la} BlockDim | "
                f"{lb} BlockDim | 瓶颈(ratio差) |"
            )
            lines.append(
                "|---|------|---------|---------|-----------|-----------|---------|"
                "---------|------------|------------|-------------|------|------|------|"
            )
            total_deg_impact = 0.0
            for i, s in enumerate(degraded[:20], 1):
                total_deg_impact += s["iter_impact_us"]
                lines.append(
                    f"| {i} | {s['semantic'][:25]} | {s['a_type'][:35]} | {s['b_type'][:35]} | "
                    f"{s['a_dur']} | {s['b_dur']} | {s['dur_diff']} | {s['dur_pct']:.1f}% | "
                    f"{s['a_per_iter_us']:.1f} | {s['b_per_iter_us']:.1f} | {s['iter_impact_us']:.1f} | "
                    f"{s.get('a_block_dim', 0)} | {s.get('b_block_dim', 0)} | {s['bottleneck']} |"
                )
            lines.append(
                f"\n**相同范围劣化对迭代总影响**: {total_deg_impact:.1f} us/iter\n"
            )
        if improved:
            lines.append(f"### 3.2 基线 {la} 优势的融合 kernel\n")
            lines.append(
                f"| # | 语义 | {la} Dur(us) | {lb} Dur(us) | 差异(us) | 迭代影响(us) |"
            )
            lines.append("|---|------|-----------|-----------|---------|-------------|")
            for i, s in enumerate(improved[:10], 1):
                lines.append(
                    f"| {i} | {s['semantic'][:25]} | {s['a_dur']} | {s['b_dur']} | "
                    f"{s['dur_diff']} | {s['iter_impact_us']:.1f} |"
                )
            lines.append("")
    else:
        lines.append("(无相同融合范围的算子可对比)\n")

    lines.append("## 4. 热点概览对比\n")
    a_iter_us = (a.total_fusion_us + a.total_non_fusion_us) / max(a.iter_count, 1)
    b_iter_us = (b.total_fusion_us + b.total_non_fusion_us) / max(b.iter_count, 1)
    a_op_count = sum(safe_int(r.get("Count", 0)) for r in a.op_statistic)
    b_op_count = sum(safe_int(r.get("Count", 0)) for r in b.op_statistic)
    a_fused_count = sum(op["count"] for op in a.fusion_ops)
    b_fused_count = sum(op["count"] for op in b.fusion_ops)
    lines.append("| 指标 | AutoFuse | 对比模式 |")
    lines.append("|------|----------|----------|")
    lines.append(f"| 单迭代耗时(us) | {a_iter_us:.1f} | {b_iter_us:.1f} |")
    lines.append(f"| 迭代算子总数 | {a_op_count} | {b_op_count} |")
    lines.append(f"| 融合算子数 | {a_fused_count} | {b_fused_count} |")
    lines.append(
        f"| 非融合算子数 | {a_op_count - a_fused_count} | {b_op_count - b_fused_count} |"
    )
    lines.append(f"| 融合算子种类 | {len(a.fusion_ops)} | {len(b.fusion_ops)} |")
    lines.append("")

    hotspots = comparator.compare_hotspots()
    unfused_hot = [h for h in hotspots if h["status"] == "未融合(可优化)"]
    degraded_hot = [h for h in hotspots if h["status"] == "融合劣化"]
    normal_hot = [h for h in hotspots if h["status"] == "融合正常"]
    only_a_hot = [h for h in hotspots if h["status"] == "仅A融合"]

    if unfused_hot:
        lines.append(f"### 4.1 AutoFuse 未融合热点（{lb} 已融合，AutoFuse 可优化）\n")
        lines.append(
            f"| # | OP Type | AutoFuse Count | AutoFuse Total(us) | AutoFuse Avg(us) | {lb} 对应融合 | 状态 |"
        )
        lines.append(
            "|---|---------|---------------|-------------------|-----------------|-------------|------|"
        )
        for i, h in enumerate(unfused_hot[:15], 1):
            lines.append(
                f"| {i} | {h['op_type'][:40]} | {h['a_count']} | {h['a_total']:.1f} | {h['a_avg']:.2f} | "
                f"{h['b_count']} | 未融合(可优化) |"
            )
        lines.append("")

    if degraded_hot:
        lines.append(
            "### 4.2 AutoFuse 融合劣化热点（相同融合范围，AutoFuse kernel 更慢）\n"
        )
        lines.append(
            f"| # | OP Type | AutoFuse Count | AutoFuse Avg(us) | {lb} Avg(us) | 劣化(%) | AutoFuse Total(us) | 状态 |"
        )
        lines.append(
            "|---|---------|---------------|-----------------|-----------|---------|-------------------|------|"
        )
        for i, h in enumerate(degraded_hot[:15], 1):
            pct = (
                ((h["a_avg"] - h["b_avg"]) / h["b_avg"] * 100) if h["b_avg"] > 0 else 0
            )
            lines.append(
                f"| {i} | {h['op_type'][:40]} | {h['a_count']} | {h['a_avg']:.2f} | {h['b_avg']:.2f} | "
                f"+{pct:.1f}% | {h['a_total']:.1f} | 融合劣化 |"
            )
        lines.append("")

    if normal_hot:
        lines.append("### 4.3 AutoFuse 融合正常/优势热点\n")
        lines.append(
            f"| # | OP Type | AutoFuse Count | AutoFuse Avg(us) | {lb} Avg(us) | AutoFuse Total(us) | 状态 |"
        )
        lines.append(
            "|---|---------|---------------|-----------------|-----------|-------------------|------|"
        )
        for i, h in enumerate(normal_hot[:10], 1):
            lines.append(
                f"| {i} | {h['op_type'][:40]} | {h['a_count']} | {h['a_avg']:.2f} | {h['b_avg']:.2f} | "
                f"{h['a_total']:.1f} | 融合正常 |"
            )
        lines.append("")

    if only_a_hot:
        lines.append("### 4.4 仅 AutoFuse 融合的算子\n")
        lines.append("| # | OP Type | AutoFuse Count | AutoFuse Total(us) | 状态 |")
        lines.append("|---|---------|---------------|-------------------|------|")
        for i, h in enumerate(only_a_hot[:10], 1):
            lines.append(
                f"| {i} | {h['op_type'][:40]} | {h['a_count']} | {h['a_total']:.1f} | 仅A融合 |"
            )
        lines.append("")

    lines.append(f"## 5. AutoFuse 未融合但 {lb} 已融合的融合范围差异\n")
    a_sem = a.get_semantic_map()
    b_sem = b.get_semantic_map()
    all_sem = set(a_sem.keys()) | set(b_sem.keys())
    b_only = []
    for sem in all_sem:
        a_ops = a_sem.get(sem, [])
        b_ops = b_sem.get(sem, [])
        if not a_ops and b_ops:
            b_total = sum(op["total_us"] for op in b_ops)
            b_count = sum(op["count"] for op in b_ops)
            b_per_iter = b_total / max(b.iter_count, 1)
            b_types = ", ".join(op["op_type"][:40] for op in b_ops[:2])
            b_only.append(
                {
                    "semantic": sem,
                    "b_types": b_types,
                    "b_count": b_count,
                    "b_total": b_total,
                    "b_per_iter": b_per_iter,
                }
            )
    b_only.sort(key=lambda x: x["b_per_iter"], reverse=True)
    if b_only:
        lines.append(
            f"以下融合范围在 {lb} 中已融合，但 AutoFuse 未融合，是 AutoFuse 的优化机会：\n"
        )
        lines.append(
            f"| # | {lb} 融合范围 | {lb} 融合算子 | 数量(Count) | 单迭代耗时(us) |"
        )
        lines.append("|---|-------------|-------------|-----------|---------------|")
        total_iter_impact = 0.0
        for i, item in enumerate(b_only[:20], 1):
            total_iter_impact += item["b_per_iter"]
            lines.append(
                f"| {i} | {item['semantic'][:30]} | {item['b_types']} | {item['b_count']} | {item['b_per_iter']:.1f} |"
            )
        lines.append(
            f"\n**AutoFuse 未融合但 {lb} 已融合的融合范围共 {len(b_only)} 种，单迭代合计 {total_iter_impact:.1f} us**\n"
        )
    else:
        lines.append(f"(未发现 {lb} 已融合但 AutoFuse 未融合的融合范围)\n")

    lines.append("## 6. AutoFuse 优化建议总结\n")
    unfused_impact = (
        sum(u["iter_impact_us"] for u in unfused_vs_fused) if unfused_vs_fused else 0
    )
    self_impact = sum(r["per_iter_us"] for r in self_analysis) if self_analysis else 0
    deg_impact = (
        sum(s["iter_impact_us"] for s in same_scope if s["a_is_slower"])
        if same_scope
        else 0
    )
    total_impact = unfused_impact + self_impact + deg_impact
    lines.append("| 影响来源 | 每迭代影响(us) | 占基线每迭代(%) |")
    lines.append("|---------|-------------|---------------|")
    lines.append(
        f"| ① 能力对比可融未融 | {unfused_impact:.1f} | {unfused_impact / a_per_iter * 100:.1f}% |"
        if a_per_iter > 0
        else f"| ① 能力对比可融未融 | {unfused_impact:.1f} | N/A |"
    )
    lines.append(
        f"| ② 自身能力可融未融 | {self_impact:.1f} | {self_impact / a_per_iter * 100:.1f}% |"
        if a_per_iter > 0
        else f"| ② 自身能力可融未融 | {self_impact:.1f} | N/A |"
    )
    lines.append(
        f"| 相同范围劣化 | {deg_impact:.1f} | {deg_impact / a_per_iter * 100:.1f}% |"
        if a_per_iter > 0
        else f"| 相同范围劣化 | {deg_impact:.1f} | N/A |"
    )
    lines.append(
        f"| **合计** | **{total_impact:.1f}** | **{total_impact / a_per_iter * 100:.1f}%** |"
        if a_per_iter > 0
        else f"| **合计** | **{total_impact:.1f}** | **N/A** |"
    )
    lines.append("")

    return "\n".join(lines)


# ============================================================
#  报告生成 — compare 子命令（before-after）
# ============================================================


def generate_before_after_report(comparator: FusionComparator) -> str:
    a, b = comparator.a, comparator.b
    la, lb = comparator.label_a, comparator.label_b
    lines = []
    lines.append("# 融合前后对比分析报告\n")
    lines.append("## 1. 数据与口径\n")
    lines.append("| 项 | 融合前 | 融合后 |")
    lines.append("|---|--------|--------|")
    lines.append(f"| 标签 | {la} | {lb} |")
    lines.append(f"| 融合模式 | {a.fusion_mode_label} | {b.fusion_mode_label} |")
    lines.append(f"| 融合算子种类 | {len(a.fusion_ops)} | {len(b.fusion_ops)} |")
    lines.append(
        f"| 融合算子总耗时 | {a.total_fusion_us:.1f} us | {b.total_fusion_us:.1f} us |"
    )
    lines.append(
        f"| 非融合算子总耗时 | {a.total_non_fusion_us:.1f} us | {b.total_non_fusion_us:.1f} us |"
    )
    a_total_us = a.total_fusion_us + a.total_non_fusion_us
    b_total_us = b.total_fusion_us + b.total_non_fusion_us
    lines.append(f"| 算子总耗时 | {a_total_us:.1f} us | {b_total_us:.1f} us |")
    lines.append(f"| 迭代次数 | {a.iter_count} | {b.iter_count} |")
    a_per_iter = a_total_us / max(a.iter_count, 1)
    b_per_iter = b_total_us / max(b.iter_count, 1)
    lines.append(
        f"| 每迭代算子耗时(算子耗时和) | {a_per_iter:.1f} us | {b_per_iter:.1f} us |"
    )
    a_op_count_per = a.total_op_count / max(a.iter_count, 1)
    b_op_count_per = b.total_op_count / max(b.iter_count, 1)
    lines.append(f"| 每迭代算子数 | {a_op_count_per:.1f} | {b_op_count_per:.1f} |")
    lines.append(f"| 每迭代差异 | {a_per_iter - b_per_iter:.1f} us | - |")
    if a.iter_wall_us is not None or b.iter_wall_us is not None:
        aw = f"{a.iter_wall_us:.1f} us" if a.iter_wall_us is not None else "N/A"
        bw = f"{b.iter_wall_us:.1f} us" if b.iter_wall_us is not None else "N/A"
        lines.append(f"| 每迭代平均耗时(step_trace) | {aw} | {bw} |")
    lines.append("")

    lines.append("## 2. AutoFuse 热点概览\n")
    hotspots = comparator.compare_hotspots()
    unfused_hot = []
    for h in hotspots:
        if h["status"] in ("未融合(可优化)", "未融合(无对比)"):
            unfused_hot.append(h)
    degraded_hot = [h for h in hotspots if h["status"] == "融合劣化"]
    improved_hot = [h for h in hotspots if h["status"] == "融合正常"]

    if degraded_hot:
        lines.append("### 2.1 融合后劣化热点\n")
        lines.append(
            "| # | OP Type | 融合前 Avg(us) | 融合后 Avg(us) | 劣化(%) | 融合后 Total(us) |"
        )
        lines.append(
            "|---|---------|---------------|---------------|---------|-----------------|"
        )
        for i, h in enumerate(degraded_hot[:15], 1):
            pct = (
                ((h["b_avg"] - h["a_avg"]) / h["a_avg"] * 100) if h["a_avg"] > 0 else 0
            )
            lines.append(
                f"| {i} | {h['op_type'][:40]} | {h['a_avg']:.2f} | {h['b_avg']:.2f} | "
                f"+{pct:.1f}% | {h['b_total']:.1f} |"
            )
        lines.append("")

    if unfused_hot:
        lines.append("### 2.2 未融合热点\n")
        lines.append(
            "| # | OP Type | 融合后 Count | 融合后 Total(us) | 融合后 Avg(us) | 状态 |"
        )
        lines.append(
            "|---|---------|-------------|-----------------|---------------|------|"
        )
        for i, h in enumerate(unfused_hot[:15], 1):
            lines.append(
                f"| {i} | {h['op_type'][:40]} | {h['b_count']} | {h['b_total']:.1f} | "
                f"{h['b_avg']:.2f} | {h['status']} |"
            )
        lines.append("")

    if improved_hot:
        lines.append("### 2.3 融合后改善/正常热点\n")
        lines.append(
            "| # | OP Type | 融合前 Avg(us) | 融合后 Avg(us) | 改善(%) | 融合后 Total(us) |"
        )
        lines.append(
            "|---|---------|---------------|---------------|---------|-----------------|"
        )
        for i, h in enumerate(improved_hot[:10], 1):
            pct = (
                ((h["a_avg"] - h["b_avg"]) / h["a_avg"] * 100) if h["a_avg"] > 0 else 0
            )
            lines.append(
                f"| {i} | {h['op_type'][:40]} | {h['a_avg']:.2f} | {h['b_avg']:.2f} | {pct:.1f}% | {h['b_total']:.1f} |"
            )
        lines.append("")

    lines.append("## 3. 融合劣化片段\n")
    degradations = comparator.find_degradation_segments()
    if degradations:
        lines.append(
            "| # | OP Type | 融合前 Avg(us) | 融合后 Avg(us) | 劣化(%) | 融合算子 |"
        )
        lines.append(
            "|---|---------|---------------|---------------|---------|:--------:|"
        )
        for i, d in enumerate(degradations[:30], 1):
            fused = "是" if d["is_fused"] else "否"
            lines.append(
                f"| {i} | {d['op_type'][:40]} | {d['a_avg']:.2f} | {d['b_avg']:.2f} | "
                f"+{d['diff_pct']:.1f}% | {fused} |"
            )
    else:
        lines.append("(未检测到融合劣化片段)")
    lines.append("")

    lines.append("## 4. 可融但未融合片段\n")
    unfused = comparator.find_unfused_opportunities()
    if unfused:
        lines.append("| 来源 | OP Type | Count | Avg(us) | Total(us) |")
        lines.append("|------|---------|-------|---------|-----------|")
        for u in unfused[:20]:
            lines.append(
                f"| {u['source']} | {u['op_type']} | {u['count']} | {u['avg_us']:.1f} | {u['total_us']:.1f} |"
            )
    else:
        lines.append("(未检测到可融但未融合片段)")
    lines.append("")

    lines.append("## 5. 融合范围差异\n")
    a_sem = a.get_semantic_map()
    b_sem = b.get_semantic_map()
    all_sem = set(a_sem.keys()) | set(b_sem.keys())
    lines.append(
        "| 语义 | 融合前范围 | 融合前 Count | 融合前 Total(us) | 融合后范围 | 融合后 Count | 融合后 Total(us) | 差异类型 |"
    )
    lines.append(
        "|------|-----------|-------------|-----------------|-----------|-------------|-----------------|---------|"
    )
    for sem in sorted(
        all_sem,
        key=lambda s: abs(
            sum(op["total_us"] for op in a_sem.get(s, []))
            - sum(op["total_us"] for op in b_sem.get(s, []))
        ),
        reverse=True,
    )[:20]:
        a_ops = a_sem.get(sem, [])
        b_ops = b_sem.get(sem, [])
        a_total = sum(op["total_us"] for op in a_ops)
        b_total = sum(op["total_us"] for op in b_ops)
        a_count = sum(op["count"] for op in a_ops)
        b_count = sum(op["count"] for op in b_ops)
        a_types = ", ".join(op["op_type"][:30] for op in a_ops[:2]) if a_ops else "-"
        b_types = ", ".join(op["op_type"][:30] for op in b_ops[:2]) if b_ops else "-"
        diff = "A only" if not b_ops else ("B only" if not a_ops else "both")
        lines.append(
            f"| {sem[:25]} | {a_types} | {a_count} | {a_total:.1f} | "
            f"{b_types} | {b_count} | {b_total:.1f} | {diff} |"
        )
    lines.append("")

    return "\n".join(lines)


# ============================================================
#  自动发现图文件（all 子命令）
# ============================================================


def auto_discover_graph_file(profiling_dir: str) -> Optional[str]:
    """从 Profiling 目录附近自动查找图文件 (.pbtxt / output_code.py)"""
    candidates = []
    parent = os.path.dirname(profiling_dir.rstrip(os.sep))
    if not parent:
        parent = profiling_dir
    parent_len = len(parent)
    for root, dirs, files in os.walk(parent):
        depth = root[parent_len:].count(os.sep)
        if depth > 3:
            dirs[:] = []
            continue
        for fn in files:
            if fn.endswith(".pbtxt") and "ge_onnx" in fn:
                candidates.append(os.path.join(root, fn))
            elif fn == "output_code.py":
                candidates.append(os.path.join(root, fn))
        candidate_names = {os.path.basename(c) for c in candidates}
        for fn in files:
            if fn.endswith(".pbtxt") and fn not in candidate_names:
                candidates.append(os.path.join(root, fn))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return candidates[0]


# ============================================================
#  主入口 — 子命令调度
# ============================================================


def _cmd_profile(args) -> int:
    if not os.path.isdir(args.profiling_dir):
        logging.error(f"错误: 目录不存在: {args.profiling_dir}")
        return 1
    logging.info(f"分析目录: {args.profiling_dir}\n")
    p = ProfilingParser(args.profiling_dir)
    data = p.parse()
    report = generate_profile_report(data)
    _output_report(report, args.output)
    return 0


def _cmd_graph(args) -> int:
    if not os.path.exists(args.graph_file):
        logging.error(f"文件不存在: {args.graph_file}")
        return 1
    gtype = detect_graph_type(args.graph_file)
    logging.info(f"解析文件: {args.graph_file}")
    logging.info(f"文件类型: {gtype}")
    if args.profiling_dir:
        logging.info(f"Profiling: {args.profiling_dir}")
    logging.info("")
    if gtype == "pbtxt":
        p = PbtxtParser()
        nodes = p.parse_file(args.graph_file)
    else:
        p = FXGraphParser()
        nodes = p.parse_file(args.graph_file)
    logging.info(f"解析完成: {len(nodes)} 个算子节点")
    analyzer = GraphAnalyzer(nodes, gtype, profiling_dir=args.profiling_dir)
    report = analyzer.format_report()
    _output_report(report, args.output)
    return 0


def _cmd_compare(args) -> int:
    if args.compare_mode == "before-after":
        if not args.before_prof or not args.after_prof:
            logging.error("before-after 模式需要 --before-prof 和 --after-prof")
            return 1
        data_a = FusionDataLoader(args.before_prof, "融合前")
        data_b = FusionDataLoader(args.after_prof, "融合后")
    else:
        if not args.mode_a_prof or not args.mode_b_prof:
            logging.error("cross-mode 模式需要 --mode-a-prof 和 --mode-b-prof")
            return 1
        data_a = FusionDataLoader(args.mode_a_prof, args.mode_a_name)
        data_b = FusionDataLoader(args.mode_b_prof, args.mode_b_name)
    logging.info(f"加载 {data_a.label} Profiling: {data_a.profiling_dir}")
    data_a.load()
    logging.info(
        f"  融合模式: {data_a.fusion_mode_label}, 融合算子: {len(data_a.fusion_ops)}种"
    )
    logging.info(f"加载 {data_b.label} Profiling: {data_b.profiling_dir}")
    data_b.load()
    logging.info(
        f"  融合模式: {data_b.fusion_mode_label}, 融合算子: {len(data_b.fusion_ops)}种"
    )
    logging.info("")
    comparator = FusionComparator(data_a, data_b, data_a.label, data_b.label)
    if args.compare_mode == "before-after":
        report = generate_before_after_report(comparator)
    else:
        report = generate_cross_mode_report(comparator)
    _output_report(report, args.output)
    return 0


def _cmd_all(args) -> int:
    if not os.path.isdir(args.profiling_dir):
        logging.error(f"错误: 目录不存在: {args.profiling_dir}")
        return 1
    parts = []
    logging.info(f"[profile] 分析目录: {args.profiling_dir}")
    p = ProfilingParser(args.profiling_dir)
    data = p.parse()
    parts.append(generate_profile_report(data))
    logging.info(f"  融合模式: {data['fusion_mode']['mode_label']}, 报告生成完成")
    graph_file = args.graph_file
    if not graph_file and args.graph_dir:
        for root, _dirs, files in os.walk(args.graph_dir):
            for fn in files:
                if fn.endswith(".pbtxt") or fn == "output_code.py":
                    graph_file = os.path.join(root, fn)
                    break
            if graph_file:
                break
    if not graph_file:
        graph_file = auto_discover_graph_file(args.profiling_dir)
    if graph_file:
        logging.info(f"[graph] 解析图文件: {graph_file}")
        gtype = detect_graph_type(graph_file)
        if gtype == "pbtxt":
            gp = PbtxtParser()
            nodes = gp.parse_file(graph_file)
        else:
            gp = FXGraphParser()
            nodes = gp.parse_file(graph_file)
        logging.info(f"  文件类型: {gtype}, 解析完成: {len(nodes)} 个算子节点")
        analyzer = GraphAnalyzer(nodes, gtype, profiling_dir=args.profiling_dir)
        parts.append(analyzer.format_report())
    else:
        logging.warning("未找到图文件，跳过 graph 分析")
        parts.append("## 图结构分析\n\n(未找到图文件，跳过)\n")
    report = "\n\n---\n\n".join(parts)
    _output_report(report, args.output)
    return 0


def _output_report(report: str, output_path: Optional[str]):
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logging.info(f"报告已保存至: {output_path}")
    else:
        print(report)


def main():
    parser = argparse.ArgumentParser(
        description="AutoFuse 融合性能分析统一脚本 (profile / graph / compare / all)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
        "  python autofuse_analysis.py profile PROF_DIR -o report.md\n"
        "  python autofuse_analysis.py graph dump_graph.pbtxt --profiling-dir PROF_DIR -o report.md\n"
        "  python autofuse_analysis.py compare --mode cross-mode --mode-a-prof AF_PROF "
        "--mode-a-name AutoFuse --mode-b-prof TRITON_PROF --mode-b-name Triton -o report.md\n"
        "  python autofuse_analysis.py compare --mode before-after --before-prof BEFORE --after-prof AFTER\n"
        "  python autofuse_analysis.py all PROF_DIR --graph-file dump_graph.pbtxt -o report.md\n",
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    p_profile = sub.add_parser("profile", help="单组 Profiling 分析")
    p_profile.add_argument("profiling_dir", help="MindStudio Profiler 输出目录")
    p_profile.add_argument("--output", "-o", default=None, help="Markdown 报告输出路径")

    p_graph = sub.add_parser("graph", help="Dump 图结构分析 (pbtxt / output_code.py)")
    p_graph.add_argument("graph_file", help="图文件路径 (.pbtxt 或 .py)")
    p_graph.add_argument(
        "--profiling-dir", default="", help="Profiling 输出目录（可选，用于关联耗时）"
    )
    p_graph.add_argument("--output", "-o", default=None, help="Markdown 报告输出路径")

    p_cmp = sub.add_parser("compare", help="融合对比 (before-after / cross-mode)")
    p_cmp.add_argument(
        "--mode",
        dest="compare_mode",
        choices=["before-after", "cross-mode"],
        required=True,
        help="对比模式: before-after=融合前后, cross-mode=双模式(以AutoFuse为基线)",
    )
    p_cmp.add_argument(
        "--before-prof", default="", help="before-after: 融合前 Profiling 目录"
    )
    p_cmp.add_argument(
        "--before-graph", default="", help="before-after: 融合前 Dump 图文件"
    )
    p_cmp.add_argument(
        "--after-prof", default="", help="before-after: 融合后 Profiling 目录"
    )
    p_cmp.add_argument(
        "--after-graph", default="", help="before-after: 融合后 Dump 图文件"
    )
    p_cmp.add_argument(
        "--mode-a-prof", default="", help="cross-mode: 模式A(基线) Profiling 目录"
    )
    p_cmp.add_argument(
        "--mode-a-graph", default="", help="cross-mode: 模式A(基线) Dump 图文件"
    )
    p_cmp.add_argument(
        "--mode-a-name", default="模式A", help="cross-mode: 模式A(基线) 名称"
    )
    p_cmp.add_argument(
        "--mode-b-prof", default="", help="cross-mode: 模式B(对比) Profiling 目录"
    )
    p_cmp.add_argument(
        "--mode-b-graph", default="", help="cross-mode: 模式B(对比) Dump 图文件"
    )
    p_cmp.add_argument(
        "--mode-b-name", default="模式B", help="cross-mode: 模式B(对比) 名称"
    )
    p_cmp.add_argument("--output", "-o", default=None, help="Markdown 报告输出路径")

    p_all = sub.add_parser("all", help="单模式综合分析 (profile + graph)")
    p_all.add_argument("profiling_dir", help="MindStudio Profiler 输出目录")
    p_all.add_argument("--graph-dir", default="", help="图文件搜索目录（可选）")
    p_all.add_argument(
        "--graph-file", default="", help="直接指定图文件（可选，优先于自动发现）"
    )
    p_all.add_argument("--output", "-o", default=None, help="Markdown 报告输出路径")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    dispatch = {
        "profile": _cmd_profile,
        "graph": _cmd_graph,
        "compare": _cmd_compare,
        "all": _cmd_all,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
