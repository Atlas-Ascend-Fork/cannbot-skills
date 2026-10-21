#!/usr/bin/env python3
# --------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with it.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the repository root for the full text of the License.
# --------------------------------------------------------------------------------------------------
"""Create a small, standalone Ascend C operator test for mssanitizer."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CASE_SHAPES = (
    ("small-vector", (128,)),
    ("medium-vector", (1024,)),
    ("large-vector", (4096,)),
    ("unaligned-three", (3,)),
    ("unaligned-five", (5,)),
    ("matrix-32x512", (32, 512)),
    ("matrix-64x768", (64, 768)),
    ("tensor-8x16x64", (8, 16, 64)),
)
OPERATOR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def case_matrix(dtypes: list[str]) -> list[dict[str, object]]:
    """Return serializable test cases consumed by the generated program."""
    return [
        {"name": f"{label}-{dtype}", "shape": list(shape), "dtype": dtype}
        for label, shape in CASE_SHAPES
        for dtype in dtypes
    ]


def generated_program(operator: str, dtypes: list[str]) -> str:
    """Build source without embedding executable user input in Python syntax."""
    config = json.dumps(
        {"operator": operator, "cases": case_matrix(dtypes)}, ensure_ascii=False
    )
    return f'''#!/usr/bin/env python3
"""Auto-generated mssanitizer workload for {operator}."""

import argparse
import json
import os
from datetime import datetime, timezone

import torch
import torch_npu


CONFIG = json.loads({config!r})
TORCH_DTYPES = {{
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}}


def print_runtime():
    print("mssanitizer workload:", CONFIG["operator"])
    print("PyTorch:", torch.__version__)
    print("torch_npu:", torch_npu.__version__)
    print("NPU available:", torch.npu.is_available())
    if torch.npu.is_available():
        print("NPU count:", torch.npu.device_count())


def resolve_operator():
    namespace = getattr(torch.ops, "customize", None)
    operation = getattr(namespace, CONFIG["operator"], None) if namespace else None
    if operation is None:
        raise RuntimeError(
            "目标算子未注册: torch.ops.customize." + CONFIG["operator"]
        )
    return operation


def execute_case(operation, case):
    dtype_name = case["dtype"]
    dtype = TORCH_DTYPES.get(dtype_name, getattr(torch, dtype_name, None))
    if dtype is None:
        raise ValueError("torch 不支持数据类型: " + dtype_name)
    tensor = torch.randn(tuple(case["shape"]), dtype=dtype, device="npu:0")
    output = operation(tensor)
    torch.npu.synchronize()
    shape = list(output.shape) if hasattr(output, "shape") else None
    return {{"output_shape": shape}}


def run_cases():
    if not torch.npu.is_available():
        raise RuntimeError("NPU 不可用，无法执行目标算子测试")
    operation = resolve_operator()
    records = []
    for case in CONFIG["cases"]:
        record = {{**case, "status": "PASS"}}
        try:
            record.update(execute_case(operation, case))
            print("PASS", case["name"], case["shape"], case["dtype"])
        except Exception as error:
            record.update(status="FAIL", error=str(error))
            print("FAIL", case["name"], str(error))
        records.append(record)
    return records


def save_report(destination, records):
    failed = sum(item["status"] == "FAIL" for item in records)
    report = {{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_type": "mssanitizer 内存检测测试",
        "operator": CONFIG["operator"],
        "total_tests": len(records),
        "passed": len(records) - failed,
        "failed": failed,
        "results": records,
    }}
    with open(destination, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print("report:", destination)
    return failed


def main():
    command = argparse.ArgumentParser(description="运行 Ascend C mssanitizer workload")
    command.add_argument("-o", "--output", required=True, help="输出 JSON 报告")
    arguments = command.parse_args()
    os.environ.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "1")
    os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
    print_runtime()
    return int(save_report(arguments.output, run_cases()) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
'''


def build_parser() -> argparse.ArgumentParser:
    """Create the generator command-line parser."""
    parser = argparse.ArgumentParser(description="生成 mssanitizer 测试脚本")
    parser.add_argument("--operator", required=True, help="torch.ops.customize 下的算子名称")
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["float16", "float32"],
        help="需要覆盖的数据类型",
    )
    parser.add_argument("-o", "--output", required=True, help="生成脚本路径")
    return parser


def main() -> int:
    """Validate arguments and write the generated workload."""
    arguments = build_parser().parse_args()
    if not OPERATOR_NAME.fullmatch(arguments.operator):
        raise SystemExit(f"非法算子名称: {arguments.operator}")
    destination = Path(arguments.output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        generated_program(arguments.operator, arguments.dtypes), encoding="utf-8"
    )
    print(f"已生成测试脚本: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
