# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""为精度看护用例做冒烟智能采样。

从全量用例 JSON 中抽取常规用例，并按边界类别各抽一条，形成可快速执行的冒烟集。
当前为控制执行时长，冒烟集暂不包含上边界（upper_border）用例；该类用例保留在全量集中。
"""

from __future__ import annotations

import argparse
import json
import random


BOUNDARY_SHAPE_CATEGORIES = {
    "empty_shape",
    "zero_dim",
    "scalar",
    "tiny_shape",
    "large_shape",
}

BOUNDARY_PRIORITY = [
    "empty_shape",
    "zero_dim",
    "scalar",
    "tiny_shape",
    "large_shape",
    "dtype_bf16",
    "dtype_fp16",
    "dtype_fp32",
    "infnan",
    "lower_border",
    "upper_border",
    "general_boundary",
]


def iter_tensor_inputs(inputs):
    for input_group in inputs:
        if isinstance(input_group, list):
            for inp in input_group:
                if isinstance(inp, dict) and inp.get("type") in ["tensor", "tensors"]:
                    yield inp
        elif isinstance(input_group, dict) and input_group.get("type") in ["tensor", "tensors"]:
            yield input_group


def categorize_tensor_shape(shape):
    categories = []
    if not isinstance(shape, (list, tuple)):
        return categories

    has_zero_dim = False
    total = 1
    for dim in shape:
        if not isinstance(dim, int):
            return []
        if dim == 0:
            has_zero_dim = True
        total *= dim

    if len(shape) == 0:
        categories.append("empty_shape")
    elif has_zero_dim:
        categories.append("zero_dim")
    elif total == 1:
        categories.append("scalar")
    elif total <= 4:
        categories.append("tiny_shape")

    max_dim = max(shape) if shape else 0
    if max_dim >= 1024:
        categories.append("large_shape")

    return categories


def dedupe_categories(categories):
    return list(dict.fromkeys(categories))


def is_upper_border_case(case):
    return "upper_border" in get_boundary_category(case)


def is_boundary_case(case):
    if case.get("is_boundary", False):
        return True
    if "boundary_type" in case:
        return True
    if case.get("case_type") == "boundary":
        return True

    description = case.get("description", "")
    if isinstance(description, str) and any(
        key in description.lower()
        for key in ["boundary", "empty", "infnan", "scalar", "border"]
    ):
        return True

    tensor_inputs = list(iter_tensor_inputs(case.get("inputs", [])))
    for tensor_input in tensor_inputs:
        shape_categories = categorize_tensor_shape(tensor_input.get("shape", []))
        if any(category in BOUNDARY_SHAPE_CATEGORIES for category in shape_categories):
            return True

    case_text = json.dumps(case, ensure_ascii=False).lower()
    return any(key in case_text for key in ["infnan", "lower_border", "upper_border"])


def get_boundary_category(case):
    categories = []
    inputs = case.get("inputs", [])

    for tensor_input in iter_tensor_inputs(inputs):
        categories.extend(categorize_tensor_shape(tensor_input.get("shape", [])))

        x_dtype = tensor_input.get("dtype", "")
        if x_dtype in ["bf16", "fp16", "fp32", "int32"]:
            categories.append(f"dtype_{x_dtype}")

    case_text = json.dumps(case, ensure_ascii=False).lower()
    if "infnan" in case_text:
        categories.append("infnan")
    if "lower_border" in case_text:
        categories.append("lower_border")
    if "upper_border" in case_text:
        categories.append("upper_border")

    categories = dedupe_categories(categories)
    if not categories:
        categories.append("general_boundary")

    return categories


def sample_boundary_cases_smart(boundary_cases, seed=42):
    random.seed(seed)

    category_groups = {}
    for case in boundary_cases:
        for cat in get_boundary_category(case):
            category_groups.setdefault(cat, []).append(case)

    sampled = []
    sampled_ids = set()

    for cat in BOUNDARY_PRIORITY:
        if cat not in category_groups:
            continue
        candidates = [c for c in category_groups[cat] if c["id"] not in sampled_ids]
        if candidates:
            selected = random.choice(candidates)
            sampled.append(selected)
            sampled_ids.add(selected["id"])

    return sampled, category_groups


def sample_quality_cases(input_json, output_json, normal_count=25, boundary_mode="all", seed=42):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data if isinstance(data, list) else data.get("cases", data)

    boundary_cases = []
    normal_cases = []

    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        case.setdefault("id", i)
        if is_boundary_case(case):
            boundary_cases.append(case)
        else:
            normal_cases.append(case)

    boundary_cases = [case for case in boundary_cases if not is_upper_border_case(case)]

    random.seed(seed)
    sampled_normal = random.sample(normal_cases, min(normal_count, len(normal_cases)))

    sampled_boundary = []
    category_info = {}

    if boundary_mode == "all":
        sampled_boundary = boundary_cases
    elif boundary_mode == "smart":
        sampled_boundary, category_info = sample_boundary_cases_smart(boundary_cases, seed)
    else:
        try:
            count = int(boundary_mode)
            sampled_boundary = random.sample(boundary_cases, min(count, len(boundary_cases)))
        except ValueError:
            sampled_boundary = boundary_cases

    sampled_cases = sampled_normal + sampled_boundary

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(sampled_cases, f, indent=2)

    print("[INFO] 采样完成")
    print(f"[INFO] 原始用例总数: {len(cases)}")
    print(f"[INFO] 常规用例: {len(normal_cases)} -> 采样: {len(sampled_normal)}")
    print(f"[INFO] 边界用例: {len(boundary_cases)} -> 采样: {len(sampled_boundary)}")

    if category_info:
        print("[INFO] 边界用例分类覆盖:")
        for cat, cat_cases in category_info.items():
            sampled_ids = [s["id"] for s in sampled_boundary]
            status = "✅" if any(c["id"] in sampled_ids for c in cat_cases) else "❌"
            print(f"[INFO]   {status} {cat}: {len(cat_cases)}条")

    print(f"[INFO] 输出文件: {output_json}")

    return sampled_cases


def main():
    parser = argparse.ArgumentParser(description="为精度看护用例做冒烟智能采样")
    parser.add_argument("-i", "--input", required=True, help="输入用例 JSON 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出采样后用例 JSON 文件路径")
    parser.add_argument("-n", "--normal", type=int, default=25, help="常规用例采样数量")
    parser.add_argument(
        "-b",
        "--boundary",
        default="smart",
        help="边界用例采样模式：all(全部测试)、smart(智能分类各抽1条)、具体数字(随机抽取N条)",
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    sample_quality_cases(
        input_json=args.input,
        output_json=args.output,
        normal_count=args.normal,
        boundary_mode=args.boundary,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()