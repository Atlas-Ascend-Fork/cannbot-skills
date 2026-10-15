# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Targeted checks for explicit Contraction C++ contract violations.

The callable entry point is validate_gates(design_text).
These checks do not decide whether a skill can generate code, require a test
CLI, or require framework-owned CopyOut/VF implementations. Empty results
mean no recognized violation, not proof of coverage, ABI or device precision.
Only unambiguous assignments are inspected; symbolic configurations require
the documented tiling/SWAT checks and source review. A min or CeilDiv expression
is not itself a violation, and class names do not select a different policy.
"""

from __future__ import annotations

import re


def _assignments(code: str, name: str) -> list[str]:
    return re.findall(rf"\b{re.escape(name)}\s*=(?!=)\s*([^;\n]+)", code)


def _literal(code: str, name: str) -> int | None:
    values = _assignments(code, name)
    if len(values) != 1:
        return None
    value = values[0].strip()
    if not re.fullmatch(r"-?[0-9]+[uUlL]*", value):
        return None
    return int(re.sub(r"[uUlL]+$", "", value))


def validate_gates(design_text: str) -> list[str]:
    """Return actionable errors in C++ design snippets, without new metadata."""
    errors: list[str] = []
    blocks = re.findall(r"```(?:cpp|c\+\+)\s*\n(.*?)```", design_text, re.S)
    for index, raw in enumerate(blocks, 1):
        # Comments and string contents cannot count as implementation evidence.
        code = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', "", raw, flags=re.S)
        prefix = f"Contraction code block {index}: "
        # Both public strategy names and AscendC field bindings denote positive parts.
        # Symbolic expressions are reviewed against their actual strategy derivation.
        for name in ("mTailParts", "nTailParts", "mTailCnt", "nTailCnt"):
            value = _literal(code, name)
            if value is not None and value <= 0:
                errors.append(prefix + f"{name} must be positive.")
        # Only reject a known void* source, not aliases/RAII wrappers we cannot resolve.
        if "aclrtMalloc" in code:
            for name in re.findall(r"\bvoid\s*\*\s*(\w+)", code):
                if re.search(
                    rf"reinterpret_cast\s*<\s*GM_ADDR\s*>\s*\(\s*{name}\s*\)", code
                ):
                    errors.append(
                        prefix + "void* allocation cast to GM_ADDR; "
                        "use an address-space-compatible typed device pointer."
                    )
        # A recognized one-byte allocation contradicting an explicit BF16 output type.
        bf16_output = bool(
            re.search(
                r"using\s+(?:CType|OutputType)\s*=\s*(?:bfloat16_t|bf16_t)\b", code
            )
        )
        if bf16_output:
            for name in ("cBytes", "outputBytes"):
                if any(
                    re.fullmatch(
                        r"(?:numel|cNumel|outputNumel|elementCount)", rhs.strip()
                    )
                    for rhs in _assignments(code, name)
                ):
                    errors.append(
                        prefix + f"{name} contains element count only; "
                        "multiply checked output numel by the BF16 storage width."
                    )
    return errors
