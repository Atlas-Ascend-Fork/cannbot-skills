# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression checks for documented Contraction static diagnostics, not device proof."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "references/paradigms/contraction/validate_gates.py"
)
module_spec = importlib.util.spec_from_file_location("contraction_gates", SCRIPT)
gates = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(gates)


def snippet(code, route=""):
    return route + "\n```cpp\n" + code + "\n```"


class ContractionGatesTest(unittest.TestCase):
    def test_symbolic_public_strategy_is_not_rejected(self):
        # Parameter derivation and actual consumers still require source review.
        cases = [
            "usedCoreNum = std::min(P, T);",
            "blockNum = std::min(availableAic, totalTiles);",
            "launchBlockNum = min(availableAic, totalTiles);",
            "mTailCnt = splitM; nTailCnt = splitN;",
            "step = CeilDiv(lastValid, requestedParts); "
            "mTailCnt = CeilDiv(lastValid, step);",
        ]
        for route in ("", "MatmulMultiBlockBasic BlockSchedulerMatmulBasic"):
            for code in cases:
                with self.subTest(route=route, code=code):
                    self.assertEqual(gates.validate_gates(snippet(code, route)), [])

    def test_nonpositive_literal_split_counts(self):
        for name in ("mTailParts", "nTailParts", "mTailCnt", "nTailCnt"):
            for value in ("0", "0u", "-1"):
                with self.subTest(name=name, value=value):
                    self.assertEqual(
                        gates.validate_gates(snippet(f"int {name} = {value};")),
                        [f"Contraction code block 1: {name} must be positive."],
                    )

    def test_positive_parts_are_not_locked_to_one(self):
        # SWAT documents T=6, P=4, requested parts=(2,1); normalization is separate.
        cases = [
            "mTailParts = 2; nTailParts = 1;",
            "mTailCnt = 2u; nTailCnt = 1;",
            "mTailCnt = 1; mTailCnt = selectedParts;",
            "Kernel kernel; kernel(params);",
        ]
        for code in cases:
            with self.subTest(code=code):
                self.assertEqual(gates.validate_gates(snippet(code)), [])
        self.assertEqual(gates.validate_gates(""), [])

    def test_comments_and_strings_are_not_assignments(self):
        cases = [
            "// mTailCnt = 0;\nmTailCnt = 1;",
            "/* nTailParts = -1; */ nTailParts = 2;",
            'const char* note = "mTailCnt = 0;"; mTailCnt = 1;',
            'const char* note = "using CType = bfloat16_t; cBytes = numel;";',
        ]
        for code in cases:
            with self.subTest(code=code):
                self.assertEqual(gates.validate_gates(snippet(code)), [])

    def test_class_names_do_not_select_a_policy(self):
        names = "MatmulMultiBlockBasic BlockSchedulerMatmulBasic"
        code = "blockNum = std::min(availableAic, totalTiles); mTailCnt = 0;"
        expected = ["Contraction code block 1: mTailCnt must be positive."]
        cases = [
            snippet(code, "CustomScheduler"),
            snippet(code, names),
            snippet("// " + names + "\n" + code, "CustomScheduler"),
            snippet("/* " + names + " */\n" + code, "CustomScheduler"),
            snippet('const char* note = "' + names + '";\n' + code),
        ]
        for design in cases:
            with self.subTest(design=design):
                self.assertEqual(gates.validate_gates(design), expected)

    def test_code_blocks_have_separate_diagnostics(self):
        design = snippet(
            "Kernel kernel; kernel(params);",
            "MatmulMultiBlockBasic BlockSchedulerMatmulBasic",
        ) + snippet("blockNum = std::min(availableAic, totalTiles);")
        design += snippet("using CType = bfloat16_t; cBytes = numel;")
        self.assertEqual(
            gates.validate_gates(design),
            [
                "Contraction code block 3: cBytes contains element count only; "
                "multiply checked output numel by the BF16 storage width."
            ],
        )

    def test_pointer_binding(self):
        invalid = snippet(
            "void* a; aclrtMalloc(&a, bytes, mode); launch(reinterpret_cast<GM_ADDR>(a));"
        )
        self.assertEqual(
            gates.validate_gates(invalid),
            [
                "Contraction code block 1: void* allocation cast to GM_ADDR; "
                "use an address-space-compatible typed device pointer."
            ],
        )
        valid = snippet(
            "uint8_t* a; aclrtMalloc(reinterpret_cast<void**>(&a), bytes, mode); launch(a);"
        )
        self.assertEqual(gates.validate_gates(valid), [])

    def test_bf16_storage_bytes(self):
        invalid = snippet("using CType = bfloat16_t; cBytes = numel;")
        self.assertEqual(
            gates.validate_gates(invalid),
            [
                "Contraction code block 1: cBytes contains element count only; "
                "multiply checked output numel by the BF16 storage width."
            ],
        )
        valid = snippet("using CType = bfloat16_t; cBytes = numel * sizeof(CType);")
        self.assertEqual(gates.validate_gates(valid), [])


if __name__ == "__main__":
    unittest.main()
