#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ascendc-st-design 约束表达式安全求值回归测试（#698）。

验证：
1. spec 中定义的合法 expression（索引/切片/列表构造/算术/get_broadcast_result）行为不变；
2. 白名单外结构（属性访问、lambda、推导式、exec、open、__import__ 等）一律拒绝并阻断；
3. 恶意表达式的实际副作用（命令执行/文件读/文件写）不发生（哨兵文件保持原样）；
4. 合法表达式求值失败仍走软失败（返回 None），不是异常。
"""

import itertools
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "ops" / "ascendc-st-design" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from generate_factor_values import (  # noqa: E402
    FactorValueGenerator,
    UnsafeExpressionError,
    safe_eval_expression,
)


def _env(sources=None, broadcast=None):
    """构造与 solve_constraint_range 一致的求值环境"""
    if broadcast is None:
        def broadcast(*shapes):
            # 简化版广播，测试只需确定性结果
            result = list(shapes[0])
            for shape in shapes[1:]:
                for i, dim in enumerate(shape):
                    if i < len(result) and dim != 1:
                        result[i] = max(result[i], dim)
            return result
    return {
        "sources": sources if sources is not None else [],
        "get_broadcast_result": broadcast,
    }


class TestValidExpressions(unittest.TestCase):
    """spec（dependency-yaml-spec.md）中定义的合法表达式行为不变"""

    def test_simple_index(self):
        self.assertEqual(safe_eval_expression("sources[0]", _env([[1, 2, 3]])), [1, 2, 3])

    def test_list_construction(self):
        self.assertEqual(
            safe_eval_expression("[sources[0][1], sources[1][2]]", _env([[0, 4, 0], [0, 0, 5]])),
            [4, 5],
        )

    def test_slice_and_concat(self):
        self.assertEqual(
            safe_eval_expression("sources[0][:3] + [2]", _env([[1, 2, 3, 4]])),
            [1, 2, 3, 2],
        )

    def test_arithmetic(self):
        self.assertEqual(
            safe_eval_expression("sources[0][1] * sources[1][2]", _env([[0, 3, 0], [0, 0, 4]])),
            12,
        )

    def test_broadcast_call(self):
        self.assertEqual(
            safe_eval_expression("get_broadcast_result(sources[0], sources[1])", _env([[2, 1], [1, 3]])),
            [2, 3],
        )

    def test_numeric_constants_and_unary(self):
        self.assertEqual(safe_eval_expression("-1 + 2 ** 3", _env([])), 7)


class TestBlockedExpressions(unittest.TestCase):
    """白名单外结构一律拒绝"""

    def _assert_blocked(self, expression, env=None):
        with self.assertRaises(UnsafeExpressionError):
            safe_eval_expression(expression, env or _env([[1, 2]]))

    def test_import_and_command_execution(self):
        self._assert_blocked("__import__('os').system('echo pwned')")

    def test_open_file_read(self):
        self._assert_blocked("open('/etc/passwd').read()")

    def test_open_file_write(self):
        self._assert_blocked("open('/tmp/pwned.txt', 'w').write('pwned')")

    def test_attribute_access(self):
        self._assert_blocked("sources[0].__class__")

    def test_lambda_call(self):
        self._assert_blocked("(lambda: __import__('os'))()")

    def test_list_comprehension(self):
        self._assert_blocked("[x for x in sources[0]]")

    def test_dict_and_fstring(self):
        self._assert_blocked("{'a': 1}")

    def test_disallowed_callable(self):
        self._assert_blocked("eval('1+1')")


class TestSideEffectsDoNotHappen(unittest.TestCase):
    """恶意表达式的副作用被拦截（哨兵文件保持原样）"""

    _counter = itertools.count()

    def _make_tmp_dir(self):
        # 每个用例独立的临时目录（makedirs 方式，兼容受限执行环境）
        tmp = Path(tempfile.gettempdir()) / f"st-design-safe-eval-{os.getpid()}-{next(self._counter)}"
        os.makedirs(tmp, exist_ok=True)
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp

    def test_command_execution_has_no_effect(self):
        tmp = self._make_tmp_dir()
        sentinel = tmp / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        payload = f"__import__('os').remove({str(sentinel)!r})"
        with self.assertRaises(UnsafeExpressionError):
            safe_eval_expression(payload, _env([]))
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_file_write_has_no_effect(self):
        tmp = self._make_tmp_dir()
        victim = tmp / "victim.txt"
        payload = f"open({str(victim)!r}, 'w').write('pwned')"
        with self.assertRaises(UnsafeExpressionError):
            safe_eval_expression(payload, _env([]))
        self.assertFalse(victim.exists())


class TestGeneratorIntegration(unittest.TestCase):
    """solve_constraint_range 集成：安全拦截向上阻断，合法失败仍软返回"""

    def setUp(self):
        self.gen = FactorValueGenerator()

    def test_unsafe_constraint_blocks(self):
        constraint = {
            "type": "calculate",
            "target": "y.shape",
            "sources": ["x.shape"],
            "expression": "__import__('os').system('echo pwned')",
        }
        with self.assertRaises(UnsafeExpressionError):
            self.gen.solve_constraint_range(constraint, {"x.shape": [2, 3]})

    def test_legit_failure_stays_soft(self):
        constraint = {
            "type": "calculate",
            "target": "y.value",
            "sources": ["x.value"],
            "expression": "sources[5] + 1",  # 越界：合法表达式求值失败
        }
        self.assertIsNone(self.gen.solve_constraint_range(constraint, {"x.value": 1}))

    def test_valid_constraint_still_solves(self):
        constraint = {
            "type": "calculate",
            "target": "y.shape",
            "sources": ["x.shape"],
            "expression": "sources[0][:3] + [2]",
        }
        self.assertEqual(
            self.gen.solve_constraint_range(constraint, {"x.shape": [1, 2, 3, 4]}),
            [1, 2, 3, 2],
        )


if __name__ == "__main__":
    unittest.main()
