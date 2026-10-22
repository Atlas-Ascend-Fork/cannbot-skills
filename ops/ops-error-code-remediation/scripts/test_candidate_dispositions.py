#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Behavioral regressions for discovered-but-unresolved remediation candidates."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from check_candidate_dispositions import check, initialize


class DispositionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "sample.cpp"
        self.source.write_text(
            'OP_CHECK_NULL_WITH_CONTEXT(context, ptr);\nOP_LOGE(context, "bad shape");\n',
            encoding="utf-8",
        )
        self.ledger = initialize(self.root)

    def decision(self, index, status, entry=None, line=None):
        row = self.ledger["decisions"][index]
        row.update(
            status=status,
            root_cause="tested cause",
            reason="see report decision",
            evidence={
                key: "report.md#verified-path"
                for key in (
                    "macro_definition",
                    "call_chain",
                    "registration",
                    "build",
                    "negative_test",
                    "first_error",
                    "retention_basis",
                    "removal_basis",
                )
            },
            after={
                "path": "sample.cpp",
                "line": line or index + 1,
                "entry": entry or self.ledger["baseline"]["findings"][index]["entry"],
            },
        )
        return row

    def test_initial_candidates_are_not_completed(self):
        self.assertEqual(2, len(self.ledger["decisions"]))
        self.assertTrue(check(self.ledger))

    def test_patch_only_fixes_explicit_log_but_leaves_null_macro(self):
        self.source.write_text(
            'OP_CHECK_NULL_WITH_CONTEXT(context, ptr);\nOP_LOGE_FOR_INVALID_SHAPE(context, "x", "a", "b");\n',
            encoding="utf-8",
        )
        self.decision(1, "FIXED", "OP_LOGE_FOR_INVALID_SHAPE")
        errors = check(self.ledger)
        self.assertTrue(any("OP_CHECK_NULL_WITH_CONTEXT" in error for error in errors))

    def test_claiming_unchanged_null_check_fixed_is_rejected(self):
        self.decision(0, "FIXED")
        self.decision(1, "RETAINED")
        self.assertTrue(
            any("source is unchanged" in error for error in check(self.ledger))
        )

    def test_changed_source_but_null_check_still_present_is_rejected(self):
        self.source.write_text(
            self.source.read_text() + "// unrelated edit\n", encoding="utf-8"
        )
        self.decision(0, "FIXED")
        self.decision(1, "RETAINED")
        self.assertTrue(
            any("still a governance candidate" in error for error in check(self.ledger))
        )

    def test_justified_retention_and_real_fix_reconcile(self):
        self.source.write_text(
            'OP_CHECK_NULL_WITH_CONTEXT(context, ptr);\nOP_LOGE_FOR_INVALID_SHAPE(context, "x", "a", "b");\n',
            encoding="utf-8",
        )
        self.decision(0, "RETAINED")
        self.decision(1, "FIXED", "OP_LOGE_FOR_INVALID_SHAPE")
        self.assertEqual([], check(self.ledger))

    def test_blanket_retention_without_evidence_fails(self):
        for i in range(2):
            self.decision(i, "RETAINED")["evidence"] = {}
        self.assertTrue(check(self.ledger))

    def test_blocked_is_explicit_but_not_closed(self):
        self.decision(0, "BLOCKED")
        self.decision(1, "RETAINED")
        self.assertTrue(any("blocked:" in error for error in check(self.ledger)))

    def test_missing_duplicate_or_foreign_decision_fails_closed(self):
        for change in (
            lambda rows: rows.pop(),
            lambda rows: rows.append(copy.deepcopy(rows[0])),
            lambda rows: rows[0].update(id="unknown"),
        ):
            ledger = copy.deepcopy(self.ledger)
            change(ledger["decisions"])
            with self.assertRaises(ValueError):
                check(ledger)

    def test_changed_line_requires_explicit_mapping(self):
        self.source.write_text("// moved\n" + self.source.read_text(), encoding="utf-8")
        self.decision(0, "RETAINED")
        self.decision(1, "RETAINED")
        self.assertTrue(check(self.ledger))
        self.ledger["decisions"][0]["after"]["line"] = 2
        self.ledger["decisions"][1]["after"]["line"] = 3
        self.assertEqual([], check(self.ledger))

    def test_real_removal_passes_but_new_candidate_blocks(self):
        self.source.write_text("int valid = 1;\n", encoding="utf-8")
        for i in range(2):
            self.decision(i, "REMOVED")["after"] = None
        self.assertEqual([], check(self.ledger))
        (self.root / "new.cpp").write_text(
            "OP_CHECK_NULL_WITH_CONTEXT(context, other);\n", encoding="utf-8"
        )
        self.assertTrue(any("new.cpp" in error for error in check(self.ledger)))

    def test_deprecated_cannot_be_retained(self):
        self.source.write_text('REPORT("EZ0001");\n', encoding="utf-8")
        self.ledger = initialize(self.root)
        self.decision(0, "RETAINED", "REPORT")
        self.assertTrue(any("deprecated code" in error for error in check(self.ledger)))

    def test_tampered_baseline_and_missing_root_fail_closed(self):
        ledger = copy.deepcopy(self.ledger)
        ledger["baseline"]["scope"]["excludes"].append("hidden")
        with self.assertRaises(ValueError):
            check(ledger)
        self.source.unlink()
        self.root.rmdir()
        with self.assertRaises(OSError):
            check(self.ledger)

    def test_cli_refuses_overwrite_and_returns_nonzero_on_pending(self):
        output = self.root / "decisions.json"
        script = str(Path(__file__).with_name("check_candidate_dispositions.py"))

        def run(*args):
            return subprocess.run(
                [sys.executable, script, *args], capture_output=True, text=True
            )

        self.assertEqual(
            0, run("init", "--root", str(self.root), "--output", str(output)).returncode
        )
        original = output.read_bytes()
        self.assertEqual(
            2, run("init", "--root", str(self.root), "--output", str(output)).returncode
        )
        self.assertEqual(original, output.read_bytes())
        self.assertEqual(1, run("check", "--ledger", str(output)).returncode)
        output.write_text(json.dumps({}), encoding="utf-8")
        self.assertEqual(2, run("check", "--ledger", str(output)).returncode)


if __name__ == "__main__":
    unittest.main(verbosity=2)
