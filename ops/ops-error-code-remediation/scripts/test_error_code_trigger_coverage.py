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
"""Regression tests for the shared E*9*** trigger coverage contract."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from audit_operator_error_codes import audit_source, unmatched_candidates
from error_code_triggers import DEFAULT_EXCLUDES, SOURCE_SUFFIXES
from scan_error_code_candidates import scan


HEADER = r"""
#define OP_LOGE(context, ...) REPORT_INNER_ERR_MSG("EZ9999", __VA_ARGS__)
#define OP_CHECK_NULL_WITH_CONTEXT(context, value) OP_LOGE(context, "null")
#define OPS_CHECK_NULL_WITH_CONTEXT(context, value) OP_LOGE(context, "null")
#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, value, ret) OP_LOGE(context, "null")
#define OPS_CHECK_NOT_NULL_WITH_CONTEXT(context, value) OP_LOGE(context, "null")
#define MOE_OPS_CHECK_NULL_WITH_CONTEXT_RET(context, value, ret) OP_LOGE(context, "null")
#define VECTOR_INNER_ERR_REPORT_TILIING(context, ...) OP_LOGE(context, __VA_ARGS__)
#define OPS_REPORT_VECTOR_INNER_ERR(context, ...) OP_LOGE(context, __VA_ARGS__)
#define MOE_VECTOR_INNER_ERR_REPORT_TILIING(context, ...) OP_LOGE(context, __VA_ARGS__)
#define OPS_REPORT_CUBE_INNER_ERR(context, ...) OP_LOGE(context, __VA_ARGS__)
#define CUBE_INNER_ERR_REPORT_PLUGIN(context, ...) OP_LOGE(context, __VA_ARGS__)
#define ONNX_PLUGIN_LOGE(context, ...) OP_LOGE(context, __VA_ARGS__)
#define GE_OP_LOGE_WITH_REPORT(context, ...) OP_LOGE(context, __VA_ARGS__)
#define INFER_AXIS_TYPE_ERR_REPORT(context, ...) OP_LOGE(context, __VA_ARGS__)
#define TR_EXECUTE(...) OP_LOGE(context, "tiling execution failed")
#define REGISTER_OP_TILING_V3_WITH_VECTOR(...) VECTOR_INNER_ERR_REPORT_TILIING(context, "v3")
#define REGISTER_OP_TILING_V4_CUSTOM(...) VECTOR_INNER_ERR_REPORT_TILIING(context, "v4")
#define PROJECT_REPORT_ALIAS(context, ...) OP_LOGE(context, __VA_ARGS__)
#define CHECK_RET(condition, err) do { if (!(condition)) { return err; } } while (0)
"""

SOURCE = r"""
#include "log.h"
int AuditAll(void *context, void *value) {
  OP_CHECK_NULL_WITH_CONTEXT(context, value);
  OPS_CHECK_NULL_WITH_CONTEXT(context, value);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, value, ACLNN_ERR_INNER_NULLPTR);
  OPS_CHECK_NOT_NULL_WITH_CONTEXT(context, value);
  MOE_OPS_CHECK_NULL_WITH_CONTEXT_RET(context, value, ACLNN_ERR_INNER_NULLPTR);
  VECTOR_INNER_ERR_REPORT_TILIING(context, "vector");
  OPS_REPORT_VECTOR_INNER_ERR(context, "vector");
  MOE_VECTOR_INNER_ERR_REPORT_TILIING(context, "vector");
  OPS_REPORT_CUBE_INNER_ERR(context, "cube");
  CUBE_INNER_ERR_REPORT_PLUGIN(context, "cube");
  ONNX_PLUGIN_LOGE(context, "onnx");
  GE_OP_LOGE_WITH_REPORT(context, "ge");
  INFER_AXIS_TYPE_ERR_REPORT(context, "axis");
  TR_EXECUTE(DoTiling());
  REGISTER_OP_TILING_V3_WITH_VECTOR(Op, Func, Key, Optional);
  REGISTER_OP_TILING_V4_CUSTOM(Op, Func, Parse, Info);
  PROJECT_REPORT_ALIAS(context, "project alias");
  ReportErrMessage("EZ1001", {});
  ReportInterErrMessage("EZ9999", "internal");
  ReportErrorMessage(ACLNN_ERR_RUNTIME_ERROR, "runtime");
  CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);
  OP_LOGE(
      ACL_ERROR_FAILURE,
      "lower layer failed");
  OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bad argument");
  return ACLNN_ERR_INNER;
}
"""


class TriggerCoverageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "log.h").write_text(textwrap.dedent(HEADER), encoding="utf-8")
        self.source = self.root / "sample.cpp"
        self.source.write_text(textwrap.dedent(SOURCE), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_scanner_and_auditor_share_all_entry_families(self) -> None:
        candidates = scan(self.root, DEFAULT_EXCLUDES, SOURCE_SUFFIXES, False)
        records, _, _ = audit_source(self.source, self.root, [self.root], {"EZ1001"})
        entries = {row.entry.split("::")[-1] for row in records}
        expected = {
            "OP_CHECK_NULL_WITH_CONTEXT",
            "OPS_CHECK_NULL_WITH_CONTEXT",
            "OPS_CHECK_NULL_WITH_CONTEXT_RET",
            "OPS_CHECK_NOT_NULL_WITH_CONTEXT",
            "MOE_OPS_CHECK_NULL_WITH_CONTEXT_RET",
            "VECTOR_INNER_ERR_REPORT_TILIING",
            "OPS_REPORT_VECTOR_INNER_ERR",
            "MOE_VECTOR_INNER_ERR_REPORT_TILIING",
            "OPS_REPORT_CUBE_INNER_ERR",
            "CUBE_INNER_ERR_REPORT_PLUGIN",
            "ONNX_PLUGIN_LOGE",
            "GE_OP_LOGE_WITH_REPORT",
            "INFER_AXIS_TYPE_ERR_REPORT",
            "TR_EXECUTE",
            "REGISTER_OP_TILING_V3_WITH_VECTOR",
            "REGISTER_OP_TILING_V4_CUSTOM",
            "PROJECT_REPORT_ALIAS",
            "ReportErrMessage",
            "ReportInterErrMessage",
            "ReportErrorMessage",
            "CHECK_RET",
            "OP_LOGE",
        }
        self.assertTrue(expected.issubset(entries), expected - entries)
        self.assertEqual([], unmatched_candidates(candidates, records))

    def test_errno_mapping_and_multiline_call(self) -> None:
        records, _, _ = audit_source(self.source, self.root, [self.root], {"EZ1001"})
        runtime = next(row for row in records if row.entry == "ReportErrorMessage")
        self.assertEqual("EZ9903", runtime.final_code)
        self.assertEqual("INTERNAL_E9", runtime.code_status)

        failure = next(row for row in records if row.entry == "OP_LOGE" and row.final_code == "EZ9999")
        self.assertEqual("INTERNAL_REPORTED", failure.report_status)

        parameter = next(row for row in records if row.entry == "OP_LOGE" and row.final_code == "EZ1001")
        self.assertEqual("DOCUMENTED_FORMAL", parameter.code_status)
        self.assertEqual("FORMAL_REPORTED", parameter.report_status)


if __name__ == "__main__":
    unittest.main()
