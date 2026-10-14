#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""List implemented and planned RegBase microbenchmark cases."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "config/calibration_cases.json").read_text(encoding="utf-8")
    )
    print("Implemented cases:")
    for case in manifest["implemented_cases"]:
        print(f"  {case['case_name']:24s} {case['op']:12s} {case['expected_class']}")
    print("\nPlanned calibration cases:")
    for case in manifest.get("planned_calibration_cases", []):
        print(f"  {case['case_name']:24s} {case['status']}")
    print("\nUnknown document ops to measure after calibration:")
    for op in manifest.get("unknown_doc_ops_to_measure_after_calibration", []):
        print(f"  {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
