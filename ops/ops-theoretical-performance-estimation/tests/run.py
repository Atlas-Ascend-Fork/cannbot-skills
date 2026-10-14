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

"""离线 ST 单一入口：一把跑完 offline/ 全部测试，聚合退出码（非 0 = 有失败）。

可当 CI / pre-commit 门禁：
    python3 tests/run.py
onboard/（理论 vs 实测，需 950 实测 CSV）不在此入口，单独跑 tests/onboard/compare.py。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFFLINE = [
    "offline/test_builders.py",  # 自洽性
    "offline/test_design_conformance.py",  # 设计符合性（builder vs flow）
    "offline/test_negative.py",  # 负向：门禁会咬
    "offline/test_public_instruction_rates.py",  # asc-devkit 公开吞吐表与别名回归
    "offline/test_overlap_model.py",  # 引擎 overlap 标定回归
    "offline/test_engine_knobs.py",  # 引擎各建模旋钮回归
    "offline/test_core_balance.py",  # 核间尾块均衡：分核单位(非 tile_count)、resident 非整除行
    "offline/test_compare.py",  # onboard/compare.py 判定逻辑自测（合成数据）
    "calibration/check_calibration.py",  # 标定一致性：JSON 常量 ⟷ 上板 bench 实测（防凑数）
]


def main():
    rc = 0
    for t in OFFLINE:
        print(f"\n===== {t} =====")
        r = subprocess.run([sys.executable, str(HERE / t)])
        rc |= r.returncode or 0
    print("\n" + ("✅ ALL OFFLINE PASS" if rc == 0 else "❌ SOME OFFLINE TESTS FAILED"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
