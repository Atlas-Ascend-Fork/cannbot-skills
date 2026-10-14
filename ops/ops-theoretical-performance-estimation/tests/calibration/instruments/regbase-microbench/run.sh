#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CASE_NAME="${1:-add_fp32_independent}"
ITERATIONS="${2:-20000}"
UNROLL="${3:-16}"
BLOCK_DIM="${4:-1}"

[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "ASCEND_HOME_PATH is not set" >&2; exit 1; }
source "${ASCEND_HOME_PATH}/set_env.sh"

mkdir -p build output
cmake -S . -B build
cmake --build build -j4

RAW="output/raw_${CASE_NAME}.jsonl"
REPORT="output/report_${CASE_NAME}.json"

./build/regbase_microbench "${CASE_NAME}" "${ITERATIONS}" "${UNROLL}" "${BLOCK_DIM}" | tee "${RAW}"
python3 scripts/analyze_results.py --manifest config/calibration_cases.json --input "${RAW}" --output "${REPORT}"
cat "${REPORT}"
