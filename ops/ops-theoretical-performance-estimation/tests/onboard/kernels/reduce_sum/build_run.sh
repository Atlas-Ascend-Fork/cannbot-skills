#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# reduce_sum 直调 ST：构建 + 跑用例（shape 对齐 reduce_sum_cases.json 的 fp32 用例）。
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build"; BIN="$BUILD/reduce_sum_direct"
ARCH="${NPU_ARCH:-dav-3510}"; MODE="${1:-all}"; ITERS="${ITERS:-20}"
build() { cmake -S "$HERE" -B "$BUILD" -DNPU_ARCH="$ARCH"; cmake --build "$BUILD" -j; echo "BUILD_OK $BIN"; }
CASES=(
    "reduce_sum_s_512x2048    512   2048"
    "reduce_sum_m_8192x4096   8192  4096"
    "reduce_sum_l_16384x8192  16384 8192"
)
run() {
    [ -x "$BIN" ] || { echo "先 build"; exit 1; }
    for row in "${CASES[@]}"; do
        read -r cid rows cols <<<"$row"
        printf "[%s] " "$cid"
        "$BIN" --rows "$rows" --cols "$cols" --iters "$ITERS"
    done
}
case "$MODE" in build) build ;; run) run ;; all) build; run ;; *) echo "用法: $0 {build|run|all}"; exit 1 ;; esac
