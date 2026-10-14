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

# vchain 直调 ST：构建 + 跑 vector-bound 用例(shape/K 对齐 vchain_cases.json)。
# 前置：source <toolkit>/set_env.sh。 用法: build_run.sh {build|run|all}
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build"; BIN="$BUILD/vchain_direct"
ARCH="${NPU_ARCH:-dav-3510}"; MODE="${1:-all}"; ITERS="${ITERS:-20}"

build() { cmake -S "$HERE" -B "$BUILD" -DNPU_ARCH="$ARCH"; cmake --build "$BUILD" -j; echo "BUILD_OK $BIN"; }

# case_id  rows  cols  K  —— 同源 references/examples/vchain_cases.json
CASES=(
    "vchain_m_4096x2048_k64   4096 2048 64"
    "vchain_m_4096x2048_k128  4096 2048 128"
    "vchain_l_8192x4096_k64   8192 4096 64"
)
run() {
    [ -x "$BIN" ] || { echo "先 build"; exit 1; }
    for row in "${CASES[@]}"; do
        read -r cid rows cols K <<<"$row"; printf "[%s] " "$cid"
        "$BIN" --rows "$rows" --cols "$cols" --K "$K" --iters "$ITERS"
    done
}
case "$MODE" in
    build) build ;; run) run ;; all) build; run ;;
    *) echo "用法: $0 {build|run|all}"; exit 1 ;;
esac
