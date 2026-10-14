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

# rms_norm 直调 ST：构建 + 跑全载/非全载用例（shape 对齐 rms_norm_cases.json）。
# 当前 Ascend950 环境使用 `build_run.sh all` 编译并运行、打印 RESULT 行；
# 使用 `build_run.sh build` / `build_run.sh run` 可分别编译和运行。
#
# 前置：已 source <toolkit>/set_env.sh（$ASCEND_HOME_PATH 生效）。
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build"
BIN="$BUILD/rms_norm_direct"
ARCH="${NPU_ARCH:-dav-3510}"
MODE="${1:-all}"
ITERS="${ITERS:-20}"

build() {
    echo "== cmake configure (NPU_ARCH=$ARCH) =="
    cmake -S "$HERE" -B "$BUILD" -DNPU_ARCH="$ARCH"
    echo "== build =="
    cmake --build "$BUILD" -j
    echo "BUILD_OK $BIN"
}

# case_id  variant  rows  cols  —— 与 references/examples/rms_norm_cases.json 同源（fp32 全载/非全载）
CASES=(
    "rms_norm_s_512x4096           full   512   4096"
    "rms_norm_m_4096x4096          full   4096  4096"
    "rms_norm_l_16384x8192         full   16384 8192"
    "rms_norm_xl_2048x16384_nonresident split 2048 16384 4096"
)

run() {
    [ -x "$BIN" ] || { echo "未找到可执行 $BIN，先 build"; exit 1; }
    echo "== run cases (iters=$ITERS) =="
    for row in "${CASES[@]}"; do
        read -r cid variant rows cols chunk <<<"$row"
        printf "[%s] " "$cid"
        if [ -n "$chunk" ]; then
            "$BIN" --variant "$variant" --rows "$rows" --cols "$cols" --chunk "$chunk" --iters "$ITERS"
        else
            "$BIN" --variant "$variant" --rows "$rows" --cols "$cols" --iters "$ITERS"
        fi
    done
}

case "$MODE" in
    build) build ;;
    run)   run ;;
    all)   build; run ;;
    *) echo "用法: $0 {build|run|all}"; exit 1 ;;
esac
