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

# membw 带宽标定：HBM(大 buffer) + L2(小 buffer) × read/write/copy。前置 source set_env.sh。
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; BUILD="$HERE/build"; BIN="$BUILD/membw"
ARCH="${NPU_ARCH:-dav-3510}"; MODE="${1:-all}"
build(){ cmake -S "$HERE" -B "$BUILD" -DNPU_ARCH="$ARCH"; cmake --build "$BUILD" -j; echo "BUILD_OK $BIN"; }
run(){
  [ -x "$BIN" ] || { echo "先 build"; exit 1; }
  echo "== HBM (512MB, 远超 L2) =="
  for m in read write copy; do "$BIN" --mode $m --mb 512 --iters 30; done
  echo "== L2 (16MB, 驻 L2) =="
  for m in read write copy; do "$BIN" --mode $m --mb 16 --iters 200; done
}
case "$MODE" in build) build;; run) run;; all) build; run;; *) echo "用法: $0 {build|run|all}";exit 1;; esac
