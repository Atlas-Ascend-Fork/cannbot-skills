#!/bin/bash
# --------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with it.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the repository root for the full text of the License.
# --------------------------------------------------------------------------------------------------

set -euo pipefail

ENTRY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${ENTRY_DIR}/../common/mssanitizer_runner.py"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "用法: bash run_mssanitizer_geir.sh <project_dir> [cann_root]" >&2
    exit 1
fi

python3 "${RUNNER}" geir "$@"
