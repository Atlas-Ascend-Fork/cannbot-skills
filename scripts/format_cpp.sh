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

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")

FORMAT_DIR="${1:-${REPO_ROOT}}"
if [[ ! -d ${FORMAT_DIR} ]]; then
    echo "Error: directory not found: ${FORMAT_DIR}" >&2
    exit 1
fi
# Normalize to an absolute path: find prints paths rooted at the string passed
# on the command line, so the prune patterns below must be built from the
# resolved FORMAT_DIR, not REPO_ROOT.
FORMAT_DIR=$(cd "${FORMAT_DIR}" && pwd)

CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"
STYLE_FILE="$REPO_ROOT/.clang-format"

if [ ! -f "$STYLE_FILE" ]; then
    echo "Error: .clang-format not found at $STYLE_FILE" >&2
    exit 1
fi

if ! command -v "$CLANG_FORMAT" &>/dev/null; then
    echo "Error: $CLANG_FORMAT not found. Install clang-format or set CLANG_FORMAT env var." >&2
    exit 1
fi

echo "Using $CLANG_FORMAT ..."
"$CLANG_FORMAT" --version

EXCLUDE_DIRS=(
    "build"
    "build_out"
    "third_party"
    ".git"
)

FIND_EXPR=()
for dir in "${EXCLUDE_DIRS[@]}"; do
    # Prune paths are anchored to FORMAT_DIR (the actual traversal root);
    # anchoring to REPO_ROOT breaks exclusion when FORMAT_DIR is customized.
    FIND_EXPR+=( -path "$FORMAT_DIR/$dir" -prune -o )
done

mapfile -d '' FILES < <(
    find "$FORMAT_DIR" "${FIND_EXPR[@]}" \
        \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.cc' -o -name '*.hh' -o -name '*.cxx' -o -name '*.hxx' -o -name '*.asc' \) \
        -type f -print0
)

TOTAL="${#FILES[@]}"
echo "Found $TOTAL files to format."

if [ "$TOTAL" -eq 0 ]; then
    echo "No files to format."
    exit 0
fi

"$CLANG_FORMAT" -i --style=file --verbose "${FILES[@]}" 2>&1

echo ""
echo "Done! Formatted $TOTAL files."
