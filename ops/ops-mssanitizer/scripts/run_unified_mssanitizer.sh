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
# 统一 mssanitizer 检测入口脚本
# 自动识别算子类型并调用对应检测脚本

set -euo pipefail

# 动态获取 SKILL 目录（相对于脚本所在位置）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"

# 参数解析
PROJECT_OR_SCRIPT="${1:-}"
CANN_ROOT="${2:-${ASCEND_HOME_PATH:-}}"

if [ -z "$PROJECT_OR_SCRIPT" ]; then
    echo "Usage: $0 <project_or_script> [cann_root]"
    echo ""
    echo "参数说明:"
    echo "  project_or_script: 算子工程目录或测试脚本路径"
    echo "  cann_root: CANN 安装根路径（可选，默认 \$ASCEND_HOME_PATH）"
    exit 1
fi

if [ -z "$CANN_ROOT" ]; then
    echo "错误：未指定 CANN 根路径，且 ASCEND_HOME_PATH 未设置"
    echo "请通过以下方式之一指定 CANN 路径:"
    echo "  1. 传递第二个参数: $0 <project> <cann_root>"
    echo "  2. 设置环境变量: source <cann_root>/set_env.sh"
    exit 1
fi

# 判断算子类型
OPERATOR_TYPE="unknown"

# 检查是否为目录
if [ -d "$PROJECT_OR_SCRIPT" ]; then
    PROJECT_DIR="$PROJECT_OR_SCRIPT"

    # AscendC ops 算子仓：存在 op_graph/ 或 examples/
    if [ -d "${PROJECT_DIR}/op_graph" ] || [ -d "${PROJECT_DIR}/examples" ]; then
        OPERATOR_TYPE="ascendc_ops"
        echo "识别为 AscendC ops 算子仓 (C++ 模式)"
    # AscendC 自定义算子仓：无上述目录，但存在 op_host/
    elif [ -d "${PROJECT_DIR}/op_host" ] && [ ! -d "${PROJECT_DIR}/op_graph" ]; then
        OPERATOR_TYPE="ascendc_custom"
        echo "识别为 AscendC 自定义算子仓 (Python 模式)"
    else
        # 无法识别，提示用户手动指定
        echo "无法自动识别算子类型，请检查目录结构"
        echo ""
        echo "AscendC ops 算子仓应包含：op_graph/ 或 examples/"
        echo "AscendC 自定义算子仓应包含：op_host/（无 op_graph/）"
        echo "Triton 算子应使用 Python 脚本路径而非目录"
        exit 1
    fi

elif [ -f "$PROJECT_OR_SCRIPT" ]; then
    TEST_SCRIPT="$PROJECT_OR_SCRIPT"

    # Triton 算子：Python 文件包含 @triton.jit
    if grep -q "@triton.jit" "$TEST_SCRIPT" 2>/dev/null; then
        OPERATOR_TYPE="triton"
        echo "识别为 Triton 算子 (Python 模式)"
    # AscendC 自定义算子：Python 文件包含 torch.ops.customize
    elif grep -q "torch.ops.customize" "$TEST_SCRIPT" 2>/dev/null; then
        OPERATOR_TYPE="ascendc_custom"
        echo "识别为 AscendC 自定义算子 (Python 模式)"
    else
        # 无法识别，提示用户手动指定
        echo "无法自动识别算子类型，请检查脚本内容"
        echo ""
        echo "Triton 算子脚本应包含：@triton.jit 装饰器"
        echo "AscendC 自定义算子脚本应包含：torch.ops.customize 调用"
        exit 1
    fi

else
    echo "错误：路径不存在: $PROJECT_OR_SCRIPT"
    exit 1
fi

# 调用对应检测脚本
case "$OPERATOR_TYPE" in
    "ascendc_ops")
        echo "调用 AscendC C++ 模式检测脚本..."
        bash "${SKILL_DIR}/scripts/ascendc/run_mssanitizer_geir.sh" "$PROJECT_DIR" "$CANN_ROOT"
        ;;

    "ascendc_custom")
        if [ -d "$PROJECT_OR_SCRIPT" ]; then
            # 目录模式：需要生成测试脚本
            echo "检测到算子工程目录，需要先生成测试脚本"
            echo ""
            echo "请使用以下命令生成测试脚本:"
            echo "  python3 ${SKILL_DIR}/scripts/ascendc/gen_test_script.py \\"
            echo "      --operator <operator_name> \\"
            echo "      --dtypes float16 float32 \\"
            echo "      --output ${PROJECT_DIR}/mssanitizer_test/<operator>_test.py"
            echo ""
            echo "然后使用测试脚本路径重新运行:"
            echo "  bash $0 ${PROJECT_DIR}/mssanitizer_test/<operator>_test.py $CANN_ROOT"
            exit 1
        else
            # 脚本模式：直接检测
            echo "调用 AscendC Python 模式检测脚本..."
            bash "${SKILL_DIR}/scripts/ascendc/run_mssanitizer.sh" "$TEST_SCRIPT" "$CANN_ROOT"
        fi
        ;;

    "triton")
        echo "调用 Triton 检测脚本..."
        bash "${SKILL_DIR}/scripts/triton/run_mssanitizer_triton.sh" "$TEST_SCRIPT" "$CANN_ROOT"
        ;;

    *)
        echo "错误：未知的算子类型: $OPERATOR_TYPE"
        exit 1
        ;;
esac

echo ""
echo "检测完成，请查看 mssanitizer_logs/ 目录下的报告文件"
