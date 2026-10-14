# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 在 project() 之前把工具链指向 Bisheng（ASC 语言所需）。依赖已 source set_env.sh
# 注入的 $ASCEND_HOME_PATH；asc-config.cmake 由 CMAKE_PREFIX_PATH(=$ASCEND_HOME_PATH/lib64/cmake) 解析。
if(NOT DEFINED ENV{ASCEND_HOME_PATH} OR "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    message(FATAL_ERROR "ASCEND_HOME_PATH 未设置：先 source <toolkit>/set_env.sh")
endif()
set(ASCEND_DIR $ENV{ASCEND_HOME_PATH})

execute_process(COMMAND uname -m OUTPUT_VARIABLE _arch OUTPUT_STRIP_TRAILING_WHITESPACE)
set(SYSTEM_PREFIX "${_arch}-linux")
set(BISHENG "${ASCEND_DIR}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng")
if(NOT EXISTS "${BISHENG}")
    set(BISHENG "${ASCEND_DIR}/tools/ccec_compiler/bin/bisheng")
endif()
message(STATUS "Ascend toolkit: ${ASCEND_DIR}")
message(STATUS "Bisheng: ${BISHENG}")

set(CMAKE_PREFIX_PATH "${ASCEND_DIR}/lib64/cmake;${CMAKE_PREFIX_PATH}")
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "${BISHENG}" CACHE FILEPATH "C compiler" FORCE)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${BISHENG}" CACHE FILEPATH "CXX compiler" FORCE)
endif()
if(NOT DEFINED CMAKE_LINKER)
    set(CMAKE_LINKER "${BISHENG}" CACHE FILEPATH "Linker" FORCE)
endif()
