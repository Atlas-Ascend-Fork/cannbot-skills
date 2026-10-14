# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# reduce_sum 直调 ST 工程（自包含）。构建见 build_run.sh。
# 用法: cmake -S . -B build -DNPU_ARCH=dav-3510 && cmake --build build -j
cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED NPU_ARCH OR NPU_ARCH STREQUAL "")
    set(NPU_ARCH dav-3510)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ascend_bootstrap.cmake)
find_package(ASC REQUIRED)
project(reduce_sum_direct LANGUAGES C CXX ASC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(reduce_sum_direct reduce_sum_direct.asc)
target_compile_options(reduce_sum_direct PRIVATE
    "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
    "$<$<COMPILE_LANGUAGE:ASC>:-O2>"
)
target_link_directories(reduce_sum_direct PRIVATE ${ASCEND_DIR}/lib64)
target_link_libraries(reduce_sum_direct
    m
    dl
    platform
    tiling_api
)
