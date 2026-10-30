/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_example.h
 * \brief
 */
#ifndef ADD_EXAMPLE_H
#define ADD_EXAMPLE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "add_example_tiling_data.h"
#include "add_example_tiling_key.h"

namespace NsAddExample {

using namespace AscendC;

static constexpr uint32_t THREAD_NUM = 512;

template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpComputeSimt(uint64_t count, __gm__ T* x1,
                                                                               __gm__ T* x2, __gm__ T* y)
{
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x); index < count;
         index += (blockDim.x * gridDim.x)) {
        y[index] = x1[index] + x2[index];
    }
}

template <typename T>
__aicore__ inline void Process(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const AddExampleTilingData* tilingData)
{
    int64_t count = tilingData->totalNum;
    __gm__ T* x1_gm = (__gm__ T*)x1;
    __gm__ T* x2_gm = (__gm__ T*)x2;
    __gm__ T* y_gm = (__gm__ T*)y;
    asc_vf_call<OpComputeSimt<T>>(dim3(THREAD_NUM), count, x1_gm, x2_gm, y_gm);
}
} // namespace NsAddExample
#endif // ADD_EXAMPLE_H
