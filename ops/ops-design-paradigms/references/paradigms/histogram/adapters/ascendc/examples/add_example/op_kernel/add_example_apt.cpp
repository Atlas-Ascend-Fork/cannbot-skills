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
 * \file add_example.cpp
 * \brief
 */

#include "arch35/add_example.h"

// dtype 由编译期 DTYPE_X1（输入 x1 的 dtype，x1/x2/y 同 dtype）决定，由 opc 工具按
// op def 的 DataType profile 注入；不通过 tiling key（schMode）分派 dtype。
// schMode 分派调度模式，定义见 add_example_tiling_key.h；当前仅 ELEMWISE 单值。
// 后续新增调度模式时，在 ADD_EXAMPLE_SCH_MODE_ELEMWISE 分支后追加
// else if constexpr (schMode == 新模式) 分支即可。
template <uint32_t schMode>
__global__ __aicore__ void add_example(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AddExampleTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddExampleTilingData, tilingData, tiling);
    if constexpr (schMode == ADD_EXAMPLE_SCH_MODE_ELEMWISE) {
        NsAddExample::Process<DTYPE_X1>(x1, x2, y, &tilingData);
    }
}
