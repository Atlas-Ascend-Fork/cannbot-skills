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
 * \file add_example_tiling.cpp
 * \brief Tiling implementation for add_example operator
 */

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "../../op_kernel/arch35/add_example_tiling_data.h"
#include "../../op_kernel/arch35/add_example_tiling_key.h"

static const gert::Shape kOneDim = {1};

static const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return kOneDim;
    }
    return inShape;
}

namespace optiling {

constexpr int64_t PER_CORE_MIN = 1024;
constexpr uint32_t DCACHE_SIZE = 32 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;

// ========== input index constants ==========
static constexpr int32_t kX1Idx = 0;
static constexpr int32_t kX2Idx = 1;

struct AddExampleCompileInfo {};

// ========== GetPlatformInfo ==========
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ========== ValidateDtype (SE §2.2 + §2.6) ==========
static ge::graphStatus ValidateDtype(gert::TilingContext* context, ge::DataType& dataType)
{
    auto inputDesc = context->GetInputDesc(kX1Idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    OP_CHECK_IF(dataType != ge::DT_FLOAT && dataType != ge::DT_INT32,
                OP_LOGE(context, "x1 dtype must be float or int32, got %d", dataType), return ge::GRAPH_FAILED);

    // SE §2.6: x2 dtype == x1 dtype
    auto x2Desc = context->GetInputDesc(kX2Idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
    OP_CHECK_IF(x2Desc->GetDataType() != dataType,
                OP_LOGE(context, "x2 dtype must match x1, got %d vs %d", x2Desc->GetDataType(), dataType),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ========== ValidateShape (SE §2.5) ==========
static ge::graphStatus ValidateShape(gert::TilingContext* context, int64_t& totalNum)
{
    auto x1Input = context->GetInputShape(kX1Idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Input);
    auto x1Shape = EnsureNotScalar(x1Input->GetStorageShape());

    auto x2Input = context->GetInputShape(kX2Idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Input);
    auto x2Shape = EnsureNotScalar(x2Input->GetStorageShape());

    // SE §2.5 C1: ndim ≤ 8
    int32_t ndim = static_cast<int32_t>(x1Shape.GetDimNum());
    OP_CHECK_IF(ndim > 8, OP_LOGE(context, "x1 ndim %d exceeds max 8", ndim), return ge::GRAPH_FAILED);

    // SE §2.5 C2: x1 shape == x2 shape
    OP_CHECK_IF(x2Shape != x1Shape, OP_LOGE(context, "x2 shape must equal x1 shape"), return ge::GRAPH_FAILED);

    totalNum = x1Shape.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

// ========== ValidateInputs ==========
static ge::graphStatus ValidateInputs(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    // SE §2.2 + §2.6
    OP_CHECK_IF(ValidateDtype(context, dataType) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateDtype failed"),
                return ge::GRAPH_FAILED);
    // SE §2.5
    OP_CHECK_IF(ValidateShape(context, totalNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateShape failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ========== DumpTilingData (DFX) ==========
static void DumpTilingData(gert::TilingContext* context, const AddExampleTilingData* tiling)
{
    OP_LOGD(context, "AddExampleTilingData: totalNum=%ld, needCoreNum=%ld", tiling->totalNum, tiling->needCoreNum);
}

// ========== GetWorkspaceSize ==========
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

// ========== ComputeTiling ==========
static ge::graphStatus ComputeTiling([[maybe_unused]] gert::TilingContext* context, AddExampleTilingData* tiling,
                                     int64_t totalNum, int64_t coreNum)
{
    tiling->totalNum = totalNum;
    int64_t blockFactor = (totalNum + coreNum - 1) / coreNum;
    if (blockFactor < PER_CORE_MIN) {
        blockFactor = PER_CORE_MIN;
    }
    tiling->needCoreNum = (totalNum + blockFactor - 1) / blockFactor;
    if (tiling->needCoreNum > coreNum) {
        tiling->needCoreNum = coreNum;
    }
    if (tiling->needCoreNum <= 0) {
        tiling->needCoreNum = 1;
    }
    return ge::GRAPH_SUCCESS;
}

// ========== GetTilingKey ==========
static uint64_t GetTilingKey()
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(ADD_EXAMPLE_SCH_MODE_ELEMWISE));
}

// ========== TilingFunc ==========
static ge::graphStatus AddExampleTilingFunc(gert::TilingContext* context)
{
    // 1. validate inputs
    int64_t totalNum;
    ge::DataType dataType;
    OP_CHECK_IF(ValidateInputs(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateInputs failed"), return ge::GRAPH_FAILED);

    // 2. get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 3. compute tiling
    AddExampleTilingData* tiling = context->GetTilingData<AddExampleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(AddExampleTilingData), 0, sizeof(AddExampleTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ComputeTiling(context, tiling, totalNum, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ComputeTiling error"), return ge::GRAPH_FAILED);

    DumpTilingData(context, tiling);

    // 4. get workspace size
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    // 5. set block dim and local memory
    context->SetBlockDim(tiling->needCoreNum);
    OP_CHECK_IF((ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE),
                OP_LOGE(context, "ubSize %lu <= DCACHE_SIZE + STATIC_UB_ESTIMATE", ubSize), return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context, "SetLocalMemorySize failed, ubSize=%lu", ubSize),
                return ge::GRAPH_FAILED);

    // 6. set tiling key
    context->SetTilingKey(GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

// ========== TilingParse ==========
static ge::graphStatus TilingParseForAddExample([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// ========== Registration ==========
IMPL_OP_OPTILING(AddExample).Tiling(AddExampleTilingFunc).TilingParse<AddExampleCompileInfo>(TilingParseForAddExample);

} // namespace optiling
