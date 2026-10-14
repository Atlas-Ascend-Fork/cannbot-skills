/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

constexpr uint32_t REGBASE_BENCH_MAGIC = 0x52424D42U; // RBMB
constexpr uint32_t REGBASE_BENCH_VERSION = 1;
constexpr uint32_t REGBASE_BENCH_WORDS = 16;
constexpr uint32_t REGBASE_BENCH_F32_ELEMS = 64;
constexpr uint32_t REGBASE_BENCH_F32_LANES = 4;
constexpr uint32_t REGBASE_BENCH_F32_BUFFER_ELEMS = REGBASE_BENCH_F32_ELEMS * REGBASE_BENCH_F32_LANES;
constexpr uint32_t REGBASE_BENCH_F16_ELEMS = 128;
constexpr uint32_t REGBASE_BENCH_F16_BUFFER_ELEMS = REGBASE_BENCH_F16_ELEMS * REGBASE_BENCH_F32_LANES;

enum RegbaseBenchCaseId : uint32_t {
    BENCH_ADD_FP32 = 1,
    BENCH_MUL_FP32 = 2,
    BENCH_CAST_F32_TO_F16 = 3,
    BENCH_MULADDDST_FP32 = 4,
    BENCH_SQRT_FP32 = 5,
    BENCH_REDUCESUM_FP32 = 6,
    BENCH_SUB_FP32 = 7,
    BENCH_MAX_FP32 = 8,
    BENCH_MIN_FP32 = 9,
    BENCH_ABS_FP32 = 10,
    BENCH_RELU_FP32 = 11,
    BENCH_NEG_FP32 = 12,
    BENCH_LOG_FP32 = 13,
    BENCH_LOG2_FP32 = 14,
    BENCH_LOG10_FP32 = 15,
    BENCH_LEAKYRELU_FP32 = 16,
    BENCH_PRELU_FP32 = 17,
    BENCH_ABSSUB_FP32 = 18,
    BENCH_EXPSUB_FP32 = 19,
    BENCH_MULSCAST_F32_TO_F16 = 20,
    BENCH_ADDC_U32 = 21,
    BENCH_SUBC_U32 = 22,
    BENCH_INTERLEAVE_U32 = 23,
    BENCH_DEINTERLEAVE_U32 = 24,
    BENCH_PACK_U32_TO_U16 = 25,
    BENCH_UNPACK_U16_TO_U32 = 26,
    BENCH_EXP_FP32 = 27,
    BENCH_EXP_THEN_SUB_FP32 = 28,
    BENCH_ADD_U32 = 29,
    BENCH_ADD_S32 = 30,
    BENCH_EXP_FP16 = 31,
    BENCH_EXP_ADD2_FP32 = 32,
    BENCH_EXP_ADD4_FP32 = 33,
    BENCH_EXP_ADD8_FP32 = 34,
    BENCH_EXP_MUL_ADD_FP32 = 35,
    BENCH_SQRT_ADD4_FP32 = 36,
    BENCH_EXP_ADD4_INDEP_FP32 = 37,
    BENCH_EXP_ADD8_INDEP_FP32 = 38,
    BENCH_DIV_FP32 = 39,
    BENCH_REDUCEMAX_FP32 = 40,
    BENCH_CAST_RT_F16 = 41
};

enum RegbaseBenchStatus : uint32_t {
    BENCH_STATUS_OK = 0,
    BENCH_STATUS_UNSUPPORTED_CASE = 1
};

struct RegbaseBenchTilingData {
    uint32_t caseId;
    uint32_t iterations;
    uint32_t unroll;
    uint32_t reserved;
};

enum RegbaseBenchRecordWord : uint32_t {
    REC_MAGIC = 0,
    REC_VERSION = 1,
    REC_CASE_ID = 2,
    REC_BLOCK_IDX = 3,
    REC_STATUS = 4,
    REC_ITERATIONS = 5,
    REC_UNROLL = 6,
    REC_LOGICAL_OPS_PER_UNROLL = 7,
    REC_ELEMENTS_PER_INST = 8,
    REC_EMPTY_CYCLES = 9,
    REC_MEASURED_CYCLES = 10,
    REC_BODY_CYCLES = 11,
    REC_SINK = 12,
    REC_RESERVED0 = 13,
    REC_RESERVED1 = 14,
    REC_RESERVED2 = 15
};
