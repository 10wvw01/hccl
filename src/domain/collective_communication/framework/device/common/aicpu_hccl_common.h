/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_HCCL_COMMON_H__
#define __AICPU_HCCL_COMMON_H__

#include "common/aicpu_hccl_def.h"
#include "profiling_manager_device.h"
#include "dtype_common.h"
#include "log.h"
#include "sal_pub.h"

template <typename T>
inline T MathCeil(T num1, T num2)
{
    if (num2 == 0) {
        return num1;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
inline T MathFloor(T num1, T num2)
{
    if (num2 == 0) {
        return num1;
    }
    return num1 / num2;
}

template <typename T>
inline T AlignUp(T num1, T num2)
{
    return MathCeil(num1, num2) * num2;
}

template <typename T>
inline T AlignDown(T num1, T num2)
{
    return MathFloor(num1, num2) * num2;
}

inline uint32_t DataUnitSize(HcclDataType dataType)
{
    if (dataType >= HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR("[dataUnitSize]data type[%s] out of range[%d, %d]", GetDataTypeEnumStr(dataType).c_str(),
            HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_RESERVED - 1);
        return 0;
    }

    return SIZE_TABLE[dataType];
}

#define CLOCK_MONOTONIC 1
#define NSEC_PER_SEC 1000000000U

inline u64 GetCurCpuTimestamp(bool isProfTime = false)
{
#ifndef CCL_LLT
    if (isProfTime && dfx::ProfilingManager::GetProfL1State()) {
        uint64_t cntvct;
        AsmCntvc(cntvct);
        return cntvct;
    } else {
        struct timespec timestamp;
        (void)clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp);
        return static_cast<u64>((timestamp.tv_sec * NSEC_PER_SEC) + (timestamp.tv_nsec));
    }
#else
    struct timespec timestamp;
    (void)clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp);
    return static_cast<u64>((timestamp.tv_sec * NSEC_PER_SEC) + (timestamp.tv_nsec));
#endif
}

#define RECORD_PROF_TIME(VAR)                                                                              \
    do {                                                                                                   \
        AicpuComContext *commctx__ = AicpuGetComContext();                                                 \
        if (!MC2AicpuUtils::NeedRecordTimeTaken(*commctx__)) { break; }                                       \
        uint32_t recordIndex = commctx__->acprof[g_proxLoopCnt].workCnt;                                  \
        recordIndex = (recordIndex >= AC_MAX_PROF_COMM_CNT) ? (AC_MAX_PROF_COMM_CNT - 1) : recordIndex; \
        commctx__->acprof[g_proxLoopCnt].commLoop[recordIndex].VAR = GetCurCpuTimestamp(true);                \
    } while (0)

#define KFC_GET_START_TIME() ((MC2AicpuUtils::NeedRecordTimeTaken(*AicpuGetComContext())) ? GetCurCpuTimestamp() : 0)

#define RECORD_FILL_SQE_TIME(START_TIME)                                                                   \
    do {                                                                                                   \
        AicpuComContext *commctx__ = AicpuGetComContext();                                                 \
        if (!MC2AicpuUtils::NeedRecordTimeTaken(*commctx__)) { break; }                                       \
        commctx__->acprof[g_proxLoopCnt].fillSqeTimes += GetCurCpuTimestamp() - startTime;                 \
    } while (0)

#endif // __AICPU_HCCL_COMMON_HPP__