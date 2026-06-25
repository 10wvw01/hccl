/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_SAL_H
#define HCCL_INC_SAL_H

#include <climits>
#include <chrono>
#include <hccl/hccl_types.h>
#include <hccl/base.h>

constexpr int HCCL_BASE_DECIMAL = 10; // 10进制字符串转换

using HcclUs = std::chrono::steady_clock::time_point;
#define TIME_NOW() ({ std::chrono::steady_clock::now(); })
#define DURATION_US(x) (std::chrono::duration_cast<std::chrono::microseconds>(x))

struct AivProfilingData {
    uint64_t selectorUs = 0;
    uint64_t resAcquireUs = 0;
    uint64_t dfxRegUs = 0;
    uint64_t aivEntranceUs = 0;
    uint64_t cacheLogicUs = 0;
    uint64_t cacheCheckUs = 0;
    uint64_t orchestrateUs = 0;
    uint64_t cacheStoreUs = 0;
    uint64_t reportUs = 0;
    uint64_t templatePrepUs = 0;
    uint64_t kernelLaunchUs = 0;
    uint64_t aclrtLaunchUs = 0;
    uint32_t kernelLaunchCount = 0;
};
extern thread_local AivProfilingData g_aivProfiling;
#define AIV_PROF_BEGIN(field) HcclUs field##_t0 = TIME_NOW()
#define AIV_PROF_END(field) g_aivProfiling.field += DURATION_US(TIME_NOW() - field##_t0).count()

HcclResult  SalStrToULong(const std::string str, int base, u32 &val);
HcclResult SalStrToDouble(const std::string str, double &val);

HcclResult IsAllDigit(const char *strNum);
u32 SalStrLen(const char *s, u32 maxLen = INT_MAX);

#define weak_alias(name, aliasname) _weak_alias(name, aliasname)
#define _weak_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((weak, alias(#name)))

#endif  // HCCL_INC_SAL_H