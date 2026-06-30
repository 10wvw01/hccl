/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
 */

#ifndef REFACTOR_TEMPLATE_DEBUG_CCU_TYPES_H
#define REFACTOR_TEMPLATE_DEBUG_CCU_TYPES_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// planner-only debug target 不执行 CCU kernel；这些 typedef 只用于让 alg_param.h 的资源结构可编译。
typedef enum {
    CCU_SUCCESS = 0,
    CCU_E_PARA = 1,
    CCU_E_PTR = 2,
    CCU_E_INTERNAL = 4,
    CCU_E_NOT_SUPPORT = 5,
    CCU_E_NOT_FOUND = 6,
    CCU_E_UNAVAIL = 7,
    CCU_E_RESERVED = 9216
} CcuResult;

typedef enum {
    CCU_CONDITION_EQ = 0,
    CCU_CONDITION_NE = 1,
} CcuConditionType;

typedef uint64_t CcuLoop;
typedef uint64_t CcuLoopGroup;
typedef uint64_t CcuLoopExecutors;

typedef struct {
    uint64_t addrOffset;
    uint64_t iterNum;
} CcuLoopConfig;

typedef struct {
    uint32_t cloneNum;
    uint32_t cloneLoopOffset;
    uint32_t addrOffset;
    uint32_t ccuBufferOffset;
    uint32_t eventOffset;
} CcuLoopGroupConfig;

typedef uint64_t CcuInsHandle;
typedef uint64_t CcuKernelHandle;
typedef uint64_t CcuVariableHandle;
typedef uint64_t CcuAddressHandle;
typedef uint64_t CcuEventHandle;
typedef uint64_t CcuBufferHandle;
typedef uint64_t CcuLocalAddrHandle;
typedef uint64_t CcuRemoteAddrHandle;
typedef void *CcuKernelArg;
typedef CcuResult (*CcuKernelFunc)(CcuKernelArg arg);

#ifdef __cplusplus
}
#endif

#endif
