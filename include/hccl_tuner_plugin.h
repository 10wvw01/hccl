/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TUNER_PLUGIN_H_
#define HCCL_TUNER_PLUGIN_H_

#include <stdint.h>
#include <stddef.h>
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===== 常量 ===== */

#define HCCL_TUNER_API_VERSION 1

#define HCCL_NUM_ENGINES   5
#define HCCL_NUM_EXECUTORS 7
#define HCCL_NUM_TEMPLATES 16
#define HCCL_ALGO_IGNORE   (-1.0f)

/* ===== 枚举 ===== */

typedef enum {
    HCCL_OP_ALLREDUCE = 0,
    HCCL_OP_ALLGATHER,
    HCCL_OP_BROADCAST,
    HCCL_OP_REDUCE,
    HCCL_OP_REDUCE_SCATTER,
    HCCL_OP_SCATTER,
    HCCL_OP_ALLTOALL,
    HCCL_OP_ALLTOALLV,
    HCCL_OP_INVALID = 255
} hcclOpType_t;

/* ===== 通信域信息（per-comm，init 时传入）===== */

typedef struct {
    uint32_t nRanks;
    uint32_t nNpusPerServer;
    uint32_t nServers;
    uint32_t nPods;
    uint32_t nSuperPods;
    const char* commName;
    uint64_t bufferSize;
    uint32_t structSize;
} hcclTunerCommInfo_t;

/* ===== 集合通信调用信息（per-op，getCollInfo 时传入）===== */

typedef struct {
    hcclOpType_t collType;
    size_t nBytes;
    HcclDataType dataType;
    int nEngine;
    int nExecutor;
    int nTemplate;
    uint32_t structSize;
} hcclTunerCollInfo_t;

/* ===== Host 函数表 ===== */

typedef void (*hcclTunerLogFn_t)(int level, const char* file, int line, const char* fmt, ...);

typedef struct {
    HcclResult (*ctxCreate)(HcclComm comm, const char* ctxTag, uint64_t size, void** ctx);
    HcclResult (*ctxGet)(HcclComm comm, const char* ctxTag, void** ctx, uint64_t* size);
    HcclResult (*ctxDestroy)(HcclComm comm, const char* ctxTag);
    hcclTunerLogFn_t logFunction;
    uint32_t structSize;
} hcclTunerHostFunctions_t;

/* ===== 回调函数签名 ===== */

typedef HcclResult (*hcclTunerInit_t)(
    HcclComm comm,
    const hcclTunerCommInfo_t* commInfo,
    const hcclTunerHostFunctions_t* hostFuncs
);

typedef HcclResult (*hcclTunerGetCollInfo_t)(
    HcclComm comm,
    const hcclTunerCollInfo_t* collInfo,
    float* collCostTable
);

/* ===== 函数表 ===== */

typedef struct {
    hcclTunerInit_t        init;
    hcclTunerGetCollInfo_t getCollInfo;
} hcclTunerFuncs_t;

/* ===== 插件描述符 ===== */

typedef struct {
    uint32_t apiVersion;
    uint32_t hcclVersion;
    const char* pluginName;
    uint32_t pluginVersion;
} hcclPluginDescriptor_t;

/* ===== 入口符号（插件 .so 导出）===== */

extern hcclPluginDescriptor_t hcclTunerPlugin;
HcclResult hcclTunerGetFuncs(hcclTunerFuncs_t* funcs);

/* ===== Helper 宏 ===== */

#define HCCL_TUNER_SELECT_ALGO(table, e, ex, t) do { \
    for (int _i = 0; _i < HCCL_NUM_ENGINES * HCCL_NUM_EXECUTORS * \
                             HCCL_NUM_TEMPLATES; _i++) \
        (table)[_i] = HCCL_ALGO_IGNORE; \
    (table)[(e) * HCCL_NUM_EXECUTORS * HCCL_NUM_TEMPLATES \
           + (ex) * HCCL_NUM_TEMPLATES + (t)] = 0.0f; \
} while(0)

#define HCCL_TUNER_DISABLE_ALGO(table, e, ex, t) do { \
    (table)[(e) * HCCL_NUM_EXECUTORS * HCCL_NUM_TEMPLATES \
           + (ex) * HCCL_NUM_TEMPLATES + (t)] = HCCL_ALGO_IGNORE; \
} while(0)

#define HCCL_TUNER_DISABLE_ENGINE(table, e) do { \
    for (int _ex = 0; _ex < HCCL_NUM_EXECUTORS; _ex++) \
        for (int _t = 0; _t < HCCL_NUM_TEMPLATES; _t++) \
            (table)[(e) * HCCL_NUM_EXECUTORS * HCCL_NUM_TEMPLATES \
                   + _ex * HCCL_NUM_TEMPLATES + _t] = HCCL_ALGO_IGNORE; \
} while(0)

#define HCCL_TUNER_DISABLE_EXECUTOR(table, ex) do { \
    for (int _e = 0; _e < HCCL_NUM_ENGINES; _e++) \
        for (int _t = 0; _t < HCCL_NUM_TEMPLATES; _t++) \
            (table)[_e * HCCL_NUM_EXECUTORS * HCCL_NUM_TEMPLATES \
                   + (ex) * HCCL_NUM_TEMPLATES + _t] = HCCL_ALGO_IGNORE; \
} while(0)

/* ===== 日志级别 ===== */

#define HCCL_TUNER_LOG_ERROR 0
#define HCCL_TUNER_LOG_WARN  1
#define HCCL_TUNER_LOG_INFO  2
#define HCCL_TUNER_LOG_DEBUG 3

#ifdef __cplusplus
}
#endif

#endif /* HCCL_TUNER_PLUGIN_H_ */
