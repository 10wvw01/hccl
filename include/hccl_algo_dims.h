/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ALGO_DIMS_H
#define HCCL_ALGO_DIMS_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===== OpType 维度（集合通信算子类型）===== */
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

/* ===== Engine 维度（5 种）===== */
typedef enum {
    HCCL_ENGINE_AICPU = 0,
    HCCL_ENGINE_CCU_MS,
    HCCL_ENGINE_CCU_SCHED,
    HCCL_ENGINE_AIV,
    HCCL_ENGINE_DPU,
    HCCL_ENGINE_COUNT
} hcclEngineType_t;

/* ===== Executor 维度（5 种）===== */
typedef enum {
    HCCL_EXEC_SEQUENCE = 0,
    HCCL_EXEC_SOLE,
    HCCL_EXEC_PARALLEL,
    HCCL_EXEC_PIPILINE,
    HCCL_EXEC_CONCUR,
    HCCL_EXEC_COUNT
} hcclExecutorType_t;

/* ===== Template 维度（6 种）===== */
typedef enum {
    HCCL_TPL_MESH = 0,
    HCCL_TPL_NHR,
    HCCL_TPL_MESH_TWO_SHOT,
    HCCL_TPL_MESH_ONE_SHOT,
    HCCL_TPL_MESH_CHUNK,
    HCCL_TPL_MESH_2DIE,
    HCCL_TPL_COUNT
} hcclTemplateType_t;

/* ===== 维度名字对：PascalCase（算法名拼接用）+ user（JSON 配置用）===== */
typedef struct {
    const char *pascal;
    const char *user;
} hcclDimName_t;

/* ===== 维度名字对定义（header-only，单一权威来源）===== */
static const hcclDimName_t g_hcclEngines[HCCL_ENGINE_COUNT] = {
    {"Aicpu", "aicpu"}, {"CcuMs", "ccu_ms"}, {"CcuSched", "ccu_sched"},
    {"Aiv", "aiv"}, {"Dpu", "dpu"},
};

static const hcclDimName_t g_hcclExecutors[HCCL_EXEC_COUNT] = {
    {"Sequence", "sequence"}, {"Sole", "sole"}, {"Parallel", "parallel"},
    {"Pipiline", "pipiline"}, {"Concur", "concur"},
};

static const hcclDimName_t g_hcclTemplates[HCCL_TPL_COUNT] = {
    {"Mesh", "mesh"}, {"NHR", "NHR"}, {"MeshTwoShot", "mesh_two_shot"},
    {"MeshOneShot", "mesh_one_shot"}, {"MeshChunk", "mesh_chunk"}, {"Mesh2die", "mesh_2die"},
};

/* ===== 查找函数（static inline，header-only）===== */

static inline const char *HcclEngineToUser(int engine)
{
    if (engine < 0 || engine >= HCCL_ENGINE_COUNT) {
        return NULL;
    }
    return g_hcclEngines[engine].user;
}

static inline const char *HcclExecutorToUser(int executor)
{
    if (executor < 0 || executor >= HCCL_EXEC_COUNT) {
        return NULL;
    }
    return g_hcclExecutors[executor].user;
}

static inline const char *HcclTemplateToUser(int tmpl)
{
    if (tmpl < 0 || tmpl >= HCCL_TPL_COUNT) {
        return NULL;
    }
    return g_hcclTemplates[tmpl].user;
}

static inline int HcclUserToEngine(const char *user)
{
    if (user == NULL) {
        return -1;
    }
    for (int i = 0; i < HCCL_ENGINE_COUNT; i++) {
        if (strcmp(user, g_hcclEngines[i].user) == 0) {
            return i;
        }
    }
    return -1;
}

static inline int HcclUserToExecutor(const char *user)
{
    if (user == NULL) {
        return -1;
    }
    for (int i = 0; i < HCCL_EXEC_COUNT; i++) {
        if (strcmp(user, g_hcclExecutors[i].user) == 0) {
            return i;
        }
    }
    return -1;
}

static inline int HcclUserToTemplate(const char *user)
{
    if (user == NULL) {
        return -1;
    }
    for (int i = 0; i < HCCL_TPL_COUNT; i++) {
        if (strcmp(user, g_hcclTemplates[i].user) == 0) {
            return i;
        }
    }
    return -1;
}

#ifdef __cplusplus
}
#endif

/* ===== OpType 查找函数（switch-case，编译器 -Wswitch-enum 强制完整性）===== */

static inline const char *HcclOpTypeToPascal(int opType)
{
    switch (opType) {
        case HCCL_OP_ALLREDUCE:      return "AllReduce";
        case HCCL_OP_ALLGATHER:      return "AllGather";
        case HCCL_OP_BROADCAST:      return "Broadcast";
        case HCCL_OP_REDUCE:         return "Reduce";
        case HCCL_OP_REDUCE_SCATTER: return "ReduceScatter";
        case HCCL_OP_SCATTER:        return "Scatter";
        case HCCL_OP_ALLTOALL:       return "AlltoAll";
        case HCCL_OP_ALLTOALLV:      return "AlltoAllV";
        default:                     return NULL;
    }
}

static inline int HcclOpTypeFromName(const char *name)
{
    if (name == NULL) {
        return HCCL_OP_INVALID;
    }
    if (strcmp(name, "allreduce") == 0)      return HCCL_OP_ALLREDUCE;
    if (strcmp(name, "allgather") == 0)      return HCCL_OP_ALLGATHER;
    if (strcmp(name, "broadcast") == 0)      return HCCL_OP_BROADCAST;
    if (strcmp(name, "reduce") == 0)         return HCCL_OP_REDUCE;
    if (strcmp(name, "reduce_scatter") == 0) return HCCL_OP_REDUCE_SCATTER;
    if (strcmp(name, "scatter") == 0)        return HCCL_OP_SCATTER;
    if (strcmp(name, "alltoall") == 0)       return HCCL_OP_ALLTOALL;
    if (strcmp(name, "alltoallv") == 0)      return HCCL_OP_ALLTOALLV;
    return HCCL_OP_INVALID;
}

#endif /* HCCL_ALGO_DIMS_H */
