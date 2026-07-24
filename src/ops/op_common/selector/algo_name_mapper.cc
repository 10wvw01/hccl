/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "algo_name_mapper.h"

#include "log.h"

namespace ops_hccl {

/* ===== 单例 ===== */
AlgoNameMapper *AlgoNameMapper::Global()
{
    static AlgoNameMapper *instance = new AlgoNameMapper;
    return instance;
}

/* ===== 维度值表 ===== */
const AlgoNameMapper::DimEntry AlgoNameMapper::g_engines[] = {
    {"Aicpu", "aicpu"}, {"CcuMs", "ccu_ms"}, {"CcuSched", "ccu_sched"},
    {"Aiv", "aiv"}, {"Dpu", "dpu"},
};
const int AlgoNameMapper::g_engineCount = 5;

const AlgoNameMapper::DimEntry AlgoNameMapper::g_executors[] = {
    {"Sequence", "sequence"}, {"Sole", "sole"}, {"Parallel", "parallel"},
    {"Pipiline", "pipiline"}, {"Concur", "concur"},
};
const int AlgoNameMapper::g_executorCount = 5;

const AlgoNameMapper::DimEntry AlgoNameMapper::g_templates[] = {
    {"MeshOneShot", "mesh_one_shot"}, {"MeshTwoShot", "mesh_two_shot"},
    {"MeshChunk", "mesh_chunk"}, {"Mesh2die", "mesh_2die"},
    {"Mesh", "mesh"}, {"NHR", "NHR"},
};
const int AlgoNameMapper::g_templateCount = 6;

const char * const AlgoNameMapper::g_opTypes[] = {
    "AllReduce", "AllGather", "ReduceScatter", "Broadcast",
    "AlltoAllV", "AlltoAll", "Reduce", "Scatter",
};
const int AlgoNameMapper::g_opTypeCount = 8;

/* ===== opType 枚举 → PascalCase ===== */
std::string AlgoNameMapper::OpTypeToPascal(int opType)
{
    if (opType >= 0 && opType < g_opTypeCount) {
        return g_opTypes[opType];
    }
    return "";
}

/* ===== 构建 2D 表（30 条）===== */
void AlgoNameMapper::BuildMap2D()
{
    for (int ex = 0; ex < g_executorCount; ex++) {
        for (int t = 0; t < g_templateCount; t++) {
            std::string key = std::string(g_executors[ex].pascal)
                            + g_templates[t].pascal;
            map2D_[key] = {g_executors[ex].user, g_templates[t].user};
        }
    }
    HCCL_DEBUG("[AlgoNameMapper] 2D map built, %zu entries.", map2D_.size());
}

/* ===== 2D 查表（仅 init 时调用）===== */
bool AlgoNameMapper::Lookup2D(const std::string &algName,
                               const std::string &opTypePascal,
                               AlgoDims &dims) const
{
    /* 1. 定位 optype，拆出 engine 和 execTpl */
    size_t pos = algName.find(opTypePascal);
    if (pos == std::string::npos) {
        return false;
    }

    /* 2. engine = optype 前面，查 5 项 engine 表 */
    std::string enginePascal = algName.substr(0, pos);
    dims.engineUser = nullptr;
    for (int i = 0; i < g_engineCount; i++) {
        if (enginePascal == g_engines[i].pascal) {
            dims.engineUser = g_engines[i].user;
            break;
        }
    }
    if (dims.engineUser == nullptr) {
        return false;
    }

    /* 3. execTpl = optype 后面，查 30 项 2D 表 */
    std::string execTpl = algName.substr(pos + opTypePascal.size());
    auto it = map2D_.find(execTpl);
    if (it == map2D_.end()) {
        return false;
    }
    dims.executorUser = it->second.first;
    dims.templateUser = it->second.second;
    return true;
}

/* ===== init：建表 + 缓存所有算法（一次性）===== */
void AlgoNameMapper::Init(const AlgoRegInfo *algos, int count)
{
    BuildMap2D(); /* 30 条，<0.1ms */

    for (int i = 0; i < count; ++i) {
        const std::string &algName = algos[i].algName;
        std::string opTypePascal = OpTypeToPascal(algos[i].opType);
        if (opTypePascal.empty()) {
            HCCL_WARNING("[AlgoNameMapper] unknown opType=%d, skip algName=%s.",
                         algos[i].opType, algName.c_str());
            continue;
        }

        AlgoDims dims = {};
        if (Lookup2D(algName, opTypePascal, dims)) {
            cache_[algName] = dims;
        } else {
            HCCL_WARNING("[AlgoNameMapper] lookup failed, algName=%s.", algName.c_str());
        }
    }
    HCCL_DEBUG("[AlgoNameMapper] init done, cached %zu algorithms.", cache_.size());
}

/* ===== enrich：填 3D 名到 entry 数组（每次 op）===== */
void AlgoNameMapper::Enrich(hcclTunerAlgoEntry_t *entries, int count)
{
    if (entries == nullptr) {
        return;
    }
    for (int i = 0; i < count; i++) {
        auto it = cache_.find(entries[i].algName);
        if (it != cache_.end()) {
            entries[i].engineName = it->second.engineUser;
            entries[i].executorName = it->second.executorUser;
            entries[i].templateName = it->second.templateUser;
        } else {
            entries[i].engineName = "";
            entries[i].executorName = "";
            entries[i].templateName = "";
        }
        entries[i].structSize = sizeof(hcclTunerAlgoEntry_t);
    }
}

/* ===== query：按需查单个算法 ===== */
bool AlgoNameMapper::Query(const std::string &algName, AlgoDims &dims) const
{
    auto it = cache_.find(algName);
    if (it == cache_.end()) {
        return false;
    }
    dims = it->second;
    return true;
}

} /* namespace ops_hccl */
