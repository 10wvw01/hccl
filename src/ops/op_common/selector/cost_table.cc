/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cost_table.h"

#include <algorithm>
#include <new>
#include <set>

namespace ops_hccl {

HcclResult CostTableManager::FilterCMByConfig(CostModel &cm, CostTable &ct,
                                              const TopoInfoWithNetLayerDetails *topoInfo,
                                              const OpParam &opParam)
{
    HCCL_DEBUG("[FilterCMByConfig] filter cost model by config, opType=%d, algCount=%d.",
               static_cast<int>(opParam.opType), cm.count);
    switch (opParam.opType) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            return FilterAllReduce(cm, ct, topoInfo, opParam);
        default:
            HCCL_WARNING("[FilterCMByConfig] opType=%d not supported yet, keep all algorithms.",
                         static_cast<int>(opParam.opType));
            return HcclResult::HCCL_SUCCESS;
    }
}

namespace {
const std::set<std::string> &GetCcuMsAlgos()
{
    static const std::set<std::string> ccuMsAlgos = {
        "CcuAllReduceMesh1DOneShot", "CcuAllReduceMesh1D", "CcuAllReduceMesh2Die",
        "CcuAllreduceMesh2DieBigMs", "CcuAllReduceConcurrentMs", "CcuV2AllReduceOmniPipe2DMs"
    };
    return ccuMsAlgos;
}

OpExecuteConfig GetAlgEngine(const std::string &algName)
{
    if (algName.rfind("Aiv", 0) == 0) {
        return OpExecuteConfig::AIV;
    }
    if (algName.rfind("Ccu", 0) == 0) {
        return GetCcuMsAlgos().count(algName) > 0 ? OpExecuteConfig::CCU_MS : OpExecuteConfig::CCU_SCHED;
    }
    return OpExecuteConfig::AICPU;
}

bool IsEngineMatched(OpExecuteConfig config, OpExecuteConfig algEngine)
{
    if (config == OpExecuteConfig::CCU_MS) {
        return algEngine == OpExecuteConfig::CCU_MS || algEngine == OpExecuteConfig::CCU_SCHED ||
               algEngine == OpExecuteConfig::AICPU;
    }
    if (config == OpExecuteConfig::AIV || config == OpExecuteConfig::AIV_ONLY) {
        return algEngine == OpExecuteConfig::AIV;
    }
    if (config == OpExecuteConfig::CCU_SCHED) {
        return algEngine == OpExecuteConfig::CCU_SCHED;
    }
    if (config == OpExecuteConfig::AICPU) {
        return algEngine == OpExecuteConfig::AICPU;
    }
    return true;
}
} // namespace

HcclResult CostTableManager::FilterAllReduce(CostModel &cm, CostTable &ct,
                                             const TopoInfoWithNetLayerDetails *topoInfo,
                                             const OpParam &opParam)
{
    (void)topoInfo;
    HCCL_DEBUG("[FilterAllReduce] filter, algCount=%d, opExecuteConfig=%d.",
               cm.count, static_cast<int>(opParam.opExecuteConfig));

    ct.costs = nullptr;
    ct.count = 0;
    if (cm.count <= 0) {
        return HcclResult::HCCL_SUCCESS;
    }
    ct.costs = new (std::nothrow) AlgoCost[cm.count];
    if (ct.costs == nullptr) {
        HCCL_ERROR("[FilterAllReduce] alloc AlgoCost failed, count=%d.", cm.count);
        return HcclResult::HCCL_E_PARA;
    }

    u64 dataSize = opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];

    for (int i = 0; i < cm.count; ++i) {
        const char *algName = cm.costAlgoParams[i].algName;
        std::string name = (algName != nullptr) ? algName : "";

        OpExecuteConfig algEngine = GetAlgEngine(name);
        if (!IsEngineMatched(opParam.opExecuteConfig, algEngine)) {
            HCCL_DEBUG("[FilterAllReduce] algName=%s engine=%d filtered out by config=%d.", name.c_str(),
                       static_cast<int>(algEngine), static_cast<int>(opParam.opExecuteConfig));
            continue;
        }

        float cost = CalcAlgCost(name, dataSize, cm.costAlgoParams[i]);
        ct.costs[ct.count].algName = algName;
        ct.costs[ct.count].cost = cost;
        ++ct.count;
        HCCL_DEBUG("[FilterAllReduce] algName=%s engine=%d cost=%f.", name.c_str(),
                   static_cast<int>(algEngine), cost);
    }
    return HcclResult::HCCL_SUCCESS;
}

float CostTableManager::CalcAlgCost(const std::string &algName, u64 dataSize, const CostAlgoParams &algoParams) const
{
    AlgNetMeta meta;
    AlgNetMetaRegistry::Global()->Query(algName, meta);

    const CostModelParam *params = algoParams.param;
    std::vector<u32> groups = meta.groupSizes;
    if (groups.empty()) {
        groups.assign(static_cast<size_t>(algoParams.count), 1);
    }

    float cost = 0.0f;
    u32 idx = 0;
    for (u32 g = 0; g < groups.size() && idx < static_cast<u32>(algoParams.count); ++g) {
        float groupCost = 0.0f;
        for (u32 k = 0; k < groups[g] && idx < static_cast<u32>(algoParams.count); ++k, ++idx) {
            AlgNetType nt = (idx < meta.netTypes.size()) ? meta.netTypes[idx] : AlgNetType::MESH;
            float util = 1.0f;
            if (QueryUbUtil(nt, dataSize, util) != HcclResult::HCCL_SUCCESS) {
                util = 1.0f;
            }
            float segCost = (params[idx].A * util + params[idx].B) * static_cast<float>(dataSize) + params[idx].C;
            groupCost = (meta.intraGroupMode == CostAggMode::MAX) ? std::max(groupCost, segCost) : groupCost + segCost;
        }
        cost += groupCost;
    }
    HCCL_DEBUG("[CalcAlgCost] algName=%s intraGroupMode=%d groupCount=%zu cost=%f.", algName.c_str(),
               static_cast<int>(meta.intraGroupMode), groups.size(), cost);
    return cost;
}

HcclResult CostTableManager::CostTableGen(CostModel &cm, CostTable &ct,
                                          const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam)
{
    HCCL_DEBUG("[CostTableGen] generate cost table, algCount=%d.", cm.count);
    HcclResult ret = FilterCMByConfig(cm, ct, topoInfo, opParam);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CostTableGen] FilterCMByConfig failed, ret=%d.", static_cast<int>(ret));
    }
    return ret;
}

const std::vector<UbUtilEntry> CostTableManager::closUbUtilTable_ = {
    {0.125 * 1024 * 1024ULL, 0.02755f},
    {0.25 * 1024 * 1024ULL, 0.05357f},
    {0.5 * 1024 * 1024ULL, 0.10388f},
    {1 * 1024 * 1024ULL, 0.1855f},
    {2 * 1024 * 1024ULL, 0.3f},
    {4 * 1024 * 1024ULL, 0.4288f},
    {8 * 1024 * 1024ULL, 0.5302f},
    {16 * 1024 * 1024ULL, 0.568f},
    {32 * 1024 * 1024ULL, 0.6549f},
    {64 * 1024 * 1024ULL, 0.7184f},
    {128 * 1024 * 1024ULL, 0.7408f},
    {256 * 1024 * 1024ULL, 0.7644f}
};

const std::vector<UbUtilEntry> CostTableManager::meshUbUtilTable_ = {
    {1 * 1024 * 1024ULL, 0.7135f},
    {2 * 1024 * 1024ULL, 0.7758f},
    {4 * 1024 * 1024ULL, 0.8112f},
    {8 * 1024 * 1024ULL, 0.8301f},
    {16 * 1024 * 1024ULL, 0.84f},
    {32 * 1024 * 1024ULL, 0.8449f},
    {64 * 1024 * 1024ULL, 0.8475f},
    {128 * 1024 * 1024ULL, 0.8487f},
    {256 * 1024 * 1024ULL, 0.8494f}
};

CostTableManager::~CostTableManager() {}

HcclResult CostTableManager::QueryUbUtil(AlgNetType netType, u64 dataSize, float &utilization) const
{
    const std::vector<UbUtilEntry> &table = (netType == AlgNetType::CLOS) ? closUbUtilTable_ : meshUbUtilTable_;
    if (table.empty()) {
        HCCL_WARNING("[CostTableManager] ub util table empty, netType=%d dataSize=%llu.",
                     static_cast<int>(netType), dataSize);
        return HcclResult::HCCL_E_PARA;
    }
    auto it = std::lower_bound(table.begin(), table.end(), dataSize,
        [](const UbUtilEntry &e, u64 ds) { return e.upperBound < ds; });
    if (it == table.end()) {
        utilization = table.back().utilization;
    } else {
        utilization = it->utilization;
    }
    HCCL_DEBUG("[CostTableManager] QueryUbUtil netType=%d dataSize=%llu utilization=%f.",
               static_cast<int>(netType), dataSize, utilization);
    return HcclResult::HCCL_SUCCESS;
}

CostTableManager *CostTableManager::Global()
{
    static CostTableManager *globalCostTableManager = new CostTableManager;
    return globalCostTableManager;
}

HcclResult CostTableManager::Load()
{
    const std::lock_guard<std::mutex> lock(mu_);
    HCCL_DEBUG("[CostTableManager] load cost table, count=%d.", costTable_.count);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CostTableManager::Query(const std::string &algName, u64 dataSize, double &cost) const
{
    const std::lock_guard<std::mutex> lock(mu_);
    for (int i = 0; i < costTable_.count; ++i) {
        if (algName == costTable_.costs[i].algName) {
            cost = costTable_.costs[i].cost;
            return HcclResult::HCCL_SUCCESS;
        }
    }
    HCCL_WARNING("[CostTableManager] no entry matched algName=%s dataSize=%llu.", algName.c_str(), dataSize);
    return HcclResult::HCCL_E_PARA;
}

} // namespace ops_hccl
