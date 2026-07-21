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

#include <new>

namespace ops_hccl {

HcclResult CostTableManager::FilterCMByConfig(CostModel &cm, CostTable &ct,
                                              const TopoInfoWithNetLayerDetails *topoInfo,
                                              const OpParam &opParam)
{
    (void)ct;
    HCCL_DEBUG("[FilterCMByConfig] filter cost model by config, opType=%d, algCount=%d.",
               static_cast<int>(opParam.opType), cm.count);
    switch (opParam.opType) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            return FilterAllReduce(cm, topoInfo, opParam);
        default:
            HCCL_WARNING("[FilterCMByConfig] opType=%d not supported yet, keep all algorithms.",
                         static_cast<int>(opParam.opType));
            return HcclResult::HCCL_SUCCESS;
    }
}

HcclResult CostTableManager::FilterAllReduce(CostModel &cm, CostTable &ct,
                                             const TopoInfoWithNetLayerDetails *topoInfo,
                                             const OpParam &opParam)
{
    constexpr u32 AIV_MAX_RANK_SIZE = 512;
    HCCL_DEBUG("[FilterAllReduce] filter, algCount=%d.", cm.count);

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
    HcclDataType dataType = opParam.DataDes.dataType;
    bool isInt8 = (dataType == HcclDataType::HCCL_DATA_TYPE_INT8);
    bool is64Bit = (dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
                    dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
                    dataType == HcclDataType::HCCL_DATA_TYPE_FP64);
    bool isProd = (opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD);
    bool isUboe3Level = (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_3 && topoInfo->level2Uboe);
    u32 rankSize = topoInfo->userRankSize;
    const auto &layerSize = topoInfo->netLayerDetails.localNetInsSizeOfLayer;
    bool singleCardPerBox = (!layerSize.empty() && layerSize[0] == 1);

    for (int i = 0; i < cm.count; ++i) {
        const char *algName = cm.costAlgoParams[i].algName;
        std::string name = (algName != nullptr) ? algName : "";
        bool applicable = true;

        if (name == "InsAllReduceMesh1DOneShot") {
            applicable = (topoInfo->level0Topo == Level0Shape::MESH_1D);
        } else if (name == "InsAllReduceNHR") {
            applicable = (topoInfo->topoLevelNums > 1) ||
                         (topoInfo->level0Topo == Level0Shape::CLOS) ||
                         topoInfo->Level1Nhr || singleCardPerBox;
        } else if (name == "CcuAllReduceMesh1D") {
            applicable = !isInt8 && !is64Bit && !isProd &&
                         (topoInfo->level0Topo == Level0Shape::MESH_1D);
        } else if (name == "AivAllReduceMesh1DOneShot") {
            applicable = !isInt8 && !is64Bit && !isProd && !isUboe3Level &&
                         (rankSize <= AIV_MAX_RANK_SIZE) &&
                         (topoInfo->level0Topo == Level0Shape::MESH_1D);
        } else {
            applicable = true;
        }

        if (!applicable) {
            HCCL_DEBUG("[FilterAllReduce] algName=%s filtered out.", name.c_str());
            continue;
        }

        float cost = 0.0f;
        const CostModelParam *params = cm.costAlgoParams[i].param;
        for (int j = 0; j < cm.costAlgoParams[i].count; ++j) {
            cost += (params[j].A + params[j].B) * static_cast<float>(dataSize) + params[j].C;
        }
        ct.costs[ct.count].algName = algName;
        ct.costs[ct.count].cost = cost;
        ++ct.count;
        HCCL_DEBUG("[FilterAllReduce] algName=%s cost=%f.", name.c_str(), cost);
    }
    return HcclResult::HCCL_SUCCESS;
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

CostTableManager *CostTableManager::Global()
{
    static CostTableManager *globalCostTableManager = new CostTableManager;
    return globalCostTableManager;
}

CostTableManager::~CostTableManager() {}

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
