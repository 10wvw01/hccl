/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cost_model.h"

#include <new>
#include <memory>

#include "coll_alg_v2_exec_registry.h"

namespace ops_hccl {

AllAlgos *GetAllAlgos()
{
    static AllAlgos globalAllAlgos{nullptr, 0, 0};
    return &globalAllAlgos;
}

HcclResult AddAlgToAllAlgos(HcclCMDType opType, const char *algName, const char *executorName,
                            const char **templateName, int templateNum)
{
    AllAlgos *allAlgos = GetAllAlgos();
    if (allAlgos->count >= allAlgos->capacity) {
        int newCapacity = (allAlgos->capacity == 0) ? 16 : allAlgos->capacity * 2;
        AlgElement *newElements = new (std::nothrow) AlgElement[newCapacity];
        if (newElements == nullptr) {
            HCCL_ERROR("[AllAlgos] alloc failed, newCapacity=%d.", newCapacity);
            return HcclResult::HCCL_E_PARA;
        }
        for (int i = 0; i < allAlgos->count; ++i) {
            newElements[i] = allAlgos->algElements[i];
        }
        delete[] allAlgos->algElements;
        allAlgos->algElements = newElements;
        allAlgos->capacity = newCapacity;
    }
    allAlgos->algElements[allAlgos->count] = {algName, executorName, templateName, templateNum, opType};
    ++allAlgos->count;
    HCCL_DEBUG("[AllAlgos] add algName=%s executorName=%s templateNum=%d opType=%d, total=%d.",
               algName, executorName, templateNum, opType, allAlgos->count);
    return HcclResult::HCCL_SUCCESS;
}

CostModelManager::CostModelManager()
{
    InitBandwidth();
}

CostModelManager::~CostModelManager()
{
    FreeCostModel();
}

void CostModelManager::FreeCostModel()
{
    delete[] costModel_.costAlgoParams;
    costModel_.costAlgoParams = nullptr;
    costModel_.count = 0;
}

void CostModelManager::InitBandwidth()
{
    HCCL_DEBUG("[CostModelManager] InitBandwidth.");
    localCopyBw_ = 750;
    localReduceBw_ = 483;
    crossChipBw_ = 56;
    crossChipReduceBw_ = 56;
    ccuLocalCopyBw_ = 200;
    ccuLocalReduceBw_ = 200;
    HCCL_DEBUG("[CostModelManager] localCopyBw=%f localReduceBw=%f crossChipBw=%f crossChipReduceBw=%f "
               "ccuLocalCopyBw=%f ccuLocalReduceBw=%f.",
               localCopyBw_, localReduceBw_, crossChipBw_, crossChipReduceBw_, ccuLocalCopyBw_, ccuLocalReduceBw_);
}

HcclResult CostModelManager::InitCostModel(const TopoInfoWithNetLayerDetails *topoInfo)
{
    FreeCostModel();

    const AllAlgos &allAlgos = *GetAllAlgos();
    int algNum = allAlgos.count;
    if (algNum <= 0) {
        HCCL_WARNING("[CostModelManager] InitCostModel with empty AllAlgos.");
        return HcclResult::HCCL_SUCCESS;
    }

    costModel_.costAlgoParams = new (std::nothrow) CostAlgoParams[algNum];
    if (costModel_.costAlgoParams == nullptr) {
        HCCL_ERROR("[CostModelManager] alloc CostAlgoParams failed, algNum=%d.", algNum);
        return HcclResult::HCCL_E_PARA;
    }
    costModel_.count = 0;

    for (int i = 0; i < algNum; ++i) {
        const AlgElement &alg = allAlgos.algElements[i];

        std::unique_ptr<InsCollAlgBase> exec =
            CollAlgExecRegistryV2::Instance().GetAlgExec(alg.opType, alg.algName);
        if (exec == nullptr) {
            HCCL_WARNING("[CostModelManager] executor not registered, skip algName=%s opType=%d.",
                         alg.algName, alg.opType);
            continue;
        }

        CostAlgoParams cap = exec->CalcCostCoeff(topoInfo->userRankSize);
        if (cap.count == 0 || cap.param == nullptr) {
            HCCL_WARNING("[CostModelManager] CalcCostCoeff uncalibrated, skip algName=%s.", alg.algName);
            continue;
        }
        cap.algName = alg.algName;

        AlgNetMetaRegistry::Global()->Register(alg.algName, exec->GetAlgNetMeta());

        costModel_.costAlgoParams[costModel_.count] = cap;
        ++costModel_.count;
    }

    if (costModel_.count == 0) {
        delete[] costModel_.costAlgoParams;
        costModel_.costAlgoParams = nullptr;
    }

    HCCL_DEBUG("[CostModelManager] InitCostModel done, total=%d calibrated=%d.",
               algNum, costModel_.count);
    return HcclResult::HCCL_SUCCESS;
}

void CostModelManager::CalcMeshParam(float n, int netType, int portNum, float &A)
{
    A = 0.0f;
    if (netType == 0) {
        // cost = D/B(write)
        A = 1 / crossChipBw_;
    } else if (netType == 1) {
        // cost = nD/B(write)
        A = n / (portNum * crossChipBw_);
    } else {
        HCCL_ERROR("[CostModelManager] CalcMeshParams unsupported netType=%d.", netType);
    }
    HCCL_DEBUG("[CostModelManager] CalcMeshParams n=%f netType=%d portNum=%d A=%f.", n, netType, portNum, A);
    return;
}

void CostModelManager::CalcNHRParams(float n, int netType, int portNum, float &A)
{
    A = 0.0f;
    if (netType == 0) {
        // 
        A = 1 / crossChipBw_;
    } else if (netType == 1) {
        // 
        A = n / (portNum * crossChipBw_);
    } else {
        HCCL_ERROR("[CostModelManager] CalcNHRParams unsupported netType=%d.", netType);
    }
    HCCL_DEBUG("[CostModelManager] CalcNHRParams n=%f netType=%d portNum=%d A=%f.", n, netType, portNum, A);
    return;
}

void CostModelManager::CalcLocalCopyParams(float n, int scene, float &B)
{
    float bw = (scene == 1) ? ccuLocalCopyBw_ : localCopyBw_;
    B = n / bw;
    HCCL_DEBUG("[CostModelManager] CalcLocalCopyParams n=%f scene=%d B=%f.", n, scene, B);
    return;
}

void CostModelManager::CalcLocalReduceParams(float n, int scene, float &B)
{
    float bw = (scene == 1) ? ccuLocalReduceBw_ : localReduceBw_;
    B = n / bw;
    HCCL_DEBUG("[CostModelManager] CalcLocalReduceParams n=%f scene=%d B=%f.", n, scene, B);
    return;
}

void CostModelManager::CalcLatencyParams(int taskNum, float &C)
{
    C = 0.0f;
    HCCL_DEBUG("[CostModelManager] CalcLatencyParams taskNum=%d, C=%f.", taskNum, C);
    return;
}

AlgNetMetaRegistry *AlgNetMetaRegistry::Global()
{
    static AlgNetMetaRegistry *globalRegistry = new AlgNetMetaRegistry;
    return globalRegistry;
}

void AlgNetMetaRegistry::Register(const std::string &algName, AlgNetMeta meta)
{
    const std::lock_guard<std::mutex> lock(mu_);
    metas_[algName] = meta;
    HCCL_DEBUG("[AlgNetMetaRegistry] register algName=%s netTypes=%zu intraGroupMode=%d groupSizes=%zu.",
               algName.c_str(), meta.netTypes.size(), static_cast<int>(meta.intraGroupMode), meta.groupSizes.size());
}

bool AlgNetMetaRegistry::Query(const std::string &algName, AlgNetMeta &meta) const
{
    const std::lock_guard<std::mutex> lock(mu_);
    auto it = metas_.find(algName);
    if (it == metas_.end()) {
        return false;
    }
    meta = it->second;
    return true;
}

} // namespace ops_hccl
