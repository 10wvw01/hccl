/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "selector_engine.h"

#include <cstring>
#include <algorithm>

#include "alg_env_config.h"

namespace ops_hccl {

// ====== algo / tuner 模块接口声明 ======
// 由 algo 模块提供;若未链接,下方 weak 桩生效
// algo 模块根据 candidateEngines 过滤掉不会选择到的引擎算法
__attribute__((weak)) HcclResult FilterCmByHcclAlgo(CostModel &cm, const std::vector<OpExecuteConfig> &candidateEngines)
{
    (void)cm;
    (void)candidateEngines;
    HCCL_INFO("[FilterCmByHcclAlgo] weak stub, no filtering applied.");
    return HCCL_SUCCESS;
}

// 由 tuner 模块提供;若未链接,下方 weak 桩生效
__attribute__((weak)) HcclResult Tuner(CostTable &ct, const OpParam &param)
{
    (void)param;
    (void)ct;
    HCCL_INFO("[Tuner] weak stub, no tuning applied.");
    return HCCL_SUCCESS;
}

// 由 tuner 模块提供;若未链接,下方 weak 桩生效
__attribute__((weak)) HcclResult InitTuner()
{
    HCCL_INFO("[InitTuner] weak stub, no tuner plugin initialized.");
    return HCCL_SUCCESS;
}

// ====== SelectorEngine ======

SelectorEngine *SelectorEngine::Global()
{
    static SelectorEngine *globalSelectorEngine = new SelectorEngine;
    return globalSelectorEngine;
}

OpExecuteConfig SelectorEngine::GetEngineByAlgName(const std::string &algName)
{
    if (algName.rfind("Ccu", 0) == 0) {
        // CCU 算法按名称区分 CCU_MS / CCU_SCHED
        if (algName.find("Sche") != std::string::npos) {
            return OpExecuteConfig::CCU_SCHED;
        }
        return OpExecuteConfig::CCU_MS;
    }
    if (algName.rfind("Aiv", 0) == 0) {
        return OpExecuteConfig::AIV;
    }
    if (algName.rfind("Ins", 0) == 0) {
        return OpExecuteConfig::AICPU_TS;
    }
    if (algName.find("DPU") != std::string::npos) {
        return OpExecuteConfig::HOSTCPU;
    }
    return OpExecuteConfig::AICPU_TS;
}

std::vector<OpExecuteConfig> SelectorEngine::GetEnginePriority(TopoInfoWithNetLayerDetails *topoInfo,
                                                               OpExecuteConfig opExecuteConfig)
{
    // hostDpuOnly 已在 HcclCalcTopoInfo 中计算并缓存
    if (topoInfo != nullptr && topoInfo->hostDpuOnly) {
        HCCL_INFO("[GetEnginePriority] hostDpuOnly=true, return [HOSTCPU].");
        return {OpExecuteConfig::HOSTCPU};
    }

    switch (opExecuteConfig) {
        case OpExecuteConfig::CCU_MS:
            // CCU_MS → CCU_SCHED → AIV → AICPU 回退链
            return {OpExecuteConfig::CCU_MS,
                    OpExecuteConfig::CCU_SCHED,
                    OpExecuteConfig::AIV,
                    OpExecuteConfig::AICPU_TS};
        case OpExecuteConfig::CCU_SCHED:
            // CCU_SCHED → AIV → AICPU 回退链
            return {OpExecuteConfig::CCU_SCHED,
                    OpExecuteConfig::AIV,
                    OpExecuteConfig::AICPU_TS};
        case OpExecuteConfig::AIV:
            // AIV → AICPU 回退链
            return {OpExecuteConfig::AIV,
                    OpExecuteConfig::AICPU_TS};
        case OpExecuteConfig::AIV_ONLY:
            // AIV only, 不回退
            return {OpExecuteConfig::AIV};
        case OpExecuteConfig::AICPU_TS:
            // AICPU only
            return {OpExecuteConfig::AICPU_TS};
        case OpExecuteConfig::HOSTCPU:
            // HOSTCPU(DPU)
            return {OpExecuteConfig::HOSTCPU};
        default:
            // 安全回退:仅 AICPU
            return {OpExecuteConfig::AICPU_TS};
    }
}

HcclResult SelectorEngine::GetOrInitCostModel(HcclComm comm, const std::vector<OpExecuteConfig> &candidateEngines,
                                              CostModel *&cm)
{
    static constexpr const char *COST_MODEL_TAG = "costmodel";
    void *ctxPtr = nullptr;
    uint64_t ctxSize = 0;

    // 尝试从通信域 ctx 获取已缓存的 costModel
    if (HcclEngineCtxGet(comm, COST_MODEL_TAG, CommEngine::COMM_ENGINE_CPU, &ctxPtr, &ctxSize) == HCCL_SUCCESS) {
        cm = static_cast<CostModel *>(ctxPtr);
        HCCL_DEBUG("[SelectorEngine] costModel found in comm ctx, count=%d.", cm->count);
        return HCCL_SUCCESS;
    }

    // 未缓存，初始化 costModel
    HCCL_INFO("[SelectorEngine] Initializing costModel for comm.");

    // TODO 调用costmodel初始化
    CostModelManager costModelMgr;
    CHK_RET(costModelMgr.InitCostModel());

    const CostModel &srcCm = costModelMgr.GetCostModel();
    CostModel &mutableCm = const_cast<CostModel &>(srcCm);

    // 调用 algo 模块过滤 costModel(传入 candidateEngines,过滤不会选择到的引擎算法)
    CHK_RET(FilterCmByHcclAlgo(mutableCm, candidateEngines));

    // 将 CostModel 扁平化存入通信域 ctx: [CostModel header][CostAlgoParams array]
    uint64_t flatSize = sizeof(CostModel) + static_cast<uint64_t>(srcCm.count) * sizeof(CostAlgoParams);
    CHK_RET(HcclEngineCtxCreate(comm, COST_MODEL_TAG, CommEngine::COMM_ENGINE_CPU, flatSize, &ctxPtr));

    CostModel *storedCm = static_cast<CostModel *>(ctxPtr);
    storedCm->count = srcCm.count;
    storedCm->costAlgoParams = reinterpret_cast<CostAlgoParams *>(storedCm + 1);
    if (srcCm.count > 0 && srcCm.costAlgoParams != nullptr) {
        memcpy_s(storedCm->costAlgoParams, static_cast<uint64_t>(srcCm.count) * sizeof(CostAlgoParams),
                 srcCm.costAlgoParams, static_cast<uint64_t>(srcCm.count) * sizeof(CostAlgoParams));
    }

    cm = storedCm;

    // 初始化 tuner 插件 TODO
    CHK_RET(InitTuner());

    HCCL_INFO("[SelectorEngine] costModel initialized and stored in comm ctx, count=%d.", srcCm.count);
    return HCCL_SUCCESS;
}

HcclResult SelectorEngine::Run(HcclComm comm, OpParam &param, TopoInfoWithNetLayerDetails *topoInfo,
                               std::string &algName)
{
    HCCL_INFO("[SelectorEngine] Run start, opType=%d, opExecuteConfig=%d.",
              static_cast<int>(param.opType), static_cast<int>(param.opExecuteConfig));

    // step 0: 获取候选引擎列表(顺序即优先级,从高到低)
    std::vector<OpExecuteConfig> candidateEngines = GetEnginePriority(topoInfo, param.opExecuteConfig);

    // step 1: 从通信域 ctx 获取或初始化 costModel(传 candidateEngines 给 algo 模块过滤)
    CostModel *cm = nullptr;
    CHK_RET(GetOrInitCostModel(comm, candidateEngines, cm));

    // step 2: costTable 生成 + tuner
    CostTable ct{nullptr, 0};
    CHK_RET(GenAndProcessCostTable(param, topoInfo, ct, *cm));

    // step 3: min(ct)
    HcclResult ret = SelectMinCost(ct, candidateEngines, param, algName);

    delete[] ct.costs;
    ct.costs = nullptr;
    ct.count = 0;

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[SelectorEngine] Run failed, no algorithm selected.");
        return ret;
    }

    HCCL_INFO("[SelectorEngine] The opExecuteConfig is %d, the selected algo type is %s",
               static_cast<int>(param.opExecuteConfig), algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult SelectorEngine::GenAndProcessCostTable(OpParam &param, TopoInfoWithNetLayerDetails *topoInfo,
                                                   CostTable &ct, CostModel &cm)
{
    // step 2.1: 从 costModel 生成 costTable TODO
    CHK_RET(CostTableManager::Global()->CostTableGen(cm, ct, topoInfo, param));

    if (ct.count <= 0) {
        HCCL_WARNING("[SelectorEngine] costTable is empty after CostTableGen.");
        return HCCL_SUCCESS;
    }

    // step 2.2: tuner(外部模块,优先级最高,可覆盖 costTable) TODO
    CHK_RET(Tuner(ct, param));

    return HCCL_SUCCESS;
}

HcclResult SelectorEngine::SelectMinCost(const CostTable &ct, const std::vector<OpExecuteConfig> &candidateEngines,
                                         OpParam &param, std::string &algName)
{
    // candidateEngines 顺序即优先级,从高到低逐引擎查找 min(cost)
    for (OpExecuteConfig engine : candidateEngines) {
        int minIdx = -1;
        float minCost = 0.0f;
        for (int i = 0; i < ct.count; ++i) {
            if (ct.costs[i].cost < 0.0f || ct.costs[i].algName == nullptr) {
                continue;
            }
            // 这里应该缓存好每个引擎的所有算法，不能每次计算 TODO
            if (GetEngineByAlgName(ct.costs[i].algName) != engine) {
                continue;
            }
            if (minIdx == -1 || ct.costs[i].cost < minCost) {
                minIdx = i;
                minCost = ct.costs[i].cost;
            } else if (ct.costs[i].cost == minCost) {
                HCCL_WARNING("[SelectorEngine] multiple algos with same cost=%f: %s vs %s, selecting first.",
                             minCost, ct.costs[minIdx].algName, ct.costs[i].algName);
            }
        }

        if (minIdx >= 0) {
            algName = ct.costs[minIdx].algName;
            param.opExecuteConfig = engine;
            HCCL_INFO("[SelectorEngine] SelectMinCost: engine=%d, algName=%s, cost=%f.",
                       static_cast<int>(engine), algName.c_str(), minCost);
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[SelectorEngine] SelectMinCost: no valid algorithm found.");
    return HCCL_E_NOT_SUPPORT;
}

} // namespace ops_hccl
