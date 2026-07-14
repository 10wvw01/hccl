/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTO_SELECTOR_BASE
#define AUTO_SELECTOR_BASE

#include <string>
#include <unordered_map>
#include "alg_param.h"
#include "hccl_algorithm.h"

namespace ops_hccl {

constexpr uint64_t SMALL_COUNT_512KB = 512*1024; // Byte, UB协议一次传输的最大size
constexpr uint64_t LARGE_COUNT_1024KB = 1024*1024; // Byte, 可掩盖多mission尾块开销

constexpr u64 CCU_PARALLEL_MAX_DATA_SIZE = 64 * 1024 * 1024;

enum class SelectorStatus { MATCH, NOT_MATCH };

class AutoSelectorBase {
public:
    virtual ~AutoSelectorBase() = default;
    SelectorStatus Select(OpParam &opParam, TopoInfoWithNetLayerDetails* topoInfo,
                          HcclAlgorithm &alg) const;
    bool IsDefaultAlg(const HcclAlgoType algoType) const;
    bool IsSmallData(const u64 dataSize) const;
    bool IsLargeData(const u64 dataSize) const;
    virtual SelectorStatus SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                 const OpParam &opParam,
                                 const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                 HcclAlgorithm &alg) const;
    virtual SelectorStatus SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                 const OpParam &opParam,
                                 const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                 HcclAlgorithm &alg) const;
    virtual SelectorStatus SelectAicpuAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                   const OpParam &opParam,
                                   const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                   HcclAlgorithm &alg) const;
    virtual SelectorStatus SelectAivAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                   const OpParam &opParam,
                                   const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                   HcclAlgorithm &alg) const;
    virtual SelectorStatus SelectDPUAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                   const OpParam &opParam,
                                   const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                   HcclAlgorithm &alg) const;
    bool IsStarsState(const OpExecuteConfig &opExecuteConfig) const;
    bool IsLayerAllConnetedWithTopo(const TopoInfoWithNetLayerDetails *topoInfo, const u32 netLayer, const CommTopo topoType) const;
    HcclResult CheckMeshNumEqualToClosNum(const TopoInfoWithNetLayerDetails *topoInfo, bool &isEqual) const;
    HcclResult CheckClosNumMultipleOfMeshNum(const TopoInfoWithNetLayerDetails *topoInfo, bool &isMultiple) const;
    bool IsTwoLevelNetLayer(const TopoInfoWithNetLayerDetails *topoInfo) const;
    bool IsInputOutputOverlap(const OpParam &opParam) const;
    bool IsSmallDataCCU(const u64 dataSize, const u64 rankSize) const;

private:
    bool ProcessAivConfig(OpParam &opParam, TopoInfoWithNetLayerDetails* topoInfo,
                          const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                          HcclAlgorithm &alg, SelectorStatus &ret) const;
};

inline bool Is64BitDataType(const HcclDataType dataType)
{
    return dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
           dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
           dataType == HcclDataType::HCCL_DATA_TYPE_FP64;
}

} // namespace ops_hccl

#endif
