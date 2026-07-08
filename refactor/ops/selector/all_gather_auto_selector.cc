/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_auto_selector.h"
#include "selector_registry.h"
#include "hccl_aiv_utils.h"

namespace ops_hccl {
constexpr u64 AG_2D_SMALL_DATA_SIZE = 1024 * 1024;
constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4;
constexpr u64 AG_CCU_SMALL_DATA_SIZE = 4 * 1024 * 1024;
constexpr u32 AG_FLATTEN_MAX_DATA_SIZE = 64 * 1024;
constexpr u64 AG_CCU_SEQUENCE_MAX_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AG_AICPU_SMALL_DATA_SIZE = 1 * 1024 * 1024;
constexpr u64 AG_AICPU_1D_TWO_LEVER_DATA_SIZE_THRESHOLD = 1 * 1024 * 1024 * 1024;
constexpr u64 AG_CCU_CLOS_SMALL_DATA_SIZE = 1 * 1024 * 1024;
constexpr u64 AG_AICPU_SEQUENCE_DATA_SIZE = 4ULL * 1024 * 1024 * 1024;
constexpr u32 OMNI_PCIE_AG_DATA_SIZE = 4 * 1024 * 1024;
constexpr u32 TOPO_LEVEL_NUM_3 = 3;
constexpr u32 DEVICE_NUM_PER_MODULE_8 = 8;

// 全局 AICPU AllGather 算法表（定义在 gen_algorithm.cc）
extern const std::map<HcclAicpuAllGatherAlgoType, HcclAlgorithm> g_aicpuAllGatherAlgoMap;

// 算法名到 HcclAicpuAllGatherAlgoType 枚举的映射
static const std::map<std::string, HcclAicpuAllGatherAlgoType> AICPU_ALLGATHER_NAME_TO_ENUM = {
    {"InsV2AllGatherOmniPipeUboe",         HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_OMNIPIPE_UBOE},
    {"InsAllGatherNHR",                    HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_NHR},
    {"InsAllGatherParallelMesh1DNHRUboe",  HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_PARALLEL_MESH1D_NHR_UBOE},
    {"InsAllGatherSequenceNHRMesh1D",      HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_SEQUENCE_NHR_MESH1D},
    {"InsAllGatherParallelMesh1DNHR",      HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_PARALLEL_MESH1D_NHR},
    {"InsAllGatherMesh1D1DZAxisDetour",    HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_MESH1D1D_ZAXIS_DETOUR},
    {"InsAllGatherMesh1D",                 HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_MESH1D},
    {"InsAllGatherParallelMesh1DNHRPcie",  HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_PARALLEL_MESH1D_NHR_PCIE},
    {"InsV2AllGatherOmniPipePcie",         HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_OMNIPIPE_PCIE},
    {"InsAllGatherConcurrentMesh1DNHR",    HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_CONCURRENT_MESH1D_NHR},
    {"InsAllGatherParallelMesh1DNHRMultiJetty",
        HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_PARALLEL_MESH1D_NHR_MULTIJETTY},
};

/**
 * 根据算法名字符串查 g_aicpuAllGatherAlgoMap 填充 HcclAlgorithm。
 * 返回 true 表示成功找到并填充，false 表示未找到。
 */
static bool FillAicpuAllGatherAlgoByName(const std::string &selectAlgName, HcclAlgorithm &alg)
{
    auto enumIter = AICPU_ALLGATHER_NAME_TO_ENUM.find(selectAlgName);
    if (enumIter == AICPU_ALLGATHER_NAME_TO_ENUM.end()) {
        HCCL_ERROR("[AllGatherAutoSelector] algName[%s] has no matching HcclAicpuAllGatherAlgoType enum.",
            selectAlgName.c_str());
        return false;
    }
    auto algoIter = g_aicpuAllGatherAlgoMap.find(enumIter->second);
    if (algoIter == g_aicpuAllGatherAlgoMap.end()) {
        HCCL_ERROR("[AllGatherAutoSelector] enum[%d] has no matching HcclAlgorithm in g_aicpuAllGatherAlgoMap.",
            static_cast<int>(enumIter->second));
        return false;
    }
    alg = algoIter->second;
    return true;
}

SelectorStatus AllGatherAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    HcclAlgorithm &alg) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start, topoInfo topoLevelNums[%u]", __func__, topoInfo->topoLevelNums);
    (void)configAlgMap;
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    HCCL_INFO("[AllGatherAutoSelector][SelectAicpuAlgo] topoLevelNums=[%d], deviceNumPerModule=[%d], level0Topo=[%d]",
              topoInfo->topoLevelNums, topoInfo->deviceNumPerModule, topoInfo->level0Topo);
    std::string selectAlgName;
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_3) {
            if (topoInfo->deviceNumPerModule == DEVICE_NUM_PER_MODULE_8) {
                selectAlgName = "InsV2AllGatherOmniPipeUboe";
            } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[1] == 1) {
                selectAlgName = "InsAllGatherNHR";
            } else {
                selectAlgName = "InsAllGatherParallelMesh1DNHRUboe";
            }
        } else if (topoInfo->Level1Nhr) {
            selectAlgName = "InsAllGatherNHR";
            HCCL_INFO("[AllGatherAutoSelector] Level1Nhr=true, select [%s]", selectAlgName.c_str());
        } else if (topoInfo->Level0Nhr) {
            selectAlgName = "InsAllGatherNHR"; // 预留给NHRNHR
        } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
            selectAlgName = "InsAllGatherNHR";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (dataSize > AG_AICPU_SMALL_DATA_SIZE) {
                selectAlgName = (dataSize * topoInfo->userRankSize > AG_AICPU_SEQUENCE_DATA_SIZE) ?
                    "InsAllGatherSequenceNHRMesh1D" : "InsAllGatherParallelMesh1DNHR";
            } else {
                selectAlgName = "InsAllGatherNHR";
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "InsAllGatherNHR";
        } else {
            HCCL_ERROR("[AllGatherAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (IsTwoLevelNetLayer(topoInfo) && dataSize * topoInfo->userRankSize > AG_AICPU_1D_TWO_LEVER_DATA_SIZE_THRESHOLD) {
                selectAlgName = "InsAllGatherMesh1D1DZAxisDetour";
            } else {
                selectAlgName = "InsAllGatherMesh1D";
            }
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            // PCIE-SW定制机型，Mesh无法链接全卡时，需要跨pcie链路，选择适配算法
            if (topoInfo->level0PcieMix) {
                if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                    selectAlgName = "InsAllGatherMesh1D";
                } else {
                    selectAlgName = (dataSize < OMNI_PCIE_AG_DATA_SIZE) ? "InsAllGatherParallelMesh1DNHRPcie" :
                                                                          "InsV2AllGatherOmniPipePcie";
                }
                HCCL_DEBUG("[AllGatherAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
                if (!FillAicpuAllGatherAlgoByName(selectAlgName, alg)) {
                    return SelectorStatus::NOT_MATCH;
                }
                return SelectorStatus::MATCH;
            }
            // UBX机型
            bool isMeshNumEqualToClosNum = false;
            bool isClosNumMultipleOfMeshNum = false;
            CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
            HCCL_ERROR("[AllGatherAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
            CHK_PRT_RET(CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
            HCCL_ERROR("[AllGatherAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
            if (isMeshNumEqualToClosNum && topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) {
                if (dataSize > SMALL_COUNT_512KB) {
                    selectAlgName = "InsAllGatherConcurrentMesh1DNHR";
                } else {
                    selectAlgName = "InsAllGatherMesh1D";
                }
            } else if(isClosNumMultipleOfMeshNum && dataSize > SMALL_COUNT_512KB) {
                selectAlgName = "InsAllGatherParallelMesh1DNHRMultiJetty";
            } else {
                // 4P外非对称场景，大小数据量都用NHR算法
                selectAlgName = "InsAllGatherNHR";
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "InsAllGatherNHR";
        } else {
            HCCL_ERROR("[AllGatherAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    if (!FillAicpuAllGatherAlgoByName(selectAlgName, alg)) {
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}


REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLGATHER, 18, AllGatherAutoSelector);

}  // namespace ops_hccl
