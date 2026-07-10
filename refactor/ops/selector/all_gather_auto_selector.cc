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

namespace ops_hccl { namespace refactor {
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

// 全局 AICPU AllGather 算法表（定义在 gen_algorithm.cc），以 HcclAicpuAllGatherAlgoType 枚举值为数组下标。
// extern 声明位于 hccl_algorithm.h。

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
    HcclAicpuAllGatherAlgoType selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_ALGO_TYPE_COUNT;
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_3) {
            if (topoInfo->deviceNumPerModule == DEVICE_NUM_PER_MODULE_8) {
                selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_OMNIPIPE_UBOE;
            } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[1] == 1) {
                selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_NHR;
            } else {
                selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_PARALLEL_MESH1D_NHR_UBOE;
            }
        } else if (topoInfo->Level1Nhr) {
            selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_NHR;
            HCCL_INFO("[AllGatherAutoSelector] Level1Nhr=true, select [%d]", static_cast<int>(selectAlgEnum));
        } else if (topoInfo->Level0Nhr) {
            selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_NHR; // 预留给NHRNHR
        } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
            selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_NHR;
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (dataSize > AG_AICPU_SMALL_DATA_SIZE) {
                selectAlgEnum = (dataSize * topoInfo->userRankSize > AG_AICPU_SEQUENCE_DATA_SIZE) ?
                    HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_SEQUENCE_NHR_MESH1D :
                    HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_PARALLEL_MESH1D_NHR;
            } else {
                selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_NHR;
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_NHR;
        } else {
            HCCL_ERROR("[AllGatherAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (IsTwoLevelNetLayer(topoInfo) && dataSize * topoInfo->userRankSize > AG_AICPU_1D_TWO_LEVER_DATA_SIZE_THRESHOLD) {
                selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_MESH1D1D_ZAXIS_DETOUR;
            } else {
                selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_MESH1D;
            }
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            // PCIE-SW定制机型，Mesh无法链接全卡时，需要跨pcie链路，选择适配算法
            if (topoInfo->level0PcieMix) {
                if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                    selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_MESH1D;
                } else {
                    selectAlgEnum = (dataSize < OMNI_PCIE_AG_DATA_SIZE) ?
                        HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_PARALLEL_MESH1D_NHR_PCIE :
                        HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_OMNIPIPE_PCIE;
                }
                HCCL_DEBUG("[AllGatherAutoSelector][%s] Algo match[%d]", __func__, static_cast<int>(selectAlgEnum));
                alg = g_aicpuAllGatherAlgoMap[static_cast<size_t>(selectAlgEnum)];
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
                    selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_CONCURRENT_MESH1D_NHR;
                } else {
                    selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_MESH1D;
                }
            } else if(isClosNumMultipleOfMeshNum && dataSize > SMALL_COUNT_512KB) {
                selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_PARALLEL_MESH1D_NHR_MULTIJETTY;
            } else {
                // 4P外非对称场景，大小数据量都用NHR算法
                selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_NHR;
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgEnum = HcclAicpuAllGatherAlgoType::AICPU_ALLGATHER_NHR;
        } else {
            HCCL_ERROR("[AllGatherAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] Algo match[%d]", __func__, static_cast<int>(selectAlgEnum));
    alg = g_aicpuAllGatherAlgoMap[static_cast<size_t>(selectAlgEnum)];
    return SelectorStatus::MATCH;
}


REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLGATHER, 18, AllGatherAutoSelector);

} } // namespace refactor::ops_hccl
