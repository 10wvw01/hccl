/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_concurrent.h"
#include "op_common.h"

namespace ops_hccl {
TopoMatchConcurrent::TopoMatchConcurrent() : TopoMatchBase()
{
}

TopoMatchConcurrent::~TopoMatchConcurrent()
{
}

HcclResult TopoMatchConcurrent::MatchTopo(
    const HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    uint32_t layerNum = topoInfo->topoInstDetailsOfLayer.size();
    uint32_t myRank = topoInfo->userRank;

    // serverRankSize = layer0 第 0 个 inst 的 rank 数（单 server 卡数）
    uint32_t serverRankSize = 0;
    if (layerNum > 0) {
        const TopoInstDetails &layer0Details = topoInfo->topoInstDetailsOfLayer.at(0);
        if (layer0Details.topoInstNum > 0 && !layer0Details.ranksInTopo.empty()) {
            serverRankSize = layer0Details.ranksInTopo.at(0).size();
        }
    }

    algHierarchyInfo.infos.clear();
    for (uint32_t i = 0; i < layerNum; i++) {
        TopoInstDetails topoInstDetails = topoInfo->topoInstDetailsOfLayer.at(i);
        uint32_t topoInstNum = topoInstDetails.topoInstNum;
        for (uint32_t j = 0; j < topoInstNum; j++) {
            const std::vector<uint32_t> &allRanks = topoInstDetails.ranksInTopo.at(j);
            CommTopo topoType = topoInstDetails.typeOfTopo.at(j);
            std::vector<uint32_t> subCommRanks;
            if (topoType == CommTopo::COMM_TOPO_1DMESH) {
                // MESH：取 inst 全集（server 内 Mesh 子域）
                subCommRanks = allRanks;
            } else if (serverRankSize > 0) {
                // 非 MESH（CLOS / NHR）：跨 server 同 idx 子集
                for (auto r : allRanks) {
                    if (r % serverRankSize == myRank % serverRankSize) {
                        subCommRanks.push_back(r);
                    }
                }
            } else {
                subCommRanks = allRanks;
            }
            algHierarchyInfo.infos.emplace_back(std::vector<std::vector<u32>>{subCommRanks});
            if (topoType == CommTopo::COMM_TOPO_1DMESH && j == 0) {
                // MESH inst 复制一份，作为同 server 内的另一条 sub-comm（Concurrent 语义）
                algHierarchyInfo.infos.emplace_back(std::vector<std::vector<u32>>{subCommRanks});
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
