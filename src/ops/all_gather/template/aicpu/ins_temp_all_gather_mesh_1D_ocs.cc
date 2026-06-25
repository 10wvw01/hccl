/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_1D_ocs.h"
#include "channel.h"

namespace ops_hccl {

InsTempAllGatherMesh1DOcs::InsTempAllGatherMesh1DOcs(const OpParam &param, const u32 rankId,
                                                     const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAllGatherMesh1D(param, rankId, subCommRanks)
{
}

InsTempAllGatherMesh1DOcs::~InsTempAllGatherMesh1DOcs()
{
}

HcclResult InsTempAllGatherMesh1DOcs::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    // 线程/notify 语义与父类 InsTempAllGatherMesh1D::GetRes 一致
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    CHK_PRT_RET(topoInfo == nullptr,
        HCCL_ERROR("[InsTempAllGatherMesh1DOcs][CalcRes] topoInfo is nullptr"), HCCL_E_PARA);

    // 仅在 OCS 层(net_layer_3)上建立 channel，避免误取低层(layer0/1/2)链路
    std::vector<HcclChannelDesc> ocsChannels;
    CHK_RET(CalcChannelRequestMesh1DByLayer(comm, param, topoInfo, subCommRanks_, ocsChannels, ocsNetLayer_));
    resourceRequest.channels.push_back(ocsChannels);

    HCCL_DEBUG("[InsTempAllGatherMesh1DOcs][CalcRes] myRank[%u], ocsNetLayer_[%u], channels[%zu], "
        "notifyNumOnMainThread[%u], slaveThreadNum[%u]",
        myRank_, ocsNetLayer_, ocsChannels.size(),
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
