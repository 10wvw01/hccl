/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1D_ocs.h"
#include "channel.h"

namespace ops_hccl {

InsTempReduceScatterMesh1DOcs::InsTempReduceScatterMesh1DOcs(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempReduceScatterMesh1D(param, rankId, subCommRanks)
{
}

InsTempReduceScatterMesh1DOcs::~InsTempReduceScatterMesh1DOcs()
{
}

HcclResult InsTempReduceScatterMesh1DOcs::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    CHK_PRT_RET(topoInfo == nullptr,
        HCCL_ERROR("[InsTempReduceScatterMesh1DOcs][CalcRes] topoInfo is nullptr"), HCCL_E_PARA);

    // 仅在 OCS 层(net_layer_3)上建立 channel，避免误取低层(layer0/1/2)链路
    std::vector<HcclChannelDesc> ocsChannels;
    CHK_RET(CalcChannelRequestMesh1DByLayer(comm, param, topoInfo, subCommRanks_, ocsChannels, ocsNetLayer_));
    resourceRequest.channels.push_back(ocsChannels);

    HCCL_DEBUG("[InsTempReduceScatterMesh1DOcs][CalcRes] myRank[%u], ocsNetLayer_[%u], channels[%zu], "
        "notifyNumOnMainThread[%u], slaveThreadNum[%u]",
        myRank_, ocsNetLayer_, ocsChannels.size(),
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
