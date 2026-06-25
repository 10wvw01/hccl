/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_NHR_COMMON_H
#define INS_TEMP_REDUCE_SCATTER_NHR_COMMON_H

#include "alg_v2_template_base.h"
#include "executor_v2_base.h"

namespace ops_hccl {
constexpr u32 RS_NHR_SMALL_COUNT_512KB = 512 * 1024;

inline HcclResult CalcReduceScatterNhrRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, const std::vector<std::vector<u32>> &subCommRanks,
    AlgResourceRequest& resourceRequest, u32 &channelsPerRank)
{
    std::vector<HcclChannelDesc> channels;
    std::vector<HcclChannelDesc> myChannelDescs;
    u64 perDataSize = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * perDataSize;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        bool isIsolation = !(IsAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH) ||
            dataSize < RS_NHR_SMALL_COUNT_512KB);
        CHK_RET(CalcChannelRequestNhrMultiJetty(comm, param, topoInfo, subCommRanks, myChannelDescs, isIsolation));
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                channels.push_back(channel);
            }
        }
    } else {
        CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks, myChannelDescs));
        channels = myChannelDescs;
    }
    resourceRequest.channels.push_back(channels);
    channelsPerRank = CalcChannelsPerRank(channels);
    HCCL_INFO("[ReduceScatterNHR][CalcRes] channelsPerRank: [%u].", channelsPerRank);
    if (channelsPerRank > MAX_JETTY_NUM) {
        HCCL_ERROR(" %s  channelsPerRank %u is greater than MAX_JETTY_NUM %u", __func__, channelsPerRank,
            MAX_JETTY_NUM);
    } else {
        HCCL_DEBUG(" %s channelsPerRank is %u ", __func__, channelsPerRank);
    }
    return HCCL_SUCCESS;
}

inline HcclResult FillReduceScatterNhrRes(u32 channelsPerRank, AlgResourceRequest& resourceRequest)
{
    u32 threadNum = channelsPerRank;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    HCCL_INFO("[GetRes] channelsPerRank: [%u], slaveThreadNum: [%u].", channelsPerRank,
        resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

#endif
