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
    CHK_PRT_RET(topoInfo == nullptr,
        HCCL_ERROR("[InsTempReduceScatterMesh1DOcs][CalcRes] topoInfo is nullptr"), HCCL_E_PARA);

    // 仅在 OCS 层(net_layer_3)上建立 channel，避免误取低层(layer0/1/2)链路
    std::vector<HcclChannelDesc> ocsChannels;
    CHK_RET(CalcChannelRequestMesh1DByLayer(comm, param, topoInfo, subCommRanks_, ocsChannels, ocsNetLayer_));
    resourceRequest.channels.push_back(ocsChannels);

    // 关键: channelsPerRank_ 必须在 GetRes 之前算出。父类 GetRes 会通过(被本类 override 的)
    // GetThreadNum() 读取 channelsPerRank_，以 (templateRankSize_-1) * channelsPerRank_ 口径
    // 申请从线程数。RunReduceScatter 内部按同样口径通过 threads[queIdx] 分发每条 channel 的收发，
    // 若此处漏算将导致从线程数不足、threads[queIdx] 越界取到空 handle(thread[0x0] is nullptr)。
    // 与 InsTempReduceScatterMesh1DZAxisDetour::CalcRes 同样的时序约定。
    channelsPerRank_ = CalcChannelsPerRank(ocsChannels);

    // GetRes 为虚函数分发: 这里调用本类 override 的 GetThreadNum()，保证 slaveThreadNum 与
    // RunReduceScatter 的实际线程访问上界一致。
    CHK_RET(GetRes(resourceRequest));

    HCCL_INFO("[InsTempReduceScatterMesh1DOcs][CalcRes] myRank[%u], ocsNetLayer_[%u], channels[%zu], "
        "channelsPerRank_[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u]",
        myRank_, ocsNetLayer_, ocsChannels.size(), channelsPerRank_,
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1DOcs::GetThreadNum() const
{
    // 单 rank 时 RunReduceScatter 的循环不会执行(queIdx 不增长)，1 个主线程即可;
    // 多 rank 时每条 channel 各占一个从线程: (templateRankSize_-1) * channelsPerRank_ + 1。
    u32 threadNum = templateRankSize_ > 1 ? ((templateRankSize_ - 1) * channelsPerRank_ + 1) : 1;
    HCCL_INFO("[InsTempReduceScatterMesh1DOcs][GetThreadNum] templateRankSize_[%u] channelsPerRank_[%u] threadNum[%u]",
        templateRankSize_, channelsPerRank_, threadNum);
    return threadNum;
}

// ---- RunReduceScatter 偏移计算（OCS 重载：else 分支，使用 (rank+1)%maxBlockNum 偏移公式） ----
u64 InsTempReduceScatterMesh1DOcs::GetRxSrcOffset(const TemplateDataParams &tempAlgParam, u32 repeatIdx,
                                                  u32 myAlgRank, u32 channelIdx) const
{
    return tempAlgParam.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParam.inputRepeatStride +
           myAlgRank * tempAlgParam.inputSliceStride + elemOffset_[channelIdx];
}

u64 InsTempReduceScatterMesh1DOcs::GetRxDstOffset(const TemplateDataParams &tempAlgParam, u32 repeatIdx,
                                                  u32 nextRank, u64 outputSliceStride, u32 channelIdx) const
{
    return repeatIdx * tempAlgParam.outputRepeatStride +
           ((subCommRanks_[0][nextRank] + 1) % hcclBufMaxBlockNum_) * outputSliceStride + elemOffset_[channelIdx];
}

u64 InsTempReduceScatterMesh1DOcs::GetTxDstOffset(const TemplateDataParams &tempAlgParam, u32 repeatIdx,
                                                  u32 myAlgRank, u64 outputSliceStride, u32 channelIdx) const
{
    return repeatIdx * tempAlgParam.outputRepeatStride +
           (myRank_ + 1) % hcclBufMaxBlockNum_ * outputSliceStride + elemOffset_[channelIdx];
}

// ---- PostCopy 偏移计算（OCS 重载：else 分支，使用 (rank+1)%maxBlockNum 偏移公式） ----
u64 InsTempReduceScatterMesh1DOcs::GetPostCopySrcOffset(const TemplateDataParams &tempAlgParams, u32 repeatIdx,
                                                        u32 tmpRank, u64 buffSliceStride) const
{
    return repeatIdx * tempAlgParams.outputRepeatStride +
           ((subCommRanks_[0][tmpRank] + 1) % hcclBufMaxBlockNum_) * buffSliceStride;
}

void InsTempReduceScatterMesh1DOcs::SetHcclBufMaxBlockNum(u32 blockNum)
{
    hcclBufMaxBlockNum_ = blockNum;
    return;
}

} // namespace ops_hccl
