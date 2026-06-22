/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/ins_temp_all_to_all_v_ab_inline_no_memcpy.h"
#include <algorithm>

namespace ops_hccl {

InsTempAlltoAllVABInlineNoMemcpy::InsTempAlltoAllVABInlineNoMemcpy(
    const OpParam &param, u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

HcclResult InsTempAlltoAllVABInlineNoMemcpy::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllVABInlineNoMemcpy::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

void InsTempAlltoAllVABInlineNoMemcpy::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    for (u32 i = 1; i < threadNum_; ++i) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAlltoAllVABInlineNoMemcpy::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    for (u32 i = 1; i < threadNum_; ++i) {
        notifyIdxSubToMain.push_back(i - 1);
    }
}

HcclResult InsTempAlltoAllVABInlineNoMemcpy::KernelRun(
    const OpParam &param, const TemplateDataParams &tempAlgParams, TemplateResource &templateResource)
{
    threadNum_ = templateResource.threads.size();
    rankSize_ = tempAlgParams.sendCounts.size();
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(dataType_ >= HCCL_DATA_TYPE_RESERVED || dataTypeSize_ == 0,
                HCCL_ERROR("[A2AV_AB_INLINE][Template] invalid datatype[%d].", static_cast<int>(dataType_)),
                HcclResult::HCCL_E_INTERNAL);
    CHK_RET(CheckParams(tempAlgParams, templateResource));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdx;
        GetNotifyIdxMainToSub(notifyIdx);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdx));
    }
    CHK_RET(RunBClos(tempAlgParams, templateResource));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdx;
        GetNotifyIdxSubToMain(notifyIdx);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdx));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVABInlineNoMemcpy::CheckParams(
    const TemplateDataParams &params, const TemplateResource &resource) const
{
    CHK_PRT_RET(params.sendCounts.empty() || params.recvCounts.size() < params.sendCounts.size() ||
                    params.sdispls.size() < params.sendCounts.size() ||
                    params.rdispls.size() < params.sendCounts.size() ||
                    params.remoteRdispls.size() < params.sendCounts.size() ||
                    params.remoteRecvCounts.size() < params.sendCounts.size(),
                HCCL_ERROR("[A2AV_AB_INLINE][Template] invalid vectors. rank=%u send=%zu recv=%zu "
                           "sdispls=%zu rdispls=%zu remoteRdispls=%zu remoteRecvCounts=%zu",
                           myRank_, params.sendCounts.size(), params.recvCounts.size(), params.sdispls.size(),
                           params.rdispls.size(), params.remoteRdispls.size(), params.remoteRecvCounts.size()),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(resource.threads.empty(),
                HCCL_ERROR("[A2AV_AB_INLINE][Template] empty thread resource. rank=%u", myRank_),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(resource.threads.size() < resource.channels.size() + 1,
                HCCL_ERROR("[A2AV_AB_INLINE][Template] insufficient threads. rank=%u threads=%zu peers=%zu",
                           myRank_, resource.threads.size(), resource.channels.size()),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVABInlineNoMemcpy::CheckSliceRange(
    u32 peerRank, u64 srcOffset, u64 dstOffset, u64 byteSize, u64 inputSize, u64 remoteOutputSize) const
{
    CHK_PRT_RET(srcOffset + byteSize > inputSize || dstOffset + byteSize > remoteOutputSize,
                HCCL_ERROR("[A2AV_AB_INLINE][Template] slice out of range. rank=%u peer=%u "
                           "srcOff=%llu dstOff=%llu size=%llu inputSize=%llu remoteOutputSize=%llu",
                           myRank_, peerRank, srcOffset, dstOffset, byteSize, inputSize, remoteOutputSize),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVABInlineNoMemcpy::RunPeer(
    u32 peerRank, const ChannelInfo &channel, const TemplateDataParams &params, const ThreadHandle &thread) const
{
    u64 txCount = params.sendCounts[peerRank];
    u64 rxCount = params.recvCounts[peerRank];
    u64 txByteSize = txCount * dataTypeSize_;
    u64 rxByteSize = rxCount * dataTypeSize_;
    u64 txSrcOffset = params.sdispls[peerRank] * dataTypeSize_;
    u64 txDstOffset = params.remoteRdispls[peerRank] * dataTypeSize_;
    CHK_PRT_RET(params.remoteRecvCounts[peerRank] != txCount,
                HCCL_ERROR("[A2AV_AB_INLINE][Template] remote recv count mismatch. rank=%u peer=%u "
                           "localSend=%llu remoteRecvForLocal=%llu",
                           myRank_, peerRank, txCount, params.remoteRecvCounts[peerRank]),
                HcclResult::HCCL_E_INTERNAL);

    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    if (txByteSize > 0) {
        CHK_PRT_RET(channel.remoteOutputGraphMode.addr == nullptr,
                    HCCL_ERROR("[A2AV_AB_INLINE][Template] remote output unavailable. rank=%u peer=%u txSize=%llu",
                               myRank_, peerRank, txByteSize),
                    HcclResult::HCCL_E_INTERNAL);
        CHK_RET(CheckSliceRange(peerRank, txSrcOffset, txDstOffset, txByteSize,
                                params.buffInfo.inputSize, channel.remoteOutputGraphMode.size));
        txSrcSlices.emplace_back(params.buffInfo.inputPtr, txSrcOffset, txByteSize, txCount);
        txDstSlices.emplace_back(channel.remoteOutputGraphMode.addr, txDstOffset, txByteSize, txCount);
    } else {
        txSrcSlices.emplace_back(nullptr, 0, 0, 0);
        txDstSlices.emplace_back(nullptr, 0, 0, 0);
    }
    rxSrcSlices.emplace_back(params.buffInfo.outputPtr, params.rdispls[peerRank] * dataTypeSize_, rxByteSize, rxCount);
    rxDstSlices.emplace_back(params.buffInfo.outputPtr, params.rdispls[peerRank] * dataTypeSize_, rxByteSize, rxCount);

    SendRecvInfo sendRecvInfo{{channel, channel}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}},
                              dataType_};
    HCCL_WARNING("[A2AV_AB_INLINE][B_CLOS] rank=%u peer=%u txCount=%llu rxCount=%llu "
                 "txSrcOff=%llu txDstOff=%llu thread=%llu",
                 myRank_, peerRank, txCount, rxCount, txSrcOffset, txDstOffset,
                 static_cast<unsigned long long>(thread));
    CHK_PRT_RET(SendRecvWrite(sendRecvInfo, thread),
                HCCL_ERROR("[A2AV_AB_INLINE][Template] SendRecvWrite failed. rank=%u peer=%u",
                           myRank_, peerRank),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVABInlineNoMemcpy::RunBClos(
    const TemplateDataParams &params, const TemplateResource &resource)
{
    u32 threadIdx = 1;
    for (const auto &item : resource.channels) {
        u32 peerRank = item.first;
        if (peerRank == myRank_ || item.second.empty()) {
            continue;
        }
        const ChannelInfo &channel = item.second[0];
        ThreadHandle thread = resource.threads[threadIdx];
        CHK_RET(RunPeer(peerRank, channel, params, thread));
        ++threadIdx;
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
