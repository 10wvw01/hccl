/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_mesh_1D_Z_axis_detour.h"

namespace ops_hccl {

InsTempScatterMesh1DZAxisDetour::InsTempScatterMesh1DZAxisDetour(
    const OpParam& param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempScatterMesh1D(param, rankId, subCommRanks)
{
}

InsTempScatterMesh1DZAxisDetour::~InsTempScatterMesh1DZAxisDetour()
{
}

HcclResult InsTempScatterMesh1DZAxisDetour::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(topoInfo == nullptr,
        HCCL_ERROR("[InsTempScatterMesh1DZAxisDetour][CalcRes] topoInfo is nullptr"), HCCL_E_PARA);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel0(comm, param, topoInfo, subCommRanks_, level0Channels));
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel1(comm, param, topoInfo, subCommRanks_, level1Channels));
    std::vector<HcclChannelDesc> mergedChannels;
    mergedChannels.insert(mergedChannels.end(), level0Channels.begin(), level0Channels.end());
    mergedChannels.insert(mergedChannels.end(), level1Channels.begin(), level1Channels.end());
    resourceRequest.channels.push_back(mergedChannels);
    level0ChannelNumPerRank_ = level0Channels.empty() ? 0 : CalcChannelsPerRank(level0Channels);
    level1ChannelNumPerRank_ = level1Channels.empty() ? 0 : CalcChannelsPerRank(level1Channels);
    channelsPerRank_ = level0ChannelNumPerRank_ + level1ChannelNumPerRank_;
    CHK_RET(GetRes(resourceRequest));
    HCCL_INFO("[InsTempScatterMesh1DZAxisDetour][CalcRes]myRank[%u], channelsPerRank_[%u], "
               "level0ChannelNumPerRank_[%u], level1ChannelNumPerRank_[%u], level0DataRatio_[%.2f]",
               myRank_, channelsPerRank_, level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
    return HCCL_SUCCESS;
}

u64 InsTempScatterMesh1DZAxisDetour::GetThreadNum() const
{
    u32 threadNum = templateRankSize_ > 1 ? ((templateRankSize_ - 1) * channelsPerRank_) : 1;
    HCCL_INFO("[InsTempScatterMesh1DZAxisDetour][GetThreadNum] templateRankSize_[%u] channelsPerRank_[%u] threadNum[%u]",
              templateRankSize_, channelsPerRank_, threadNum);
    return threadNum;
}

HcclResult InsTempScatterMesh1DZAxisDetour::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

void InsTempScatterMesh1DZAxisDetour::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempScatterMesh1DZAxisDetour::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult InsTempScatterMesh1DZAxisDetour::CalcDataSplitByPortGroup(
    const u64 totalDataCount, const u64 dataTypeSize,
    const std::vector<ChannelInfo> &channels,
    std::vector<u64> &elemCountOut, std::vector<u64> &sizeOut,
    std::vector<u64> &elemOffset)
{
    HCCL_INFO("[InsTempScatterMesh1DZAxisDetour][CalcDataSplitByPortGroup] Run Start");
    return CalcDataSplitByPortGroupZAxisDetour(totalDataCount, dataTypeSize, channels,
        elemCountOut, sizeOut, elemOffset,
        level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
}

HcclResult InsTempScatterMesh1DZAxisDetour::SetchannelsPerRank(
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[SetchannelsPerRank] channels is empty."), HCCL_E_INTERNAL);
    channelsPerRank_ = CalcChannelsPerRank(channels);
    if (channelsPerRank_ > 1) {
        level0ChannelNumPerRank_ = MESH_CHANNELS_NUM;
        level1ChannelNumPerRank_ = channelsPerRank_ - level0ChannelNumPerRank_;
        level0DataRatio_ = 0.5f;
    }
    HCCL_INFO("[InsTempScatterMesh1DZAxisDetour][SetchannelsPerRank], channelsPerRank_[%u], "
              "level0ChannelNumPerRank_[%u], level1ChannelNumPerRank_[%u], level0DataRatio_[%.2f]",
              channelsPerRank_, level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1DZAxisDetour::KernelRun(const OpParam& param,
                     const TemplateDataParams &tempAlgParams,
                     TemplateResource& templateResource)
{
    CHK_PRT_RET(templateResource.threads.empty(),
                HCCL_ERROR("[InsTempScatterMesh1DZAxisDetour][KernelRun] threads is empty"),
                HCCL_E_INTERNAL);
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    threadNum_ = templateResource.threads.size(); //
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    if (tempAlgParams.tailSize != 0 && myAlgRank == templateRankSize_ - 1) {
        processSize_ = tempAlgParams.tailSize;
        count_ = processSize_ / dataTypeSize;
    }
    HCCL_INFO("[InsTempScatterMesh1DZAxisDetour] Run Start, myRank[%u], channelsPerRank_[%u]",
              myRank_, channelsPerRank_);
    CHK_RET(PreCopy(tempAlgParams, templateResource.threads));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunMesh(templateResource.channels, templateResource.threads, tempAlgParams));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));
    HCCL_INFO("[InsTempScatterMesh1DZAxisDetour] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1DZAxisDetour::PreCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads) const
{
    // 非根节点 或 in/outBuffType均为CCL BUFFER 则跳过前拷贝
    if (u32(myRank_) != root_ ||
        (tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER &&
         tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER)) {
        HCCL_INFO("[InsTempScatterMesh1DZAxisDetour][PreCopy] skip precopy, myRank = %u, root = %u", myRank_, root_);    
        return HCCL_SUCCESS;
    }

    // inplace场景跳过前拷贝
    if (tempAlgParams.buffInfo.inputPtr == tempAlgParams.buffInfo.outputPtr) {
         HCCL_INFO("[InsTempScatterMesh1DZAxisDetour][PreCopy] skip precopy due to inplace, myRank = %u, root = %u", myRank_, root_);
        return HCCL_SUCCESS;
    }
    u32 myAlgRank = 0;
    GetAlgRank(myRank_, subCommRanks_[0], myAlgRank);
    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        u64 srcOffset = tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER
                            ? r * tempAlgParams.inputRepeatStride + tempAlgParams.inputSliceStride * myAlgRank +
                                  tempAlgParams.buffInfo.hcclBuffBaseOff
                            : r * tempAlgParams.inputRepeatStride + tempAlgParams.inputSliceStride * myAlgRank +
                                  tempAlgParams.buffInfo.inBuffBaseOff;
        u64 dstOffset = tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER
                            ? r * tempAlgParams.outputRepeatStride + tempAlgParams.outputSliceStride * myAlgRank +
                                  tempAlgParams.buffInfo.hcclBuffBaseOff
                            : r * tempAlgParams.outputRepeatStride + tempAlgParams.outputSliceStride * myAlgRank +
                                  tempAlgParams.buffInfo.outBuffBaseOff;
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, processSize_, count_);
        DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr, dstOffset, processSize_, count_);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1DZAxisDetour::PostCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads) const
{
    bool isInplaceRootRank = u32(myRank_) == root_ &&
        tempAlgParams.buffInfo.inputPtr == tempAlgParams.buffInfo.outputPtr;
    // 通信结束之后，非root rank数据都在 cclBuffer 上，需要搬运到对应的输出位置
    if (tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER || enableRemoteMemAccess_) {
        HCCL_INFO("[InsTempScatterMesh1DZAxisDetour][PostCopy] skip postcopy, myRank = %u, root = %u", myRank_, root_);
        return HCCL_SUCCESS;
    }

    //root rank数据在非inplace场景不需要进行搬运
    if (u32(myRank_) == root_ && tempAlgParams.buffInfo.inputPtr != tempAlgParams.buffInfo.outputPtr) {
        HCCL_INFO("[InsTempScatterMesh1DZAxisDetour][PostCopy] root rank skip postcopy, myRank = %u, root = %u", myRank_, root_);
        return HCCL_SUCCESS;
    }

    u32 myAlgRank = 0;
    GetAlgRank(myRank_, subCommRanks_[0], myAlgRank);
    void* srcPtr = isInplaceRootRank ? tempAlgParams.buffInfo.inputPtr : tempAlgParams.buffInfo.hcclBuff.addr;
    u64 srcOffset = isInplaceRootRank
                        ? tempAlgParams.buffInfo.inBuffBaseOff + tempAlgParams.inputSliceStride * myAlgRank
                        : tempAlgParams.buffInfo.hcclBuffBaseOff + tempAlgParams.outputSliceStride * myAlgRank;
    DataSlice srcSlice = DataSlice(srcPtr, srcOffset,
        processSize_ * tempAlgParams.repeatNum, count_ * tempAlgParams.repeatNum);
    DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
        tempAlgParams.buffInfo.outBuffBaseOff + tempAlgParams.outputSliceStride * myAlgRank,
        processSize_ * tempAlgParams.repeatNum, count_ * tempAlgParams.repeatNum);
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1DZAxisDetour::RunMesh(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    u32 myAlgRank = 0;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    GetAlgRank(myRank_, subCommRanks_[0], myAlgRank);
    HCCL_INFO("[InsTempScatterMesh1DZAxisDetour][RunMesh] myRank[%d], myAlgRank[%d], channelsPerRank_[%u]",
              myRank_, myAlgRank, channelsPerRank_);
    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        if (root_ == u32(myRank_)) {
            u32 queIdx = 0;
            for (u32 algRank = 0; algRank < subCommRanks_[0].size(); algRank++) {
                if (myAlgRank == algRank) {
                    continue;
                }
                u32 remoteRank = subCommRanks_[0][algRank];
                u64 curSliceSize = (tempAlgParams.tailSize != 0 && algRank == templateRankSize_ - 1)
                                       ? tempAlgParams.tailSize : processSize_;
                u64 curCount = curSliceSize / dataTypeSize;
                CHK_PRT_RET(channels.find(remoteRank) == channels.end() || channels.at(remoteRank).empty(),
                            HCCL_ERROR("[InsTempScatterMesh1DZAxisDetour][RunMesh] remoteRank[%d] not found",
                                       remoteRank),
                            HCCL_E_INTERNAL);
                const std::vector<ChannelInfo> &curChannels = channels.at(remoteRank);
                CHK_RET(CalcDataSplitByPortGroup(curCount, dataTypeSize, curChannels,
                                                  elemCountOut_, sizeOut_, elemOffset_));
                for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
                    const ChannelInfo &linkSend = curChannels[channelIdx];
                    u64 sliceSize = sizeOut_[channelIdx];
                    u64 sliceCount = elemCountOut_[channelIdx];
                    u64 elemOff = elemOffset_[channelIdx];
                    u64 srcOffset = tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER
                                        ? tempAlgParams.buffInfo.hcclBuffBaseOff + r * tempAlgParams.inputRepeatStride +
                                              algRank * tempAlgParams.inputSliceStride + elemOff
                                        : r * tempAlgParams.inputRepeatStride + algRank * tempAlgParams.inputSliceStride +
                                              tempAlgParams.buffInfo.inBuffBaseOff + elemOff;
                    u64 dstOffset = (!enableRemoteMemAccess_)
                                        ? tempAlgParams.buffInfo.hcclBuffBaseOff + algRank * tempAlgParams.outputSliceStride +
                                              r * tempAlgParams.outputRepeatStride + elemOff
                                        : tempAlgParams.buffInfo.outBuffBaseOff + algRank * tempAlgParams.outputSliceStride +
                                              r * tempAlgParams.outputRepeatStride + elemOff;
                    void* txDstPtr = (!enableRemoteMemAccess_) ? linkSend.remoteCclMem.addr
                                                               : linkSend.remoteOutputGraphMode.addr;
                    DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, sliceSize, sliceCount);
                    DataSlice dstSlice = DataSlice(txDstPtr, dstOffset, sliceSize, sliceCount);
                    SlicesList txSlicesList({srcSlice}, {dstSlice});
                    DataInfo sendData(linkSend, txSlicesList);
                    CHK_PRT_RET(queIdx >= threads.size(),
                                HCCL_ERROR("[InsTempScatterMesh1DZAxisDetour][RunMesh] queIdx[%d] >= threads.size[%d]",
                                           queIdx, threads.size()),
                                HCCL_E_INTERNAL);
                    CHK_PRT_RET(static_cast<HcclResult>(SendWrite(sendData, threads.at(queIdx))),
                                HCCL_ERROR("[InsTempScatterMesh1DZAxisDetour] RunMesh Send failed"),
                                HcclResult::HCCL_E_INTERNAL);
                    queIdx++;
                }
            }
        } else {
            if (channels.size() == 0 || channels.count(root_) == 0) {
                continue;
            }
            CHK_PRT_RET(channels.find(root_) == channels.end() || channels.at(root_).empty(),
                        HCCL_ERROR("[InsTempScatterMesh1DZAxisDetour][RunMesh] root[%d] not found", root_),
                        HCCL_E_INTERNAL);
            const std::vector<ChannelInfo> &curChannels = channels.at(root_);
            u64 curSliceSize = (tempAlgParams.tailSize != 0 && myAlgRank == templateRankSize_ - 1)
                                   ? tempAlgParams.tailSize : processSize_;
            u64 curCount = curSliceSize / dataTypeSize;
            CHK_RET(CalcDataSplitByPortGroup(curCount, dataTypeSize, curChannels,
                                              elemCountOut_, sizeOut_, elemOffset_));
            u32 queIdx = 0;
            for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
                const ChannelInfo &linkRecv = curChannels[channelIdx];
                u64 sliceSize = sizeOut_[channelIdx];
                u64 sliceCount = elemCountOut_[channelIdx];
                u64 elemOff = elemOffset_[channelIdx];
                u64 dstOffset = (!enableRemoteMemAccess_)
                                    ? tempAlgParams.buffInfo.hcclBuffBaseOff + myAlgRank * tempAlgParams.outputSliceStride +
                                          r * tempAlgParams.outputRepeatStride + elemOff
                                    : tempAlgParams.buffInfo.outBuffBaseOff + myAlgRank * tempAlgParams.outputSliceStride +
                                          r * tempAlgParams.outputRepeatStride + elemOff;
                void* rxDstPtr = (!enableRemoteMemAccess_) ? tempAlgParams.buffInfo.hcclBuff.addr
                                                           : tempAlgParams.buffInfo.outputPtr;
                DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, 0, sliceSize, sliceCount);
                DataSlice dstSlice = DataSlice(rxDstPtr, dstOffset, sliceSize, sliceCount);
                SlicesList rxSlicesList({srcSlice}, {dstSlice});
                DataInfo recvData(linkRecv, rxSlicesList);
                CHK_PRT_RET(queIdx >= threads.size(),
                            HCCL_ERROR("[InsTempScatterMesh1DZAxisDetour][RunMesh] queIdx[%d] >= threads.size[%d]",
                                       queIdx, threads.size()),
                            HCCL_E_INTERNAL);
                CHK_PRT_RET(static_cast<HcclResult>(RecvWrite(recvData, threads.at(queIdx))),
                            HCCL_ERROR("[InsTempScatterMesh1DZAxisDetour] RunMesh Recv failed"),
                            HcclResult::HCCL_E_INTERNAL);
                queIdx++;
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
