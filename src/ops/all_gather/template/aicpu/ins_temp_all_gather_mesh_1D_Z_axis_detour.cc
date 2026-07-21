/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {
InsTempAllGatherMesh1D1DZAxisDetour::InsTempAllGatherMesh1D1DZAxisDetour(const OpParam &param, const u32 rankId,
                                               const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAllGatherMesh1D(param, rankId, subCommRanks)
{
}
InsTempAllGatherMesh1D1DZAxisDetour::~InsTempAllGatherMesh1D1DZAxisDetour() {}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                           AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][CalcRes] start");
    CHK_PRT_RET(topoInfo == nullptr,
        HCCL_ERROR("[InsTempAllGatherMesh1D1DZAxisDetour][CalcRes] topoInfo is nullptr"), HCCL_E_PARA);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel0(comm, param, topoInfo, subCommRanks_, level0Channels));
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel1(comm, param, topoInfo, subCommRanks_, level1Channels));
    level0ChannelNumPerRank_ = level0Channels.empty() ? 0 : CalcChannelsPerRank(level0Channels);
    level1ChannelNumPerRank_ = level1Channels.empty() ? 0 : CalcChannelsPerRank(level1Channels);
    channelsPerRank_ = level0ChannelNumPerRank_ + level1ChannelNumPerRank_;
    std::vector<HcclChannelDesc> mergedChannels;
    HCCL_INFO("level0Channels[%d]level1Channels[%d]\n", level0Channels.size(), level1Channels.size());
    mergedChannels.insert(mergedChannels.end(), level0Channels.begin(), level0Channels.end());
    mergedChannels.insert(mergedChannels.end(), level1Channels.begin(), level1Channels.end());
    resourceRequest.channels.push_back(mergedChannels);
    HCCL_INFO("mergedChannels[%d]\n", mergedChannels.size());

    if(subCommRanks_.size() <= COMM_LEVEL0) {
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(templateRankSize_ > 1 && channelsPerRank_ == 0,
        HCCL_ERROR("[InsTempAllGatherMesh1D1DZAxisDetour][CalcRes] channelsPerRank_ is 0"), HCCL_E_INTERNAL);
    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherMesh1D1DZAxisDetour::GetThreadNum() const
{
    u32 threadNum = templateRankSize_ > 1 ? ((templateRankSize_ - 1) * channelsPerRank_) : 1;
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][GetThreadNum] templateRankSize_[%u] channelsPerRank_[%u] threadNum[%u]", templateRankSize_, channelsPerRank_, threadNum);
    return threadNum;
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::CalcDataSplitByPortGroup(
    const u64 totalDataCount, const u64 dataTypeSize,
    const std::vector<ChannelInfo> &channels,
    std::vector<u64> &elemCountOut, std::vector<u64> &sizeOut,
    std::vector<u64> &elemOffset)
{
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][CalcDataSplitByPortGroup] Run Start[%u][%u][%f]\n", level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
    return CalcDataSplitByPortGroupZAxisDetour(totalDataCount, dataTypeSize, channels,
        elemCountOut, sizeOut, elemOffset,
        level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::SetchannelsPerRank(
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[SetchannelsPerRank] channels is empty."), HCCL_E_INTERNAL);
    channelsPerRank_ = CalcChannelsPerRank(channels);
    if (channelsPerRank_ > 1) {
        level0ChannelNumPerRank_ = MESH_CHANNELS_NUM;
        level1ChannelNumPerRank_ = channelsPerRank_ - level0ChannelNumPerRank_;
        level0DataRatio_ = 0.5f;
    }
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][SetchannelsPerRank], channelsPerRank_[%u], "
              "level0ChannelNumPerRank_[%u], level1ChannelNumPerRank_[%u], level0DataRatio_[%.2f]",
              channelsPerRank_, level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] LocalDataCopy.");
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 sliceSize = tempAlgParams_.sliceSize;
    if (tempAlgParams_.tailSize != 0 && myAlgRank == templateRankSize_ - 1) {
        sliceSize = tempAlgParams_.tailSize;
    }
    u64 sliceCount = sliceSize / dataTypeSize;

    // Z轴绕路场景下，input已经是框间NHR写入cclBuff的一维[rpt]布局，无需再做input->cclBuff的自拷贝，
    // 仅保留input->output的OUT-COPY即可。基类的ccl-copy会按[sliceSize*myAlgRank]二维布局重排一维cclBuff，
    // 导致cclBuff溢出与数据错乱，故此处override去掉ccl-copy分支。
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff;
        const u64 outOff = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        bool skipOutCopy = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr && inOff == outOff);
        if (!skipOutCopy) {
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, sliceSize, sliceCount);
            LocalCopy(threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::RunAllGatherMesh(const std::vector<ThreadHandle> &threads,
                                                                 const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] RunAllGatherMesh RankIDs[%d].", myRank_);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u32 queIdx = 0;
    for (u32 rankIdx = 1; rankIdx < templateRankSize_; rankIdx++) {
        u32 connectedAlgRank = (myAlgRank + rankIdx) % templateRankSize_;
        u32 connectedRank = subCommRanks_[0][connectedAlgRank];

        HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour] RunAllGatherMesh RankIDs[%d], connectedRank[%d], connectedAlgRank[%d].",
                  myRank_, connectedRank, connectedAlgRank);

        CHK_PRT_RET(channels.count(connectedRank) == 0 || channels.at(connectedRank).empty(),
                    HCCL_ERROR("[InsTempAllGatherMesh1D1DZAxisDetour][RankID]=%u connectedRank=%d, channels.size=%zu",
                               myRank_, connectedRank, channels.size()),
                    HcclResult::HCCL_E_INTERNAL);

        const std::vector<ChannelInfo> &curChannels = channels.at(connectedRank);
        u64 sliceSize = tempAlgParams_.sliceSize;
        if (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1) {
            sliceSize = tempAlgParams_.tailSize;
        }
        u64 sliceCount = sliceSize / dataTypeSize;
        CHK_RET(CalcDataSplitByPortGroup(sliceCount, dataTypeSize, curChannels, elemCountOut_, sizeOut_, elemOffset_));

        void *remoteOut = nullptr;
        if (supportSymmetricMemory_) {
            HcclResult ret = HcclSymWinGetPeerPointer(outputSymWindow_, outputOffset_, connectedRank, &remoteOut);
            CHK_PRT_RET(ret != HCCL_SUCCESS || remoteOut == nullptr,
                        HCCL_ERROR("[InsTempAllGatherSymmetryMemoryMesh1D] HcclSymWinGetPeerPointer failed, "
                            "remoteRank[%u] outputRet[%d] out[%p]", connectedRank, ret, remoteOut),
                        HcclResult::HCCL_E_INTERNAL);
            HCCL_INFO("[InsTempAllGatherSymmetryMemoryMesh1D] HcclSymWinGetPeerPointer success, "
                "remoteRank[%u] out[%p]", connectedRank, remoteOut);
        }

        const u32 curChannelNum = std::min(static_cast<u32>(curChannels.size()), channelsPerRank_);
        for (u32 channelIdx = 0; channelIdx < curChannelNum; channelIdx++) {
            CHK_PRT_RET(queIdx >= threads.size(),
                        HCCL_ERROR("[InsTempAllGatherMesh1D1DZAxisDetour][RankID]=%u queIdx=%u, threads.size=%zu",
                                   myRank_, queIdx, threads.size()),
                        HcclResult::HCCL_E_INTERNAL);
            const ChannelInfo &linkRemote = curChannels[channelIdx];
            void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
            std::vector<DataSlice> rxDstSlicesAll;
            std::vector<DataSlice> rxSrcSlicesAll;

            for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
                const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
                // Z轴绕路场景下，框间NHR把每个rpt段按inputRepeatStride步进写入cclBuff，形成一维[rpt]布局，
                // 每段只存remote自己那份数据；rx读取时所有connectedAlgRank都应从该rpt段起点读，
                // 即用inputSliceStride(框间写入步进，intra阶段为0)索引，而非基类standalone的sliceSize。
                u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
                if (tempAlgParams_.buffInfo.inBuffType != BufferType::HCCL_BUFFER) {
                    const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
                    scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
                }

                u64 rxOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff + elemOffset_[channelIdx];
                u64 rxSrcOffset = 0;
                void *rxSrcPtr = nullptr;
                void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;

                if (!supportSymmetricMemory_) {
                    u64 rxScratchOffset = scratchBase + tempAlgParams_.inputSliceStride * connectedAlgRank + elemOffset_[channelIdx];
                    rxSrcOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;
                    rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
                } else {
                    rxSrcOffset = rxOutOffset;
                    rxSrcPtr = remoteOut;
                }

                rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sizeOut_[channelIdx], elemCountOut_[channelIdx]);
                rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sizeOut_[channelIdx], elemCountOut_[channelIdx]);

                HCCL_DEBUG("[InsTempAllGatherMesh1D1DZAxisDetour][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] "
                           "channelIdx[%u] offset[%llu] sliceSize[%llu] count[%llu].",
                           myRank_, connectedRank, rpt, channelIdx, elemOffset_[channelIdx],
                           sizeOut_[channelIdx], elemCountOut_[channelIdx]);
            }

            std::vector<DataSlice> emptySlices;
            TxRxSlicesList sendRecvSlicesList({emptySlices, emptySlices}, {rxSrcSlicesAll, rxDstSlicesAll});
            TxRxChannels sendRecvChannels(linkRemote, linkRemote);
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
            CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[queIdx]),
                        HCCL_ERROR("[InsTempAllGatherMesh1D1DZAxisDetour] RunAllGather Send failed"),
                        HcclResult::HCCL_E_INTERNAL);
            queIdx++;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl
