/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_register.h"
#include "all_gather_birs.h"

namespace ops_hccl_experimental {


AllGatherBIRS::AllGatherBIRS() : AlgTemplateBaseExperimental()
{
}

AllGatherBIRS::~AllGatherBIRS()
{
}

HcclResult AllGatherBIRS::Prepare(u32 interRank, u32 interRankSize)
{
    interRank_ = interRank;
    interRankSize_ = interRankSize;
    return HCCL_SUCCESS;
}

HcclResult AllGatherBIRS::Prepare(HcclMem &inputMem, HcclMem &outputMem, HcclMem &scratchMem,
                                 const u64 count,
                                 const HcclDataType dataType, ThreadHandle thread, const std::vector<ThreadHandle> &slaveThreads,
                                 const HcclReduceOp reductionOp,
                                 const u32 root, const std::vector<Slice> &slices, const u64 baseOffset,
                                 const bool disableDMAReduce)
{
    mainThread = thread;
    subThreads = slaveThreads;
    AlgTemplateBase::Prepare(inputMem, outputMem, scratchMem,
                                 count, dataType, thread, reductionOp, root, slices, baseOffset,
                                 disableDMAReduce);
    return HCCL_SUCCESS;
}

void AllGatherBIRS::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = 3;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void AllGatherBIRS::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = 3;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult AllGatherBIRS::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
{
    slices_.resize(rankSize);
    u64 sliceSize = totalCount * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
        HCCL_DEBUG(" default slice[%u]: offset: [%llu] size[%llu]", i, i * sliceSize, sliceSize);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBIRS::Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    if (rankSize == 1) {
        if (inputMem_.addr != outputMem_.addr) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, outputMem_.addr, inputMem_.addr, inputMem_.size)));
        }
        return HCCL_SUCCESS;
    }

    if (channels.size() < rankSize) {
        HCCL_ERROR("[AllGatherBIRS][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]",
            rank, channels.size(), rankSize);
        return HCCL_E_INTERNAL;
    }
    
    unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[AllGatherBIRS][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, rankSize);
    }
    return HCCL_SUCCESS;
}


HcclResult AllGatherBIRS::HCCSProcessMainLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize) 
{
    if (round != hccs_ranks.size()) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links[round].handle, NOTIFY_IDX_ACK)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(subThreads[0], hccs_links_reversed[round].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            u64 localOffsetByte = 0;
            u64 remoteOffsetByte = (rank / rankSizeX_) * localStrideSize;
            void* src = static_cast<void *>(static_cast<u8 *>(inputMem_.addr));
            void* dst = static_cast<void *>(static_cast<u8 *>(hccs_links[round].remoteOutput.addr) + remoteOffsetByte);
            
            HcommWriteOnThread(subThreads[0], hccs_links[round].handle, dst, src, sliceSize);

            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links[round].handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(subThreads[0], hccs_links_reversed[round].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBIRS::SIOProcessMainLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize) 
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    
    u64 localOffsetByte = (round != 0) ? (hccs_ranks_reversed[round - 1] ^ 1) * sliceSize : sio_rank * sliceSize;
    u64 remoteOffsetByte = (round != 0) ? (hccs_ranks_reversed[round - 1] / rankSizeX_) * localStrideSize : (sio_rank / rankSizeX_) * localStrideSize;
    void* src = static_cast<void *>(static_cast<u8 *>(sio_link.remoteOutput.addr) + remoteOffsetByte);
    void* dst = static_cast<void *>(static_cast<u8 *>(outputMem_.addr) + localOffsetByte);

    HcommReadOnThread(mainThread, sio_link.handle, dst, src, sliceSize);
    
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult AllGatherBIRS::LocalCopyMainLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize) 
{
    if (round != 0) {
        void* srcSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + (hccs_ranks_reversed[round - 1] / rankSizeX_) * localStrideSize);
        void* dstSlice = static_cast<void *>(static_cast<u8 *>(outputMem_.addr) + hccs_ranks_reversed[round - 1] * sliceSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(subThreads[1], dstSlice, srcSlice, sliceSize)));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherBIRS::FinalStep(const u32 rank, const u32 rankSize, u64 sliceSize)
{
    //Local copy to output
    void* srcSlice = static_cast<void *>(static_cast<u8 *>(inputMem_.addr));
    void* dstSlice =  static_cast<void *>(static_cast<u8 *>(outputMem_.addr) + rank * sliceSize);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize)));
    return HCCL_SUCCESS;
}

HcclResult AllGatherBIRS::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    HCCL_INFO("AllGatherBIRS run: rank[%u] rankSize[%u] inputMem[%p] to outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.addr, outputMem_.addr, count_);
    
    Preprocess(rank, rankSize, channels);

    u32 rankSizeX_ = 2;
    if (rankSize % rankSizeX_ != 0) {
        HCCL_ERROR("[AllGatherBIRS][RunAsync]rankSize[%u] is not evenly divisible by rankSizeX_[%u]", rankSize, rankSizeX_);
        return HCCL_E_INTERNAL;
    }
    u32 rankSizeY_ = rankSize / rankSizeX_;

    sio_rank = rank ^ 1;
    sio_link = channels[sio_rank];

    for (u32 i = 1; i < rankSize / rankSizeX_; ++i) {
        u32 current_hccs_rank = (rank + rankSizeX_ * i) % (rankSizeX_ * rankSizeY_);
        hccs_ranks.push_back(current_hccs_rank);
        hccs_neighbour_rank.push_back(current_hccs_rank ^ 1);
        hccs_links.push_back(channels[hccs_ranks[i-1]]);
    }
    hccs_ranks_reversed.assign(hccs_ranks.rbegin(), hccs_ranks.rend());
    hccs_links_reversed.assign(hccs_links.rbegin(), hccs_links.rend());
    
    u64 sliceSize = count_ * unitSize;
    u64 localStrideSize = RoundUpWithDivisor(sliceSize, HCCL_MIN_SLICE_ALIGN_910B);

    //MainRecordSub + SubWaitMain
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
    
    void* srcSlice = static_cast<void *>(static_cast<u8 *>(inputMem_.addr));
    void* dstSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + rank / rankSizeX_ * localStrideSize);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize)));

    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);

    for (u32 round = 0; round < hccs_ranks.size() + 1; round++) {
        //MainRecordSub + SubWaitMain
        PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
        
        HCCSProcessMainLoop(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);
        
        SIOProcessMainLoop(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);

        LocalCopyMainLoop(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);

        //SubRecordMain + MainWaitSub
        PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);
    }

    // MainRecordSub + SubWaitMain
    PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);

    FinalStep(rank, rankSize, sliceSize);

    PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);

    HCCL_INFO("AllGatherBIRS finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TEMPLATE_ALL_GATHER_BIRS, AllGatherBIRS);
}
