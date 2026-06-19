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
#include "all_reduce_biar.h"

namespace ops_hccl_experimental {


AllReduceBIAR::AllReduceBIAR() : AlgTemplateBaseExperimental()
{
}

AllReduceBIAR::~AllReduceBIAR()
{
}

HcclResult AllReduceBIAR::Prepare(u32 interRank, u32 interRankSize)
{
    interRank_ = interRank;
    interRankSize_ = interRankSize;
    return HCCL_SUCCESS;
}

HcclResult AllReduceBIAR::Prepare(HcclMem &inputMem, HcclMem &outputMem, HcclMem &scratchMem,
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

void AllReduceBIAR::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = 3;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void AllReduceBIAR::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = 3;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult AllReduceBIAR::Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    if (rankSize == 1) {
        if (inputMem_.addr != outputMem_.addr) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, outputMem_.addr, inputMem_.addr, inputMem_.size)));
        }
        return HCCL_SUCCESS;
    }

    if (channels.size() < rankSize) {
        HCCL_ERROR("[AllReduceBIAR][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]",
            rank, channels.size(), rankSize);
        return HCCL_E_INTERNAL;
    }
    
    unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[AllReduceBIAR][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult AllReduceBIAR::LocalReduceCCLToCCL(u64 srcOffset, u64 dstOffset, u64 size, ThreadHandle thread) {
    void* srcSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + srcOffset);
    void* dstSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + dstOffset);
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, dstSlice, srcSlice, size / unitSize, static_cast<HcommDataType>(dataType_), static_cast<HcommReduceOp>(reductionOp_))));
    return HCCL_SUCCESS;
}

HcclResult AllReduceBIAR::HCCSProcessRsLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize) 
{
    if (round != 0) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links[round - 1].handle, NOTIFY_IDX_ACK)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(subThreads[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            u64 localOffsetByte = hccs_ranks[round - 1] / rankSizeX_ * localStrideSize;
            u64 remoteOffsetByte = ((rankSize / rankSizeX_) + rank / rankSizeX_) * localStrideSize;
            void* src = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + localOffsetByte);
            void* dst = static_cast<void *>(static_cast<u8 *>(hccs_links[round - 1].remoteOutput.addr) + remoteOffsetByte);
            
            HcommWriteOnThread(subThreads[0], hccs_links[round - 1].handle, dst, src, sliceSize);

            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(subThreads[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceBIAR::SIOProcessRsLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize) 
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    
    u64 localOffsetByte = (round != hccs_ranks.size()) ? slices_[hccs_neighbour_rank[round]].offset : slices_[sio_rank].offset;
    u64 remoteOffsetByte = (round != hccs_ranks.size()) ? hccs_ranks[round] / rankSizeX_ * localStrideSize : rank / rankSizeX_ * localStrideSize;
    void* src = static_cast<void *>(static_cast<u8 *>(inputMem_.addr) + localOffsetByte);
    void* dst = static_cast<void *>(static_cast<u8 *>(sio_link.remoteOutput.addr) + remoteOffsetByte);

    HcommWriteReduceOnThread(mainThread, sio_link.handle, dst, src, sliceSize / unitSize, static_cast<HcommDataType>(dataType_), static_cast<HcommReduceOp>(reductionOp_));
    
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult AllReduceBIAR::LocalCopyRsLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize) 
{
    if (round < hccs_ranks.size()) {
        u32 rank_idx = (round < hccs_ranks.size() - 1) ? hccs_ranks[round + 1] : rank;
        void* srcSlice = static_cast<void *>(static_cast<u8 *>(inputMem_.addr) + slices_[rank_idx].offset);
        void* dstSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + rank_idx / rankSizeX_ * localStrideSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(subThreads[1], dstSlice, srcSlice, sliceSize)));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceBIAR::TreeReduceRs(const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize)
{
    std::vector<u32> vec;
    for (u32 i = 0; i < (rankSize / rankSizeX_); i++){
        if (i == (rank / rankSizeX_)) {
            vec.push_back((rank / rankSizeX_) * localStrideSize);
        } else {
            vec.push_back(((rankSize / rankSizeX_) + i) * localStrideSize);
        }
    }
    //Tree local reduce
    auto ind = rankSize / rankSizeX_;
    for (u32 stride = 1; stride < ind; stride *= 2) {
        for (u32 i = stride; i < ind; i += stride * 2) {
            LocalReduceCCLToCCL(vec[i], vec[i - stride], sliceSize, mainThread);
        }
    }

    u64 AllGatherOffset = rank / rankSizeX_ * localStrideSize;
    
    if (vec[0] != AllGatherOffset){ 
        void* srcSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + vec[0]);
        void* dstSlice =  static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + AllGatherOffset);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize))); 
    }

    return HCCL_SUCCESS;
}


HcclResult AllReduceBIAR::RunReduceScatter(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels, u64 sliceSize, u64 localStrideSize)
{
    for (u32 round = 0; round < hccs_ranks.size() + 1; round++) {
        //MainRecordSub + SubWaitMain
        PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
        
        HCCSProcessRsLoop(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);
        SIOProcessRsLoop(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);
        LocalCopyRsLoop(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);

        //SubRecordMain + MainWaitSub
        PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);
    }

    PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
    
    TreeReduceRs(rank, rankSize, rankSizeX_, sliceSize, localStrideSize);
    
    PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);

    return HCCL_SUCCESS; 
}


HcclResult AllReduceBIAR::HCCSProcessAgLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize) 
{
    u64 AllGatherOffset = rank / rankSizeX_ * localStrideSize;
    if (round != hccs_ranks.size()) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links[round].handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(subThreads[0], hccs_links_reversed[round].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        u64 localOffsetByte = 0;
        u64 remoteOffsetByte = (rank / rankSizeX_) * localStrideSize;
        void* src = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + AllGatherOffset);
        void* dst = static_cast<void *>(static_cast<u8 *>(hccs_links[round].remoteOutput.addr) + remoteOffsetByte);
        
        HcommWriteOnThread(subThreads[0], hccs_links[round].handle, dst, src, sliceSize);

        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links[round].handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(subThreads[0], hccs_links_reversed[round].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceBIAR::SIOProcessAgLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize) 
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

HcclResult AllReduceBIAR::LocalCopyAgLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize) 
{
    if (round != 0) {
        void* srcSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + (hccs_ranks_reversed[round - 1] / rankSizeX_) * localStrideSize);
        void* dstSlice = static_cast<void *>(static_cast<u8 *>(outputMem_.addr) + hccs_ranks_reversed[round - 1] * sliceSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(subThreads[1], dstSlice, srcSlice, sliceSize)));
    }
    return HCCL_SUCCESS;
}


HcclResult AllReduceBIAR::RunAllGather(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels, u64 sliceSize, u64 localStrideSize)
{
    for (u32 round = 0; round < hccs_ranks.size() + 1; round++) {
        //MainRecordSub + SubWaitMain
        PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
        HCCSProcessAgLoop(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);
        SIOProcessAgLoop(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);
        LocalCopyAgLoop(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);

        //SubRecordMain + MainWaitSub
        PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);
    }
    PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
    u64 AllGatherOffset = rank / rankSizeX_ * localStrideSize;
    void* srcSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + AllGatherOffset);
    void* dstSlice =  static_cast<void *>(static_cast<u8 *>(outputMem_.addr) + rank * sliceSize);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize)));

    PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);
    return HCCL_SUCCESS; 
}

HcclResult AllReduceBIAR::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    HCCL_INFO("AllReduceBIAR run: rank[%u] rankSize[%u] inputMem[%p] to outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.addr, outputMem_.addr, count_);
    
    Preprocess(rank, rankSize, channels);

    rankSizeX_ = 2;
    if (rankSize % rankSizeX_ != 0) {
        HCCL_ERROR("[AllReduceBIAR][RunAsync]rankSize[%u] is not evenly divisible by rankSizeX_[%u]", rankSize, rankSizeX_);
        return HCCL_E_INTERNAL;
    }
    rankSizeY_ = rankSize / rankSizeX_;

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
    
    u64 totalSize = count_ * unitSize;
    u64 sliceSize = (totalSize + rankSize - 1) / rankSize;
    u64 localStrideSize = RoundUpWithDivisor(sliceSize, HCCL_MIN_SLICE_ALIGN_910B);

    slices_.resize(rankSize);
    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
        HCCL_DEBUG(" default slice[%u]: offset: [%llu] size[%llu]", i, i * sliceSize, sliceSize);
    }
    
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
    
    void* srcSlice = static_cast<void *>(static_cast<u8 *>(inputMem_.addr) + slices_[hccs_ranks[0]].offset);
    void* dstSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr) + hccs_ranks[0] / rankSizeX_ * localStrideSize);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize)));

    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);
    
    //MainRecordSub + SubWaitMain
    RunReduceScatter(rank, rankSize, channels, sliceSize, localStrideSize);
    RunAllGather(rank, rankSize, channels, sliceSize, localStrideSize);
    
    HCCL_INFO("AllReduceBIAR finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TEMPLATE_ALL_REDUCE_BIAR, AllReduceBIAR);
}
