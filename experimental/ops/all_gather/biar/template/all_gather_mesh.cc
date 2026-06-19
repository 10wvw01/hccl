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
#include "all_gather_mesh.h"

namespace ops_hccl_experimental {


AllGatherMesh::AllGatherMesh() : AlgTemplateBaseExperimental()
{
}

AllGatherMesh::~AllGatherMesh()
{
}

HcclResult AllGatherMesh::Prepare(u32 interRank, u32 interRankSize)
{
    interRank_ = interRank;
    interRankSize_ = interRankSize;
    return HCCL_SUCCESS;
}

HcclResult AllGatherMesh::Prepare(HcclMem &inputMem, HcclMem &outputMem, HcclMem &scratchMem,
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

void AllGatherMesh::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub, const u32 rankSize)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = rankSize;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void AllGatherMesh::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain, const u32 rankSize)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = rankSize;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult AllGatherMesh::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
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

HcclResult AllGatherMesh::Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    if (rankSize == 1) {
        if (inputMem_.addr != outputMem_.addr) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, outputMem_.addr, inputMem_.addr, inputMem_.size)));
        }
        return HCCL_SUCCESS;
    }

    if (channels.size() < rankSize) {
        HCCL_ERROR("[AllGatherMesh][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]",
            rank, channels.size(), rankSize);
        return HCCL_E_INTERNAL;
    }
    
    unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[AllGatherMesh][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, rankSize);
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherMesh::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    HCCL_INFO("AllGatherMesh run: rank[%u] rankSize[%u] inputMem[%p] to outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.addr, outputMem_.addr, count_);
    
    Preprocess(rank, rankSize, channels);

    u64 sliceSize = count_ * unitSize;

    //MainRecordSub + SubWaitMain
    GetNotifyIdxMainToSub(notifyIdxMainToSub_, rankSize);
    PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
    void* srcSlice = static_cast<void *>(static_cast<u8 *>(inputMem_.addr));
    void* dstSlice = static_cast<void *>(static_cast<u8 *>(scratchMem_.addr));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize)));
    for (u32 round = 0; round < rankSize - 1; round++) {
        auto dstRank = (rank + (round + 1)) % rankSize;
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(subThreads[round], channels[dstRank].handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(subThreads[round], channels[dstRank].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));    
    }

    GetNotifyIdxSubToMain(notifyIdxSubToMain_, rankSize);
    PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);

    PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_);
    srcSlice = dstSlice;
    dstSlice = static_cast<void *>(static_cast<u8 *>(outputMem_.addr) + rank * sliceSize);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize)));

    for (u32 round = 0; round < rankSize - 1; round++) {
        auto dstRank = (rank + (round + 1)) % rankSize;
        u64 localOffsetByte = dstRank * sliceSize;
        u64 remoteOffsetByte = 0;
        void* src = static_cast<void *>(static_cast<u8 *>(channels[dstRank].remoteOutput.addr) + remoteOffsetByte);
        void* dst = static_cast<void *>(static_cast<u8 *>(outputMem_.addr) + localOffsetByte);
        HcommReadOnThread(subThreads[round], channels[dstRank].handle, dst, src, sliceSize);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(subThreads[round], channels[dstRank].handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(subThreads[round], channels[dstRank].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_);

    HCCL_INFO("AllGatherMesh finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TEMPLATE_ALL_GATHER_MESH, AllGatherMesh);
}
