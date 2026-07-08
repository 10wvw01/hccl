/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_mini_reduce_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "channel.h"
#include "alg_v2_template_register.h"
#include "hccl_common.h"

namespace ops_hccl {

InsTempMiniReduceMesh1D::InsTempMiniReduceMesh1D(const OpParam &param, const u32 rankId,
                                                 const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

HcclResult InsTempMiniReduceMesh1D::CalcRes(HcclComm comm, const OpParam &param,
                                            const TopoInfoWithNetLayerDetails *topoInfo,
                                            AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempMiniReduceMesh1D][CalcRes] start, rank[%u] rankSize[%u]", myRank_, templateRankSize_);

    u32 level0RankSize = templateRankSize_;
    // 每个非 root rank 需要一个 thread 与 root 通信
    // root 需要 (rankSize - 1) 个 thread 并行接收
    u32 threadNum = (level0RankSize > 1) ? level0RankSize : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    // 申请 Mesh 1D channel (所有 rank 对之间)
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);

    HCCL_INFO("[InsTempMiniReduceMesh1D][CalcRes] slaveThreadNum[%u] notifyNumOnMainThread[%u] channels[%zu]",
              resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, level0Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempMiniReduceMesh1D::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ : 1;
}

u64 InsTempMiniReduceMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return (myRank_ == root_) ? 1 : 0;
}

HcclResult InsTempMiniReduceMesh1D::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                              TemplateResource &templateResource)
{
    HCCL_INFO("[InsTempMiniReduceMesh1D] KernelRun start, rank[%u] rankSize[%u] root[%u]",
              myRank_, templateRankSize_, root_);

    // 单rank: root 直接拷贝 input → output
    if (templateRankSize_ == 1) {
        if (myRank_ == root_) {
            u64 dataSize = param.DataDes.count * DATATYPE_SIZE_TABLE[param.DataDes.dataType];
            DataSlice srcSlice(tempAlgParams.buffInfo.inputPtr, 0, dataSize, param.DataDes.count);
            DataSlice dstSlice(tempAlgParams.buffInfo.outputPtr, 0, dataSize, param.DataDes.count);
            CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        }
        return HCCL_SUCCESS;
    }

    threadNum_ = templateResource.threads.size();
    dataType_ = param.DataDes.dataType;

    // 多线程同步: main 线程通知所有 sub 线程开始
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    // 根据是否是 root 执行不同的逻辑
    if (myRank_ == root_) {
        CHK_RET(RunRootRecv(param, tempAlgParams, templateResource.threads, templateResource.channels));
    } else {
        CHK_RET(RunNonRootSend(param, tempAlgParams, templateResource.threads, templateResource.channels));
    }

    // 多线程同步: sub 线程通知 main 线程完成
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempMiniReduceMesh1D] KernelRun End, rank[%u]", myRank_);
    return HCCL_SUCCESS;
}

HcclResult InsTempMiniReduceMesh1D::RunRootRecv(const OpParam &param,
                                                const TemplateDataParams &tempAlgParams,
                                                const std::vector<ThreadHandle> &threads,
                                                const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempMiniReduceMesh1D] RunRootRecv rank[%u]", myRank_);

    u64 dataSize = param.DataDes.count * DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    void *inputPtr = tempAlgParams.buffInfo.inputPtr;
    void *outputPtr = tempAlgParams.buffInfo.outputPtr;
    HcclMem cclMem = tempAlgParams.buffInfo.hcclBuff;

    // Root: 从每个非 root rank 接收数据
    u32 threadIdx = 0;
    for (const auto &rankId : subCommRanks_[0]) {
        if (rankId == myRank_) {
            continue;
        }

        CHK_PRT_RET(threadIdx >= threads.size() || channels.count(rankId) == 0 ||
                    channels.at(rankId).empty(),
                    HCCL_ERROR("[InsTempMiniReduceMesh1D][RunRootRecv] rank[%u] threadIdx[%u] "
                               "connectedRank[%u] channels invalid", myRank_, threadIdx, rankId),
                    HCCL_E_INTERNAL);

        const ChannelInfo &linkRemote = channels.at(rankId)[0];

        // 从远端读取数据到 CCL buffer (不同 rank 使用不同偏移避免覆盖)
        u64 rankOffset = threadIdx * dataSize;
        DataSlice recvSlice(cclMem.addr, rankOffset, dataSize, param.DataDes.count);
        std::vector<DataSlice> emptySlices;
        TxRxSlicesList sendRecvSlicesList({emptySlices, emptySlices}, {recvSlice, emptySlices});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
        CHK_RET(SendRecvRead(sendRecvInfo, threads[threadIdx]));

        threadIdx++;
    }

    // 先将 root 自身的 input 拷贝到 output
    DataSlice srcSlice(inputPtr, 0, dataSize, param.DataDes.count);
    DataSlice dstSlice(outputPtr, 0, dataSize, param.DataDes.count);
    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    // 对远端数据逐份做 LocalReduce 到 output
    threadIdx = 0;
    for (const auto &rankId : subCommRanks_[0]) {
        if (rankId == myRank_) {
            continue;
        }
        u64 rankOffset = threadIdx * dataSize;
        DataSlice remoteSlice(cclMem.addr, rankOffset, dataSize, param.DataDes.count);
        // 将远端数据 reduce 到 output
        CHK_RET(LocalReduce(threads[0], remoteSlice, dstSlice, param.DataDes.dataType, param.reduceType));
        threadIdx++;
    }

    HCCL_INFO("[InsTempMiniReduceMesh1D] RunRootRecv End, rank[%u]", myRank_);
    return HCCL_SUCCESS;
}

HcclResult InsTempMiniReduceMesh1D::RunNonRootSend(const OpParam &param,
                                                   const TemplateDataParams &tempAlgParams,
                                                   const std::vector<ThreadHandle> &threads,
                                                   const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempMiniReduceMesh1D] RunNonRootSend rank[%u] root[%u]", myRank_, root_);

    CHK_PRT_RET(channels.count(root_) == 0 || channels.at(root_).empty(),
                HCCL_ERROR("[InsTempMiniReduceMesh1D][RunNonRootSend] rank[%u] no channel to root[%u]",
                           myRank_, root_),
                HCCL_E_INTERNAL);

    const ChannelInfo &linkToRoot = channels.at(root_)[0];
    u64 dataSize = param.DataDes.count * DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    void *inputPtr = tempAlgParams.buffInfo.inputPtr;

    // 将本地 input 发送给 root
    DataSlice sendSlice(inputPtr, 0, dataSize, param.DataDes.count);
    std::vector<DataSlice> emptySlices;
    TxRxSlicesList sendRecvSlicesList({sendSlice, emptySlices}, {emptySlices, emptySlices});
    TxRxChannels sendRecvChannels(linkToRoot, linkToRoot);
    SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
    CHK_RET(SendRecvWrite(sendRecvInfo, threads[0]));

    HCCL_INFO("[InsTempMiniReduceMesh1D] RunNonRootSend End, rank[%u]", myRank_);
    return HCCL_SUCCESS;
}

void InsTempMiniReduceMesh1D::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempMiniReduceMesh1D::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

// 注册模板到全局注册表
REGISTER_TEMPLATE_V2("InsTempMiniReduceMesh1D", InsTempMiniReduceMesh1D);

}  // namespace ops_hccl
