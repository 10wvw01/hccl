/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_alltoall_mesh_2d_v3_no_memcpy.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {
namespace {
bool IsAlltoAllVNoMemcpyParams(const TemplateDataParams &params, u32 rankSize)
{
    return params.sendCounts.size() >= rankSize &&
           params.recvCounts.size() >= rankSize &&
           params.sdispls.size() >= rankSize &&
           params.rdispls.size() >= rankSize;
}

HcclResult CheckNoMemcpySliceRange(const char *tag, u32 myRank, u32 peerRank, u64 srcOffset, u64 dstOffset,
                                   u64 byteSize, u64 inputSize, u64 remoteOutputSize)
{
    CHK_PRT_RET(srcOffset + byteSize > inputSize || dstOffset + byteSize > remoteOutputSize,
                HCCL_ERROR("[%s] slice out of registered range. myRank=%u peer=%u "
                           "srcOff=%llu dstOff=%llu size=%llu inputSize=%llu remoteOutputSize=%llu",
                           tag, myRank, peerRank, srcOffset, dstOffset, byteSize, inputSize, remoteOutputSize),
                HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}
}

InsTempAlltoAllMesh2DV3NoMemcpy::InsTempAlltoAllMesh2DV3NoMemcpy(const OpParam &param, const u32 rankId,
                                                 const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempAlltoAllMesh2DV3NoMemcpy::~InsTempAlltoAllMesh2DV3NoMemcpy() {}

std::string InsTempAlltoAllMesh2DV3NoMemcpy::Describe() const
{
    std::string info = "Template of alltoall mesh 2D v2 with tempRankSize ";
    info += std::to_string(templateRankSize_);
    return info;
}

HcclResult InsTempAlltoAllMesh2DV3NoMemcpy::CalcRes(HcclComm comm, const OpParam &param,
                                            const TopoInfoWithNetLayerDetails *topoInfo,
                                            AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAlltoAllMesh2DV3NoMemcpy][CalcRes] start");
    GetRes(resourceRequest);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV3NoMemcpy::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllMesh2DV3NoMemcpy::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
}

u64 InsTempAlltoAllMesh2DV3NoMemcpy::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;

    if (opMode_ == OpMode::OPBASE) {
        return std::max(templateRankSize_, 1u);
    }
    return 0;
}

void InsTempAlltoAllMesh2DV3NoMemcpy::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAlltoAllMesh2DV3NoMemcpy::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult InsTempAlltoAllMesh2DV3NoMemcpy::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                              TemplateResource &templateResource)
{
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    slaveErrs_.clear();
    slaveErrs_.resize(templateResource.threads.size(), HCCL_SUCCESS);
    failedRanks_.assign(templateRankSize_, 0);

    // RAII PostSync guard: on ANY exit path (success, error, return),
    // this ensures PostSyncInterThreads is signaled to sub-threads
    // iff PreSync was previously called, preventing sub-thread hangs.
    // This is the guarantee required by design §10.1 (Fix 3).
    bool preSyncCalled = false;

    if (tempAlgParams.sliceSize == 0 && !IsAlltoAllVNoMemcpyParams(tempAlgParams, rankSize_)) {
        HCCL_INFO("[InsTempAlltoAllMesh2DV3NoMemcpy] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }

    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;

    dataType_ = param.all2AllVDataDes.sendType;
    HCCL_INFO("[InsTempAlltoAllMesh2DV3NoMemcpy] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);

    const bool isPcie = IsPcieProtocol(templateResource.channels);
    // if (isPcie) {
    //     // 远端读需要先把数据搬运到 hccl buffer内
    //     CHK_RET(LocalDataCopy(templateResource.threads));
    // }

    if (templateRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1,
                                             templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
        preSyncCalled = true;
    }

    CHK_RET(RunAlltoAllMesh(templateResource.threads, templateResource.channels));

    if (preSyncCalled) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1,
                                             templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
        preSyncCalled = false;
    }

    // if (!isPcie) {
    //     // 远端写 输出数据在 HCCL Buffer内 需要copy到 output
    //     CHK_RET(PostLocalCopy(templateResource.threads));
    // }

    HCCL_INFO("[InsTempAlltoAllMesh2DV3NoMemcpy][KernelRun] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV3NoMemcpy::RunAlltoAllMesh(
    const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    const u64 totalSliceSize = tempAlgParams_.sliceSize;

    const u64 actualChunkSize = (totalSliceSize + rankSize_ - 1) / rankSize_;
    const u64 chunkCount = actualChunkSize / dataTypeSize;
    const bool isAlltoAllV = IsAlltoAllVNoMemcpyParams(tempAlgParams_, rankSize_);

    HCCL_WARNING("[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] Stride config: "
        "inputSliceStride=%llu outputSliceStride=%llu inBuffType=%d outBuffType=%d "
        "inBuffBaseOff=%llu outBuffBaseOff=%llu outputSize=%llu actualChunkSize=%llu isAlltoAllV=%d",
        tempAlgParams_.inputSliceStride, tempAlgParams_.outputSliceStride,
        static_cast<int>(tempAlgParams_.buffInfo.inBuffType),
        static_cast<int>(tempAlgParams_.buffInfo.outBuffType),
        tempAlgParams_.buffInfo.inBuffBaseOff,
        tempAlgParams_.buffInfo.outBuffBaseOff,
        tempAlgParams_.buffInfo.outputSize,
        actualChunkSize, isAlltoAllV);

    const bool isPcie = IsPcieProtocol(channels);
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    HCCL_INFO("[InsTempAlltoAllMesh2DV3NoMemcpy][RunAlltoAllMesh] start. templateRankSize=%u isPcie=%d myAlgRank=%u "
              "actualChunkSize=%llu totalSlice=%llu",
              templateRankSize_, isPcie, myAlgRank, actualChunkSize, totalSliceSize);

    for (u32 neighborIdx = 0; neighborIdx < subCommRanks_[0].size() - 1; neighborIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + neighborIdx) % subCommRanks_[0].size()];
        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));

        if (failedRanks_[connectedAlgRank]) {
            HCCL_ERROR("[InsTempAlltoAllMesh2DV3NoMemcpy][RunAlltoAllMesh] peer[%u] algRank[%u] failed.",
                      connectedRank, connectedAlgRank);
            return HcclResult::HCCL_E_INTERNAL;
        }

        CHK_PRT_RET(
            neighborIdx >= threads.size() || channels.count(connectedRank) == 0 || channels.at(connectedRank).empty(),
            HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] Channel validation FAILED: "
                        "myRank=%u neighborIdx=%u threads=%zu connectedRank=%u channels.has=%d",
                        myRank_, neighborIdx, threads.size(), connectedRank,
                        channels.count(connectedRank)),
            HcclResult::HCCL_E_INTERNAL
        );

        const ChannelInfo &linkRemote = channels.at(connectedRank)[0];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
        if (!remoteCclBuffAddr) {
            HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] remoteCclMem.addr is NULL for peer %u. "
                       "myRank=%u connectedRank=%u templateRank=%u",
                       connectedRank, myRank_, connectedRank, templateRankSize_);
            return HCCL_E_INTERNAL;
        }

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;

        // tx 远端写
        void *txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
        u64 txSrcOffset = isAlltoAllV ? tempAlgParams_.sdispls[connectedRank] * dataTypeSize :
                                        tempAlgParams_.buffInfo.inBuffBaseOff + connectedRank * actualChunkSize;
        u64 txByteSize = isAlltoAllV ? tempAlgParams_.sendCounts[connectedRank] * dataTypeSize : actualChunkSize;
        u64 txCount = isAlltoAllV ? tempAlgParams_.sendCounts[connectedRank] : chunkCount;
        if (isAlltoAllV && txByteSize > 0) {
            CHK_PRT_RET(connectedRank >= tempAlgParams_.remoteRdispls.size() ||
                            connectedRank >= tempAlgParams_.remoteRecvCounts.size(),
                        HCCL_ERROR("[ALLTOALL_NO_MEMCPY][Mesh2D] missing remote exchange table. "
                                   "myRank=%u peer=%u remoteRdispls=%zu remoteRecvCounts=%zu",
                                   myRank_, connectedRank, tempAlgParams_.remoteRdispls.size(),
                                   tempAlgParams_.remoteRecvCounts.size()),
                        HcclResult::HCCL_E_INTERNAL);
            CHK_PRT_RET(tempAlgParams_.remoteRecvCounts[connectedRank] != txCount,
                        HCCL_ERROR("[ALLTOALL_NO_MEMCPY][Mesh2D] remote recv count mismatch. "
                                   "myRank=%u peer=%u localSend=%llu remoteRecvForLocal=%llu",
                                   myRank_, connectedRank, txCount,
                                   tempAlgParams_.remoteRecvCounts[connectedRank]),
                        HcclResult::HCCL_E_INTERNAL);
        }
        u64 txDstOffset = isAlltoAllV ? tempAlgParams_.remoteRdispls[connectedRank] * dataTypeSize :
                                        tempAlgParams_.buffInfo.outBuffBaseOff + myRank_ * actualChunkSize;
        u64 rxByteSize = isAlltoAllV ? tempAlgParams_.recvCounts[connectedRank] * dataTypeSize : actualChunkSize;
        u64 rxCount = isAlltoAllV ? tempAlgParams_.recvCounts[connectedRank] : chunkCount;
        if (txByteSize == 0 && rxByteSize == 0) {
            HCCL_INFO("[ALLTOALL_NO_MEMCPY][Mesh2D] skip zero send/recv. myRank=%u peer=%u",
                      myRank_, connectedRank);
            continue;
        }

        void *txDstPtr = linkRemote.remoteOutputGraphMode.addr;
        if (txByteSize > 0) {
            CHK_PRT_RET(!enableRemoteMemAccess_ || txDstPtr == nullptr,
                        HCCL_ERROR("[ALLTOALL_NO_MEMCPY][Mesh2D] remote output is unavailable. "
                                   "myRank=%u connectedRank=%u enableRemoteMemAccess=%d txSize=%llu rxSize=%llu",
                                   myRank_, connectedRank, enableRemoteMemAccess_, txByteSize, rxByteSize),
                        HcclResult::HCCL_E_INTERNAL);
            CHK_RET(CheckNoMemcpySliceRange("[ALLTOALL_NO_MEMCPY][Mesh2D]", myRank_, connectedRank, txSrcOffset,
                                            txDstOffset, txByteSize, tempAlgParams_.buffInfo.inputSize,
                                            linkRemote.remoteOutputGraphMode.size));
            txSrcSlicesAll.emplace_back(txSrcPtr, txSrcOffset, txByteSize, txCount);
            txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, txByteSize, txCount);
        }

        // no-memcpy mode only posts remote write. Keep rx placeholder identical
        // to tx so SendRecvWrite sees symmetric slice metadata in AllToAllV.
        void *rxSrcPtr = txByteSize > 0 ? linkRemote.remoteOutputGraphMode.addr : tempAlgParams_.buffInfo.outputPtr;
        u64 rxSrcOffset = txDstOffset;
        rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, rxByteSize, rxCount);

        void *rxDstPtr = txSrcPtr;
        u64 rxOutOffset = txSrcOffset;
        rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, rxByteSize, rxCount);

        HCCL_WARNING(
            "[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] rank[%d] peer[%d] "
            "txSrcOff=%llu txDstOff=%llu rxSrcOff=%llu rxDstOff=%llu "
            "txSize=%llu rxSize=%llu",
            myRank_, connectedRank,
            txSrcOffset, txDstOffset, rxSrcOffset, rxOutOffset,
            txByteSize, rxByteSize
        );

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll},
                                          {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);
        DataInfo sendInfo(linkRemote, {txSrcSlicesAll, txDstSlicesAll}, dataType_);
        DataInfo recvInfo(linkRemote, {rxSrcSlicesAll, rxDstSlicesAll}, dataType_);

        HCCL_WARNING("[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] round[%u/%zu] connectedRank=%u connectedAlgRank=%u "
                  "actualChunkSize=%llu txByteSize=%llu txCount=%llu isPcie=%d repeatNum=%u",
                  neighborIdx, subCommRanks_[0].size() - 1,
                  connectedRank, connectedAlgRank, actualChunkSize, txByteSize, txCount, isPcie,
                  tempAlgParams_.repeatNum);

        CHK_PRT_RET(isPcie,
                    HCCL_ERROR("[ALLTOALL_NO_MEMCPY][Mesh2D] pcie/read protocol is not supported."),
                    HcclResult::HCCL_E_NOT_SUPPORT);
        HcclResult dmaResult = HCCL_SUCCESS;
        if (isAlltoAllV || (txByteSize > 0 && rxByteSize > 0)) {
            dmaResult = SendRecvWrite(sendRecvInfo, threads[neighborIdx]);
        } else if (txByteSize > 0) {
            dmaResult = SendWrite(sendInfo, threads[neighborIdx]);
        } else {
            dmaResult = RecvWrite(recvInfo, threads[neighborIdx]);
        }

        if (dmaResult == HcclResult::HCCL_E_INTERNAL) {
            failedRanks_[connectedAlgRank] = 1;
            HCCL_WARNING("[ALLTOALL_V3_DEBUG][Mesh2D] Ring round %d: peer %u timed out. "
                         "templateRank=%u myRank=%u myAlgRank=%u",
                         neighborIdx, connectedRank, templateRankSize_, myRank_, myAlgRank);
            continue;
        }

        if (dmaResult != HCCL_SUCCESS) {
            HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D] RunAlltoAllMesh send/recv FAILED: "
                       "connectedRank=%u connectedAlgRank=%u round=%u err=0x%x",
                       connectedRank, connectedAlgRank, neighborIdx, dmaResult);
            return dmaResult;
        }
    }

    // 本地拷贝 输入 到 输出
    u64 inputOffset = isAlltoAllV ? tempAlgParams_.sdispls[myRank_] * dataTypeSize :
                                    tempAlgParams_.buffInfo.inBuffBaseOff + myRank_ * actualChunkSize;
    u64 outputOffsetBase = isAlltoAllV ? tempAlgParams_.rdispls[myRank_] * dataTypeSize :
                                         tempAlgParams_.buffInfo.outBuffBaseOff +  myRank_ * actualChunkSize;
    u64 localByteSize = isAlltoAllV ? tempAlgParams_.sendCounts[myRank_] * dataTypeSize : actualChunkSize;
    u64 localCount = isAlltoAllV ? tempAlgParams_.sendCounts[myRank_] : chunkCount;

    DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inputOffset, localByteSize, localCount);
    DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffsetBase, localByteSize, localCount);

    if (localByteSize > 0) {
        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    for (u32 i = 0; i < failedRanks_.size(); i++) {
        if (failedRanks_[i]) {
            HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][RunAlltoAllMesh] Failed rank[%u] detected. "
                       "templateRank=%u myRank=%u totalRounds=%zu",
                       i, templateRankSize_, myRank_, subCommRanks_[0].size() - 1);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV3NoMemcpy::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (rankSize_ == 0) {
        HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][LocalDataCopy] totalRankSize_ is 0.");
        return HCCL_E_INTERNAL;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSize + rankSize_ - 1) / rankSize_;
    u64 cellCount = cellSize / dataTypeSize;

    u32 myRank = myRank_;
    u32 meshDataOffset = (myRank / meshSize_) * meshSize_;

    u64 inputOffset = tempAlgParams_.buffInfo.inBuffBaseOff + cellSize * meshDataOffset;
    u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + cellSize * meshDataOffset;

    DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inputOffset, cellSize * meshSize_, cellCount * meshSize_);
    DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * meshSize_, cellCount * meshSize_);

    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllMesh2DV3NoMemcpy::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (rankSize_ == 0) {
        HCCL_ERROR("[ALLTOALL_V3_DEBUG][Mesh2D][LocalDataCopy] totalRankSize_ is 0.");
        return HCCL_E_INTERNAL;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalSize = tempAlgParams_.sliceSize;
    u64 cellSize = (totalSize + rankSize_ - 1) / rankSize_;
    u64 cellCount = cellSize / dataTypeSize;

    u32 myRank = myRank_;
    u32 meshDataOffset = (myRank / meshSize_) * meshSize_;

    u64 scratchOffset = tempAlgParams_.buffInfo.hcclBuffBaseOff + cellSize * meshDataOffset;
    u64 outputOffset = tempAlgParams_.buffInfo.outBuffBaseOff + cellSize * meshDataOffset;

    DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, cellSize * meshSize_, cellCount * meshSize_);
    DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outputOffset, cellSize * meshSize_, cellCount * meshSize_);

    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl
