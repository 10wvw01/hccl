/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_omnipipe_mesh_1D.h"
#include <sstream>
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "hccl_sym_win.h"
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */

namespace ops_hccl {
InsTempAllGatherOmniPipeMesh1D::InsTempAllGatherOmniPipeMesh1D(const OpParam& param,
                                                               const u32 rankId,  // 传通信域的rankId，userRank
                                                               const std::vector<std::vector<u32>>& subCommRanks)
    : InsTempAllGatherMesh1D(param, rankId, subCommRanks)
{
}
InsTempAllGatherOmniPipeMesh1D::~InsTempAllGatherOmniPipeMesh1D()
{
}

HcclResult InsTempAllGatherOmniPipeMesh1D::KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
                                                     TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherOmniPipeMesh1D] Run start");
    if (templateRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    tempAlgParams_.buffInfo.outputPtr = param.outputPtr;
    omniLastStepRead_ = tempAlgParams.omniLastStepRead_;
    dataType_ = param.DataDes.dataType;
    inputSymWindow_ = param.inputSymWindow;
    outputSymWindow_ = param.outputSymWindow;
    inputOffset_ = param.inputOffset;
    outputOffset_ = param.outputOffset;
    supportSymmetricMemory_ = param.supportSymmetricMemory;
    HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(RunAllGatherMesh(templateResource.threads, templateResource.channels));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    HCCL_INFO("[InsTempAllGatherOmniPipeMesh1D] Run End");
    return HcclResult::HCCL_SUCCESS;
}

// 当前仅支持strach->strach
HcclResult InsTempAllGatherOmniPipeMesh1D::RunAllGatherMesh(const std::vector<ThreadHandle>& threads,
                                                            const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    HCCL_INFO("[InsTempAllGatherOmniPipeMesh1D] RunAllGatherMesh RankIDs[%d].", myRank_);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    for (u32 threadIdx = 0; threadIdx < subCommRanks_[0].size() - 1; threadIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + threadIdx) % subCommRanks_[0].size()];

        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
        HCCL_INFO("[InsTempAllGatherOmniPipeMesh1D] RunAllGatherMesh RankIDs[%d], connectedRank[%d], "
                    "connectedAlgRank[%d].",
                    myRank_, connectedRank, connectedAlgRank);

        // 异常检查
        CHK_PRT_RET(threadIdx >= threads.size() || !channels.count(connectedRank),
                    HCCL_ERROR("[InsTempAllGatherOmniPipeMesh1D][RankID]=%u threadIdx=%u, threads.size=%u, "
                                "connectedRank=%d, channels.size=%u",
                                myRank_, threadIdx, threads.size(), connectedRank, channels.size()),
                    HcclResult::HCCL_E_INTERNAL);

        ThreadHandle currQue = threads[threadIdx];

        std::vector<DataSlice> txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices;

        const ChannelInfo& linkRemote = channels.at(connectedRank)[0];
        void* remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
        void* remoteIn = nullptr;
        void* remoteOut = nullptr;
        // 对称内存：通过 SymWindow 直接映射对端用户 input/output 缓冲区（input 仅用于校验对称映射可用，
        // 传输实际使用 output）；非对称内存：使用对端 CCL 通信缓冲区。
        if (supportSymmetricMemory_) {
            HcclResult ret = HcclSymWinGetPeerPointer(inputSymWindow_, inputOffset_, connectedRank, &remoteIn);
            CHK_PRT_RET(ret != HCCL_SUCCESS || remoteIn == nullptr,
                        HCCL_ERROR("[InsTempAllGatherOmniSymmetryMemoryMesh1D] HcclSymWinGetPeerPointer failed, "
                            "remoteRank[%u] inputRet[%d] in[%p]", connectedRank, ret, remoteIn),
                            HcclResult::HCCL_E_INTERNAL);

            ret = HcclSymWinGetPeerPointer(outputSymWindow_, outputOffset_, connectedRank, &remoteOut);
            CHK_PRT_RET(ret != HCCL_SUCCESS || remoteOut == nullptr,
                        HCCL_ERROR("[InsTempAllGatherOmniSymmetryMemoryMesh1D] HcclSymWinGetPeerPointer failed, "
                            "remoteRank[%u] outputRet[%d] out[%p]", connectedRank, ret, remoteOut),
                            HcclResult::HCCL_E_INTERNAL);
            HCCL_INFO("[InsTempAllGatherOmniSymmetryMemoryMesh1D] HcclSymWinGetPeerPointer success, "
                "remoteRank[%u] in[%p] out[%p]", connectedRank, remoteIn, remoteOut);
        }

        // 远端指针：对称模式直接访问对端 user output；非对称模式访问对端 CCL buff
        void* remotePeerPtr = supportSymmetricMemory_ ? remoteOut : remoteCclBuffAddr;
        // 本地指针：仅"非对称 + write"走 CCL buff，其余均直读直写 user output
        void* localPtr = (!supportSymmetricMemory_ && !omniLastStepRead_)
                             ? tempAlgParams_.buffInfo.hcclBuff.addr
                             : tempAlgParams_.buffInfo.outputPtr;
        const char* mode = omniLastStepRead_ ? "read" : "write";

        for (u32 rpt = 0; rpt < tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size(); ++rpt) {
            // stepSliceInfo 基础偏移：非对称 write 全程使用，其余分支的 txDst/rxSrc 也复用
            u64 txBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff +
                            tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank][rpt];
            u64 rxBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff +
                            tempAlgParams_.stepSliceInfo.outputOmniPipeSliceStride[connectedAlgRank][rpt];
            u64 txOffset = tempAlgParams_.stepSliceInfo.stepInputSliceStride[myAlgRank] + txBaseOff;
            u64 rxOffset = tempAlgParams_.stepSliceInfo.stepOutputSliceStride[connectedAlgRank] + rxBaseOff;

            // 本轮四个 slice 的 (offset, sliceSize, count)
            u64 txSrcOff, txSrcSize, txSrcCount;
            u64 txDstOff, txDstSize, txDstCount;
            u64 rxDstOff, rxDstSize, rxDstCount;
            u64 rxSrcOff, rxSrcSize, rxSrcCount;

            if (supportSymmetricMemory_) {
                // 对称模式：tx/rx 直读直写 user output，偏移基于 omniReadDstStepSliceInfo 并叠加 processedDataCount
                u64 txWriteSrcBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff +
                                tempAlgParams_.omniReadDstStepSliceInfo.inputOmniPipeSliceStride[myAlgRank][rpt];
                u64 rxReadDstBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff +
                                tempAlgParams_.omniReadDstStepSliceInfo.outputOmniPipeSliceStride[connectedAlgRank][rpt];
                u64 txWriteSrcOffset = tempAlgParams_.omniReadDstStepSliceInfo.stepInputSliceStride[myAlgRank] +
                                       txWriteSrcBaseOff + tempAlgParams_.processedDataCount * dataTypeSize;
                u64 rxReadDstOffset = tempAlgParams_.omniReadDstStepSliceInfo.stepOutputSliceStride[connectedAlgRank] +
                                      rxReadDstBaseOff + tempAlgParams_.processedDataCount * dataTypeSize;

                txSrcOff = txDstOff = txWriteSrcOffset;
                txSrcSize = txDstSize = tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[myAlgRank][rpt];
                txSrcCount = txDstCount = tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt];

                rxSrcOff = rxDstOff = rxReadDstOffset;
                rxSrcSize = rxDstSize = tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[connectedAlgRank][rpt];
                rxSrcCount = tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt];
                // read 模式下本地接收量取 omniReadDst 步进量，write 模式取 step 步进量
                rxDstCount = omniLastStepRead_
                                 ? tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[connectedAlgRank][rpt]
                                 : tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt];
            } else if (!omniLastStepRead_) {
                // 非对称 write：数据在 CCL buff 间环形汇聚，全部基于 stepSliceInfo
                txSrcOff = txDstOff = txOffset;
                txSrcSize = txDstSize = tempAlgParams_.stepSliceInfo.stepSliceSize[myAlgRank][rpt];
                txSrcCount = txDstCount = tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt];

                rxDstOff = rxSrcOff = rxOffset;
                rxDstSize = rxSrcSize = tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt];
                rxDstCount = rxSrcCount = tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt];
            } else {
                // 非对称 read：最后一步从对端 CCL buff 读出写回 user output
                u64 txWriteSrcBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff +
                                tempAlgParams_.omniReadDstStepSliceInfo.inputOmniPipeSliceStride[myAlgRank][rpt];
                u64 rxReadDstBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff +
                                tempAlgParams_.omniReadDstStepSliceInfo.outputOmniPipeSliceStride[connectedAlgRank][rpt];
                u64 rxReadDstOffset = tempAlgParams_.omniReadDstStepSliceInfo.stepOutputSliceStride[connectedAlgRank] +
                                      rxReadDstBaseOff + tempAlgParams_.processedDataCount * dataTypeSize;

                txSrcOff = txWriteSrcBaseOff;  // 本地源用 baseOff
                txSrcSize = tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[myAlgRank][rpt];
                txSrcCount = tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt];
                txDstOff = txOffset;           // 远端目标用 txOffset
                txDstSize = tempAlgParams_.stepSliceInfo.stepSliceSize[myAlgRank][rpt];
                txDstCount = tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt];
                rxDstOff = rxReadDstOffset;
                rxDstSize = tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[connectedAlgRank][rpt];
                rxDstCount = tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[connectedAlgRank][rpt];
                rxSrcOff = rxOffset;
                rxSrcSize = tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt];
                rxSrcCount = tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt];
            }

            txSrcSlices.emplace_back(localPtr, txSrcOff, txSrcSize, txSrcCount);       // 本地(send)
            txDstSlices.emplace_back(remotePeerPtr, txDstOff, txDstSize, txDstCount);  // 远程(send)
            rxDstSlices.emplace_back(localPtr, rxDstOff, rxDstSize, rxDstCount);       // 本地(recv)
            rxSrcSlices.emplace_back(remotePeerPtr, rxSrcOff, rxSrcSize, rxSrcCount);  // 远程(recv)

            HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][%s] rankId[%d] connectedRank[%d] rpt[%u] "
                    "txSrc[off=%llu,sz=%llu,cnt=%llu] txDst[off=%llu,sz=%llu,cnt=%llu] "
                    "rxDst[off=%llu,sz=%llu,cnt=%llu] rxSrc[off=%llu,sz=%llu,cnt=%llu].",
                    mode, myRank_, connectedRank, rpt,
                    txSrcOff, txSrcSize, txSrcCount, txDstOff, txDstSize, txDstCount,
                    rxDstOff, rxDstSize, rxDstCount, rxSrcOff, rxSrcSize, rxSrcCount);
        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        if (!omniLastStepRead_) {
            CHK_PRT_RET(SendRecvBatchWrite(sendRecvInfo, threads[threadIdx]),
                        HCCL_ERROR("[InsTempAllGatherOmniPipeMesh1D] RunAllGather Send failed"),
                        HcclResult::HCCL_E_INTERNAL);
        }
        else {
            CHK_PRT_RET(SendRecvBatchRead(sendRecvInfo, threads[threadIdx]),
                        HCCL_ERROR("[InsTempAllGatherOmniPipeMesh1D]omniLastStepRead_ RunAllGather SendRecvBatchRead failed"),
                        HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
}  // namespace ops_hccl
