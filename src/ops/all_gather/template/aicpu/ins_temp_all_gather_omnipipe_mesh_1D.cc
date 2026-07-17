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
    HCCL_INFO("[InsTempAllGatherOmniPipeMesh1D][KernelRun] start Mesh all-gather template, "
              "rank[%u], symmetric[%d].", myRank_, param.supportSymmetricMemory);
    if (templateRankSize_ == 1) {
        HCCL_INFO("[InsTempAllGatherOmniPipeMesh1D][KernelRun] skip communication for single-rank template, "
                  "rank[%u].", myRank_);
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
    HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][KernelRun] communication threads are ready, "
               "rank[%u], threadNum[%u].", myRank_, threadNum_);

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
    HCCL_INFO("[InsTempAllGatherOmniPipeMesh1D][KernelRun] Mesh all-gather template completed, rank[%u].",
              myRank_);
    return HcclResult::HCCL_SUCCESS;
}

// 普通路径在 ccl scratch 间通信；对称路径直接在本端与对端的 user output 窗口间通信。
HcclResult InsTempAllGatherOmniPipeMesh1D::RunAllGatherMesh(const std::vector<ThreadHandle>& threads,
                                                            const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    HCCL_INFO("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] start exchanging Mesh slices, rank[%u].",
              myRank_);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    for (u32 threadIdx = 0; threadIdx < subCommRanks_[0].size() - 1; threadIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + threadIdx) % subCommRanks_[0].size()];

        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
        HCCL_INFO("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] prepare peer slice exchange, "
                  "localRank[%u], remoteRank[%u], remoteAlgRank[%u].",
                  myRank_, connectedRank, connectedAlgRank);

        // 异常检查
        CHK_PRT_RET(threadIdx >= threads.size() || !channels.count(connectedRank),
                    HCCL_ERROR("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] thread or channel resource is "
                               "missing, localRank[%u], remoteRank[%u], threadIdx[%u], threadNum[%zu], "
                               "channelCount[%zu].", myRank_, connectedRank, threadIdx, threads.size(),
                               channels.size()),
                    HcclResult::HCCL_E_INTERNAL);

        ThreadHandle currQue = threads[threadIdx];

        std::vector<DataSlice> txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices;

        const ChannelInfo& linkRemote = channels.at(connectedRank)[0];
        void* remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
        void* remoteIn = nullptr;
        void* remoteOut = nullptr;
        // 通过对称窗口取得对端 input/output 的可访问地址。AG 数据面只使用 remoteOut；
        // remoteIn 用于确认对端输入窗口也已完成注册和交换。
        if (supportSymmetricMemory_) {
            HcclResult ret = HcclSymWinGetPeerPointer(inputSymWindow_, inputOffset_, connectedRank, &remoteIn);
            CHK_PRT_RET(ret != HCCL_SUCCESS || remoteIn == nullptr,
                        HCCL_ERROR("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] failed to get peer input "
                                   "pointer for window validation, remoteRank[%u], ret[%d], ptr[%p].",
                                   connectedRank, ret, remoteIn),
                            HcclResult::HCCL_E_INTERNAL);

            ret = HcclSymWinGetPeerPointer(outputSymWindow_, outputOffset_, connectedRank, &remoteOut);
            CHK_PRT_RET(ret != HCCL_SUCCESS || remoteOut == nullptr,
                        HCCL_ERROR("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] failed to get peer output "
                                   "pointer for data transfer, remoteRank[%u], ret[%d], ptr[%p].",
                                   connectedRank, ret, remoteOut),
                            HcclResult::HCCL_E_INTERNAL);
            HCCL_INFO("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] peer symmetric pointers are ready, "
                      "remoteRank[%u], inputPtr[%p], outputPtr[%p].", connectedRank, remoteIn, remoteOut);
        }
        // 对称路径将通信基地址切换到 user output：本端收发使用 outputPtr，
        // 对端收发使用 remoteOut 指向的对端 output 窗口。
        if (supportSymmetricMemory_) {
            void* txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
            void* txDstPtr = remoteOut;
            void* rxSrcPtr = remoteOut;
            void* rxDstPtr = tempAlgParams_.buffInfo.outputPtr;

            for (u32 rpt = 0; rpt < tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size(); ++rpt) {

                u64 txWriteSrcBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff +
                                tempAlgParams_.omniReadDstStepSliceInfo.inputOmniPipeSliceStride[myAlgRank][rpt];
                u64 rxReadDstBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff +
                                tempAlgParams_.omniReadDstStepSliceInfo.outputOmniPipeSliceStride[connectedAlgRank][rpt];
                u64 txWriteSrcOffset = tempAlgParams_.omniReadDstStepSliceInfo.stepInputSliceStride[myAlgRank] +
                                       txWriteSrcBaseOff + tempAlgParams_.processedDataCount * dataTypeSize;
                u64 rxReadDstOffset = tempAlgParams_.omniReadDstStepSliceInfo.stepOutputSliceStride[connectedAlgRank] +
                                      rxReadDstBaseOff + tempAlgParams_.processedDataCount * dataTypeSize;

                u64 rxDstCount = omniLastStepRead_ ?
                    tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[connectedAlgRank][rpt] :
                    tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt];
                const char* logTag = omniLastStepRead_ ? "last-step-read" : "symmetric-output";

                DataSlice txSrcSlice =
                    DataSlice(txSrcPtr, txWriteSrcOffset,
                            tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[myAlgRank][rpt],
                            tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt]);  // 本端发送源
                DataSlice txDstSlice =
                    DataSlice(txDstPtr, txWriteSrcOffset,
                            tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[myAlgRank][rpt],
                            tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt]);  // 对端发送目标
                // 读模式使用接收方向的切片。
                DataSlice rxDstSlice =
                    DataSlice(rxDstPtr, rxReadDstOffset,
                            tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                            rxDstCount);  // 本端接收目标
                DataSlice rxSrcSlice =
                    DataSlice(rxSrcPtr, rxReadDstOffset,
                            tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                            tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt]);  // 对端接收源

                rxSrcSlices.push_back(rxSrcSlice);
                rxDstSlices.push_back(rxDstSlice);
                txSrcSlices.push_back(txSrcSlice);
                txDstSlices.push_back(txDstSlice);

                HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] build send source slice, "
                        "mode[%s], localRank[%u], remoteRank[%u], offset[%llu], sliceSize[%llu], count[%llu].",
                        logTag, myRank_, connectedRank, txWriteSrcOffset,
                        tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[myAlgRank][rpt],
                        tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt]);

                HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] build send destination slice, "
                        "mode[%s], localRank[%u], remoteRank[%u], offset[%llu], sliceSize[%llu], count[%llu].",
                        logTag, myRank_, connectedRank, txWriteSrcOffset,
                        tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[myAlgRank][rpt],
                        tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt]);

                HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] build receive source slice, "
                        "mode[%s], localRank[%u], remoteRank[%u], offset[%llu], sliceSize[%llu], count[%llu].",
                        logTag, myRank_, connectedRank, rxReadDstOffset,
                        tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                        tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt]);

                HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] build receive destination slice, "
                        "mode[%s], localRank[%u], remoteRank[%u], offset[%llu], sliceSize[%llu], count[%llu].",
                        logTag, myRank_, connectedRank, rxReadDstOffset,
                        tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                        rxDstCount);
            }
        }
        else if (!supportSymmetricMemory_) {
            void* txSrcPtr;
            void* txDstPtr = remoteCclBuffAddr;
            void* rxSrcPtr = remoteCclBuffAddr;
            void* rxDstPtr;

            // 写模式使用发送方向的切片，接收方向仅使用通道完成同步。
            // 读模式使用接收方向的切片，发送方向仅使用通道完成同步。

            for (u32 rpt = 0; rpt < tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size(); ++rpt) {

                u64 txBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff +
                                tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank][rpt];
                u64 rxBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff +
                                tempAlgParams_.stepSliceInfo.outputOmniPipeSliceStride[connectedAlgRank][rpt];
                u64 txOffset = tempAlgParams_.stepSliceInfo.stepInputSliceStride[myAlgRank] + txBaseOff;
                u64 rxOffset = tempAlgParams_.stepSliceInfo.stepOutputSliceStride[connectedAlgRank] + rxBaseOff;

                if (!omniLastStepRead_) {

                    txSrcPtr = tempAlgParams_.buffInfo.hcclBuff.addr;
                    rxDstPtr = tempAlgParams_.buffInfo.hcclBuff.addr;

                    DataSlice txSrcSlice =
                        DataSlice(txSrcPtr, txOffset, tempAlgParams_.stepSliceInfo.stepSliceSize[myAlgRank][rpt],
                                tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt]);  // 本端发送源
                    DataSlice txDstSlice =
                        DataSlice(txDstPtr, txOffset, tempAlgParams_.stepSliceInfo.stepSliceSize[myAlgRank][rpt],
                                tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt]);  // 对端发送目标
                    // 读模式使用接收方向的切片。
                    DataSlice rxDstSlice =
                        DataSlice(rxDstPtr, rxOffset, tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                                tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt]);  // 本端接收目标
                    DataSlice rxSrcSlice =
                        DataSlice(rxSrcPtr, rxOffset, tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                                tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt]);  // 对端接收源

                    rxSrcSlices.push_back(rxSrcSlice);
                    rxDstSlices.push_back(rxDstSlice);
                    txSrcSlices.push_back(txSrcSlice);
                    txDstSlices.push_back(txDstSlice);

                    HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] build send source slice, "
                            "mode[scratch-write], localRank[%u], remoteRank[%u], offset[%llu], "
                            "sliceSize[%llu], count[%llu].",
                            myRank_, connectedRank, txOffset, tempAlgParams_.stepSliceInfo.stepSliceSize[myAlgRank][rpt],
                            tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt]);

                    HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] build send destination slice, "
                            "mode[scratch-write], localRank[%u], remoteRank[%u], offset[%llu], "
                            "sliceSize[%llu], count[%llu].",
                            myRank_, connectedRank, txOffset, tempAlgParams_.stepSliceInfo.stepSliceSize[myAlgRank][rpt],
                            tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt]);

                    HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] build receive source slice, "
                            "mode[scratch-write], localRank[%u], remoteRank[%u], offset[%llu], "
                            "sliceSize[%llu], count[%llu].",
                            myRank_, connectedRank, rxOffset,
                            tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                            tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt]);

                    HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] build receive destination slice, "
                            "mode[scratch-write], localRank[%u], remoteRank[%u], offset[%llu], "
                            "sliceSize[%llu], count[%llu].",
                            myRank_, connectedRank, rxOffset,
                            tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                            tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt]);
                }
                else {
                    txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
                    rxDstPtr = tempAlgParams_.buffInfo.outputPtr;

                    u64 txWriteSrcBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff +
                                    tempAlgParams_.omniReadDstStepSliceInfo.inputOmniPipeSliceStride[myAlgRank][rpt];
                    u64 rxReadDstBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff +
                                    tempAlgParams_.omniReadDstStepSliceInfo.outputOmniPipeSliceStride[connectedAlgRank][rpt];
                    u64 txWriteSrcOffset = tempAlgParams_.omniReadDstStepSliceInfo.stepInputSliceStride[myAlgRank] + txWriteSrcBaseOff + tempAlgParams_.processedDataCount*dataTypeSize;
                    u64 rxReadDstOffset = tempAlgParams_.omniReadDstStepSliceInfo.stepOutputSliceStride[connectedAlgRank] + rxReadDstBaseOff + tempAlgParams_.processedDataCount*dataTypeSize;

                    DataSlice txSrcSlice =
                        DataSlice(txSrcPtr, txWriteSrcBaseOff, tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[myAlgRank][rpt],
                                tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt]);  // 本端发送源
                    DataSlice txDstSlice =
                        DataSlice(txDstPtr, txOffset, tempAlgParams_.stepSliceInfo.stepSliceSize[myAlgRank][rpt],
                                tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt]);  // 对端发送目标
                    // 读模式使用接收方向的切片。
                    DataSlice rxDstSlice =
                        DataSlice(rxDstPtr, rxReadDstOffset, tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                                tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[connectedAlgRank][rpt]);  // 本端接收目标
                    DataSlice rxSrcSlice =
                        DataSlice(rxSrcPtr, rxOffset, tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                                tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt]);  // 对端接收源

                    rxSrcSlices.push_back(rxSrcSlice);
                    rxDstSlices.push_back(rxDstSlice);
                    txSrcSlices.push_back(txSrcSlice);
                    txDstSlices.push_back(txDstSlice);

                    HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] build send source slice, "
                            "mode[last-step-read], localRank[%u], remoteRank[%u], offset[%llu], "
                            "sliceSize[%llu], count[%llu].",
                            myRank_, connectedRank, txWriteSrcBaseOff, tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[myAlgRank][rpt],
                            tempAlgParams_.omniReadDstStepSliceInfo.stepCount[myAlgRank][rpt]);

                    HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] build send destination slice, "
                            "mode[last-step-read], localRank[%u], remoteRank[%u], offset[%llu], "
                            "sliceSize[%llu], count[%llu].",
                            myRank_, connectedRank, txOffset, tempAlgParams_.stepSliceInfo.stepSliceSize[myAlgRank][rpt],
                            tempAlgParams_.stepSliceInfo.stepCount[myAlgRank][rpt]);

                    HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] build receive source slice, "
                            "mode[last-step-read], localRank[%u], remoteRank[%u], offset[%llu], "
                            "sliceSize[%llu], count[%llu].",
                            myRank_, connectedRank, rxOffset,
                            tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                            tempAlgParams_.stepSliceInfo.stepSliceSize[connectedAlgRank][rpt]);

                    HCCL_DEBUG("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] build receive destination slice, "
                            "mode[last-step-read], localRank[%u], remoteRank[%u], offset[%llu], "
                            "sliceSize[%llu], count[%llu].",
                            myRank_, connectedRank, rxReadDstOffset,
                            tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[connectedAlgRank][rpt],
                            tempAlgParams_.omniReadDstStepSliceInfo.stepSliceSize[connectedAlgRank][rpt]);
                }
            }
        }
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        if (!omniLastStepRead_) {
            CHK_PRT_RET(SendRecvBatchWrite(sendRecvInfo, threads[threadIdx]),
                        HCCL_ERROR("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] batch write communication "
                                   "failed, localRank[%u], remoteRank[%u], threadIdx[%u].",
                                   myRank_, connectedRank, threadIdx),
                        HcclResult::HCCL_E_INTERNAL);
        }
        else {
            CHK_PRT_RET(SendRecvBatchRead(sendRecvInfo, threads[threadIdx]),
                        HCCL_ERROR("[InsTempAllGatherOmniPipeMesh1D][RunAllGatherMesh] last-step batch read "
                                   "communication failed, localRank[%u], remoteRank[%u], threadIdx[%u].",
                                   myRank_, connectedRank, threadIdx),
                        HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
}  // namespace ops_hccl
