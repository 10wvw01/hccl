/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "hccl_sym_win.h"
#include <iomanip>
#include <sstream>
namespace ops_hccl {

static void PrintSliceData1D(const char *tag, const void *baseAddr, u64 offset, u64 byteSize,
    HcclDataType dataType)
{
    if (baseAddr == nullptr || byteSize == 0) {
        HCCL_INFO("[%s] addr NULL or size 0, skip", tag);
        return;
    }
    u32 typeSize = DATATYPE_SIZE_TABLE[dataType];
    u64 elemCount = byteSize / typeSize;
    if (elemCount == 0) {
        HCCL_INFO("[%s] elemCount 0, skip", tag);
        return;
    }
    u64 printCount = std::min(elemCount, static_cast<u64>(256));
    const u8 *addr = static_cast<const u8 *>(baseAddr) + offset;

    std::stringstream ss;
    ss << "[" << tag << "] offset[" << offset << "] bytes[" << byteSize << "] elems[" << elemCount << "]";

    if (dataType == HcclDataType::HCCL_DATA_TYPE_FP32) {
        const float *ptr = reinterpret_cast<const float *>(addr);
        ss << " fp32: [";
        for (u64 i = 0; i < printCount; i++) {
            if (i > 0) ss << ", ";
            ss << std::setprecision(9) << ptr[i];
        }
        ss << "]";
    } else if (dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        const double *ptr = reinterpret_cast<const double *>(addr);
        ss << " fp64: [";
        for (u64 i = 0; i < printCount; i++) {
            if (i > 0) ss << ", ";
            ss << std::setprecision(17) << ptr[i];
        }
        ss << "]";
    } else if (dataType == HcclDataType::HCCL_DATA_TYPE_INT32) {
        const int32_t *ptr = reinterpret_cast<const int32_t *>(addr);
        ss << " int32: [";
        for (u64 i = 0; i < printCount; i++) {
            if (i > 0) ss << ", ";
            ss << ptr[i];
        }
        ss << "]";
    } else {
        ss << " raw_hex: [";
        for (u64 i = 0; i < std::min(byteSize, static_cast<u64>(64)); i++) {
            if (i > 0) ss << " ";
            ss << std::hex << static_cast<unsigned>(addr[i]) << std::dec;
        }
        ss << "]";
    }
    HCCL_INFO("%s", ss.str().c_str());
}
InsTempAllGatherMesh1D::InsTempAllGatherMesh1D(const OpParam &param, const u32 rankId,
                                               const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}
InsTempAllGatherMesh1D::~InsTempAllGatherMesh1D() {}

HcclResult InsTempAllGatherMesh1D::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                           AlgResourceRequest &resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherMesh1D][CalcRes] start");
    GetRes(resourceRequest);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}
HcclResult InsTempAllGatherMesh1D::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherMesh1D::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
}

u64 InsTempAllGatherMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 0;
    if (opMode_ == OpMode::OPBASE){
        scratchMultiple = templateRankSize_;
    }
    return scratchMultiple;
}

HcclResult InsTempAllGatherMesh1D::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                             TemplateResource &templateResource)
{
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    HCCL_INFO("[InsTempAllGatherMesh1D] Run start");
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize ==0) {
        HCCL_INFO("[InsTempAllGatherMesh1D] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    inputSymWindow_ = param.inputSymWindow;
    outputSymWindow_ = param.outputSymWindow;
    inputOffset_ = param.inputOffset;
    outputOffset_ = param.outputOffset;
    supportSymmetricMemory_ = param.supportSymmetricMemory;
    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;

    // Step1: LocalDataCopy - 打印输入数据，执行后同步再打印
    HCCL_INFO("[AG_Mesh1D] >>> Step1: LocalDataCopy");
    PrintSliceData1D("AG_Step1_BEFORE_input", tempAlgParams.buffInfo.inputPtr,
        tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.inputSize, dataType_);
    // 逐个rank slice打印输入
    {
        u32 myAlgRank = 0;
        CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
        for (u32 i = 0; i < subCommRanks_[0].size(); i++) {
            u32 algRank = 0;
            CHK_RET(GetAlgRank(subCommRanks_[0][i], subCommRanks_[0], algRank));
            u64 sliceSize = tempAlgParams.sliceSize;
            if (tempAlgParams.tailSize != 0 && algRank == templateRankSize_ - 1) {
                sliceSize = tempAlgParams.tailSize;
            }
            u64 sliceOffset = tempAlgParams.outputSliceStride * algRank;
            if (sliceSize > 0) {
                PrintSliceData1D("AG_Step1_BEFORE_input_rankSlice", tempAlgParams.buffInfo.inputPtr,
                    sliceOffset + tempAlgParams.buffInfo.inBuffBaseOff, sliceSize, dataType_);
            }
        }
    }
    CHK_RET(LocalDataCopy(templateResource.threads));
    HCCL_INFO("[AG_Mesh1D] Step1: LocalDataCopy submitted (async)");

    // DEBUG SYNC: 等待LocalDataCopy DMA完成
    HCCL_INFO("[AG_Mesh1D] DEBUG_SYNC: waiting for LocalDataCopy DMA completion...");
    CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
    CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
    for (const auto &thread : templateResource.threads) {
        CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
    }
    HCCL_INFO("[AG_Mesh1D] DEBUG_SYNC: LocalDataCopy DMA complete");
    // LocalDataCopy后：打印output中本rank的slice + scratch中的数据
    {
        u32 myAlgRank = 0;
        CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
        u64 mySliceSize = tempAlgParams.sliceSize;
        if (tempAlgParams.tailSize != 0 && myAlgRank == templateRankSize_ - 1) {
            mySliceSize = tempAlgParams.tailSize;
        }
        u64 mySliceOffset = tempAlgParams.outputSliceStride * myAlgRank;
        PrintSliceData1D("AG_Step1_AFTER_output_mySlice", tempAlgParams.buffInfo.outputPtr,
            mySliceOffset + tempAlgParams.buffInfo.outBuffBaseOff, mySliceSize, dataType_);
        // 打印scratch中的本rank区域
        if (!enableRemoteMemAccess_) {
            u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
            u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff;
            u64 myScratchOff = scratchBase + tempAlgParams.sliceSize * myAlgRank;
            PrintSliceData1D("AG_Step1_AFTER_scratch_myRank", tempAlgParams.buffInfo.hcclBuff.addr,
                myScratchOff, mySliceSize, dataType_);
        }
    }

    if (templateRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    // Step2: RunAllGatherMesh - 异步通信，同步后再打印
    HCCL_INFO("[AG_Mesh1D] >>> Step2: RunAllGatherMesh");
    CHK_RET(RunAllGatherMesh(templateResource.threads, templateResource.channels));
    HCCL_INFO("[AG_Mesh1D] Step2: AllGatherMesh submitted (async)");

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    // DEBUG SYNC: 等待AllGatherMesh通信完成
    HCCL_INFO("[AG_Mesh1D] DEBUG_SYNC: waiting for AllGatherMesh completion...");
    CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
    CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
    for (const auto &thread : templateResource.threads) {
        CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
    }
    HCCL_INFO("[AG_Mesh1D] DEBUG_SYNC: AllGatherMesh complete, all rank data now visible");
    // AllGatherMesh后：打印scratch + output中每个rank的slice
    {
        for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
            u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
            u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
            u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
            if (!enableRemoteMemAccess_) {
                PrintSliceData1D("AG_Step2_AFTER_cclBuff_scratch", tempAlgParams_.buffInfo.hcclBuff.addr,
                    scratchBase, scratchRepeatStride, dataType_);
            }
            for (u32 i = 0; i < subCommRanks_[0].size(); i++) {
                u32 algRank = 0;
                CHK_RET(GetAlgRank(subCommRanks_[0][i], subCommRanks_[0], algRank));
                u64 sliceSize = tempAlgParams_.sliceSize;
                if (tempAlgParams_.tailSize != 0 && algRank == templateRankSize_ - 1) {
                    sliceSize = tempAlgParams_.tailSize;
                }
                u64 sliceOffset = tempAlgParams_.outputSliceStride * algRank + outBaseOff;
                if (sliceSize > 0) {
                    PrintSliceData1D("AG_Step2_AFTER_output_rankSlice", tempAlgParams.buffInfo.outputPtr,
                        sliceOffset, sliceSize, dataType_);
                }
            }
        }
    }

    // Step3: PostLocalCopy - 异步DMA，同步后再打印
    HCCL_INFO("[AG_Mesh1D] >>> Step3: PostLocalCopy");
    CHK_RET(PostLocalCopy(templateResource.threads));
    HCCL_INFO("[AG_Mesh1D] Step3: PostLocalCopy submitted (async)");

    // DEBUG SYNC: 等待PostLocalCopy DMA完成
    HCCL_INFO("[AG_Mesh1D] DEBUG_SYNC: waiting for PostLocalCopy DMA completion...");
    CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
    CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
    for (const auto &thread : templateResource.threads) {
        CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
    }
    HCCL_INFO("[AG_Mesh1D] DEBUG_SYNC: PostLocalCopy DMA complete, final output now visible");
    // PostLocalCopy后：打印最终output（每个rank的slice + 全量）
    {
        for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
            u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
            for (u32 i = 0; i < subCommRanks_[0].size(); i++) {
                u32 algRank = 0;
                CHK_RET(GetAlgRank(subCommRanks_[0][i], subCommRanks_[0], algRank));
                u64 sliceSize = tempAlgParams_.sliceSize;
                if (tempAlgParams_.tailSize != 0 && algRank == templateRankSize_ - 1) {
                    sliceSize = tempAlgParams_.tailSize;
                }
                u64 sliceOffset = tempAlgParams_.outputSliceStride * algRank + outBaseOff;
                if (sliceSize > 0) {
                    PrintSliceData1D("AG_Step3_AFTER_output_rankSlice", tempAlgParams.buffInfo.outputPtr,
                        sliceOffset, sliceSize, dataType_);
                }
            }
        }
        PrintSliceData1D("AG_Step3_AFTER_output_full", tempAlgParams.buffInfo.outputPtr,
            tempAlgParams.buffInfo.outBuffBaseOff, tempAlgParams.buffInfo.outputSize, dataType_);
    }
    HCCL_INFO("[InsTempAllGatherMesh1D] Run End");
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::RunAllGatherMesh(const std::vector<ThreadHandle> &threads,
                                                    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherMesh1D] RunAllGatherMesh RankIDs[%d].", myRank_);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    for (u32 threadIdx = 0; threadIdx < subCommRanks_[0].size() - 1; threadIdx++) {
        u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + threadIdx) % subCommRanks_[0].size()];

        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
        HCCL_INFO("[InsTempAllGatherMesh1D] RunAllGatherMesh RankIDs[%d], connectedRank[%d], connectedAlgRank[%d].",
                    myRank_, connectedRank, connectedAlgRank);

        CHK_PRT_RET(threadIdx >= threads.size() || channels.count(connectedRank) == 0 ||
                    channels.at(connectedRank).empty(),
                    HCCL_ERROR("[InsTempAllGatherMesh1D][RankID]=%u threadIdx=%u, threads.size=%u, "
                                "connectedRank=%d, channels.size=%u",
                                myRank_, threadIdx, threads.size(), connectedRank, channels.size()),
                    HcclResult::HCCL_E_INTERNAL);

        const ChannelInfo &linkRemote = channels.at(connectedRank)[0];
        void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
        
        // 对称内存下，远端地址需要通过HcclSymWinGetPeerPointer获取
        void *remoteIn = nullptr;
        void *remoteOut = nullptr;
        if (supportSymmetricMemory_) {
            HcclResult ret = HcclSymWinGetPeerPointer(inputSymWindow_, inputOffset_, connectedRank, &remoteIn);
            CHK_PRT_RET(ret != HCCL_SUCCESS || remoteIn == nullptr,
                        HCCL_ERROR("[InsTempAllGatherSymmetryMemoryMesh1D] HcclSymWinGetPeerPointer failed, "
                            "remoteRank[%u] inputRet[%d] in[%p]", connectedRank, ret, remoteIn),
                            HcclResult::HCCL_E_INTERNAL);

            ret = HcclSymWinGetPeerPointer(outputSymWindow_, outputOffset_, connectedRank, &remoteOut);
            CHK_PRT_RET(ret != HCCL_SUCCESS || remoteOut == nullptr,
                        HCCL_ERROR("[InsTempAllGatherSymmetryMemoryMesh1D] HcclSymWinGetPeerPointer failed, "
                            "remoteRank[%u] outputRet[%d] out[%p]", connectedRank, ret, remoteOut),
                            HcclResult::HCCL_E_INTERNAL);
            HCCL_INFO("[InsTempAllGatherSymmetryMemoryMesh1D] HcclSymWinGetPeerPointer success, "
                "remoteRank[%u] in[%p] out[%p]", connectedRank, remoteIn, remoteOut);
        }

        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;

        for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
            const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
            const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
            const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

            u64 sliceSize = tempAlgParams_.sliceSize;
            if (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1) {
                sliceSize = tempAlgParams_.tailSize;
            }
            u64 txOutOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;
            u64 rxOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff;
            u64 txDstOffset = 0;
            u64 rxSrcOffset = 0;
            void *txDstPtr = nullptr;
            void *rxSrcPtr = nullptr;
            void *txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
            void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;

            if (!supportSymmetricMemory_) {
                u64 txScratchOffset = scratchBase + tempAlgParams_.sliceSize * myAlgRank;
                txDstOffset = (!enableRemoteMemAccess_) ? txScratchOffset : txOutOffset;
                u64 rxScratchOffset = scratchBase + tempAlgParams_.sliceSize * connectedAlgRank;
                rxSrcOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;
                txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
                rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
            } else {
                txDstOffset = txOutOffset;
                rxSrcOffset = rxOutOffset;
                txDstPtr = remoteOut;
                rxSrcPtr = remoteOut;
            }
            u64 sliceCount = sliceSize / dataTypeSize;

            txSrcSlicesAll.emplace_back(txSrcPtr, txOutOffset, sliceSize, sliceCount);
            txDstSlicesAll.emplace_back(txDstPtr, txDstOffset, sliceSize, sliceCount);
            rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sliceSize, sliceCount);
            rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sliceSize, sliceCount);

            HCCL_DEBUG("[InsTempAllGatherMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] txSrcSlices: "
                        "offset[%d] sliceSize[%d] count[%d].",
                        myRank_, connectedRank, rpt, txOutOffset, sliceSize, sliceCount);

            HCCL_DEBUG("[InsTempAllGatherMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] txDstSlices: "
                        "offset[%d] sliceSize[%d] count[%d].",
                        myRank_, connectedRank, rpt, txDstOffset, sliceSize, sliceCount);

            HCCL_DEBUG("[InsTempAllGatherMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] rxSrcSlices: "
                        "offset[%d] sliceSize[%d] count[%d].",
                        myRank_, connectedRank, rpt, rxOutOffset, sliceSize, sliceCount);

            HCCL_DEBUG("[InsTempAllGatherMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] rxDrcSlices: "
                        "offset[%d] sliceSize[%d] count[%d].",
                        myRank_, connectedRank, rpt, rxSrcOffset, sliceSize, sliceCount);
        }

        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(linkRemote, linkRemote);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
        CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[threadIdx]),
                    HCCL_ERROR("[InsTempAllGatherMesh1D] RunAllGather Send failed"), HcclResult::HCCL_E_INTERNAL);
        }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1D] LocalDataCopy.");
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 sliceSize = tempAlgParams_.sliceSize;

    if (tempAlgParams_.tailSize !=0 && myAlgRank == templateRankSize_ -1) {
        sliceSize = tempAlgParams_.tailSize;
    }

    u64 sliceCount = sliceSize / dataTypeSize;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff;
        const u64 outOff = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        bool skipOutCopy = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr && inOff == outOff);

        if (!skipOutCopy) {
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, sliceSize, sliceCount);
            HCCL_DEBUG("[InsTempAllGatherMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] srcSlice: inBaseOff[%llu] inOff[%llu] "
                       "sliceSize[%llu] count[%llu].",
                       myRank_, myAlgRank, inBaseOff, inOff, sliceSize, sliceCount);
            HCCL_DEBUG("[InsTempAllGatherMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] dstSlice: outBaseoff[%llu] "
                       "outOff[%llu] sliceSize[%llu] count[%llu].",
                       myRank_, myAlgRank, outBaseOff, outOff, sliceSize, sliceCount);
            LocalCopy(threads[0], srcSlice, dstSlice);
        }

        if (!enableRemoteMemAccess_ && !supportSymmetricMemory_) {
            const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
            const u64 cclBaseOff = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
            u64 cclOff = cclBaseOff + tempAlgParams_.sliceSize * myAlgRank;
            DataSlice cclDstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, cclOff, sliceSize, sliceCount);
            bool skipCclCopy = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr &&
                                inOff == cclOff);
            if (!skipCclCopy) {
                HCCL_DEBUG("[InsTempAllGatherMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] copy to ccl: "
                        "cclBaseOff[%llu] cclOff[%llu] sliceSize[%llu] count[%llu].",
                        myRank_, myAlgRank, cclBaseOff, cclOff, sliceSize, sliceCount);
                LocalCopy(threads[0], srcSlice, cclDstSlice);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1D] PostLocalCopy.");
    if (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAllGatherMesh1D] PostLocalCopy skip because output is scratch" );
        return HcclResult::HCCL_SUCCESS;
    }
    if (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAllGatherMesh1D] PostLocalCopy skip because input is scratch and should be read to output" );
        return HcclResult::HCCL_SUCCESS;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 sliceSize = tempAlgParams_.sliceSize;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            if (rank == myRank_) {
                continue;
            }
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            // 尾块模式
            if (tempAlgParams_.tailSize !=0 && algRank == templateRankSize_ -1) {
                sliceSize = tempAlgParams_.tailSize;
            }
            u64 scratchOffset = tempAlgParams_.sliceSize * algRank + scratchBase;
            u64 outOffset = tempAlgParams_.outputSliceStride * algRank + outBaseOff;
            u64 sliceCount = sliceSize / dataTypeSize;
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, sliceSize, sliceCount);
            HCCL_DEBUG("[InsTempAllGatherMesh1D] LocalDataCopy RankID [%d] dataRank [%d] dataAlgRank[%d] "
                       "scratchBase[%d] outBaseOff[%d] scratchOffset[%d] outOffset[%d].",
                       myRank_, rank, algRank, outBaseOff, outBaseOff, scratchOffset, outOffset);
            LocalCopy(threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherMesh1D::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAllGatherMesh1D::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

}  // namespace ops_hccl