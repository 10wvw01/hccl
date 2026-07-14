/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_omnipipe_nhr.h"
#include "omnipipe_template_utils.h"
#include "exec_timeout_manager.h"
#include "hcomm_primitives_dl.h"
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "hccl_sym_win.h"
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
constexpr u32 SMALL_COUNT_512KB = 512 * 1024;
namespace ops_hccl {
namespace {
HcclResult CalcDirectUserInputOffset(u64 packedOffset, u64 sliceSize, u64 loopBaseOffset,
                                     u64 rankStride, u64 loopSize, u64 inputSize, u64& userInputOffset)
{
    CHK_PRT_RET(rankStride == 0 || loopSize == 0 || loopBaseOffset > inputSize,
                HCCL_ERROR("[%s] invalid input layout, loopBase[%llu] rankStride[%llu] loopSize[%llu] "
                           "inputSize[%llu]", __func__, loopBaseOffset, rankStride, loopSize, inputSize),
                HCCL_E_PARA);
    const u64 rankIndex = packedOffset / loopSize;
    const u64 offsetInLoop = packedOffset % loopSize;
    CHK_PRT_RET(sliceSize > loopSize - offsetInLoop ||
                rankIndex > (inputSize - loopBaseOffset) / rankStride,
                HCCL_ERROR("[%s] invalid packed input slice, packedOffset[%llu] sliceSize[%llu]", __func__,
                           packedOffset, sliceSize),
                HCCL_E_PARA);
    const u64 rankBaseOffset = loopBaseOffset + rankIndex * rankStride;
    CHK_PRT_RET(offsetInLoop > inputSize - rankBaseOffset,
                HCCL_ERROR("[%s] input offset exceeds input buffer, rankBase[%llu] offsetInLoop[%llu]",
                           __func__, rankBaseOffset, offsetInLoop),
                HCCL_E_PARA);
    userInputOffset = rankBaseOffset + offsetInLoop;
    CHK_PRT_RET(sliceSize > inputSize - userInputOffset,
                HCCL_ERROR("[%s] input slice exceeds input buffer, offset[%llu] sliceSize[%llu] inputSize[%llu]",
                           __func__, userInputOffset, sliceSize, inputSize),
                HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult DirectWriteReduceWithStepSync(const ChannelInfo& sendChannel, const ChannelInfo& recvChannel,
                                         const std::vector<DataSlice>& srcSlices,
                                         const std::vector<DataSlice>& dstSlices, HcclDataType dataType,
                                         HcclReduceOp reduceOp, const ThreadHandle& thread, bool needReadySync)
{
    CHK_PRT_RET(srcSlices.size() != dstSlices.size(),
                HCCL_ERROR("[%s] src slice num[%zu] does not match dst slice num[%zu]", __func__,
                           srcSlices.size(), dstSlices.size()),
                HCCL_E_PARA);

    const u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    // Every rank's original input is ready at step 0. Later steps consume partial sums produced by the
    // preceding step, so only those steps need the READY handshake.
    if (needReadySync) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
        CHK_RET(HcclChannelNotifyWaitOnThreadDefault(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout));
    }

    const bool useBatchTransfer = IsHcommBatchTransferOnThreadSupported();
    std::vector<HcclHcommBatchTransferDesc> transferDescs;
    if (useBatchTransfer) {
        transferDescs.reserve(srcSlices.size());
    }
    const u32 sliceNum = static_cast<u32>(srcSlices.size());
    for (u32 i = 0; i < sliceNum; ++i) {
        const DataSlice& srcSlice = srcSlices[i];
        const DataSlice& dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) {
            continue;
        }
        CHK_PRT_RET(srcSlice.size_ != dstSlice.size_ ||
                    srcSlice.count_ * DATATYPE_SIZE_TABLE[dataType] != srcSlice.size_ ||
                    dstSlice.count_ * DATATYPE_SIZE_TABLE[dataType] != dstSlice.size_,
                    HCCL_ERROR("[%s] invalid slice[%u], srcSize[%llu] srcCount[%llu] dstSize[%llu] "
                               "dstCount[%llu] dataType[%d]", __func__, i, srcSlice.size_, srcSlice.count_,
                               dstSlice.size_, dstSlice.count_, static_cast<int>(dataType)),
                    HCCL_E_PARA);
        void* dst = static_cast<void*>(static_cast<s8*>(dstSlice.addr_) + dstSlice.offset_);
        void* src = static_cast<void*>(static_cast<s8*>(srcSlice.addr_) + srcSlice.offset_);
        if (useBatchTransfer) {
            HcclHcommBatchTransferDesc desc = {};
            desc.transType = HCCL_HCOMM_TRANSFER_TYPE_WRITE_REDUCE;
            desc.transferInfo.reduce.count = srcSlice.count_;
            desc.transferInfo.reduce.dst = dst;
            desc.transferInfo.reduce.src = src;
            desc.transferInfo.reduce.dataType = static_cast<HcommDataType>(dataType);
            desc.transferInfo.reduce.reduceOp = static_cast<HcommReduceOp>(reduceOp);
            transferDescs.push_back(desc);
        } else {
            CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(
                thread, sendChannel.handle, dst, src, srcSlice.count_, static_cast<HcommDataType>(dataType),
                static_cast<HcommReduceOp>(reduceOp))));
        }
    }
    if (useBatchTransfer && !transferDescs.empty()) {
        CHK_RET(static_cast<HcclResult>(HcclHcommBatchTransferOnThread(
            thread, sendChannel.handle, transferDescs.data(), static_cast<u32>(transferDescs.size()))));
    }

    // Keep DONE synchronization for every step. It protects in-place partial sums and guarantees the
    // last remote write-reduce has completed before the operator copies its local result to user output.
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(HcclChannelNotifyWaitOnThreadDefault(
        thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout));
    return HCCL_SUCCESS;
}
} // namespace

InsTempReduceScatterOmniPipeNHR::InsTempReduceScatterOmniPipeNHR(
    const OpParam& param, const u32 rankId, // 传通信域的u32，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempReduceScatterNHR(param, rankId, subCommRanks)
{
    supportSymmetricMemory_ = param.supportSymmetricMemory;
    inputSymWindow_ = param.inputSymWindow;
    inputOffset_ = param.inputOffset;
}

InsTempReduceScatterOmniPipeNHR::~InsTempReduceScatterOmniPipeNHR()
{
}

// 语义改为返回当前template的类型，mesh返回1，nhr返回0
u64 InsTempReduceScatterOmniPipeNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return 0;
}

HcclResult InsTempReduceScatterOmniPipeNHR::KernelRun(const OpParam& param,
                                              const TemplateDataParams& tempAlgParams,
                                              TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempReduceScatterOmniPipeNHR] GenExtIns start");
    if (templateRankSize_ == 1) {
        HCCL_INFO("[InsTempReduceScatterOmniPipeNHR] Rank [%d], template ranksize is 1.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    tempAlgParams_       = tempAlgParams;
    channels_            = templateResource.channels;
    dataType_ = param.DataDes.dataType;

    // Direct mode maps the packed OmniPipe offsets back to user input and keeps every NHR step there.
    // The legacy step-0-only optimization remains available when direct mode is not requested.
    inputLoopBaseOff_ = tempAlgParams_.processedDataCount * DATATYPE_SIZE_TABLE[dataType_];
    inputRankStride_ = param.outputSize;
    inputLoopSize_ = tempAlgParams_.inputRepeatStride;
    inputTotalSize_ = param.inputSize;
    useSymmetricDirect_ = supportSymmetricMemory_ && tempAlgParams_.supportSymmetricMemory &&
                          tempAlgParams_.enableRemoteMemAccess;
    CHK_PRT_RET(useSymmetricDirect_ && (inputSymWindow_ == nullptr || tempAlgParams_.buffInfo.inputPtr == nullptr ||
                inputRankStride_ == 0 || inputLoopSize_ == 0),
                HCCL_ERROR("[%s] invalid direct symmetric input, inputWin[%p] inputPtr[%p] rankStride[%llu] "
                           "loopSize[%llu]", __func__, inputSymWindow_, tempAlgParams_.buffInfo.inputPtr,
                           inputRankStride_, inputLoopSize_),
                HCCL_E_PARA);
    useSymmetricInput_ = !useSymmetricDirect_ && supportSymmetricMemory_ && tempAlgParams_.supportSymmetricMemory &&
                         inputSymWindow_ != nullptr && inputRankStride_ != 0 && inputLoopSize_ != 0 &&
                         !IsPcieProtocol(channels_);
    if (supportSymmetricMemory_ && !useSymmetricDirect_ && !useSymmetricInput_) {
        HCCL_INFO("[%s] symmetric NHR fallback: firstAxis[%d] inputWin[%p] rankStride[%llu] loopSize[%llu] "
                  "isPcie[%d]", __func__, tempAlgParams_.supportSymmetricMemory, inputSymWindow_, inputRankStride_,
                  inputLoopSize_, IsPcieProtocol(channels_));
    } else if (useSymmetricDirect_) {
        HCCL_INFO("[%s] direct symmetric NHR write-reduce enabled: inputWin[%p] offset[%llu] jettyNum[%u]",
                  __func__, inputSymWindow_, inputOffset_, channelsPerRank_);
    }

    threadNum_ = GetThreadNum();

    // 这里的步骤nhr无需流同步、前后copy单独拿出来在executor中控制执行
    CHK_RET(PrepareOmniPipeDataSplitForMultiChannel(static_cast<CommonAlgTemplateBase*>(this), tempAlgParams_, dataType_, templateResource, 
        dataSplitVec_, dataOffsetVec_));
    HCCL_DEBUG("MT channelsPerRank_ = %llu, templateRankSize_ = %llu", channelsPerRank_, templateRankSize_);
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        // todo 这里的subThreads的size是0，就是说templateResource.threads给的值不对
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    HCCL_DEBUG("MT channelsPerRank_ = %llu, templateRankSize_ = %llu", channelsPerRank_, templateRankSize_);
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(RunNHR(templateResource.threads, channelIdx));
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    HCCL_INFO("[InsTempAllGatherOmniPipeNHR] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeNHR::DoLocalCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempReduceScatterOmniPipeNHR][DoLocalCopy] DoLocalCopy myRank_ = [%u]", myRank_);
    u32 rankIdx = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        rankIdx = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[%s]subCommRanks_ or myRank_ is error.", __func__);
        return HCCL_E_INTERNAL;
    }

    // 区分前后搬运
    void* srcAddr;
    void* dstAddr;
    if (tempAlgParams.buffInfo.inBuffType == BufferType::INPUT) {
        srcAddr = tempAlgParams.buffInfo.inputPtr;
        dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    } else if (tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
        srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        dstAddr = tempAlgParams.buffInfo.outputPtr;
    } else {
        HCCL_ERROR("[%s]InputBufferType Error.", __func__);
        return HCCL_E_PARA;
    }
    // 这里的循环precopy是ranksize-1，postcopy是1
    for (auto i = 0; i < tempAlgParams.repeatNum; ++i) {
        auto srcSlice = DataSlice(srcAddr,
                                  tempAlgParams.buffInfo.inBuffBaseOff +
                                  i * tempAlgParams.inputSliceStride,
                                  tempAlgParams.sliceSize, tempAlgParams.count);
        auto dstSlice = DataSlice(dstAddr,
                                  tempAlgParams.buffInfo.outBuffBaseOff +
                                  i * tempAlgParams.outputSliceStride,
                                  tempAlgParams.sliceSize, tempAlgParams.count);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeNHR::GetNHRDataSize(const AicpuNHRStepInfo& st, const u32 channelIdx, 
                    void* sendCclBuffAddr, void* recvCclBuffAddr, const u32 dataTypeSize, const u64 rptNum,
                    std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices, 
                    std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices){
    for (u32 i = 0; i < st.nSlices; ++i) {
        const u32 txIdx = st.txSliceIdxs[i]; // 算法序
        const u32 rxIdx = st.rxSliceIdxs[i];
        for (u64 rpt = 0; rpt < rptNum; ++rpt) {
            u64 scratchBaseTx = tempAlgParams_.buffInfo.inBuffBaseOff + tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt];
            u64 scratchBaseRx = tempAlgParams_.buffInfo.inBuffBaseOff + tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[rxIdx][rpt];
            scratchBaseTx += dataOffsetVec_[txIdx][rpt][channelIdx];
            scratchBaseRx += dataOffsetVec_[rxIdx][rpt][channelIdx];
            // 已对齐，这边都用input
            const u64 txScOff = scratchBaseTx + tempAlgParams_.stepSliceInfo.stepInputSliceStride[txIdx];
            const u64 rxScOff = scratchBaseRx + tempAlgParams_.stepSliceInfo.stepInputSliceStride[rxIdx];

            DataSlice txSrcSlice = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr,
                txScOff,
                dataSplitVec_[txIdx][rpt][channelIdx],
                dataSplitVec_[txIdx][rpt][channelIdx]/dataTypeSize);  // 发送源
            DataSlice txDstSlice = DataSlice(sendCclBuffAddr,
                txScOff,
                dataSplitVec_[txIdx][rpt][channelIdx],
                dataSplitVec_[txIdx][rpt][channelIdx]/dataTypeSize);  // 发送目标
            DataSlice rxSrcSlice = DataSlice(recvCclBuffAddr,
                rxScOff,
                dataSplitVec_[rxIdx][rpt][channelIdx],
                dataSplitVec_[rxIdx][rpt][channelIdx]/dataTypeSize);
            DataSlice rxDstSlice = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr,
                rxScOff,
                dataSplitVec_[rxIdx][rpt][channelIdx],
                dataSplitVec_[rxIdx][rpt][channelIdx]/dataTypeSize);
            txSrcSlices.emplace_back(txSrcSlice);
            txDstSlices.emplace_back(txDstSlice);
            rxSrcSlices.emplace_back(rxSrcSlice);
            rxDstSlices.emplace_back(rxDstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeNHR::RunSymmetricStep0(const AicpuNHRStepInfo& stepInfo,
                                                               const std::vector<ThreadHandle>& threads,
                                                               u32 channelIdx, u32 dataTypeSize)
{
    CHK_PRT_RET(threads.size() <= channelIdx,
                HCCL_ERROR("[RS-NHR][RunSymmetricStep0] missing thread for channelIdx[%u]", channelIdx),
                HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(inputLoopSize_ == 0 || inputRankStride_ == 0,
                HCCL_ERROR("[RS-NHR][RunSymmetricStep0] invalid input layout, rankStride[%llu] loopSize[%llu]",
                           inputRankStride_, inputLoopSize_),
                HcclResult::HCCL_E_INTERNAL);

    const u32 recvFromRank = subCommRanks_[0].at(stepInfo.fromRank);
    void* peerInputAddr = nullptr;
    HcclResult ret = HcclSymWinGetPeerPointer(inputSymWindow_, inputOffset_, recvFromRank, &peerInputAddr);
    CHK_PRT_RET(ret != HCCL_SUCCESS || peerInputAddr == nullptr,
                HCCL_ERROR("[RS-NHR][RunSymmetricStep0] HcclSymWinGetPeerPointer failed, peerRank[%u] ret[%d] "
                           "addr[%p]", recvFromRank, ret, peerInputAddr),
                HcclResult::HCCL_E_INTERNAL);

    const u64 rptNum = std::max<u64>(1, tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[0].size());
    for (u32 i = 0; i < stepInfo.nSlices; ++i) {
        const u32 rxIdx = stepInfo.rxSliceIdxs[i];
        for (u64 rpt = 0; rpt < rptNum; ++rpt) {
            u64 localCclOffset = tempAlgParams_.buffInfo.inBuffBaseOff +
                                  tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[rxIdx][rpt] +
                                  dataOffsetVec_[rxIdx][rpt][channelIdx] +
                                  tempAlgParams_.stepSliceInfo.stepInputSliceStride[rxIdx];
            u64 sliceSize = dataSplitVec_[rxIdx][rpt][channelIdx];
            u64 inputRank = localCclOffset / inputLoopSize_;
            u64 inputOffsetInLoop = localCclOffset % inputLoopSize_;
            CHK_PRT_RET(inputOffsetInLoop + sliceSize > inputLoopSize_,
                        HCCL_ERROR("[RS-NHR][RunSymmetricStep0] input slice crosses packed rank block, "
                                   "offset[%llu] size[%llu] loopSize[%llu]", localCclOffset, sliceSize,
                                   inputLoopSize_),
                        HcclResult::HCCL_E_INTERNAL);

            u64 peerInputOffset = inputLoopBaseOff_ + inputRank * inputRankStride_ + inputOffsetInLoop;
            CHK_PRT_RET(peerInputOffset > inputTotalSize_ || sliceSize > inputTotalSize_ - peerInputOffset,
                        HCCL_ERROR("[RS-NHR][RunSymmetricStep0] peer input slice exceeds registered input, "
                                   "offset[%llu] size[%llu] inputSize[%llu]", peerInputOffset, sliceSize,
                                   inputTotalSize_),
                        HcclResult::HCCL_E_INTERNAL);
            DataSlice srcSlice(peerInputAddr, peerInputOffset, sliceSize, sliceSize / dataTypeSize);
            DataSlice dstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, localCclOffset, sliceSize,
                               sliceSize / dataTypeSize);
            CHK_RET(static_cast<HcclResult>(
                LocalReduce(threads[channelIdx], srcSlice, dstSlice, dataType_, reduceOp_)));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeNHR::RunSymmetricDirectWriteStep(
    const AicpuNHRStepInfo& stepInfo, const std::vector<ThreadHandle>& threads, u32 channelIdx, u32 dataTypeSize)
{
    CHK_PRT_RET(threads.size() <= channelIdx,
                HCCL_ERROR("[RS-NHR][%s] missing thread for channelIdx[%u]", __func__, channelIdx),
                HCCL_E_INTERNAL);
    const u32 recvFromRank = subCommRanks_[0].at(stepInfo.fromRank);
    const u32 sendToRank = subCommRanks_[0].at(stepInfo.toRank);
    CHK_PRT_RET(channels_.count(recvFromRank) == 0 || channels_.count(sendToRank) == 0 ||
                channels_.at(recvFromRank).size() <= channelIdx || channels_.at(sendToRank).size() <= channelIdx,
                HCCL_ERROR("[RS-NHR][%s] link missing, recvFrom[%u] sendTo[%u] channelIdx[%u]", __func__,
                           recvFromRank, sendToRank, channelIdx),
                HCCL_E_INTERNAL);
    const ChannelInfo& linkRecv = channels_.at(recvFromRank)[channelIdx];
    const ChannelInfo& linkSend = channels_.at(sendToRank)[channelIdx];

    void* peerInputAddr = nullptr;
    HcclResult ret = HcclSymWinGetPeerPointer(inputSymWindow_, inputOffset_, sendToRank, &peerInputAddr);
    CHK_PRT_RET(ret != HCCL_SUCCESS || peerInputAddr == nullptr,
                HCCL_ERROR("[RS-NHR][%s] HcclSymWinGetPeerPointer failed, peerRank[%u] ret[%d] addr[%p]",
                           __func__, sendToRank, ret, peerInputAddr),
                HCCL_E_INTERNAL);

    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[0].size());
    for (u32 i = 0; i < stepInfo.nSlices; ++i) {
        const u32 txIdx = stepInfo.txSliceIdxs[i];
        for (u64 rpt = 0; rpt < rptNum; ++rpt) {
            const u64 packedOffset = tempAlgParams_.buffInfo.inBuffBaseOff +
                tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt] +
                dataOffsetVec_[txIdx][rpt][channelIdx] +
                tempAlgParams_.stepSliceInfo.stepInputSliceStride[txIdx];
            const u64 sliceSize = dataSplitVec_[txIdx][rpt][channelIdx];
            u64 userInputOffset = 0;
            CHK_RET(CalcDirectUserInputOffset(packedOffset, sliceSize, inputLoopBaseOff_, inputRankStride_,
                                              inputLoopSize_, inputTotalSize_, userInputOffset));
            txSrcSlices.emplace_back(tempAlgParams_.buffInfo.inputPtr, userInputOffset, sliceSize,
                                     sliceSize / dataTypeSize);
            txDstSlices.emplace_back(peerInputAddr, userInputOffset, sliceSize,
                                     sliceSize / dataTypeSize);
        }
    }
    CHK_PRT_RET(DirectWriteReduceWithStepSync(linkSend, linkRecv, txSrcSlices, txDstSlices, dataType_, reduceOp_,
                                             threads[channelIdx], stepInfo.step != 0),
                HCCL_ERROR("[RS-NHR][%s] direct write-reduce failed, step[%u]", __func__, stepInfo.step),
                HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeNHR::RunNHR(const std::vector<ThreadHandle> &threads, u32 channelIdx)
{
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(threads.empty(),
        HCCL_ERROR("[RS-NHR][RunNHR] empty queue"), HcclResult::HCCL_E_INTERNAL);

    if (templateRankSize_ <= 1) return HcclResult::HCCL_SUCCESS;
    bool isPcieProtocal = IsPcieProtocol(channels_);
    // 步进参数，片数由inputOmniPipeSliceStride确定
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[0].size());

    // 预计算步骤列表（算法序）
    std::vector<AicpuNHRStepInfo> steps;
    CHK_RET(GetStepInfoList(steps));
    // 基础位置由 hcclBuffBaseOff 和 inputOmniPipeSliceStride 确定
    for (u32 s = 0; s < steps.size(); ++s) {
        const auto &st = steps[s];
        if (useSymmetricDirect_) {
            CHK_RET(RunSymmetricDirectWriteStep(st, threads, channelIdx, dataTypeSize));
            continue;
        } else if (useSymmetricInput_ && st.step == 0) {
            // 所有 rank 在首轮都通过 peer input 做本地规约；同一 thread 上的后续 SendRecv 保持顺序依赖。
            CHK_RET(RunSymmetricStep0(st, threads, channelIdx, dataTypeSize));
            continue;
        }
        const u32 recvFromRank = subCommRanks_[0].at(st.fromRank);
        const u32 sendToRank   = subCommRanks_[0].at(st.toRank);
        CHK_PRT_RET(recvFromRank == static_cast<u32>(-1) || sendToRank == static_cast<u32>(-1),
            HCCL_ERROR("[RS-NHR][RunNHR] rank map failed: from[%u] to[%u]", st.fromRank, st.toRank),
            HcclResult::HCCL_E_INTERNAL);

        CHK_PRT_RET(channels_.count(recvFromRank) == 0 || channels_.count(sendToRank) == 0 ||
                    channels_[recvFromRank].size() == 0 || channels_[sendToRank].size() == 0,
                    HCCL_ERROR("[RS-NHR][RunNHR] link missing: recvFrom=%d sendTo=%d", recvFromRank, sendToRank),
            HcclResult::HCCL_E_INTERNAL);

        ChannelInfo linkRecv = channels_[recvFromRank].at(channelIdx);
        ChannelInfo linkSend = channels_[sendToRank].at(channelIdx);
        HCCL_DEBUG("recvFromRank=[%u], sendToRank=[%u], linkRecv.remoteRank=[%u], linkSend.remoteRank=[%u]", recvFromRank, sendToRank, linkRecv.remoteRank, linkSend.remoteRank);

        std::vector<DataSlice> txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices;

        void* sendCclBuffAddr = linkSend.remoteCclMem.addr;
        void* recvCclBuffAddr = linkRecv.remoteCclMem.addr;
        // RS：在 SCRATCH 上进行规约交换
        CHK_RET(GetNHRDataSize(st, channelIdx, sendCclBuffAddr, recvCclBuffAddr, dataTypeSize, rptNum, 
                    txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices));

        SendRecvReduceInfo info{
            { linkSend, linkRecv }, { { txSrcSlices, txDstSlices }, { rxSrcSlices, rxDstSlices } }, dataType_, reduceOp_
        };
        if (isPcieProtocal) {
            CHK_PRT_RET(SendRecvReadReduce(info, threads[channelIdx]),
                HCCL_ERROR("[RS-NHR][RunNHR] SendRecvReduce failed (step=%u)", st.step), HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(SendRecvBatchWriteReduce(info, threads[channelIdx]),
                HCCL_ERROR("[RS-NHR][RunNHR] SendRecvReduce failed (step=%u)", st.step), HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

//  计算每轮收发的对端以及slice编号
HcclResult InsTempReduceScatterOmniPipeNHR::GetStepInfoList(std::vector<AicpuNHRStepInfo> &stepInfoList)
{
    // 将本 rank 号转换成算法使用的索引号
    u32 u32x = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], u32x));
    stepInfoList.clear();
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    stepInfoList.resize(nSteps);

    for (u32 step = 0; step < nSteps; step++) {
        // 计算通信对象
        u32 deltaRank = 1 << step;
        u32 sendTo = (u32x + templateRankSize_ - deltaRank) % templateRankSize_;
        u32 recvFrom = (u32x + deltaRank) % templateRankSize_;

        // 数据份数和数据编号增量
        u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
        u32 deltaSliceIndex = 1 << (step + 1);
        u32 rxSliceIdx = u32x;
        u32 txSliceIdx = sendTo;

        AicpuNHRStepInfo &currStepInfo = stepInfoList[step];
        currStepInfo.step = step;
        currStepInfo.toRank = sendTo;
        currStepInfo.myRank = u32x;
        currStepInfo.fromRank = recvFrom;
        currStepInfo.nSlices = nSlices;

        // 计算本rank在每轮收/发中的slice编号
        currStepInfo.txSliceIdxs.reserve(nSlices);
        currStepInfo.rxSliceIdxs.reserve(nSlices);
        for (u32 i = 0; i < nSlices; i++) {
            currStepInfo.txSliceIdxs.push_back(txSliceIdx);
            currStepInfo.rxSliceIdxs.push_back(rxSliceIdx);
            HCCL_DEBUG("[InsTempReduceScatterOmniPipeNHR][GetStepInfoList] i[%u] txSliceIdx[%u] rxSliceIdx[%u], step[%u]", i, txSliceIdx, rxSliceIdx, step);
            txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
            rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
