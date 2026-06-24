/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_v_aicpu_reduce_nhr.h"

namespace ops_hccl {
InsTempReduceScatterVAicpuReduceNHR::InsTempReduceScatterVAicpuReduceNHR(
    const OpParam& param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempReduceScatterVAicpuReduceNHR::~InsTempReduceScatterVAicpuReduceNHR()
{
}

HcclResult InsTempReduceScatterVAicpuReduceNHR::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                             AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, channels));
    resourceRequest.channels.push_back(channels);
    HCCL_INFO("[InsTempReduceScatterVAicpuReduceNHR][CalcRes] slaveThreadNum: [%u], notifyNumOnMainThread: [%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterVAicpuReduceNHR::GetThreadNum() const
{
    return 1;
}

HcclResult InsTempReduceScatterVAicpuReduceNHR::KernelRun(const OpParam& param,
                                               const TemplateDataParams& tempAlgParams,
                                               TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempReduceScatterVAicpuReduceNHR] KernelRun start");

    tempAlgParams_ = tempAlgParams;
    channels_ = templateResource.channels;
    dataType_ = param.vDataDes.dataType;

    allRankProcessSize_ = tempAlgParams.allRankSliceSize;
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    allRankCounts_.resize(templateRankSize_);
    for (u64 i = 0; i < allRankProcessSize_.size(); ++i) {
        allRankCounts_[i] = allRankProcessSize_[i] / dataTypeSize;
    }
    // 每个循环只处理一份数据
    for (u32 sliceIdx = 0; sliceIdx < templateRankSize_; ++sliceIdx) {
        CHK_RET(LocalDataCopy(templateResource.threads, sliceIdx));
        // 如果是本卡数据，拷贝到output
        if (sliceIdx == myRank_) {
            CHK_RET(LocalCopyToOutput(templateResource.threads, sliceIdx));
        }
        // 这里通过allgather来传递数据，实际上只有一个rank需要这份数据
        CHK_RET(RunAllGather(templateResource.threads, sliceIdx));
        // 每个循环都必须确保通信任务完成后，才去做本地归约
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        for (const auto &thread : templateResource.threads) {
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
        }

        if (sliceIdx == myRank_) {
            CHK_RET(PostLocalReduce(templateResource.threads));
        }
    }

    HCCL_INFO("[InsTempReduceScatterVAicpuReduceNHR] KernelRun end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterVAicpuReduceNHR::LocalDataCopy(const std::vector<ThreadHandle> &threads, u32 sliceIdx)
{
    CHK_PRT_RET(threads.empty(),
        HCCL_ERROR("[InsTempReduceScatterVAicpuReduceNHR][LocalDataCopy] empty threads"), HcclResult::HCCL_E_INTERNAL);

    ThreadHandle q = threads[0];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);

    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));

    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        // 当前只处理sliceIdx对应的数据
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff +
                              rpt * tempAlgParams_.inputRepeatStride;
        const u64 inOff = inBaseOff + tempAlgParams_.allRankDispls[sliceIdx];
        // 拷贝到本卡的cclbuffer，地址是myAlgIdx对应的位置，方便后续做nhr
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff +
                                rpt * tempAlgParams_.outputRepeatStride;
        const u64 scOff = scratchBase + tempAlgParams_.outputSliceStride * myAlgIdx;

        DataSlice src = DataSlice(tempAlgParams_.buffInfo.inputPtr, inOff,
            allRankProcessSize_[sliceIdx], allRankCounts_[sliceIdx]);
        DataSlice dst = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scOff,
            allRankProcessSize_[sliceIdx], allRankCounts_[sliceIdx]);

        HCCL_INFO("[InsTempReduceScatterVAicpuReduceNHR][LocalDataCopy] rpt[%u] inOff[%llu] scOff[%llu] processSize[%llu]",
            rpt, inOff, scOff, allRankProcessSize_[sliceIdx]);

        if (tempAlgParams_.buffInfo.inBuffType != tempAlgParams_.buffInfo.hcclBuffType || inOff != scOff) {
            CHK_RET(LocalCopy(q, src, dst));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterVAicpuReduceNHR::LocalCopyToOutput(const std::vector<ThreadHandle> &threads, u32 sliceIdx)
{
    CHK_PRT_RET(threads.empty(),
        HCCL_ERROR("[InsTempReduceScatterVAicpuReduceNHR][LocalCopyToOutput] empty threads"), HcclResult::HCCL_E_INTERNAL);
    ThreadHandle q = threads[0];
    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff +
                              rpt * tempAlgParams_.inputRepeatStride;
        const u64 inOff = inBaseOff + tempAlgParams_.allRankDispls[sliceIdx];
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff +
                                rpt * tempAlgParams_.outputRepeatStride;
        DataSlice src = DataSlice(tempAlgParams_.buffInfo.inputPtr, inOff,
            allRankProcessSize_[myAlgIdx], allRankCounts_[myAlgIdx]);
        DataSlice dst = DataSlice(tempAlgParams_.buffInfo.outputPtr, outBaseOff,
            allRankProcessSize_[myAlgIdx], allRankCounts_[myAlgIdx]);
        HCCL_INFO("[InsTempReduceScatterVAicpuReduceNHR][LocalCopyToOutput] rpt[%u] inOff[%llu] outBaseOff[%llu] processSize[%llu]",
            rpt, inOff, outBaseOff, allRankProcessSize_[myAlgIdx]);
        if (tempAlgParams_.buffInfo.inBuffType != tempAlgParams_.buffInfo.outBuffType || inOff != outBaseOff) {
            CHK_RET(LocalCopy(q, src, dst));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterVAicpuReduceNHR::PostLocalReduce(const std::vector<ThreadHandle> &threads)
{
    CHK_PRT_RET(threads.empty(),
        HCCL_ERROR("[InsTempReduceScatterVAicpuReduceNHR][PostLocalReduce] empty threads"), HcclResult::HCCL_E_INTERNAL);

    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));
    ThreadHandle q = threads[0];

    const u64 rptNum = std::max<u64>(1, tempAlgParams_.repeatNum);
    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.outputSliceStride * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (u32 rankIdx = 0; rankIdx < templateRankSize_; ++rankIdx) {
            if (rankIdx == myAlgIdx) {
                continue;
            }

            const u64 otherScratchOff = scratchBase + tempAlgParams_.outputSliceStride * rankIdx;
            DataSlice reduceSrcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, otherScratchOff,
                allRankProcessSize_[myAlgIdx], allRankCounts_[myAlgIdx]);
            DataSlice reduceDstSlice(tempAlgParams_.buffInfo.outputPtr, outBaseOff,
                allRankProcessSize_[myAlgIdx], allRankCounts_[myAlgIdx]);

            HCCL_INFO("[InsTempReduceScatterVAicpuReduceNHR][PostLocalReduce] reduce from rank[%u] otherScratchOff[%llu]",
                rankIdx, otherScratchOff);

            CHK_RET(LocalReduce(q, reduceSrcSlice, reduceDstSlice, dataType_, reduceOp_));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterVAicpuReduceNHR::RunAllGather(const std::vector<ThreadHandle> &threads, u32 sliceIdx)
{
    const u32 nSteps = GetNHRStepNum(templateRankSize_);

    for (u32 step = 0; step < nSteps; ++step) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));

        const ChannelInfo &channelRecv = channels_.at(GetRankFromMap(stepInfo.fromRank))[0];
        const ChannelInfo &channelSend = channels_.at(GetRankFromMap(stepInfo.toRank))[0];
        std::vector<DataSlice> txSrcSlicesAll;
        std::vector<DataSlice> txDstSlicesAll;
        std::vector<DataSlice> rxSrcSlicesAll;
        std::vector<DataSlice> rxDstSlicesAll;
        void *sendCclBuffAddr = channelSend.remoteCclMem.addr;
        void *recvCclBuffAddr = channelRecv.remoteCclMem.addr;

        HCCL_DEBUG(
            "[InsTempReduceScatterVAicpuReduceNHR] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u]",
            myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

        for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
            const u64 scratchRepeatStride = tempAlgParams_.outputSliceStride * templateRankSize_;
            const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

            for (u32 i = 0; i < stepInfo.nSlices; ++i) {
                const u32 txIdx = stepInfo.txSliceIdxs[i];
                const u32 rxIdx = stepInfo.rxSliceIdxs[i];

                const u64 txScratchOff = scratchBase + tempAlgParams_.outputSliceStride * txIdx;

                const u64 rxScratchOff = scratchBase + tempAlgParams_.outputSliceStride * rxIdx;

                txSrcSlicesAll.emplace_back(tempAlgParams_.buffInfo.hcclBuff.addr, txScratchOff,
                    allRankProcessSize_[sliceIdx], allRankCounts_[sliceIdx]);
                txDstSlicesAll.emplace_back(sendCclBuffAddr, txScratchOff,
                    allRankProcessSize_[sliceIdx], allRankCounts_[sliceIdx]);
                rxSrcSlicesAll.emplace_back(recvCclBuffAddr, rxScratchOff,
                    allRankProcessSize_[sliceIdx], allRankCounts_[sliceIdx]);
                rxDstSlicesAll.emplace_back(tempAlgParams_.buffInfo.hcclBuff.addr, rxScratchOff,
                    allRankProcessSize_[sliceIdx], allRankCounts_[sliceIdx]);
            }
        }
        TxRxSlicesList sendRecvSlicesList({txSrcSlicesAll, txDstSlicesAll}, {rxSrcSlicesAll, rxDstSlicesAll});
        TxRxChannels sendRecvChannels(channelSend, channelRecv);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[0]),
            HCCL_ERROR("[InsTempReduceScatterVAicpuReduceNHR] sendrecv failed (step=%u)", step),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterVAicpuReduceNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void) inBuffType;
    (void) outBuffType;
    HCCL_INFO(
        "[InsTempReduceScatterVAicpuReduceNHR][CalcScratchMultiple] templateScratchMultiplier[%llu]", templateRankSize_);
    return templateRankSize_;
}

u32 InsTempReduceScatterVAicpuReduceNHR::GetRankFromMap(const u32 algRankIdx)
{
    return subCommRanks_[0].at(algRankIdx);
}

HcclResult InsTempReduceScatterVAicpuReduceNHR::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = myAlgRank;

    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (myAlgRank + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 sendTo = (myAlgRank + deltaRank) % templateRankSize_;

    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = myAlgRank;
    u32 rxSliceIdx = (myAlgRank - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.fromRank = recvFrom;
    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG("[ReduceScatterVAicpu][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempReduceScatterVAicpuReduceNHR::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    (void)notifyIdxMianToSub;
}

void InsTempReduceScatterVAicpuReduceNHR::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    (void)notifyIdxSubToMain;
}
}
