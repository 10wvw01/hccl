/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_nhr_dpu.h"
#include "dpu_alg_nhr_opt_wrapper.h"

namespace ops_hccl {

InsTempReduceScatterNhrDpu::InsTempReduceScatterNhrDpu()
{
}

InsTempReduceScatterNhrDpu::InsTempReduceScatterNhrDpu(const OpParam& param,
                                                        const u32 rankId,
                                                        const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempReduceScatterNhrDpu::~InsTempReduceScatterNhrDpu()
{
}

HcclResult InsTempReduceScatterNhrDpu::CalcRes(HcclComm comm, const OpParam& param,
                                               const TopoInfoWithNetLayerDetails* topoInfo,
                                               AlgResourceRequest& resourceRequest)
{
    // DPU 模式：不需要从流，只使用主线程
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);

    HCCL_INFO("[InsTempReduceScatterNhrDpu][CalcRes] slaveThreadNum[%u] notifyNumOnMainThread[%u] level1Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread,
        level1Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterNhrDpu::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void) inBufferType;
    (void) outBufferType;
    // 方案3：CCL 缓冲区需要双倍空间
    // - 前半部分（templateRankSize_）：存储本地数据
    // - 后半部分（templateRankSize_）：临时存储远程数据，用于规约
    u64 scratchMultiple = 2 * subCommRanks_[0].size();
    return scratchMultiple;
}

HcclResult InsTempReduceScatterNhrDpu::KernelRun(const OpParam& param,
                                                  const TemplateDataParams& tempAlgParams,
                                                  TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    tempAlgParams_ = tempAlgParams;
    channels_ = templateResource.channels;

    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempReduceScatterNhrDpu] Rank [%d], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // 切换到 eager 模式
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempReduceScatterNhrDpu] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
        HCCL_ERROR("[InsTempReduceScatterNhrDpu] HcommThreadSynchronize failed");
        return HCCL_E_INTERNAL;
    }

    // Step 1: 将输入数据拷贝到 cclBuff
    CHK_RET(LocalDataCopy(param, tempAlgParams, templateResource.threads));

    // Step 2: 获取所有 step 信息
    std::vector<AicpuNHRStepInfo> stepInfoList;
    CHK_RET(GetStepInfoList(stepInfoList));

    // Step 3: 逐个 step 执行通信 + 规约
    for (u32 step = 0; step < stepInfoList.size(); ++step) {
        // 3.1 执行当前 step 的 DPU 数据交换
        CHK_RET(RunSingleStep(param, tempAlgParams, templateResource, step));

        // 3.2 立即规约接收到的数据（关键！）
        CHK_RET(LocalReduceAfterStep(param, tempAlgParams, templateResource.threads, step, stepInfoList[step]));

        HCCL_INFO("[InsTempReduceScatterNhrDpu] Step[%u] completed, toRank[%u] fromRank[%u]",
            step, stepInfoList[step].toRank, stepInfoList[step].fromRank);
    }

    // Step 4: 将结果拷贝到输出
    CHK_RET(PostLocalCopy(param, tempAlgParams, templateResource.threads));

    // 切换回 batch 模式
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempReduceScatterNhrDpu] failed set batch mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[InsTempReduceScatterNhrDpu] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterNhrDpu::RunSingleStep(const OpParam& param,
                                                     const TemplateDataParams& tempAlgParams,
                                                     TemplateResource& templateResource,
                                                     u32 step)
{
    // 构造单步 DPU 运行信息
    DPURunInfo dpuRunInfo;
    dpuRunInfo.templateName = "InsTempReduceScatterNhrDpu";
    dpuRunInfo.tempAlgParams = tempAlgParams;
    // 复用 root 字段传递 currentStep（ReduceScatter 不需要 root）
    dpuRunInfo.tempAlgParams.root = step;
    dpuRunInfo.channels = templateResource.channels;
    dpuRunInfo.myRank = myRank_;
    dpuRunInfo.subCommRanks = subCommRanks_;

    u32 sendMsgId = 0;
    auto dpuRunInfoSeqData = dpuRunInfo.Serialize();
    if (HcommSendRequest(reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
        static_cast<void*>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId) != 0) {
        HCCL_ERROR("[InsTempReduceScatterNhrDpu] HcommSendRequest failed at step[%u]", step);
        return HCCL_E_INTERNAL;
    }

    // 等待 DPU 完成当前 step
    void *recvData = nullptr;
    u32 recvMsgId = 0;
    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("[InsTempReduceScatterNhrDpu] HcommWaitResponse failed at step[%u]", step);
        return HCCL_E_INTERNAL;
    }

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR("[InsTempReduceScatterNhrDpu] recvMsgId[%u] not equal to sendMsgId[%u] at step[%u]",
            recvMsgId, sendMsgId, step);
        return HCCL_E_INTERNAL;
    }

    HCCL_DEBUG("[InsTempReduceScatterNhrDpu] DPU step[%u] completed", step);
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterNhrDpu::LocalReduceAfterStep(const OpParam& param,
                                                            const TemplateDataParams& tempAlgParams,
                                                            const std::vector<ThreadHandle>& threads,
                                                            u32 step,
                                                            const AicpuNHRStepInfo& stepInfo)
{
    // 方案3：规约逻辑
    // DPU 已经将远程数据写入到 cclBuff[rxIdx + templateRankSize_]
    // 需要手动规约：cclBuff[rxIdx] + cclBuff[rxIdx + templateRankSize_] → cclBuff[rxIdx]

    if (stepInfo.rxSliceIdxs.empty()) {
        return HCCL_SUCCESS;
    }

    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 count = processSize_ / dataTypeSize;
    const u64 rptNum = std::max<u64>(1, tempAlgParams.repeatNum);

    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff
                              + rpt * tempAlgParams.outputRepeatStride;

        for (u32 i = 0; i < stepInfo.nSlices; ++i) {
            const u32 rxIdx = stepInfo.rxSliceIdxs[i];

            // 本地数据位置：cclBuff[rxIdx]
            DataSlice localSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                scratchBase + rxIdx * tempAlgParams.sliceSize,
                processSize_, count);

            // 远程数据位置：cclBuff[rxIdx + templateRankSize_]
            DataSlice remoteSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                scratchBase + (rxIdx + templateRankSize_) * tempAlgParams.sliceSize,
                processSize_, count);

            // 规约：remote + local → local
            CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], remoteSlice, localSlice, dataType_, reduceOp_)));

            HCCL_DEBUG("[InsTempReduceScatterNhrDpu] Step[%u] LocalReduce: "
                "cclBuff[%u] + cclBuff[%u] → cclBuff[%u]",
                step, rxIdx + templateRankSize_, rxIdx, rxIdx);
        }
    }

    HCCL_INFO("[InsTempReduceScatterNhrDpu] Step[%u] LocalReduceAfterStep completed, nSlices[%u]",
        step, stepInfo.nSlices);
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterNhrDpu::LocalDataCopy(const OpParam& param,
                                                     const TemplateDataParams& tempAlgParams,
                                                     const std::vector<ThreadHandle>& threads)
{
    // 将输入数据从 inputPtr 搬运到 cclBuff
    const u64 rptNum = std::max<u64>(1, tempAlgParams.repeatNum);

    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
        u64 count = processSize_ / dataTypeSize;

        const u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff
                              + rpt * tempAlgParams.outputRepeatStride;

        // 将每个 rank 对应的输入数据拷贝到 cclBuff 的对应位置
        for (u32 rankIdx = 0; rankIdx < templateRankSize_; ++rankIdx) {
            DataSlice srcSlice = DataSlice(
                tempAlgParams.buffInfo.inputPtr,
                tempAlgParams.buffInfo.inBuffBaseOff
                    + rankIdx * tempAlgParams.inputSliceStride
                    + rpt * tempAlgParams.inputRepeatStride,
                processSize_, count);

            DataSlice dstSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                scratchBase + rankIdx * tempAlgParams.sliceSize,
                processSize_, count);

            CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
        }
    }

    HCCL_DEBUG("[InsTempReduceScatterNhrDpu] LocalDataCopy completed");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterNhrDpu::PostLocalCopy(const OpParam& param,
                                                     const TemplateDataParams& tempAlgParams,
                                                     const std::vector<ThreadHandle>& threads)
{
    u32 myAlgRank = 0;
    std::vector<u32> rankIds = subCommRanks_[0];
    auto iter = std::find(rankIds.begin(), rankIds.end(), myRank_);
    if (iter != rankIds.end()) {
        myAlgRank = std::distance(rankIds.begin(), iter);
    } else {
        HCCL_ERROR("[InsTempReduceScatterNhrDpu][PostLocalCopy] rankIds or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }

    const u64 rptNum = std::max<u64>(1, tempAlgParams.repeatNum);

    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
        u64 count = processSize_ / dataTypeSize;

        const u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff
                              + rpt * tempAlgParams.outputRepeatStride;

        // 将 cclBuff[myAlgRank] 拷贝到 outputPtr
        DataSlice srcSlice = DataSlice(
            tempAlgParams.buffInfo.hcclBuff.addr,
            scratchBase + myAlgRank * tempAlgParams.sliceSize,
            processSize_, count);

        DataSlice dstSlice = DataSlice(
            tempAlgParams.buffInfo.outputPtr,
            tempAlgParams.buffInfo.outBuffBaseOff
                + rpt * tempAlgParams.outputRepeatStride,
            processSize_, count);

        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    }

    HCCL_DEBUG("[InsTempReduceScatterNhrDpu] PostLocalCopy completed");
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterNhrDpu::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo) const
{
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));

    stepInfo.step = step;
    stepInfo.myRank = myAlgIdx;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();

    // ReduceScatter NHR 计算通信对象
    u32 deltaRank = 1 << step;
    u32 sendTo = (myAlgIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 recvFrom = (myAlgIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);

    // ReduceScatter NHR 的 slice 索引计算
    u32 txSliceIdx = sendTo;
    u32 rxSliceIdx = myAlgIdx;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    stepInfo.txSliceIdxs.reserve(nSlices);
    stepInfo.rxSliceIdxs.reserve(nSlices);

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }

    HCCL_DEBUG("[InsTempReduceScatterNhrDpu][GetStepInfo] step[%u] toRank[%u] fromRank[%u] nSlices[%u]",
        step, sendTo, recvFrom, nSlices);
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterNhrDpu::GetStepInfoList(std::vector<AicpuNHRStepInfo> &stepInfoList) const
{
    u32 myAlgIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgIdx));

    stepInfoList.clear();
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    stepInfoList.resize(nSteps);

    for (u32 step = 0; step < nSteps; step++) {
        // ReduceScatter NHR 计算通信对象
        // 与 AllGather NHR 相反的方向
        u32 deltaRank = 1 << step;
        u32 sendTo = (myAlgIdx + templateRankSize_ - deltaRank) % templateRankSize_;
        u32 recvFrom = (myAlgIdx + deltaRank) % templateRankSize_;

        // 数据份数和数据编号增量
        u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
        u32 deltaSliceIndex = 1 << (step + 1);

        // ReduceScatter NHR 的 slice 索引计算
        u32 txSliceIdx = sendTo;
        u32 rxSliceIdx = myAlgIdx;

        AicpuNHRStepInfo &currStepInfo = stepInfoList[step];
        currStepInfo.step = step;
        currStepInfo.myRank = myAlgIdx;
        currStepInfo.nSlices = nSlices;
        currStepInfo.toRank = sendTo;
        currStepInfo.fromRank = recvFrom;

        currStepInfo.txSliceIdxs.reserve(nSlices);
        currStepInfo.rxSliceIdxs.reserve(nSlices);

        for (u32 i = 0; i < nSlices; i++) {
            currStepInfo.txSliceIdxs.push_back(txSliceIdx);
            currStepInfo.rxSliceIdxs.push_back(rxSliceIdx);

            HCCL_DEBUG("[InsTempReduceScatterNhrDpu][GetStepInfoList] step[%u] i[%u] txSliceIdx[%u] rxSliceIdx[%u]",
                step, i, txSliceIdx, rxSliceIdx);

            txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
            rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterNhrDpu::DPUKernelRun(const TemplateDataParams& tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
#ifndef AICPU_COMPILE
    myRank_ = myRank;
    templateRankSize_ = subCommRanks[0].size();
    subCommRanks_ = subCommRanks;

    // 从 tempAlgParams.root 获取当前 step（ReduceScatter 不需要 root）
    u32 currentStep = static_cast<u32>(tempAlgParams.root);

    // 只获取当前 step 的信息（避免冗余）
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    AicpuNHRStepInfo st;
    CHK_RET(GetStepInfo(currentStep, nSteps, st));

    const u32 recvFromRank = subCommRanks_[0][st.fromRank];
    const u32 sendToRank = subCommRanks_[0][st.toRank];

    if (channels.count(recvFromRank) == 0 || channels.count(sendToRank) == 0) {
        HCCL_ERROR("[InsTempReduceScatterNhrDpu][DPUKernelRun] link missing: recvFrom=%u sendTo=%u",
            recvFromRank, sendToRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo &linkRecv = channels.at(recvFromRank)[0];
    const ChannelInfo &linkSend = channels.at(sendToRank)[0];

    DpuTransferCtx ctx;
    ctx.txCh = &linkSend;
    ctx.rxCh = &linkRecv;

    const u64 rptNum = std::max<u64>(1, tempAlgParams.repeatNum);
    ctx.txSrcSlices.reserve(st.nSlices * rptNum);
    ctx.txDstSlices.reserve(st.nSlices * rptNum);
    ctx.rxSrcSlices.reserve(st.nSlices * rptNum);
    ctx.rxDstSlices.reserve(st.nSlices * rptNum);

    void* sendRemoteCclBuffAddr = linkSend.remoteCclMem.addr;
    void* recvRemoteCclBuffAddr = linkRecv.remoteCclMem.addr;

    for (u64 rpt = 0; rpt < rptNum; ++rpt) {
        const u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff
                              + rpt * tempAlgParams.outputRepeatStride;

        for (u32 i = 0; i < st.nSlices; ++i) {
            const u32 txIdx = st.txSliceIdxs[i];
            const u32 rxIdx = st.rxSliceIdxs[i];

            // TX: 发送本地 cclBuff[txIdx] 到远程 cclBuff[txIdx]
            DataSlice txSrcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                scratchBase + tempAlgParams.sliceSize * txIdx,
                tempAlgParams.sliceSize, tempAlgParams.count);
            DataSlice txDstSlice = DataSlice(
                sendRemoteCclBuffAddr,
                scratchBase + tempAlgParams.sliceSize * txIdx,
                tempAlgParams.sliceSize, tempAlgParams.count);

            // RX: 从远程接收数据到本地 cclBuff[rxIdx + templateRankSize_]（后半部分）
            // 这样避免覆盖本地数据，为后续规约做准备
            DataSlice rxSrcSlice = DataSlice(
                recvRemoteCclBuffAddr,
                scratchBase + tempAlgParams.sliceSize * rxIdx,
                tempAlgParams.sliceSize, tempAlgParams.count);
            DataSlice rxDstSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                scratchBase + tempAlgParams.sliceSize * (rxIdx + templateRankSize_),  // 后半部分
                tempAlgParams.sliceSize, tempAlgParams.count);

            ctx.txSrcSlices.push_back(txSrcSlice);
            ctx.txDstSlices.push_back(txDstSlice);
            ctx.rxSrcSlices.push_back(rxSrcSlice);
            ctx.rxDstSlices.push_back(rxDstSlice);
        }
    }

    std::vector<DpuTransferCtx> pairs = {ctx};
    CHK_RET(DpuBatchTransfer(pairs));

    HCCL_INFO("[InsTempReduceScatterNhrDpu][DPUKernelRun] step[%u] toRank[%u] fromRank[%u] nSlices[%u], "
        "remote data written to cclBuff[idx + %u]",
        currentStep, sendToRank, recvFromRank, st.nSlices, templateRankSize_);
#endif
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
REGISTER_TEMPLATE_V2("InsTempReduceScatterNhrDpu", InsTempReduceScatterNhrDpu);
#endif

} // namespace ops_hccl