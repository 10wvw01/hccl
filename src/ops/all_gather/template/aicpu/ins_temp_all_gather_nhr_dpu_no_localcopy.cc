/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_nhr_dpu_no_localcopy.h"
#include "dpu_alg_nhr_opt_wrapper.h"
#include "alg_data_trans_wrapper.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "channel.h"
#include "alg_v2_template_register.h"

namespace ops_hccl {
InsTempAllGatherNHRDPUNoLocalCopy::InsTempAllGatherNHRDPUNoLocalCopy(const OpParam& param, const uint32_t rankId,
                                               const std::vector<std::vector<uint32_t>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks) {}

HcclResult InsTempAllGatherNHRDPUNoLocalCopy::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                           AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO("[InsTempAllGatherNHRDPUNoLocalCopy][CalcRes]slaveThreadNum[%u] notifyNumOnMainThread[%u] level1Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, level1Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherNHRDPUNoLocalCopy::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void) inBufferType;
    (void) outBufferType;
    u64 scratchMultiple = templateRankSize_;
    HCCL_INFO(
        "[InsTempAllGatherNHRDPUNoLocalCopy][CalcScratchMultiple] templateScratchMultiplier[%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempAllGatherNHRDPUNoLocalCopy::KernelRun(const OpParam& param,
                                             const TemplateDataParams& tempAlgParams,
                                             TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherNHRDPUNoLocalCopy] Run Start");

    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempAllGatherNHRDPUNoLocalCopy] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // 去掉LocalDataCopy步骤：由executor在调用前自行完成input→cclBuff拷贝

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempAllGatherNHRDPUNoLocalCopy] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
        HCCL_ERROR("[InsTempAllGatherNHRDPUNoLocalCopy] HcommThreadSynchronize failed");
        return HCCL_E_INTERNAL;
    }

    DPURunInfo dpuRunInfo;
    dpuRunInfo.templateName = "InsTempAllGatherNHRDPUNoLocalCopy";
    dpuRunInfo.tempAlgParams = tempAlgParams;
    dpuRunInfo.channels = templateResource.channels;
    dpuRunInfo.myRank = myRank_;
    dpuRunInfo.subCommRanks = subCommRanks_;
    auto dpuRunInfoSeqData = dpuRunInfo.Serialize();

    u32 sendMsgId = 0;
    if (HcommSendRequest(reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
        static_cast<void*>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId) != 0) {
        HCCL_ERROR("[InsTempAllGatherNHRDPUNoLocalCopy] HcommSendRequest failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempAllGatherNHRDPUNoLocalCopy] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);

    // 等待DPU数据传输，然后回写结果回来
    void *recvData = nullptr;
    u32 recvMsgId = 0;
    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("[InsTempAllGatherNHRDPUNoLocalCopy] HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempAllGatherNHRDPUNoLocalCopy] HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR("[InsTempAllGatherNHRDPUNoLocalCopy] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
        return HCCL_E_INTERNAL;
    }

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempAllGatherNHRDPUNoLocalCopy] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    // 去掉PostLocalCopy步骤：由executor在调用后自行完成cclBuff→output拷贝

    HCCL_INFO("[InsTempAllGatherNHRDPUNoLocalCopy] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRDPUNoLocalCopy::DPUKernelRun(const TemplateDataParams& tempAlgParams,
                                                const std::map<u32, std::vector<ChannelInfo>>& channels,
                                                const u32 myRank,
                                                const std::vector<std::vector<uint32_t>>& subCommRanks)
{
    myRank_ = myRank;
    templateRankSize_ = subCommRanks[0].size();
    subCommRanks_ = subCommRanks;
    CHK_RET(RunNHR(tempAlgParams, channels));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRDPUNoLocalCopy::GetStepInfo(uint32_t step, uint32_t nSteps, AicpuNHRStepInfo &stepInfo) const
{
    uint32_t rankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx));
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();

    // 计算通信对象
    uint32_t deltaRank = 1 << (nSteps - 1 - step);
    uint32_t recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    uint32_t sendTo = (rankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    uint32_t nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    uint32_t deltaSliceIndex = 1 << (nSteps - step);
    uint32_t txSliceIdx = rankIdx;
    uint32_t rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.fromRank = recvFrom;
    stepInfo.toRank = sendTo;

    for (uint32_t i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG("[InsTempAllGatherNHRDPUNoLocalCopy][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempAllGatherNHRDPUNoLocalCopy::GetRankFromMap(const uint32_t rankIdx) const
{
    return subCommRanks_[0].at(rankIdx);
}

HcclResult InsTempAllGatherNHRDPUNoLocalCopy::RunNHR(const TemplateDataParams& tempAlgParams,
                                          const std::map<u32, std::vector<ChannelInfo>>& channels) const
{
#ifndef AICPU_COMPILE
    const uint32_t nSteps = GetNHRStepNum(templateRankSize_);

    for (uint32_t rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        for (uint32_t step = 0; step < nSteps; ++step) {
            AicpuNHRStepInfo stepInfo;
            CHK_RET(GetStepInfo(step, nSteps, stepInfo));

            HCCL_DEBUG("[InsTempAllGatherNHRDPUNoLocalCopy] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u]",
                myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

            stepInfo.toRank = GetRankFromMap(stepInfo.toRank);
            stepInfo.fromRank = GetRankFromMap(stepInfo.fromRank);
            HCCL_INFO("[InsTempAllGatherNHRDPUNoLocalCopy][RunNHR] converted toRank=%u fromRank=%u channelsSize=%zu",
                stepInfo.toRank, stepInfo.fromRank, channels.size());
            CHK_RET(BatchTransferNHR(stepInfo, channels, tempAlgParams, rpt, myRank_, templateRankSize_));
        }
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempAllGatherNHRDPUNoLocalCopy", InsTempAllGatherNHRDPUNoLocalCopy);
}