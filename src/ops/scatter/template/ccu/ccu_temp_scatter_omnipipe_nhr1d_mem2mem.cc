/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "channel.h"
#include "ccu_kernel_scatter_omnipipe_nhr1d_mem2mem.h"
#include "ccu_temp_scatter_omnipipe_nhr1d_mem2mem.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempScatterOmniPipeNHR1DMem2Mem::CcuTempScatterOmniPipeNHR1DMem2Mem(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    templateRankSize_ = ranks.size();
    auto itRoot = std::find(ranks.begin(), ranks.end(), param.root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
    ifRealRoot_ = (rankId == param.root);
    std::string ranksStr = "";
    for (auto r : ranks) { ranksStr += std::to_string(r) + ", "; }
    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] subCommRanks[%s] templateRankSize[%u] subCommRootId_[%d] ifRealRoot_[%d]",
        __func__, rankId, mySubCommRank_, ranksStr.c_str(), templateRankSize_, subCommRootId_, ifRealRoot_);
}

CcuTempScatterOmniPipeNHR1DMem2Mem::~CcuTempScatterOmniPipeNHR1DMem2Mem()
{
}

void CcuTempScatterOmniPipeNHR1DMem2Mem::SetRoot(u32 root)
{
    HCCL_DEBUG("[CcuTempScatterOmniPipeNHR1DMem2Mem][SetRoot] myRank_ [%u], set root [%u] ", myRank_, root);
    std::vector<u32> ranks = subCommRanks_[0];
    std::string ranksStr = "";
    for (auto r : ranks) { ranksStr += std::to_string(r) + ", "; }
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] subCommRanks[%s] subCommRootId_[%d]",
        __func__, myRank_, mySubCommRank_,  ranksStr.c_str(), subCommRootId_);
}

void CcuTempScatterOmniPipeNHR1DMem2Mem::UnsetRoot(u32 rank)
{
    HCCL_DEBUG("[CcuTempScatterOmniPipeNHR1DMem2Mem][UnsetRoot] myRank_ [%u], unset root [%u] ", myRank_, rank);
    if (!ifRealRoot_) {
        subCommRootId_ = 1000;
    }
}

u64 CcuTempScatterOmniPipeNHR1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    GetRes(resourceRequest);
    resourceRequest.ccuKernelNum.push_back(1);

    HCCL_DEBUG("[%s]notifyNumOnMainThread[%u] slaveThreadNum[%u]", __func__,
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    CHK_SAFETY_FUNC_RET(strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuScatterOmniPipeNHR1DMem2MemKernel"));
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuScatterOmniPipeNHR1DMem2MemKernel);
    std::vector<HcclChannelDesc> channelDescs;
    // CommTopo priorityTopo = COMM_TOPO_CLOS;
    // CHK_RET(CalcChannelRequestNHRWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, priorityTopo));
    CHK_RET(CalcChannelRequestNhrMultiJetty(comm, param,topoInfo, subCommRanks_, channelDescs));
    for (auto channel : channelDescs) {
        HCCL_DEBUG("[%s] channel myrank[%u], remoteRank [%u]", __func__, myRank_,  channel.remoteRank);
        if (channel.channelProtocol != COMM_PROTOCOL_UBC_CTP) {
            HCCL_ERROR("[%s] channelProtocol: %u", __func__, channel.channelProtocol);
            return HCCL_E_INTERNAL;
        }
    }

    HCCL_DEBUG("[%s] Get Clos Channel Success!", __func__);
    std::vector<NHRStepInfo>     stepInfoVector; 
    std::map<u32, u32>           rank2ChannelIdx; // rankId和channel匹配
    std::map<u32, u32>           subRankIdx2RankIdx;
    for (u32 i = 0; i < channelDescs.size(); ++i) {
        u32 remoteRank = channelDescs[i].remoteRank;
        u32 subRankIdx = RemoteRankId2RankId(remoteRank);
        rank2ChannelIdx[subRankIdx] = i;
        subRankIdx2RankIdx[subRankIdx] = remoteRank;
        HCCL_DEBUG("[%s] channel myrank[%u], mySubCommRank_[%u],  rank2ChannelIdx[%u]=%u, subRankIdx2RankIdx[%u]=%u ", __func__, myRank_,  mySubCommRank_, 
                subRankIdx , i, subRankIdx, remoteRank);
    }
    CHK_RET(CalcNHRInfo(stepInfoVector));

    auto kernelArg = std::make_shared<CcuKernelArgScatterOmniPipeNHR1DMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->rootId = subCommRootId_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelArg->stepInfoVector = stepInfoVector;
    kernelArg->ifRealRoot = ifRealRoot_;
    kernelArg->myrealrank = myRank_;
    kernelArg->rank2ChannelIdx = rank2ChannelIdx;

    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    HCCL_DEBUG("[%s]channelDescs.size()=%llu, dimsize=%llu, ccuKernelInfos.size()=%llu", __func__,
        channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::KernelRun(
    const OpParam &param, const TemplateDataParams &templateDataParams, TemplateResource &templateResource)
{
    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    HCCL_DEBUG("[%s] myRank[%u] mySubCommRank_[%u] isStepone[%d] isLastStep[%d] localCopyFlag[%d] start", __func__, myRank_,
        mySubCommRank_, isStepOne_, isLastStep_, localCopyFlag);
    buffInfo_ = templateDataParams.buffInfo;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);

    uint64_t inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    uint64_t outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    HCCL_DEBUG("buffInfo_.inputSize[%llu] buffInfo_.inputAddr[%llu] buffInfo_.inBuffType[%llu]", buffInfo_.inputSize,
        inputAddrBase, buffInfo_.inBuffType);
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    if (localCopyFlag == 0) {
        uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
        bool ifNewRoot = (subCommRootId_ == mySubCommRank_);

        uint64_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[myRank_% xRankSize_].size() / (templateRankSize_ - 1);
        HCCL_DEBUG("[%s] myRank[%u] mySubCommRank_[%u] subCommRootId_[%u] ifDoTask_[%d] isStepone[%d] isLastStep[%d] "
                   "repeatNum[%llu]",
            __func__, myRank_, mySubCommRank_, subCommRootId_, ifDoTask_, isStepOne_, isLastStep_, repeatNum);
        for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
            uint64_t sliceSize = 0;
            uint64_t inputOmniPipeSliceStride = 0;
            uint64_t outputOmniPipeSliceStride = 0;
            std::vector<uint64_t> inputOmniSliceStrideVec = {};
            std::vector<uint64_t> outputOmniSliceStrideVec = {};
            std::vector<uint64_t> inputOmniSliceSizeVec = {};
            if (ifDoTask_) {
                sliceSize = stepSliceInfo.stepSliceSize[myRank_ % xRankSize_][rpt]; //  TODO [TODO： 》》这块的对于关系 对齐下] 1、rpt 对应的大小，需要计算下，当前默认发的都是一样大小 2、xRankSize_传过来
                inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[myRank_% xRankSize_][rpt];
                outputOmniPipeSliceStride= stepSliceInfo.outputOmniPipeSliceStride[myRank_% xRankSize_][rpt];
                uint32_t originIndex = 0;
                for (uint32_t ridx = 0; ridx < templateRankSize_; ridx++) {
                    if (ridx == subCommRootId_) {
                        outputOmniSliceStrideVec.push_back(0);
                        inputOmniSliceStrideVec.push_back(0);
                        inputOmniSliceSizeVec.push_back(0);
                    } else {
                        HCCL_DEBUG("[checkSliceStride] myrank:%d,mySubCommRank_:%d, ridx:%d, tag0", myRank_, mySubCommRank_, ridx);
                        uint32_t sliceStrideiIndex = repeatNum * originIndex  + rpt;
                        originIndex = originIndex + 1;
                        HCCL_DEBUG("[checkSliceStride] myrank:%d,mySubCommRank_:%d, ridx:%d, tag2 sliceStrideiIndex:%d", myRank_, mySubCommRank_, ridx, sliceStrideiIndex);
                        uint64_t outputOmniSliceStrideTmp = stepSliceInfo.outputOmniPipeSliceStride[myRank_%xRankSize_][sliceStrideiIndex];
                        outputOmniSliceStrideVec.push_back(outputOmniSliceStrideTmp);
                        uint64_t inputOmniSliceStrideTmp = stepSliceInfo.inputOmniPipeSliceStride[myRank_%xRankSize_][sliceStrideiIndex];
                        inputOmniSliceStrideVec.push_back(inputOmniSliceStrideTmp);
                        inputOmniSliceSizeVec.push_back(stepSliceInfo.stepSliceSize[myRank_%xRankSize_][sliceStrideiIndex]); // TODO： 》》这块的对于关系 对齐下
                        HCCL_DEBUG("[checkSliceSize] myrank:%d,mySubCommRank_:%d, ridx:%d, stepSliceSize:%d", myRank_, mySubCommRank_, ridx, stepSliceInfo.stepSliceSize[myRank_%xRankSize_][sliceStrideiIndex]);
                        HCCL_DEBUG(
                            "[checkSliceStride] myrank:%d,mySubCommRank_:%d, "
                            "ridx:%d,sliceStrideiIndex:%d,outputOmniSliceStrideTmp:%llu, inputOmniSliceStrideTmp:%llu",
                            myRank_, mySubCommRank_, ridx, sliceStrideiIndex, outputOmniSliceStrideTmp,
                            inputOmniSliceStrideTmp);
                    }
                }
            } else {
                for (uint32_t ridx = 0; ridx < templateRankSize_; ridx++) {
                    outputOmniSliceStrideVec.push_back(0);
                    inputOmniSliceStrideVec.push_back(0);
                    inputOmniSliceSizeVec.push_back(0);
                }
            }

            for (uint32_t i = 0; i < outputOmniSliceStrideVec.size(); i++) {
                HCCL_DEBUG("myRank_[%u] subCommRootId_[%u] rpt[%u] sliceSize[%llu] outputOmniSliceStrideVec[%d] = %llu "
                          "isStepone[%d] isLastStep[%d]",
                    myRank_, subCommRootId_, rpt, sliceSize, i, outputOmniSliceStrideVec[i], isStepOne_, isLastStep_);
            }
            for (uint32_t i = 0; i < inputOmniSliceStrideVec.size(); i++) {
                HCCL_DEBUG("myRank_[%u] subCommRootId_[%u] rpt[%u] sliceSize[%llu] inputOmniSliceStrideVec[%d] = %llu "
                          "isStepone[%d] isLastStep[%d]",
                    myRank_, subCommRootId_, rpt, sliceSize, i, inputOmniSliceStrideVec[i], isStepOne_, isLastStep_);
            }

            for (uint32_t i = 0; i < inputOmniSliceSizeVec.size(); i++) {
                HCCL_DEBUG("myRank_[%u] subCommRootId_[%u] rpt[%u] sliceSize[%llu] inputOmniSliceSizeVec[%d] = %llu "
                          "isStepone[%d] isLastStep[%d]",
                    myRank_, subCommRootId_, rpt, sliceSize, i, inputOmniSliceSizeVec[i], isStepOne_, isLastStep_);
            }

            std::vector<uint64_t> taskArgs
                = {inputAddr, outputAddr, sliceSize, token, isStepOne_, isLastStep_, ifNewRoot};
            taskArgs.insert(taskArgs.end(), outputOmniSliceStrideVec.begin(), outputOmniSliceStrideVec.end());
            taskArgs.insert(taskArgs.end(), inputOmniSliceStrideVec.begin(), inputOmniSliceStrideVec.end());
            taskArgs.insert(taskArgs.end(), inputOmniSliceSizeVec.begin(), inputOmniSliceSizeVec.end());
            uint64_t argSize = taskArgs.size();
            HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] rpt[%u] argSize[%llu]", __func__, myRank_, mySubCommRank_, rpt, argSize);

            CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), argSize);
            if (launchRet != CCU_SUCCESS) {
                HCCL_ERROR("[%s] myRank[%u] HcommCcuKernelLaunch failed, ccuRet is:[%d]", __func__, myRank_, launchRet);
                return ConvertCcuToHccl(launchRet);
            }
        }
    } else if (localCopyFlag == 1) {
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy start", __func__, myRank_);
        DataSlice srcSlice(buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        DataSlice dstSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy inputAddrBase[%llu] inputAddrOffset[%llu] outputAddrBase[%llu] "
                "outputAddrOffset[%llu] sliceSize[%llu]",
            __func__, myRank_, inputAddrBase, buffInfo_.inBuffBaseOff, outputAddrBase, buffInfo_.outBuffBaseOff,
            templateDataParams.sliceSize);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy end", __func__, myRank_);
    }

    HCCL_DEBUG("[%s] run success", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::FastLaunch(const OpParam &param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    HCCL_DEBUG("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempScatterOmniPipeNHR1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::CalcNHRInfo(std::vector<NHRStepInfo> &stepInfoVector) const
{
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 CcuTempScatterOmniPipeNHR1DMem2Mem::GetNHRStepNum(u32 rankSize) const
{
    u32 nSteps = 0;
    if (rankSize == 0) {
        return 0;
    }
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[%s] rankSize[%u] nSteps[%u]", __func__, rankSize, nSteps);
    return nSteps;
}

uint32_t CcuTempScatterOmniPipeNHR1DMem2Mem::RemoteRankId2RankId(const uint32_t remoteRankId) const
{
    uint32_t subCommRankId = 0;
    std::vector<u32> ranks = subCommRanks_[0];
    auto it = std::find(ranks.begin(), ranks.end(), remoteRankId);
    if (it != ranks.end()) {
        subCommRankId = std::distance(ranks.begin(), it);
    }
    return subCommRankId;
}

HcclResult CcuTempScatterOmniPipeNHR1DMem2Mem::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo& stepInfo) const
{
    u32 virtRankIdx = mySubCommRank_;
    std::vector<u32> ranks = subCommRanks_[0];
    for (auto rank: ranks) { // [0 2] [1 3] [0 2] [1 3]
        HCCL_DEBUG("GetStepInfo rank is [%u]", rank);
    }
    HCCL_DEBUG("GetStepInfo templateRankSize_ is [%u] subCommRootId_ is [%u] virtRankIdx is [%u]", templateRankSize_,
        subCommRootId_, virtRankIdx);

    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.toRank = templateRankSize_;
    stepInfo.fromRank = templateRankSize_;
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;

    uint32_t rootId = subCommRootId_;
    u32 deltaRankPair = 1 << step;
    u32 deltaRoot = (rootId + templateRankSize_ - virtRankIdx) % templateRankSize_;

    u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);

    bool isPowerOfTwo = (templateRankSize_ & (templateRankSize_ - 1)) == 0;
    u32 nRanks = 0;
    if (!isPowerOfTwo && step == nSteps - 1) {
        nRanks = templateRankSize_ - deltaRankPair;
    } else {
        nRanks = deltaRankPair;
    }

    if (deltaRoot < nRanks) {
        u32 sendTo = (virtRankIdx + templateRankSize_ - deltaRankPair) % templateRankSize_;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            stepInfo.txSliceIdxs.push_back(txSliceIdx);
            HCCL_DEBUG("GetStepInfo [%s] step[%u] myRank[%u] mySubCommRank[%u] txSliceIdx[%u] sendTo[%u] slice-i[%u]",
            __func__, step, myRank_, mySubCommRank_, txSliceIdx, sendTo, i);
            txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
        stepInfo.toRank = sendTo; //ranks[sendTo]
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) {
        u32 recvFrom = (virtRankIdx + deltaRankPair) % templateRankSize_;
        u32 rxSliceIdx = virtRankIdx;
        for (u32 i = 0; i < nSlices; i++) {
            stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
            HCCL_DEBUG("GetStepInfo [%s] step[%u] myRank[%u] mySubCommRank[%u] rxSliceIdx[%u] recvFrom[%u] slice-i[%u]",
            __func__, step, myRank_, mySubCommRank_, rxSliceIdx, recvFrom, i);
            rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
        stepInfo.fromRank = recvFrom; //ranks[recvFrom]
        stepInfo.nSlices = nSlices;
    }

    HCCL_DEBUG("[%s] myRank[%u] StepInfo step[%u] nSteps[%u] nSlices[%u] fromRank[%u] toRank[%u] ", __func__, myRank_, step , nSteps, stepInfo.nSlices, stepInfo.fromRank, stepInfo.toRank);
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
