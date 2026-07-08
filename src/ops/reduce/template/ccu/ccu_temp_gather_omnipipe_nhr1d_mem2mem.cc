/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "channel.h"
#include "ccu_kernel_gather_omnipipe_nhr1d_mem2mem.h"
#include "ccu_temp_gather_omnipipe_nhr1d_mem2mem.h"
#include "alg_data_trans_wrapper.h" 
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempGatherOmniPipeNHR1DMem2Mem::CcuTempGatherOmniPipeNHR1DMem2Mem(const OpParam& param, const u32 rankId,
                                                                        const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }

    // 子通信域的root卡号
    auto itRoot = std::find(ranks.begin(), ranks.end(), param.root);
    if (itRoot != ranks.end()) {
        subCommRootId_ = std::distance(ranks.begin(), itRoot);
    }
    ifRealRoot_ = (rankId == param.root);
}

CcuTempGatherOmniPipeNHR1DMem2Mem::~CcuTempGatherOmniPipeNHR1DMem2Mem()
{
}

void CcuTempGatherOmniPipeNHR1DMem2Mem::SetRoot(u32 root)
{
    HCCL_INFO("[CcuTempGatherOmniPipeNHR1DMem2Mem][SetRoot] myRank_ [%u], set root [%u] ", myRank_, root);
    std::string ranksStr = "";
    std::vector<u32> ranks = subCommRanks_[0];
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
    for (auto r : ranks) { ranksStr += std::to_string(r) + ", "; }
    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] subCommRanks[%s] subCommRootId_[%d]",
        __func__, myRank_, mySubCommRank_,  ranksStr.c_str(), subCommRootId_);
}

void CcuTempGatherOmniPipeNHR1DMem2Mem::UnsetRoot(u32 rank)
{
    HCCL_INFO("[CcuTempGatherOmniPipeNHR1DMem2Mem][UnsetRoot] myRank_ [%u], unset root [%u] ", myRank_, rank);
    if (!ifRealRoot_) {
        subCommRootId_ = UINT32_MAX;
    }
}


u64 CcuTempGatherOmniPipeNHR1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempGatherOmniPipeNHR1DMem2Mem::GetRes(AlgResourceRequest &resourceRequest) const
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);

    return HCCL_SUCCESS;
}

HcclResult CcuTempGatherOmniPipeNHR1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param,
                                                        const TopoInfoWithNetLayerDetails* topoInfo,
                                                        AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    GetRes(resourceRequest);
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempGatherOmniPipeNHR1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    CHK_SAFETY_FUNC_RET(strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuGatherOmniPipeNHR1DMem2MemKernel"));
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuGatherOmniPipeNHR1DMem2MemKernel);

    std::vector<HcclChannelDesc> channelDescs;

    // NHR
    CHK_RET(CalcChannelRequestNhrMultiJetty(comm, param, topoInfo, subCommRanks_, channelDescs));
    for (auto channel : channelDescs) {
        HCCL_DEBUG("[%s] channel myrank[%u], remoteRank [%u]", __func__, myRank_,  channel.remoteRank);
        if (channel.channelProtocol != COMM_PROTOCOL_UBC_CTP) {
            HCCL_ERROR("[%s] channelProtocol: %u", __func__, channel.channelProtocol);
            return HCCL_E_INTERNAL;
        }
    }

    std::vector<NHRStepInfo>     stepInfoVector; 
    std::map<u32, u32>           rank2ChannelIdx; // rankId和channel匹配
    for (u32 i = 0; i < channelDescs.size(); ++i) {
        u32 remoteRank = channelDescs[i].remoteRank;
        u32 subRankIdx = RemoteRankId2RankId(remoteRank);
        rank2ChannelIdx[subRankIdx] = i;
    }

    CHK_RET(CalcNHRInfo(stepInfoVector));
    kernelInfo.channels = channelDescs;

    auto kernelArg = std::make_shared<CcuKernelArgGatherOmniPipeNHR1DMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->rootId = subCommRootId_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelArg->myrealrank = myRank_;
    kernelArg->stepInfoVector = stepInfoVector;
    kernelArg->rank2ChannelIdx = rank2ChannelIdx;
    
    kernelInfo.setKernelArg(kernelArg);
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[%s] ------------------zq---------------------", __func__);
    HCCL_DEBUG("[%s]channelDescs.size()=%llu, dimsize=%llu, ccuKernelInfos.size()=%llu", __func__,
        channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());
    return HcclResult::HCCL_SUCCESS;
}


HcclResult CcuTempGatherOmniPipeNHR1DMem2Mem::KernelRun(const OpParam& param,
                                                          const TemplateDataParams& templateDataParams,
                                                          TemplateResource& templateResource)
{
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempGatherOmniPipeNHR1DMem2Mem::KernelRun] start1");

    buffInfo_ = templateDataParams.buffInfo;
    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);
    uint64_t inBuffBaseOff = buffInfo_.inBuffBaseOff;
    uint64_t outBuffBaseOff = buffInfo_.outBuffBaseOff;
    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
    uint64_t scratchAddr = PointerToAddr(buffInfo_.hcclBuff.addr) +  inBuffBaseOff;

    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    if (localCopyFlag == 0) {
        uint64_t sliceSize;
        uint64_t inputOmniPipeSliceStride;
        uint64_t outputOmniPipeSliceStride;
        uint64_t repeatNum = stepSliceInfo.stepSliceSize[0].size();
        for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
            sliceSize = stepSliceInfo.stepSliceSize[0][rpt];//peerId为0,默认发送大小都一致
            // if (subCommRootId_ == UINT32_MAX) {
            //     sliceSize = 0;
            //     HCCL_DEBUG("[CcuTempGatherOmniPipeNHR1DMem2Mem::KernelRun] stepSliceSize is zero");
            // }
            inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_][rpt];
            outputOmniPipeSliceStride = stepSliceInfo.outputOmniPipeSliceStride[mySubCommRank_][rpt];
            // bool ifNewRoot = (subCommRootId_ == mySubCommRank_);
            std::vector<uint64_t> inputOmniSliceStrideVec = {};
            std::vector<uint64_t> outputOmniSliceStrideVec = {};
            if (ifDoTask_) {
                for (uint32_t ridx = 0; ridx < templateRankSize_; ridx++) {
                    uint64_t inputOmniSliceStrideTmp = stepSliceInfo.inputOmniPipeSliceStride[ridx][rpt];
                    uint64_t outputOmniSliceStrideTmp = stepSliceInfo.outputOmniPipeSliceStride[ridx][rpt];
                    inputOmniSliceStrideVec.push_back(inputOmniSliceStrideTmp);
                    outputOmniSliceStrideVec.push_back(outputOmniSliceStrideTmp);
                    HCCL_INFO("[zjq checkSliceStride] myrank:%d,mySubCommRank_:%d, ridx:%d,rpt:%d,inputOmniSliceStrideTmp:%d", myRank_, mySubCommRank_,
                                ridx,rpt, inputOmniSliceStrideTmp);
                }  
            }else {
                sliceSize = 0;
                for (uint32_t ridx = 0; ridx < templateRankSize_; ridx++) {
                    inputOmniSliceStrideVec.push_back(0);
                    outputOmniSliceStrideVec.push_back(0);
                }
            }

            std::vector<uint64_t> taskArgs = {
                inputAddr, 
                outputAddr,
                scratchAddr,
                token, 
                localCopyFlag, 
                sliceSize, 
                inputOmniPipeSliceStride, 
                outputOmniPipeSliceStride, 
                isStepOne_, 
                isLastStep_
            };
            taskArgs.insert(taskArgs.end(), inputOmniSliceStrideVec.begin(), inputOmniSliceStrideVec.end());
            taskArgs.insert(taskArgs.end(), outputOmniSliceStrideVec.begin(), outputOmniSliceStrideVec.end());
            uint64_t argSize = taskArgs.size();
            CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), argSize);
            if (launchRet != CCU_SUCCESS) {
                HCCL_ERROR("[%s] myRank[%u] HcommCcuKernelLaunch failed, ccuRet is:[%d]", __func__, myRank_, launchRet);
                return ConvertCcuToHccl(launchRet);
            }
        }
    }
    else if (localCopyFlag == 1) {
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy start", __func__, myRank_);
        DataSlice srcSlice(buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        DataSlice dstSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy inputAddrBase[%llu] inputAddrOffset[%llu] outputAddrBase[%llu]"
                   "outputAddrOffset[%llu] sliceSize[%llu]",
            __func__, myRank_, inputAddrBase, buffInfo_.inBuffBaseOff, outputAddrBase, buffInfo_.outBuffBaseOff,
            templateDataParams.sliceSize);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy end", __func__, myRank_);
    }

    HCCL_DEBUG("[CcuTempGatherOmniPipeNHR1DMem2Mem::KernelRun] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempGatherOmniPipeNHR1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

HcclResult CcuTempGatherOmniPipeNHR1DMem2Mem::CalcNHRInfo(std::vector<NHRStepInfo> &stepInfoVector) const
{
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 CcuTempGatherOmniPipeNHR1DMem2Mem::GetNHRStepNum(u32 rankSize) const
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

uint32_t CcuTempGatherOmniPipeNHR1DMem2Mem::RemoteRankId2RankId(const uint32_t remoteRankId) const
{
    uint32_t subCommRankId = 0;
    std::vector<u32> ranks = subCommRanks_[0];
    auto it = std::find(ranks.begin(), ranks.end(), remoteRankId);
    if (it != ranks.end()) {
        subCommRankId = std::distance(ranks.begin(), it);
    }
    return subCommRankId;
}

HcclResult CcuTempGatherOmniPipeNHR1DMem2Mem::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo) const
{
    u32 virtRankIdx = mySubCommRank_;
    std::vector<u32> ranks = subCommRanks_[0];
    for (auto rank: ranks) { // [0 2] [1 3] [0 2] [1 3] TODO 删除
        HCCL_DEBUG("GetStepInfo rank is [%u]", rank);
    }

    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.toRank = templateRankSize_;
    stepInfo.fromRank = templateRankSize_;
    stepInfo.step = step;
    stepInfo.myRank = virtRankIdx;
    uint32_t rootId = subCommRootId_;
    u32 deltaRoot = (rootId + templateRankSize_ - virtRankIdx) % templateRankSize_;
    // Gather: 用 nSteps-1-step
    u32 deltaRankPair = 1 << (nSteps - 1 - step);
    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    // 是否为2的幂
    u32 nRanks = 0;
    bool isPowerOfTwo = (templateRankSize_ & (templateRankSize_ - 1)) == 0;
    if (!isPowerOfTwo && step == nSteps - 1) {
        nRanks = templateRankSize_ - (1 << (nSteps - 1 - step));
    } else {
        nRanks = deltaRankPair;
    }

    if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) {
        u32 sendTo = (virtRankIdx + deltaRankPair) % templateRankSize_;
        u32 txSliceIdx = virtRankIdx;
        for (u32 i = 0; i < nSlices; i++) {
            stepInfo.txSliceIdxs.push_back(txSliceIdx);
            HCCL_DEBUG("GetStepInfo [%s] step[%u] myRank[%u] mySubCommRank[%u] txSliceIdx[%u] sendTo[%u] slice-i[%u]",
                __func__, step, myRank_, mySubCommRank_, txSliceIdx, sendTo, i);
            txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
        stepInfo.toRank = sendTo; // TODO ranks[sendTo];
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot < nRanks) {
        u32 recvFrom = (virtRankIdx + templateRankSize_ - deltaRankPair) % templateRankSize_;
        u32 rxSliceIdx = recvFrom;
        for (u32 i = 0; i < nSlices; i++) {
            stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
            HCCL_DEBUG("GetStepInfo [%s] step[%u] myRank[%u] mySubCommRank[%u] rxSliceIdx[%u] recvFrom[%u] slice-i[%u]",
                __func__, step, myRank_, mySubCommRank_, rxSliceIdx, recvFrom, i);
            rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
        stepInfo.fromRank = recvFrom; // TODO ranks[recvFrom];
        stepInfo.nSlices = nSlices;
    }

    HCCL_DEBUG("[%s] myRank[%u] StepInfo step[%u] nSteps[%u] nSlices[%u] fromRank[%u] toRank[%u] subRoot[%u] subCommRootId_[%u]", __func__, myRank_, step , nSteps, stepInfo.nSlices, stepInfo.fromRank, stepInfo.toRank, subCommRootId_, subCommRootId_);
    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl