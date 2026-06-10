/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "ccu_kernel_gather_omnipipe_nhr1d_mem2mem.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID   = 0;
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int STEP_SYNC_ID = 3;
constexpr int POST_SYNC_ID   = 4;
constexpr int CKE_IDX_0     = 0;

CcuKernelGatherOmniPipeNHR1DMem2Mem::CcuKernelGatherOmniPipeNHR1DMem2Mem(const hcomm::CcuKernelArg& arg)
    : CcuKernelAlgBase(arg)
{
    HCCL_INFO("[CcuKernelGatherOmniPipeNHR1DMem2Mem] STARTS");
    const CcuKernelArgGatherOmniPipeNHR1DMem2Mem* kernelArg =
        dynamic_cast<const CcuKernelArgGatherOmniPipeNHR1DMem2Mem*>(&arg);
    rankId_        = kernelArg->rankId_;
    rankSize_      = kernelArg->dimSize_;
    channels_      = kernelArg->channels;
    rootId_        = kernelArg->rootId_;
    subCommRanks_  = kernelArg->subCommRanks_;
    userRank_      = subCommRanks_[0][rankId_];
    dataType_      = kernelArg->opParam_.DataDes.dataType;

    stepInfoVector_    = kernelArg->stepInfoVector_;
    rank2ChannelIdx_   = kernelArg->rank2ChannelIdx_;
    localSize_         = rank2ChannelIdx_.size(); // nhr 算法通信rank数
    myRankIdx_         = rank2ChannelIdx_.size(); // InitResources中将本端放在末尾，此处为对应的idx
    subRankIdx2RankIdx_   = kernelArg->subRankIdx2RankIdx_;

    HCCL_INFO("[CcuKernelGatherOmniPipeNHR1DMem2Mem] Init, KernelArgs are rankId[%u], "
        "rankSize_[%u], dataType[%d], rootId[%u], localSize_[%d] myRankIdx_[%u] ",
        rankId_, rankSize_, dataType_, rootId_, localSize_, myRankIdx_);
}

HcclResult CcuKernelGatherOmniPipeNHR1DMem2Mem::InitResource()
{
    HCCL_DEBUG("[%s] start", __func__);
    HCCL_DEBUG("[%s] start, channels_ is [%u]", __func__, channels_.size());
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelGatherOmniPipeNHR1DMem2Mem] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    for (uint32_t channelIdx = 0; channelIdx < localSize_; channelIdx++) {
        CcuRep::Variable inputVar, scratchVar, tokenVar;
        CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
        input_.push_back(inputVar);
        CHK_RET(CreateVariable(channels_[channelIdx], SCRATCH_XN_ID, &scratchVar));
        scratch_.push_back(scratchVar);
        CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
        token_.push_back(tokenVar);
    }

    // 本端的input和token放在最后
    input_.push_back(CreateVariable());
    scratch_.push_back(CreateVariable());
    token_.push_back(CreateVariable());

    output_ = CreateVariable();
    sliceSize_ = CreateVariable();
    inputOmniPipeSliceStride_ = CreateVariable();
    outputOmniPipeSliceStride_ = CreateVariable();
    localCopyFlag_ = CreateVariable();
    isStepOne_ = CreateVariable();
    isLastStep_ = CreateVariable();
    ifNewRoot_ = CreateVariable();
    for (uint32_t peerId = 0; peerId < rankSize_; peerId++) {
        inputOmniSliceStrideVec_.push_back(CreateVariable());
    }

    event_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelGatherOmniPipeNHR1DMem2Mem::LoadArgs()
{
    Load(input_[myRankIdx_]);
    Load(output_);
    Load(scratch_[myRankIdx_]);
    Load(token_[myRankIdx_]);
    Load(localCopyFlag_);
    Load(sliceSize_);
    Load(inputOmniPipeSliceStride_);
    Load(outputOmniPipeSliceStride_);
    Load(isStepOne_);
    Load(isLastStep_);
    Load(ifNewRoot_);
    for (int i = 0; i < rankSize_; i++) {
        Load(inputOmniSliceStrideVec_[i]);
    }
}

void CcuKernelGatherOmniPipeNHR1DMem2Mem::PreSync()
{
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << SCRATCH_XN_ID |1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[myRankIdx_], 1 << INPUT_XN_ID); // 同步并置位远端
        NotifyRecord(channel, CKE_IDX_0, SCRATCH_XN_ID, scratch_[myRankIdx_], 1 << SCRATCH_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[myRankIdx_], 1 << TOKEN_XN_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    return;
}

void CcuKernelGatherOmniPipeNHR1DMem2Mem::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}


std::vector<uint64_t> CcuKernelGatherOmniPipeNHR1DMem2Mem::GeneArgs(const hcomm::CcuTaskArg& arg)
{
    const CcuTaskArgGatherOmniPipeNHR1DMem2Mem *taskArg
        = dynamic_cast<const CcuTaskArgGatherOmniPipeNHR1DMem2Mem *>(&arg);
    uint64_t inputAddr                 = taskArg->inputAddr_;
    uint64_t outputAddr                = taskArg->outputAddr_;
    uint64_t scratchAddr               = taskArg->scratchAddr_;
    uint64_t token                     = taskArg->token_;
    uint64_t localCopyFlag             = taskArg->localCopyFlag_;
    uint64_t sliceSize                 = taskArg->sliceSize_;
    uint64_t inputOmniPipeSliceStride  = taskArg->inputOmniPipeSliceStride_;
    uint64_t outputOmniPipeSliceStride = taskArg->outputOmniPipeSliceStride_;
    uint64_t isStepOne                 = taskArg->isStepOne_;
    uint64_t isLastStep                = taskArg->isLastStep_;
    uint64_t ifNewRoot                 = taskArg->ifNewRoot_;
    std::vector<uint64_t> inputOmniSliceStrideVec = taskArg->inputOmniSliceStrideVec_;

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, scratchAddr, token, localCopyFlag, sliceSize, 
                                    inputOmniPipeSliceStride, outputOmniPipeSliceStride, isStepOne, isLastStep, ifNewRoot};
    taskArgs.insert(taskArgs.end(), inputOmniSliceStrideVec.begin(), inputOmniSliceStrideVec.end());

    HCCL_INFO("[CcuTaskArgGatherOmniPipeNHR1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
            "scratchAddr[%llu], sliceSize[%llu],"
            "inputOmniPipeSliceStride_[%llu], outputOmniPipeSliceStride_[%llu],"
            "localCopyFlag[%llu], isStepOne[%d], isLastStep[%d], ifNewRoot[%d]",
        inputAddr, outputAddr, scratchAddr, sliceSize,
        inputOmniPipeSliceStride, outputOmniPipeSliceStride,
        localCopyFlag, isStepOne, isLastStep, ifNewRoot);
    return taskArgs;
}

HcclResult CcuKernelGatherOmniPipeNHR1DMem2Mem::Algorithm()
{
    CHK_RET(InitResource());
    LoadArgs();
    PreSync();
    DoGatherOmniPipeNHR();
    PostSync();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelGatherOmniPipeNHR1DMem2Mem::DoGatherOmniPipeNHR()
{
    for (auto& nhrStepInfo : stepInfoVector_) {
        DoGatherOmniPipeNHRSingleStep(nhrStepInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelGatherOmniPipeNHR1DMem2Mem::DoGatherOmniPipeNHRSingleStep(const NHRStepInfo& nhrStepInfo)
{
    CcuRep::RemoteAddr src = CreateRemoteAddr();
    CcuRep::LocalAddr dst = CreateLocalAddr();
    u32                    &toRankIdx        = rank2ChannelIdx_[nhrStepInfo.toRank];
    u32                    &fromRankIdx      = rank2ChannelIdx_[nhrStepInfo.fromRank];
    ChannelHandle          &sendChannel      = channels_[toRankIdx];
    ChannelHandle          &recvChannel      = channels_[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs; // 发送
    const std::vector<u32> &recvSliceIdxList = nhrStepInfo.rxSliceIdxs; // 接受

    // 发送端：先通知接收方可以读取自己的数据
    if (sendSliceIdxList.size() != 0) {
        u32& toRankIdx = rank2ChannelIdx_[nhrStepInfo.toRank];
        ChannelHandle sendChannel = channels_[toRankIdx];
        NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_SYNC_ID);
    }

    if (recvSliceIdxList.size() != 0) {
        u32& fromRankIdx  = rank2ChannelIdx_[nhrStepInfo.fromRank];
        u32  recvSliceIdx = 0;
        ChannelHandle recvChannel        = channels_[fromRankIdx];
        src.token                        = token_[myRankIdx_];
        dst.token                        = token_[fromRankIdx];

        NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_SYNC_ID);
        u32 recvSliceIdxSize = recvSliceIdxList.size();
        for (u32 i = 0; i < recvSliceIdxSize; i++) {
            recvSliceIdx = recvSliceIdxList[i];
            if (nhrStepInfo.fromRank == recvSliceIdx) {
                src.addr = input_[fromRankIdx];
            } else {
                src.addr = scratch_[fromRankIdx];
            }
            src.addr += inputOmniSliceStrideVec_[recvSliceIdx];

            dst.addr = output_;
            dst.addr += inputOmniSliceStrideVec_[recvSliceIdx];

            event_.SetMask(1 << i);
            CCU_IF(sliceSize_ != 0) {
                ReadNb(recvChannel, dst, src, sliceSize_, event_);
            }
            CCU_IF(sliceSize_ == 0) {
                RecordEvent(event_);
            }
        }
        event_.SetMask((1 << recvSliceIdxSize) - 1);
        WaitEvent(event_);
    }

    HCCL_INFO("[DoGatherOmniPipeNHRSingleStep] step %u, toRank=%u, fromRank=%u, sendSliceNum=%lu",
        nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl