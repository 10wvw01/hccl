/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_omnipipe_nhr1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID       = 1;
constexpr uint16_t TOKEN_XN_ID        = 2;
constexpr uint16_t POST_SYNC_ID       = 3;
constexpr uint16_t STEP_PRE_SYNC_ID   = 4;
constexpr uint16_t STEP_POST_SYNC_ID  = 5;
constexpr uint16_t CKE_IDX_0          = 0; // 每个channel有单独cke(16个bit)，只需一个cke
constexpr uint16_t BIT_NUM_PER_CKE    = 16; // 本rank给远端置位时应当写的CKE，16个对端一个CKE

CcuKernelAllGatherOmniPipeNHR1DMem2Mem::CcuKernelAllGatherOmniPipeNHR1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const auto kernelArg
        = dynamic_cast<const CcuKernelArgAllGatherOmniPipeNHR1DMem2Mem *>(&arg);
    rankId_            = kernelArg->rankId_;
    rankSize_          = kernelArg->rankSize_; // 子通信域rank数
    channels_          = kernelArg->channels;
    userRank_          = kernelArg->subCommRanks_[0][rankId_];
    stepInfoVector_    = kernelArg->stepInfoVector_;
    rank2ChannelIdx_   = kernelArg->rank2ChannelIdx_;
    localSize_         = rank2ChannelIdx_.size(); // nhr 算法通信rank数
    myRankIdx_         = rank2ChannelIdx_.size(); // InitResources中将本端放在末尾 此处为对应的idx
    // subRankIdx2RankIdx_   = kernelArg->subRankIdx2RankIdx_;

    HCCL_INFO("[%s] kernelArg: rankId[%u] userRank[%u] myRankIdx[%u] rankSize[%u] localSize[%u]", __func__, rankId_,
        userRank_, myRankIdx_, rankSize_, localSize_);
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMem2Mem::InitResources()
{
    HCCL_DEBUG("[%s] start", __func__);
    if (channels_.size() == 0) {
        HCCL_ERROR("[%s] channels is empty!", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    input_       = CreateVariable();
    for (uint32_t channelIdx = 0; channelIdx < localSize_; channelIdx++) {
        HCCL_DEBUG("[%s] rankId[%u] channelIdx[%u]", __func__, rankId_, channelIdx);
        CcuRep::Variable outputVar;
        CcuRep::Variable tokenVar;
        CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
        output_.push_back(outputVar);
        CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
        token_.push_back(tokenVar);
    }
    HCCL_DEBUG("[%s] push backkk", __func__);
    output_.push_back(CreateVariable()); // 将本端数据加在末尾
    token_.push_back(CreateVariable());

    sliceStride_               = CreateVariable();
    localCopyFlag_             = CreateVariable();
    sliceSize_                 = CreateVariable();
    event_                     = CreateCompletedEvent();
    inputOmniPipeSliceStride_  = CreateVariable();
    groupOpSize_               = CreateGroupOpSize();
    for (uint32_t peerId = 0; peerId < rankSize_; peerId++) {
        inputOmniSliceStrideVec_.push_back(CreateVariable());
    }
    inputSliceStride_          = CreateVariable();
    HCCL_DEBUG("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMem2Mem::LoadArgs()
{
    Load(input_);
    Load(output_[myRankIdx_]);
    Load(token_[myRankIdx_]);
    Load(sliceSize_);
    Load(sliceStride_);
    Load(localCopyFlag_);
    Load(inputOmniPipeSliceStride_);
    Load(groupOpSize_);
    for (int i = 0; i < rankSize_; i++) {
        Load(inputOmniSliceStrideVec_[i]);
    }
    Load(inputSliceStride_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMem2Mem::PreSync()
{
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[localSize_], 1 << OUTPUT_XN_ID)); // 同步并置位远端
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[localSize_], 1 << TOKEN_XN_ID));
    }

    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyWait(channel, CKE_IDX_0, allBit));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMem2Mem::PostSync()
{
    for (auto &ch : channels_) {
        CHK_RET(NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }

    for (auto &ch : channels_) {
        CHK_RET(NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMem2Mem::DoRepeatAllGatherNHR()
{
    // 本地拷贝
    CCU_IF(localCopyFlag_ == 1) {
        HCCL_DEBUG("[%s] rankId[%u] localcopy start", __func__, rankId_);
        CcuRep::LocalAddr myOutput = CreateLocalAddr();
        CcuRep::LocalAddr myInput  = CreateLocalAddr();

        CcuRep::Variable outSliceStride = CreateVariable();
        outSliceStride = 0;
        for (uint32_t i = 0; i < userRank_; i++) {
            outSliceStride += sliceStride_;
        }

        myInput.addr            =  input_;
        myInput.token = token_[myRankIdx_];
        myOutput.addr           =  output_[rankId_];
        myOutput.addr           += outSliceStride;
        myOutput.token = token_[myRankIdx_];

        event_.SetMask(1 << rankId_);
        CCU_IF(sliceSize_ != 0) {
            LocalCopyNb(myOutput, myInput, sliceSize_, event_);
        }
        CCU_IF(sliceSize_ == 0) {
            RecordEvent(event_);
        }
        WaitEvent(event_);

        HCCL_DEBUG("[%s] rankId[%u] localcopy end", __func__, rankId_);
    }

    CCU_IF(localCopyFlag_ == 0) {
        for (auto &nhrStepInfo : stepInfoVector_) {
            CHK_RET(DoRepeatAllGatherNHRSingleStep(nhrStepInfo));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMem2Mem::DoRepeatAllGatherNHRSingleStep(const NHRStepInfoAG &nhrStepInfo)
{
    u32                    &toRankIdx        = rank2ChannelIdx_[nhrStepInfo.toRank];
    u32                    &fromRankIdx      = rank2ChannelIdx_[nhrStepInfo.fromRank];
    ChannelHandle          &sendChannel      = channels_[toRankIdx];
    ChannelHandle          &recvChannel      = channels_[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    const std::vector<u32> &recvSliceIdxList = nhrStepInfo.rxSliceIdxs;

    CcuRep::LocalAddr src = CreateLocalAddr();
    CcuRep::RemoteAddr dst = CreateRemoteAddr();
    src.token = token_[myRankIdx_];
    dst.token = token_[toRankIdx];

    // 通知对端rank自己准备好了-前同步
    CHK_RET(NotifyRecord(recvChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID)); // 通知fromrank可以写入
    CHK_RET(NotifyWait(sendChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID)); // 等待torank准备好

    for (uint32_t idx = 0; idx < sendSliceIdxList.size(); idx++) {
        u32 sendSliceIdx = sendSliceIdxList[idx];
        if (sendSliceIdx == rankId_) {
            src.addr = output_[myRankIdx_];
            src.addr += sliceStride_;
            src.addr += inputOmniPipeSliceStride_;

            dst.addr = output_[toRankIdx];
            dst.addr += sliceStride_;
            dst.addr += inputOmniPipeSliceStride_;
        } else {
            src.addr = output_[myRankIdx_];
            dst.addr = output_[toRankIdx];
            src.addr += inputOmniSliceStrideVec_[sendSliceIdx];
            dst.addr += inputOmniSliceStrideVec_[sendSliceIdx];
        }
        // event_.SetMask(1);
        event_.SetMask(1 << idx);
        CCU_IF(sliceSize_ != 0) {
            WriteNb(sendChannel, dst, src, sliceSize_, event_);
        }
        CCU_IF(sliceSize_ == 0) {
            RecordEvent(event_);
        }
        // WaitEvent(event_);
    }
    event_.SetMask((1 << sendSliceIdxList.size()) - 1);
    WaitEvent(event_);

    // 写之后告诉对端写完了-后同步
    CHK_RET(NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID)); // 通知torank已写完
    CHK_RET(NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID)); // 等待fromrank写完
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherOmniPipeNHR1DMem2Mem::Algorithm()
{
    HCCL_DEBUG("[%s] start", __func__);

    InitResources();
    LoadArgs();
    PreSync();
    DoRepeatAllGatherNHR();
    PostSync();

    HCCL_DEBUG("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherOmniPipeNHR1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllGatherOmniPipeNHR1DMem2Mem *taskArg
        = dynamic_cast<const CcuTaskArgAllGatherOmniPipeNHR1DMem2Mem *>(&arg);

    uint64_t inputAddr                = taskArg->inputAddr_;
    uint64_t outputAddr               = taskArg->outputAddr_;
    uint64_t tokenInfo                = taskArg->token_;
    uint64_t sliceSize                = taskArg->sliceSize_;
    uint64_t sliceStride              = taskArg->sliceStride_;
    uint64_t localCopyFlag            = taskArg->localCopyFlag_;
    uint64_t inputOmniPipeSliceStride = taskArg->inputOmniPipeSliceStride_;
    std::vector<uint64_t> inputOmniSliceStrideVec = taskArg->inputOmniSliceStrideVec_;
    uint64_t inputSliceStride         = taskArg->inputSliceStride_;
    auto goSize                       = CalGoSize(sliceSize);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, tokenInfo, sliceSize, sliceStride,
        localCopyFlag, inputOmniPipeSliceStride, goSize[0], goSize[1], goSize[2], goSize[3]};
    taskArgs.insert(taskArgs.end(), inputOmniSliceStrideVec.begin(), inputOmniSliceStrideVec.end());
    taskArgs.push_back(inputSliceStride);

    HCCL_DEBUG(
        "[%s] rankId[%u] userRank[%u] TaskArgs(size[%u]): (0)inputAddr[%llu] (1)outputAddr[%llu] "
        "(3)sliceSize[%llu] (4)sliceStride[%llu] (5)localCopyFlag[%llu] (6)inputOmniPipeSliceStride[%llu] "
        "(7)goSize[0][%llu] (8)goSize[1][%llu] (9)goSize[2][%llu] (10)goSize[3][%llu] inputSliceStride[%llu]",
        __func__, rankId_, userRank_, taskArgs.size(), inputAddr, outputAddr, sliceSize, sliceStride,
        localCopyFlag, inputOmniPipeSliceStride, goSize[0], goSize[1], goSize[2], goSize[3], inputSliceStride);

    HCCL_DEBUG("[%s] end", __func__);
    return taskArgs;
}
} // namespace Hccl
