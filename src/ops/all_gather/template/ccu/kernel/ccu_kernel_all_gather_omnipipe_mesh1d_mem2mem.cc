/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_omnipipe_mesh1d_mem2mem.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID   = 0;
constexpr int OUTPUT_XN_ID  = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID  = 3;
constexpr int CKE_IDX_0     = 0;

CcuKernelAllGatherOmniPipeMesh1DMem2Mem::CcuKernelAllGatherOmniPipeMesh1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem *kernelArg
        = dynamic_cast<const CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem *>(&arg);
    rankIdx_        = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    subCommRanks_   = kernelArg->subCommRanks_;
    userRank_       = subCommRanks_[0][rankIdx_];

    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_INFO("[%s] outputDataType is [INVALID], set outputDataType to[%d]", __func__, outputDataType_);
    }
    HCCL_INFO("[%s] Init, KernelArgs are rankIdx[%u], userRank[%u], rankSize_[%u], dataType[%d], "
              "outputDataType[%d]",
        __func__, rankIdx_, userRank_, rankSize_, dataType_, outputDataType_);
}

HcclResult CcuKernelAllGatherOmniPipeMesh1DMem2Mem::InitResource()
{
    HCCL_INFO("[%s] start", __func__);
    if (channels_.size() == 0) {
        HCCL_ERROR("[%s] channels is empty!", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }

    HCCL_INFO("[%s] channels_.size[%u]", __func__, channels_.size());

    input_ = CreateVariable();
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    uint16_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankIdx_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO("[%s] MyRank[%u], userRank_[%d], PeerId[%u], ChannelId[%u]", __func__, rankIdx_, userRank_, peerId, channelIdx);
            CcuRep::Variable outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    sliceStride_               = CreateVariable();
    localCopyFlag_             = CreateVariable();
    sliceSize_                 = CreateVariable();
    event_                     = CreateCompletedEvent();
    inputOmniPipeSliceStride_  = CreateVariable();
    groupOpSize_               = CreateGroupOpSize();

    HCCL_INFO("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGatherOmniPipeMesh1DMem2Mem::LoadArgs()
{
    HCCL_INFO("[%s] start", __func__);

    Load(input_);
    Load(output_[rankIdx_]);
    Load(token_[rankIdx_]);
    Load(sliceSize_);
    Load(sliceStride_);
    Load(localCopyFlag_);
    Load(inputOmniPipeSliceStride_);
    Load(groupOpSize_);

    HCCL_INFO("[%s] end", __func__);
}

void CcuKernelAllGatherOmniPipeMesh1DMem2Mem::PreSync()
{
    HCCL_INFO("[%s] channels_ size %u", __func__, channels_.size());
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankIdx_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankIdx_], 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
}

void CcuKernelAllGatherOmniPipeMesh1DMem2Mem::PostSync()
{
    HCCL_INFO("[%s] channels_ size %u", __func__, channels_.size());
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelAllGatherOmniPipeMesh1DMem2Mem::DoRepeatAllGather()
{
    HCCL_INFO("[%s] start", __func__);

    // 本地拷贝
    CCU_IF(localCopyFlag_ == 1) {
        HCCL_INFO("[%s] rankIdx[%u] userRank[%u] localcopy begin", __func__, rankIdx_, userRank_);
        CcuRep::LocalAddr myOutput = CreateLocalAddr();
        CcuRep::LocalAddr myInput  = CreateLocalAddr();

        CcuRep::Variable outSliceStride = CreateVariable();
        outSliceStride = 0;
        for (uint32_t i = 0; i < userRank_; i++) {
            outSliceStride += sliceStride_;
        }

        myInput.addr            =  input_;
        myInput.token = token_[rankIdx_];
        myOutput.addr           =  output_[rankIdx_];
        myOutput.addr           += outSliceStride;
        myOutput.token = token_[rankIdx_];

        event_.SetMask(1 << rankIdx_);
        CCU_IF(sliceSize_ != 0) {
            // LocalCopyNb(myOutput, myInput, sliceSize_, event_);
            GroupCopy(myOutput, myInput, groupOpSize_);
            RecordEvent(event_);
        }
        CCU_IF(sliceSize_ == 0) {
            RecordEvent(event_);
        }
        WaitEvent(event_);
        HCCL_INFO("[%s] rankIdx[%u] userRank[%u] localcopy end", __func__, rankIdx_, userRank_);
    }

    CCU_IF(localCopyFlag_ == 0) {
        CcuRep::LocalAddr src = CreateLocalAddr();
        std::vector<CcuRep::RemoteAddr> dst;
        for (uint64_t idx = 0; idx < rankSize_; idx++) {
            if (idx == rankIdx_) {
                dst.push_back({});
            }
            else {
                dst.push_back(CreateRemoteAddr());
            }
        }

        HCCL_INFO("[%s] kernel rankId %llu, rankIdx %llu, rankSize %llu", __func__, userRank_, rankIdx_, rankSize_);

        src.addr = output_[rankIdx_];
        src.addr += sliceStride_;
        src.addr += inputOmniPipeSliceStride_;
        src.token = token_[rankIdx_];

        HCCL_INFO("[%s] Mem2Mem start", __func__);
        for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
            if (peerId == rankIdx_) {
                continue;
            }
            dst[peerId].addr = output_[peerId];
            dst[peerId].addr += sliceStride_;
            dst[peerId].addr += inputOmniPipeSliceStride_;
            dst[peerId].token = token_[peerId];
        }

        uint16_t channelId = 0;
        for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
            if (peerId == rankIdx_) {
                event_.SetMask(1 << rankIdx_);
                RecordEvent(event_);
            } else {
                event_.SetMask(1 << peerId);
                CCU_IF(sliceSize_ != 0) {
                    WriteNb(channels_[channelId], dst[peerId], src, sliceSize_, event_);
                }
                CCU_IF(sliceSize_ == 0) {
                    RecordEvent(event_);
                }
                channelId++;
            }
        }
        event_.SetMask((1 << rankSize_) - 1);
        WaitEvent(event_);
        HCCL_INFO("[%s] Mem2Mem end", __func__);
    }
    HCCL_INFO("[%s] end", __func__);
}

HcclResult CcuKernelAllGatherOmniPipeMesh1DMem2Mem::Algorithm()
{
    HCCL_INFO("[%s] start", __func__);
    CHK_RET(InitResource());
    LoadArgs();
    PreSync();
    DoRepeatAllGather();
    PostSync();
    HCCL_INFO("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherOmniPipeMesh1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    HCCL_INFO("[%s] start", __func__);
    const CcuTaskArgAllGatherOmniPipeMesh1DMem2Mem *taskArg
        = dynamic_cast<const CcuTaskArgAllGatherOmniPipeMesh1DMem2Mem *>(&arg);
    uint64_t inputAddr                = taskArg->inputAddr_;
    uint64_t outputAddr               = taskArg->outputAddr_;
    uint64_t tokenInfo                = taskArg->token_;
    uint64_t sliceSize                = taskArg->sliceSize_;
    uint64_t sliceStride              = taskArg->sliceStride_;
    uint64_t localCopyFlag            = taskArg->localCopyFlag_;
    uint64_t inputOmniPipeSliceStride = taskArg->inputOmniPipeSliceStride_;
    auto goSize                       = CalGoSize(sliceSize);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, tokenInfo, sliceSize, sliceStride,
        localCopyFlag, inputOmniPipeSliceStride, goSize[0], goSize[1], goSize[2], goSize[3]};

    HCCL_INFO(
        "[%s] rankIdx[%u] userRank[%u] TaskArgs(size[%u]): (0)inputAddr[%llu] (1)outputAddr[%llu] "
        "(3)sliceSize[%llu] (4)sliceStride[%llu] (5)localCopyFlag[%llu] (6)inputOmniPipeSliceStride[%llu] "
        "(7)goSize[0][%llu] (8)goSize[1][%llu] (9)goSize[2][%llu] (10)goSize[3][%llu]",
        __func__, rankIdx_, userRank_, taskArgs.size(), inputAddr, outputAddr, sliceSize, sliceStride,
        localCopyFlag, inputOmniPipeSliceStride, goSize[0], goSize[1], goSize[2], goSize[3]);

    HCCL_INFO("[%s] end", __func__);
    return taskArgs;
}

} // namespace ops_hccl