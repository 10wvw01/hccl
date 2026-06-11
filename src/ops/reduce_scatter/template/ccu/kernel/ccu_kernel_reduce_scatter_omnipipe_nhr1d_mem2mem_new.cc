/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_omnipipe_nhr1d_mem2mem.h"

namespace ops_hccl {

constexpr uint16_t INPUT_XN_ID = 0;
constexpr uint16_t TOKEN_XN_ID = 1;

constexpr uint16_t CKE_IDX_INPUT = 0;
constexpr uint16_t CKE_IDX_TOKEN = 1;
constexpr uint16_t CKE_IDX_READY = 2;
constexpr uint16_t CKE_IDX_DONE = 3;
constexpr uint16_t POST_XN_ID = 4;
constexpr uint16_t BIT_NUM_PER_CKE = 16;

static uint32_t GetSignalIndex(const int signalBit)
{
    return static_cast<uint32_t>(signalBit) / BIT_NUM_PER_CKE;
}

static uint16_t GetSignalMask(const int signalBit)
{
    return (1 << (static_cast<uint32_t>(signalBit) % BIT_NUM_PER_CKE));
}

static CcuResult ParseKernelArg(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx, 
                                CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.localSize = kernelArg->rank2ChannelIdx_.size();
    ctx.myRankIdx = kernelArg->rank2ChannelIdx_.size();
    return CCU_SUCCESS;
}

static CcuResult InitResources(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuReduceScatterOmniPipeNHR1DMem2Mem] channels is empty!");
        return CCU_E_INTERNAL;
    }
    
    HCCL_INFO("[CcuReduceScatterOmniPipeNHR1DMem2Mem] channels.size: [%u]", arg->channelCount);
    
    ctx.input.resize(ctx.localSize + 1);
    ctx.token.resize(ctx.localSize + 1);
    
    for (uint32_t channelIdx = 0; channelIdx < ctx.localSize; channelIdx++) {
        ctx.input[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
        ctx.token[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
    }
    
    ctx.input[ctx.myRankIdx] = ccu::Variable();
    ctx.token[ctx.myRankIdx] = ccu::Variable();
    
    ctx.inputOmniSliceStrideVec.resize(arg->rankSize);
    
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t argId = 0;
    
    CCU_CHK_RET(ccu::LoadArg(ctx.input[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyFlag, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniPipeSliceStride, argId++));
    
    for (uint32_t i = 0; i < arg->rankSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniSliceStrideVec[i], argId++));
    }
    
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, argId++));
    
    return CCU_SUCCESS;
}

static CcuResult PreSync(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    
    const uint16_t signalBitInput = GetSignalMask(CKE_IDX_INPUT);
    const uint16_t signalBitToken = GetSignalMask(CKE_IDX_TOKEN);
    const uint32_t signalIndexInput = GetSignalIndex(CKE_IDX_INPUT);
    const uint32_t signalIndexToken = GetSignalIndex(CKE_IDX_TOKEN);
    
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[ctx.myRankIdx],
            CKE_IDX_INPUT, signalIndexInput, signalBitInput));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.myRankIdx],
            CKE_IDX_TOKEN, signalIndexToken, signalBitToken));
    }
    
    const uint16_t waitMask = signalBitInput | signalBitToken;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], signalIndexInput, waitMask));
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], signalIndexToken, waitMask));
    }
    
    return CCU_SUCCESS;
}

static CcuResult PostSync(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    
    const uint16_t selfBitInput = GetSignalMask(POST_XN_ID);
    const uint32_t signalIndexInput = GetSignalIndex(POST_XN_ID);
    
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], signalIndexInput, selfBitInput));
    }
    
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], signalIndexInput, selfBitInput));
    }
    
    return CCU_SUCCESS;
}

static CcuResult DoNHRStep(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx, const NHRStepInfoRS &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    
    if (nhrStepInfo.fromRank != nhrStepInfo.myRank) {
        ccu::RemoteAddr srcAddr;
        srcAddr.addr = ctx.input[nhrStepInfo.fromRank];
        srcAddr.addr += ctx.inputOmniSliceStrideVec[nhrStepInfo.fromRank];
        srcAddr.token = ctx.token[nhrStepInfo.fromRank];
        
        ccu::LocalAddr dstAddr;
        dstAddr.addr = ctx.output;
        dstAddr.addr += ctx.inputOmniSliceStrideVec[nhrStepInfo.fromRank];
        dstAddr.token = ctx.token[ctx.myRankIdx];
        
        uint32_t channelIdx = arg->rank2ChannelIdx_[nhrStepInfo.fromRank];
        
        CCU_CHK_RET(ccu::Read(arg->channels[channelIdx], dstAddr, srcAddr, 
            ctx.sliceSize, ctx.event, 1));
    }
    
    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatterNHR(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    
    for (const auto &stepInfo : arg->stepInfoVector_) {
        CCU_CHK_RET(DoNHRStep(ctx, stepInfo));
    }
    
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterOmniPipeNHR1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem *>(arg);
    
    ReduceScatterOmniPipeNHR1DMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    
    HCCL_INFO("[CcuReduceScatterOmniPipeNHR1DMem2Mem] ReduceScatterOmniPipeNHR1DMem2Mem run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    
    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoRepeatReduceScatterNHR(ctx));
    CCU_CHK_RET(PostSync(ctx));
    
    HCCL_INFO("[CcuReduceScatterOmniPipeNHR1DMem2Mem] ReduceScatterOmniPipeNHR1DMem2Mem end");
    
    return CCU_SUCCESS;
}

} // namespace ops_hccl