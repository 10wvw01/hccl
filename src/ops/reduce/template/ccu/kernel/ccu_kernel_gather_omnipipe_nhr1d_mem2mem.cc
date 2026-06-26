/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_gather_omnipipe_nhr1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID   = 0;
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int STEP_SYNC_ID = 3;
constexpr int POST_SYNC_ID   = 4;
constexpr int CKE_IDX_0     = 0;

static CcuResult ParseKernelArg(GatherOmniPipeNHR1DMem2MemContext &ctx, 
                                CcuKernelArgGatherOmniPipeNHR1DMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankSize = kernelArg->rankSize;
    ctx.rankId = kernelArg->rankId;
    ctx.rootId = kernelArg->rootId;
    ctx.stepInfoVector = kernelArg->stepInfoVector;
    ctx.rank2ChannelIdx = kernelArg->rank2ChannelIdx;
    ctx.localSize = static_cast<uint32_t>(ctx.rank2ChannelIdx.size());
    ctx.myRankIdx = ctx.localSize;
    return CCU_SUCCESS;
}

static CcuResult InitResource(GatherOmniPipeNHR1DMem2MemContext &ctx)
{
    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[CcuGatherOmniPipeNHR1DMem2Mem] channels is empty!");
        return CCU_E_INTERNAL;
    }
    
    HCCL_INFO("[CcuGatherOmniPipeNHR1DMem2Mem] channels.size: [%u]", ctx.arg->channelCount);
    
    ctx.input.resize(ctx.localSize);
    ctx.token.resize(ctx.localSize);
    
    for (uint64_t channelIdx = 0; channelIdx < ctx.localSize; channelIdx++) {
        ctx.input[channelIdx] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
        ctx.scratch[channelIdx] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], SCRATCH_XN_ID);
        ctx.token[channelIdx] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    ctx.inputOmniSliceStrideVec.resize(ctx.rankSize);
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(GatherOmniPipeNHR1DMem2MemContext &ctx)
{
    uint32_t argId = 0;
    
    CCU_CHK_RET(ccu::LoadArg(ctx.input[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyFlag, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniPipeSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputOmniPipeSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isStepOne, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isLastStep, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.ifNewRoot, argId++));
    for (uint64_t i = 0; i < ctx.rankSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniSliceStrideVec[i], argId++));
    }
    return CCU_SUCCESS;
}

static CcuResult PreSync(GatherOmniPipeNHR1DMem2MemContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.input[ctx.rankId],
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.input[ctx.rankId],
            SCRATCH_XN_ID, CKE_IDX_0, 1 << SCRATCH_XN_ID);
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token[ctx.rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    
    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << SCRATCH_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit));
    }
    
    return CCU_SUCCESS;
}

static CcuResult PostSync(GatherOmniPipeNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    
    return CCU_SUCCESS;
}

static CcuResult DoGatherOmniPipeNHRSingleStep(GatherOmniPipeNHR1DMem2MemContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    ccu::RemoteAddr src;
    ccu::LocalAddr dst;

    const auto &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    const auto &recvSliceIdxList = nhrStepInfo.rxSliceIdxs;
    u32 toRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.toRank];
    u32 fromRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.fromRank];
    // 发送端：先通知接收方可以读取自己的数据
    if (sendSliceIdxList.size() != 0) {
        if (ctx.rank2ChannelIdx.count(nhrStepInfo.toRank) == 0) {
            return CCU_E_INTERNAL;
        }
        if (toRankIdx >= ctx.arg->channelCount) {
            return CCU_E_INTERNAL;
        }
        ccu::NotifyRecord(ctx.arg->channels[toRankIdx], CKE_IDX_0, 1 << STEP_SYNC_ID);
        ccu::NotifyWait(ctx.arg->channels[fromRankIdx], CKE_IDX_0, 1 << STEP_SYNC_ID);
    }

    if (recvSliceIdxList.size() != 0) {
        ChannelHandle recvChannel        = ctx.arg->channels[fromRankIdx];
        src.token                        = ctx.token[ctx.myRankIdx];
        dst.token                        = ctx.token[fromRankIdx];

        u32 recvSliceIdxSize = recvSliceIdxList.size();
        for (u32 i = 0; i < recvSliceIdxSize; i++) {
            u32 recvSliceIdx = recvSliceIdxList[i];
            if (nhrStepInfo.fromRank == recvSliceIdx) {
                src.addr = ctx.input[fromRankIdx];
            } else {
                src.addr = ctx.scratch[fromRankIdx];
            }
            src.addr += ctx.inputOmniSliceStrideVec[recvSliceIdx];

            dst.addr = ctx.output;
            dst.addr += ctx.inputOmniSliceStrideVec[recvSliceIdx];

            CCU_IF(ctx.sliceSize != 0) {
                ccu::Read(recvChannel, dst, src, ctx.sliceSize, ctx.event, 1 << i);
            }
            CCU_IF(ctx.sliceSize == 0) {
                ccu::EventRecord(ctx.event, 1 << i);
            }
        }
        ccu::EventWait(ctx.event, (1 << recvSliceIdxSize) - 1);
    }

    HCCL_INFO("[DoGatherOmniPipeNHRSingleStep] step %u, toRank=%u, fromRank=%u, sendSliceNum=%lu",
        nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
    return CCU_SUCCESS;
}

static CcuResult DoGatherOmniPipeNHR(GatherOmniPipeNHR1DMem2MemContext &ctx)
{
    for (auto &step : ctx.stepInfoVector) {
        CCU_CHK_RET(DoGatherOmniPipeNHRSingleStep(ctx, step));
    }
    return CCU_SUCCESS;
}

CcuResult CcuGatherOmniPipeNHR1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGatherOmniPipeNHR1DMem2Mem *>(arg);
    GatherOmniPipeNHR1DMem2MemContext ctx;

    HCCL_INFO("[CcuGatherOmniPipeNHR1DMem2Mem] GatherOmniPipeNHR1DMem2Mem run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    
    // CCU_CHK_RET(PreSync(ctx));
    
    // CCU_CHK_RET(DoGatherOmniPipeNHR(ctx));
    
    // CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuGatherOmniPipeNHR1DMem2Mem] GatherOmniPipeNHR1DMem2Mem end");
    
    return CCU_SUCCESS;
}

} // namespace ops_hccl