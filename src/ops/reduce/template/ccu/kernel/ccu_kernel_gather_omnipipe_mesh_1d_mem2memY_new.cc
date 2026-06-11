/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_gather_omnipipe_mesh_1d_mem2mem.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;

static CcuResult ParseKernelArg(GatherOmniPipeMesh1DMem2MemYContext &ctx, 
                                CcuKernelArgGatherOmniPipeMesh1DMem2MemY *kernelArg)
{
    ctx.arg = kernelArg;
    return CCU_SUCCESS;
}

static CcuResult InitResource(GatherOmniPipeMesh1DMem2MemYContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;
    
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuGatherOmniPipeMesh1DMem2MemY] channels is empty!");
        return CCU_E_INTERNAL;
    }
    
    HCCL_INFO("[CcuGatherOmniPipeMesh1DMem2MemY] channels.size: [%u]", arg->channelCount);
    
    ctx.input.resize(arg->dimSize);
    ctx.token.resize(arg->dimSize);
    
    for (uint64_t peerId = 0; peerId < arg->dimSize; peerId++) {
        if (peerId != arg->rankId) {
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }
    
    ctx.inputMem.resize(arg->dimSize);
    ctx.outputMem.resize(arg->dimSize);
    
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(GatherOmniPipeMesh1DMem2MemYContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t argId = 0;
    
    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyFlag, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniPipeSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputOmniPipeSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isStepOne, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isLastStep, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.ifNewRoot, argId++));
    
    return CCU_SUCCESS;
}

static CcuResult PreSync(GatherOmniPipeMesh1DMem2MemYContext &ctx)
{
    const auto *arg = ctx.arg;
    
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId],
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    
    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    
    return CCU_SUCCESS;
}

static CcuResult PostSync(GatherOmniPipeMesh1DMem2MemYContext &ctx)
{
    const auto *arg = ctx.arg;
    
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    
    return CCU_SUCCESS;
}

static CcuResult DoGather(GatherOmniPipeMesh1DMem2MemYContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelId = 0;
    
    for (uint64_t rankIdx = 0; rankIdx < arg->dimSize; rankIdx++) {
        const uint16_t rankMask = 1 << rankIdx;
        
        CCU_IF(ctx.sliceSize != 0)
        {
            if (rankIdx == arg->rankId) {
                CCU_CHK_RET(ccu::EventRecord(ctx.event, rankMask));
            } else {
                CCU_CHK_RET(ccu::Read(arg->channels[channelId], ctx.outputMem[rankIdx], 
                    ctx.inputMem[rankIdx], ctx.sliceSize, ctx.event, rankMask));
                channelId++;
            }
        }
    }
    
    ctx.event.SetMask((1 << arg->dimSize) - 1);
    CCU_CHK_RET(ccu::EventWait(ctx.event, (1 << arg->dimSize) - 1));
    
    return CCU_SUCCESS;
}

static CcuResult DoRepeatGather(GatherOmniPipeMesh1DMem2MemYContext &ctx)
{
    const auto *arg = ctx.arg;
    
    for (uint64_t rankIdx = 0; rankIdx < arg->dimSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            ctx.outputMem[rankIdx].addr = ctx.output;
            ctx.outputMem[rankIdx].addr += ctx.outputSliceStride * arg->rankId;
            ctx.outputMem[rankIdx].token = ctx.token[arg->rankId];
        } else {
            ctx.inputMem[rankIdx].addr = ctx.input[rankIdx];
            ctx.inputMem[rankIdx].addr += ctx.inputSliceStride * rankIdx;
            ctx.inputMem[rankIdx].token = ctx.token[rankIdx];
        }
    }
    
    CCU_CHK_RET(DoGather(ctx));
    
    return CCU_SUCCESS;
}

CcuResult CcuGatherOmniPipeMesh1DMem2MemYKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGatherOmniPipeMesh1DMem2MemY *>(arg);
    
    GatherOmniPipeMesh1DMem2MemYContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    
    HCCL_INFO("[CcuGatherOmniPipeMesh1DMem2MemY] GatherOmniPipeMesh1DMem2MemY run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    
    CCU_CHK_RET(PreSync(ctx));
    
    CCU_CHK_RET(DoRepeatGather(ctx));
    
    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuGatherOmniPipeMesh1DMem2MemY] GatherOmniPipeMesh1DMem2MemY end");
    
    return CCU_SUCCESS;
}

} // namespace ops_hccl
