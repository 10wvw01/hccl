/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_gather_omnipipe_mesh_1d_mem2memY.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;

static CcuResult ParseKernelArg(GatherOmniPipeMesh1DMem2MemContextY &ctx, CcuKernelArgGatherOmniPipeMesh1DMem2MemY *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankSize = kernelArg->rankSize;
    ctx.rankId = kernelArg->rankId;
    ctx.rootId = kernelArg->rootId;
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.subRankIdx2RankIdx = kernelArg->subRankIdx2RankIdx;
    return CCU_SUCCESS;
}

static CcuResult InitResource(GatherOmniPipeMesh1DMem2MemContextY &ctx)
{
    uint32_t channelIdx = 0;
    
    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[CcuGatherOmniPipeMesh1DMem2MemKernelY] channels is empty!");
        return CCU_E_INTERNAL;
    }
    
    HCCL_INFO("[CcuGatherOmniPipeMesh1DMem2MemY] channels.size: [%u]", ctx.arg->channelCount);
    
    ctx.input.resize(ctx.rankSize);
    ctx.token.resize(ctx.rankSize);
    
    for (uint64_t peerId = 0; peerId < ctx.rankSize; peerId++) {
        if (peerId != ctx.rankId) {
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }
    
    ctx.inputMem.resize(ctx.rankSize);
    ctx.outputMem.resize(ctx.rankSize);
    
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(GatherOmniPipeMesh1DMem2MemContextY &ctx)
{
    uint32_t argId = 0;
    
    CCU_CHK_RET(ccu::LoadArg(ctx.input[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyFlag, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniPipeSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputOmniPipeSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isStepOne, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isLastStep, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.ifNewRoot, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.peerId, argId++));
    
    return CCU_SUCCESS;
}

static CcuResult PreSync(GatherOmniPipeMesh1DMem2MemContextY &ctx)
{
    HCCL_INFO("-------start--------,RankId[%u]", ctx.rankId);
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        HCCL_INFO("-------1--------,i[%u] RankId[%u]", i, ctx.rankId);
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.input[ctx.rankId],
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token[ctx.rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    
    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        HCCL_INFO("-------2--------,i[%u] RankId[%u]", i, ctx.rankId);
        ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO("-------end--------,RankId[%u]", ctx.rankId);
    return CCU_SUCCESS;
}

static CcuResult PostSync(GatherOmniPipeMesh1DMem2MemContextY &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    
    return CCU_SUCCESS;
}

static CcuResult DoGather(GatherOmniPipeMesh1DMem2MemContextY &ctx)
{
    uint32_t channelId = 0;
    
    for (uint64_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        uint16_t rankMask = 1 << rankIdx;
        if (rankIdx == ctx.rankId) {
            ccu::EventRecord(ctx.event, rankMask);
            continue;
        }
        CCU_IF(ctx.sliceSize != 0) {
            if (ctx.peerId == rankIdx) {
                ccu::Read(ctx.arg->channels[channelId], ctx.outputMem[rankIdx], ctx.inputMem[rankIdx], ctx.sliceSize, ctx.event, rankMask);
            } else {
                ccu::EventRecord(ctx.event, rankMask);
            }
        }

        CCU_IF(ctx.sliceSize == 0)
        {
            ccu::EventRecord(ctx.event, rankMask);
        }
        channelId++;
    }
    
    ccu::EventWait(ctx.event, (1 << ctx.rankSize) - 1);
    return CCU_SUCCESS;
}

static CcuResult DoRepeatGather(GatherOmniPipeMesh1DMem2MemContextY &ctx)
{
    for (uint64_t curId = 0; curId < ctx.rankSize; curId++) {
        if (curId == ctx.rankId) {
            continue;
        }
        ctx.inputMem[curId].token = ctx.token[curId];
        ctx.inputMem[curId].addr = ctx.input[curId];
        // ctx.inputMem[curId].addr += ctx.inputOmniPipeSliceStride;

        ctx.outputMem[curId].token = ctx.token[curId];
        ctx.outputMem[curId].addr = ctx.output;
        // ctx.outputMem[curId].addr += ctx.outputOmniPipeSliceStride;
    }
    CCU_IF(ctx.ifNewRoot == true)
    {
        CCU_CHK_RET(DoGather(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuGatherOmniPipeMesh1DMem2MemKernelY(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGatherOmniPipeMesh1DMem2MemY *>(arg);
    
    GatherOmniPipeMesh1DMem2MemContextY ctx;
    
    HCCL_INFO("[CcuGatherOmniPipeMesh1DMem2MemY] GatherOmniPipeMesh1DMem2MemY run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx)); 
    
    CCU_CHK_RET(PreSync(ctx));
    
    CCU_CHK_RET(DoRepeatGather(ctx));
    
    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuGatherOmniPipeMesh1DMem2MemY] new GatherOmniPipeMesh1DMem2MemY end");
    
    return CCU_SUCCESS;
}

} // namespace ops_hccl