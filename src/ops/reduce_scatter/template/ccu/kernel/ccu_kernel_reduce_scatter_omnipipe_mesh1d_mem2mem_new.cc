/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_omnipipe_mesh1d_mem2mem.h"

namespace ops_hccl {

constexpr int CKE_IDX_0 = 0;
constexpr int INPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;

static CcuResult ParseKernelArg(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx, 
                                CcuKernelArgReduceScatterOmniPipeMesh1DMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    return CCU_SUCCESS; 
}

static CcuResult InitResource(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;
    
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuReduceScatterOmniPipeMesh1DMem2Mem] channels is empty!");
        return CCU_E_INTERNAL;
    }
    
    HCCL_INFO("[CcuReduceScatterOmniPipeMesh1DMem2Mem] channels.size: [%u]", arg->channelCount);
    
    ctx.input.resize(arg->dimSize);
    ctx.token.resize(arg->dimSize);
    
    for (uint64_t peerId = 0; peerId < arg->dimSize; peerId++) {
        if (peerId != arg->rankId) {
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }
    
    ctx.selfBit = 1 << arg->rankId;
    
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t argId = 0;
    
    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.offSet, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyFlag, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniPipeSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.residual, argId++));
    
    return CCU_SUCCESS;
}

static CcuResult PreSync(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId],
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    
    uint16_t allBit = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    
    return CCU_SUCCESS;
}

static CcuResult PostSync(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx)
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

static CcuResult DoRepeatReduceScatter(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    
    CCU_IF(ctx.localCopyFlag == 0)
    {
        ccu::LocalAddr dst;
        std::vector<ccu::RemoteAddr> src;
        src.resize(arg->dimSize);
        
        dst.addr = ctx.input[arg->rankId];
        dst.addr += ctx.inputSliceStride;
        dst.addr += ctx.inputOmniPipeSliceStride;
        dst.token = ctx.token[arg->rankId];
        
        src[arg->dimSize - 1].addr = dst.addr;
        src[arg->dimSize - 1].token = dst.token;
        
        uint32_t idx = 0;
        for (uint64_t i = 0; i < arg->dimSize; i++) {
            if (i == arg->rankId) {
                continue;
            }
            
            src[idx].addr = ctx.input[i];
            src[idx].addr += ctx.inputSliceStride;
            src[idx].addr += ctx.inputOmniPipeSliceStride;
            src[idx].token = ctx.token[i];
            idx++;
        }
        
        std::vector<ccu::LocalAddr> scratchMem;
        scratchMem.resize(arg->dimSize);
        
        ccu::Variable scratchOffset;
        scratchOffset = 0;
        
        for (uint64_t i = 0; i < arg->dimSize; i++) {
            scratchMem[i].addr = ctx.scratch;
            scratchMem[i].addr += scratchOffset;
            scratchMem[i].token = ctx.token[arg->rankId];
            scratchOffset += ctx.sliceSize;
        }
        
        uint32_t channelId = 0;
        for (uint64_t i = 0; i < arg->dimSize; i++) {
            const uint16_t rankMask = 1 << i;
            
            if (i == arg->rankId) {
                CCU_CHK_RET(ccu::EventRecord(ctx.event, rankMask));
                continue;
            }
            
            CCU_CHK_RET(ccu::Read(arg->channels[channelId], scratchMem[i], src[channelId], 
                ctx.sliceSize, ctx.event, rankMask));
            channelId++;
        }
        
        CCU_CHK_RET(ccu::EventWait(ctx.event, (1 << arg->dimSize) - 1));
        
        scratchMem[arg->rankId].addr = dst.addr;
        scratchMem[arg->rankId].token = dst.token;
        
        for (uint64_t i = 0; i < arg->dimSize; i++) {
            const uint16_t rankMask = 1 << i;
            
            if (i == arg->rankId) {
                CCU_CHK_RET(ccu::EventRecord(ctx.event, rankMask));
                continue;
            }
            
            CCU_CHK_RET(ccu::LocalReduce(dst, scratchMem[i], ctx.sliceSize, 
                arg->opParam_.DataDes.dataType, arg->opParam_.reduceType, ctx.event, rankMask));
        }
        
        CCU_CHK_RET(ccu::EventWait(ctx.event, (1 << arg->dimSize) - 1));
    }
    
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterOmniPipeMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterOmniPipeMesh1DMem2Mem *>(arg);
    
    ReduceScatterOmniPipeMesh1DMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    
    HCCL_INFO("[CcuReduceScatterOmniPipeMesh1DMem2Mem] ReduceScatterOmniPipeMesh1DMem2Mem run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    
    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoRepeatReduceScatter(ctx));
    CCU_CHK_RET(PostSync(ctx));
    
    HCCL_INFO("[CcuReduceScatterOmniPipeMesh1DMem2Mem] ReduceScatterOmniPipeMesh1DMem2Mem end");
    
    return CCU_SUCCESS;
}

} // namespace ops_hccl