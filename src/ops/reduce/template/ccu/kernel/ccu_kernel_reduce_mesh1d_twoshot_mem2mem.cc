/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_mesh1d_twoshot_mem2mem.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID     = 0;
constexpr int SCRATCH_XN_ID   = 1;
constexpr int OUTPUT_XN_ID    = 2;
constexpr int TOKEN_XN_ID     = 3;
constexpr int POST_SYNC_ID    = 4;
constexpr int SCATTER_SYNC_ID = 5;
constexpr int CKE_IDX_0       = 0;
constexpr uint16_t BIT_NUM_PER_CKE = 16;

static CcuResult ParseKernelArg(ReduceMesh1DTwoShotMem2MemContext &ctx,
                                CcuKernelArgReduceMesh1DTwoShotMem2Mem *kernelArg)
{
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceMesh1DTwoShotMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    ctx.input.resize(arg->rankSize);
    ctx.scratch.resize(arg->rankSize);
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    ctx.scratchMem.resize(arg->rankSize);
    ctx.constVar1 = 1;

    uint32_t eventNum = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    ctx.events.resize(eventNum);

    if (arg->channelCount == 0) {
        return CCU_SUCCESS;
    }
    HCCL_INFO("[CcuKernelReduceMesh1DTwoShotMem2Mem] channels.size: [%u]", arg->channelCount);

    uint32_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            continue;
        }
        ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
        ctx.scratch[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], SCRATCH_XN_ID);
        ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
        channelIdx++;
    }
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceMesh1DTwoShotMem2MemContext &ctx)
{
    uint32_t cnt = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[ctx.arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch[ctx.arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, cnt++));
    return CCU_SUCCESS;
}

static void PreSync(ReduceMesh1DTwoShotMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.scratch[arg->rankId], SCRATCH_XN_ID, CKE_IDX_0, 1 << SCRATCH_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << SCRATCH_XN_ID) | (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
}

static void PostSync(ReduceMesh1DTwoShotMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

static void InitScratchMem(ReduceMesh1DTwoShotMem2MemContext &ctx)
{
    if (ctx.arg->rankId == ctx.arg->rankSize - 1) {
        ctx.mySliceSize = ctx.lastSliceSize;
    } else {
        ctx.mySliceSize = ctx.normalSliceSize;
    }
    ctx.myScratchOffset = 0;
    for (uint32_t k = 0; k < ctx.arg->rankId; k++) {
        ctx.myScratchOffset += ctx.normalSliceSize;
    }
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t k = 0; k < ctx.arg->rankSize; k++) {
        ctx.scratchMem[k].addr = ctx.scratch[ctx.arg->rankId];
        ctx.scratchMem[k].addr += scratchOffset;
        ctx.scratchMem[k].token = ctx.token[ctx.arg->rankId];
        scratchOffset += ctx.normalSliceSize;
    }
}

static CcuResult DoScatterWriteWait(ReduceMesh1DTwoShotMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t numEvents = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    for (uint32_t i = 0; i < numEvents; i++) {
        uint16_t eventMask;
        if (i == numEvents - 1 && arg->rankSize % BIT_NUM_PER_CKE != 0) {
            eventMask = (1 << (arg->rankSize % BIT_NUM_PER_CKE)) - 1;
        } else {
            eventMask = (1 << BIT_NUM_PER_CKE) - 1;
        }
        ccu::EventWait(ctx.events[i], eventMask);
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterWrite(ReduceMesh1DTwoShotMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::Variable inputOffset;
    inputOffset = 0;
    uint32_t channelIdx = 0;

    for (uint32_t peerId = 0; peerId < arg->rankSize; peerId++) {
        uint32_t eventIdx = peerId / BIT_NUM_PER_CKE;
        uint16_t rankMask = 1 << (peerId % BIT_NUM_PER_CKE);
        if (peerId == arg->rankId) {
            ccu::EventRecord(ctx.events[eventIdx], rankMask);
        } else {
            ctx.peerSliceSize = (peerId == arg->rankSize - 1) ? ctx.lastSliceSize : ctx.normalSliceSize;
            ctx.myInput.addr = ctx.input[arg->rankId];
            ctx.myInput.addr += inputOffset;
            ctx.myInput.token = ctx.token[arg->rankId];
            ctx.remoteScratch.addr = ctx.scratch[peerId];
            ctx.remoteScratch.addr += ctx.myScratchOffset;
            ctx.remoteScratch.token = ctx.token[peerId];
            CCU_IF(ctx.peerSliceSize != 0) {
                ccu::Write(arg->channels[channelIdx], ctx.remoteScratch, ctx.myInput, ctx.peerSliceSize,
                           ctx.events[eventIdx], rankMask);
            } CCU_ELSE {
                ccu::EventRecord(ctx.events[eventIdx], rankMask);
            }
            channelIdx++;
        }
        inputOffset += ctx.normalSliceSize;
    }

    ctx.myInput.addr = ctx.input[arg->rankId];
    ctx.myInput.addr += ctx.myScratchOffset;
    ctx.myInput.token = ctx.token[arg->rankId];
    CCU_IF(ctx.mySliceSize != 0) {
        CCU_CHK_RET(GroupCopy(ctx, ctx.scratchMem[arg->rankId], ctx.myInput, ctx.goSize));
    }

    CCU_CHK_RET(DoScatterWriteWait(ctx));
    return CCU_SUCCESS;
}

static CcuResult DoLocalReduce(ReduceMesh1DTwoShotMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    CCU_IF(ctx.mySliceSize != 0)
    {
        for (uint32_t peerId = 0; peerId < arg->rankSize; peerId++) {
            if (peerId == arg->rankId) {
                continue;
            }
            ccu::LocalReduce(ctx.scratchMem[arg->rankId], ctx.scratchMem[peerId], ctx.mySliceSize,
                             ctx.dataType, ctx.reduceOp, ctx.events[0]);
            ccu::EventWait(ctx.events[0]);
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoReduceScatter(ReduceMesh1DTwoShotMem2MemContext &ctx)
{
    CCU_CHK_RET(DoScatterWrite(ctx));
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << SCATTER_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << SCATTER_SYNC_ID);
    }
    CCU_CHK_RET(DoLocalReduce(ctx));
    return CCU_SUCCESS;
}

static CcuResult DoGather(ReduceMesh1DTwoShotMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    ctx.myScratchResult.addr = ctx.scratchMem[arg->rankId].addr;
    ctx.myScratchResult.token = ctx.scratchMem[arg->rankId].token;

    CCU_IF(ctx.mySliceSize != 0)
    {
        if (arg->rankId == arg->rootId) {
            ctx.myOutput.addr = ctx.output[arg->rankId];
            ctx.myOutput.addr += ctx.myScratchOffset;
            ctx.myOutput.token = ctx.token[arg->rankId];
            CCU_CHK_RET(GroupCopy(ctx, ctx.myOutput, ctx.myScratchResult, ctx.goSize));
        } else {
            uint16_t channelToRoot = (arg->rootId < arg->rankId) ? arg->rootId : arg->rootId - 1;
            ctx.remoteOutput.addr = ctx.output[arg->rootId];
            ctx.remoteOutput.addr += ctx.myScratchOffset;
            ctx.remoteOutput.token = ctx.token[arg->rootId];
            ccu::Write(arg->channels[channelToRoot], ctx.remoteOutput, ctx.myScratchResult, ctx.mySliceSize,
                       ctx.events[0], 1);
            ccu::EventWait(ctx.events[0], 1);
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoRepeatTwoshot(ReduceMesh1DTwoShotMem2MemContext &ctx)
{
    InitScratchMem(ctx);
    ctx.flag = 0;

    CCU_WHILE(ctx.repeatNumVar != UINT64_MAX)
    {
        CCU_IF(ctx.flag != 0)
        {
            ctx.input[ctx.arg->rankId] += ctx.inputRepeatStride;
            ctx.output[ctx.arg->rootId] += ctx.outputRepeatStride;
        }
        CCU_CHK_RET(DoReduceScatter(ctx));
        CCU_CHK_RET(DoGather(ctx));
        ctx.repeatNumVar += ctx.constVar1;
        ctx.flag = 1;
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceMesh1DTwoShotMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceMesh1DTwoShotMem2Mem *>(arg);
    ReduceMesh1DTwoShotMem2MemContext ctx;
    ctx.arg = kernelArg;
    ctx.enginePool = 0;
    // moConfig/moRes 初始化为 0，GroupCopy 首次调用时由 AllocGoResource 惰性分配
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    HCCL_INFO("[CcuKernelReduceMesh1DTwoShotMem2Mem] run, rankId[%u], rankSize[%llu], rootId[%u]",
              kernelArg->rankId, kernelArg->rankSize, kernelArg->rootId);
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    PreSync(ctx);

    CCU_IF(ctx.normalSliceSize != 0)
    {
        CCU_CHK_RET(DoRepeatTwoshot(ctx));
    }

    PostSync(ctx);
    HCCL_INFO("[CcuKernelReduceMesh1DTwoShotMem2Mem] end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl
