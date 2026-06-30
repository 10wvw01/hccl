/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_mesh1d_mem2mem.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int POST_SYNC_ID = 3;
constexpr uint16_t BIT_NUM_PER_CKE = 16;
constexpr uint32_t INVALID_CHANNEL_IDX = static_cast<uint32_t>(-1);

static CcuResult ParseKernelArg(AllGatherMesh1DMem2MemContext &ctx, CcuKernelArgAllGatherMesh1DMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    return CCU_SUCCESS;
}

static bool HasSharedChannel(const AllGatherMesh1DMem2MemContext &ctx, uint64_t peerId)
{
    return peerId < ctx.sharedChannelIdxByRank.size() &&
        ctx.sharedChannelIdxByRank[peerId] != INVALID_CHANNEL_IDX &&
        ctx.sharedChannelIdxByRank[peerId] < ctx.arg->channelCount;
}

static bool HasAnySharedChannel(const AllGatherMesh1DMem2MemContext &ctx)
{
    for (uint64_t peerId = 0; peerId < ctx.arg->rankSize; peerId++) {
        if (peerId != ctx.arg->rankId && HasSharedChannel(ctx, peerId)) {
            return true;
        }
    }
    return false;
}

static CcuResult InitChannelIdxByRank(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankId >= arg->rankSize) {
        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] rankId[%u] is invalid, rankSize[%llu]",
            arg->rankId, arg->rankSize);
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.mainChannelIdxByRank = arg->mainChannelIdxByRank;
    ctx.sharedChannelIdxByRank = arg->sharedChannelIdxByRank;
    if (ctx.mainChannelIdxByRank.empty()) {
        ctx.mainChannelIdxByRank.assign(arg->rankSize, INVALID_CHANNEL_IDX);
        uint32_t channelIdx = 0;
        for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
            if (peerId == arg->rankId) {
                continue;
            }
            if (channelIdx >= arg->channelCount) {
                HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] insufficient channels, channelIdx[%u], "
                           "channelCount[%u], peerId[%llu]",
                    channelIdx, arg->channelCount, peerId);
                return CcuResult::CCU_E_INTERNAL;
            }
            ctx.mainChannelIdxByRank[peerId] = channelIdx++;
        }
    }
    if (ctx.sharedChannelIdxByRank.empty()) {
        ctx.sharedChannelIdxByRank.assign(arg->rankSize, INVALID_CHANNEL_IDX);
    }

    if (ctx.mainChannelIdxByRank.size() != arg->rankSize || ctx.sharedChannelIdxByRank.size() != arg->rankSize) {
        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] invalid channel idx size, main[%zu], shared[%zu], "
                   "rankSize[%llu]",
            ctx.mainChannelIdxByRank.size(), ctx.sharedChannelIdxByRank.size(), arg->rankSize);
        return CcuResult::CCU_E_INTERNAL;
    }
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            continue;
        }
        if (ctx.mainChannelIdxByRank[peerId] >= arg->channelCount) {
            HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] invalid main channel idx[%u], channelCount[%u], "
                       "peerId[%llu]",
                ctx.mainChannelIdxByRank[peerId], arg->channelCount, peerId);
            return CcuResult::CCU_E_INTERNAL;
        }
        if (ctx.sharedChannelIdxByRank[peerId] != INVALID_CHANNEL_IDX &&
            ctx.sharedChannelIdxByRank[peerId] >= arg->channelCount) {
            HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] invalid shared channel idx[%u], channelCount[%u], "
                       "peerId[%llu]",
                ctx.sharedChannelIdxByRank[peerId], arg->channelCount, peerId);
            return CcuResult::CCU_E_INTERNAL;
        }
    }
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] channels.size: [%u]", arg->channelCount);

    CCU_CHK_RET(InitChannelIdxByRank(ctx));
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    if (HasAnySharedChannel(ctx)) {
        ctx.sharedOutput.resize(arg->rankSize);
        ctx.sharedToken.resize(arg->rankSize);
    }

    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) {
            uint32_t mainChannelIdx = ctx.mainChannelIdxByRank[peerId];
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(
                arg->channels[mainChannelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(
                arg->channels[mainChannelIdx], TOKEN_XN_ID);
            if (HasSharedChannel(ctx, peerId)) {
                uint32_t sharedChannelIdx = ctx.sharedChannelIdxByRank[peerId];
                ctx.sharedOutput[peerId] = ccu::GetResByChannel<ccu::Variable>(
                    arg->channels[sharedChannelIdx], OUTPUT_XN_ID);
                ctx.sharedToken[peerId] = ccu::GetResByChannel<ccu::Variable>(
                    arg->channels[sharedChannelIdx], TOKEN_XN_ID);
            }
        }
    }

    const uint32_t eventNum = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    ctx.events.resize(AG_UNROLL_NUM * eventNum);
    if (HasAnySharedChannel(ctx)) {
        ctx.sharedEvents.resize(AG_UNROLL_NUM * eventNum);
    }
    ctx.sharedEventMasks.assign(eventNum, 0);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId || !HasSharedChannel(ctx, peerId)) {
            continue;
        }
        uint16_t eventIdx = peerId / BIT_NUM_PER_CKE;
        uint16_t rankMask = 1 << (peerId % BIT_NUM_PER_CKE);
        ctx.sharedEventMasks[eventIdx] |= rankMask;
    }

    ctx.resourceAllocated = false;

    ctx.constVar1 = 1;
    ctx.repeatTimeflag = 0;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.tmpRepeatNum, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argId++)); 
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.mainSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sharedSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argId++));

    return CCU_SUCCESS;
}

static CcuResult PreSync(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllGatherMesh1DMem2MemContext &ctx)
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

static CcuResult InitAllGatherAddr(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.src.addr = ctx.input;
    ctx.src.addr += ctx.currentRankSliceInputOffset;
    ctx.src.token = ctx.token[arg->rankId];

    ctx.sharedSrc.addr = ctx.src.addr;
    ctx.sharedSrc.addr += ctx.mainSliceSize;
    ctx.sharedSrc.token = ctx.token[arg->rankId];

    ctx.src_loccopy.addr = ctx.input;
    ctx.src_loccopy.addr += ctx.currentRankSliceInputOffset;
    ctx.src_loccopy.token = ctx.token[arg->rankId];

    ctx.dst.resize(arg->rankSize);
    if (HasAnySharedChannel(ctx)) {
        ctx.sharedDst.resize(arg->rankSize);
    }
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            ctx.localDst.addr = ctx.output[arg->rankId];
            ctx.localDst.addr += ctx.currentRankSliceOutputOffset;
            ctx.localDst.token = ctx.token[arg->rankId];
        } else {
            ctx.dst[rankIdx].addr = ctx.output[rankIdx];
            ctx.dst[rankIdx].addr += ctx.currentRankSliceOutputOffset;
            ctx.dst[rankIdx].token = ctx.token[rankIdx];
            if (HasSharedChannel(ctx, rankIdx)) {
                ctx.sharedDst[rankIdx].addr = ctx.sharedOutput[rankIdx];
                ctx.sharedDst[rankIdx].addr += ctx.currentRankSliceOutputOffset;
                ctx.sharedDst[rankIdx].addr += ctx.mainSliceSize;
                ctx.sharedDst[rankIdx].token = ctx.sharedToken[rankIdx];
            }
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherWrite(AllGatherMesh1DMem2MemContext &ctx, const ccu::LocalAddr &src,
    const std::vector<ccu::RemoteAddr> &dst, const ccu::Variable &sliceSize, uint32_t unrollIdx)
{
    const auto *arg = ctx.arg;
    uint32_t numEventsPerIter = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;

    CCU_IF(ctx.sharedSliceSize != 0)
    {
        for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
            if (rankIdx == arg->rankId || !HasSharedChannel(ctx, rankIdx)) {
                continue;
            }
            uint32_t eventIdx = unrollIdx * numEventsPerIter + rankIdx / BIT_NUM_PER_CKE;
            uint16_t rankMask = 1 << (rankIdx % BIT_NUM_PER_CKE);
            uint32_t sharedChannelIdx = ctx.sharedChannelIdxByRank[rankIdx];
            CCU_CHK_RET(ccu::Write(arg->channels[sharedChannelIdx], ctx.sharedDst[rankIdx],
                ctx.sharedSrc, ctx.sharedSliceSize, ctx.sharedEvents[eventIdx], rankMask));
        }
    }

    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        uint32_t eventIdx = unrollIdx * numEventsPerIter + rankIdx / BIT_NUM_PER_CKE;
        uint16_t rankMask = 1 << (rankIdx % BIT_NUM_PER_CKE);
        if (rankIdx == arg->rankId) {
            CCU_CHK_RET(ccu::EventRecord(ctx.events[eventIdx], rankMask));
        } else {
            uint32_t mainChannelIdx = ctx.mainChannelIdxByRank[rankIdx];
            if (HasSharedChannel(ctx, rankIdx)) {
                CCU_IF(ctx.sharedSliceSize != 0)
                {
                    CCU_CHK_RET(ccu::Write(arg->channels[mainChannelIdx], dst[rankIdx],
                        src, ctx.mainSliceSize, ctx.events[eventIdx], rankMask));
                }
                CCU_IF(ctx.sharedSliceSize == 0)
                {
                    CCU_CHK_RET(ccu::Write(arg->channels[mainChannelIdx], dst[rankIdx],
                        src, sliceSize, ctx.events[eventIdx], rankMask));
                }
            } else {
                CCU_CHK_RET(ccu::Write(arg->channels[mainChannelIdx], dst[rankIdx],
                    src, sliceSize, ctx.events[eventIdx], rankMask));
            }
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherWait(AllGatherMesh1DMem2MemContext &ctx, uint32_t unrollIdx)
{
    const auto *arg = ctx.arg;
    uint32_t numEventsPerIter = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;

    for (uint32_t i = 0; i < numEventsPerIter; i++) {
        uint32_t eventIdx = unrollIdx * numEventsPerIter + i;
        uint16_t eventMask;
        if (i == numEventsPerIter - 1) {
            if (arg->rankSize % BIT_NUM_PER_CKE == 0) {
                eventMask = (1 << BIT_NUM_PER_CKE) - 1;
            } else {
                eventMask = (1 << (arg->rankSize % BIT_NUM_PER_CKE)) - 1;
            }
        } else {
            eventMask = (1 << BIT_NUM_PER_CKE) - 1;
        }
        CCU_CHK_RET(ccu::EventWait(ctx.events[eventIdx], eventMask));
    }
    CCU_IF(ctx.sharedSliceSize != 0)
    {
        for (uint32_t i = 0; i < numEventsPerIter; i++) {
            if (ctx.sharedEventMasks[i] != 0) {
                uint32_t eventIdx = unrollIdx * numEventsPerIter + i;
                CCU_CHK_RET(ccu::EventWait(ctx.sharedEvents[eventIdx], ctx.sharedEventMasks[i]));
            }
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherGroupCopy(AllGatherMesh1DMem2MemContext &ctx)
{
    CCU_IF(ctx.isInputOutputEqual == 0)
    {
        CCU_IF(ctx.groupCopyRepeatNum != UINT64_MAX)
        {
            ctx.repeatTimeflag = 0;
            CCU_WHILE(ctx.groupCopyRepeatNum != UINT64_MAX)
            {
                ctx.groupCopyRepeatNum += ctx.constVar1;
                CCU_IF(ctx.repeatTimeflag != 0)
                {
                    ctx.localDst.addr += ctx.outputRepeatStride;
                    ctx.src_loccopy.addr += ctx.inputRepeatStride;
                }
                CCU_CHK_RET(GroupCopy(ctx, ctx.localDst, ctx.src_loccopy, ctx.goSize));
                ctx.repeatTimeflag = 1;
            }
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGather(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    CCU_CHK_RET(InitAllGatherAddr(ctx));
    ctx.waitRepeatNum = ctx.tmpRepeatNum;
    ctx.groupCopyRepeatNum = ctx.tmpRepeatNum;

    // Phase 1: 先下发所有WriteNb（非阻塞，event错开），不包含GroupCopy
    CCU_IF(ctx.tmpRepeatNum != UINT64_MAX)
    {
        ctx.tmpRepeatNum += ctx.constVar1;
        CCU_CHK_RET(DoAllGatherWrite(ctx, ctx.src, ctx.dst, ctx.normalSliceSize, 0));
    }

    for (uint32_t i = 1; i < AG_UNROLL_NUM; i++) {
        CCU_IF(ctx.tmpRepeatNum != UINT64_MAX)
        {
            ctx.tmpRepeatNum += ctx.constVar1;
            ctx.src.addr += ctx.inputRepeatStride;
            ctx.sharedSrc.addr += ctx.inputRepeatStride;
            for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
                if (rankIdx != arg->rankId) {
                    ctx.dst[rankIdx].addr += ctx.outputRepeatStride;
                    if (HasSharedChannel(ctx, rankIdx)) {
                        ctx.sharedDst[rankIdx].addr += ctx.outputRepeatStride;
                    }
                }
            }
            CCU_CHK_RET(DoAllGatherWrite(ctx, ctx.src, ctx.dst, ctx.normalSliceSize, i));
        }
    }

    // Phase 2: GroupCopy使用CCU_WHILE
    CCU_CHK_RET(DoAllGatherGroupCopy(ctx));

    // Phase 3: 批量WaitEvent
    for (uint32_t i = 0; i < AG_UNROLL_NUM; i++) {
        CCU_IF(ctx.waitRepeatNum != UINT64_MAX)
        {
            ctx.waitRepeatNum += ctx.constVar1;
            CCU_CHK_RET(DoAllGatherWait(ctx, i));
        }
    }

    return CCU_SUCCESS;
}

CcuResult CcuAllGatherMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1DMem2Mem *>(arg);

    AllGatherMesh1DMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] AllGatherMesh1DMem2Mem run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    ctx.sliceSize = (kernelArg->rankId == (kernelArg->rankSize - 1)) ? ctx.lastSliceSize : ctx.normalSliceSize;
    // sliceSize == 0时不需要执行AllGather，只需前后同步
    CCU_IF(ctx.sliceSize != 0) {
        CCU_CHK_RET(DoRepeatAllGather(ctx));
    }

    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] AllGatherMesh1DMem2Mem end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
