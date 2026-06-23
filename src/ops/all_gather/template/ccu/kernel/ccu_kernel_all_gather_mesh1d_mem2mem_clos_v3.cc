/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_mesh1d_mem2mem_clos_v3.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int POST_SYNC_ID = 3;
constexpr uint16_t BIT_NUM_PER_CKE = 16;
constexpr uint32_t INVALID_CHANNEL_IDX = static_cast<uint32_t>(-1);

static CcuResult ParseKernelArg(AllGatherMesh1DMem2MemClosV3Context &ctx, CcuKernelArgAllGatherMesh1DMem2MemClosV3 *kernelArg)
{
    ctx.arg = kernelArg;
    return CCU_SUCCESS;
}

static bool HasSharedChannel(const AllGatherMesh1DMem2MemClosV3Context &ctx, uint64_t peerId)
{
    return peerId < ctx.sharedChannelIdxByRank.size() &&
        ctx.sharedChannelIdxByRank[peerId] != INVALID_CHANNEL_IDX &&
        ctx.sharedChannelIdxByRank[peerId] < ctx.arg->channelCount;
}

static CcuResult InitChannelIdxByRank(AllGatherMesh1DMem2MemClosV3Context &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankId >= arg->rankSize) {
        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2MemClosV3] rankId[%u] is invalid, rankSize[%llu]",
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
                HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2MemClosV3] insufficient channels, channelIdx[%u], "
                    "channelCount[%u], peerId[%llu]", channelIdx, arg->channelCount, peerId);
                return CcuResult::CCU_E_INTERNAL;
            }
            ctx.mainChannelIdxByRank[peerId] = channelIdx++;
        }
    }
    if (ctx.sharedChannelIdxByRank.empty()) {
        ctx.sharedChannelIdxByRank.assign(arg->rankSize, INVALID_CHANNEL_IDX);
    }

    if (ctx.mainChannelIdxByRank.size() != arg->rankSize || ctx.sharedChannelIdxByRank.size() != arg->rankSize) {
        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2MemClosV3] invalid channel idx size, main[%zu], shared[%zu], "
            "rankSize[%llu]", ctx.mainChannelIdxByRank.size(), ctx.sharedChannelIdxByRank.size(), arg->rankSize);
        return CcuResult::CCU_E_INTERNAL;
    }
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            continue;
        }
        if (ctx.mainChannelIdxByRank[peerId] >= arg->channelCount) {
            HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2MemClosV3] invalid main channel idx[%u], channelCount[%u], "
                "peerId[%llu]", ctx.mainChannelIdxByRank[peerId], arg->channelCount, peerId);
            return CcuResult::CCU_E_INTERNAL;
        }
        if (ctx.sharedChannelIdxByRank[peerId] != INVALID_CHANNEL_IDX &&
            ctx.sharedChannelIdxByRank[peerId] >= arg->channelCount) {
            HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2MemClosV3] invalid shared channel idx[%u], channelCount[%u], "
                "peerId[%llu]", ctx.sharedChannelIdxByRank[peerId], arg->channelCount, peerId);
            return CcuResult::CCU_E_INTERNAL;
        }
    }
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllGatherMesh1DMem2MemClosV3Context &ctx)
{
    const auto *arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2MemClosV3] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2MemClosV3] channels.size: [%u]", arg->channelCount);

    CCU_CHK_RET(InitChannelIdxByRank(ctx));
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    ctx.sharedOutput.resize(arg->rankSize);
    ctx.sharedToken.resize(arg->rankSize);

    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) {
            uint32_t mainChannelIdx = ctx.mainChannelIdxByRank[peerId];
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[mainChannelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[mainChannelIdx], TOKEN_XN_ID);
            if (HasSharedChannel(ctx, peerId)) {
                uint32_t sharedChannelIdx = ctx.sharedChannelIdxByRank[peerId];
                ctx.sharedOutput[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[sharedChannelIdx],
                    OUTPUT_XN_ID);
                ctx.sharedToken[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[sharedChannelIdx],
                    TOKEN_XN_ID);
            }
        }
    }

    const uint32_t eventNum = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    ctx.events.resize(eventNum);
    ctx.sharedEvents.resize(eventNum);
    ctx.sharedEventMasks.assign(eventNum, 0);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId || !HasSharedChannel(ctx, peerId)) {
            continue;
        }
        const uint16_t eventIdx = peerId / BIT_NUM_PER_CKE;
        const uint16_t rankMask = 1 << (peerId % BIT_NUM_PER_CKE);
        ctx.sharedEventMasks[eventIdx] |= rankMask;
    }

    ctx.resourceAllocated = false;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGatherMesh1DMem2MemClosV3Context &ctx)
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

static CcuResult PreSync(AllGatherMesh1DMem2MemClosV3Context &ctx)
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

static CcuResult PostSync(AllGatherMesh1DMem2MemClosV3Context &ctx)
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

static CcuResult DoAllGather(AllGatherMesh1DMem2MemClosV3Context &ctx, const ccu::LocalAddr &src,
    const ccu::LocalAddr &sharedSrc, const std::vector<ccu::RemoteAddr> &dst,
    const std::vector<ccu::RemoteAddr> &sharedDst)
{
    const auto *arg = ctx.arg;
    const uint32_t eventNum = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;

    CCU_IF(ctx.sharedSliceSize != 0)
    {
        for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
            if (rankIdx == arg->rankId || !HasSharedChannel(ctx, rankIdx)) {
                continue;
            }
            const uint16_t eventIdx = rankIdx / BIT_NUM_PER_CKE;
            const uint16_t rankMask = 1 << (rankIdx % BIT_NUM_PER_CKE);
            uint32_t sharedChannelIdx = ctx.sharedChannelIdxByRank[rankIdx];
            CCU_CHK_RET(ccu::Write(arg->channels[sharedChannelIdx], sharedDst[rankIdx],
                sharedSrc, ctx.sharedSliceSize, ctx.sharedEvents[eventIdx], rankMask));
        }
    }

    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        const uint16_t eventIdx = rankIdx / BIT_NUM_PER_CKE;
        const uint16_t rankMask = 1 << (rankIdx % BIT_NUM_PER_CKE);
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
                        src, ctx.normalSliceSize, ctx.events[eventIdx], rankMask));
                }
            } else {
                CCU_CHK_RET(ccu::Write(arg->channels[mainChannelIdx], dst[rankIdx],
                    src, ctx.normalSliceSize, ctx.events[eventIdx], rankMask));
            }
        }
    }
    CCU_IF(ctx.isInputOutputEqual == 0)
    {
        // 处理本卡情况
        CCU_CHK_RET(GroupCopy(ctx, ctx.localDst, src, ctx.goSize));
    }

    for (uint32_t i = 0; i < eventNum; i++) {
        uint16_t eventMask;
        if (i == eventNum - 1) {
            if (arg->rankSize % BIT_NUM_PER_CKE == 0) {
                eventMask = (1 << BIT_NUM_PER_CKE) - 1;
            } else {
                eventMask = (1 << (arg->rankSize % BIT_NUM_PER_CKE)) - 1;
            }
        } else {
            eventMask = (1 << BIT_NUM_PER_CKE) - 1;
        }
        CCU_CHK_RET(ccu::EventWait(ctx.events[i], eventMask));
    }
    CCU_IF(ctx.sharedSliceSize != 0)
    {
        for (uint32_t i = 0; i < eventNum; i++) {
            if (ctx.sharedEventMasks[i] != 0) {
                CCU_CHK_RET(ccu::EventWait(ctx.sharedEvents[i], ctx.sharedEventMasks[i]));
            }
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGather(AllGatherMesh1DMem2MemClosV3Context &ctx)
{
    const auto *arg = ctx.arg;

    ccu::LocalAddr src;
    ccu::LocalAddr sharedSrc;
    std::vector<ccu::RemoteAddr> dst;
    std::vector<ccu::RemoteAddr> sharedDst;

    dst.resize(arg->rankSize);
    sharedDst.resize(arg->rankSize);

    src.addr = ctx.input;
    src.addr += ctx.currentRankSliceInputOffset;
    src.token = ctx.token[arg->rankId];
    sharedSrc.addr = src.addr;
    sharedSrc.addr += ctx.mainSliceSize;
    sharedSrc.token = ctx.token[arg->rankId];

    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            ctx.localDst.addr = ctx.output[arg->rankId];
            ctx.localDst.addr += ctx.currentRankSliceOutputOffset;
            ctx.localDst.token = ctx.token[arg->rankId];
        } else {
            dst[rankIdx].addr = ctx.output[rankIdx];
            dst[rankIdx].addr += ctx.currentRankSliceOutputOffset;
            dst[rankIdx].token = ctx.token[rankIdx];
            if (HasSharedChannel(ctx, rankIdx)) {
                sharedDst[rankIdx].addr = ctx.sharedOutput[rankIdx];
                sharedDst[rankIdx].addr += ctx.currentRankSliceOutputOffset;
                sharedDst[rankIdx].addr += ctx.mainSliceSize;
                sharedDst[rankIdx].token = ctx.sharedToken[rankIdx];
            }
        }
    }

    ccu::Variable constVar1;
    ccu::Variable repeatTimeflag;
    constVar1 = 1;
    repeatTimeflag = 0;

    CCU_WHILE(ctx.tmpRepeatNum != UINT64_MAX)
    {
        ctx.tmpRepeatNum += constVar1;
        CCU_IF(repeatTimeflag != 0)
        {
            src.addr += ctx.inputRepeatStride;
            sharedSrc.addr += ctx.inputRepeatStride;
            for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
                if (rankIdx == arg->rankId) {
                    ctx.localDst.addr += ctx.outputRepeatStride;
                } else {
                    dst[rankIdx].addr += ctx.outputRepeatStride;
                    if (HasSharedChannel(ctx, rankIdx)) {
                        sharedDst[rankIdx].addr += ctx.outputRepeatStride;
                    }
                }
            }
        }
        CCU_IF(ctx.normalSliceSize != 0)
        {
            CCU_CHK_RET(DoAllGather(ctx, src, sharedSrc, dst, sharedDst));
        }
        repeatTimeflag = 1;
    }

    return CCU_SUCCESS;
}

CcuResult CcuAllGatherMesh1DMem2MemClosV3Kernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1DMem2MemClosV3 *>(arg);

    AllGatherMesh1DMem2MemClosV3Context ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2MemClosV3] AllGatherMesh1DMem2MemClosV3 run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    CCU_CHK_RET(DoRepeatAllGather(ctx));

    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2MemClosV3] AllGatherMesh1DMem2MemClosV3 end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
