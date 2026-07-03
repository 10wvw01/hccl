/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/ccu_kernel_omni.h"

namespace ops_hccl {

constexpr int CKE_IDX_0   = 0;
constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr uint16_t BIT_NUM_PER_CKE = 16;
constexpr uint16_t REMOTE_RANKID_BIT = 4;
constexpr uint32_t OMNI_LOOP_COUNT = 16;
const std::string LOCAL_REDUCE_LOOP_BLOCK_TAG{"OMNI"};
constexpr uint32_t LOOP_NUM = 2;
constexpr uint64_t REDUCE_MS_CNT = 8;
constexpr uint16_t REDUCE_SCATTER_2DIE_GROUP_REDUCE_MAX_PIECE_CNT = 8;
constexpr uint32_t REDUCE_SCATTER_LOOP_COUNT = 16;

static std::string GetLoopBlockTag(std::string loopType, int32_t index)
{
    return loopType + LOCAL_REDUCE_LOOP_BLOCK_TAG + std::to_string(index);
}

static CcuResult CreateReduceLoop(OmniContext &ctx, uint32_t size, HcclDataType dataType,
    HcclDataType outputDataType, HcclReduceOp opType)
{
    AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, OMNI_LOOP_COUNT);

    std::string loopType = ops_hccl::GetReduceTypeStr(dataType, opType);
    if (ctx.IsLoopEntityRegistered(loopType)) {
        return CCU_SUCCESS;
    }
    ctx.CreateLoopEntity(loopType);
    auto &loops = ctx.loopMap[loopType];

    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < LOOP_NUM; index++) {
        ctx.loopScratch[index].resize(size);

        uint32_t bufBase = index * ctx.moConfig.msInterleave;
        ccu::Event loopEvt = ctx.moRes.completedEvent[index];

        loops.body[index].reset(new ccu::Func(
            [&ctx, index, bufBase, loopEvt, size, expansionNum, usedBufNum, dataType, outputDataType, opType]() {
                for (uint32_t i = 0; i < size; i++) {
                    ccu::LocalCopy(ctx.moRes.ccuBuf[bufBase + i], ctx.loopScratch[index][i], ctx.loopLen[index], loopEvt, 1 << i);
                }
                ccu::EventWait(loopEvt, (1 << size) - 1);

                if (size > 1) {
                    ccu::LocalReduce(&ctx.moRes.ccuBuf[bufBase], size, dataType, outputDataType, opType, ctx.loopLen[index], loopEvt, 1);
                    ccu::EventWait(loopEvt, 1);
                }

                ccu::LocalCopy(ctx.loopDst[index], ctx.moRes.ccuBuf[bufBase], ctx.loopLenExp[index], loopEvt, 1);
                ccu::EventWait(loopEvt, 1);
        }));

        loops.loops[index].reset(
            new ccu::Loop(loops.loopParam[index], *loops.body[index]));
    }

    return CCU_SUCCESS;
}

static CcuResult ReduceLoopGroup(OmniContext &ctx, ccu::LocalAddr outDstOrg,
        std::vector<ccu::LocalAddr> &scratchOrg, HcclDataType dataType,
        HcclDataType outputDataType, HcclReduceOp opType)
{
    const uint32_t size = scratchOrg.size();

    ccu::LocalAddr dst;
    dst.addr  = outDstOrg.addr;
    dst.token = outDstOrg.token;

    std::vector<ccu::LocalAddr> scratch;
    for (uint32_t idx = 0; idx < size; idx++) {
        ccu::LocalAddr scratchAddr;
        scratchAddr.addr = scratchOrg[idx].addr;
        scratchAddr.token = scratchOrg[idx].token;
        scratch.push_back(scratchAddr);
    }

    CCU_CHK_RET(CreateReduceLoop(ctx, size, dataType, outputDataType, opType));
    auto &loops = ctx.loopMap[ops_hccl::GetReduceTypeStr(dataType, opType)];

    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    ccu::Variable sliceSizeExpansion;
    ccu::Variable tmp;
    if (expansionNum != 1) {
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }

    // m部分
    CCU_IF(ctx.goSize.loopParam != 0)
    {
        ccu::Variable loopParam;
        ccu::Variable sliceSize;
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam = loopParam + ctx.goSize.loopParam;
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定loop0的外部LocalAddr和Variable
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopSrc[0].addr  = dst.addr;
        ctx.loopSrc[0].token = dst.token;
        ctx.loopDst[0].addr  = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0]       = sliceSize;
        ctx.loopLenExp[0]    = sliceSizeExpansion;

        ccu::Variable paraCfg;
        ccu::Variable offsetCfg;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopParam;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0], *loops.loops[1] };
        ccu::LoopGroup group(paraCfg, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    CCU_IF(ctx.goSize.parallelParam != 0)
    {
        // p部分，加m的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.addrOffset;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion = sliceSizeExpansion + ctx.goSize.residual;
        }

        // 绑定loop0参数 (p部分)
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopSrc[0].addr  = dst.addr;
        ctx.loopSrc[0].token = dst.token;
        ctx.loopDst[0].addr  = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0]    = ctx.goSize.residual;
        ctx.loopLenExp[0] = sliceSizeExpansion;

        ccu::Variable sliceSize;
        // n部分，再加p的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.residual;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.residual;
        }
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定loop1参数 (n部分)
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[1][i].addr = scratch[i].addr;
            ctx.loopScratch[1][i].token = scratch[i].token;
        }
        ctx.loopSrc[1].addr  = dst.addr;
        ctx.loopSrc[1].token = dst.token;
        ctx.loopDst[1].addr  = dst.addr;
        ctx.loopDst[1].token = dst.token;
        ctx.loopLen[1]    = sliceSize;
        ctx.loopLenExp[1] = sliceSizeExpansion;

        ccu::Variable loopCfg0;
        ccu::Variable loopCfg1;
        ccu::Variable offsetCfg;
        loopCfg0 = GetLoopParam(0, 0, 1);
        loopCfg1 = GetLoopParam(0, 0, 1);
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopCfg0;
        loops.loopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0], *loops.loops[1] };
        ccu::LoopGroup group(ctx.goSize.parallelParam, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    return CCU_SUCCESS;
}


static CcuResult InitResources(OmniContext &ctx)
{
    HCCL_DEBUG("[CcuKernelOmni] InitResources begin");

    const auto *arg = ctx.arg;

    // 创建Variable，用于交换地址及token
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelOmni] RankId[%u] channels is empty", ctx.arg->rankId);
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.input.resize(arg->rankGroup.size());
    ctx.output.resize(arg->rankGroup.size());
    ctx.token.resize(arg->rankGroup.size());

    uint32_t idx = 0;
    for (uint32_t i = 0; i < arg->rankGroup.size(); i++) {
        if (arg->rankGroup[i] != ctx.arg->rankId) {
            ctx.rankId2Channel[arg->rankGroup[i]] = arg->channels[idx];
            ctx.rankId2Idx[arg->rankGroup[i]] = i;
            HCCL_DEBUG("InitResources myrank[%u] remote[%u] is %u", ctx.arg->rankId, arg->rankGroup[i], i);

            ctx.output[idx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[idx], OUTPUT_XN_ID);
            ctx.input[idx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[idx], INPUT_XN_ID);
            ctx.token[idx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[idx], TOKEN_XN_ID);
            idx++;
        } else { // 最后一个放的是本卡ID
            ctx.rankId2Idx[arg->rankGroup[i]] = i;
            HCCL_DEBUG("InitResources myrank[%u] remote[%u] is %u", ctx.arg->rankId, arg->rankGroup[i], i);
        }
    }

    ctx.moConfig.loopCount = REDUCE_SCATTER_LOOP_COUNT;
    ctx.moConfig.msInterleave = REDUCE_MS_CNT;
    ctx.moConfig.memSlice = CCU_MS_SIZE;

    ctx.resourceAllocated = false;

    ctx.events.resize(arg->rankGroup.size());

    HCCL_DEBUG("[CcuKernelOmni] rankId [%u] InitResources end", ctx.arg->rankId);

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(OmniContext &ctx)
{
    const auto *arg = ctx.arg;
    HCCL_INFO("LoadArgs begin rankid is %llu", arg->rankId);

    uint32_t argId = 0;
    HCCL_DEBUG("[LoadArgs] rankId [%llu] idx [%llu]", arg->rankId, ctx.rankId2Idx[ctx.arg->rankId]);

    CCU_CHK_RET(ccu::LoadArg(ctx.input[ctx.rankId2Idx[ctx.arg->rankId]], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.rankId2Idx[ctx.arg->rankId]], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.rankId2Idx[ctx.arg->rankId]], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.syncIdx, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argId++));

    HCCL_INFO("LoadArgs end rankid is %u", ctx.arg->rankId);
    return CCU_SUCCESS;
}

static CcuResult PreSync(OmniContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[ctx.rankId2Idx[ctx.arg->rankId]],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));

        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[ctx.rankId2Idx[ctx.arg->rankId]],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));

        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.rankId2Idx[ctx.arg->rankId]],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }


    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }

    HCCL_INFO("[CcuKernelOmni] PreSync success!");

    return CCU_SUCCESS;
}

static CcuResult PostSync(OmniContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }

    return CCU_SUCCESS;
}

static CcuResult DoRepeatOmni(OmniContext &ctx)
{
    const auto *arg = ctx.arg;
    HCCL_DEBUG("DoRepeatOmni begin rankid is %u, sendRecvInfo size %llu", ctx.arg->rankId, arg->sendRecvInfo.size());

    for (uint64_t i = 0; i < arg->sendRecvInfo.size(); i++) {
        CCU_IF(ctx.syncIdx == i) {
            HCCL_DEBUG("sync is %llu, rank id %u", i, ctx.arg->rankId);
            for (auto& signalInfo : arg->sendRecvInfo[i]) {
                if (signalInfo.opType == omni::OP_LOCAL_COPY) { // groupCopy
                    HCCL_DEBUG("groupCopy  begin rank id %u", ctx.arg->rankId);
                    ccu::LocalAddr myOutput;
                    ccu::LocalAddr myInput;
                    myInput.token = ctx.token[ctx.rankId2Idx[ctx.arg->rankId]];
                    myOutput.token = ctx.token[ctx.rankId2Idx[ctx.arg->rankId]];

                    if (signalInfo.srcSliceInfo.size() != signalInfo.dstSliceInfo.size()) {
                        HCCL_WARNING("[DoRepeatOmni] srd and dst num should be same");
                    }

                    for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {

                        HCCL_DEBUG("localcopy rank id %u, remoteRank is %u, src sliceIdx is %u, dst sliceIdx is %u",
                            ctx.arg->rankId, signalInfo.dstSliceInfo[i].remoteRank, signalInfo.srcSliceInfo[i].sliceIdx, signalInfo.dstSliceInfo[i].sliceIdx);

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::INPUT) {
                            myInput.addr = ctx.input[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::OUTPUT) {
                            myInput.addr = ctx.output[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            myInput.addr = ctx.scratchAddr;
                        }

                        for (uint64_t j = 0; j < signalInfo.srcSliceInfo[i].sliceIdx; j++) {
                            myInput.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::INPUT) { // input
                            myOutput.addr = ctx.input[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::OUTPUT) {
                            myOutput.addr = ctx.output[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            myOutput.addr = ctx.scratchAddr;
                        }

                        for (uint32_t j = 0; j < signalInfo.dstSliceInfo[i].sliceIdx; j++) {
                            HCCL_DEBUG("localcopy rank id %u, j [%u] signalInfo.dstSliceInfo[i].sliceIdx [%llu]", ctx.arg->rankId, j, signalInfo.dstSliceInfo[i].sliceIdx);
                            myOutput.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }

                        GroupCopy(ctx, myOutput, myInput, ctx.goSize);
                    }

                    HCCL_DEBUG("groupCopy  end rank id %u", ctx.arg->rankId);
                }

                if (signalInfo.opType == omni::OP_LOCAL_REDUCE) { // ReduceLoopGroup
                    HCCL_DEBUG("ReduceLoopGroup  begin rank id %u", ctx.arg->rankId);
                    ccu::LocalAddr myOutput;
                    std::vector<ccu::LocalAddr> myInput;
                    myOutput.token = ctx.token[ctx.rankId2Idx[ctx.arg->rankId]];

                    if (signalInfo.dstSliceInfo.size() > 1) {
                        HCCL_WARNING("[DoRepeatOmni] ReduceLoopGroup dstSliceInfo_ > 1");
                    }

                    for (auto& sliceInfo : signalInfo.srcSliceInfo) {
                        ccu::LocalAddr tmpInput;
                        tmpInput.token = ctx.token[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        if (sliceInfo.sliceType == omni::BufferTypeTmp::INPUT) { // input
                            tmpInput.addr = ctx.input[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        }

                        if (sliceInfo.sliceType == omni::BufferTypeTmp::OUTPUT) {
                            tmpInput.addr = ctx.output[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        }

                        if (sliceInfo.sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            tmpInput.addr = ctx.scratchAddr;
                        }

                        for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                            tmpInput.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }

                        myInput.push_back(tmpInput);
                    }

                    for (auto& sliceInfo : signalInfo.dstSliceInfo) {
                        if (sliceInfo.sliceType == omni::BufferTypeTmp::INPUT) { // input
                            myOutput.addr = ctx.input[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        }

                        if (sliceInfo.sliceType == omni::BufferTypeTmp::OUTPUT) {
                            myOutput.addr = ctx.output[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        }

                        if (sliceInfo.sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            myOutput.addr = ctx.scratchAddr;
                        }

                        for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                            myOutput.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }
                    }

                    CCU_CHK_RET(ReduceLoopGroup(ctx, myOutput, myInput, signalInfo.inputDataType, signalInfo.outputDataType, signalInfo.reduceType));
                    HCCL_DEBUG("ReduceLoopGroup end rank id %u", ctx.arg->rankId);
                }

                if (signalInfo.opType == omni::OP_SEND_RECV_WRITE || signalInfo.opType == omni::OP_SEND_WRITE) { // WriteNb
                    for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
                        ccu::RemoteAddr remoteDst;
                        ccu::LocalAddr src;
                        src.token = ctx.token[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
                        remoteDst.token = ctx.token[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::INPUT) {
                            src.addr = ctx.input[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::OUTPUT) {
                            src.addr = ctx.output[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            src.addr = ctx.scratchAddr;
                        }

                        for (uint32_t j = 0; j < signalInfo.srcSliceInfo[i].sliceIdx; j++) {
                            src.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::INPUT) {
                            remoteDst.addr = ctx.input[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::OUTPUT) {
                            remoteDst.addr = ctx.output[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            remoteDst.addr = ctx.scratchAddr;
                        }

                        for (uint32_t j = 0; j < signalInfo.dstSliceInfo[i].sliceIdx; j++) {
                            remoteDst.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }

                        uint64_t eventIdx = signalInfo.dstSliceInfo[i].remoteRecvRank >> REMOTE_RANKID_BIT; // 前面6位是event list的下标
                        uint64_t rankIdx = signalInfo.dstSliceInfo[i].remoteRecvRank & ((1 << REMOTE_RANKID_BIT) - 1); // 后4位为 eventid

                        // ctx.events[eventIdx].SetMask(1 << (rankIdx % BIT_NUM_PER_CKE));
                        // WriteNb(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRecvRank], remoteDst, src, ctx.sliceSize, ctx.events[eventIdx]);
                        ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                        ccu::Write(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRecvRank], remoteDst,
                            src, ctx.sliceSize, ctx.events[eventIdx], (1 << (rankIdx % BIT_NUM_PER_CKE)));
                        HCCL_DEBUG("WriteNb rank id %u, eventIdx is %llu, rankIdx %llu", ctx.arg->rankId, eventIdx, rankIdx);
                    }

                    HCCL_DEBUG("WriteNb  end rank id %u", ctx.arg->rankId);
                }

                if (signalInfo.opType == omni::OP_RECV_WRITE) { // recvWrite
                    // ccu 没有对应的函数
                    HCCL_DEBUG("recvWrite not support");
                }

                if (signalInfo.opType == omni::OP_SEND_RECV_WRITE_REDUCE || signalInfo.opType == omni::OP_SEND_WRITE_REDUCE) { // WriteReduceNb
                    HCCL_DEBUG("WriteReduceNb  begin rank id %u", ctx.arg->rankId);
                    for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
                        ccu::RemoteAddr remoteDst;
                        ccu::LocalAddr src;
                        src.token = ctx.token[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
                        remoteDst.token = ctx.token[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::INPUT) { // input
                            src.addr = ctx.input[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::OUTPUT) {
                            src.addr = ctx.output[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            src.addr = ctx.scratchAddr;
                        }

                        for (uint32_t j = 0; j < signalInfo.srcSliceInfo[i].sliceIdx; j++) {
                            src.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::INPUT) { // input
                            remoteDst.addr = ctx.input[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::OUTPUT) {
                            remoteDst.addr = ctx.output[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            remoteDst.addr = ctx.scratchAddr;
                        }

                        for (uint32_t j = 0; j < signalInfo.dstSliceInfo[i].sliceIdx; j++) {
                            remoteDst.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }

                        uint64_t eventIdx = signalInfo.dstSliceInfo[i].remoteRecvRank >> REMOTE_RANKID_BIT; // 前面6位是event list的下标
                        uint64_t rankIdx = signalInfo.dstSliceInfo[i].remoteRecvRank & ((1 << REMOTE_RANKID_BIT) - 1); // 后4位为 eventid

                        // ctx.events[eventIdx].SetMask(1 << (rankIdx % BIT_NUM_PER_CKE));
                        // WriteReduceNb(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRank], remoteDst, src, ctx.sliceSize, signalInfo.inputDataType, signalInfo.reduceType, ctx.events[eventIdx]);
                        ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                        ccu::WriteReduce(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRank], remoteDst, src,
                            ctx.sliceSize, signalInfo.inputDataType, signalInfo.reduceType, ctx.events[eventIdx], (1 << (rankIdx % BIT_NUM_PER_CKE)));
                        HCCL_DEBUG("WriteReduceNb rank id %u, eventIdx is %llu, rankIdx %llu", ctx.arg->rankId, eventIdx, rankIdx);
                    }
                    HCCL_DEBUG("WriteReduceNb  end rank id %u", ctx.arg->rankId);
                }

                if (signalInfo.opType == omni::OP_RECV_WRITE_REDUCE) { // OP_RECV_WRITE_REDUCE
                    // ccu 没有对应的函数
                    HCCL_DEBUG("OP_RECV_WRITE_REDUCE not support");
                }

                if (signalInfo.opType == omni::OP_SEND_RECV_READ || signalInfo.opType == omni::OP_RECV_READ) { // ReadNb
                    HCCL_DEBUG("ReadNb  begin rank id %u", ctx.arg->rankId);
                    for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
                        ccu::RemoteAddr remoteDst;
                        ccu::LocalAddr src;
                        src.token = ctx.token[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
                        remoteDst.token = ctx.token[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::INPUT) {
                            src.addr = ctx.input[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::OUTPUT) {
                            src.addr = ctx.output[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            src.addr = ctx.scratchAddr;
                        }

                        for (uint32_t j = 0; j < signalInfo.srcSliceInfo[i].sliceIdx; j++) {
                            src.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::INPUT) {
                            remoteDst.addr = ctx.input[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::OUTPUT) {
                            remoteDst.addr = ctx.output[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            remoteDst.addr = ctx.scratchAddr;
                        }

                        for (uint32_t j = 0; j < signalInfo.dstSliceInfo[i].sliceIdx; j++) {
                            remoteDst.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }

                        uint64_t eventIdx = signalInfo.dstSliceInfo[i].remoteRecvRank >> REMOTE_RANKID_BIT; // 前面6位是event list的下标
                        uint64_t rankIdx = signalInfo.dstSliceInfo[i].remoteRecvRank & ((1 << REMOTE_RANKID_BIT) - 1); // 后4位为 eventid

                        // ctx.events[eventIdx].SetMask(1 << (rankIdx % BIT_NUM_PER_CKE));
                        // ReadNb(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRank], src, remoteDst, ctx.sliceSize, ctx.events[eventIdx]);
                        ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                        CCU_CHK_RET(ccu::Read(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRank],
                            src, remoteDst, ctx.sliceSize, ctx.events[eventIdx], (1 << (rankIdx % BIT_NUM_PER_CKE))));

                    }
                    HCCL_DEBUG("ReadNb  end rank id %u", ctx.arg->rankId);
                }

                if (signalInfo.opType == omni::OP_SEND_READ) { // OP_SEND_READ
                    // ccu 没有对应的函数
                    HCCL_DEBUG("OP_SEND_READ not support");
                }

                if (signalInfo.opType == omni::OP_SEND_RECV_READ_REDUCE || signalInfo.opType == omni::OP_RECV_READ_REDUCE) { // ReadReduceNb
                    HCCL_DEBUG("ReadReduceNb  begin rank id %u", ctx.arg->rankId);
                    for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
                        ccu::RemoteAddr remoteDst;
                        ccu::LocalAddr src;
                        src.token = ctx.token[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
                        remoteDst.token = ctx.token[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::INPUT) {
                            src.addr = ctx.input[ctx.arg->rankId];
                        }

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::OUTPUT) {
                            src.addr = ctx.output[ctx.arg->rankId];
                        }

                        if (signalInfo.srcSliceInfo[i].sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            src.addr = ctx.scratchAddr;
                        }

                        for (uint32_t j = 0; j < signalInfo.srcSliceInfo[i].sliceIdx; j++) {
                            src.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::INPUT) {
                            remoteDst.addr = ctx.input[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::OUTPUT) {
                            remoteDst.addr = ctx.output[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];
                        }

                        if (signalInfo.dstSliceInfo[i].sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            remoteDst.addr = ctx.scratchAddr;
                        }

                        for (uint32_t j = 0; j < signalInfo.dstSliceInfo[i].sliceIdx; j++) {
                            remoteDst.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }

                        uint64_t eventIdx = signalInfo.dstSliceInfo[i].remoteRecvRank >> REMOTE_RANKID_BIT; // 前面6位是event list的下标
                        uint64_t rankIdx = signalInfo.dstSliceInfo[i].remoteRecvRank & ((1 << REMOTE_RANKID_BIT) - 1); // 后4位为 eventid

                        // ctx.events[eventIdx].SetMask(1 << (rankIdx % BIT_NUM_PER_CKE));
                        // ReadReduceNb(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRank], src, remoteDst, ctx.sliceSize, signalInfo.inputDataType, signalInfo.reduceType, ctx.events[eventIdx]);

                        ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                        ccu::ReadReduce(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRank], src, remoteDst,
                            ctx.sliceSize, signalInfo.inputDataType, signalInfo.reduceType, ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                    }
                    HCCL_DEBUG("ReadReduceNb  end rank id %u", ctx.arg->rankId);
                }

                if (signalInfo.opType == omni::OP_SEND_READ_REDUCE) { // OP_SEND_READ_REDUCE
                    // ccu 没有对应的函数
                    HCCL_DEBUG("OP_SEND_READ_REDUCE not support");
                }

                if (signalInfo.opType == omni::OP_GROUP_BROAD_CAST) { // GroupBroadcast
                    HCCL_DEBUG("GroupBroadcast  begin rank id %u", ctx.arg->rankId);
                    if (signalInfo.srcSliceInfo.size() > 1) {
                        HCCL_WARNING("[DoRepeatOmni] GroupBroadcast srcSliceInfo > 1");
                    }

                    ccu::LocalAddr src;
                    std::vector<ccu::RemoteAddr> dst;
                    src.token = ctx.token[signalInfo.srcSliceInfo[0].remoteRank];

                    for (auto& sliceInfo : signalInfo.srcSliceInfo) {
                        if (sliceInfo.sliceType == omni::BufferTypeTmp::INPUT) {
                            src.addr = ctx.input[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        }

                        if (sliceInfo.sliceType == omni::BufferTypeTmp::OUTPUT) {
                            src.addr = ctx.output[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        }

                        if (sliceInfo.sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            src.addr = ctx.scratchAddr;
                        }

                        for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                            src.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }
                    }

                    // std::vector<ChannelHandle> channels;
                    ChannelHandle channels[CCU_MAX_RANK_SIZE];
                    uint32_t channelIdx = 0;
                    for (auto& sliceInfo : signalInfo.dstSliceInfo) {
                        ccu::RemoteAddr tmpdst;
                        tmpdst.token = ctx.token[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        if (sliceInfo.sliceType == omni::BufferTypeTmp::INPUT) {
                            tmpdst.addr = ctx.input[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        }

                        if (sliceInfo.sliceType == omni::BufferTypeTmp::OUTPUT) {
                            tmpdst.addr = ctx.output[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        }

                        if (sliceInfo.sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            tmpdst.addr = ctx.scratchAddr;
                        }

                        for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                            tmpdst.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }

                        dst.push_back(tmpdst);
                        // channels.push_back(ctx.rankId2Channel[sliceInfo.remoteRank]);
                        channels[channelIdx++] = ctx.rankId2Channel[sliceInfo.remoteRank];
                    }

                    // 添加本卡信息加到最后
                    ccu::LocalAddr localdst;
                    localdst.token = ctx.token[ctx.rankId2Idx[ctx.arg->rankId]];

                    if (signalInfo.dstSliceInfo[0].sliceType == omni::BufferTypeTmp::INPUT) {
                        localdst.addr = ctx.input[ctx.arg->rankId];
                    }

                    if (signalInfo.dstSliceInfo[0].sliceType == omni::BufferTypeTmp::OUTPUT) {
                        localdst.addr = ctx.output[ctx.arg->rankId];
                    }

                    if (signalInfo.dstSliceInfo[0].sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                        localdst.addr = ctx.scratchAddr;
                    }

                    for (uint64_t i = 0; i < signalInfo.srcSliceInfo[0].sliceIdx; i++) {
                        localdst.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                    }

                    CCU_CHK_RET(GroupBroadcast(ctx, channels, channelIdx, localdst, dst, src, ctx.goSize));

                    HCCL_DEBUG("GroupBroadcast  end rank id %u", ctx.arg->rankId);
                }

                if (signalInfo.opType == omni::OP_GROUP_REDUCE) { // GroupReduce
                    HCCL_DEBUG("GroupReduce  begin rank id %u", ctx.arg->rankId);
                    if (signalInfo.dstSliceInfo.size() > 1) {
                        HCCL_WARNING("[DoRepeatOmni] GroupReduce dstSliceInfo > 1");
                    }

                    std::vector<ccu::RemoteAddr> src;
                    ccu::LocalAddr dst;
                    dst.token = ctx.token[ctx.rankId2Idx[ctx.arg->rankId]];

                    for (auto& sliceInfo : signalInfo.dstSliceInfo) {
                        if (sliceInfo.sliceType == omni::BufferTypeTmp::INPUT) {
                            dst.addr = ctx.input[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        }

                        if (sliceInfo.sliceType == omni::BufferTypeTmp::OUTPUT) {
                            dst.addr = ctx.output[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        }

                        if (sliceInfo.sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            dst.addr = ctx.scratchAddr;
                        }

                        for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                            dst.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }
                    }

                    // std::vector<ChannelHandle> channels;
                    ChannelHandle channels[CCU_MAX_RANK_SIZE];
                    uint32_t channelIdx = 0;
                    for (auto& sliceInfo : signalInfo.srcSliceInfo) {
                        ccu::RemoteAddr tmpdst;
                        tmpdst.token = ctx.token[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        if (sliceInfo.sliceType == omni::BufferTypeTmp::INPUT) {
                            tmpdst.addr = ctx.input[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        }

                        if (sliceInfo.sliceType == omni::BufferTypeTmp::OUTPUT) {
                            tmpdst.addr = ctx.output[ctx.rankId2Idx[sliceInfo.remoteRank]];
                        }

                        if (sliceInfo.sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                            tmpdst.addr = ctx.scratchAddr;
                        }

                        for (uint64_t i = 0; i < sliceInfo.sliceIdx; i++) {
                            tmpdst.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                        }

                        src.push_back(tmpdst);
                        // channels.push_back(ctx.rankId2Channel[sliceInfo.remoteRank]);
                        channels[channelIdx++] = ctx.rankId2Channel[sliceInfo.remoteRank];
                    }

                    // 添加本卡信息加到最后
                    ccu::LocalAddr localSrc;
                    localSrc.token = ctx.token[ctx.rankId2Idx[ctx.arg->rankId]];

                    if (signalInfo.dstSliceInfo[0].sliceType == omni::BufferTypeTmp::INPUT) {
                        localSrc.addr = ctx.input[ctx.arg->rankId];
                    }

                    if (signalInfo.dstSliceInfo[0].sliceType == omni::BufferTypeTmp::OUTPUT) {
                        localSrc.addr = ctx.output[ctx.arg->rankId];
                    }

                    if (signalInfo.dstSliceInfo[0].sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
                        localSrc.addr = ctx.scratchAddr;
                    }

                    for (uint64_t i = 0; i < signalInfo.dstSliceInfo[0].sliceIdx; i++) {
                        localSrc.addr += ctx.inputSize; // 一个sliceId 对应一份sliceSize
                    }

                    //self defined broadcast
                    // GroupReduce(channels, dst, src, ctx.goSize,signalInfo.inputDataType, signalInfo.outputDataType, signalInfo.reduceType);
                    GroupReduce(ctx, channels, channelIdx, dst, src, localSrc, ctx.goSize,
                        signalInfo.inputDataType, signalInfo.outputDataType, signalInfo.reduceType);

                    HCCL_DEBUG("GroupReduce end rank id %u", ctx.arg->rankId);
                }

                if (signalInfo.opType == omni::OP_WAIT_EVENT) { // WaitEvent
                    for (uint32_t i = 0; i < signalInfo.dstSliceInfo.size(); i++) {
                        auto &sliceInfo = signalInfo.dstSliceInfo[i];
                        uint64_t eventIdx = sliceInfo.remoteRecvRank >> REMOTE_RANKID_BIT; // 前面6位是event list的下标
                        uint64_t rankIdx = sliceInfo.remoteRecvRank & ((1 << REMOTE_RANKID_BIT) - 1); // 后4位为 eventid
                        // ctx.events[eventIdx].SetMask(1 << (rankIdx % BIT_NUM_PER_CKE));
                        HCCL_DEBUG("WaitEvent rank id %u, remoteRecvRank %llu, eventIdx is %llu, rankIdx %llu", ctx.arg->rankId, sliceInfo.remoteRecvRank, eventIdx, rankIdx);

                        CCU_CHK_RET(ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE)));
                        ccu::EventWait(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                    }
                    HCCL_DEBUG("WaitEvent end rank id %u", ctx.arg->rankId);
                }
            }
        }
    }

    HCCL_DEBUG("DoRepeatOmni end rankid is %u", ctx.arg->rankId);
    return CCU_SUCCESS;
}

static CcuResult ParseKernelArg(OmniContext &ctx, CcuKernelArgOmni *kernelArg)
{
    ctx.arg = kernelArg;
    return CCU_SUCCESS;
}

CcuResult CcuOmniKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgOmni *>(arg);

    OmniContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    HCCL_INFO("[CcuOmniKernel] omni kernel run");

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResources(ctx)); // 创建变量
    CCU_CHK_RET(LoadArgs(ctx));      // 加载变量 顺序跟args相同
    CCU_CHK_RET(PreSync(ctx));       // 前同步
    CCU_CHK_RET(DoRepeatOmni(ctx));  // 执行数据搬运
    CCU_CHK_RET(PostSync(ctx));      // 后同步

    HCCL_INFO("[CcuOmniKernel] RankId[%u] Omni run end.", ctx.arg->rankId);
    return CCU_SUCCESS;
}


}// namespace ops_hccl