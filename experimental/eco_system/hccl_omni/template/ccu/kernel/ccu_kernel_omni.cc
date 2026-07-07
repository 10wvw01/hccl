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

constexpr int CKE_IDX_0 = 0;
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

static CcuResult CreateReduceLoop(
    OmniContext &ctx, uint32_t size, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType)
{
    AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, OMNI_LOOP_COUNT);

    std::string loopType = ops_hccl::GetReduceTypeStr(dataType, opType);
    if (ctx.IsLoopEntityRegistered(loopType)) {
        return CCU_SUCCESS;
    }
    ctx.CreateLoopEntity(loopType);
    auto &loops = ctx.loopMap[loopType];

    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < LOOP_NUM; index++) {
        ctx.loopScratch[index].resize(size);

        uint32_t bufBase = index * ctx.moConfig.msInterleave;
        ccu::Event loopEvt = ctx.moRes.completedEvent[index];

        loops.body[index].reset(new ccu::Func(
            [&ctx, index, bufBase, loopEvt, size, expansionNum, usedBufNum, dataType, outputDataType, opType]() {
                for (uint32_t i = 0; i < size; i++) {
                    ccu::LocalCopy(
                        ctx.moRes.ccuBuf[bufBase + i], ctx.loopScratch[index][i], ctx.loopLen[index], loopEvt, 1 << i);
                }
                ccu::EventWait(loopEvt, (1 << size) - 1);

                if (size > 1) {
                    ccu::LocalReduce(&ctx.moRes.ccuBuf[bufBase], size, dataType, outputDataType, opType,
                        ctx.loopLen[index], loopEvt, 1);
                    ccu::EventWait(loopEvt, 1);
                }

                ccu::LocalCopy(ctx.loopDst[index], ctx.moRes.ccuBuf[bufBase], ctx.loopLenExp[index], loopEvt, 1);
                ccu::EventWait(loopEvt, 1);
            }));

        loops.loops[index].reset(new ccu::Loop(loops.loopParam[index], *loops.body[index]));
    }

    return CCU_SUCCESS;
}

static CcuResult ReduceLoopGroup(OmniContext &ctx, ccu::LocalAddr outDstOrg, std::vector<ccu::LocalAddr> &scratchOrg,
    HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType)
{
    const uint32_t size = scratchOrg.size();

    ccu::LocalAddr dst;
    dst.addr = outDstOrg.addr;
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
        sliceSize = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定loop0的外部LocalAddr和Variable
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopSrc[0].addr = dst.addr;
        ctx.loopSrc[0].token = dst.token;
        ctx.loopDst[0].addr = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0] = sliceSize;
        ctx.loopLenExp[0] = sliceSizeExpansion;

        ccu::Variable paraCfg;
        ccu::Variable offsetCfg;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopParam;
        std::vector<ccu::Loop> grpLoops{*loops.loops[0], *loops.loops[1]};
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
        ctx.loopSrc[0].addr = dst.addr;
        ctx.loopSrc[0].token = dst.token;
        ctx.loopDst[0].addr = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0] = ctx.goSize.residual;
        ctx.loopLenExp[0] = sliceSizeExpansion;

        ccu::Variable sliceSize;
        // n部分，再加p的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.residual;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.residual;
        }
        sliceSize = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定loop1参数 (n部分)
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[1][i].addr = scratch[i].addr;
            ctx.loopScratch[1][i].token = scratch[i].token;
        }
        ctx.loopSrc[1].addr = dst.addr;
        ctx.loopSrc[1].token = dst.token;
        ctx.loopDst[1].addr = dst.addr;
        ctx.loopDst[1].token = dst.token;
        ctx.loopLen[1] = sliceSize;
        ctx.loopLenExp[1] = sliceSizeExpansion;

        ccu::Variable loopCfg0;
        ccu::Variable loopCfg1;
        ccu::Variable offsetCfg;
        loopCfg0 = GetLoopParam(0, 0, 1);
        loopCfg1 = GetLoopParam(0, 0, 1);
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopCfg0;
        loops.loopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{*loops.loops[0], *loops.loops[1]};
        ccu::LoopGroup group(ctx.goSize.parallelParam, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    return CCU_SUCCESS;
}

static CcuResult InitResources(OmniContext &ctx)
{
    HCCL_DEBUG("[CcuKernelOmni] InitResources begin");

    const auto *arg = ctx.arg;
    ctx.sliceNum = arg->sendRecvInfo[0][0].sliceNum;

    ctx.sendCounts.resize(arg->rankGroup.size());
    ctx.recvCounts.resize(arg->rankGroup.size());
    ctx.sdispls.resize(arg->rankGroup.size());
    ctx.rdispls.resize(arg->rankGroup.size());
    ctx.sendRecvCountsInfo.resize(ctx.sliceNum);
    ctx.recvSliceData.resize(ctx.sliceNum);
    ctx.sendSdispls.resize(ctx.sliceNum);
    ctx.localSdispls.resize(ctx.sliceNum);

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
    ctx.constVar1 = 1;
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
    CCU_CHK_RET(ccu::LoadArg(ctx.syncIdx, argId++));

    // 全量数据的gosize
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argId++));

    for (uint64_t i = 0; i < arg->rankGroup.size(); i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.sendCounts[i], argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.recvCounts[i], argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sdispls[i], argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.rdispls[i], argId++));
    }

    for (uint32_t i = 0; i < ctx.sliceNum; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.recvSliceData[i], argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendSdispls[i], argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.localSdispls[i], argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvCountsInfo[i].tailSize, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvCountsInfo[i].loopNum, argId++));

        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvCountsInfo[i].tailGoSize.addrOffset, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvCountsInfo[i].tailGoSize.loopParam, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvCountsInfo[i].tailGoSize.parallelParam, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvCountsInfo[i].tailGoSize.residual, argId++));
    }

    HCCL_INFO("LoadArgs end rankid is %u", ctx.arg->rankId);
    return CCU_SUCCESS;
}

static CcuResult PreSync(OmniContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.input[ctx.rankId2Idx[ctx.arg->rankId]], OUTPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));

        ccu::Variable tmpAddr = ctx.output[ctx.rankId2Idx[ctx.arg->rankId]];
        tmpAddr += ctx.rdispls[arg->rankGroup[i]];
        CCU_CHK_RET(
            ccu::WriteVariableWithNotify(arg->channels[i], tmpAddr, OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));

        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.token[ctx.rankId2Idx[ctx.arg->rankId]], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
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

static ccu::Variable GetallAddrBySliceType(OmniContext &ctx, omni::BufferTypeTmp sliceType, uint32_t rankIdx,
    uint64_t sliceIdx, const std::vector<ccu::Variable> &sendSdispls, const std::vector<ccu::Variable> &localSdispls)
{
    (void)sendSdispls;
    ccu::Variable tmpAddr;
    if (sliceType == omni::BufferTypeTmp::INPUT) {
        tmpAddr = ctx.input[rankIdx];
        tmpAddr += localSdispls[sliceIdx];
    }
    if (sliceType == omni::BufferTypeTmp::OUTPUT) {
        tmpAddr = ctx.output[rankIdx];
        tmpAddr += localSdispls[sliceIdx];
    }
    if (sliceType == omni::BufferTypeTmp::HCCL_BUFFER) {
        tmpAddr = ctx.scratchAddr;
        tmpAddr += sendSdispls[sliceIdx];
    }

    return tmpAddr;
}

static CcuResult DoOpLocalCopy(OmniContext &ctx, const OmniSendRecvInfo &signalInfo)
{
    HCCL_DEBUG("groupCopy  begin rank id %u", ctx.arg->rankId);
    ccu::LocalAddr myOutput;
    ccu::LocalAddr myInput;
    myInput.token = ctx.token[ctx.rankId2Idx[ctx.arg->rankId]];
    myOutput.token = ctx.token[ctx.rankId2Idx[ctx.arg->rankId]];

    if (signalInfo.srcSliceInfo.size() != signalInfo.dstSliceInfo.size()) {
        HCCL_WARNING("[DoRepeatOmni] srd and dst num should be same");
    }

    ccu::Variable xnMaxTransportSize;
    xnMaxTransportSize = UB_MAX_TRANS_SIZE;

    for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
        HCCL_DEBUG("localcopy rank id %u, remoteRank is %u, src sliceIdx is %u, dst sliceIdx is %u", ctx.arg->rankId,
            signalInfo.dstSliceInfo[i].remoteRank, signalInfo.srcSliceInfo[i].sliceIdx,
            signalInfo.dstSliceInfo[i].sliceIdx);

        ccu::Variable loopNum = ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].loopNum;
        myInput.addr = GetallAddrBySliceType(ctx, signalInfo.srcSliceInfo[i].sliceType,
            ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank], signalInfo.srcSliceInfo[i].sliceIdx, ctx.sendSdispls,
            ctx.localSdispls);
        myOutput.addr = GetallAddrBySliceType(ctx, signalInfo.dstSliceInfo[i].sliceType,
            ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank], signalInfo.dstSliceInfo[i].sliceIdx, ctx.sendSdispls,
            ctx.localSdispls);

        CCU_WHILE(loopNum != UINT64_MAX)
        {
            CCU_IF(loopNum == UINT64_MAX - 1)
            {
                CCU_IF(ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].tailSize != 0)
                {
                    GroupCopy(
                        ctx, myOutput, myInput, ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].tailGoSize);
                }
            }

            CCU_IF(loopNum != UINT64_MAX - 1)
            {
                GroupCopy(ctx, myOutput, myInput, ctx.goSize);
                myInput.addr += xnMaxTransportSize;
                myOutput.addr += xnMaxTransportSize;
            }

            loopNum += ctx.constVar1;
        }
    }

    HCCL_DEBUG("groupCopy  end rank id %u", ctx.arg->rankId);
    return CCU_SUCCESS;
}

static CcuResult DoOpLocalReduce(OmniContext &ctx, const OmniSendRecvInfo &signalInfo)
{
    HCCL_DEBUG("ReduceLoopGroup  begin rank id %u", ctx.arg->rankId);
    ccu::LocalAddr myOutput;
    std::vector<ccu::LocalAddr> myInput;
    myOutput.token = ctx.token[ctx.rankId2Idx[ctx.arg->rankId]];

    if (signalInfo.dstSliceInfo.size() > 1) {
        HCCL_WARNING("[DoRepeatOmni] ReduceLoopGroup dstSliceInfo_ > 1");
    }

    for (auto &sliceInfo : signalInfo.srcSliceInfo) {
        ccu::LocalAddr tmpInput;
        tmpInput.token = ctx.token[ctx.rankId2Idx[sliceInfo.remoteRank]];

        tmpInput.addr = GetallAddrBySliceType(ctx, sliceInfo.sliceType, ctx.rankId2Idx[sliceInfo.remoteRank],
            sliceInfo.sliceIdx, ctx.sendSdispls, ctx.localSdispls);

        myInput.push_back(tmpInput);
    }

    for (auto &sliceInfo : signalInfo.dstSliceInfo) {
        myOutput.addr = GetallAddrBySliceType(ctx, sliceInfo.sliceType, ctx.rankId2Idx[sliceInfo.remoteRank],
            sliceInfo.sliceIdx, ctx.sendSdispls, ctx.localSdispls);
    }

    CCU_CHK_RET(ReduceLoopGroup(
        ctx, myOutput, myInput, signalInfo.inputDataType, signalInfo.outputDataType, signalInfo.reduceType));
    HCCL_DEBUG("ReduceLoopGroup end rank id %u", ctx.arg->rankId);
    return CCU_SUCCESS;
}

static CcuResult DoOpWriteNb(OmniContext &ctx, const OmniSendRecvInfo &signalInfo)
{
    ccu::Variable xnMaxTransportSize;
    xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
        ccu::RemoteAddr remoteDst;
        ccu::LocalAddr src;
        src.token = ctx.token[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
        remoteDst.token = ctx.token[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];

        ccu::Variable loopNum = ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].loopNum;
        uint64_t eventIdx = signalInfo.dstSliceInfo[i].remoteRecvRank >> REMOTE_RANKID_BIT;
        uint64_t rankIdx = signalInfo.dstSliceInfo[i].remoteRecvRank & ((1 << REMOTE_RANKID_BIT) - 1);
        src.addr = GetallAddrBySliceType(ctx, signalInfo.srcSliceInfo[i].sliceType,
            ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank], signalInfo.srcSliceInfo[i].sliceIdx, ctx.sendSdispls,
            ctx.localSdispls);
        remoteDst.addr = GetallAddrBySliceType(ctx, signalInfo.dstSliceInfo[i].sliceType,
            ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank], signalInfo.dstSliceInfo[i].sliceIdx, ctx.sendSdispls,
            ctx.localSdispls);
        CCU_WHILE(loopNum != UINT64_MAX)
        {
            CCU_IF(loopNum == UINT64_MAX - 1)
            {
                CCU_IF(ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].tailSize != 0)
                {
                    ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                    ccu::Write(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRecvRank], remoteDst, src,
                        ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].tailSize, ctx.events[eventIdx],
                        (1 << (rankIdx % BIT_NUM_PER_CKE)));
                }
            }

            CCU_IF(loopNum != UINT64_MAX - 1)
            {
                ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                ccu::Write(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRecvRank], remoteDst, src,
                    xnMaxTransportSize, ctx.events[eventIdx], (1 << (rankIdx % BIT_NUM_PER_CKE)));
                src.addr += xnMaxTransportSize;
                remoteDst.addr += xnMaxTransportSize;
            }

            loopNum += ctx.constVar1;
        }

        HCCL_DEBUG("WriteNb rank id %u, eventIdx is %llu, rankIdx %llu", ctx.arg->rankId, eventIdx, rankIdx);
    }

    HCCL_DEBUG("WriteNb  end rank id %u", ctx.arg->rankId);
    return CCU_SUCCESS;
}

static CcuResult DoOpWriteReduceNb(OmniContext &ctx, const OmniSendRecvInfo &signalInfo)
{
    HCCL_DEBUG("WriteReduceNb  begin rank id %u", ctx.arg->rankId);
    ccu::Variable xnMaxTransportSize;
    xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
        ccu::RemoteAddr remoteDst;
        ccu::LocalAddr src;
        src.token = ctx.token[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
        remoteDst.token = ctx.token[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];

        ccu::Variable loopNum = ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].loopNum;
        uint64_t eventIdx = signalInfo.dstSliceInfo[i].remoteRecvRank >> REMOTE_RANKID_BIT;
        uint64_t rankIdx = signalInfo.dstSliceInfo[i].remoteRecvRank & ((1 << REMOTE_RANKID_BIT) - 1);
        src.addr = GetallAddrBySliceType(ctx, signalInfo.srcSliceInfo[i].sliceType,
            ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank], signalInfo.srcSliceInfo[i].sliceIdx, ctx.sendSdispls,
            ctx.localSdispls);
        remoteDst.addr = GetallAddrBySliceType(ctx, signalInfo.dstSliceInfo[i].sliceType,
            ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank], signalInfo.dstSliceInfo[i].sliceIdx, ctx.sendSdispls,
            ctx.localSdispls);
        CCU_WHILE(loopNum != UINT64_MAX)
        {
            CCU_IF(loopNum == UINT64_MAX - 1)
            {
                CCU_IF(ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].tailSize != 0)
                {
                    ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                    ccu::WriteReduce(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRecvRank], remoteDst, src,
                        ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].tailSize, signalInfo.inputDataType,
                        signalInfo.reduceType, ctx.events[eventIdx], (1 << (rankIdx % BIT_NUM_PER_CKE)));
                }
            }

            CCU_IF(loopNum != UINT64_MAX - 1)
            {
                ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                ccu::WriteReduce(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRecvRank], remoteDst, src,
                    xnMaxTransportSize, signalInfo.inputDataType, signalInfo.reduceType, ctx.events[eventIdx],
                    (1 << (rankIdx % BIT_NUM_PER_CKE)));
                src.addr += xnMaxTransportSize;
                remoteDst.addr += xnMaxTransportSize;
            }

            loopNum += ctx.constVar1;
        }

        HCCL_DEBUG("WriteReduceNb rank id %u, eventIdx is %llu, rankIdx %llu", ctx.arg->rankId, eventIdx, rankIdx);
    }
    HCCL_DEBUG("WriteReduceNb  end rank id %u", ctx.arg->rankId);
    return CCU_SUCCESS;
}

static CcuResult DoOpReadNb(OmniContext &ctx, const OmniSendRecvInfo &signalInfo)
{
    HCCL_DEBUG("ReadNb  begin rank id %u", ctx.arg->rankId);
    ccu::Variable xnMaxTransportSize;
    xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
        ccu::RemoteAddr remoteDst;
        ccu::LocalAddr src;
        src.token = ctx.token[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
        remoteDst.token = ctx.token[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];

        ccu::Variable loopNum = ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].loopNum;
        uint64_t eventIdx = signalInfo.dstSliceInfo[i].remoteRecvRank >> REMOTE_RANKID_BIT;
        uint64_t rankIdx = signalInfo.dstSliceInfo[i].remoteRecvRank & ((1 << REMOTE_RANKID_BIT) - 1);
        src.addr = GetallAddrBySliceType(ctx, signalInfo.srcSliceInfo[i].sliceType,
            ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank], signalInfo.srcSliceInfo[i].sliceIdx, ctx.sendSdispls,
            ctx.localSdispls);
        remoteDst.addr = GetallAddrBySliceType(ctx, signalInfo.dstSliceInfo[i].sliceType,
            ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank], signalInfo.dstSliceInfo[i].sliceIdx, ctx.sendSdispls,
            ctx.localSdispls);
        CCU_WHILE(loopNum != UINT64_MAX)
        {
            CCU_IF(loopNum == UINT64_MAX - 1)
            {
                CCU_IF(ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].tailSize != 0)
                {
                    ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                    CCU_CHK_RET(ccu::Read(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRecvRank], src, remoteDst,
                        ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].tailSize, ctx.events[eventIdx],
                        (1 << (rankIdx % BIT_NUM_PER_CKE))));
                }
            }

            CCU_IF(loopNum != UINT64_MAX - 1)
            {
                ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                CCU_CHK_RET(ccu::Read(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRecvRank], src, remoteDst,
                    xnMaxTransportSize, ctx.events[eventIdx], (1 << (rankIdx % BIT_NUM_PER_CKE))));
                src.addr += xnMaxTransportSize;
                remoteDst.addr += xnMaxTransportSize;
            }

            loopNum += ctx.constVar1;
        }
    }
    HCCL_DEBUG("ReadNb  end rank id %u", ctx.arg->rankId);
    return CCU_SUCCESS;
}

static CcuResult DoOpReadReduceNb(OmniContext &ctx, const OmniSendRecvInfo &signalInfo)
{
    HCCL_DEBUG("ReadReduceNb  begin rank id %u", ctx.arg->rankId);
    ccu::Variable xnMaxTransportSize;
    xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    for (uint32_t i = 0; i < signalInfo.srcSliceInfo.size(); i++) {
        ccu::RemoteAddr remoteDst;
        ccu::LocalAddr src;
        src.token = ctx.token[ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank]];
        remoteDst.token = ctx.token[ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank]];

        ccu::Variable loopNum = ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].loopNum;
        uint64_t eventIdx = signalInfo.dstSliceInfo[i].remoteRecvRank >> REMOTE_RANKID_BIT;
        uint64_t rankIdx = signalInfo.dstSliceInfo[i].remoteRecvRank & ((1 << REMOTE_RANKID_BIT) - 1);
        src.addr = GetallAddrBySliceType(ctx, signalInfo.srcSliceInfo[i].sliceType,
            ctx.rankId2Idx[signalInfo.srcSliceInfo[i].remoteRank], signalInfo.srcSliceInfo[i].sliceIdx, ctx.sendSdispls,
            ctx.localSdispls);
        remoteDst.addr = GetallAddrBySliceType(ctx, signalInfo.dstSliceInfo[i].sliceType,
            ctx.rankId2Idx[signalInfo.dstSliceInfo[i].remoteRank], signalInfo.dstSliceInfo[i].sliceIdx, ctx.sendSdispls,
            ctx.localSdispls);
        CCU_WHILE(loopNum != UINT64_MAX)
        {
            CCU_IF(loopNum == UINT64_MAX - 1)
            {
                CCU_IF(ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].tailSize != 0)
                {
                    ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                    ccu::ReadReduce(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRecvRank], src, remoteDst,
                        ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[i].sliceIdx].tailSize, signalInfo.inputDataType,
                        signalInfo.reduceType, ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                }
            }

            CCU_IF(loopNum != UINT64_MAX - 1)
            {
                ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
                ccu::ReadReduce(ctx.rankId2Channel[signalInfo.dstSliceInfo[i].remoteRecvRank], src, remoteDst,
                    xnMaxTransportSize, signalInfo.inputDataType, signalInfo.reduceType, ctx.events[eventIdx],
                    1 << (rankIdx % BIT_NUM_PER_CKE));
                src.addr += xnMaxTransportSize;
                remoteDst.addr += xnMaxTransportSize;
            }

            loopNum += ctx.constVar1;
        }
    }
    HCCL_DEBUG("ReadReduceNb  end rank id %u", ctx.arg->rankId);
    return CCU_SUCCESS;
}

static CcuResult DoOpGroupBroadcast(OmniContext &ctx, const OmniSendRecvInfo &signalInfo)
{
    HCCL_DEBUG("GroupBroadcast  begin rank id %u", ctx.arg->rankId);
    if (signalInfo.srcSliceInfo.size() > 1) {
        HCCL_WARNING("[DoRepeatOmni] GroupBroadcast srcSliceInfo > 1");
    }

    if (signalInfo.srcSliceInfo.empty() || signalInfo.dstSliceInfo.empty()) {
        return CCU_SUCCESS;
    }

    ccu::Variable xnMaxTransportSize;
    xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    GroupOpSizeVars fullGoSize = ctx.goSize;
    ccu::Variable loopNum = ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[0].sliceIdx].loopNum;

    ccu::LocalAddr src;
    src.token = ctx.token[ctx.rankId2Idx[signalInfo.srcSliceInfo[0].remoteRank]];
    src.addr = GetallAddrBySliceType(ctx, signalInfo.srcSliceInfo[0].sliceType,
        ctx.rankId2Idx[signalInfo.srcSliceInfo[0].remoteRank], signalInfo.srcSliceInfo[0].sliceIdx, ctx.sendSdispls,
        ctx.localSdispls);

    std::vector<ccu::RemoteAddr> dst;
    ChannelHandle channels[CCU_MAX_RANK_SIZE];
    uint32_t channelIdx = 0;
    for (auto &sliceInfo : signalInfo.dstSliceInfo) {
        ccu::RemoteAddr tmpdst;
        tmpdst.token = ctx.token[ctx.rankId2Idx[sliceInfo.remoteRank]];
        tmpdst.addr = GetallAddrBySliceType(ctx, sliceInfo.sliceType, ctx.rankId2Idx[sliceInfo.remoteRank],
            sliceInfo.sliceIdx, ctx.sendSdispls, ctx.localSdispls);
        dst.push_back(tmpdst);
        channels[channelIdx++] = ctx.rankId2Channel[sliceInfo.remoteRank];
    }

    ccu::LocalAddr localdst;
    localdst.token = ctx.token[ctx.rankId2Idx[ctx.arg->rankId]];
    localdst.addr = GetallAddrBySliceType(ctx, signalInfo.dstSliceInfo[0].sliceType, ctx.rankId2Idx[ctx.arg->rankId],
        signalInfo.srcSliceInfo[0].sliceIdx, ctx.sendSdispls, ctx.localSdispls);

    CCU_WHILE(loopNum != UINT64_MAX)
    {
        CCU_IF(loopNum == UINT64_MAX - 1)
        {
            CCU_IF(ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[0].sliceIdx].tailSize != 0)
            {
                ctx.goSize = ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[0].sliceIdx].tailGoSize;
                CCU_CHK_RET(GroupBroadcast(ctx, channels, channelIdx, localdst, dst, src, ctx.goSize));
                ctx.goSize = fullGoSize;
            }
        }

        CCU_IF(loopNum != UINT64_MAX - 1)
        {
            CCU_CHK_RET(GroupBroadcast(ctx, channels, channelIdx, localdst, dst, src, ctx.goSize));
            src.addr += xnMaxTransportSize;
            localdst.addr += xnMaxTransportSize;
            for (uint32_t dstIdx = 0; dstIdx < signalInfo.dstSliceInfo.size(); dstIdx++) {
                dst[dstIdx].addr += xnMaxTransportSize;
            }
        }

        loopNum += ctx.constVar1;
    }

    HCCL_DEBUG("GroupBroadcast  end rank id %u", ctx.arg->rankId);
    return CCU_SUCCESS;
}

static CcuResult DoOpGroupReduce(OmniContext &ctx, const OmniSendRecvInfo &signalInfo)
{
    HCCL_DEBUG("GroupReduce  begin rank id %u", ctx.arg->rankId);
    if (signalInfo.dstSliceInfo.size() > 1) {
        HCCL_WARNING("[DoRepeatOmni] GroupReduce dstSliceInfo > 1");
    }

    if (signalInfo.srcSliceInfo.empty() || signalInfo.dstSliceInfo.empty()) {
        return CCU_SUCCESS;
    }

    ccu::Variable xnMaxTransportSize;
    xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    GroupOpSizeVars fullGoSize = ctx.goSize;
    ccu::Variable loopNum = ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[0].sliceIdx].loopNum;

    ccu::LocalAddr dst;
    dst.token = ctx.token[ctx.rankId2Idx[ctx.arg->rankId]];
    dst.addr = GetallAddrBySliceType(ctx, signalInfo.dstSliceInfo[0].sliceType,
        ctx.rankId2Idx[signalInfo.dstSliceInfo[0].remoteRank], signalInfo.dstSliceInfo[0].sliceIdx, ctx.sendSdispls,
        ctx.localSdispls);

    std::vector<ccu::RemoteAddr> src;
    ChannelHandle channels[CCU_MAX_RANK_SIZE];
    uint32_t channelIdx = 0;
    for (auto &sliceInfo : signalInfo.srcSliceInfo) {
        ccu::RemoteAddr tmpSrc;
        tmpSrc.token = ctx.token[ctx.rankId2Idx[sliceInfo.remoteRank]];
        tmpSrc.addr = GetallAddrBySliceType(ctx, sliceInfo.sliceType, ctx.rankId2Idx[sliceInfo.remoteRank],
            sliceInfo.sliceIdx, ctx.sendSdispls, ctx.localSdispls);
        src.push_back(tmpSrc);
        channels[channelIdx++] = ctx.rankId2Channel[sliceInfo.remoteRank];
    }

    ccu::LocalAddr localSrc;
    localSrc.token = ctx.token[ctx.rankId2Idx[ctx.arg->rankId]];
    localSrc.addr = GetallAddrBySliceType(ctx, signalInfo.srcSliceInfo[0].sliceType, ctx.rankId2Idx[ctx.arg->rankId],
        signalInfo.srcSliceInfo[0].sliceIdx, ctx.sendSdispls, ctx.localSdispls);

    CCU_WHILE(loopNum != UINT64_MAX)
    {
        CCU_IF(loopNum == UINT64_MAX - 1)
        {
            CCU_IF(ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[0].sliceIdx].tailSize != 0)
            {
                ctx.goSize = ctx.sendRecvCountsInfo[signalInfo.srcSliceInfo[0].sliceIdx].tailGoSize;
                GroupReduce(ctx, channels, channelIdx, dst, src, localSrc, ctx.goSize, signalInfo.inputDataType,
                    signalInfo.outputDataType, signalInfo.reduceType);
                ctx.goSize = fullGoSize;
            }
        }

        CCU_IF(loopNum != UINT64_MAX - 1)
        {
            GroupReduce(ctx, channels, channelIdx, dst, src, localSrc, ctx.goSize, signalInfo.inputDataType,
                signalInfo.outputDataType, signalInfo.reduceType);
            dst.addr += xnMaxTransportSize;
            localSrc.addr += xnMaxTransportSize;
            for (uint32_t srcIdx = 0; srcIdx < signalInfo.srcSliceInfo.size(); srcIdx++) {
                src[srcIdx].addr += xnMaxTransportSize;
            }
        }

        loopNum += ctx.constVar1;
    }

    HCCL_DEBUG("GroupReduce end rank id %u", ctx.arg->rankId);
    return CCU_SUCCESS;
}

static CcuResult DoOpWaitEvent(OmniContext &ctx, const OmniSendRecvInfo &signalInfo)
{
    for (uint32_t i = 0; i < signalInfo.dstSliceInfo.size(); i++) {
        auto &sliceInfo = signalInfo.dstSliceInfo[i];
        uint64_t eventIdx = sliceInfo.remoteRecvRank >> REMOTE_RANKID_BIT;
        uint64_t rankIdx = sliceInfo.remoteRecvRank & ((1 << REMOTE_RANKID_BIT) - 1);
        HCCL_DEBUG("WaitEvent rank id %u, remoteRecvRank %llu, eventIdx is %llu, rankIdx %llu", ctx.arg->rankId,
            sliceInfo.remoteRecvRank, eventIdx, rankIdx);

        CCU_CHK_RET(ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE)));
        ccu::EventWait(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
    }
    HCCL_DEBUG("WaitEvent end rank id %u", ctx.arg->rankId);
    return CCU_SUCCESS;
}

static CcuResult DoRepeatOmni(OmniContext &ctx)
{
    const auto *arg = ctx.arg;
    HCCL_DEBUG("DoRepeatOmni begin rankid is %u, sendRecvInfo size %llu", ctx.arg->rankId, arg->sendRecvInfo.size());

    for (uint64_t i = 0; i < arg->sendRecvInfo.size(); i++) {
        CCU_IF(ctx.syncIdx == i)
        {
            HCCL_DEBUG("sync is %llu, rank id %u", i, ctx.arg->rankId);
            for (auto &signalInfo : arg->sendRecvInfo[i]) {
                if (signalInfo.opType == omni::OP_LOCAL_COPY) {
                    CCU_CHK_RET(DoOpLocalCopy(ctx, signalInfo));
                }

                if (signalInfo.opType == omni::OP_LOCAL_REDUCE) {
                    CCU_CHK_RET(DoOpLocalReduce(ctx, signalInfo));
                }

                if (signalInfo.opType == omni::OP_SEND_RECV_WRITE || signalInfo.opType == omni::OP_SEND_WRITE) {
                    CCU_CHK_RET(DoOpWriteNb(ctx, signalInfo));
                }

                if (signalInfo.opType == omni::OP_RECV_WRITE) {
                    HCCL_DEBUG("recvWrite not support");
                }

                if (signalInfo.opType == omni::OP_SEND_RECV_WRITE_REDUCE
                    || signalInfo.opType == omni::OP_SEND_WRITE_REDUCE) {
                    CCU_CHK_RET(DoOpWriteReduceNb(ctx, signalInfo));
                }

                if (signalInfo.opType == omni::OP_RECV_WRITE_REDUCE) {
                    HCCL_DEBUG("OP_RECV_WRITE_REDUCE not support");
                }

                if (signalInfo.opType == omni::OP_SEND_RECV_READ || signalInfo.opType == omni::OP_RECV_READ) {
                    CCU_CHK_RET(DoOpReadNb(ctx, signalInfo));
                }

                if (signalInfo.opType == omni::OP_SEND_READ) {
                    HCCL_DEBUG("OP_SEND_READ not support");
                }

                if (signalInfo.opType == omni::OP_SEND_RECV_READ_REDUCE
                    || signalInfo.opType == omni::OP_RECV_READ_REDUCE) {
                    CCU_CHK_RET(DoOpReadReduceNb(ctx, signalInfo));
                }

                if (signalInfo.opType == omni::OP_SEND_READ_REDUCE) {
                    HCCL_DEBUG("OP_SEND_READ_REDUCE not support");
                }

                if (signalInfo.opType == omni::OP_GROUP_BROAD_CAST) {
                    CCU_CHK_RET(DoOpGroupBroadcast(ctx, signalInfo));
                }

                if (signalInfo.opType == omni::OP_GROUP_REDUCE) {
                    CCU_CHK_RET(DoOpGroupReduce(ctx, signalInfo));
                }

                if (signalInfo.opType == omni::OP_WAIT_EVENT) {
                    CCU_CHK_RET(DoOpWaitEvent(ctx, signalInfo));
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

} // namespace ops_hccl