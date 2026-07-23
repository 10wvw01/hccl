/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel.h"
#include "utils.h"

namespace ops_hccl_ar {

// 资源标识：用于通过channel交换远端输入地址和token
constexpr int INPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int POST_SYNC_ID = 3;

// ---------------------------------------------------------------------------
// 初始化本地拷贝（LoopGroup）所需的资源
// 与AllGather示例一致，使用ccu::LocalCopy做本卡input->output的拷贝
// ---------------------------------------------------------------------------
static void InitGroupCopyResources(AllReduceMesh1DMem2MemContext &ctx, ccu::LocalAddr *loopSrc,
                                    ccu::LocalAddr *loopDst, ccu::Variable *loopLen)
{
    if (!ctx.resourceAllocated) {
        ctx.moConfig.msInterleave = CCU_MS_INTERLEAVE;
        ctx.moConfig.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
        ctx.moConfig.memSlice = CCU_LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;

        ctx.moRes.eventCount = ctx.moConfig.loopCount;
        ctx.moRes.completedEvent = ccu::Array<ccu::Event>(ctx.moRes.eventCount);

        ctx.moRes.bufCount = ctx.moConfig.loopCount * ctx.moConfig.msInterleave;
        ctx.moRes.ccuBuf = ccu::Array<ccu::CcuBuffer>(ctx.moRes.bufCount);

        ctx.resourceAllocated = true;
    }

    std::string loopType = "localcopy";
    if (!ctx.IsLoopEntityRegistered(loopType)) {
        ctx.CreateLoopEntity(loopType);
        auto &entity = ctx.loopMap[loopType];
        for (uint32_t index = 0; index < 2; index++) {
            uint32_t bufBase = index * ctx.moConfig.msInterleave;
            ccu::Event loopEvt = ctx.moRes.completedEvent[index];
            entity.body[index].reset(new ccu::Func(
                [&ctx, index, bufBase, loopEvt, loopSrc, loopDst, loopLen]() {
                    ccu::LocalCopy(ctx.moRes.ccuBuf[bufBase], loopSrc[index], loopLen[index], loopEvt, 1);
                    ccu::EventWait(loopEvt, 1);
                    ccu::LocalCopy(loopDst[index], ctx.moRes.ccuBuf[bufBase], loopLen[index], loopEvt, 1);
                    ccu::EventWait(loopEvt, 1);
                }));
            entity.loops[index].reset(
                new ccu::Loop(entity.loopParam[index], *entity.body[index]));
        }
    }
}

// ---------------------------------------------------------------------------
// 本地拷贝：使用LoopGroup资源将src拷贝到dst
// ---------------------------------------------------------------------------
static CcuResult GroupCopy(AllReduceMesh1DMem2MemContext &ctx, ccu::LocalAddr dst, ccu::LocalAddr src,
                            GroupOpSizeVars goSize)
{
    ccu::LocalAddr loopSrc[2];
    ccu::LocalAddr loopDst[2];
    ccu::Variable loopLen[2];

    InitGroupCopyResources(ctx, loopSrc, loopDst, loopLen);

    auto &loops = ctx.loopMap["localcopy"];

    CCU_IF(goSize.addrOffset != 0)
    {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        ccu::Variable sliceSize;
        sliceSize = ctx.moConfig.memSlice;

        loopSrc[0].addr = src.addr;
        loopSrc[0].token = src.token;
        loopDst[0].addr = dst.addr;
        loopDst[0].token = dst.token;
        loopLen[0] = sliceSize;

        loops.loopParam[0] = loopParam;
        ccu::Variable paraCfg;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);

        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0] };
        ccu::LoopGroup group(paraCfg, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    CCU_IF(goSize.parallelParam != 0)
    {
        src.addr += goSize.addrOffset;
        dst.addr += goSize.addrOffset;

        loopSrc[0].addr = src.addr;
        loopSrc[0].token = src.token;
        loopDst[0].addr = dst.addr;
        loopDst[0].token = dst.token;
        loopLen[0] = goSize.residual;

        src.addr += goSize.residual;
        dst.addr += goSize.residual;

        ccu::Variable sliceSize;
        sliceSize = ctx.moConfig.memSlice;

        loopSrc[1].addr = src.addr;
        loopSrc[1].token = src.token;
        loopDst[1].addr = dst.addr;
        loopDst[1].token = dst.token;
        loopLen[1] = sliceSize;

        ccu::Variable loopCfg0;
        loopCfg0 = GetLoopParam(0, 0, 1);
        ccu::Variable loopCfg1;
        loopCfg1 = GetLoopParam(0, 0, 1);
        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopCfg0;
        loops.loopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0], *loops.loops[1] };
        ccu::LoopGroup group(goSize.parallelParam, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    return CCU_SUCCESS;
}

// ---------------------------------------------------------------------------
// 执行AllReduce传输：
// 1. 先将本卡input本地拷贝到output（初始化output为本卡数据）
// 2. 再对每个远端rank使用ccu::ReadReduce，将远端input数据归约到本卡output
//    ccu::ReadReduce会从远端地址读取数据，并与本地地址的数据做归约（如SUM），
//    结果写回本地地址。这是AllReduce与AllGather的核心区别：
//    AllGather使用ccu::Write将本卡数据写入远端，而AllReduce使用ccu::ReadReduce
//    从远端读取并归约到本地。
// ---------------------------------------------------------------------------
static CcuResult ExecuteAllReduceTransfer(AllReduceMesh1DMem2MemContext &ctx)
{
    ccu::LocalAddr localSrc;   // 本卡输入地址
    ccu::LocalAddr localDst;   // 本卡输出地址
    std::vector<ccu::RemoteAddr> remoteSrc; // 远端输入地址

    remoteSrc.resize(ctx.arg->rankSize);

    localSrc.addr = ctx.input;
    localSrc.addr += ctx.currentSliceOffset;
    localSrc.token = ctx.token;

    localDst.addr = ctx.output;
    localDst.addr += ctx.currentSliceOffset;
    localDst.token = ctx.token;

    for (uint32_t rankIdx = 0; rankIdx < ctx.arg->rankSize; rankIdx++) {
        if (rankIdx != ctx.arg->rankId) {
            remoteSrc[rankIdx].addr = ctx.remoteInput[rankIdx];
            remoteSrc[rankIdx].addr += ctx.currentSliceOffset;
            remoteSrc[rankIdx].token = ctx.remoteToken[rankIdx];
        }
    }

    // Step 1: 本地拷贝，将本卡input拷贝到output，初始化output为本卡数据
    CCU_CHK_RET(GroupCopy(ctx, localDst, localSrc, ctx.goSize));
    CCU_CHK_RET(ccu::EventRecord(ctx.event, 1 << ctx.arg->rankId));

    const uint16_t totalMask = static_cast<uint16_t>((1u << ctx.arg->rankSize) - 1);
    CCU_CHK_RET(ccu::EventWait(ctx.event, totalMask)); // 等待本地拷贝完成

    // Step 2: 对每个远端rank执行ccu::ReadReduce，将远端input归约到本卡output
    // ccu::ReadReduce：从远端地址读取数据，与本地地址数据做归约操作（如SUM），
    // 结果存回本地地址。遍历所有远端rank后，output即为所有rank input的归约结果。
    CCU_IF(ctx.sliceSize != 0)
    {
        uint32_t channelId = 0;
        for (uint64_t rankIdx = 0; rankIdx < ctx.arg->rankSize; rankIdx++) {
            const uint16_t mask = 1 << rankIdx;
            if (rankIdx != ctx.arg->rankId) {
                CCU_CHK_RET(ccu::ReadReduce(ctx.arg->channels[channelId], localDst, remoteSrc[rankIdx],
                    ctx.sliceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, mask)); // 远端读取并归约
                channelId++;
            }
        }
        CCU_CHK_RET(ccu::EventRecord(ctx.event, 1 << ctx.arg->rankId));
        CCU_CHK_RET(ccu::EventWait(ctx.event, totalMask)); // 等待所有ReadReduce完成
    }

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuAllReduceMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduceMesh1D *>(arg);

    AllReduceMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;

    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllReduceMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    // 1.初始化资源：通过channel获取远端输入地址和token的Variable
    ctx.remoteInput.resize(ctx.arg->rankSize);
    ctx.remoteToken.resize(ctx.arg->rankSize);

    uint32_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < ctx.arg->rankSize; peerId++) {
        if (peerId != ctx.arg->rankId) {
            ctx.remoteInput[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
            ctx.remoteToken[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    // 2.加载参数
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentSliceOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argId++));

    // 3.前同步：将本卡输入地址和token写入远端channel，使远端rank能读取本卡input
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.input,
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token,
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit));
    }

    // 4.执行算法：本地拷贝 + ccu::ReadReduce远端归约
    CCU_CHK_RET(ExecuteAllReduceTransfer(ctx));

    // 5.后同步：通知远端rank本卡已完成，并等待所有远端rank完成
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }

    return CcuResult::CCU_SUCCESS;
}

} // namespace ops_hccl_ar
