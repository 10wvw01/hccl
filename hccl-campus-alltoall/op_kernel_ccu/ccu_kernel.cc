/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "ccu_kernel.h"

namespace ops_ccu {

constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int POST_SYNC_ID = 3;
static CcuResult ExecuteAllToAllTransfer(AllToAllMesh1DMem2MemContext &ctx)
{
    HCCL_INFO("ExecuteAllToAllTransfer Start.");
    const auto *arg = ctx.arg;
    std::vector<ccu::LocalAddr> src(arg->rankSize);
    std::vector<ccu::RemoteAddr> dst(arg->rankSize);
    ccu::LocalAddr localDst;

    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx != arg->rankId) {
            dst[rankIdx].token = ctx.token[rankIdx];
            dst[rankIdx].addr = ctx.output[rankIdx];
            dst[rankIdx].addr += ctx.dstOffset;
        } else {
            localDst.token = ctx.token[rankIdx];
            localDst.addr = ctx.output[rankIdx];
            localDst.addr += ctx.dstOffset;
        }

        src[rankIdx].addr = ctx.srcOffset;
        src[rankIdx].token = ctx.token[rankIdx];
        for (uint64_t i = 0; i < rankIdx; i++) {
            src[rankIdx].addr += ctx.srcStride;
        }
    }

    uint32_t channelId = 0;

    for(uint64_t r = 0; r < arg->rankSize; r++) {
        if (r == arg->rankId) {
            ccu::LocalCopy(localDst, src[r], ctx.sliceSize, ctx.event, 1 << r);
        }
        else {
            ccu::Write(arg->channels[channelId], dst[r], src[r], ctx.sliceSize, ctx.event, 1 << r);
            channelId++;
        }
    }
    // 等读完所有对端
    ccu::EventWait(ctx.event, (1 << arg->rankSize) - 1);
    
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuAlltoAllMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllToAllMesh1DMem2Mem *>(arg);

    AllToAllMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;

    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllToAllMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    // 1.初始化资源
    uint32_t channelIdx = 0;
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求算法返回的Link同样是按顺序排列的
    ctx.input.resize(ctx.arg->rankSize);
    ctx.output.resize(ctx.arg->rankSize);
    ctx.token.resize(ctx.arg->rankSize);
    for (uint64_t peerId = 0; peerId < ctx.arg->rankSize; peerId++) {
        if (peerId != ctx.arg->rankId) {
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    // 2.加载参数
    const auto *kArg = ctx.arg;
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[kArg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[kArg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[kArg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.srcStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.srcOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.dstOffset, argId++));

    ctx.srcOffset += ctx.input[kArg->rankId];

    // 3.前同步
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.output[ctx.arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token[ctx.arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit));
    }

    // 4.执行算法
    CCU_CHK_RET(ExecuteAllToAllTransfer(ctx));

    // 5.后同步
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID)); // 等待远端卡数据搬运完成
    }

    return CcuResult::CCU_SUCCESS;
}

} // namespace ops_ccu
