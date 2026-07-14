/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_base_template.h"
#include "base_engine.h"

#include "log.h"

namespace ops_hccl {

HcclResult AicpuBaseTemplate::KernelRun(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                                        std::vector<u32> &ranksForOutputData)
{
    HCCL_INFO("[AicpuBaseTemplate][KernelRun] start, myRank[%u], rankSize[%zu].", myRank_, ranks_.size());

    tempAlgParams_ = tempAlgParams;
    dataType_ = tempAlgParams.dataType;
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    templateRankSize_ = static_cast<u32>(ranks_.size());

    // 数据量为 0 时直接返回。
    const u64 sliceCount = tempAlgParams.sliceCount;
    const u64 tailCount = tempAlgParams.tailCount;
    if (sliceCount == 0 && tailCount == 0) {
        HCCL_INFO("[AicpuBaseTemplate][KernelRun] sliceCount is 0, no need to do, just success.");
        ranksForOutputData.clear();
        ranksForOutputData_ = ranksForOutputData;
        return HCCL_SUCCESS;
    }

    // 1. PreCopy：本地数据预处理（input -> output / ccl buffer）。
    if (templateResource.threads.empty()) {
        HCCL_ERROR("[AicpuBaseTemplate][KernelRun] threads is empty.");
        return HCCL_E_INTERNAL;
    }
    CHK_RET(PreCopy(templateResource.threads));

    // 单 rank 时无需通信。
    if (templateRankSize_ <= 1) {
        ranksForOutputData = tempAlgParams.ranksForInputData;
        ranksForOutputData_ = ranksForOutputData;
        return HCCL_SUCCESS;
    }

    const u32 threadNum = static_cast<u32>(templateResource.threads.size());
    const bool multiThread = (threadNum > 1);

    // 2. 多线程场景下，通信前先同步（主线程通知从线程可以开始）。
    if (multiThread) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxMainToSub(subThreads.size(), 0);
        CHK_RET(ops_hccl::PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }

    // 3. RunAlgorithm：子类生成 SendRecvInfo 列表与 ranksForOutputData。
    std::vector<SendRecvInfo> sendRecvInfos;
    CHK_RET(RunAlgorithm(templateResource, sendRecvInfos, ranksForOutputData));

    // 4. SendAll：统一逐个执行 SendRecv。
    if (!sendRecvInfos.empty()) {
        CHK_RET(SendAll(sendRecvInfos, templateResource));
    }

    // 5. 多线程场景下，通信后同步（从线程通知主线程完成）。
    if (multiThread) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxSubToMain(subThreads.size(), 0);
        CHK_RET(ops_hccl::PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }

    // 6. PostCopy：后处理（ccl buffer -> output），若需要。
    CHK_RET(PostCopy(templateResource.threads));

    ranksForOutputData_ = ranksForOutputData;

    HCCL_INFO("[AicpuBaseTemplate][KernelRun] end.");
    return HCCL_SUCCESS;
}

// ───────────── SendAll：逐个执行 SendRecv 的公共逻辑 ─────────────

HcclResult AicpuBaseTemplate::SendAll(const std::vector<SendRecvInfo> &sendRecvInfos,
                                       TemplateResource &templateResource)
{
    for (size_t i = 0; i < sendRecvInfos.size(); ++i) {
        TransferContext ctx;
        ctx.enableRemoteMemAccess = tempAlgParams_.enableRemoteMemAccess;
        ctx.buffType = tempAlgParams_.cclBufferType;
        ctx.txRxSlicesList = sendRecvInfos[i].sendRecvSlices_;
        ctx.templateRes = templateResource;
        ctx.dataType = sendRecvInfos[i].dataType_;
        ctx.reduceOp = tempAlgParams_.reduceOp;
        CHK_RET(GetEngine().Send(ctx));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuBaseTemplate::PreCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[AicpuBaseTemplate][PreCopy] start, myRank[%u].", myRank_);
    if (threads.empty()) {
        HCCL_ERROR("[AicpuBaseTemplate][PreCopy] threads is empty.");
        return HCCL_E_INTERNAL;
    }

    if (tempAlgParams_.ranksForInputData.empty()) {
        HCCL_ERROR("[AicpuBaseTemplate][PreCopy] ranksForInputData is empty.");
        return HCCL_E_INTERNAL;
    }

    if (tempAlgParams_.inputBufferPtr == tempAlgParams_.cclBufferPtr) {
        HCCL_DEBUG("[AicpuBaseTemplate][PreCopy] inputBufferPtr == cclBufferPtr, skip copy.");
        return HCCL_SUCCESS;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    const u64 sliceSize = tempAlgParams_.sliceCount * dataTypeSize;
    const u64 tailSize = tempAlgParams_.tailCount * dataTypeSize;

    // 遍历 ranksForInputData，将每个 rank 的数据从 input 拷到 output / ccl buffer。
    for (u32 rank : tempAlgParams_.ranksForInputData) {
        u32 algRank = 0;
        CHK_RET(GetAlgRank(rank, ranks_, algRank));
        const u64 curSliceSize = (tailSize != 0 && algRank == templateRankSize_ - 1) ? tailSize : sliceSize;
        if (curSliceSize == 0) {
            continue;
        }
        const u64 sliceCount = curSliceSize / dataTypeSize;

        const u64 inOff = tempAlgParams_.dataOffset + tempAlgParams_.sliceOffset + rank * tempAlgParams_.stride;
        const u64 cclOff = tempAlgParams_.sliceOffset + rank * tempAlgParams_.stride;

        // input -> ccl buffer（非 remote mem 访问时需要本地中转）
        DataSlice srcSlice(tempAlgParams_.inputBufferPtr, inOff, curSliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.cclBufferPtr, cclOff, curSliceSize, sliceCount);
        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    HCCL_INFO("[AicpuBaseTemplate][PreCopy] end.");
    return HCCL_SUCCESS;
}

HcclResult AicpuBaseTemplate::PostCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[AicpuBaseTemplate][PostCopy] start, myRank[%u].", myRank_);

    // 输出本身就是 ccl buffer 时无需后处理。
    if (tempAlgParams_.outputBufferType == BufferType::HCCL_BUFFER) {
        return HCCL_SUCCESS;
    }
    // 输入是 ccl buffer 时（ccl -> output 已在通信中完成）也无需后处理。
    if (tempAlgParams_.inputBufferType == BufferType::HCCL_BUFFER) {
        return HCCL_SUCCESS;
    }
    // remote mem 访问模式下数据已直接写到 output，无需后处理。
    if (enableRemoteMemAccess_) {
        return HCCL_SUCCESS;
    }
    if (threads.empty()) {
        HCCL_ERROR("[AicpuBaseTemplate][PostCopy] threads is empty.");
        return HCCL_E_INTERNAL;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    const u64 sliceSize = tempAlgParams_.sliceCount * dataTypeSize;
    const u64 tailSize = tempAlgParams_.tailCount * dataTypeSize;

    // 将 ccl buffer 中 ranksForOutputData 对应 rank 的数据搬回 output。
    for (u32 rank : ranksForOutputData_) {
        if (rank == myRank_) {
            continue;
        }
        u32 algRank = 0;
        CHK_RET(GetAlgRank(rank, ranks_, algRank));
        const u64 curSliceSize = (tailSize != 0 && algRank == templateRankSize_ - 1) ? tailSize : sliceSize;
        if (curSliceSize == 0) {
            continue;
        }
        const u64 sliceCount = curSliceSize / dataTypeSize;
        const u64 cclOff = tempAlgParams_.sliceOffset + rank * tempAlgParams_.stride;
        const u64 outOff = tempAlgParams_.dataOffset + tempAlgParams_.sliceOffset + rank * tempAlgParams_.stride;
        DataSlice srcSlice(tempAlgParams_.cclBufferPtr, cclOff, curSliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.outputBufferPtr, outOff, curSliceSize, sliceCount);
        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    HCCL_INFO("[AicpuBaseTemplate][PostCopy] end.");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
