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

#include "log.h"

namespace ops_hccl {

HcclResult AicpuBaseTemplate::KernelRun(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource,
                                        std::vector<u32> &ranksForOutputData)
{
    HCCL_INFO("[AicpuBaseTemplate][KernelRun] start, myRank[%u], rankSize[%zu].", myRank_, ranks.size());

    tempAlgParams_ = tempAlgParams;
    dataType_ = tempAlgParams.dataType;
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    templateRankSize_ = static_cast<u32>(ranks.size());

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

    // 3. RunAlgorithm：子类实现具体的通信原语编排。
    CHK_RET(RunAlgorithm(templateResource));

    // 4. 多线程场景下，通信后同步（从线程通知主线程完成）。
    if (multiThread) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxSubToMain(subThreads.size(), 0);
        CHK_RET(ops_hccl::PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }

    // 5. PostCopy：后处理（ccl buffer -> output），若需要。
    CHK_RET(PostCopy(templateResource.threads));

    // AllGather 语义：输出对应全部 rank。
    ranksForOutputData = ranks;
    ranksForOutputData_ = ranksForOutputData;

    HCCL_INFO("[AicpuBaseTemplate][KernelRun] end.");
    return HCCL_SUCCESS;
}

// ───────────── PreCopy / PostCopy 默认实现 ─────────────
// 内存布局说明：
//   - ccl buffer (scratch)：只存放一个 loop 的数据，rank i 的偏移 = sliceOffset + i * sliceSize。
//   - input/output buffer：包含所有 loop 的数据，rank i 的偏移 = sliceOffset + i * stride * dataTypeSize。
//     其中 stride 表示 loop 间数据间隔（以元素计），当只有一个 loop 时 stride = sliceCount。

HcclResult AicpuBaseTemplate::PreCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[AicpuBaseTemplate][PreCopy] start, myRank[%u].", myRank_);
    if (threads.empty()) {
        HCCL_ERROR("[AicpuBaseTemplate][PreCopy] threads is empty.");
        return HCCL_E_INTERNAL;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    const u64 sliceSize = tempAlgParams_.sliceCount * dataTypeSize;
    const u64 tailSize = tempAlgParams_.tailCount * dataTypeSize;

    // 遍历 ranksForInputData，将每个 rank 的数据从 input 拷到 output / ccl buffer。
    for (u32 rank : tempAlgParams_.ranksForInputData) {
        u32 algRank = 0;
        CHK_RET(GetAlgRank(rank, ranks, algRank));
        const u64 curSliceSize = (tailSize != 0 && algRank == templateRankSize_ - 1) ? tailSize : sliceSize;
        if (curSliceSize == 0) {
            continue;
        }
        const u64 sliceCount = curSliceSize / dataTypeSize;

        // input/output 中 rank 的偏移（考虑 loop 间间隔 stride）。
        const u64 inOff = tempAlgParams_.sliceOffset + algRank * tempAlgParams_.stride * dataTypeSize;
        const u64 outOff = inOff;
        // ccl buffer (scratch) 中 rank 的偏移（只有一个 loop，rank 间间隔为 sliceSize）。
        const u64 cclOff = tempAlgParams_.sliceOffset + algRank * sliceSize;

        // input -> output（同址同偏移跳过）
        bool skipOutCopy = (tempAlgParams_.inputBufferPtr == tempAlgParams_.outputBufferPtr && inOff == outOff);
        if (!skipOutCopy) {
            DataSlice srcSlice(tempAlgParams_.inputBufferPtr, inOff, curSliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.outputBufferPtr, outOff, curSliceSize, sliceCount);
            CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
        }

        // input -> ccl buffer（非 remote mem 访问时需要本地中转）
        if (!enableRemoteMemAccess_) {
            bool skipCclCopy = (tempAlgParams_.inputBufferPtr == tempAlgParams_.cclBufferPtr && inOff == cclOff);
            if (!skipCclCopy) {
                DataSlice srcSlice(tempAlgParams_.inputBufferPtr, inOff, curSliceSize, sliceCount);
                DataSlice dstSlice(tempAlgParams_.cclBufferPtr, cclOff, curSliceSize, sliceCount);
                CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
            }
        }
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
        CHK_RET(GetAlgRank(rank, ranks, algRank));
        const u64 curSliceSize = (tailSize != 0 && algRank == templateRankSize_ - 1) ? tailSize : sliceSize;
        if (curSliceSize == 0) {
            continue;
        }
        const u64 sliceCount = curSliceSize / dataTypeSize;
        // ccl buffer (scratch) 中 rank 的偏移（只有一个 loop）。
        const u64 cclOff = tempAlgParams_.sliceOffset + algRank * sliceSize;
        // output 中 rank 的偏移（考虑 loop 间间隔 stride）。
        const u64 outOff = tempAlgParams_.sliceOffset + algRank * tempAlgParams_.stride * dataTypeSize;
        DataSlice srcSlice(tempAlgParams_.cclBufferPtr, cclOff, curSliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.outputBufferPtr, outOff, curSliceSize, sliceCount);
        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    HCCL_INFO("[AicpuBaseTemplate][PostCopy] end.");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
