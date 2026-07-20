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

#include <algorithm>
#include <cstdio>

namespace ops_hccl {

HcclResult AicpuBaseTemplate::KernelRun(BaseEngine &engine, const TemplateDataParams &tempAlgParams,
    TemplateResource &templateResource, std::vector<u32> &ranksForOutputData)
{
    HCCL_INFO("[AicpuBaseTemplate][KernelRun] start, myRank[%u], rankSize[%zu].", myRank_, ranks_.size());
    tempAlgParams_ = tempAlgParams;
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

    // 1. PreCopy：本地数据预处理（input -> ccl buffer），仅第一步（input 为 userBuffer）执行。
    if (templateResource.threads.empty()) {
        HCCL_ERROR("[AicpuBaseTemplate][KernelRun] threads is empty.");
        return HCCL_E_INTERNAL;
    }
    if (tempAlgParams.inputBufferType == BufferType::INPUT) {
        CHK_RET(PreCopy(templateResource.threads));
    }

    // 单 rank 时无需通信。
    if (templateRankSize_ <= 1) {
        // 定位 mesh1dclos 用例：打印短路时的关键参数，确认是否 nhr 子分支被短路导致 PostCopy 未执行。
        fprintf(stderr, "[KernelRun] SHORTCUT myRank=%u rankSize=%zu inputBufType=%d outputBufType=%d "
                "sliceOffset=%lu sliceCount=%lu dataOffset=%lu ranksForOutputDataSize=%zu (skip PostCopy)\n",
            myRank_, ranks_.size(),
            static_cast<int>(tempAlgParams.inputBufferType), static_cast<int>(tempAlgParams.outputBufferType),
            static_cast<unsigned long>(tempAlgParams.sliceOffset), static_cast<unsigned long>(tempAlgParams.sliceCount),
            static_cast<unsigned long>(tempAlgParams.dataOffset), tempAlgParams.ranksForInputData.size());
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
    std::vector<TxRxSlicesList> txRxSlicesLists;
    CHK_RET(RunAlgorithm(templateResource, txRxSlicesLists, ranksForOutputData));

    // 4. SendAll：统一逐个执行 SendRecv（子类可在通信后做本地归约）。
    if (!txRxSlicesLists.empty()) {
        CHK_RET(SendAll(engine, txRxSlicesLists, templateResource, templateResource.threads));
    }

    // 5. 多线程场景下，通信后同步（从线程通知主线程完成）。
    if (multiThread) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxSubToMain(subThreads.size(), 0);
        CHK_RET(ops_hccl::PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }

    ranksForOutputData_ = ranksForOutputData;

    // 6. PostCopy：后处理（ccl buffer -> output），仅最后一步（output 为 userBuffer）执行。
    if (tempAlgParams_.outputBufferType == BufferType::OUTPUT) {
        CHK_RET(PostCopy(templateResource.threads));
    }

    HCCL_INFO("[AicpuBaseTemplate][KernelRun] end.");
    return HCCL_SUCCESS;
}

// ───────────── SendAll：逐个执行 SendRecv 的公共逻辑 ─────────────

HcclResult AicpuBaseTemplate::SendAll(
    BaseEngine &engine, const std::vector<TxRxSlicesList> &txRxSlicesLists,
    TemplateResource &templateResource, const std::vector<ThreadHandle> &threads)
{
    (void)threads;
    for (size_t i = 0; i < txRxSlicesLists.size(); ++i) {
        TransferContext ctx;
        ctx.enableRemoteMemAccess = tempAlgParams_.enableRemoteMemAccess;
        ctx.buffType = tempAlgParams_.cclBufferType;
        ctx.txRxSlicesList = txRxSlicesLists[i];
        ctx.templateRes = templateResource;
        ctx.dataType = tempAlgParams_.dataType;
        ctx.reduceOp = tempAlgParams_.reduceOp;
        CHK_RET(engine.Send(ctx));
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

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams_.dataType];
    const u64 sliceSize = tempAlgParams_.sliceCount * dataTypeSize;
    const u64 tailSize = tempAlgParams_.tailCount * dataTypeSize;

    // input 即 ccl buffer 时无需 PreCopy（inputBufferType=HCCL_BUFFER 或指针相同）。
    if (tempAlgParams_.inputBufferType == BufferType::HCCL_BUFFER ||
        tempAlgParams_.inputBufferPtr == tempAlgParams_.cclBufferPtr) {
        return HCCL_SUCCESS;
    }

    const size_t lastInputIdx = tempAlgParams_.ranksForInputData.size() - 1;
    // 遍历 ranksForInputData，将每个 rank 的数据从 input 拷到 ccl buffer。
    // input 偏移使用循环索引（idx）而非 rank 值：
    //   AllGather: ranksForInputData=[myRank_], idx=0 → inputOff=0（输入仅含本 rank 数据）
    //   AllReduce: ranksForInputData=[0,1,...], idx=rank → inputOff=rank*stride（输入含所有 rank 数据）
    // ccl 偏移使用 rank 值：cclOff = rank * stride（每个 rank 的数据在 ccl buffer 中按 rank 排列）
    for (size_t idx = 0; idx < tempAlgParams_.ranksForInputData.size(); ++idx) {
        u32 rank = tempAlgParams_.ranksForInputData[idx];
        // ranksForInputData 最后一个为尾 rank，数据量为 sliceSize + tailSize，其余为 sliceSize。
        u64 curSliceSize = (tailSize > 0 && idx == lastInputIdx) ? sliceSize + tailSize : sliceSize;
        if (curSliceSize == 0) {
            continue;
        }
        const u64 sliceCount = curSliceSize / dataTypeSize;

        const u64 inOff = tempAlgParams_.dataOffset + tempAlgParams_.sliceOffset + idx * tempAlgParams_.dataStride;

        const u64 cclOff = tempAlgParams_.sliceOffset + rank * tempAlgParams_.scratchStride;
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

    // remote mem 访问模式下数据已直接写到 output，无需后处理。
    if (tempAlgParams_.enableRemoteMemAccess) {
        return HCCL_SUCCESS;
    }
    if (threads.empty()) {
        HCCL_ERROR("[AicpuBaseTemplate][PostCopy] threads is empty.");
        return HCCL_E_INTERNAL;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams_.dataType];
    const u64 sliceSize = tempAlgParams_.sliceCount * dataTypeSize;
    const u64 tailSize = tempAlgParams_.tailCount * dataTypeSize;
    const size_t lastOutputIdx = ranksForOutputData_.empty() ? 0 : ranksForOutputData_.size() - 1;

    // 明确算法执行到 PostCopy 时，ccl buffer 里 slot=myrank 的数据是否是经过所有 rank reduce 过的数据，
    // 且最终要 PostCopy 到 output 的 0 位置。
    // ranksForOutputData 是当前 primitive 输出的归约集合（每项对应一个 ccl slot rank）；
    // ReduceScatter 语义要求最终 PostCopy 的数据应覆盖全 rank 归约到 myRank 的那一段，
    // 且输出落在 outOff=0 位置（idx=0 时 outOff=dataOffset+sliceOffset）。
    fprintf(stderr, "[PostCopy] ENTER myRank=%u globalRankSize=%zu inputBufType=%d outputBufType=%d "
            "sliceOffset=%lu scratchStride=%lu dataOffset=%lu dataStride=%lu sliceSize=%lu tailSize=%lu "
            "ranksForOutputData(size=%zu)=[",
        myRank_, ranks_.size(),
        static_cast<int>(tempAlgParams_.inputBufferType),
        static_cast<int>(tempAlgParams_.outputBufferType),
        static_cast<unsigned long>(tempAlgParams_.sliceOffset),
        static_cast<unsigned long>(tempAlgParams_.scratchStride),
        static_cast<unsigned long>(tempAlgParams_.dataOffset),
        static_cast<unsigned long>(tempAlgParams_.dataStride),
        static_cast<unsigned long>(sliceSize), static_cast<unsigned long>(tailSize),
        ranksForOutputData_.size());
    for (size_t i = 0; i < ranksForOutputData_.size(); ++i) {
        fprintf(stderr, "%u%s", ranksForOutputData_[i],
            (ranksForOutputData_[i] == myRank_) ? "(myRank)" : "");
        if (i + 1 < ranksForOutputData_.size()) {
            fprintf(stderr, ",");
        }
    }
    fprintf(stderr, "]\n");

    // 将 ccl buffer 中 ranksForOutputData 对应 rank 的数据搬回 output。
    for (size_t idx = 0; idx < ranksForOutputData_.size(); ++idx) {
        u32 rank = ranksForOutputData_[idx];
        // ranksForOutputData_ 最后一个为尾 rank，数据量为 sliceSize + tailSize，其余为 sliceSize。
        u64 curSliceSize = (tailSize > 0 && idx == lastOutputIdx) ? sliceSize + tailSize : sliceSize;
        if (curSliceSize == 0) {
            continue;
        }
        const u64 sliceCount = curSliceSize / dataTypeSize;
        const u64 cclOff = tempAlgParams_.sliceOffset + rank * tempAlgParams_.scratchStride;
        const u64 outOff = tempAlgParams_.dataOffset + tempAlgParams_.sliceOffset + idx * tempAlgParams_.dataStride;
        // 日志聚焦：ccl slot=rank 的数据 PostCopy 到 outOff；
        //   - slotRankIsMyRank=1 表示本条 PostCopy 取的是 ccl slot(myRank) 的数据；
        //   - outOffIsZero=1 表示本条 PostCopy 目标在 output 0 位置（idx=0）；
        //   - ranksForOutputDataContainsMyRank=1 表示归约集合包含 myRank（即 slot=myRank 的数据会被 PostCopy）。
        fprintf(stderr, "[PostCopy] COPY myRank=%u idx=%zu slotRank=%u slotRankIsMyRank=%d outOff=%lu outOffIsZero=%d "
                "curSliceSize=%lu ranksForOutputDataContainsMyRank=%d\n",
            myRank_, idx, rank, static_cast<int>(rank == myRank_),
            static_cast<unsigned long>(outOff), static_cast<int>(outOff == 0),
            static_cast<unsigned long>(curSliceSize),
            static_cast<int>(std::find(ranksForOutputData_.begin(), ranksForOutputData_.end(), myRank_) !=
                             ranksForOutputData_.end()));
        DataSlice srcSlice(tempAlgParams_.cclBufferPtr, cclOff, curSliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.outputBufferPtr, outOff, curSliceSize, sliceCount);
        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }

    HCCL_INFO("[AicpuBaseTemplate][PostCopy] end.");
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
