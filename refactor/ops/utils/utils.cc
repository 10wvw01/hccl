/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/utils.h"

#include "hcomm_primitives_dl.h"
#include "exec_timeout_manager.h"
#include "log.h"

namespace ops_hccl {

namespace {

inline void *GetSliceAddr(const DataSlice &slice)
{
    return static_cast<void *>(static_cast<s8 *>(slice.addr_) + slice.offset_);
}

}  // namespace

HcclResult LocalCopy(const ThreadHandle &thread, const DataSlice &srcSlice, const DataSlice &dstSlice)
{
    if (srcSlice.size_ == 0) {
        return HCCL_SUCCESS;
    }
    if (srcSlice.size_ != dstSlice.size_) {
        HCCL_ERROR("[LocalCopy] src size[%llu] != dst size[%llu].",
                   static_cast<unsigned long long>(srcSlice.size_),
                   static_cast<unsigned long long>(dstSlice.size_));
        return HCCL_E_INTERNAL;
    }
    void *src = GetSliceAddr(srcSlice);
    void *dst = GetSliceAddr(dstSlice);
    HCCL_DEBUG("[LocalCopy] src[%p] dst[%p] len[%llu].", src, dst,
               static_cast<unsigned long long>(srcSlice.size_));
    return static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, src, srcSlice.size_));
}

HcclResult PreSyncInterThreads(const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads,
                               const std::vector<u32> &notifyIdxMainToSub)
{
    CHK_PRT_RET(notifyIdxMainToSub.size() != subThreads.size(),
                HCCL_ERROR("[PreSyncInterThreads] size mismatch, sub[%zu] idx[%zu].",
                           subThreads.size(), notifyIdxMainToSub.size()),
                HCCL_E_PARA);
    const u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    // 主thread向从thread发送record
    for (size_t i = 0; i < subThreads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(mainThread, subThreads[i], notifyIdxMainToSub[i])));
    }
    // 从thread等待主thread的record
    for (size_t i = 0; i < subThreads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(subThreads[i], notifyIdxMainToSub[i], execTimeout)));
    }
    return HCCL_SUCCESS;
}

HcclResult PostSyncInterThreads(const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads,
                                const std::vector<u32> &notifyIdxSubToMain)
{
    CHK_PRT_RET(notifyIdxSubToMain.size() != subThreads.size(),
                HCCL_ERROR("[PostSyncInterThreads] size mismatch, sub[%zu] idx[%zu].",
                           subThreads.size(), notifyIdxSubToMain.size()),
                HCCL_E_PARA);
    const u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    // 主thread等待所有从thread的record
    for (size_t i = 0; i < subThreads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(mainThread, notifyIdxSubToMain[i], execTimeout)));
    }
    // 从thread向主thread发送record
    for (size_t i = 0; i < subThreads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(subThreads[i], mainThread, notifyIdxSubToMain[i])));
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
