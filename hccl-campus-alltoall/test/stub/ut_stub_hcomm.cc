/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <cstring>
#include <map>

#include <acl/acl_rt.h>
#include <hcomm/hcomm_primitives.h>
#include <hccl/hccl_res_expt.h>

namespace {

struct CtxBuffer {
    void *ptr;
    uint64_t size;
};

static std::map<CommEngine, CtxBuffer> ctxMap;

} // namespace

extern "C" {

// ==============================================
// hccl/hccl_res_expt.h
// ==============================================

HcclResult HcclThreadAcquireWithStream(HcclComm comm, CommEngine engine, aclrtStream stream,
    uint32_t notifyNum, ThreadHandle *thread)
{
    *thread = 1;
    return HCCL_SUCCESS;
}

HcclResult HcclThreadExportToCommEngine(HcclComm comm, uint32_t threadNum, const ThreadHandle *threads,
    CommEngine dstCommEngine, ThreadHandle *exportedThreads)
{
    *exportedThreads = *threads;
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxGet(HcclComm comm, const char *ctxTag, CommEngine engine,
    void **ctx, uint64_t *size)
{
    auto it = ctxMap.find(engine);
    if (it != ctxMap.end()) {
        *ctx = it->second.ptr;
        *size = it->second.size;
        return HCCL_SUCCESS;
    }
    return HCCL_E_PARA;
}

HcclResult HcclEngineCtxCreate(HcclComm comm, const char *ctxTag, CommEngine engine,
    uint64_t size, void **ctx)
{
    void *ptr = malloc(size);
    if (ptr == nullptr) {
        return HCCL_E_MEMORY;
    }
    memset(ptr, 0, size);
    *ctx = ptr;
    ctxMap[engine] = {ptr, size};
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxCopy(HcclComm comm, CommEngine engine, const char *ctxTag,
    const void *srcCtx, uint64_t size, uint64_t dstCtxOffset)
{
    auto it = ctxMap.find(engine);
    if (it == ctxMap.end()) {
        return HCCL_E_PARA;
    }
    memcpy(static_cast<char *>(it->second.ptr) + dstCtxOffset, srcCtx, size);
    return HCCL_SUCCESS;
}

HcclResult HcclThreadAcquire(HcclComm comm, CommEngine engine, uint32_t threadNum,
    uint32_t notifyNumPerThread, ThreadHandle *threads)
{
    *threads = 1;
    return HCCL_SUCCESS;
}

// ==============================================
// hcomm/hcomm_primitives.h
// ==============================================

int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
{
    return HCCL_SUCCESS;
}

int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut)
{
    return HCCL_SUCCESS;
}

int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    return HCCL_SUCCESS;
}

int32_t HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout)
{
    return HCCL_SUCCESS;
}

int32_t HcommBatchModeStart(const char *batchTag)
{
    return HCCL_SUCCESS;
}

int32_t HcommBatchModeEnd(const char *batchTag)
{
    return HCCL_SUCCESS;
}

int32_t HcommLocalCopyOnThread(ThreadHandle thread, void *dst, const void *src, uint64_t len)
{
    return HCCL_SUCCESS;
}

// ==============================================
// hccl/hccl_comm.h
// ==============================================

HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm)
{
    void *dummy = malloc(1);
    *comm = (HcclComm)dummy;
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommName(HcclComm comm, char *commName)
{
    const char* str = "hccl_comm";
    strcpy(commName, str);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank)
{
    *rank = 0;
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize)
{
    *rankSize = 16;
    return HCCL_SUCCESS;
}
} // extern "C"
