/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_PRIMITIVES_DL_H
#define HCOMM_PRIMITIVES_DL_H

#include "dlsym_common.h"
#include "hcomm_primitives.h"   // 原头文件，包含所有类型和定义
#include "hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

DECL_WEAK_FUNC(int32_t, HcommThreadSynchronize, ThreadHandle thread);
DECL_WEAK_FUNC(int32_t, HcommSendRequest, uint64_t handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId);
DECL_WEAK_FUNC(int32_t, HcommWaitResponse, uint64_t handle, void* dst, size_t sizeByte, uint32_t* msgId);
DECL_WEAK_FUNC(HcclResult, HcommThreadJoin, ThreadHandle thread, uint32_t timeout);
DECL_WEAK_FUNC(int32_t, HcommThreadNotifyRecordOnThread, ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx);
DECL_WEAK_FUNC(int32_t, HcommThreadNotifyWaitOnThread, ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut);
DECL_WEAK_FUNC(int32_t, HcommChannelNotifyRecordOnThread, ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx);
DECL_WEAK_FUNC(int32_t, HcommChannelNotifyWaitOnThread, ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut);
DECL_WEAK_FUNC(int32_t, HcommWriteOnThread, ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len);
DECL_WEAK_FUNC(int32_t, HcommWriteReduceOnThread, ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp);
DECL_WEAK_FUNC(int32_t, HcommReadOnThread, ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len);
DECL_WEAK_FUNC(int32_t, HcommReadReduceOnThread, ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp);
DECL_WEAK_FUNC(int32_t, HcommSetNotifyWaitTimeOut, uint32_t timeOut);
DECL_WEAK_FUNC(int32_t, HcommThreadNotifyWaitOnThreadWithDefaultTimeout, ThreadHandle thread, uint32_t notifyIdx);
DECL_WEAK_FUNC(int32_t, HcommThreadResAcquireTimeOut, uint32_t timeOut);

DECL_SUPPORT_FLAG(HcommSetNotifyWaitTimeOut);
DECL_SUPPORT_FLAG(HcommThreadNotifyWaitOnThreadWithDefaultTimeout);
DECL_SUPPORT_FLAG(HcommThreadResAcquireTimeOut);

bool IsHcommDefaultTimeoutSupported();
HcclResult HcclSetNotifyWaitTimeOut(uint32_t timeout);
HcclResult HcclThreadNotifyWaitOnThreadDefault(ThreadHandle thread, uint32_t notifyIdx, uint32_t fallbackTimeout);
HcclResult HcclThreadResAcquireTimeOut(uint32_t timeout);

void HcommPrimitivesDlInit(void* libHcommHandle);  // 本模块独立初始化

#ifdef __cplusplus
}
#endif

#endif // HCOMM_PRIMITIVES_DL_H
