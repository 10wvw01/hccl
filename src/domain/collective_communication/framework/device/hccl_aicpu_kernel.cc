/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hdcs.h"
#include "hccl_cpu_kernel.h"

uint32_t HdcsGetPid(void *args)
{
    int *pid = static_cast<int *>(args);
    return DevGetPid(pid);
}
//        ʼ      psRankIdMap
uint32_t HdcsCsInit(void *args)
{
    HdcsCsInitPara *params = static_cast<HdcsCsInitPara *>(args);
    for (u32 i = 0; i < params->psSize; ++i) {
        HCCL_INFO("psList: index[%u]=[%u]", i, params->psList[i]);
    }
    DevGetPid(static_cast<s32 *>(params->devicePid));
    return DevCommInit(*params);
}

uint32_t HdcsCsDeInit()
{
    return DevCommDeInit();
}

uint32_t HdcsRegTransport(void *args)
{
    HdcsRegTransportPara *params = static_cast<HdcsRegTransportPara *>(args);
    return RegTransport(*params);
}

uint32_t HdcsRemoteLookup(void *args)
{
    return IsendLookupRequest(args);
}

uint32_t HdcsCollRemoteUpdate(void *args)
{
    return IsendUpdateRequest(args);
}

// lookup ȥ  +sendbuf ׼  
uint32_t HdcsCollLookupKeysDuplicates(void *args)
{
    return CollLookupKeysDuplicates(args);
}

// lookup send keys
uint32_t HdcsCollLookupSendKeys(void *args)
{
    return CollLookupSendKeys(args);
}

// lookup recv values
uint32_t HdcsCollLookupRecvValues(void *args)
{
    return CollLookupRecvValues(args);
}

// lookup recover value
uint32_t HdcsCollLookupRecoverValue(void *args)
{
    return CollLookupRecoverValue(args);
}

// lookup reset unique handle
uint32_t HdcsCollLookupResetUniqueHandle(void *args)
{
    return CollLookupResetUniqueHandle(args);
}

// lookup wait send key finish
uint32_t HdcsCollLookupWaitSendKeyFinish(void *args)
{
    return CollLookupWaitSendKeyFinish(args);
}

// update ReduceSum
uint32_t HdcsCollRemoteUpdateReduceSum(void *args)
{
    return RemoteUpdateReduceSum(args);
}

// lookup gather finish
uint32_t HdcsCollLookupGatherFinish(void *args)
{
    return CollLookupGatherFinish(args);
}

// update KeyReduce
uint32_t HdcsCollRemoteUpdateKeyReduce(void *args)
{
    return RemoteUpdateKeyReduce(args);
}

// update SendRequest
uint32_t HdcsCollRemoteUpdateSendRequest(void *args)
{
    return RemoteUpdateSendRequest(args);
}

uint32_t HdcsCollRemoteUpdateResetUniqueHandle(void *args)
{
    return RemoteUpdateResetUniqueHandle(args);
}

// update RecvResponse
uint32_t HdcsCollRemoteUpdateRecvResponse(void *args)
{
    return RemoteUpdateRecvResponse(args);
}

uint32_t HdcsChkServiceCancel(void *args)
{
    return ChkServiceCancel(args);
}