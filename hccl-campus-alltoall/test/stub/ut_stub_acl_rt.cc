/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <acl/acl_rt.h>

#include "common.h"

// AICPU Kernel 函数
extern "C" unsigned int HcclAICPUKernel(void *param);

// AICPU Kernel 函数入参
static OpParam *g_opParam = nullptr;

extern "C" {

aclError aclrtBinaryLoadFromFile(const char *path, aclrtBinaryLoadOptions *loadOptions, aclrtBinHandle *handle)
{
    *handle = nullptr;
    return ACL_SUCCESS;
}

aclError aclrtBinaryGetFunction(aclrtBinHandle handle, const char *kernelName, aclrtFuncHandle *funcHandle)
{
    *funcHandle = nullptr;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle *argsHandle)
{
    *argsHandle = nullptr;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void *arg, size_t size, aclrtParamHandle *paraHandle)
{
    g_opParam = static_cast<OpParam *>(arg);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernelWithConfig(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
    aclrtLaunchKernelCfg *cfg, aclrtArgsHandle argsHandle, void *reserved)
{
    // 直接调用 AICPU Kernel 函数
    int ret = HcclAICPUKernel(g_opParam);
    if (ret == 0) {
        return ACL_SUCCESS;
    } else {
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    void *ptr = malloc(size);
    if (ptr == nullptr) {
        return ACL_ERROR_INTERNAL_ERROR;
    }
    *devPtr = ptr;
    return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream *stream)
{
    void *dummy = malloc(1);
    if (dummy == nullptr) {
        return ACL_ERROR_INTERNAL_ERROR;
    }
    *stream = (aclrtStream)dummy;
    return ACL_SUCCESS;
}

} // extern "C"
