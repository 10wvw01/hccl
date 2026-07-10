/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_launch_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

AsccommCcuKernelRegisterStartFunc g_AsccommCcuKernelRegisterStart = nullptr;
AsccommCcuKernelRegisterFunc g_AsccommCcuKernelRegister = nullptr;
AsccommCcuKernelRegisterEndFunc g_AsccommCcuKernelRegisterEnd = nullptr;
AsccommCcuKernelLaunchFunc g_AsccommCcuKernelLaunch = nullptr;
AsccommCcuGetMemTokenFunc g_AsccommCcuGetMemToken = nullptr;

namespace {
template <typename Func>
Func ResolveCcuFunc(void *handle, const char *asccommName, const char *compatName)
{
    auto func = reinterpret_cast<Func>(dlsym(handle, asccommName));
    if (func != nullptr) {
        return func;
    }
    func = reinterpret_cast<Func>(dlsym(handle, compatName));
    if (func == nullptr) {
        fprintf(stderr, "[HcclWrapper] failed to resolve %s or %s: %s\n", asccommName, compatName, dlerror());
    }
    return func;
}

template <typename Func>
CcuResult CheckCcuFunc(Func func, const char *funcName)
{
    if (func == nullptr) {
        fprintf(stderr, "[HcclWrapper] %s not resolved\n", funcName);
        return CCU_E_UNAVAIL;
    }
    return CCU_SUCCESS;
}
}

void CcuLaunchDlInit(void* asccommHandle) {
    if (!asccommHandle) {
        fprintf(stderr, "[HcclWrapper] CcuLaunchDlInit: null asccomm handle\n");
        return;
    }
    dlerror();
    g_AsccommCcuKernelRegisterStart = ResolveCcuFunc<AsccommCcuKernelRegisterStartFunc>(
        asccommHandle, "AsccommCcuKernelRegisterStart", "HcommCcuKernelRegisterStart");
    g_AsccommCcuKernelRegister = ResolveCcuFunc<AsccommCcuKernelRegisterFunc>(
        asccommHandle, "AsccommCcuKernelRegister", "HcommCcuKernelRegister");
    g_AsccommCcuKernelRegisterEnd = ResolveCcuFunc<AsccommCcuKernelRegisterEndFunc>(
        asccommHandle, "AsccommCcuKernelRegisterEnd", "HcommCcuKernelRegisterEnd");
    g_AsccommCcuKernelLaunch = ResolveCcuFunc<AsccommCcuKernelLaunchFunc>(
        asccommHandle, "AsccommCcuKernelLaunch", "HcommCcuKernelLaunch");
    g_AsccommCcuGetMemToken = ResolveCcuFunc<AsccommCcuGetMemTokenFunc>(
        asccommHandle, "AsccommCcuGetMemToken", "HcommCcuGetMemToken");
}

CcuResult HcommCcuKernelRegisterStart(CcuInsHandle insHandle)
{
    CcuResult ret = CheckCcuFunc(g_AsccommCcuKernelRegisterStart, __func__);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    return g_AsccommCcuKernelRegisterStart(insHandle);
}

CcuResult HcommCcuKernelRegister(CcuInsHandle insHandle, uint32_t dieId,
    const char *kernelFuncName, const void *kernelFunc,
    const void **kernelArgs, uint32_t argNum, CcuKernelHandle *kernelHandle)
{
    CcuResult ret = CheckCcuFunc(g_AsccommCcuKernelRegister, __func__);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    return g_AsccommCcuKernelRegister(insHandle, dieId, kernelFuncName, kernelFunc,
        kernelArgs, argNum, kernelHandle);
}

CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle insHandle)
{
    CcuResult ret = CheckCcuFunc(g_AsccommCcuKernelRegisterEnd, __func__);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    return g_AsccommCcuKernelRegisterEnd(insHandle);
}

CcuResult HcommCcuKernelLaunch(ThreadHandle threadHandle,
    CcuKernelHandle kernelHandle, const void *taskArgs, uint32_t argSize)
{
    CcuResult ret = CheckCcuFunc(g_AsccommCcuKernelLaunch, __func__);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    return g_AsccommCcuKernelLaunch(threadHandle, kernelHandle, taskArgs, argSize);
}
