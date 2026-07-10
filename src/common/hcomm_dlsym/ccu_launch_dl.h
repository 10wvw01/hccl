/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_LAUNCH_DL_H
#define CCU_LAUNCH_DL_H

#include "dlsym_common.h"
#if CANN_VERSION_NUM >=90100000
#include "ccu_types.h"
#include "hccl_res.h"
#else
#include "ccu_types_dl.h"
#include "hccl_res_dl.h"
#endif // CANN_VERSION_NUM >= 90100000

#ifdef __cplusplus
extern "C" {
#endif

typedef CcuResult (*AsccommCcuKernelRegisterStartFunc)(CcuInsHandle insHandle);
typedef CcuResult (*AsccommCcuKernelRegisterFunc)(CcuInsHandle insHandle, uint32_t dieId,
    const char *kernelFuncName, const void *kernelFunc,
    const void **kernelArgs, uint32_t argNum, CcuKernelHandle *kernelHandle);
typedef CcuResult (*AsccommCcuKernelRegisterEndFunc)(CcuInsHandle insHandle);
typedef CcuResult (*AsccommCcuKernelLaunchFunc)(ThreadHandle threadHandle,
    CcuKernelHandle kernelHandle, const void *taskArgs, uint32_t argSize);
typedef CcuResult (*AsccommCcuGetMemTokenFunc)(uint64_t srcVa, uint64_t size, uint64_t *tokenInfo);

extern AsccommCcuKernelRegisterStartFunc g_AsccommCcuKernelRegisterStart;
extern AsccommCcuKernelRegisterFunc g_AsccommCcuKernelRegister;
extern AsccommCcuKernelRegisterEndFunc g_AsccommCcuKernelRegisterEnd;
extern AsccommCcuKernelLaunchFunc g_AsccommCcuKernelLaunch;
extern AsccommCcuGetMemTokenFunc g_AsccommCcuGetMemToken;

void CcuLaunchDlInit(void* asccommHandle);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C++" {
inline CcuResult HcommCcuKernelRegisterStart(CcuInsHandle insHandle) {
    return g_AsccommCcuKernelRegisterStart ? g_AsccommCcuKernelRegisterStart(insHandle) : CCU_E_NOT_SUPPORT;
}
inline CcuResult HcommCcuKernelRegister(CcuInsHandle insHandle, uint32_t dieId,
    const char *kernelFuncName, const void *kernelFunc,
    const void **kernelArgs, uint32_t argNum, CcuKernelHandle *kernelHandle) {
    return g_AsccommCcuKernelRegister ? g_AsccommCcuKernelRegister(insHandle, dieId, kernelFuncName,
        kernelFunc, kernelArgs, argNum, kernelHandle) : CCU_E_NOT_SUPPORT;
}
inline CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle insHandle) {
    return g_AsccommCcuKernelRegisterEnd ? g_AsccommCcuKernelRegisterEnd(insHandle) : CCU_E_NOT_SUPPORT;
}
inline CcuResult HcommCcuKernelLaunch(ThreadHandle threadHandle,
    CcuKernelHandle kernelHandle, const void *taskArgs, uint32_t argSize) {
    return g_AsccommCcuKernelLaunch ? g_AsccommCcuKernelLaunch(threadHandle, kernelHandle, taskArgs, argSize) : CCU_E_NOT_SUPPORT;
}
inline CcuResult HcommCcuGetMemToken(uint64_t srcVa, uint64_t size, uint64_t *tokenInfo) {
    return g_AsccommCcuGetMemToken ? g_AsccommCcuGetMemToken(srcVa, size, tokenInfo) : CCU_E_NOT_SUPPORT;
}
}
#endif

#endif // CCU_LAUNCH_DL_H
