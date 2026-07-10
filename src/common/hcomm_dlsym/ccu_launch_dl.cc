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

void CcuLaunchDlInit(void* asccommHandle) {
    if (!asccommHandle) {
        fprintf(stderr, "[HcclWrapper] CcuLaunchDlInit: null asccomm handle\n");
        return;
    }
    g_AsccommCcuKernelRegisterStart = (AsccommCcuKernelRegisterStartFunc)dlsym(asccommHandle, "AsccommCcuKernelRegisterStart");
    g_AsccommCcuKernelRegister = (AsccommCcuKernelRegisterFunc)dlsym(asccommHandle, "AsccommCcuKernelRegister");
    g_AsccommCcuKernelRegisterEnd = (AsccommCcuKernelRegisterEndFunc)dlsym(asccommHandle, "AsccommCcuKernelRegisterEnd");
    g_AsccommCcuKernelLaunch = (AsccommCcuKernelLaunchFunc)dlsym(asccommHandle, "AsccommCcuKernelLaunch");
    g_AsccommCcuGetMemToken = (AsccommCcuGetMemTokenFunc)dlsym(asccommHandle, "AsccommCcuGetMemToken");

    // 用工厂函数覆盖 ccu_launch.h 中的 weak 符号声明
    // 通过函数指针直接调用，不经过 weak stub
}
