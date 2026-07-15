/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_dlsym.h"
#include "hccl_res_dl.h"
#include "ccu_res_dl.h"
#include "hccl_ccu_res_dl.h"
#include "ccu_launch_dl.h"
#include "ccu_primitives_impl_dl.h"
#include "hccl_rank_graph_dl.h"
#include "hcomm_primitives_dl.h"
#include "hccl_inner_dl.h"
#include "hcomm_host_profiling_dl.h"
#include "hccl_host_comm_dl.h"
#include "hccl_res_expt_dl.h"
#include <pthread.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <acl/acl.h>

static void* gLibHandle = nullptr;
static int gHcommVersion = 0;
static pthread_once_t gDlInitOnce = PTHREAD_ONCE_INIT;

int GetHcommVersion(void) {
    if (gHcommVersion == 0) {
        char hcommPkgName[] = "hcomm";
        if (aclsysGetVersionNum(hcommPkgName, &gHcommVersion) != ACL_SUCCESS) {
            gHcommVersion = 0;
        }
    }

    return gHcommVersion;
}

bool HcommIsProfilingSupported()
{
    if (GetHcommVersion() >= CANN_VERSION(9, 0, 0)) {
        return true;
    } else {
        return false;
    }
}

bool HcommIsExportThreadSupported()
{
    if (GetHcommVersion() >= CANN_VERSION(9, 0, 0) && HcommIsSupportHcclThreadExportToCommEngine()) {
        return true;
    } else {
        return false;
    }
}

// 初始化
static void HcommDlInitImpl(void) {
    gLibHandle = dlopen("libhcomm.so", RTLD_NOW);
    if (!gLibHandle) {
        fprintf(stderr, "[HcclWrapper] Failed to open libhcomm: %s\n", dlerror());
        return;
    }

    dlerror();

    HcclResDlInit(gLibHandle);
    HcclRankGraphDlInit(gLibHandle);
    HcommPrimitivesDlInit(gLibHandle);
    HcclInnerDlInit(gLibHandle);
    HcommProfilingDlInit(gLibHandle);
    HcclCommDlInit(gLibHandle);
    HcclResExptDlInit(gLibHandle);
    CcuResDlInit(gLibHandle);
    HcclCcuResDlInit(gLibHandle);
    CcuLaunchDlInit(gLibHandle);
    CcuPrimitivesImplDlInit(gLibHandle);
}

void HcommDlInit(void) {
    pthread_once(&gDlInitOnce, HcommDlInitImpl);
}
