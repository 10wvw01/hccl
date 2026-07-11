/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_sche_launcher.h"

#include <hccl/hccl_comm.h>
#include "ccu_sche_res.h"
#include "ops_executor.h"
#include "log.h"

namespace ops_hccl {

HcclResult CcuScheLauncher::CreateRes(AlgResourceRequest &res)
{
    HCCL_INFO("[CcuScheLauncher][CreateRes] start.");

    // 1. 回填标量资源需求字段（迁移自 op_common.cc:1536-1538）
    resCtx_.notifyNumOnMainThread = res.notifyNumOnMainThread;
    resCtx_.slaveThreadNum = res.slaveThreadNum;
    resCtx_.notifyNumPerThread = res.notifyNumPerThread;

    // 2. 获取 CCL buffer 作为 scratch buffer（迁移自 op_common.cc:1530-1535）
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm_, &cclBufferAddr, &cclBufferSize));
    resCtx_.cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};

    // 3. 创建 thread：主线程 + slaveThreadNum 个从线程（占位，与 aicpu_launcher 一致）
    resCtx_.threads.resize(res.slaveThreadNum + 1);

    // 4. 以 kernel 为粒度申请 channel（迁移自 op_common.cc:1542-1549）
    auto channelRet = CcuScheGetChannelForCcu(comm_, res);
    if (channelRet == HCCL_E_UNAVAIL) {
        HCCL_WARNING("[CcuScheLauncher][CreateRes] CcuScheGetChannelForCcu unavailable, try to fallback.");
        return HCCL_E_UNAVAIL;
    }
    CHK_RET(channelRet);

    // 5. 注册 CCU kernel 句柄（迁移自 op_common.cc:1551-1558）
    auto kernelRet = CcuScheGetCcuKernel(comm_, res, resCtx_);
    if (kernelRet == HCCL_E_UNAVAIL) {
        HCCL_WARNING("[CcuScheLauncher][CreateRes] CcuScheGetCcuKernel unavailable, try to fallback.");
        return HCCL_E_UNAVAIL;
    }
    CHK_RET(kernelRet);

    HCCL_INFO("[CcuScheLauncher][CreateRes] success, slaveThreadNum[%u], notifyNumOnMainThread[%u], ccuKernelNum[%zu]",
              resCtx_.slaveThreadNum, resCtx_.notifyNumOnMainThread, resCtx_.ccuKernels.size());
    return HCCL_SUCCESS;
}

HcclResult CcuScheLauncher::LaunchKernel(const OpParam &param, OpsExecutor &executor)
{
    HCCL_INFO("[CcuScheLauncher][LaunchKernel] start, commName[%s], tag[%s], algTag[%s]",
              param.commName, param.tag, param.algTag);

    CHK_RET(CcuScheLaunchKernel(param, executor, resCtx_));

    HCCL_INFO("[CcuScheLauncher][LaunchKernel] end, tag[%s], algTag[%s], commName[%s]",
              param.tag, param.algTag, param.commName);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
