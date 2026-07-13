/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_ms_launcher.h"

#include <hccl/hccl_comm.h>
#include "ccu_ms_res.h"
#include "ops_executor.h"
#include "log.h"

namespace ops_hccl {

HcclResult CcuMsLauncher::CreateRes(AlgResourceRequest &res)
{
    HCCL_INFO("[CcuMsLauncher][CreateRes] start.");

    // 1. 回填标量资源需求字段（迁移自 op_common.cc:1536-1538）
    resCtx_.notifyNumOnMainThread = res.notifyNumOnMainThread;
    resCtx_.slaveThreadNum = res.slaveThreadNum;
    resCtx_.notifyNumPerThread = res.notifyNumPerThread;

    // 2. 获取 CCL buffer 作为 scratch buffer（跨 Rank 缓存）
    //    迁移自 op_common.cc:1530-1535 HcclGetHcclBuffer
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm_, &cclBufferAddr, &cclBufferSize));
    resCtx_.cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};

    // 3. 创建 thread：主线程 + slaveThreadNum 个从线程
    //    与 aicpu_launcher.cc 一致采用占位模式，真正的 HcclThreadAcquire 调用待 CreateRes 接口扩展 param/resPack 后补全
    resCtx_.threads.resize(res.slaveThreadNum + 1);

    // 4. 以 kernel 为粒度申请 channel（迁移自 op_common.cc:1542-1549 HcclGetChannelForCcu）
    auto channelRet = CcuMsGetChannelForCcu(comm_, res);
    if (channelRet == HCCL_E_UNAVAIL) {
        HCCL_WARNING("[CcuMsLauncher][CreateRes] CcuMsGetChannelForCcu unavailable, try to fallback.");
        return HCCL_E_UNAVAIL;
    }
    CHK_RET(channelRet);

    // 5. 注册 CCU kernel 句柄（迁移自 op_common.cc:1551-1558 HcclGetCcuKernel）
    auto kernelRet = CcuMsGetCcuKernel(comm_, res, resCtx_);
    if (kernelRet == HCCL_E_UNAVAIL) {
        HCCL_WARNING("[CcuMsLauncher][CreateRes] CcuMsGetCcuKernel unavailable, try to fallback.");
        return HCCL_E_UNAVAIL;
    }
    CHK_RET(kernelRet);

    HCCL_INFO("[CcuMsLauncher][CreateRes] success, slaveThreadNum[%u], notifyNumOnMainThread[%u], ccuKernelNum[%zu]",
              resCtx_.slaveThreadNum, resCtx_.notifyNumOnMainThread, resCtx_.ccuKernels.size());
    return HCCL_SUCCESS;
}

HcclResult CcuMsLauncher::LaunchKernel(const OpParam &param, OpsExecutor &executor)
{
    HCCL_INFO("[CcuMsLauncher][LaunchKernel] start, commName[%s], tag[%s], algTag[%s]",
              param.commName, param.tag, param.algTag);

    // 通过 CcuMsLaunchKernel 入口完成环境准备、算法编排与 profiling 上报
    // 传入 resCtx_（CreateRes 已回填），内部直接使用
    CHK_RET(CcuMsLaunchKernel(param, executor, resCtx_));

    HCCL_INFO("[CcuMsLauncher][LaunchKernel] end, tag[%s], algTag[%s], commName[%s]",
              param.tag, param.algTag, param.commName);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
