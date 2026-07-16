/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "order_launch.h"
#include "hcomm_primitives_dl.h"
#include "hccl_res_dl.h"

namespace ops_hccl {
/**
 * @brief OPBASE模式第一步：
 * 执行流程：
 */
HcclResult HcclOpbaseLaunchInOrderToOrderStream(HcclComm comm, OpParam &param, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout)
{
    // 获取Host侧保序流
    ThreadHandle hostOrderThread;
    ThreadHandle exportHostOrderThread;
    CHK_RET(HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE, HOST_ORDER_THREAD_NOTIFY_NUM, &hostOrderThread));
    if (hostOrderThread == 0) {
        HCCL_INFO("Communication domains Number is less than cores Number, OrderLaunch is not Required.");
        return HCCL_SUCCESS;
    }
    // 导出Host侧保序流
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &hostOrderThread, COMM_ENGINE_AICPU_TS, &exportHostOrderThread));
    param.exportHostOrderThread = exportHostOrderThread;

    // 获取Device侧保序流
    ThreadHandle deviceOrderThread;
    CHK_RET(HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, DEVICE_ORDER_THREAD_NOTIFY_NUM, &deviceOrderThread));
    param.deviceOrderThread = deviceOrderThread;

    // notify0
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(hostOrderThread, unfoldThread, notifyIdx)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(unfoldThread, notifyIdx, timeout)));
    HCCL_INFO("Opbase OrderLaunch Phase1 Success.");
    return HCCL_SUCCESS;
}

/**
 * @brief OPBASE模式第二步：
 * 执行流程：
 */
HcclResult HcclOpbaseLaunchInOrderToKernelStream(HcclComm comm, u32 notifyIdx, u32 timeout)
{

    // 获取Host侧保序流
    ThreadHandle hostOrderThread;
    CHK_RET(HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE, HOST_ORDER_THREAD_NOTIFY_NUM, &hostOrderThread));
    if (hostOrderThread == 0) {
        HCCL_INFO("Communication domains Number is less than cores Number, OrderLaunch is not Required.");
        return HCCL_SUCCESS;
    }

    // wait notify1
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(hostOrderThread, notifyIdx, timeout)));
    HCCL_INFO("Opbase OrderLaunch Phase2 Success.");
    return HCCL_SUCCESS;
}

/**
 * @brief ACLGRAPH模式第一步：
 * 执行流程：
 */
HcclResult HcclAclgraphLaunchInOrderToOrderStream(HcclComm comm, OpParam &param, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout, HcclRtEvent event)
{

    HCCL_INFO("Aclgraph OrderLaunch Phase1 Success.");
    return HCCL_SUCCESS;
}

/**
 * @brief ACLGRAPH模式第二步
 * 执行流程：
 */
HcclResult HcclAclgraphLaunchInOrderToKernelStream(HcclComm comm, u32 notifyIdx, u32 timeout, HcclRtEvent event)
{

    HCCL_INFO("Aclgraph OrderLaunch Phase2 Success.");
    return HCCL_SUCCESS;
}

/**
 * @brief GE图模式第一步
 * 执行流程：
 */
HcclResult HcclHcommLaunchInOrderToOrderStream(HcclComm comm, OpParam &param, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout)
{
    // 获取Host侧保序流
    ThreadHandle hostOrderThread;
    ThreadHandle exportHostOrderThread;
    CHK_RET(HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE, HOST_ORDER_THREAD_NOTIFY_NUM, &hostOrderThread));
    if (hostOrderThread == 0) {
        HCCL_INFO("Communication domains Number is less than cores Number, OrderLaunch is not Required.");
        return HCCL_SUCCESS;
    }

    // 导出Host侧保序流
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &hostOrderThread, COMM_ENGINE_AICPU_TS, &exportHostOrderThread));
    param.exportHostOrderThread = exportHostOrderThread;

    // 获取Device侧保序流
    ThreadHandle deviceOrderThread;
    CHK_RET(HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, DEVICE_ORDER_THREAD_NOTIFY_NUM, &deviceOrderThread));
    param.deviceOrderThread = deviceOrderThread;

    // notify0
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(hostOrderThread, unfoldThread, notifyIdx)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(unfoldThread, notifyIdx, timeout)));
    HCCL_INFO("GE OrderLaunch Phase1 Success.");
    return HCCL_SUCCESS;
}

/**
 * @brief GE图模式第二步
 * 执行流程：
 */
HcclResult HcclHcommLaunchInOrderToKernelStream(HcclComm comm, u32 notifyIdx, u32 timeout)
{
    // 获取Host侧保序流
    ThreadHandle hostOrderThread;
    CHK_RET(HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE, HOST_ORDER_THREAD_NOTIFY_NUM, &hostOrderThread));
    if (hostOrderThread == 0) {
        HCCL_INFO("Communication domains Number is less than cores Number, OrderLaunch is not Required.");
        return HCCL_SUCCESS;
    }

    // wait notify1
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(hostOrderThread, notifyIdx, timeout)));
    HCCL_INFO("GE OrderLaunch Phase2 Success.");
    return HCCL_SUCCESS;
}
}  // namespace ops_hccl