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
#include "dlhcomm_function.h"

namespace ops_hccl {
static HcclResult OpLaunchGetUnfoldStream(HcclComm comm, ThreadHandle unfoldThread, aclrtStream &resolvedStream)
{
    void *unfoldStream = nullptr;
    auto &HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    HcclResult ret;
    if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo) {
        resolvedStream = nullptr;
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcclThreadResGetInfo is invalid.");
        return HCCL_SUCCESS;
    } else {
        ret = HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, unfoldThread, 0, sizeof(void *), &unfoldStream);
        if (ret == HCCL_E_NOT_SUPPORT) {
            resolvedStream = nullptr;
            HCCL_WARNING("HcclThreadResGetInfoFunc dlHcclThreadResGetInfo not support.");
            return HCCL_SUCCESS;
        } else if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("HcclThreadResGetInfoFunc dlHcclThreadResGetInfo not success.");
            return HCCL_SUCCESS;
        } else {
            resolvedStream = unfoldStream;
        }
    }
    return HCCL_SUCCESS;
}

/* TODO
static HcclResult OpLaunchGetHostOrderStream(ThreadHandle hostOrderThread, aclrtStream &resolvedStream)
{
    void *hostOrderStream = nullptr;
    auto &HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    HcclResult ret;
    if (!HcclThreadResGetInfoFunc.dlHcommThreadResGetInfo) {
        resolvedStream = nullptr;
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcommThreadResGetInfo is invalid.");
        return HCCL_SUCCESS;
    } else {
        ret = HcclThreadResGetInfoFunc.dlHcommThreadResGetInfo(hostOrderThread, 0, sizeof(void *), &hostOrderStream);
        if (ret == HCCL_E_NOT_SUPPORT) {
            resolvedStream = nullptr;
            HCCL_WARNING("HcclThreadResGetInfoFunc dlHcommThreadResGetInfo not support.");
            return HCCL_SUCCESS;
        } else if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("HcclThreadResGetInfoFunc dlHcommThreadResGetInfo not success.");
            return HCCL_SUCCESS;
        } else {
            resolvedStream = hostOrderStream;
        }
    }
    return HCCL_SUCCESS;
}
*/

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
        param.exportHostOrderThread = 0;
        param.deviceOrderThread = 0;
        HCCL_INFO("Communication domains Number is less than cores Number, Opbase OrderLaunch is not Required.");
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
        HCCL_INFO("Communication domains Number is less than cores Number, Opbase OrderLaunch is not Required.");
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
    // 获取Host侧保序流
    ThreadHandle hostOrderThread;
    ThreadHandle exportHostOrderThread;
    CHK_RET(HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE, HOST_ORDER_THREAD_NOTIFY_NUM, &hostOrderThread));
    if (hostOrderThread == 0) {
        param.exportHostOrderThread = 0;
        param.deviceOrderThread = 0;
        HCCL_INFO("Communication domains Number is less than cores Number, Aclgraph OrderLaunch is not Required.");
        return HCCL_SUCCESS;
    }
    // 导出Host侧保序流
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &hostOrderThread, COMM_ENGINE_AICPU_TS, &exportHostOrderThread));
    param.exportHostOrderThread = exportHostOrderThread;

    /*
    // hostOrderThread、unfoldThread分别获取对应Stream
    aclrtStream hostOrderStream;
    aclrtStream unflodStream;
    HcclResult ret;
    ret = OpLaunchGetHostOrderStream(hostOrderThread, hostOrderStream);
    if (ret != HCCL_SUCCESS) {
        HCCL_INFO("OpLaunchGetHostThreadStream Fail, Aclgraph OrderLaunch is not use.");
    }
    ret = OpLaunchGetUnfoldStream(comm, unfoldThread, unflodStream);
    if (ret != HCCL_SUCCESS) {
        HCCL_INFO("OpLaunchGetUnfoldStream Fail, Aclgraph OrderLaunch is not use.");
    }

    // hostOrderStream、unflodStream的event的操作
    aclError retEvent = ACL_SUCCESS;
    retEvent = aclrtRecordEvent(event, hostOrderStream);
    CHK_PRT_RET(retEvent != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtRecordEvent failed, ret[%d]", __func__, retEvent), HCCL_E_RUNTIME);

    retEvent = aclrtStreamWaitEvent(unflodStream, event);
    CHK_PRT_RET(retEvent != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtStreamWaitEvent failed, ret[%d]", __func__, retEvent), HCCL_E_RUNTIME);
    */

    // 获取Device侧保序流
    ThreadHandle deviceOrderThread;
    CHK_RET(HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, DEVICE_ORDER_THREAD_NOTIFY_NUM, &deviceOrderThread));
    param.deviceOrderThread = deviceOrderThread;

    // notify0
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(hostOrderThread, unfoldThread, notifyIdx)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(unfoldThread, notifyIdx, timeout)));

    HCCL_INFO("Aclgraph OrderLaunch Phase1 Success.");
    return HCCL_SUCCESS;
}

/**
 * @brief ACLGRAPH模式第二步
 * 执行流程：
 */
HcclResult HcclAclgraphLaunchInOrderToKernelStream(HcclComm comm, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout, HcclRtEvent event)
{
    // 获取Host侧保序流
    ThreadHandle hostOrderThread;
    CHK_RET(HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE, HOST_ORDER_THREAD_NOTIFY_NUM, &hostOrderThread));
    if (hostOrderThread == 0) {
        HCCL_INFO("Communication domains Number is less than cores Number, Aclgraph OrderLaunch is not Required.");
        return HCCL_SUCCESS;
    }

    // wait notify1
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(hostOrderThread, notifyIdx, timeout)));

    /*
    // hostOrderThread、unfoldThread分别获取对应Stream
    aclrtStream hostOrderStream;
    aclrtStream unflodStream;
    HcclResult ret;
    ret = OpLaunchGetHostOrderStream(hostOrderThread, hostOrderStream);
    if (ret != HCCL_SUCCESS) {
        HCCL_INFO("OpLaunchGetHostThreadStream Fail, Aclgraph OrderLaunch is not use.");
    }
    ret = OpLaunchGetUnfoldStream(comm, unfoldThread, unflodStream);
    if (ret != HCCL_SUCCESS) {
        HCCL_INFO("OpLaunchGetUnfoldStream Fail, Aclgraph OrderLaunch is not use.");
    }

    // hostOrderStream、unflodStream的event的操作
    aclError retEvent = ACL_SUCCESS;
    retEvent = aclrtRecordEvent(event, hostOrderStream);
    CHK_PRT_RET(retEvent != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtRecordEvent failed, ret[%d]", __func__, retEvent), HCCL_E_RUNTIME);

    retEvent = aclrtStreamWaitEvent(unflodStream, event);
    CHK_PRT_RET(retEvent != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtStreamWaitEvent failed, ret[%d]", __func__, retEvent), HCCL_E_RUNTIME);
    */

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
        param.exportHostOrderThread = 0;
        param.deviceOrderThread = 0;
        HCCL_INFO("Communication domains Number is less than cores Number, GE OrderLaunch is not Required.");
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
        HCCL_INFO("Communication domains Number is less than cores Number, GE OrderLaunch is not Required.");
        return HCCL_SUCCESS;
    }

    // wait notify1
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(hostOrderThread, notifyIdx, timeout)));
    HCCL_INFO("GE OrderLaunch Phase2 Success.");
    return HCCL_SUCCESS;
}
}  // namespace ops_hccl