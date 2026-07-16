/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include"order_launch.h"
#include "hcomm_primitives_dl.h"

/**
 * @brief OPBASE模式第一步：
 * 执行流程：
 */
HcclResult HcclOpbaseLaunchInOrderToOrderStream(HcclComm comm, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout)
{
    // 获取Host侧保序流

    // notify0
    HcommThreadNotifyRecordOnThread( , unfoldThread, notifyIdx);
    HcommThreadNotifyWaitOnThread(unfoldThread, notifyIdx, timeout);
    HCCL_INFO("Opbase OrderLaunch Phase1 Success.");
    return HCCL_SUCCESS;
}

/**
 * @brief OPBASE模式第二步：
 * 执行流程：
 */
HcclResult HcclOpbaseLaunchInOrderToKernelStream(HcclComm comm, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout)
{

    // 获取Host侧保序流

    // wait notify1
    HcommThreadNotifyWaitOnThread( , , timeout);
    HCCL_INFO("Opbase OrderLaunch Phase2 Success.");
    return HCCL_SUCCESS;
}

/**
 * @brief ACLGRAPH模式第一步：
 * 执行流程：
 */
HcclResult HcclAclgraphLaunchInOrderToOrderStream(HcclComm comm, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout, HcclRtEvent event)
{

    HCCL_INFO("Aclgraph OrderLaunch Phase1 Success.");
    return HCCL_SUCCESS;
}

/**
 * @brief ACLGRAPH模式第二步
 * 执行流程：
 */
HcclResult HcclAclgraphLaunchInOrderToKernelStream(HcclComm comm, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout, HcclRtEvent event)
{

    HCCL_INFO("Aclgraph OrderLaunch Phase2 Success.");
    return HCCL_SUCCESS;
}

/**
 * @brief GE图模式第一步
 * 执行流程：
 */
HcclResult HcclHcommLaunchInOrderToOrderStream(HcclComm comm, ThreadHandle unfoldThread, u32 graphId, u32 notifyIdx, u32 timeout)
{

    HCCL_INFO("GE OrderLaunch Phase1 Success.");
    return HCCL_SUCCESS;
}

/**
 * @brief GE图模式第二步
 * 执行流程：
 */
HcclResult HcclHcommLaunchInOrderToKernelStream(HcclComm comm, ThreadHandle unfoldThread, u32 graphId, u32 notifyIdx, u32 timeout)
{

    HCCL_INFO("GE OrderLaunch Phase2 Success.");
    return HCCL_SUCCESS;
}