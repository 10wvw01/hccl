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

/**
 * @brief OPBASE模式第一步：
 * 执行流程：
 */
HcclResult HcclOpbaseLaunchInOrderToOrderStream(HcclComm comm, aclrtStream kernelStream, void *notify0, void *notify1, u32 timeout)
{

    return HCCL_SUCCESS;
}

/**
 * @brief OPBASE模式第二步：
 * 执行流程：
 */
HcclResult HcclOpbaseLaunchInOrderToKernelStream(HcclComm comm, aclrtStream kernelStream, void *notify0, void *notify1, u32 timeout)
{

    return HCCL_SUCCESS;
}

/**
 * @brief ACLGRAPH模式第一步：
 * 执行流程：
 */
HcclResult HcclAclgraphLaunchInOrderToOrderStream(HcclComm comm, aclrtStream kernelStream, void *notify0, void *notify1, u32 timeout, HcclRtEvent event)
{

    return HCCL_SUCCESS;
}

/**
 * @brief ACLGRAPH模式第二步
 * 执行流程：
 */
HcclResult HcclAclgraphLaunchInOrderToKernelStream(HcclComm comm, aclrtStream kernelStream, HcclRtEvent event)
{

    return HCCL_SUCCESS;
}

/**
 * @brief GE图模式第一步
 * 执行流程：
 */
HcclResult HcclHcommLaunchInOrderToOrderStream(HcclComm comm, aclrtStream kernelStream, u32 graphId, void *notify0, void *notify1, u32 timeout)
{

    return HCCL_SUCCESS;
}

/**
 * @brief GE图模式第二步
 * 执行流程：
 */
HcclResult HcclHcommLaunchInOrderToKernelStream(HcclComm comm, aclrtStream kernelStream, u32 graphId, void *notify0, void *notify1, u32 timeout)
{

    return HCCL_SUCCESS;
}