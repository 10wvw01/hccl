/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "hccl.h"
#include "alltoall_aicpu.h"
#include "alltoall_ccu.h"

HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    const char *mode = getenv("HCCL_OP_EXPANSION_MODE");
    if (mode == nullptr || strcmp(mode, "AI_CPU") == 0) {
        // 默认使用 AICPU 通信引擎
        CHK_RET(ops_aicpu::HcclAlltoAll(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream));
    } else {
        // CCU 通信引擎
        CHK_RET(ops_ccu::HcclAlltoAll(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream));
    }
    return HCCL_SUCCESS;
}
