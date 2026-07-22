/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_DATA_TRANSFER_H
#define OPS_HCCL_DATA_TRANSFER_H

#include "base_engine.h"

namespace ops_hccl {

/**
 * AICPU 数据传输入口（从 AiCpuEngine::Send 剥离为独立函数）。
 * 根据 TransferContext 选择 Send/Recv/SendRecv × Write/Read 路由并执行实际数据搬移。
 * 输入参数：
 *   - ctx: 发送数据上下文
 * 返回值：
 *   - HCCL_SUCCESS: 数据发送成功
 *   - HCCL_E_INTERNAL: 数据发送失败
 */
HcclResult DataTransferSend(const TransferContext &ctx);

}  // namespace ops_hccl

#endif  // OPS_HCCL_DATA_TRANSFER_H
