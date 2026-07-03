/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_AICPU_KERNEL_LAUNCH_H
#define OPS_HCCL_AICPU_KERNEL_LAUNCH_H

#include "alg_param.h"

namespace ops_hccl {

class OpsExecutor;

/**
 * 还原 BatchSendRecv 算子的变长数据指针。
 */
HcclResult RestoreVarDataBatchSendRecv(OpParam &param);

/**
 * 还原 AlltoAllV/AlltoAllVC/AlltoAll 算子的变长数据指针。
 */
HcclResult RestoreVarDataAlltoAllV(OpParam &param, const AlgResourceCtxSerializable &resCtx);

/**
 * 还原 ReduceScatterV 算子的变长数据指针。
 */
HcclResult RestoreVarDataReduceScatterV(OpParam &param, const AlgResourceCtxSerializable &resCtx);

/**
 * 还原 AllGatherV 算子的变长数据指针。
 */
HcclResult RestoreVarDataAllGatherV(OpParam &param, const AlgResourceCtxSerializable &resCtx);

/**
 * AICPU kernel 下发入口。
 * 工作流程：
 *   1. 获取通信域句柄（HcommAcquireComm）；
 *   2. 根据 opType 还原变长数据（如 AllGatherV 的 counts/displs）；
 *   3. 设置 batch mode，注册 DFX op 信息与 profiling；
 *   4. 主 thread 等待 Host stream 的 notify 通知；
 *   5. 调用 executor.Orchestrate 驱动算法编排；
 *   6. 上报 profiling，通知 Host stream 完成，结束 batch mode；
 *   7. 释放通信域句柄（HcommReleaseComm）。
 * 输入参数：
 *   - param: 算子参数，包含 commName、tag、opType、数据描述等
 *   - executor: 执行器引用，提供 Orchestrate 接口驱动算法编排
 *   - resCtx: 已创建的资源上下文，包含 channel/notify/thread/cclMem 等
 * 返回值：
 *   - HCCL_SUCCESS: kernel 下发并执行成功
 *   - HCCL_E_PARA: 参数非法
 *   - HCCL_E_INTERNAL: 下发或执行失败
 */
HcclResult HcclLaunchAicpuKernel(const OpParam &param, OpsExecutor &executor,
                                 AlgResourceCtxSerializable &resCtx);

}  // namespace ops_hccl
#endif
