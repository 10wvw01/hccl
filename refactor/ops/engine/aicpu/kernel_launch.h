/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
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
 * 判断缓存的 AlgResourceCtxSerializable 是否可复用。
 * 条件：param.cacheValid 为 true 且缓存的 commInfoPtr 与当前 param.hcclComm 一致。
 */
inline bool IsResCtxCacheReusable(const AlgResourceCtxSerializable &cachedResCtx, const OpParam &param)
{
    return param.cacheValid && cachedResCtx.commInfoPtr == param.hcclComm;
}

} // namespace ops_hccl

#endif // OPS_HCCL_AICPU_KERNEL_LAUNCH_H
