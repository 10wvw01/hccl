/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_SCHE_RES_H
#define OPS_HCCL_CCU_SCHE_RES_H

#include "alg_param.h"

namespace ops_hccl {

/**
 * 以 kernel 为粒度申请 channel（迁移自 op_common.cc:1564-1599 HcclGetChannelForCcu）。
 *
 * 与 CcuMsGetChannelForCcu 逻辑一致，底层 CommEngine 共用 COMM_ENGINE_CCU。
 * AddExchangeInfo 暂省略，待 CreateRes 接口扩展 param 后补全。
 */
HcclResult CcuScheGetChannelForCcu(HcclComm comm, AlgResourceRequest &res);

/**
 * 注册 CCU kernel 句柄（迁移自 op_common.cc:1601-1664 HcclGetCcuKernel）。
 *
 * 与 CcuMsGetCcuKernel 逻辑一致。
 */
HcclResult CcuScheGetCcuKernel(HcclComm comm, AlgResourceRequest &res, AlgResourceCtxSerializable &resCtx);

}  // namespace ops_hccl

#endif  // OPS_HCCL_CCU_SCHE_RES_H
