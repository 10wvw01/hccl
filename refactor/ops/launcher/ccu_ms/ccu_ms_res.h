/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_MS_RES_H
#define OPS_HCCL_CCU_MS_RES_H

#include "alg_param.h"

namespace ops_hccl {

/**
 * 以 kernel 为粒度申请 channel（迁移自 op_common.cc:1564-1599 HcclGetChannelForCcu）。
 *
 * 流程：
 *   1. 遍历 res.ccuKernelInfos，对每个 kernelInfo 的 channels 调用 HcclChannelAcquire；
 *   2. 回填 kernelArgBase->channels[i] 与 channelCount。
 *
 * 注：原实现使用 param.engine 传入 HcclChannelAcquire，此处 engine 由 CcuMsLauncher 语义决定，
 * 统一使用 CommEngine::COMM_ENGINE_CCU 常量（CCU_MS 与 CCU_SCHE 共用同一底层 CommEngine）。
 * AddExchangeInfo 暂省略（原 op_common.cc:1576 注册一致性校验信息），待 CreateRes 接口扩展 param 后补全。
 *
 * 输入参数：
 *   - comm: 通信域句柄
 *   - res: 资源请求，含 ccuKernelInfos
 * 返回值：
 *   - HCCL_SUCCESS: channel 申请成功
 *   - HCCL_E_UNAVAIL: channel 资源不足（触发回退）
 *   - HCCL_E_INTERNAL: 其他失败
 */
HcclResult CcuMsGetChannelForCcu(HcclComm comm, AlgResourceRequest &res);

/**
 * 注册 CCU kernel 句柄（迁移自 op_common.cc:1601-1664 HcclGetCcuKernel）。
 *
 * 流程：
 *   1. HcclCommQueryCcuIns 获取 CCU 实例句柄；
 *   2. 按 resGroup 分组调用 HcommCcuKernelRegisterStart/Register/End；
 *   3. 回填 resCtx.ccuKernels[i] 与 resCtx.ccuKernelNum。
 *
 * 输入参数：
 *   - comm: 通信域句柄
 *   - res: 资源请求，含 ccuKernelInfos 与 ccuKernelNum
 * 输出参数：
 *   - resCtx: 资源上下文，回填 ccuKernels 与 ccuKernelNum
 * 返回值：
 *   - HCCL_SUCCESS: kernel 注册成功
 *   - HCCL_E_UNAVAIL: kernel 注册不可用（触发回退）
 *   - HCCL_E_INTERNAL: 其他失败
 */
HcclResult CcuMsGetCcuKernel(HcclComm comm, AlgResourceRequest &res, AlgResourceCtxSerializable &resCtx);

}  // namespace ops_hccl

#endif  // OPS_HCCL_CCU_MS_RES_H
