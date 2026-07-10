/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_OP_COMMON
#define OPS_HCCL_OP_COMMON

#include <memory>

#include "hccl.h"
#include "alg_param.h"
#include "hccl_algorithm.h"

namespace ops_hccl {

class HcclAlgorithm;

/**
 * 算子执行入口。
 * 流程：
 *   1. 通过 alg 获取引擎（launcher）与执行器（executor）；
 *   2. executor 计算算法分级信息与资源需求；
 *   3. 引擎按自身方式创建运行时资源；
 *   4. 引擎下发 kernel，内部回调 executor.Orchestrate 完成算法编排。
 * 输入参数：
 *   - comm：通信域
 *   - param：算子参数
 *   - topoInfo：带网络分层的拓扑信息
 *   - alg：由 Selector 选定的算法描述对象，提供 GetEngine/GetExecutor 能力
 *   - resPack：图模式资源打包信息（streams、scratchMem 等）
 * 返回值：
 *   - HCCL_SUCCESS：执行成功
 *   - 其他：执行失败错误码
 */
HcclResult HcclExecOp(HcclComm comm, OpParam &param,
                      std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
                      HcclAlgorithm &alg, const ResPackGraphMode &resPack);

HcclResult HcclCalcTopoInfo(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo);
HcclResult CheckAsymmetricTopoSupport(HcclCMDType opType, const TopoInfoWithNetLayerDetails* topoInfo);

}  // namespace ops_hccl

#endif  // OPS_HCCL_OP_COMMON
