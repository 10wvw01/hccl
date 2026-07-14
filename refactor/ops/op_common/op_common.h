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
#include "sal.h" // for HcclUs (used by LogHcclExit declaration)

namespace ops_hccl {

class HcclAlgorithm;

/**
 * 算子执行入口。
 * 流程：
 *   1. 通过 alg 获取引擎（engine）与执行器（executor）；
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
HcclResult HcclExecOp(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    HcclAlgorithm &alg, const ResPackGraphMode &resPack);

HcclResult HcclCalcTopoInfo(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo);
HcclResult CheckAsymmetricTopoSupport(HcclCMDType opType, const TopoInfoWithNetLayerDetails *topoInfo);

// forward-declared: refactor 入口 (.cc) 拷贝自 src，调用 src/ops/op_common/op_common.cc 中定义的
// 函数。函数实现在 src，--refactor 模式下 src 也会被编译，链接器找得到符号。
HcclResult LogHcclExit(const std::string &opName, const char *tag, HcclUs startut, bool forceLog = false);
HcclResult HcclCheckTag(const char *tag);
HcclResult CheckCount(const u64 count);
HcclResult CheckDataType(const HcclDataType dataType, bool needReduce);
HcclResult SingleRankProc(HcclComm comm, OpParam &param);
HcclResult CheckHostDPUOnly(const HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, bool &hostDPUOnly);
HcclResult RegisterKernel();
HcclResult SetExecTimeout(const OpParam &param);
HcclResult SetMultipleDimensionSplitRatio(const OpParam &param);
bool ShouldUseInnerOp(OpExecuteConfig opExecuteConfig);
HcclResult SetOpParamAlgTag(OpParam &param, const std::string &algName);
} // namespace ops_hccl

#endif // OPS_HCCL_OP_COMMON
