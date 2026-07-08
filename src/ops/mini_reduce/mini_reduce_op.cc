/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mini_reduce_op.h"
#include "ins_v2_mini_reduce_sole_executor.h"
#include "ins_temp_mini_reduce_mesh_1D.h"
#include "topo_match_1d.h"
#include "op_common_ops.h"
#include "coll_alg_v2_exec_registry.h"
#include "log.h"

namespace ops_hccl {

// ============================================================
// 注册 Executor: 将 MiniReduceMesh1D 算法注册到 REDUCE 算子
// ============================================================
// 复用现有的 HCCL_CMD_REDUCE 算子类型
// 算法名: "MiniReduceMesh1D"
// 使用: TopoMatch1D (单层 Mesh 1D) + InsTempMiniReduceMesh1D (我们的模板)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_REDUCE, MiniReduceMesh1D,
                 MiniReduceSoleExecutor, TopoMatch1D, InsTempMiniReduceMesh1D);

// ============================================================
// MiniReduce 算子入口 (学习用途, 不注册到公开 API)
// ============================================================
HcclResult MiniReduce(void *sendBuf, void *recvBuf, uint64_t count,
                      HcclDataType dataType, HcclReduceOp op,
                      uint32_t root, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("[MiniReduce] Start, root[%u] count[%llu] dataType[%d] op[%d]",
              root, count, static_cast<int>(dataType), static_cast<int>(op));

    // 参数校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(count == 0, HCCL_WARNING("[MiniReduce] count is 0, return success"), HCCL_SUCCESS);

    // 构建 OpParam
    OpParam param;
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));

    // 设置 tag (用于资源缓存)
    CHK_RET(HcclGetCommName(comm, param.commName));
    int ret = sprintf_s(param.tag, sizeof(param.tag), "MiniReduce_%s", param.commName);
    CHK_PRT_RET((ret <= 0), HCCL_ERROR("failed to fill param.tag"), HCCL_E_INTERNAL);

    // 填充参数
    param.stream = stream;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.reduceType = op;
    param.root = root;
    param.userRank = userRank;
    param.DataDes.count = count;
    param.DataDes.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE;  // 复用 REDUCE 类型
    param.opMode = OpMode::OPBASE;
    param.inputSize = count * DATATYPE_SIZE_TABLE[dataType];
    param.outputSize = param.inputSize;

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    param.deviceType = deviceType;

    // 显式指定算法名为我们注册的 MiniReduceMesh1D
    CHK_SAFETY_FUNC_RET(strcpy_s(param.algName, sizeof(param.algName), "MiniReduceMesh1D"));

    // 强制使用 AICPU 引擎
    param.opExecuteConfig = OpExecuteConfig::AICPU;

    HCCL_INFO("[MiniReduce] tag[%s] algName[%s] rank[%u/%u]",
              param.tag, param.algName, userRank, rankSize);

    // 计算拓扑信息
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo =
        std::make_unique<TopoInfoWithNetLayerDetails>();
    CHK_RET(HcclCalcTopoInfo(comm, param, topoInfo));

    // 单卡: 直接拷贝
    if (rankSize == 1) {
        HCCL_WARNING("[MiniReduce] rankSize==1, direct copy");
        CHK_RET(SingleRankProc(comm, param));
        return HCCL_SUCCESS;
    }

    // 执行算子
    std::string algName = param.algName;
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName));

    HCCL_INFO("[MiniReduce] Success");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
