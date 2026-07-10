/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_common.h"

#include "ops_executor.h"
#include "base_launcher.h"
#include "log.h"

namespace ops_hccl {

/**
 * 算子执行入口。
 * 流程：
 *   1. 通过 alg 获取引擎（launcher）与执行器（executor）；
 *   2. executor 计算算法分级信息与资源需求；
 *   3. 引擎按自身方式创建运行时资源；
 *   4. 引擎下发 kernel，内部回调 executor.Orchestrate 完成算法编排。
 */
HcclResult HcclExecOp(HcclComm comm, OpParam &param,
                      std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
                      HcclAlgorithm &alg, const ResPackGraphMode &resPack)
{
    // Todo：Kernel缓存

    // 1. 获取引擎与执行器
    auto engine = alg.GetEngine(comm);
    auto executor = alg.GetExecutor(param);

    // 2. 计算算法分级信息（topoMatch → algHierarchyInfo_）
    CHK_RET(executor->CalcAlgHierarchyInfo(comm, topoInfo.get()));

    // 3. 计算资源需求：executor.CalcRes 递归遍历 algoExecDesc，
    //    汇聚各层级 channel/notify/thread 需求到单个 AlgResourceRequest
    AlgResourceRequest resReq;
    CHK_RET(executor->CalcRes(resReq));

    // 4. 引擎创建运行时资源：
    //    AICPU → channel、notify、thread
    //    AIV   → channel、共享内存
    //    CCU   → cclMem、notify、thread、channel
    CHK_RET(engine->CreateRes(resReq));

    // 5. 下发 kernel：引擎内部回调 executor.Orchestrate 完成算法编排
    CHK_RET(engine->LaunchKernel(param, *executor));

    return HCCL_SUCCESS;
}

HcclResult HcclCalcTopoInfo(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo)
{
    HCCL_INFO("[%s] HcclCalcTopoInfo start.", __func__);
    uint64_t size = 0;
    void *ctx = nullptr;
    // 若获取Context失败，表示对应Context尚未缓存
    HcclResult ret = HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size);
    if (ret == HCCL_E_NOT_FOUND || ret == HCCL_E_PARA) {
        // 初始化topoInfo
        CHK_RET(InitRankInfo(comm, topoInfo.get()));
        // 序列化
        std::vector<char> seq = topoInfo->Serialize();
        size = seq.size();
        // 创建新的Context保存
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
        CHK_SAFETY_FUNC_RET(memcpy_s(ctx, size, seq.data(), size));
        return HCCL_SUCCESS;
    }
    char *ctxTemp = reinterpret_cast<char*>(ctx);
    std::vector<char> seq(ctxTemp, ctxTemp + size);
    TopoInfoWithNetLayerDetails topoInfoTemp;
    topoInfoTemp.DeSerialize(seq);
    topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>(std::move(topoInfoTemp));
    HCCL_INFO("[%s] HcclCalcTopoInfo end.", __func__);
    return HCCL_SUCCESS;
}

// 检查非对称拓扑支持情况
// 仅 AllGather, AllReduce, ReduceScatter 支持跨框非对称拓扑，其他算子拦截
HcclResult CheckAsymmetricTopoSupport(HcclCMDType opType, const TopoInfoWithNetLayerDetails* topoInfo)
{
    // 仅在跨框非对称场景下检查
    if (topoInfo->topoLevelNums > 1 && topoInfo->multiModuleDiffDeviceNumMode) {
        // 已适配非对称的算子：AllGather, AllReduce, ReduceScatter, AllToAll(V/VC)
        bool isSupportedOp = (opType == HcclCMDType::HCCL_CMD_ALLGATHER ||
                             opType == HcclCMDType::HCCL_CMD_ALLREDUCE ||
                             opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
                             opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
                             opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
                             opType == HcclCMDType::HCCL_CMD_ALLTOALLVC);
        if (!isSupportedOp) {
            HCCL_ERROR("[CheckAsymmetricTopoSupport] OpType[%d] does not support asymmetric topology "
                "(multi-module diff device num mode), only ALLGATHER/ALLREDUCE/REDUCE_SCATTER/ALLTOALL are supported.",
                opType);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}


}  // namespace ops_hccl
