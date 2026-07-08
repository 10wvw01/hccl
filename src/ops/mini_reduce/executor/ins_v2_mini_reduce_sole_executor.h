/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_V2_MINI_REDUCE_SOLE_EXECUTOR_H
#define INS_V2_MINI_REDUCE_SOLE_EXECUTOR_H

#include "executor_common_ops.h"
#include "topo_match_base.h"

namespace ops_hccl {

// MiniReduce Sole Executor
// 模板化设计，复用 ReduceSoleExecutor 的框架
// 第一个模板参数 AlgTopoMatch: 拓扑匹配 (如 TopoMatch1D)
// 第二个模板参数 AlgTemplate:  算法模板 (如 InsTempMiniReduceMesh1D)
template <typename AlgTopoMatch, typename AlgTemplate>
class MiniReduceSoleExecutor : public InsCollAlgBase {
public:
    explicit MiniReduceSoleExecutor();
    ~MiniReduceSoleExecutor() override = default;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
        AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

    HcclResult CalcRes(HcclComm comm, const OpParam &param,
                       const TopoInfoWithNetLayerDetails *topoInfo,
                       const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
                       AlgResourceRequest &resourceRequest) override;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

private:
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;
};

// ================ 模板实现 ================

template <typename AlgTopoMatch, typename AlgTemplate>
MiniReduceSoleExecutor<AlgTopoMatch, AlgTemplate>::MiniReduceSoleExecutor()
{
}

template <typename AlgTopoMatch, typename AlgTemplate>
HcclResult MiniReduceSoleExecutor<AlgTopoMatch, AlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    CHK_PTR_NULL(topoInfo);
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate>
HcclResult MiniReduceSoleExecutor<AlgTopoMatch, AlgTemplate>::CalcRes(
    HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
    // 使用 L0 子通信域信息创建 Template
    std::shared_ptr<AlgTemplate> algTemplate =
        std::make_shared<AlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename AlgTemplate>
HcclResult MiniReduceSoleExecutor<AlgTopoMatch, AlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[MiniReduceSoleExecutor][Orchestrate] Start, channels[%zu]", resCtx.channels.size());

    // 设置基本参数
    maxTmpMemSize_ = resCtx.cclMem.size;
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    // 准备 Template 资源
    TemplateResource templateAlgRes;
    if (remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    templateAlgRes.threads = resCtx.threads;

    // 准备 Template 数据参数
    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.count = param.DataDes.count;

    // 创建 Template 实例并执行
    std::shared_ptr<AlgTemplate> algTemplate =
        std::make_shared<AlgTemplate>(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);

    CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));

    HCCL_INFO("[MiniReduceSoleExecutor][Orchestrate] End.");
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl

#endif  // INS_V2_MINI_REDUCE_SOLE_EXECUTOR_H
