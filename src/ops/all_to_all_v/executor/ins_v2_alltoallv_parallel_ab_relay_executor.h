/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALLTOALLV_PARALLEL_AB_RELAY_EXECUTOR_H
#define HCCLV2_INS_V2_ALLTOALLV_PARALLEL_AB_RELAY_EXECUTOR_H

#include "executor_common_ops.h"

namespace ops_hccl {

template <typename AlgTopoMatch>
class InsV2AlltoAllVParallelABRelayExecutor : public InsCollAlgBase {
public:
    explicit InsV2AlltoAllVParallelABRelayExecutor() = default;
    ~InsV2AlltoAllVParallelABRelayExecutor() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
                       AlgResourceRequest &resourceRequest) override;
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
                                    AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

private:
    struct TemplateResMeta {
        u32 slaveThreadNum{0};
        u32 notifyNumOnMainThread{0};
        std::vector<u32> notifyNumPerThread;
    };

    HcclResult BuildHierarchyInfo(const TopoInfoWithNetLayerDetails *topoInfo,
                                  const AlgHierarchyInfoForAllLevel &algHierarchyInfo);
    HcclResult RestoreChannelMaps(const AlgResourceCtxSerializable &resCtx);
    HcclResult BuildBaseParams(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                               TemplateDataParams &params);
    HcclResult SplitABParams(const TemplateDataParams &baseParams, double ratio,
                             TemplateDataParams &aParams, TemplateDataParams &bParams) const;
    HcclResult RunAOriginalAndPreroute(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                                       const TemplateDataParams &aParams, const TemplateDataParams &bParams);
    HcclResult RunBRelay(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                         const TemplateDataParams &bParams);
    HcclResult BuildRuntimeMetas();
    HcclResult PrepareTemplateResources(const AlgResourceCtxSerializable &resCtx);
    HcclResult CheckRelayScratch(const TemplateDataParams &bParams, const AlgResourceCtxSerializable &resCtx) const;
    HcclResult GetGlobalMaxSendCount(u64 &globalMaxSend) const;
    double GetABRatio(const OpParam &param) const;
    u64 GetRankSize(const std::vector<std::vector<u32>> &vTopo) const;

    u32 myRank_{0};
    u64 rankSize_{0};
    u64 meshSize_{0};
    u64 groupNum_{0};
    u64 rankSizeLevel1_{0};
    HcclDataType dataType_{HCCL_DATA_TYPE_RESERVED};
    u32 dataTypeSize_{0};
    u64 dataCount_{0};
    u64 dataSize_{0};
    u64 slotStride_{0};

    std::vector<std::vector<u32>> intraHierarchyInfo_;
    std::vector<std::vector<u32>> interHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::map<u32, std::vector<ChannelInfo>> intraLinkMap_;
    std::map<u32, std::vector<ChannelInfo>> interLinkMap_;
    std::map<u32, std::vector<ChannelInfo>> aInterLinkMap_;
    std::map<u32, std::vector<ChannelInfo>> bRelayLinkMap_;
    std::vector<u64> remoteTotalSendCountsWithoutSelf_;
    std::vector<u64> remoteMaxSendCountsWithoutSelf_;
    std::vector<bool> remoteTotalSendCountsValid_;

    std::vector<ThreadHandle> threads_;
    ThreadHandle mainThread_{0};
    std::vector<ThreadHandle> aIntraThreads_;
    std::vector<ThreadHandle> aInterThreads_;
    std::vector<ThreadHandle> bRelayThreads_;

    TemplateResMeta aIntraMeta_;
    TemplateResMeta aInterMeta_;
    TemplateResMeta bRelayMeta_;
};

} // namespace ops_hccl

#endif // HCCLV2_INS_V2_ALLTOALLV_PARALLEL_AB_RELAY_EXECUTOR_H
