/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */

#ifndef HCCLV2_INS_V2_ALLTOALLV_PARALLEL_V2_AB_4P_NO_MEMCPY_EXECUTOR_H
#define HCCLV2_INS_V2_ALLTOALLV_PARALLEL_V2_AB_4P_NO_MEMCPY_EXECUTOR_H

#include "executor_common_ops.h"

namespace ops_hccl {

enum class A2AVV2AB4PBMode {
    SPRAY = 0,
    COLOR = 1,
};

template <typename AlgTopoMatch>
class InsV2AlltoAllVParallelV2AB4PNoMemcpyExecutor : public InsCollAlgBase {
public:
    explicit InsV2AlltoAllVParallelV2AB4PNoMemcpyExecutor() = default;
    ~InsV2AlltoAllVParallelV2AB4PNoMemcpyExecutor() override = default;

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
    HcclResult GetPairCountFromRecvMatrix(const TemplateDataParams &params, u32 srcRank, u32 dstRank,
                                          u64 &count) const;
    HcclResult BuildStageLinkMaps(const TemplateDataParams &params);
    HcclResult BuildBLinkMap();
    HcclResult BuildRuntimeMetas();
    HcclResult PrepareTemplateResources(const AlgResourceCtxSerializable &resCtx);
    HcclResult BuildABExactSlotOffsets(TemplateDataParams &params);
    HcclResult CheckScratch(const AlgResourceCtxSerializable &resCtx) const;
    HcclResult RunAAndBParallel(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                                const TemplateDataParams &aParams, const TemplateDataParams &bParams);
    HcclResult CopySelfToOutputAfterCommunication(const TemplateDataParams &params);
    HcclResult GetGlobalMaxSendCount(u64 &globalMaxSend) const;
    u64 GetRankSize(const std::vector<std::vector<u32>> &vTopo) const;
    HcclResult AddStageLink(u32 peerRank, std::map<u32, std::vector<ChannelInfo>> &stageLinkMap) const;
    HcclResult SelectFourthClosChannel(u32 peerRank, std::vector<ChannelInfo> &channels) const;
    bool IsSameGroup(u32 rankA, u32 rankB) const;
    double GetABRatio(const OpParam &param) const;
    double GetPathSplitRatio(const OpParam &param) const;
    void SplitABPairCount(u64 count, u64 &aCount, u64 &bCount) const;

    u32 myRank_{0};
    u64 rankSize_{0};
    u64 meshSize_{0};
    u64 groupNum_{0};
    HcclDataType dataType_{HCCL_DATA_TYPE_RESERVED};
    u32 dataTypeSize_{0};
    u64 dataCount_{0};
    u64 slotStride_{0};
    u64 exactSlotBytes_{0};
    double abRatio_{0.75};
    double pathSplitRatio_{0.5};
    A2AVV2AB4PBMode bMode_{A2AVV2AB4PBMode::SPRAY};

    std::vector<std::vector<u32>> intraHierarchyInfo_;
    std::vector<std::vector<u32>> interHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::map<u32, std::vector<ChannelInfo>> intraLinkMap_;
    std::map<u32, std::vector<ChannelInfo>> interLinkMap_;
    std::map<u32, std::vector<ChannelInfo>> allLinkMap_;
    std::map<u32, std::vector<ChannelInfo>> stage0LinkMap_;
    std::map<u32, std::vector<ChannelInfo>> stage1LinkMap_;
    std::map<u32, std::vector<ChannelInfo>> bLinkMap_;
    std::vector<u64> remoteMaxSendCountsWithoutSelf_;
    std::vector<bool> remoteCountInfoValid_;

    std::vector<ThreadHandle> threads_;
    ThreadHandle mainThread_{0};
    std::vector<ThreadHandle> stage0Threads_;
    std::vector<ThreadHandle> stage1Threads_;
    std::vector<ThreadHandle> bThreads_;
    TemplateResMeta stage0Meta_;
    TemplateResMeta stage1Meta_;
    TemplateResMeta bMeta_;
};

} // namespace ops_hccl

#endif // HCCLV2_INS_V2_ALLTOALLV_PARALLEL_V2_AB_4P_NO_MEMCPY_EXECUTOR_H
