/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_GATHER_SEQUENCE_EXECUTOR_4LEVEL_OCS_H
#define HCCLV2_INS_V2_ALL_GATHER_SEQUENCE_EXECUTOR_4LEVEL_OCS_H

#include "executor_common_ops.h"
#include "topo_match_base.h"
#include "topo_match_multilevel.h"

namespace ops_hccl {
constexpr u32 SEQUENCE_EXECUTOR_4_LEVEL_NUM_OCS = 4;

// 4层(net_layer 0/1/2/3, SuperNode 级 OCS) AllGather sequence executor。
// 执行序: 外层(level3)先、内层(level0)后，buffer 链 INPUT -> HCCL -> HCCL -> HCCL -> OUTPUT。
// 数据随层级向内递增(repeatNum 增大)，地址用 rankSize 乘积表达，不依赖 rankIdx。
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1,
    typename InsAlgTemplate2, typename InsAlgTemplate3>
class InsV2AllGatherSequenceExecutor4LevelOCS : public InsCollAlgBase {
public:
    explicit InsV2AllGatherSequenceExecutor4LevelOCS();
    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    /* *************** 资源计算 *************** */
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
                       AlgResourceRequest &resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
                                    AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

protected:
    HcclResult InitExectorInfo(const OpParam &param);
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
    void GenTemplateAlgParamsLevel3(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                                    const u64 curCount, const u64 dataOffset,
                                    TemplateDataParams &tempAlgParamsLevel3) const;
    void GenTemplateAlgParamsLevel2(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                                    const u64 curCount, const u64 dataOffset,
                                    TemplateDataParams &tempAlgParamsLevel2) const;
    void GenTemplateAlgParamsLevel1(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                                    const u64 curCount, const u64 dataOffset,
                                    TemplateDataParams &tempAlgParamsLevel1) const;
    void GenTemplateAlgParamsLevel0(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                                    const u64 curCount, const u64 dataOffset,
                                    TemplateDataParams &tempAlgParamsLevel0) const;
    template <typename InsAlgTemplate>
    HcclResult GenTempResource(const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<InsAlgTemplate> &algTemplate, TemplateResource &tempResource) const;

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};
    uint64_t rankSizeLevel2_{0};
    uint64_t rankSizeLevel3_{0};
    bool skipLevel1_{false};
    bool skipLevel2_{false};
    bool skipLevel3_{false};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<ThreadHandle> threads_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
};

}  // namespace ops_hccl

#endif  // HCCLV2_INS_V2_ALL_GATHER_SEQUENCE_EXECUTOR_4LEVEL_OCS_H
