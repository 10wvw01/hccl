/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_BROADCAST_SEQUENCE_EXECUTOR_3LEVEL_H
#define HCCLV2_INS_V2_BROADCAST_SEQUENCE_EXECUTOR_3LEVEL_H

#include "alg_param.h"
#include "topo_host.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "log.h"
#include "workflow.h"
#include "sal.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "topo_match_multilevel.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
class InsV2BroadcastSequenceExecutor3Level : public InsCollAlgBase {
public:
    explicit InsV2BroadcastSequenceExecutor3Level();
    ~InsV2BroadcastSequenceExecutor3Level() = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable& resCtx) override;

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                    AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

protected:
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable& resCtx);
    HcclResult InitCommInfo(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                            const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    void GenBaseTempAlgParams(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
        TemplateDataParams &tempAlgParamsScatterL0, TemplateDataParams &tempAlgParamsScatterL1,
        TemplateDataParams &tempAlgParamsScatterL2, TemplateDataParams &tempAlgParamsAGL2,
        TemplateDataParams &tempAlgParamsAGL1, TemplateDataParams &tempAlgParamsAGL0) const;
    void GenTempAlgParamsScatterL0(const u64 loop, const u64 currDataCount, const u64 processedDataCount,
        TemplateDataParams &tempAlgParamsScatterL0) const;
    void GenTempAlgParamsScatterL1(const u64 loop, const u64 currDataCount, const u64 sliceSizeL0,
        const u64 tailSizeL0, TemplateDataParams &tempAlgParamsScatterL1) const;
    void GenTempAlgParamsScatterL2(const u64 loop, const u64 currDataCount, const u64 sliceSizeL1,
        const u64 tailSizeL1, TemplateDataParams &tempAlgParamsScatterL2) const;
    void GenTempAlgParamsAGL2(const u64 loop, const u64 currDataCount, const u64 sliceSizeL2,
        const u64 tailSizeL2, const u64 sliceSizeL1, TemplateDataParams &tempAlgParamsAGL2) const;
    void GenTempAlgParamsAGL1(const u64 loop, const u64 currDataCount, const u64 sliceSize,
        const u64 tailSize, TemplateDataParams &tempAlgParamsAGL1) const;
    void GenTempAlgParamsAGL0(const u64 loop, const u64 currDataCount, const u64 processedDataCount,
        const u64 sliceSize, const u64 tailSize, TemplateDataParams &tempAlgParamsAGL0) const;
    template <typename InsAlgTemplate>
    HcclResult GenTempResource(const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<InsAlgTemplate> &algTemplate, TemplateResource &tempReousrce) const;

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};
    uint64_t rankSizeLevel2_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};
    uint64_t rankIdxLevel2_{0};

    uint64_t rootIdx0_{0};
    uint64_t rootIdx1_{0};
    uint64_t rootIdx2_{0};

    bool skipLevel1_{false};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;
};
}

#endif
