/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_OMNI_RUN_EXECUTOR_OMNI_SOLE_CCU_EXECUTOR_H
#define OPS_HCCL_OMNI_RUN_EXECUTOR_OMNI_SOLE_CCU_EXECUTOR_H

#include "executor_common_ops.h"
#include "topo_match_1d.h"
#include "topo_match_base.h"
#include "topo_match_ubx.h"
#include "omni_parser.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate> class InsOmniSoleCcuExecutor : public InsCollAlgBase {
public:
    explicit InsOmniSoleCcuExecutor();
    ~InsOmniSoleCcuExecutor() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(HcclComm comm, const OpParam& param,
                       const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
                       AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                    AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

    #ifndef AICPU_COMPILE
    HcclResult FastLaunch(const OpParam &param, const CcuFastLaunchCtx *resCtx) override;
    HcclResult FastLaunchSaveCtx(const OpParam &param, const TemplateResource &templateAlgRes, u32 notifyNumOnMainThread);
#endif

protected:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
    HcclResult InitCommInfo(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo);

    std::vector<std::vector<std::vector<u32>>> algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;                 // 相当于之前的std::vector<InsQuePtr> tempInsQue_;
    ThreadHandle controlThread_;
    std::vector<ThreadHandle> tmpThread_;
    std::vector<u32>          notifyIdxControlToTemplates_;
    std::vector<u32>          notifyIdxTemplatesToControl_;

private:
    uint32_t syncNum_ = 0;

};
}

#endif