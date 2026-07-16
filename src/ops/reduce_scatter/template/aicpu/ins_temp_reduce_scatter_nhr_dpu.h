/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_NHR_DPU_H
#define INS_TEMP_REDUCE_SCATTER_NPU_H

#include "alg_v2_template_base.h"
#include "alg_v2_template_register.h"
#include "alg_param.h"
#include "executor_base.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempReduceScatterNhrDpu : public InsAlgTemplateBase {
public:
    explicit InsTempReduceScatterNhrDpu();
    explicit InsTempReduceScatterNhrDpu(const OpParam& param, const u32 rankId,
                                        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempReduceScatterNhrDpu() override;

    std::string Describe() const override
    {
        std::string info = "Template of ReduceScatter NHR DPU with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType) override;
    HcclResult KernelRun(const OpParam& param,
                        const TemplateDataParams& tempAlgParams,
                        TemplateResource& templateResource) override;
    HcclResult DPUKernelRun(const TemplateDataParams& tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
        const std::vector<std::vector<uint32_t>>& subCommRanks) override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub) override {}
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override {}

private:
    HcclResult GetStepInfoList(std::vector<AicpuNHRStepInfo> &stepInfoList) const;
    HcclResult GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo) const;
    HcclResult RunSingleStep(const OpParam& param,
                             const TemplateDataParams& tempAlgParams,
                             TemplateResource& templateResource,
                             u32 step);
    HcclResult LocalReduceAfterStep(const OpParam& param,
                                    const TemplateDataParams& tempAlgParams,
                                    const std::vector<ThreadHandle>& threads,
                                    u32 step,
                                    const AicpuNHRStepInfo& stepInfo);
    HcclResult LocalDataCopy(const OpParam& param,
                             const TemplateDataParams& tempAlgParams,
                             const std::vector<ThreadHandle>& threads);
    HcclResult PostLocalCopy(const OpParam& param,
                             const TemplateDataParams& tempAlgParams,
                             const std::vector<ThreadHandle>& threads);

    u64 count_{0};
    u64 processSize_{0};
    TemplateDataParams tempAlgParams_;
    std::map<u32, std::vector<ChannelInfo>> channels_;
};

} // namespace ops_hccl

#endif // INS_TEMP_REDUCE_SCATTER_NHR_DPU_H