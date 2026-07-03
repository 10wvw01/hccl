/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_OMNI_RUN_TEMPLATE_AICPU_OMNI_TEMP_AICPU_H
#define OPS_HCCL_OMNI_RUN_TEMPLATE_AICPU_OMNI_TEMP_AICPU_H

#include "alg_param.h"
#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"
#include "utils.h"
#include "omni_parser.h"

namespace ops_hccl {

class OmniTempAicpu : public InsAlgTemplateBase {
public:
    explicit OmniTempAicpu(
        const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks);

    ~OmniTempAicpu() override;

    std::string Describe() const override
    {
        std::string info = "Template of omni_run OMNI AICPU with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    // 现在的RunAsync就是之前的GenExtIns
    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
        TemplateResource &templateResource, const omni::XmlInfo &xmlInfo);
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        AlgResourceRequest &resourceRequest, const omni::XmlInfo &xmlInfo);
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    HcclResult CalcChannelRequestOmni(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const std::vector<std::vector<u32>> &subcommInfo, std::vector<HcclChannelDesc> &channels);
    HcclResult RunOmni(const std::map<u32, std::vector<ChannelInfo>> &channels,
        const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams,
        const omni::XmlInfo &xmlInfo);

    // 操作类型处理函数
    HcclResult HandlePreSyncInterThreads(
        const omni::OmniNormalInstruction &signalInfo, const std::vector<ThreadHandle> &threads);
    HcclResult HandlePostSyncInterThreads(
        const omni::OmniNormalInstruction &signalInfo, const std::vector<ThreadHandle> &threads);
    HcclResult HandleLocalCopy(const omni::OmniNormalInstruction &signalInfo, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendRecvWrite(const omni::OmniNormalInstruction &signalInfo,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendWrite(const omni::OmniNormalInstruction &signalInfo,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult HandleRecvWrite(const omni::OmniNormalInstruction &signalInfo,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendRecvRead(const omni::OmniNormalInstruction &signalInfo,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendRead(const omni::OmniNormalInstruction &signalInfo,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult HandleRecvRead(const omni::OmniNormalInstruction &signalInfo,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult HandleGroupBroadcast(const omni::OmniNormalInstruction &signalInfo,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult HandleGroupReduce(const omni::OmniNormalInstruction &signalInfo,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendRecvWriteDPU(const omni::OmniNormalInstruction &signalInfo,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendWriteDPU(const omni::OmniNormalInstruction &signalInfo,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult HandleRecvWriteDPU(const omni::OmniNormalInstruction &signalInfo,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);

    // XML配置相关
    u32 roundRobinIndex_;
};

} // namespace ops_hccl

#endif // OPS_HCCL_OMNI_RUN_TEMPLATE_AICPU_OMNI_TEMP_AICPU_H
