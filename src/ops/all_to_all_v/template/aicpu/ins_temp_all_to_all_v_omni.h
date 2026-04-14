/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_TO_ALL_V_OMNI_H
#define INS_TEMP_ALL_TO_ALL_V_OMNI_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"
#include "utils.h"

namespace ops_hccl {

class InsTempAlltoAllVOmni : public InsAlgTemplateBase {
public:
    explicit InsTempAlltoAllVOmni(const OpParam& param, const u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);

    ~InsTempAlltoAllVOmni() override;

    std::string Describe() const override
    {
        std::string info = "Template of alltoallv OMNI with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    // 现在的RunAsync就是之前的GenExtIns
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& tempAlgParams,
                         TemplateResource& templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                        AlgResourceRequest& resourceRequest, const XmlInfo& xmlInfo);
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    HcclResult RunOmni(const std::map<u32, std::vector<ChannelInfo>> &channels,
        const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams);
    void DoRepeatOmni(const std::map<u32, std::vector<ChannelInfo>> &channels,
        const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams);

    // 操作类型处理函数
    HcclResult HandleLocalCopy(const OmniSendRecvInfo& signalInfo,
                              const std::vector<ThreadHandle> &threads,
                              const TemplateDataParams &tempAlgParams);
    HcclResult HandleLocalReduce(const OmniSendRecvInfo& signalInfo,
                                const std::vector<ThreadHandle> &threads,
                                const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendRecvWrite(const OmniSendRecvInfo& signalInfo,
                                  const std::map<u32, std::vector<ChannelInfo>> &channels,
                                  const std::vector<ThreadHandle> &threads,
                                  const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendWrite(const OmniSendRecvInfo& signalInfo,
                              const std::map<u32, std::vector<ChannelInfo>> &channels,
                              const std::vector<ThreadHandle> &threads,
                              const TemplateDataParams &tempAlgParams);
    HcclResult HandleRecvWrite(const OmniSendRecvInfo& signalInfo,
                              const std::map<u32, std::vector<ChannelInfo>> &channels,
                              const std::vector<ThreadHandle> &threads,
                              const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendRecvWriteReduce(const OmniSendRecvInfo& signalInfo,
                                        const std::map<u32, std::vector<ChannelInfo>> &channels,
                                        const std::vector<ThreadHandle> &threads,
                                        const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendWriteReduce(const OmniSendRecvInfo& signalInfo,
                                    const std::map<u32, std::vector<ChannelInfo>> &channels,
                                    const std::vector<ThreadHandle> &threads,
                                    const TemplateDataParams &tempAlgParams);
    HcclResult HandleRecvWriteReduce(const OmniSendRecvInfo& signalInfo,
                                    const std::map<u32, std::vector<ChannelInfo>> &channels,
                                    const std::vector<ThreadHandle> &threads,
                                    const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendRecvRead(const OmniSendRecvInfo& signalInfo,
                                 const std::map<u32, std::vector<ChannelInfo>> &channels,
                                 const std::vector<ThreadHandle> &threads,
                                 const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendRead(const OmniSendRecvInfo& signalInfo,
                             const std::map<u32, std::vector<ChannelInfo>> &channels,
                             const std::vector<ThreadHandle> &threads,
                             const TemplateDataParams &tempAlgParams);
    HcclResult HandleRecvRead(const OmniSendRecvInfo& signalInfo,
                             const std::map<u32, std::vector<ChannelInfo>> &channels,
                             const std::vector<ThreadHandle> &threads,
                             const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendRecvReadReduce(const OmniSendRecvInfo& signalInfo,
                                       const std::map<u32, std::vector<ChannelInfo>> &channels,
                                       const std::vector<ThreadHandle> &threads,
                                       const TemplateDataParams &tempAlgParams);
    HcclResult HandleSendReadReduce(const OmniSendRecvInfo& signalInfo,
                                   const std::map<u32, std::vector<ChannelInfo>> &channels,
                                   const std::vector<ThreadHandle> &threads,
                                   const TemplateDataParams &tempAlgParams);
    HcclResult HandleRecvReadReduce(const OmniSendRecvInfo& signalInfo,
                                   const std::map<u32, std::vector<ChannelInfo>> &channels,
                                   const std::vector<ThreadHandle> &threads,
                                   const TemplateDataParams &tempAlgParams);
    HcclResult HandleGroupBroadcast(const OmniSendRecvInfo& signalInfo,
                                   const std::map<u32, std::vector<ChannelInfo>> &channels,
                                   const std::vector<ThreadHandle> &threads,
                                   const TemplateDataParams &tempAlgParams);
    HcclResult HandleGroupReduce(const OmniSendRecvInfo& signalInfo,
                                const std::map<u32, std::vector<ChannelInfo>> &channels,
                                const std::vector<ThreadHandle> &threads,
                                const TemplateDataParams &tempAlgParams);

    // XML配置相关
    XmlInfo xmlInfo_;  // 从执行器传递的XML配置信息

    // 算法参数
    u64 count_{0};
    u64 processSize_{0};
    u64 dataTypeSize_{0};
    HcclDataType dataType_{HCCL_DATA_TYPE_RESERVED};

    // 线程相关
    u32 threadNum_{0};
    std::vector<u32> notifyIdxMainToSub_;
    std::vector<u32> notifyIdxSubToMain_;
};

} // namespace ops_hccl

#endif // INS_TEMP_ALL_TO_ALL_V_OMNI_H