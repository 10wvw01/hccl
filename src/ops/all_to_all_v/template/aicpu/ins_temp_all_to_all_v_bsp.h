/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_TO_ALL_V_BSP_H
#define INS_TEMP_ALL_TO_ALL_V_BSP_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempAlltoAllVBsp : public InsAlgTemplateBase {
public:
    InsTempAlltoAllVBsp() = default;
    explicit InsTempAlltoAllVBsp(const OpParam &param, const u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAlltoAllVBsp() override;

    std::string Describe() const override;

    HcclResult KernelRun(const OpParam &param,
                         const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    struct BspSlot {
        u32 txRank = 0;
        u32 rxRank = 0;
        u32 plane = 0;
        u32 deltaC = 0;
        u32 deltaR = 0;
    };

    HcclResult NormalizeSubCommRanks(const TopoInfoWithNetLayerDetails *topoInfo);
    HcclResult LocalCopyForMyRank(const TemplateDataParams &tempAlgParams, const ThreadHandle &thread) const;
    HcclResult BuildBspOffsetPlan(const TemplateDataParams &tempAlgParams, std::vector<u32> &offsetPlan) const;
    HcclResult CalcBspRoundPlan(u32 deltaC, const std::vector<u32> &offsetPlan,
                                std::vector<BspSlot> &slotPlans) const;
    HcclResult SelectBspChannel(const std::map<u32, std::vector<ChannelInfo>> &channels, u32 remoteRank,
                                u32 plane, ChannelInfo &channel) const;
    HcclResult RunBspSlot(const TemplateDataParams &tempAlgParams,
                          const std::map<u32, std::vector<ChannelInfo>> &channels,
                          const BspSlot &slot,
                          const ThreadHandle &sendThread,
                          const ThreadHandle &recvThread) const;
    HcclResult InitBspShape(u32 rowNum, u32 rankNum);
    HcclResult CheckBspShape() const;

    u32 GetRowNum() const;
    u32 GetColNum() const;
    u32 GetRankNum() const;
    u32 GetBspThreadNum() const;
    u32 SelectPlane(u32 deltaC, u32 deltaR, const std::vector<u32> &offsetPlan) const;

    u32 rowNum_{0};
    u32 colNum_{0};
    u32 rankNum_{0};
    u64 dataTypeSize_{0};
    bool isDmaRead_{false};
};

} // namespace ops_hccl

#endif // INS_TEMP_ALL_TO_ALL_V_BSP_H
