/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_2DIE_H_
#define HCCLV2_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_2DIE_H_

#include <set>
#include "utils.h"
#include "ccu_alg_template_base.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using RankId = u32;
using RankGroup = std::vector<RankId>;

class CcuTempAllToAllVMesh1D2Die : public CcuAlgTemplateBase {
public:
    CcuTempAllToAllVMesh1D2Die() = default;
    explicit CcuTempAllToAllVMesh1D2Die(const OpParam &param, RankId rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempAllToAllVMesh1D2Die() override;

    std::string Describe() const override
    {
        return StringFormat("Template of alltoallv ccu mesh 1D 2Die with rankSize[%u]", templateRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        AlgResourceRequest &resourceRequest) override;

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
        TemplateResource& templateResource) override;

    void SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo);

private:
    HcclResult PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs, uint32_t &meshDieId,
                                std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc);
    void FillRankGroupTaskArgs(uint32_t dieId, const LoopGroupConfig &config, std::vector<uint64_t> &taskArgs);
    // void FillRankGroupInfo();

    const uint32_t DIE_NUM = 2;

    std::map<uint32_t, std::vector<HcclChannelDesc>> channels_;
    std::map<uint32_t, RankGroup> rankGroup_;
    std::map<uint32_t, std::vector<HcclChannelDesc>> rankIdToChannelDesc_;
    std::set<RankId> closPeers_;
    uint32_t closMinorDieId_ = 0;
    uint32_t closMajorDieId_ = 1;
    uint32_t closBwCoeff_[2] = {0, 0};
    uint32_t totalBwCoeff_ = 0;

    A2ASendRecvInfo localSendRecvInfo_;
};

} // namespace ops_hccl
#endif // HCCLV2_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_2DIE_H_
