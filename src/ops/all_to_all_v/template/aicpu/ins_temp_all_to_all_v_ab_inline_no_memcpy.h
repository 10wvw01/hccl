/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_TO_ALL_V_AB_INLINE_NO_MEMCPY_H
#define INS_TEMP_ALL_TO_ALL_V_AB_INLINE_NO_MEMCPY_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempAlltoAllVABInlineNoMemcpy : public InsAlgTemplateBase {
public:
    InsTempAlltoAllVABInlineNoMemcpy() = default;
    explicit InsTempAlltoAllVABInlineNoMemcpy(const OpParam &param, u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAlltoAllVABInlineNoMemcpy() override = default;

    std::string Describe() const override
    {
        return "Template of alltoallv AB inline no-memcpy";
    }

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    HcclResult RunBClos(const TemplateDataParams &params, const TemplateResource &resource);
    HcclResult RunPeer(u32 peerRank, const ChannelInfo &channel, const TemplateDataParams &params,
                       const ThreadHandle &thread) const;
    HcclResult CheckParams(const TemplateDataParams &params, const TemplateResource &resource) const;
    HcclResult CheckSliceRange(u32 peerRank, u64 srcOffset, u64 dstOffset, u64 byteSize,
                               u64 inputSize, u64 remoteOutputSize) const;

    u32 rankSize_{0};
    u32 dataTypeSize_{0};
    HcclDataType dataType_{HCCL_DATA_TYPE_RESERVED};
};

} // namespace ops_hccl

#endif // INS_TEMP_ALL_TO_ALL_V_AB_INLINE_NO_MEMCPY_H
