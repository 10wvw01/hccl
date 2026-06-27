/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_TO_ALL_V_AB_RELAY_NO_MEMCPY_H
#define INS_TEMP_ALL_TO_ALL_V_AB_RELAY_NO_MEMCPY_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

enum class A2AVABRelayPhase {
    PREROUTE_TO_RELAY = 0,
    RELAY_TO_OUTPUT = 1,
};

class InsTempAlltoAllVABRelayNoMemcpy : public InsAlgTemplateBase {
public:
    InsTempAlltoAllVABRelayNoMemcpy() = default;
    explicit InsTempAlltoAllVABRelayNoMemcpy(const OpParam &param, u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAlltoAllVABRelayNoMemcpy() override = default;

    std::string Describe() const override
    {
        return "Template of alltoallv AB relay no-memcpy";
    }

    void SetRelayInfo(A2AVABRelayPhase phase, u32 rankSize, u32 meshSize, u64 slotStride)
    {
        phase_ = phase;
        rankSize_ = rankSize;
        meshSize_ = meshSize;
        groupNum_ = meshSize == 0 ? 0 : rankSize / meshSize;
        slotStride_ = slotStride;
    }

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;

private:
    HcclResult RunPrerouteToRelay(const TemplateDataParams &params, const TemplateResource &resource);
    HcclResult RunRelayToOutput(const TemplateDataParams &params, const TemplateResource &resource);
    HcclResult RunPeerSendRecv(const ChannelInfo &channel, const std::vector<DataSlice> &txSrcSlices,
                               const std::vector<DataSlice> &txDstSlices, const ThreadHandle &thread) const;
    HcclResult BuildPrerouteSlices(u32 peerRank, const ChannelInfo &channel, const TemplateDataParams &params,
                                   std::vector<DataSlice> &txSrcSlices,
                                   std::vector<DataSlice> &txDstSlices) const;
    HcclResult BuildRelaySlices(u32 peerRank, const std::vector<ChannelInfo> &channels, u32 channelIdx,
                                const TemplateDataParams &params, std::vector<DataSlice> &txSrcSlices,
                                std::vector<DataSlice> &txDstSlices) const;
    HcclResult CheckCommonParams(const TemplateDataParams &params, const TemplateResource &resource) const;
    HcclResult CheckSliceRange(const char *tag, u32 peerRank, u64 srcOffset, u64 dstOffset, u64 byteSize,
                               u64 srcLimit, u64 dstLimit) const;
    u64 CalcRelaySlotOffset(u32 srcRank, u32 dstRank) const;
    void CalcChannelSplit(u64 count, const std::vector<ChannelInfo> &channels, u32 channelIdx,
                          u64 &splitCount, u64 &splitOffsetCount) const;

    A2AVABRelayPhase phase_{A2AVABRelayPhase::PREROUTE_TO_RELAY};
    u32 rankSize_{0};
    u32 meshSize_{0};
    u32 groupNum_{0};
    u64 slotStride_{0};
    u32 dataTypeSize_{0};
    HcclDataType dataType_{HCCL_DATA_TYPE_RESERVED};
};

} // namespace ops_hccl

#endif // INS_TEMP_ALL_TO_ALL_V_AB_RELAY_NO_MEMCPY_H
