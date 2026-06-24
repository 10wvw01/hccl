/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */

#ifndef INS_TEMP_ALL_TO_ALL_V_V2_STAGE1_NO_MEMCPY_H
#define INS_TEMP_ALL_TO_ALL_V_V2_STAGE1_NO_MEMCPY_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

enum class A2AVV2Stage1NoMemcpyPhase {
    STAGE0_TO_RELAY = 0,
    STAGE1_TO_OUTPUT = 1,
};

class InsTempAlltoAllVV2Stage1NoMemcpy : public InsAlgTemplateBase {
public:
    InsTempAlltoAllVV2Stage1NoMemcpy() = default;
    explicit InsTempAlltoAllVV2Stage1NoMemcpy(const OpParam &param, u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAlltoAllVV2Stage1NoMemcpy() override = default;

    std::string Describe() const override
    {
        return "Template of alltoallv V2 stage1 no-memcpy";
    }

    void SetV2Stage1NoMemcpyInfo(A2AVV2Stage1NoMemcpyPhase phase, u32 rankSize, u32 meshSize, u64 slotStride)
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
    struct SplitPart {
        u32 relayRank{0};
        u64 count{0};
        u64 offsetCount{0};
    };

    HcclResult CheckParams(const TemplateDataParams &params, const TemplateResource &resource) const;
    HcclResult CheckSliceRange(const char *tag, u32 peerRank, u64 srcOffset, u64 dstOffset, u64 byteSize,
                               u64 srcLimit, u64 dstLimit) const;
    HcclResult RunPeerSendRecv(const ChannelInfo &channel, const std::vector<DataSlice> &txSrcSlices,
                               const std::vector<DataSlice> &txDstSlices, const ThreadHandle &thread) const;
    HcclResult RunStage0ToRelay(const TemplateDataParams &params, const TemplateResource &resource) const;
    HcclResult RunStage1ToOutput(const TemplateDataParams &params, const TemplateResource &resource) const;
    HcclResult CopySelfToOutput(const TemplateDataParams &params, const ThreadHandle &thread) const;
    HcclResult CopyLocalRelayToOutput(const TemplateDataParams &params, const ThreadHandle &thread) const;
    HcclResult CopyFinalDstLocalRelayToOutput(const TemplateDataParams &params, const ThreadHandle &thread) const;
    HcclResult BuildStage0Slices(u32 relayRank, const ChannelInfo &channel, const TemplateDataParams &params,
                                 std::vector<DataSlice> &txSrcSlices,
                                 std::vector<DataSlice> &txDstSlices) const;
    HcclResult BuildStage1Slices(u32 finalDst, const ChannelInfo &channel, const TemplateDataParams &params,
                                 std::vector<DataSlice> &txSrcSlices,
                                 std::vector<DataSlice> &txDstSlices) const;
    bool IsV2Peer(u32 peerRank) const;
    u32 SelectChannelIdx(u32 peerRank, const std::vector<ChannelInfo> &channels) const;
    void SplitPairCount(u64 count, u64 &part0, u64 &part1) const;
    void GetSplitParts(u32 srcRank, u32 dstRank, u64 count, std::vector<SplitPart> &parts) const;
    u64 CalcRelaySlotOffset(u32 srcRank, u32 dstRank, u32 partIdx) const;

    A2AVV2Stage1NoMemcpyPhase phase_{A2AVV2Stage1NoMemcpyPhase::STAGE0_TO_RELAY};
    u32 rankSize_{0};
    u32 meshSize_{0};
    u32 groupNum_{0};
    u64 slotStride_{0};
    u32 dataTypeSize_{0};
    HcclDataType dataType_{HCCL_DATA_TYPE_RESERVED};
};

} // namespace ops_hccl

#endif // INS_TEMP_ALL_TO_ALL_V_V2_STAGE1_NO_MEMCPY_H
