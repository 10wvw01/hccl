/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
 */

#include "template_utils.h"

namespace ops_hccl {

// Debug target 只验证 Mesh AllGather planner，需要的 template_utils 依赖面很小。
// 这里保留和 src/ops/op_common/template/template_utils.cc 相同的 port group 切分公式，
// 避免为了 planner dump 去编完整主库及 hcomm dlsym 依赖。
HcclResult CalcDataSplitByPortGroupCommon(const u64 totalDataCount,
                                          const u64 dataTypeSize,
                                          const std::vector<ChannelInfo> &channels,
                                          std::vector<u64> &elemCountOut,
                                          std::vector<u64> &sizeOut,
                                          std::vector<u64> &elemOffset,
                                          const u32 channelsPerRank)
{
    elemCountOut.clear();
    sizeOut.clear();
    elemOffset.clear();

    std::vector<u32> portGroups;
    u32 totalPorts = 0;
    u32 taskCount = (static_cast<u32>(channels.size()) > channelsPerRank) ?
        channelsPerRank : static_cast<u32>(channels.size());
    for (u32 i = 0; i < taskCount; i++) {
        const auto &ch = channels[i];
        portGroups.push_back(ch.portGroupSize);
        totalPorts += ch.portGroupSize;
    }

    u32 channelSize = static_cast<u32>(portGroups.size());
    u64 accumCount = 0;
    u64 offset = 0;
    for (u32 channelIdx = 0; channelIdx < channelSize; channelIdx++) {
        u64 elemCount = 0;
        if (channelIdx == channelSize - 1) {
            elemCount = totalDataCount - accumCount;
        } else {
            CHK_PRT_RET(totalPorts == 0,
                        HCCL_ERROR("[CalcDataSplitByPortGroup] totalPorts [%u] is 0.", totalPorts),
                        HcclResult::HCCL_E_INTERNAL);
            elemCount = static_cast<u64>((totalDataCount * portGroups[channelIdx]) / totalPorts);
        }
        elemOffset.push_back(offset);
        elemCountOut.push_back(elemCount);
        u64 elemSize = elemCount * dataTypeSize;
        sizeOut.push_back(elemSize);
        offset += elemSize;
        accumCount += elemCount;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CalcDataSplitByPortGroupZAxisDetour(const u64 totalDataCount,
                                                const u64 dataTypeSize,
                                                const std::vector<ChannelInfo> &channels,
                                                std::vector<u64> &elemCountOut,
                                                std::vector<u64> &sizeOut,
                                                std::vector<u64> &elemOffset,
                                                const u32 level0ChannelNumPerRank,
                                                const u32 level1ChannelNumPerRank,
                                                const float level0DataRatio)
{
    elemCountOut.clear();
    sizeOut.clear();
    elemOffset.clear();

    CHK_PRT_RET(level0DataRatio < 0.0f || level0DataRatio > 1.0f,
                HCCL_ERROR("[CalcDataSplitByPortGroupZAxisDetour] level0DataRatio[%f] is invalid.",
                           level0DataRatio),
                HcclResult::HCCL_E_PARA);

    u64 level0DataCount;
    if (level1ChannelNumPerRank == 0) {
        level0DataCount = totalDataCount;
    } else {
        level0DataCount = static_cast<u64>(static_cast<double>(totalDataCount) * level0DataRatio);
        level0DataCount = std::min(level0DataCount, totalDataCount);
    }
    u64 level1DataCount = totalDataCount - level0DataCount;

    std::vector<ChannelInfo> level0Chs(channels.begin(), channels.begin() + level0ChannelNumPerRank);
    std::vector<u64> level0ElemCount;
    std::vector<u64> level0Size;
    std::vector<u64> level0Offset;
    CHK_RET(CalcDataSplitByPortGroupCommon(level0DataCount, dataTypeSize, level0Chs,
                                           level0ElemCount, level0Size, level0Offset,
                                           level0ChannelNumPerRank));

    std::vector<ChannelInfo> level1Chs(channels.begin() + level0ChannelNumPerRank, channels.end());
    std::vector<u64> level1ElemCount;
    std::vector<u64> level1Size;
    std::vector<u64> level1Offset;
    CHK_RET(CalcDataSplitByPortGroupCommon(level1DataCount, dataTypeSize, level1Chs,
                                           level1ElemCount, level1Size, level1Offset,
                                           level1ChannelNumPerRank));

    u64 level0TotalSize = 0;
    for (auto sz : level0Size) {
        level0TotalSize += sz;
    }
    for (auto &off : level1Offset) {
        off += level0TotalSize;
    }

    elemCountOut = level0ElemCount;
    elemCountOut.insert(elemCountOut.end(), level1ElemCount.begin(), level1ElemCount.end());
    sizeOut = level0Size;
    sizeOut.insert(sizeOut.end(), level1Size.begin(), level1Size.end());
    elemOffset = level0Offset;
    elemOffset.insert(elemOffset.end(), level1Offset.begin(), level1Offset.end());
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
