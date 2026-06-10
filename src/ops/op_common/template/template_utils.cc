/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "template_utils.h"

#include <limits>
namespace ops_hccl {

HcclResult GetAlgRank(const u32 virtRank, const std::vector<u32> &rankIds, u32 &algRank)
{
    std::vector<u32>::const_iterator topoVecIter = std::find(rankIds.begin(), rankIds.end(), virtRank);
    CHK_PRT_RET(topoVecIter == rankIds.end(), HCCL_ERROR("[GetAlgRank] Invalid virtual Rank!"),
                HcclResult::HCCL_E_PARA);
    algRank = distance(rankIds.begin(), topoVecIter);

    return HcclResult::HCCL_SUCCESS;
}

u32 GetNHRStepNum(u32 rankSize)
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[NHRBase][GetStepNumInterServer] rankSize[%u] nSteps[%u]", rankSize, nSteps);

    return nSteps;
}

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
    u32 taskCount =  (static_cast<int>(channels.size()) > channelsPerRank) ? channelsPerRank : static_cast<int>(channels.size());
    for (u32 i = 0; i < taskCount; i++) {
        const auto &ch = channels[i];
        portGroups.push_back(ch.portGroupSize);
        totalPorts += ch.portGroupSize;
        HCCL_INFO("[CalcDataSplitByPortGroup] ch.portGroupSize[%u], totalPorts[%u], channelsPerRank[%u]",
                    ch.portGroupSize, totalPorts, channelsPerRank);
    }

    u32 channelsize = portGroups.size();
    u64 accumCount = 0;
    u64 offset = 0;
    for (u32 channelIdx = 0; channelIdx < channelsize; channelIdx++) {
        u64 elemCount = 0;
        u64 elemSize = 0;
        if (channelIdx == channelsize - 1) {
            elemCount = totalDataCount - accumCount;
        } else {
            CHK_PRT_RET(totalPorts == 0,
                        HCCL_ERROR("[CalcDataSplitByPortGroup] totalPorts [%u] is 0.", totalPorts),
                        HcclResult::HCCL_E_INTERNAL);
            elemCount = static_cast<u64>((totalDataCount * portGroups[channelIdx]) / totalPorts);
        }
        elemOffset.push_back(offset);
        elemCountOut.push_back(elemCount);
        elemSize = elemCount * dataTypeSize;
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
        HCCL_ERROR("[CalcDataSplitByPortGroupZAxisDetour] level0DataRatio[%f] is invalid.", level0DataRatio),
        HcclResult::HCCL_E_PARA);

    u64 level0DataCount;
    if (level1ChannelNumPerRank == 0) {
        level0DataCount = totalDataCount;
    } else {
        level0DataCount = static_cast<u64>(static_cast<double>(totalDataCount) * level0DataRatio);
        level0DataCount = std::min(level0DataCount, totalDataCount);
    }
    u64 level1DataCount = totalDataCount - level0DataCount;

    std::vector<ChannelInfo> level0Chs(channels.begin(),
        channels.begin() + level0ChannelNumPerRank);
    std::vector<u64> l0ElemCount, l0Size, l0Offset;
    CHK_RET(CalcDataSplitByPortGroupCommon(level0DataCount, dataTypeSize,
        level0Chs, l0ElemCount, l0Size, l0Offset, level0ChannelNumPerRank));

    std::vector<ChannelInfo> level1Chs(channels.begin() + level0ChannelNumPerRank,
        channels.end());
    std::vector<u64> l1ElemCount, l1Size, l1Offset;

    CHK_RET(CalcDataSplitByPortGroupCommon(level1DataCount, dataTypeSize,
        level1Chs, l1ElemCount, l1Size, l1Offset, level1ChannelNumPerRank));
    u64 level0TotalSize = 0;
    for (auto sz : l0Size) {
        level0TotalSize += sz;
    }
    for (auto &off : l1Offset) {
        off += level0TotalSize;
    }

    elemCountOut = l0ElemCount;
    elemCountOut.insert(elemCountOut.end(), l1ElemCount.begin(), l1ElemCount.end());
    sizeOut = l0Size;
    sizeOut.insert(sizeOut.end(), l1Size.begin(), l1Size.end());
    elemOffset = l0Offset;
    elemOffset.insert(elemOffset.end(), l1Offset.begin(), l1Offset.end());

    HCCL_INFO("[CalcDataSplitByPortGroupZAxisDetour] totalDataCount[%llu], level0DataCount[%llu], "
              "level1DataCount[%llu], level0ChannelNumPerRank[%u], level1ChannelNumPerRank[%u], "
              "level0DataRatio[%f], elemCountOut.size[%zu]",
              totalDataCount, level0DataCount, level1DataCount,
              level0ChannelNumPerRank, level1ChannelNumPerRank,
              level0DataRatio, elemCountOut.size());

    return HcclResult::HCCL_SUCCESS;
}

bool GetPortGroupSize(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    uint64_t &portGroupSize)
{
    portGroupSize = 0;
    for (const auto &entry : channels) {
        const auto &channelGroup = entry.second;
        if (!channelGroup.empty()) {
            for (const auto &ch : channelGroup) {
                portGroupSize += ch.portGroupSize;
            }
            return true;
        }
    }
    return false;
}

static bool IsPodInterChannelGroup(const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    constexpr size_t podChannelNum = 2;
    for (const auto &entry : channels) {
        const auto &channelGroup = entry.second;
        if (channelGroup.empty()) {
            continue;
        }
        if (channelGroup.size() != podChannelNum) {
            return false;
        }
        const u32 firstDieId = channelGroup[0].dieId;
        const u32 secondDieId = channelGroup[1].dieId;
        return firstDieId != INVALID_VALUE_RANKID && secondDieId != INVALID_VALUE_RANKID &&
            firstDieId != secondDieId;
    }
    return false;
}

const char* ParallelDataSplitTypeToStr(ParallelDataSplitType splitType)
{
    switch (splitType) {
        case ParallelDataSplitType::REDUCE_SCATTER_WITH_LOCAL_REDUCE:
            return "REDUCE_SCATTER_WITH_LOCAL_REDUCE";
        case ParallelDataSplitType::SCATTER:
            return "SCATTER";
        case ParallelDataSplitType::ALL_GATHER:
            return "ALL_GATHER";
        default:
            return "UNKNOWN";
    }
}

double CalcParallelDataSplitRatio(
    uint64_t intraRankSize,
    uint64_t interRankSize,
    const std::map<u32, std::vector<ChannelInfo>> &intraChannels,
    const std::map<u32, std::vector<ChannelInfo>> &interChannels,
    ParallelDataSplitType splitType,
    double fallbackRatio)
{
    const double validFallback = std::isfinite(fallbackRatio)
        ? std::max(0.0, std::min(fallbackRatio, 1.0))
        : 0.5;
    bool needFallback = false;
    std::string fallbackReason;

    if (intraRankSize == 0) {
        needFallback = true;
        fallbackReason = "intraRankSize is 0";
    } else if (interRankSize == 0) {
        needFallback = true;
        fallbackReason = "interRankSize is 0";
    } else if (intraChannels.empty()) {
        needFallback = true;
        fallbackReason = "intraChannels is empty";
    } else if (interChannels.empty()) {
        needFallback = true;
        fallbackReason = "interChannels is empty";
    }

    uint64_t intraPortGroupSize = 0;
    uint64_t interPortGroupSize = 0;
    double effectiveInterPortGroupSize = 0.0;
    bool isPod = false;
    if (!needFallback) {
        if (!GetPortGroupSize(intraChannels, intraPortGroupSize)) {
            needFallback = true;
            fallbackReason = "no non-empty channel group in intraChannels";
        } else if (!GetPortGroupSize(interChannels, interPortGroupSize)) {
            needFallback = true;
            fallbackReason = "no non-empty channel group in interChannels";
        } else if (intraPortGroupSize == 0) {
            needFallback = true;
            fallbackReason = "intraPortGroupSize is 0";
        } else if (interPortGroupSize == 0) {
            needFallback = true;
            fallbackReason = "interPortGroupSize is 0";
        } else if (intraRankSize - 1 > std::numeric_limits<uint64_t>::max() / intraPortGroupSize) {
            needFallback = true;
            fallbackReason = "intraPortGroupSize scaling overflow";
        } else {
            intraPortGroupSize *= intraRankSize - 1;
            isPod = IsPodInterChannelGroup(interChannels);
            effectiveInterPortGroupSize = static_cast<double>(interPortGroupSize) / (isPod ? 2.0 : 1.0);
            if (intraPortGroupSize == 0) {
                needFallback = true;
                fallbackReason = "scaled intraPortGroupSize is 0";
            } else if (effectiveInterPortGroupSize == 0.0 || !std::isfinite(effectiveInterPortGroupSize)) {
                needFallback = true;
                fallbackReason = "effectiveInterPortGroupSize is 0 or not finite";
            }
        }
    }

    if (needFallback) {
        HCCL_WARNING("[CalcParallelDataSplitRatio] fallback due to: %s, "
                     "intraRankSize[%llu], interRankSize[%llu], "
                     "intraPortGroupSize[%llu], interPortGroupSize[%llu], "
                     "splitType[%s], fallbackRatio[%f]",
                     fallbackReason.c_str(),
                     intraRankSize, interRankSize,
                     intraPortGroupSize, interPortGroupSize,
                     ParallelDataSplitTypeToStr(splitType), validFallback);
        return validFallback;
    }

    double meshTimeCoeff = 0.0;
    double closTimeCoeff = 0.0;

    switch (splitType) {
        case ParallelDataSplitType::REDUCE_SCATTER_WITH_LOCAL_REDUCE:
            meshTimeCoeff = 21.0 * static_cast<double>(intraRankSize - 1) /
                (20.0 * static_cast<double>(intraRankSize) * intraPortGroupSize);
            closTimeCoeff = static_cast<double>(interRankSize - 1) /
                (static_cast<double>(interRankSize) * effectiveInterPortGroupSize);
            break;
        case ParallelDataSplitType::SCATTER:
            meshTimeCoeff = static_cast<double>(intraRankSize - 1) /
                (static_cast<double>(intraRankSize) * intraPortGroupSize);
            closTimeCoeff = static_cast<double>(interRankSize - 1) /
                (static_cast<double>(interRankSize) * effectiveInterPortGroupSize);
            break;
        case ParallelDataSplitType::ALL_GATHER:
            meshTimeCoeff = static_cast<double>(intraRankSize - 1) / intraPortGroupSize;
            closTimeCoeff = static_cast<double>(interRankSize - 1) / effectiveInterPortGroupSize;
            break;
        default:
            HCCL_WARNING("[CalcParallelDataSplitRatio] fallback due to: unknown splitType[%d], "
                         "fallbackRatio[%f]", static_cast<int>(splitType), validFallback);
            return validFallback;
    }

    double denominator = closTimeCoeff + meshTimeCoeff;
    if (denominator == 0.0 || !std::isfinite(denominator)) {
        HCCL_WARNING("[CalcParallelDataSplitRatio] fallback due to: denominator is 0 or not finite, "
                     "intraRankSize[%llu], interRankSize[%llu], "
                     "intraPortGroupSize[%llu], interPortGroupSize[%llu], "
                     "splitType[%s], fallbackRatio[%f]",
                     intraRankSize, interRankSize,
                     intraPortGroupSize, interPortGroupSize,
                     ParallelDataSplitTypeToStr(splitType), validFallback);
        return validFallback;
    }

    double ratio = closTimeCoeff / denominator;
    if (!std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0) {
        HCCL_WARNING("[CalcParallelDataSplitRatio] fallback due to: ratio[%f] is not finite or out of range[0,1], "
                     "intraRankSize[%llu], interRankSize[%llu], "
                     "intraPortGroupSize[%llu], interPortGroupSize[%llu], "
                     "splitType[%s], fallbackRatio[%f]",
                     ratio,
                     intraRankSize, interRankSize,
                     intraPortGroupSize, interPortGroupSize,
                     ParallelDataSplitTypeToStr(splitType), validFallback);
        return validFallback;
    }

    constexpr double ratioStep = 1.0 / 8.0;
    constexpr double minRatioIndex = 1.0;
    constexpr double maxRatioIndex = 7.0;
    const double nearestRatioIndex = std::round(ratio / ratioStep);
    const double clampedRatioIndex = std::max(minRatioIndex, std::min(nearestRatioIndex, maxRatioIndex));
    const double quantizedRatio = clampedRatioIndex * ratioStep;
    HCCL_INFO("[CalcParallelDataSplitRatio] intraRankSize[%llu], interRankSize[%llu], "
              "intraPortGroupSize[%llu], interPortGroupSize[%llu], effectiveInterPortGroupSize[%f], isPod[%d], "
              "splitType[%s], rawRatio[%f], quantizedRatio[%f]",
              intraRankSize, interRankSize,
              intraPortGroupSize, interPortGroupSize, effectiveInterPortGroupSize, isPod,
              ParallelDataSplitTypeToStr(splitType), ratio, quantizedRatio);
    return quantizedRatio;
}
}
