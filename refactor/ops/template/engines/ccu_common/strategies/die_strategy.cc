/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "die_strategy.h"

#include "ccu_alg_utils.h"
#include "log.h"

namespace ops_hccl {

HcclResult SingleDieStrategy::Decide(HcclComm comm, u32 myRank,
                                     const std::vector<HcclChannelDesc> &channelDescs,
                                     u64 sliceSize, u32 dataTypeSize, u32 templateRankSize,
                                     DieDecision &decision)
{
    (void)comm;
    (void)myRank;
    (void)sliceSize;
    (void)dataTypeSize;
    (void)templateRankSize;

    decision.dieNum = 1;
    decision.myDieId = 0;
    decision.ifHandleSelfRank = true;
    decision.axisId = 0;
    decision.die0Size = sliceSize;
    decision.die1Size = 0;
    decision.die0LastSize = 0;
    decision.die1LastSize = 0;

    decision.channelsPerDie.clear();
    decision.rankIdGroupPerDie.clear();
    decision.channelsPerDie.push_back(channelDescs);
    std::vector<u32> rankGroup;
    rankGroup.reserve(channelDescs.size());
    for (const auto &ch : channelDescs) {
        rankGroup.push_back(ch.remoteRank);
    }
    decision.rankIdGroupPerDie.push_back(std::move(rankGroup));
    return HCCL_SUCCESS;
}

HcclResult TwoDiePeerSetStrategy::Decide(HcclComm comm, u32 myRank,
                                         const std::vector<HcclChannelDesc> &channelDescs,
                                         u64 sliceSize, u32 dataTypeSize, u32 templateRankSize,
                                         DieDecision &decision)
{
    (void)sliceSize;
    (void)dataTypeSize;
    (void)templateRankSize;

    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc;
    CHK_RET(ccu_alg_utils::RestoreChannelMap(channelDescs, rankIdToChannelDesc));

    u32 dieNum = 1;
    u32 myDieId = 0;
    CHK_RET(ccu_alg_utils::GetDieInfoFromChannelDescs(comm, rankIdToChannelDesc, myRank, dieNum, myDieId));
    decision.dieNum = dieNum;
    decision.myDieId = myDieId;

    if (dieNum == 1) {
        // 退化为单 Die：所有 channel 归 die0
        decision.ifHandleSelfRank = true;
        decision.channelsPerDie.clear();
        decision.rankIdGroupPerDie.clear();
        decision.channelsPerDie.push_back(channelDescs);
        std::vector<u32> rankGroup;
        rankGroup.reserve(channelDescs.size());
        for (const auto &ch : channelDescs) {
            rankGroup.push_back(ch.remoteRank);
        }
        decision.rankIdGroupPerDie.push_back(std::move(rankGroup));
        return HCCL_SUCCESS;
    }

    // 2 Die：按 dieId 0/1 分组 channel（迁移自 ClassifyChannelByDieId）
    decision.channelsPerDie.assign(2, {});
    decision.rankIdGroupPerDie.assign(2, {});
    for (const auto &ch : channelDescs) {
        u32 tmpDieId = 0;
        CHK_RET(ccu_alg_utils::GetChannelDieId(comm, myRank, ch, tmpDieId));
        if (tmpDieId == 0) {
            decision.channelsPerDie[0].push_back(ch);
            decision.rankIdGroupPerDie[0].push_back(ch.remoteRank);
        } else {
            decision.channelsPerDie[1].push_back(ch);
            decision.rankIdGroupPerDie[1].push_back(ch.remoteRank);
        }
    }

    // peer 数量大的 die 处理 self rank 的本地拷贝（迁移自 ccu_temp_all_gather_2dies_mesh_1D.cc:58-60）
    const auto &group0 = decision.rankIdGroupPerDie[0];
    const auto &group1 = decision.rankIdGroupPerDie[1];
    if ((group0.size() > group1.size() && group1.size() != 0) || group0.size() == 0) {
        decision.ifHandleSelfRank = false;
    } else {
        decision.ifHandleSelfRank = true;
    }

    // 端口数不一致时调整 die 顺序（端口数多的放前面）
    CHK_RET(ccu_alg_utils::ReverseChannelPerDieIfNeed(comm, myRank, decision.channelsPerDie));
    if (decision.channelsPerDie.size() == 2) {
        std::swap(decision.rankIdGroupPerDie[0], decision.rankIdGroupPerDie[1]);
    }
    return HCCL_SUCCESS;
}

HcclResult TwoDieDataHalvingStrategy::Decide(HcclComm comm, u32 myRank,
                                             const std::vector<HcclChannelDesc> &channelDescs,
                                             u64 sliceSize, u32 dataTypeSize, u32 templateRankSize,
                                             DieDecision &decision)
{
    (void)comm;

    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc;
    CHK_RET(ccu_alg_utils::RestoreChannelMap(channelDescs, rankIdToChannelDesc));

    u32 dieNum = 1;
    u32 myDieId = 0;
    CHK_RET(ccu_alg_utils::GetDieInfoFromChannelDescs(comm, rankIdToChannelDesc, myRank, dieNum, myDieId));
    decision.dieNum = dieNum;
    decision.myDieId = myDieId;
    decision.axisId = myDieId;  // axisId 由本卡 dieId 决定
    decision.ifHandleSelfRank = true;

    // 所有 channel 归 die0（单 kernel 通过 axisId 区分处理哪一半数据）
    decision.channelsPerDie.clear();
    decision.rankIdGroupPerDie.clear();
    decision.channelsPerDie.push_back(channelDescs);
    std::vector<u32> rankGroup;
    rankGroup.reserve(channelDescs.size());
    for (const auto &ch : channelDescs) {
        rankGroup.push_back(ch.remoteRank);
    }
    decision.rankIdGroupPerDie.push_back(std::move(rankGroup));

    // SplitDataFor2Dies：按 portGroupSize 6:2 比例切分（迁移自 ccu_temp_all_gather_nhr_1D_mem2mem.cc:164-184）
    constexpr u64 MULTIPLIER = 4;
    constexpr u8 DIE0_PORT_GROUP_SIZE = 6;
    constexpr u8 DIE1_PORT_GROUP_SIZE = 2;
    const u64 dataCount = (dataTypeSize == 0) ? 0 : (sliceSize / dataTypeSize);

    if (dieNum == 1 || dataCount <= static_cast<u64>(templateRankSize) * MULTIPLIER) {
        // 数据量极小或单 die，不划分
        decision.die0Size = sliceSize;
        decision.die1Size = 0;
        decision.die0LastSize = 0;
        decision.die1LastSize = 0;
    } else {
        decision.die0Size = (dataCount * DIE0_PORT_GROUP_SIZE /
                             (DIE0_PORT_GROUP_SIZE + DIE1_PORT_GROUP_SIZE)) * dataTypeSize;
        decision.die1Size = sliceSize - decision.die0Size;
        // 尾块按 dieNum 均分（迁移自 PrepareLaunchArgs 中的 die0LastSize 计算）
        decision.die0LastSize = 0;  // 由 baseTemplate 在 KernelRun 中按 tailSize 填充
        decision.die1LastSize = 0;
    }
    HCCL_DEBUG("[TwoDieDataHalvingStrategy::Decide] dieNum=%u axisId=%u die0Size=%llu die1Size=%llu",
               decision.dieNum, decision.axisId, decision.die0Size, decision.die1Size);
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
