/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_DIE_STRATEGY_H
#define OPS_HCCL_DIE_STRATEGY_H

#include <memory>
#include <string>
#include <vector>

#include "hccl/base.h"
#include "alg_param.h"

namespace ops_hccl {

/**
 * Die 决策结果
 * 不同策略填充不同字段：
 * - SingleDieStrategy: dieNum=1, myDieId=0, ifHandleSelfRank=true
 * - TwoDiePeerSetStrategy: dieNum=2, 按 dieId 分组 channel, ifHandleSelfRank 由 peer 数量大的 die 处理
 * - TwoDieDataHalvingStrategy: dieNum=1|2, axisId 区分, die0Size/die1Size 按端口比切分
 */
struct DieDecision {
    u32 dieNum{1};
    u32 myDieId{0};

    // 策略 A (TwoDiePeerSetStrategy) 字段
    bool ifHandleSelfRank{true};

    // 策略 B (TwoDieDataHalvingStrategy) 字段
    u32 axisId{0};
    u64 die0Size{0};
    u64 die1Size{0};
    u64 die0LastSize{0};
    u64 die1LastSize{0};

    // 通用：按 die 分组的 channel 与 peer rank
    std::vector<std::vector<HcclChannelDesc>> channelsPerDie;
    std::vector<std::vector<u32>> rankIdGroupPerDie;
};

/**
 * Die 划分策略接口
 * baseTemplate 持有 IDieStrategy 实例，运行时按拓扑注入具体策略。
 */
class IDieStrategy {
public:
    virtual ~IDieStrategy() = default;

    /**
     * 根据 channelDescs 与 sliceSize 决策 Die 划分。
     * 输入参数：
     *   - comm: 通信域句柄
     *   - myRank: 本卡 rankId
     *   - channelDescs: 所有 channel 描述符列表
     *   - sliceSize: 当前数据切片字节大小（策略 B 用于数据半分）
     *   - dataTypeSize: 数据类型字节大小（策略 B 用于 dataCount 计算）
     *   - templateRankSize: 模板通信域 rank 总数（策略 B 用于判断数据量是否过小）
     * 输出参数：
     *   - decision: Die 决策结果
     */
    virtual HcclResult Decide(HcclComm comm, u32 myRank,
                              const std::vector<HcclChannelDesc> &channelDescs,
                              u64 sliceSize, u32 dataTypeSize, u32 templateRankSize,
                              DieDecision &decision) = 0;

    virtual std::string Name() const = 0;
};

/** 单 Die 策略：所有 channel 归 die0，ifHandleSelfRank=true。 */
class SingleDieStrategy : public IDieStrategy {
public:
    HcclResult Decide(HcclComm comm, u32 myRank,
                      const std::vector<HcclChannelDesc> &channelDescs,
                      u64 sliceSize, u32 dataTypeSize, u32 templateRankSize,
                      DieDecision &decision) override;
    std::string Name() const override { return "SingleDieStrategy"; }
};

/**
 * 两 Die 按 peer 集分片策略（用于 mesh 拓扑）。
 * 调用 GetDieInfoFromChannelDescs 决策 dieNum，按 dieId 分组 channel。
 * ifHandleSelfRank 由 peer 数量大的 die 处理（迁移自 ccu_temp_all_gather_2dies_mesh_1D.cc:37-62）。
 */
class TwoDiePeerSetStrategy : public IDieStrategy {
public:
    HcclResult Decide(HcclComm comm, u32 myRank,
                      const std::vector<HcclChannelDesc> &channelDescs,
                      u64 sliceSize, u32 dataTypeSize, u32 templateRankSize,
                      DieDecision &decision) override;
    std::string Name() const override { return "TwoDiePeerSetStrategy"; }
};

/**
 * 两 Die 按数据半分策略（用于 nhr 拓扑）。
 * 单 kernel（axisId 区分），按 portGroupSize 比例 6:2 切分数据（迁移自
 * ccu_temp_all_gather_nhr_1D_mem2mem.cc:164-184 的 SplitDataFor2Dies）。
 */
class TwoDieDataHalvingStrategy : public IDieStrategy {
public:
    HcclResult Decide(HcclComm comm, u32 myRank,
                      const std::vector<HcclChannelDesc> &channelDescs,
                      u64 sliceSize, u32 dataTypeSize, u32 templateRankSize,
                      DieDecision &decision) override;
    std::string Name() const override { return "TwoDieDataHalvingStrategy"; }
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_DIE_STRATEGY_H
