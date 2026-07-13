/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_JETTY_STRATEGY_H
#define OPS_HCCL_JETTY_STRATEGY_H

#include <string>

#include "hccl/base.h"

namespace ops_hccl {

/**
 * Jetty 决策结果
 * - SingleJettyStrategy: jettyNum=1, sliceSizePerJetty=sliceSize
 * - MultiJettyStrategy: jettyNum=N, 按 HCCL_MIN_SLICE_ALIGN 对齐切分
 */
struct JettyDecision {
    u32 jettyNum{1};
    u64 sliceSizePerJetty{0};
    u64 lastSliceSizePerJetty{0};
};

/**
 * Jetty 切分策略接口
 * 仅 CCU_MS (mem2mem) 模式支持 MultiJetty，CCU_SCHE (调度) 模式不持有此策略。
 */
class IMultiJettyStrategy {
public:
    virtual ~IMultiJettyStrategy() = default;

    /**
     * 根据 sliceSize 与 dataTypeSize 决策 jetty 切分。
     * 输入参数：
     *   - sliceSize: 当前数据切片字节大小
     *   - dataTypeSize: 数据类型字节大小
     * 输出参数：
     *   - decision: Jetty 决策结果
     */
    virtual HcclResult Decide(u64 sliceSize, u32 dataTypeSize, JettyDecision &decision) = 0;

    virtual std::string Name() const = 0;
};

/** 单 Jetty 策略：jettyNum=1，sliceSizePerJetty=sliceSize。 */
class SingleJettyStrategy : public IMultiJettyStrategy {
public:
    HcclResult Decide(u64 sliceSize, u32 dataTypeSize, JettyDecision &decision) override;
    std::string Name() const override { return "SingleJettyStrategy"; }
};

/**
 * 多 Jetty 策略：按 HCCL_MIN_SLICE_ALIGN(128) 对齐切分。
 * 迁移自 ccu_temp_all_gather_nhr_1D_multi_jetty_mem2mem.cc:174-177。
 * jettyNum 通过构造函数注入（当前硬件多为 1，预留扩展）。
 */
class MultiJettyStrategy : public IMultiJettyStrategy {
public:
    explicit MultiJettyStrategy(u32 jettyNum) : jettyNum_(jettyNum) {}

    HcclResult Decide(u64 sliceSize, u32 dataTypeSize, JettyDecision &decision) override;
    std::string Name() const override { return "MultiJettyStrategy"; }

private:
    u32 jettyNum_{1};
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_JETTY_STRATEGY_H
