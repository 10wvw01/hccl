/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include <gtest/gtest.h>
#include "reduce_scatter_birs_selector.h"

using ops_hccl::TopoInfo;
using ops_hccl_experimental::BirsSelectResult;
using ops_hccl_experimental::DecideReduceScatterBirsAlg;

namespace {
TopoInfo MakeTopo(u32 userRankSize, u32 serverNum, DevType deviceType)
{
    TopoInfo topo{};
    topo.userRankSize = userRankSize;
    topo.serverNum = serverNum;
    topo.deviceType = deviceType;
    return topo;
}

const std::string BIRS_ALG_NAME = "ReduceScatterBIRSExecutor";
}

TEST(ReduceScatterBirsSelector, ServerNumZeroIsRejectedInsteadOfDivByZero)
{
    std::string algName;
    TopoInfo topo = MakeTopo(8, 0, DevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterBirsAlg(topo, algName), BirsSelectResult::kRejectServerNumZero);
    EXPECT_NE(algName, BIRS_ALG_NAME);
}

TEST(ReduceScatterBirsSelector, DualCardSingleServerIsRejected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(2, 1, DevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterBirsAlg(topo, algName), BirsSelectResult::kRejectRanksPerServerLT4);
    EXPECT_NE(algName, BIRS_ALG_NAME);
}

TEST(ReduceScatterBirsSelector, ThreeRanksPerServerIsRejected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(6, 2, DevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterBirsAlg(topo, algName), BirsSelectResult::kRejectRanksPerServerLT4);
    EXPECT_NE(algName, BIRS_ALG_NAME);
}

TEST(ReduceScatterBirsSelector, RankSizeOneIsRejected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(1, 1, DevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterBirsAlg(topo, algName), BirsSelectResult::kRejectRankSizeOne);
    EXPECT_NE(algName, BIRS_ALG_NAME);
}

TEST(ReduceScatterBirsSelector, FourRanksPerServerIsSelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(8, 2, DevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterBirsAlg(topo, algName), BirsSelectResult::kSelected);
    EXPECT_EQ(algName, BIRS_ALG_NAME);
}

TEST(ReduceScatterBirsSelector, ExactlyFourRanksPerServerBoundarySelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(4, 1, DevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterBirsAlg(topo, algName), BirsSelectResult::kSelected);
    EXPECT_EQ(algName, BIRS_ALG_NAME);
}

TEST(ReduceScatterBirsSelector, NonA3DeviceIsNotSelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(8, 2, DevType::DEV_TYPE_910B);
    EXPECT_EQ(DecideReduceScatterBirsAlg(topo, algName), BirsSelectResult::kNotSelected);
    EXPECT_NE(algName, BIRS_ALG_NAME);
}

TEST(ReduceScatterBirsSelector, OddRankSizeIsNotSelected)
{
    std::string algName;
    TopoInfo topo = MakeTopo(9, 1, DevType::DEV_TYPE_910_93);
    EXPECT_EQ(DecideReduceScatterBirsAlg(topo, algName), BirsSelectResult::kNotSelected);
    EXPECT_NE(algName, BIRS_ALG_NAME);
}
