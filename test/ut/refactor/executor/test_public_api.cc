/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * OpsExecutor 对外三个函数测试：CalcAlgHierarchyInfo / CalcRes / Orchestrate
 */

#include "test_helpers.h"

namespace ops_hccl {
namespace testing {

// ============================================================
// 1. CalcAlgHierarchyInfo — 拓扑匹配 + rankSize 计算
// ============================================================

class CalcAlgHierarchyInfoTest : public OpsExecutorTest {};

TEST_F(CalcAlgHierarchyInfoTest, SingleLevelRankSize)
{
    AlgHierarchyInfoForAllLevel info;
    info.infos = {{{0, 1, 2, 3}}}; // 1 level, 4 ranks
    executor_->SetTopoMatch(info);

    HcclResult ret = executor_->CalcAlgHierarchyInfo(nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(executor_->GetRankSize(), 4u);
}

TEST_F(CalcAlgHierarchyInfoTest, TwoLevelRankSize)
{
    AlgHierarchyInfoForAllLevel info;
    info.infos = {{{0, 1, 2, 3}}, {{0, 1}}}; // 2 levels: 4 ranks * 2 ranks = 8
    executor_->SetTopoMatch(info);

    HcclResult ret = executor_->CalcAlgHierarchyInfo(nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(executor_->GetRankSize(), 8u);
}

TEST_F(CalcAlgHierarchyInfoTest, ThreeLevelRankSize)
{
    AlgHierarchyInfoForAllLevel info;
    info.infos = {{{0, 1, 2, 3, 4, 5, 6, 7}}, {{0, 1, 2, 3, 4, 5, 6, 7}}, {{0, 1}}}; // 8*8*2 = 128
    executor_->SetTopoMatch(info);

    HcclResult ret = executor_->CalcAlgHierarchyInfo(nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(executor_->GetRankSize(), 128u);
}

} // namespace testing
} // namespace ops_hccl
