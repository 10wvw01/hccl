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
    info.infos = {{{0, 1}}, {{0, 1}}, {{0, 1}}}; // 2*2*2 = 8
    executor_->SetTopoMatch(info);

    HcclResult ret = executor_->CalcAlgHierarchyInfo(nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(executor_->GetRankSize(), 8u);
}

// ============================================================
// 2. Orchestrate — 数据编排主入口
// ============================================================

class OrchestrateTest : public OpsExecutorTest {
protected:
    void SetUp() override
    {
        OpsExecutorTest::SetUp();
        executor_->SetDataTypeSize(4);
        executor_->SetScratchMultiple(2);
    }

    AlgResourceCtxSerializable MakeMinimalResCtx()
    {
        AlgResourceCtxSerializable ctx;
        ctx.cclMem.addr = nullptr;
        ctx.cclMem.size = 1024;
        ctx.threads = {1};
        ctx.algHierarchyInfo.infos = {{{0}}};
        ctx.channels = {{}};
        return ctx;
    }
};

TEST_F(OrchestrateTest, DataCountZeroReturnsEarly)
{
    executor_->SetDataInfoInput(nullptr, 0);
    auto ctx = MakeMinimalResCtx();
    HcclResult ret = executor_->Orchestrate(ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// Orchestrate 正常循环流程依赖 algo tree + BaseTemplate 基础设施就绪
TEST_F(OrchestrateTest, DISABLED_SingleLoopWithLargeBuffer)
{
    executor_->SetDataInfoInput((void *)0x1000, 400);
    executor_->SetCclBufferSize(8000);
    auto ctx = MakeMinimalResCtx();
    HcclResult ret = executor_->Orchestrate(ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ============================================================
// 3. CalcRes — 资源计算
// Note: 依赖 subThreads_/requestChannels_ 等内部状态预分配，待资源模块就绪后启用
// ============================================================

class CalcResTest : public OpsExecutorTest {
protected:
    void SetUp() override
    {
        OpsExecutorTest::SetUp();
        executor_->SetTopoMatch();
    }
};

TEST_F(CalcResTest, DISABLED_EmptyAlgoTreeCalculatesBaseResources)
{
    AlgHierarchyInfoForAllLevel info;
    info.infos = {{{0, 1}}};
    executor_->SetAlgHierarchyInfo(info);
    AlgResourceRequest req;
    HcclResult ret = executor_->CalcRes(req);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

} // namespace testing
} // namespace ops_hccl
