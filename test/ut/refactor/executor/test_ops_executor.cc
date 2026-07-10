/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * OpsExecutor::GetMaxProcCntPerLoop 单元测试
 */

#include <gtest/gtest.h>

#include "ops_executor.h"

namespace ops_hccl {
namespace {

// 可测试子类：暴露 protected 成员和函数
class TestableOpsExecutor : public OpsExecutor {
public:
    using OpsExecutor::GetMaxProcCntPerLoop;

    TestableOpsExecutor(HcclAlgorithm &algo, OpParam &param) : OpsExecutor(algo, param) {}

    void SetCclBufferSize(u64 size) { cclBufferInfo_.size = size; }
    void SetScratchMultiple(u32 v) { scratchMultiple_ = v; }
    void SetDataTypeSize(u64 v) { dataTypeSize_ = v; }
    void SetRankSize(u32 v) { rankSize_ = v; }
    void SetCmdType(HcclCMDType t) { algo_.hcclCmdType = t; }
};

class GetMaxProcCntPerLoopTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        algo_.engineType = HcclAlgEngineType::AICPU;
        algo_.hcclCmdType = HCCL_CMD_ALLREDUCE; // 默认非 AllGather，需要 rankSize 对齐
        executor_ = new TestableOpsExecutor(algo_, param_);
    }

    void TearDown() override
    {
        delete executor_;
        executor_ = nullptr;
    }

    HcclAlgorithm algo_;
    OpParam param_;
    TestableOpsExecutor *executor_ = nullptr;
};

// ============================================================
// scratchMultiple_ == 0 → 直接返回 dataCount
// ============================================================
TEST_F(GetMaxProcCntPerLoopTest, ZeroScratchMultipleReturnsDataCount)
{
    executor_->SetScratchMultiple(0);
    executor_->SetDataTypeSize(4);
    EXPECT_EQ(executor_->GetMaxProcCntPerLoop(1000), 1000);
}

// ============================================================
// CCL buffer 约束：maxByCcl = cclBufferSize / (scratchMultiple * dataTypeSize)
// ============================================================
TEST_F(GetMaxProcCntPerLoopTest, CclBufferBottleneck)
{
    executor_->SetScratchMultiple(2);
    executor_->SetDataTypeSize(4);        // 4 bytes/element
    executor_->SetCclBufferSize(800);     // 800 bytes CCL buffer
    executor_->SetRankSize(1);
    // maxByCcl = 800 / (2 * 4) = 100
    EXPECT_EQ(executor_->GetMaxProcCntPerLoop(1000), 100);
}

TEST_F(GetMaxProcCntPerLoopTest, DataCountBottleneck)
{
    executor_->SetScratchMultiple(2);
    executor_->SetDataTypeSize(4);
    executor_->SetCclBufferSize(8000);    // large buffer, not a bottleneck
    executor_->SetRankSize(1);
    // maxByCcl = 8000/8 = 1000, but dataCount=200 is smaller
    EXPECT_EQ(executor_->GetMaxProcCntPerLoop(200), 200);
}

// ============================================================
// UB 传输约束：maxByUb = UB_MAX_DATA_SIZE / dataTypeSize
// UB_MAX_DATA_SIZE = 256 * 1024 * 1024 = 268435456
// ============================================================
TEST_F(GetMaxProcCntPerLoopTest, UbBottleneck)
{
    executor_->SetScratchMultiple(1);
    executor_->SetDataTypeSize(2);
    executor_->SetCclBufferSize(268435456 * 10); // huge CCL, not bottleneck
    executor_->SetRankSize(1);
    // maxByUb = 256MB / 2 = 134217728
    EXPECT_EQ(executor_->GetMaxProcCntPerLoop(200000000ULL),
              134217728ULL);
}

// ============================================================
// AllGather 场景：不需要 rankSize 对齐
// ============================================================
TEST_F(GetMaxProcCntPerLoopTest, AllGatherNoRankAlign)
{
    executor_->SetScratchMultiple(2);
    executor_->SetDataTypeSize(4);
    executor_->SetCclBufferSize(1000);
    executor_->SetRankSize(8);
    executor_->SetCmdType(HCCL_CMD_ALLGATHER);
    // maxByCcl = 1000/8 = 125, no rank-alignment needed for AllGather
    EXPECT_EQ(executor_->GetMaxProcCntPerLoop(1000), 125);
}

// ============================================================
// 非 AllGather：需要 rankSize 对齐
// ============================================================
TEST_F(GetMaxProcCntPerLoopTest, NonAllGatherRankAlign)
{
    executor_->SetScratchMultiple(2);
    executor_->SetDataTypeSize(4);
    executor_->SetCclBufferSize(1000);
    executor_->SetRankSize(8);
    executor_->SetCmdType(HCCL_CMD_ALLREDUCE);
    // maxByCcl = 1000/8 = 125, rankAlign: 125/8*8 = 120
    EXPECT_EQ(executor_->GetMaxProcCntPerLoop(1000), 120);
}

// ============================================================
// zero-division guard: resCount = 0 → returns 1
// ============================================================
TEST_F(GetMaxProcCntPerLoopTest, ReturnsAtLeastOne)
{
    executor_->SetScratchMultiple(100);
    executor_->SetDataTypeSize(100);
    executor_->SetCclBufferSize(1);       // tiny buffer
    executor_->SetRankSize(1);
    // maxByCcl = 1/10000 = 0 → resCount = min(dataCount,0,UB) = 0 → std::max(0,1) = 1
    EXPECT_EQ(executor_->GetMaxProcCntPerLoop(100), 1);
}

} // namespace
} // namespace ops_hccl

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
