/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * OpsExecutor UT 公共辅助：TestableOpsExecutor + fixture
 */

#pragma once

#include <gtest/gtest.h>

#include "ops_executor.h"

namespace ops_hccl {
namespace testing {

class TestableOpsExecutor : public OpsExecutor {
public:
    using OpsExecutor::GetMaxProcCntPerLoop;
    using OpsExecutor::RestoreChannelMap;
    using OpsExecutor::InitAlgoExecDataDesc;
    using OpsExecutor::GenTemplateDataParams;
    using OpsExecutor::GenTemplateRes;
    using OpsExecutor::UpdateSubCommMaskMap;
    using OpsExecutor::UpdateDataSplitParallel;
    using OpsExecutor::UpdateDataSplitSequence;
    using OpsExecutor::MergeChildrenOutput;

    TestableOpsExecutor(HcclAlgorithm &algo, OpParam &param) : OpsExecutor(algo, param) {}

    // setters
    void SetCclBufferSize(u64 s)          { cclBufferInfo_.size = s; }
    void SetScratchMultiple(u32 v)        { scratchMultiple_ = v; }
    void SetDataTypeSize(u64 v)           { dataTypeSize_ = v; }
    void SetRankSize(u32 v)               { rankSize_ = v; }
    void SetCmdType(HcclCMDType t)        { algo_.hcclCmdType = t; }
    void SetRoot(u32 r)                   { root_ = r; }
    void SetOpMode(OpMode m)              { opMode_ = m; }
    void SetMyRank(u32 r)                 { myRank_ = r; }
    void SetDataInfoInput(void *p, u64 s)  { dataInfo_.inputPtr = p; dataInfo_.inputSize = s; }
    void SetDataInfoOutput(void *p, u64 s) { dataInfo_.outputPtr = p; dataInfo_.outputSize = s; }
    void SetDataInfoDataType(HcclDataType t) { dataInfo_.dataType = t; }
    void SetDataInfoReduceOp(HcclReduceOp r) { dataInfo_.reduceOp = r; }
    void SetCclBufferPtr(void *p)         { cclBufferInfo_.ptr = p; }
    void SetChannelTable(std::vector<std::map<u32, std::vector<ChannelInfo>>> &t) { channelTable_ = t; }
    void SetSubThreads(std::vector<std::vector<ThreadHandle>> &t) { subThreads_ = t; }
};

class OpsExecutorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        algo_.engineType = HcclAlgEngineType::AICPU;
        algo_.hcclCmdType = HCCL_CMD_ALLREDUCE;
        executor_.reset(new TestableOpsExecutor(algo_, param_));
    }
    void TearDown() override { executor_.reset(); }

    HcclAlgorithm algo_;
    OpParam param_;
    std::unique_ptr<TestableOpsExecutor> executor_;
};

} // namespace testing
} // namespace ops_hccl
