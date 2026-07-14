/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * OpsExecutor UT 公共辅助：TestableOpsExecutor + fixture
 */

#pragma once

#include <gtest/gtest.h>

#include "ops_executor.h"

namespace ops_hccl {

// Concrete mock: fills algHierarchyInfo with pre-set test data
class MockTopoMatch : public TopoMatchBase {
public:
    AlgHierarchyInfoForAllLevel mockInfo;
    MockTopoMatch() = default;
    std::string Describe() const override { return "MockTopoMatch"; }
    HcclResult MatchTopo(const HcclComm, TopoInfoWithNetLayerDetails *,
                         AlgHierarchyInfoForAllLevel &algHierarchyInfo) override
    {
        algHierarchyInfo = mockInfo;
        return HCCL_SUCCESS;
    }
};

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
    using OpsExecutor::InitRes;
    using OpsExecutor::Orchestrate; 

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
    void SetExecDataInfoInput(void *p, u64 s)  { dataInfo_.inputPtr = p; dataInfo_.inputSize = s; }
    void SetExecDataInfoOutput(void *p, u64 s) { dataInfo_.outputPtr = p; dataInfo_.outputSize = s; }
    void SetExecDataInfoDataType(HcclDataType t) { dataInfo_.dataType = t; }
    void SetCclBufferPtr(void *p)         { cclBufferInfo_.ptr = p; }
    void SetChannelTable(std::vector<std::map<u32, std::vector<ChannelInfo>>> &t) { channelTable_ = t; }
    void SetThreads(std::vector<ThreadHandle> &t) { threads_ = t; }
    void SetSubThreads(std::vector<std::vector<ThreadHandle>> &t) { subThreads_ = t; }
    void SetTopoMatch(AlgHierarchyInfoForAllLevel info = {})
    {
        auto m = std::make_shared<MockTopoMatch>();
        m->mockInfo = std::move(info);
        algo_.topoMatch = std::move(m);
    }
    void SetAlgHierarchyInfo(AlgHierarchyInfoForAllLevel &info) { algHierarchyInfo_ = info; }

    // getters
    u32 GetRankSize() const { return rankSize_; }
    u32 GetScratchMultiple() const { return scratchMultiple_; }
    const ops_hccl::ExecDataInfo &GetExecDataInfo() const { return dataInfo_; }
    u64 GetDataTypeSize() const { return dataTypeSize_; }
    const AlgHierarchyInfoForAllLevel &GetAlgHierarchyInfo() const { return algHierarchyInfo_; }
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

} // namespace ops_hccl
