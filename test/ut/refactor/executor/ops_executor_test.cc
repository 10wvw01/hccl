/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * OpsExecutor 对外三个函数测试：CalcAlgHierarchyInfo / CalcRes / Orchestrate
 */

#include "test_helpers.h"
#include "algorithm/all_gather/all_gather_template_desc.h"

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

// ============================================================
// 2. OmniPipe AllGather 算法树构造（递归嵌套 AlgoExecDesc）
// ============================================================

/*
 * 算法树结构（用户设计）:
 *
 *   algoExecDesc6 (SEQ, 4:4)
 *   ├── algoExecDesc4 (PAR, 2:2)
 *   │   ├── scatterNHR  (leaf template, subCommIndex=1)
 *   │   └── algoExecDesc3 (SEQ, 2:2)
 *   │       ├── algoExecDesc1 (PAR, 1:1)
 *   │       │   ├── scatterMseh (subCommIndex=0)
 *   │       │   └── scatterNHR  (subCommIndex=0)
 *   │       └── algoExecDesc2 (PAR, 1:1)
 *   │           ├── scatterNHR  (subCommIndex=0)
 *   │           └── scatterMseh (subCommIndex=0)
 *   └── algoExecDesc5 (PAR, 2:2)
 *       ├── algoExecDesc3 (shared_ptr, 同 d4 中的 d3)
 *       └── scatterNHR  (leaf template, subCommIndex=1)
 */

class OmniPipeTest : public OpsExecutorTest {
protected:
    void SetUp() override
    {
        OpsExecutorTest::SetUp();
        meshTmpl_ = g_allGatherTemplateDescMap[static_cast<size_t>(
            HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_FULLMESH_SINGLE_JETTY)];
        nhrTmpl_ = g_allGatherTemplateDescMap[static_cast<size_t>(
            HcclAllGatherTemplateDescType::ALLGATHER_TEMPLATE_NHR_SINGLE_JETTY)];
    }

    AlgoExecDesc BuildOmniPipeTree()
    {
        AlgoExecDesc d1;
        d1.execPolicy = HcclAlgExecPolicy::PARALLEL;
        d1.children = {
            TemplateExecDesc{meshTmpl_, SUB_COMM_INDEX_INTRA},
            TemplateExecDesc{nhrTmpl_,  SUB_COMM_INDEX_INTRA},
        };
        d1.dataSplitRatio = {1, 1};

        AlgoExecDesc d2;
        d2.execPolicy = HcclAlgExecPolicy::PARALLEL;
        d2.children = {
            TemplateExecDesc{nhrTmpl_,  SUB_COMM_INDEX_INTRA},
            TemplateExecDesc{meshTmpl_, SUB_COMM_INDEX_INTRA},
        };
        d2.dataSplitRatio = {1, 1};

        AlgoExecDesc d3;
        d3.execPolicy = HcclAlgExecPolicy::SEQUENCE;
        d3.children = {
            std::make_shared<AlgoExecDesc>(std::move(d1)),
            std::make_shared<AlgoExecDesc>(std::move(d2)),
        };
        d3.dataSplitRatio = {2, 2};

        auto sharedD3 = std::make_shared<AlgoExecDesc>(d3);

        AlgoExecDesc d4;
        d4.execPolicy = HcclAlgExecPolicy::PARALLEL;
        d4.children = {
            TemplateExecDesc{nhrTmpl_, SUB_COMM_INDEX_INTER},
            sharedD3,
        };
        d4.dataSplitRatio = {2, 2};

        AlgoExecDesc d5;
        d5.execPolicy = HcclAlgExecPolicy::PARALLEL;
        d5.children = {
            sharedD3,
            TemplateExecDesc{nhrTmpl_, SUB_COMM_INDEX_INTER},
        };
        d5.dataSplitRatio = {2, 2};

        AlgoExecDesc d6;
        d6.execPolicy = HcclAlgExecPolicy::SEQUENCE;
        d6.children = {
            std::make_shared<AlgoExecDesc>(std::move(d4)),
            std::make_shared<AlgoExecDesc>(std::move(d5)),
        };
        d6.dataSplitRatio = {4, 4};
        return d6;
    }

    TemplateDesc meshTmpl_;
    TemplateDesc nhrTmpl_;

    // 构造 AllGather 测试用的 TestableOpsExecutor 实例
    std::unique_ptr<TestableOpsExecutor> MakeOmniPipeExecutor(
        u32 rankSize = 128, u64 elemCount = 1024, HcclDataType dtype = HCCL_DATA_TYPE_FP32)
    {
        auto algo = std::make_unique<HcclAlgorithm>();
        algo->hcclCmdType  = HCCL_CMD_ALLGATHER;
        algo->engineType   = HcclAlgEngineType::AICPU;
        algo->topoMatch    = std::make_shared<MockTopoMatch>();
        algo->algoExecDesc = BuildOmniPipeTree();

        u64 dtSize = DATATYPE_SIZE_TABLE[dtype];
        u64 inSize = elemCount * dtSize;

        bufPool_.input.resize(inSize);
        bufPool_.output.resize(inSize * rankSize);

        auto param = std::make_unique<OpParam>();
        param->userRank         = 0;
        param->inputPtr         = bufPool_.input.data();
        param->inputSize        = inSize;
        param->outputPtr        = bufPool_.output.data();
        param->outputSize       = inSize * rankSize;
        param->DataDes.dataType = dtype;
        param->DataDes.count    = elemCount;
        param->reduceType       = HcclReduceOp::HCCL_REDUCE_RESERVED;
        param->opMode           = OpMode::OPBASE;

        return std::make_unique<TestableOpsExecutor>(*algo, *param);
    }

    // buffer 内存池，生命周期与 Test Fixture 一致
    struct {
        std::vector<char> input;
        std::vector<char> output;
    } bufPool_;
};

// ============================================================
// 3. 构造完整 Executor 实例：ALLGATHER + AICPU + OmniPipe 树
// ============================================================

TEST_F(OmniPipeTest, ConstructExecutorWithOmniPipeAlgo)
{
    auto exe = MakeOmniPipeExecutor();

    // 验证 rankSize 初始为 0，调 CalcAlgHierarchyInfo 后正确计算
    EXPECT_EQ(exe->GetRankSize(), 0u);
    AlgHierarchyInfoForAllLevel info;
    info.infos = {
        {{0, 1, 2, 3, 4, 5, 6, 7}},
        {{0, 1, 2, 3, 4, 5, 6, 7}},
        {{0, 1}}
    };
    exe->SetTopoMatch(info);
    EXPECT_EQ(exe->CalcAlgHierarchyInfo(nullptr, nullptr), HCCL_SUCCESS);
    EXPECT_EQ(exe->GetRankSize(), 128u);    // 8×8×2

    // 验证 scratchMultiple 初始为 0
    EXPECT_EQ(exe->GetScratchMultiple(), 0u);

    // 验证 dataInfo_ 从 OpParam 正确传递
    const auto &d = exe->GetExecDataInfo();
    EXPECT_NE(d.inputPtr, nullptr);
    EXPECT_EQ(d.inputSize, 4096u);           // 1024 × sizeof(float)
    EXPECT_NE(d.outputPtr, nullptr);
    EXPECT_EQ(d.outputSize, 4096u * 128);   // AllGather: rankSize 倍
    EXPECT_EQ(d.dataType, HCCL_DATA_TYPE_FP32);
    EXPECT_EQ(exe->GetDataTypeSize(), 4u);  // sizeof(float)
}

TEST_F(OmniPipeTest, ConstructExecutorWithFp16)
{
    auto exe = MakeOmniPipeExecutor(128, 1024, HCCL_DATA_TYPE_FP16);

    const auto &d = exe->GetExecDataInfo();
    EXPECT_EQ(d.dataType, HCCL_DATA_TYPE_FP16);
    EXPECT_EQ(d.inputSize, 2048u);          // 1024 × 2 (sizeof half)
    EXPECT_EQ(d.outputSize, 2048u * 128);  // AllGather: rankSize 倍
    EXPECT_EQ(exe->GetDataTypeSize(), 2u);  // sizeof(half)
}

} // namespace testing
} // namespace ops_hccl
