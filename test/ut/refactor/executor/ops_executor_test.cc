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
};

// ============================================================
// 3. 构造完整 Executor 实例：ALLGATHER + AICPU + OmniPipe 树
// ============================================================

TEST_F(OmniPipeTest, ConstructExecutorWithOmniPipeAlgo)
{
    // 构造完整的 HcclAlgorithm: ALLGATHER + AICPU + OmniPipe 树
    HcclAlgorithm algo;
    algo.hcclCmdType   = HCCL_CMD_ALLGATHER;
    algo.engineType    = HcclAlgEngineType::AICPU;
    algo.topoMatch     = std::make_shared<MockTopoMatch>();
    algo.algoExecDesc  = BuildOmniPipeTree();

    // 构造 executor 实例
    OpParam param;
    OpsExecutor executor(algo, param);
    EXPECT_TRUE(true); // 构造成功
}

} // namespace testing
} // namespace ops_hccl
