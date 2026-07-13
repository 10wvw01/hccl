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

TEST_F(OmniPipeTest, OmniPipeTreeTopology)
{
    auto root = BuildOmniPipeTree();
    EXPECT_EQ(root.execPolicy, HcclAlgExecPolicy::SEQUENCE);
    EXPECT_EQ(root.dataSplitRatio, std::vector<u32>({4, 4}));

    // d4 = root.children[0]
    auto *d4 = std::get_if<std::shared_ptr<AlgoExecDesc>>(&root.children[0]);
    ASSERT_NE(d4, nullptr);
    EXPECT_EQ((*d4)->execPolicy, HcclAlgExecPolicy::PARALLEL);

    // d5 = root.children[1]
    auto *d5 = std::get_if<std::shared_ptr<AlgoExecDesc>>(&root.children[1]);
    ASSERT_NE(d5, nullptr);
    EXPECT_EQ((*d5)->execPolicy, HcclAlgExecPolicy::PARALLEL);

    // d3 shared between d4.children[1] and d5.children[0]
    auto *d4_d3 = std::get_if<std::shared_ptr<AlgoExecDesc>>(&(*d4)->children[1]);
    auto *d5_d3 = std::get_if<std::shared_ptr<AlgoExecDesc>>(&(*d5)->children[0]);
    ASSERT_NE(d4_d3, nullptr);
    ASSERT_NE(d5_d3, nullptr);
    EXPECT_EQ(d4_d3->get(), d5_d3->get());        // 同一个 shared_ptr 对象
    EXPECT_EQ((*d4_d3)->execPolicy, HcclAlgExecPolicy::SEQUENCE);

    // d3 → d1
    auto *d1 = std::get_if<std::shared_ptr<AlgoExecDesc>>(&(*d4_d3)->children[0]);
    ASSERT_NE(d1, nullptr);
    EXPECT_EQ((*d1)->execPolicy, HcclAlgExecPolicy::PARALLEL);

    // d1 leaf: Mesh, NHR
    auto *d1_t0 = std::get_if<TemplateExecDesc>(&(*d1)->children[0]);
    auto *d1_t1 = std::get_if<TemplateExecDesc>(&(*d1)->children[1]);
    ASSERT_NE(d1_t0, nullptr);
    ASSERT_NE(d1_t1, nullptr);
    EXPECT_EQ(d1_t0->templateDesc.algType, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH);
    EXPECT_EQ(d1_t1->templateDesc.algType, HcclAlgoType::HCCL_ALGO_TYPE_NHR);

    // d2 leaf: NHR, Mesh (opposite)
    auto *d3_d2 = std::get_if<std::shared_ptr<AlgoExecDesc>>(&(*d4_d3)->children[1]);
    ASSERT_NE(d3_d2, nullptr);
    auto *d2_t0 = std::get_if<TemplateExecDesc>(&(*d3_d2)->children[0]);
    auto *d2_t1 = std::get_if<TemplateExecDesc>(&(*d3_d2)->children[1]);
    ASSERT_NE(d2_t0, nullptr);
    ASSERT_NE(d2_t1, nullptr);
    EXPECT_EQ(d2_t0->templateDesc.algType, HcclAlgoType::HCCL_ALGO_TYPE_NHR);
    EXPECT_EQ(d2_t1->templateDesc.algType, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH);
}

} // namespace testing
} // namespace ops_hccl
