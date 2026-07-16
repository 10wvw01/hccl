/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * topo unit test (CalGCD, CalcGeneralTopoInfoFor*, rank mapping).
 */

#include "test_helpers.h"

using namespace ops_hccl;
using ops_hccl::testing::OpCommonTest;
using ops_hccl::testing::MakeDefaultTopoInfo;

// ───────────── CalGCD(u32, u32) ─────────────

TEST_F(OpCommonTest, CalGCD_12_8_Returns4)
{
    EXPECT_EQ(CalGCD(12, 8), 4u);
}

TEST_F(OpCommonTest, CalGCD_0_5_Returns1)
{
    EXPECT_EQ(CalGCD(0, 5), 1u);
}

TEST_F(OpCommonTest, CalGCD_5_0_Returns1)
{
    EXPECT_EQ(CalGCD(5, 0), 1u);
}

TEST_F(OpCommonTest, CalGCD_7_7_Returns7)
{
    EXPECT_EQ(CalGCD(7, 7), 7u);
}

TEST_F(OpCommonTest, CalGCD_15_25_Returns5)
{
    EXPECT_EQ(CalGCD(15, 25), 5u);
}

// ───────────── CalGCD(vector<u32>) ─────────────

TEST_F(OpCommonTest, CalGCDVector_12_8_4_Returns4)
{
    std::vector<u32> nums = {12, 8, 4};
    EXPECT_EQ(CalGCD(nums), 4u);
}

TEST_F(OpCommonTest, CalGCDVector_EmptyReturns1)
{
    std::vector<u32> nums;
    EXPECT_EQ(CalGCD(nums), 1u);
}

TEST_F(OpCommonTest, CalGCDVector_SingleElementReturnsSelf)
{
    std::vector<u32> nums = {7};
    EXPECT_EQ(CalGCD(nums), 7u);
}

// ───────────── CalcGeneralTopoInfoForA2 ─────────────

TEST_F(OpCommonTest, CalcGeneralTopoInfoForA2)
{
    TopoInfo topo = MakeDefaultTopoInfo();
    topo.userRank = 5;
    topo.deviceNumPerModule = 8;
    topo.moduleNum = 2;
    topo.moduleIdx = 0;

    AlgHierarchyInfo algHierarchyInfo;
    HcclResult ret = CalcGeneralTopoInfoForA2(nullptr, &topo, algHierarchyInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(algHierarchyInfo.levels, 2u);
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL0].localRank, 5u);
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL0].localRankSize, 8u);
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL1].localRank, 0u);
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL1].localRankSize, 2u);
}

// ───────────── CalcGeneralTopoInfoForA3 ─────────────

TEST_F(OpCommonTest, CalcGeneralTopoInfoForA3)
{
    TopoInfo topo = MakeDefaultTopoInfo();
    topo.userRank = 10;
    topo.deviceNumPerModule = 8;
    topo.serverNumPerSuperPod = 4;
    topo.superPodNum = 2;
    topo.serverIdx = 5;

    AlgHierarchyInfo algHierarchyInfo;
    HcclResult ret = CalcGeneralTopoInfoForA3(nullptr, &topo, algHierarchyInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(algHierarchyInfo.levels, 3u);
    // l0: userRank % deviceNumPerModule = 10 % 8 = 2
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL0].localRank, 2u);
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL0].localRankSize, 8u);
    // l1: serverIdx % serverNumPerSuperPod = 5 % 4 = 1
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL1].localRank, 1u);
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL1].localRankSize, 4u);
    // l2: serverIdx / serverNumPerSuperPod = 5 / 4 = 1
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL2].localRank, 1u);
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL2].localRankSize, 2u);
}

// ───────────── CalcGeneralTopoInfoForComm ─────────────

TEST_F(OpCommonTest, CalcGeneralTopoInfoForComm)
{
    TopoInfo topo = MakeDefaultTopoInfo();
    topo.userRank = 3;
    topo.userRankSize = 8;

    AlgHierarchyInfo algHierarchyInfo;
    HcclResult ret = CalcGeneralTopoInfoForComm(nullptr, &topo, algHierarchyInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(algHierarchyInfo.levels, 2u);
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL0].localRank, 0u);
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL0].localRankSize, 1u);
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL1].localRank, 3u);
    EXPECT_EQ(algHierarchyInfo.infos[COMM_LEVEL1].localRankSize, 8u);
}

// ───────────── GetUserRankBySubCommRank ─────────────

TEST_F(OpCommonTest, GetUserRankBySubCommRank_Level0)
{
    // 2-level hierarchy: l0: localRank=2, localRankSize=8; l1: localRank=1, localRankSize=2
    AlgHierarchyInfo algHierarchyInfo;
    algHierarchyInfo.levels = 2;
    algHierarchyInfo.infos[0].localRank = 2;
    algHierarchyInfo.infos[0].localRankSize = 8;
    algHierarchyInfo.infos[1].localRank = 1;
    algHierarchyInfo.infos[1].localRankSize = 2;

    // curLevel=0, subCommRank=5:
    // level 0 (==curLevel): userRank += 5 * 1 = 5; preLevelsRankSize = 8
    // level 1 (!=curLevel): userRank += 1 * 8 = 8; preLevelsRankSize = 16
    // total = 13
    u32 userRank = 0;
    HcclResult ret = GetUserRankBySubCommRank(5, 0, algHierarchyInfo, userRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(userRank, 13u);
}

TEST_F(OpCommonTest, GetUserRankBySubCommRank_Level1)
{
    AlgHierarchyInfo algHierarchyInfo;
    algHierarchyInfo.levels = 2;
    algHierarchyInfo.infos[0].localRank = 2;
    algHierarchyInfo.infos[0].localRankSize = 8;
    algHierarchyInfo.infos[1].localRank = 1;
    algHierarchyInfo.infos[1].localRankSize = 2;

    // curLevel=1, subCommRank=1:
    // level 0 (!=curLevel): userRank += 2 * 1 = 2; preLevelsRankSize = 8
    // level 1 (==curLevel): userRank += 1 * 8 = 8; preLevelsRankSize = 16
    // total = 10
    u32 userRank = 0;
    HcclResult ret = GetUserRankBySubCommRank(1, 1, algHierarchyInfo, userRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(userRank, 10u);
}

// ───────────── GetSubCommRankByUserRank ─────────────

TEST_F(OpCommonTest, GetSubCommRankByUserRank_Level0)
{
    AlgHierarchyInfo algHierarchyInfo;
    algHierarchyInfo.levels = 2;
    algHierarchyInfo.infos[0].localRank = 2;
    algHierarchyInfo.infos[0].localRankSize = 8;
    algHierarchyInfo.infos[1].localRank = 1;
    algHierarchyInfo.infos[1].localRankSize = 2;

    // curLevel=0, userRank=13: subCommRank = 13 / 1 % 8 = 5
    u32 subCommRank = 0;
    HcclResult ret = GetSubCommRankByUserRank(13, 0, algHierarchyInfo, subCommRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(subCommRank, 5u);
}

TEST_F(OpCommonTest, GetSubCommRankByUserRank_Level1)
{
    AlgHierarchyInfo algHierarchyInfo;
    algHierarchyInfo.levels = 2;
    algHierarchyInfo.infos[0].localRank = 2;
    algHierarchyInfo.infos[0].localRankSize = 8;
    algHierarchyInfo.infos[1].localRank = 1;
    algHierarchyInfo.infos[1].localRankSize = 2;

    // curLevel=1, userRank=10: subCommRank = 10 / 8 % 2 = 1
    u32 subCommRank = 0;
    HcclResult ret = GetSubCommRankByUserRank(10, 1, algHierarchyInfo, subCommRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(subCommRank, 1u);
}
