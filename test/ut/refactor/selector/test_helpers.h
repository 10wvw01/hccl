/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Selector UT 公共辅助：SelectorTest fixture + 构造 TopoInfo / OpParam 的 helper
 */

#pragma once

#include <gtest/gtest.h>
#include <memory>

#include "hccl_algorithm.h"
#include "alg_param.h"
#include "execute_selector.h"

namespace ops_hccl {

// 构造触发 AICPU_ALLGATHER_SEQUENCE_NHR_MESH1D / PARALLEL_MESH1D_NHR 分支的 TopoInfo
// 分支条件（all_gather_auto_selector.cc L42-L65）：
//   topoLevelNums > 1 且 != 3、Level1Nhr=false、Level0Nhr=false、
//   localNetInsSizeOfLayer[0] != 1、level0Topo == MESH_1D
inline std::unique_ptr<TopoInfoWithNetLayerDetails> MakeTwoLevelMesh1DTopo(u32 userRankSize)
{
    auto topo = std::make_unique<TopoInfoWithNetLayerDetails>();
    topo->userRankSize = userRankSize;
    topo->topoLevelNums = 2; // > 1 且 != 3
    topo->level0Topo = Level0Shape::MESH_1D;
    topo->Level0Nhr = false;
    topo->Level1Nhr = false;
    topo->deviceNumPerModule = 8;
    // localNetInsSizeOfLayer[0] != 1，避免进入 NHR 分支
    topo->netLayerDetails.localNetInsSizeOfLayer = {2, 1};
    return topo;
}

// 构造 AICPU ALLGATHER OpParam
inline std::unique_ptr<OpParam> MakeAicpuAllGatherParam(u64 count, HcclDataType dtype = HCCL_DATA_TYPE_FP32)
{
    auto param = std::make_unique<OpParam>();
    param->opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    param->opExecuteConfig = OpExecuteConfig::AICPU_TS; // 触发 IsStarsState -> SelectAicpuAlgo
    param->isMc2 = false;
    param->DataDes.count = count;
    param->DataDes.dataType = dtype;
    param->engine = CommEngine::COMM_ENGINE_AICPU_TS; // 匹配 LoadAICPUKernel 分支
    return param;
}

class SelectorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

} // namespace ops_hccl
