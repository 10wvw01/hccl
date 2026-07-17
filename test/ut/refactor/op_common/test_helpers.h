/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * op_common UT helpers.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "alg_param.h"
#include "exec_timeout_manager.h"
#include "op_common.h"
#include "topo/topo.h"
#include "channel.h"

namespace ops_hccl {
namespace testing {

class OpCommonTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

inline TopoInfo MakeDefaultTopoInfo()
{
    TopoInfo topo;
    topo.userRank = 0;
    topo.userRankSize = 8;
    topo.serverIdx = 0;
    topo.superPodIdx = 0;
    topo.deviceType = DevType::DEV_TYPE_COUNT;
    topo.deviceNumPerModule = 8;
    topo.serverNumPerSuperPod = 4;
    topo.serverNum = 1;
    topo.moduleNum = 1;
    topo.superPodNum = 1;
    topo.moduleIdx = 0;
    topo.isDiffDeviceModule = false;
    topo.multiModuleDiffDeviceNumMode = false;
    topo.multiSuperPodDiffServerNumMode = false;
    return topo;
}

} // namespace testing
} // namespace ops_hccl
