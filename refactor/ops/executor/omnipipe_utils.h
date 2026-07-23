/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_OMNIPIPE_UTILS_H
#define HCCL_OMNIPIPE_UTILS_H

#include <cmath>
#include <stdint.h>
#include <vector>
#include "utils/utils.h"


// ominipie只能是快轴和慢轴
constexpr uint32_t OMIN_MAX_STEP_NUM = 5;

//Mesh单链路带宽56G， CLOS单链路带宽待定
constexpr double OMIN_MESH_BW = 56;
constexpr double OMIN_CLOS_BW = 56 * 2;

extern double CalcBandwidth2D(double xB, double yB, u64 xRankSize, u64 yRankSize, int maxStepNum);
extern u64 CalcOmniPipeSteps(double bandwidthRatio, u64 xRankSize, u64 maxStep, double &scale);
extern u64 CalcOmniPipeData(double *xStepP2pDataSize, double *yStepP2pDataSize, double bandwidthRatio, u64 xRankSize,
    u64 yRankSize, u64 maxStep);
#endif