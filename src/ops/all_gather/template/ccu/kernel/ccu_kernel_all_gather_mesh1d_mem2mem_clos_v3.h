/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_MESH_1D_MEM2MEM_CLOS_V3
#define HCCL_CCU_KERNEL_ALL_GATHER_MESH_1D_MEM2MEM_CLOS_V3

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgAllGatherMesh1DMem2MemClosV3 : CcuKernelArgBase {
    uint64_t                                rankSize;
    uint32_t                                rankId;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
    std::vector<uint32_t>                   mainChannelIdxByRank;
    std::vector<uint32_t>                   sharedChannelIdxByRank;
};

struct AllGatherMesh1DMem2MemClosV3Context : CcuKernelCtxBase {
    const CcuKernelArgAllGatherMesh1DMem2MemClosV3 *arg;

    ccu::Variable input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    std::vector<ccu::Variable> sharedOutput;
    std::vector<ccu::Variable> sharedToken;
    ccu::Variable currentRankSliceInputOffset;
    ccu::Variable currentRankSliceOutputOffset;
    ccu::Variable tmpRepeatNum;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable normalSliceSize;
    ccu::Variable lastSliceSize;
    ccu::Variable isInputOutputEqual;
    ccu::Variable mainSliceSize;
    ccu::Variable sharedSliceSize;
    GroupOpSizeVars goSize;
    ccu::LocalAddr src_loccopy;
    ccu::LocalAddr localDst;
    ccu::LocalAddr sharedSrc;
    std::vector<ccu::Event> events;
    std::vector<ccu::Event> sharedEvents;
    std::vector<uint16_t> sharedEventMasks;
    std::vector<uint32_t> mainChannelIdxByRank;
    std::vector<uint32_t> sharedChannelIdxByRank;
};

CcuResult CcuAllGatherMesh1DMem2MemClosV3Kernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_GATHER_MESH_1D_MEM2MEM_CLOS_V3
