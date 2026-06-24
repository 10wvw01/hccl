/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEMY_H
#define HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEMY_H

#include <vector>
#include <ios>
#include <map>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgGatherOmniPipeMesh1DMem2MemY : CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
    uint32_t rootId;
    OpParam opParam;
    std::vector<std::vector<uint32_t>> subCommRanks;
    std::map<uint32_t, uint32_t> subRankIdx2RankIdx;
    bool ifRealRoot; 
    uint32_t myrealrank;
};

struct GatherOmniPipeMesh1DMem2MemContextY {
    CcuKernelArgGatherOmniPipeMesh1DMem2MemY* arg;
    uint64_t rankSize{0};
    uint32_t rankId{0};
    uint32_t rootId{0};
    std::map<uint32_t, uint32_t> subRankIdx2RankIdx;
    HcclDataType dataType{HcclDataType::HCCL_DATA_TYPE_RESERVED};

    std::vector<ccu::Variable> input;
    ccu::Variable output;
    std::vector<ccu::Variable> token;
    ccu::Variable sliceSize;
    ccu::Variable inputSliceStride;
    ccu::Variable outputSliceStride;
    ccu::Variable inputOmniPipeSliceStride;
    ccu::Variable outputOmniPipeSliceStride;
    ccu::Variable localCopyFlag;
    ccu::Variable isStepOne;
    ccu::Variable isLastStep;
    ccu::Variable ifNewRoot;
    
    std::vector<ccu::RemoteAddr> inputMem;
    std::vector<ccu::LocalAddr> outputMem;
    ccu::Event event;
};

CcuResult CcuGatherOmniPipeMesh1DMem2MemKernelY(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEMY_H