/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_MESH_1D_TWOSHOT_MEM2MEM
#define HCCL_CCU_KERNEL_REDUCE_MESH_1D_TWOSHOT_MEM2MEM

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgReduceMesh1DTwoShotMem2Mem: CcuKernelArgBase {
    uint64_t                                rankSize;
    uint32_t                                rankId;
    uint32_t                                rootId;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct ReduceMesh1DTwoShotMem2MemContext: CcuKernelCtxBase {
    const CcuKernelArgReduceMesh1DTwoShotMem2Mem *arg;

    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;

    std::vector<ccu::Variable> input;
    std::vector<ccu::Variable> scratch;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;

    ccu::Variable normalSliceSize;
    ccu::Variable lastSliceSize;
    ccu::Variable mySliceSize;
    ccu::Variable myScratchOffset;
    ccu::Variable peerSliceSize;
    ccu::Variable repeatNumVar;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable flag;
    ccu::Variable constVar1;
    GroupOpSizeVars goSize;

    std::vector<ccu::Event> events;

    ccu::LocalAddr myInput;
    ccu::LocalAddr myScratchResult;
    ccu::LocalAddr myOutput;
    ccu::RemoteAddr remoteScratch;
    ccu::RemoteAddr remoteOutput;

    std::vector<ccu::LocalAddr> scratchMem;
};

CcuResult CcuReduceMesh1DTwoShotMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_REDUCE_MESH_1D_TWOSHOT_MEM2MEM
