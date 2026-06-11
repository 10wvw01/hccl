/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_MESH_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_MESH_1D_MEM2MEM_H

#include <vector>
#include <map>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "omnipipe_data_slice_calc.h"

namespace ops_hccl {

class CcuKernelArgReduceScatterOmniPipeMesh1DMem2Mem : public CcuKernelArg {
public:
    explicit CcuKernelArgReduceScatterOmniPipeMesh1DMem2Mem(uint64_t dimSize, uint32_t rankId,  
                                                            const OpParam& opParam,
                                                            const std::vector<std::vector<uint32_t>>& subCommRanks)
        : dimSize_(dimSize),
          rankId_(rankId),
          opParam_(opParam),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgReduceScatterOmniPipeMesh1DMem2Mem] dimSize=%llu, rankId=%u, reduceOp=%d, dataType=%d",
                   dimSize_, rankId_, opParam.reduceType, opParam.DataDes.dataType);
    }

    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceScatterOmniPipeMesh1DMem2Mem", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t dimSize_;
    uint32_t rankId_;
    OpParam opParam_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgReduceScatterOmniPipeMesh1DMem2Mem : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterOmniPipeMesh1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, 
                                                           uint64_t scratchAddr, uint64_t sliceSize,
                                                           uint64_t offSet, uint64_t token, 
                                                           uint64_t localCopyFlag, uint64_t inputSliceStride,
                                                           uint64_t outputSliceStride, 
                                                           uint64_t inputOmniPipeSliceStride)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), scratchAddr_(scratchAddr), 
          sliceSize_(sliceSize), offSet_(offSet), token_(token),
          localCopyFlag_(localCopyFlag), inputSliceStride_(inputSliceStride),
          outputSliceStride_(outputSliceStride), inputOmniPipeSliceStride_(inputOmniPipeSliceStride)
    {
        HCCL_DEBUG("[CcuTaskArgReduceScatterOmniPipeMesh1DMem2Mem] inputAddr=%llu outputAddr=%llu sliceSize=%llu offSet=%llu localCopyFlag=%llu",
                   inputAddr_, outputAddr_, sliceSize_, offSet_, localCopyFlag_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t scratchAddr_;
    uint64_t sliceSize_;
    uint64_t offSet_;
    uint64_t token_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t localCopyFlag_;
    uint64_t inputOmniPipeSliceStride_;
};

struct ReduceScatterOmniPipeMesh1DMem2MemContext {
    CcuKernelArgReduceScatterOmniPipeMesh1DMem2Mem* arg;
    
    std::vector<ccu::Variable> input;
    ccu::Variable output;
    ccu::Variable scratch;
    std::vector<ccu::Variable> token;
    
    ccu::Variable offSet;
    ccu::Variable sliceSize;
    ccu::Variable inputSliceStride;
    ccu::Variable outputSliceStride;
    ccu::Variable inputOmniPipeSliceStride;
    ccu::Variable localCopyFlag;
    
    ccu::GroupOpSize groupOpSize;
    ccu::Event event;
    
    uint16_t selfBit;
    uint16_t allBit;
    
    bool resourceAllocated;
    ccu::MoConfig moConfig;
    ccu::MoRes moRes;
};

CcuResult CcuReduceScatterOmniPipeMesh1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_MESH_1D_MEM2MEM_H