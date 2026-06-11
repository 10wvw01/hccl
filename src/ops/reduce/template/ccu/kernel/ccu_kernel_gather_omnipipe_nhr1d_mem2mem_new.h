/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H

#include <vector>
#include <map>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

class CcuKernelArgGatherOmniPipeMesh1DMem2Mem : public CcuKernelArg {
public:
    explicit CcuKernelArgGatherOmniPipeMesh1DMem2Mem(uint64_t dimSize, uint32_t rankId, uint32_t rootId,
                                                      const OpParam& opParam,
                                                      const std::vector<std::vector<uint32_t>>& subCommRanks, 
                                                      std::map<u32, u32> subRankIdx2RankIdx, 
                                                      bool ifRealRoot, uint32_t realrank)
        : dimSize_(dimSize),
          rankId_(rankId),
          rootId_(rootId),
          opParam_(opParam),
          subCommRanks_(subCommRanks),
          subRankIdx2RankIdx_(subRankIdx2RankIdx),
          ifRealRoot_(ifRealRoot),
          myrealrank_(realrank)
    {
        HCCL_DEBUG("[CcuKernelArgGatherOmniPipeMesh1DMem2Mem] dimSize=%llu, rankId=%u, rootId=%u",
                   dimSize_, rankId_, rootId_);
    }

    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgGatherOmniPipeMesh1DMem2Mem", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t dimSize_;
    uint32_t rankId_;
    uint32_t rootId_;
    OpParam opParam_;
    std::map<uint32_t, uint32_t> subRankIdx2RankIdx_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
    bool ifRealRoot_; 
    uint32_t myrealrank_;
};

class CcuTaskArgGatherOmniPipeMesh1DMem2Mem : public CcuTaskArg {
public:
    explicit CcuTaskArgGatherOmniPipeMesh1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t token, 
                                                    uint64_t localCopyFlag,
                                                    uint64_t sliceSize, uint64_t inputSliceStride, 
                                                    uint64_t outputSliceStride,
                                                    u64 inputOmniPipeSliceStride, u64 outputOmniPipeSliceStride,
                                                    bool isStepOne, bool isLastStep, bool ifNewRoot)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), localCopyFlag_(localCopyFlag), 
          sliceSize_(sliceSize), inputSliceStride_(inputSliceStride), outputSliceStride_(outputSliceStride),
          inputOmniPipeSliceStride_(inputOmniPipeSliceStride), outputOmniPipeSliceStride_(outputOmniPipeSliceStride),
          isStepOne_(isStepOne), isLastStep_(isLastStep), ifNewRoot_(ifNewRoot)
    {
        HCCL_DEBUG("[CcuTaskArgGatherOmniPipeMesh1DMem2Mem] inputAddr=%llu, outputAddr=%llu, token=%llu, "
                   "sliceSize=%llu, sliceStride=%llu, localCopyFlag=%llu, inputOmniPipeSliceStride=%llu",
                   inputAddr_, outputAddr_, token_, sliceSize_, inputSliceStride_, localCopyFlag_,
                   inputOmniPipeSliceStride_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t localCopyFlag_;
    uint64_t sliceSize_;
    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    u64 inputOmniPipeSliceStride_;
    u64 outputOmniPipeSliceStride_;
    bool isStepOne_;
    bool isLastStep_;
    bool ifNewRoot_;
};

struct GatherOmniPipeMesh1DMem2MemContext {
    CcuKernelArgGatherOmniPipeMesh1DMem2Mem* arg;
    
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
    
    bool resourceAllocated;
    ccu::MoConfig moConfig;
    ccu::MoRes moRes;
};

CcuResult CcuGatherOmniPipeMesh1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H