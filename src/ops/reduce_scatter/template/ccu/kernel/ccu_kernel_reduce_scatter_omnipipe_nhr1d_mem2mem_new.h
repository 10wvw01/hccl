/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H

#include <vector>
#include <map>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

using NHRStepInfoRS = struct NHRStepInfoDefRS {
    uint32_t step = 0;
    uint32_t myRank = 0;
    uint32_t nSlices;
    uint32_t toRank = 0;
    uint32_t fromRank = 0;
    std::vector<uint32_t> txSliceIdxs;
    std::vector<uint32_t> rxSliceIdxs;

    NHRStepInfoDefRS() : nSlices(0)
    {
    }
};

class CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem : public CcuKernelArg {
public:
    explicit CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem(uint64_t rankSize, uint32_t rankId,
                                                            const OpParam& opParam,
                                                            const std::vector<NHRStepInfoRS> stepInfoVector,
                                                            const std::map<uint32_t, uint32_t> rank2ChannelIdx,
                                                            const std::vector<std::vector<uint32_t>>& subCommRanks)
        : rankSize_(rankSize),
          rankId_(rankId),
          opParam_(opParam),
          stepInfoVector_(stepInfoVector),
          rank2ChannelIdx_(rank2ChannelIdx),
          subCommRanks_(subCommRanks)
    {
        HCCL_DEBUG("[CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem] rankId=%u, rankSize=%llu", rankId_, rankSize_);
    }

    CcuKernelSignature GetKernelSignature() const override
    {
        CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t rankSize_;
    uint32_t rankId_;
    OpParam opParam_;
    std::vector<NHRStepInfoRS> stepInfoVector_;
    std::map<uint32_t, uint32_t> rank2ChannelIdx_;
    std::vector<std::vector<uint32_t>> subCommRanks_;
};

class CcuTaskArgReduceScatterOmniPipeNHR1DMem2Mem : public CcuTaskArg {
public:
    explicit CcuTaskArgReduceScatterOmniPipeNHR1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
                                                         uint64_t sliceSize, uint64_t sliceStride, 
                                                         uint64_t localCopyFlag,
                                                         uint64_t inputOmniPipeSliceStride, 
                                                         std::vector<uint64_t> inputOmniSliceStrideVec,
                                                         uint64_t inputSliceStride)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), sliceSize_(sliceSize),
          sliceStride_(sliceStride), localCopyFlag_(localCopyFlag), 
          inputOmniPipeSliceStride_(inputOmniPipeSliceStride),
          inputOmniSliceStrideVec_(inputOmniSliceStrideVec), inputSliceStride_(inputSliceStride)
    {
        HCCL_DEBUG("[CcuTaskArgReduceScatterOmniPipeNHR1DMem2Mem] inputAddr=%llu outputAddr=%llu token=%llu "
                   "sliceSize=%llu sliceStride=%llu localCopyFlag=%llu inputOmniPipeSliceStride=%llu",
                   inputAddr_, outputAddr_, token_, sliceSize_, sliceStride_, localCopyFlag_,
                   inputOmniPipeSliceStride_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    uint64_t sliceSize_;
    uint64_t sliceStride_;
    uint64_t localCopyFlag_;
    uint64_t inputOmniPipeSliceStride_;
    std::vector<uint64_t> inputOmniSliceStrideVec_;
    uint64_t inputSliceStride_;
};

struct ReduceScatterOmniPipeNHR1DMem2MemContext {
    CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem* arg;
    
    std::vector<ccu::Variable> input;
    ccu::Variable output;
    std::vector<ccu::Variable> token;
    
    ccu::Variable sliceStride;
    ccu::Variable localCopyFlag;
    ccu::Variable sliceSize;
    ccu::Variable inputOmniPipeSliceStride;
    std::vector<ccu::Variable> inputOmniSliceStrideVec;
    ccu::Variable inputSliceStride;
    
    ccu::GroupOpSize groupOpSize;
    ccu::Event event;
    
    uint32_t myRankIdx;
    uint32_t localSize;
    
    bool resourceAllocated;
    ccu::MoConfig moConfig;
    ccu::MoRes moRes;
};

CcuResult CcuReduceScatterOmniPipeNHR1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_OMNIPIPE_NHR1D_MEM2MEM_H