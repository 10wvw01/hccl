/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_GATHER_OMNIPIPE_NHR1D_MEM2MEMY
#define HCCL_CCU_KERNEL_GATHER_OMNIPIPE_NHR1D_MEM2MEMY

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

using NHRStepInfo = struct NHRStepInfo {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfo() : nSlices(0) {}
};

class CcuKernelArgGatherOmniPipeNHR1DMem2Mem : public hcomm::CcuKernelArg {
public:
    explicit CcuKernelArgGatherOmniPipeNHR1DMem2Mem(uint64_t dimSize, uint32_t rankId, uint32_t rootId,
                                                      const OpParam& opParam,
                                                      const std::vector<std::vector<uint32_t>>& subCommRanks, bool ifRealRoot, uint32_t realrank,
                                                      const std::vector<NHRStepInfo> stepInfoVector, const std::map<uint32_t, uint32_t> rank2ChannelIdx,
                                                      const std::map<uint32_t, uint32_t> subRankIdx2RankIdx)
        : dimSize_(dimSize),
        rankId_(rankId),
        rootId_(rootId),
        opParam_(opParam),
        subCommRanks_(subCommRanks),
        ifRealRoot_(ifRealRoot),
        myrealrank_(realrank),
        stepInfoVector_(stepInfoVector),
        rank2ChannelIdx_(rank2ChannelIdx),
        subRankIdx2RankIdx_(subRankIdx2RankIdx)
    {
        HCCL_DEBUG("[CcuKernelArgGatherOmniPipeNHR1DMem2Mem] dimSize=%llu, rankId=%u, rootId=%u",
                   dimSize_, rankId_, rootId_);
    }

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgGatherOmniPipeNHR1DMem2Mem", opParam_, subCommRanks_);
        return signature;
    }

    uint64_t dimSize_;
    uint32_t rankId_;
    uint32_t rootId_;
    OpParam opParam_;
    bool ifRealRoot_;
    uint32_t myrealrank_;
    std::vector<std::vector<uint32_t>> subCommRanks_;

    std::vector<NHRStepInfo> stepInfoVector_;           // NHR步骤信息向量
    std::map<u32, u32> rank2ChannelIdx_;                // 虚拟rank ID到channel索引的映射
    std::map<uint32_t, uint32_t> subRankIdx2RankIdx_;
};

class CcuTaskArgGatherOmniPipeNHR1DMem2Mem : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgGatherOmniPipeNHR1DMem2Mem(uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr, uint64_t token, // uint64_t scratchAddr,
                                                    uint64_t sliceSize,
                                                    uint64_t inputOmniPipeSliceStride, uint64_t outputOmniPipeSliceStride,
        bool isStepOne, bool isLastStep, bool ifNewRoot, std::vector<uint64_t> inputOmniSliceStrideVec)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), scratchAddr_(scratchAddr), token_(token), // scratchAddr_(scratchAddr), 
          sliceSize_(sliceSize), inputOmniPipeSliceStride_(inputOmniPipeSliceStride), outputOmniPipeSliceStride_(outputOmniPipeSliceStride),
          isStepOne_(isStepOne),
          isLastStep_(isLastStep),
          ifNewRoot_(ifNewRoot),
          inputOmniSliceStrideVec_(inputOmniSliceStrideVec)
    {
        HCCL_DEBUG("[CcuTaskArgGatherOmniPipeNHR1DMem2Mem] inputAddr=%llu, outputAddr=%llu, scratchAddr=%llu, token=%llu, "
                   "sliceSize=%llu, inputOmniPipeSliceStride=%llu",
                   inputAddr_, outputAddr_, scratchAddr, token_, sliceSize_, inputOmniPipeSliceStride_);
    }

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t scratchAddr_;
    uint64_t token_;
    uint64_t sliceSize_;
    uint64_t inputOmniPipeSliceStride_;
    uint64_t outputOmniPipeSliceStride_;
    bool isStepOne_;
    bool isLastStep_;
    bool ifNewRoot_;
    std::vector<uint64_t> inputOmniSliceStrideVec_;
};

class CcuKernelGatherOmniPipeNHR1DMem2Mem : public CcuKernelAlgBase {
public:
    CcuKernelGatherOmniPipeNHR1DMem2Mem(const hcomm::CcuKernelArg& arg);
    ~CcuKernelGatherOmniPipeNHR1DMem2Mem() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg& arg) override;

private:
    HcclResult InitResource();
    void LoadArgs();
    void PreSync();
    void PostSync();
    HcclResult DoGatherOmniPipeNHR();
    HcclResult DoGatherOmniPipeNHRSingleStep(const NHRStepInfo& nhrStepInfo);

    uint64_t rankSize_{0};
    uint32_t userRank_{0};
    uint32_t myrealrank_{0};
    uint32_t rankId_{0};
    uint32_t rootId_{0};
    std::vector<std::vector<uint32_t>> subCommRanks_;
    HcclDataType dataType_;
    HcclDataType outputDataType_;
    std::vector<ChannelHandle> channels_;
    std::vector<CcuRep::Variable> input_;
    CcuRep::Variable output_;
    std::vector<CcuRep::Variable> scratch_;
    std::vector<CcuRep::Variable> token_;
    hcomm::CcuRep::Variable sliceSize_;
    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    GroupOpSize groupOpSize_;
    hcomm::CcuRep::Variable inputOmniPipeSliceStride_;
    hcomm::CcuRep::Variable outputOmniPipeSliceStride_;
    HcclReduceOp reduceOp_;
    CcuRep::Variable isStepOne_;
    CcuRep::Variable isLastStep_;
    CcuRep::Variable ifNewRoot_;
    bool ifRealRoot_{false};

    hcomm::CcuRep::CompletedEvent event_;
    std::vector<CcuRep::RemoteAddr> inputMem_;
    std::vector<CcuRep::LocalAddr> outputMem_;
    std::vector<CcuRep::LocalAddr> scratchMem_;
    std::vector<CcuRep::RemoteAddr> remoteScratchMem_;
    std::vector<CcuRep::RemoteAddr> remoteInputhMem_;

    std::vector<hcomm::CcuRep::Variable> inputOmniSliceStrideVec_;

    uint32_t myRankIdx_{0};
    uint32_t localSize_{0};  // 本rank所在行或列的总rank数
    std::vector<NHRStepInfo> stepInfoVector_;
    std::map<uint32_t, uint32_t> rank2ChannelIdx_;
    std::map<uint32_t, uint32_t> subRankIdx2RankIdx_;
};

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_GATHER_OMNIPIPE_NHR1D_MEM2MEMY