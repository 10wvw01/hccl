/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_KERNEL_OMNI_H_
#define HCCLV2_CCU_KERNEL_OMNI_H_

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {


using RankId = u32;

class CcuKernelArgOmni : public hcomm::CcuKernelArg{
public:
    CcuKernelArgOmni(uint32_t rankId, const OpParam &opParam,
        const std::vector<std::vector<RankId>> &subCommRanks, const std::vector<RankId> &rankGroup,
        bool handleSelfRank, const std::vector<OmniSendRecvInfo>& sendRecvInfo)
        : rankId_(rankId), opParam_(opParam), subCommRanks_(subCommRanks), rankGroup_(rankGroup),
        handleSelfRank_(handleSelfRank), sendRecvInfo_(sendRecvInfo){}
    ~CcuKernelArgOmni() override {}

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgOmni", opParam_, subCommRanks_);
        return signature;
    }

    uint32_t rankId_;
    OpParam opParam_;
    bool handleSelfRank_;
    std::vector<std::vector<RankId>> subCommRanks_;
    std::vector<RankId> rankGroup_;
    std::vector<OmniSendRecvInfo> sendRecvInfo_;
};

class CcuTaskArgOmni : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgOmni(uint64_t inputAddr, uint64_t outputAddr, uint64_t scratchAddr,
        uint64_t token,  uint64_t sliceSize, uint64_t repeatNum, uint64_t inputRepeatStride, uint64_t outputRepeatStride) :
        inputAddr_(inputAddr), outputAddr_(outputAddr), scratchAddr_(scratchAddr), token_(token), sliceSize_(sliceSize), repeatNum_(repeatNum),
        inputRepeatStride_(inputRepeatStride), outputRepeatStride_(outputRepeatStride){}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t scratchAddr_;
    uint64_t sliceSize_;
    uint64_t token_;
    uint64_t repeatNum_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;
};

class CcuKernelOmni : public CcuKernelAlgBase {
public:
    CcuKernelOmni(const hcomm::CcuKernelArg &arg);
    ~CcuKernelOmni() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

// protected:
//     struct CcuSliceInfo {
//         hcomm::CcuRep::Variable sliceType_;
//         hcomm::CcuRep::Variable sliceIdx_;
//     };

//     struct CcuSendRecvInfo {
//         hcomm::CcuRep::Variable              optype_;
//         hcomm::CcuRep::Variable              sliceSize_;
//         std::vector<CcuSliceInfo>            srcSliceInfo_;
//         std::vector<CcuSliceInfo>            dstSliceInfo_;
//         std::vector<hcomm::CcuRep::Variable> remoteRank_;  // 对于GroupBroadcast和GroupReduce会有多个rank, 其余都是一对一
//     };

private:
    HcclResult InitResources();
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoRepeatOmni();
    void ReduceLoopGroup(CcuRep::LocalAddr &outDstOrg, std::vector<CcuRep::LocalAddr> &srcOrg,
                                                        GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType,
                                                        HcclReduceOp opType, std::string loopName);
    std::string GetLoopBlockTag(std::string loopType, int32_t index);
    void CreateReduceLoop(uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
                                                         HcclReduceOp opType, std::string loopName);

    uint32_t rankId_;
    OpParam opParam_;
    std::vector<std::vector<RankId>> subCommRanks_;
    std::vector<RankId> rankGroup_;
    std::map<uint32_t, ChannelHandle> rankId2Channel_;
    std::vector<OmniSendRecvInfo> ccuSendRecvInfo_;
    bool handleSelfRank_;

    std::vector<hcomm::CcuRep::Variable> input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;
    std::vector<ChannelHandle> channels_;
    hcomm::CcuRep::Variable scratchAddr_;
    hcomm::CcuRep::Variable sliceSize_;
    hcomm::CcuRep::Variable repeatNum_;
    hcomm::CcuRep::Variable inputRepeatStride_;
    hcomm::CcuRep::Variable outputRepeatStride_;

    GroupOpSize groupOpSize_;

    // 在本地的搬运完成标记
    hcomm::CcuRep::CompletedEvent event_;
};
} // namespace ops_hccl

#endif // HCCLV2_CCU_KERNEL_OMNI_H_