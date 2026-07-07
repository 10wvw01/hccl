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
#include "omni_parser.h"

namespace ops_hccl {
struct OmniSendRecvInfo {
    omni::OpType opType;
    HcclDataType inputDataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceType;
    uint64_t sliceNum;
    uint64_t threadIdx;
    uint64_t netlayerId;
    std::vector<omni::OmniSliceInfo> srcSliceInfo;
    std::vector<omni::OmniSliceInfo> dstSliceInfo;
};

struct CcuKernelArgOmni : public CcuKernelArgBase {
    uint32_t rankId;
    OpParam opParam;
    bool handleSelfRank;
    std::vector<std::vector<u32>> subCommRanks;
    std::vector<u32> rankGroup;
    std::vector<std::vector<OmniSendRecvInfo>> sendRecvInfo;
};

struct A2AsingleSendRecvInfo {
    ccu::Variable tailSize;
    ccu::Variable loopNum;
    GroupOpSizeVars tailGoSize;
};

struct OmniContext : public CcuKernelCtxBase {
    const CcuKernelArgOmni *arg;
    uint64_t sliceNum;

    // rankSize个rank的原始数据
    std::vector<ccu::Variable> sendCounts;
    std::vector<ccu::Variable> recvCounts;
    std::vector<ccu::Variable> sdispls;
    std::vector<ccu::Variable> rdispls;

    // sliceNum个rank的搬运信息
    std::vector<A2AsingleSendRecvInfo> sendRecvCountsInfo;
    std::vector<ccu::Variable> recvSliceData;
    std::vector<ccu::Variable> sendSdispls;
    std::vector<ccu::Variable> localSdispls;

    std::map<uint32_t, ChannelHandle> rankId2Channel;
    std::map<u32, uint32_t> rankId2Idx;

    std::vector<ccu::Variable> input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;

    std::vector<ChannelHandle> channels;
    ccu::Variable scratchAddr;
    ccu::Variable syncIdx;

    GroupOpSizeVars goSize;

    // 在本地的搬运完成标记
    ccu::Event event;
    std::vector<ccu::Event> events;

    // Loop机制相关变量
    std::array<std::vector<ccu::LocalAddr>, 2> loopScratch;
    ccu::LocalAddr loopSrc[2];
    ccu::LocalAddr loopDst[2];
    ccu::Variable loopLen[2];
    ccu::Variable loopLenExp[2];

    ccu::Variable constVar1;
};

CcuResult CcuOmniKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCLV2_CCU_KERNEL_OMNI_H_