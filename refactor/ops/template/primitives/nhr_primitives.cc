/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nhr_primitives.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

u32 CalcNhrStepNum(u32 rankSize)
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    return nSteps;
}

std::pair<std::vector<u32>, std::vector<u32>> GenNhrReduceScatterSliceIdxs(
    u32 rankSize, u32 myRankIdx, u32 step)
{
    u32 deltaSliceIndex = 1 << (step + 1);
    u32 nSlices = (rankSize - 1 + (1 << step)) / deltaSliceIndex;
    u32 txSliceIdx = (myRankIdx + rankSize - (1 << step)) % rankSize;
    u32 rxSliceIdx = myRankIdx;

    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;
    txSliceIdxs.reserve(nSlices);
    rxSliceIdxs.reserve(nSlices);
    for (u32 i = 0; i < nSlices; i++) {
        txSliceIdxs.push_back(txSliceIdx);
        rxSliceIdxs.push_back(rxSliceIdx);
        txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
    }
    return {txSliceIdxs, rxSliceIdxs};
}

std::pair<std::vector<u32>, std::vector<u32>> GenNhrAllGatherSliceIdxs(
    u32 rankSize, u32 myRankIdx, u32 step, u32 nSteps)
{
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 nSlices = (rankSize - 1 + (1 << (nSteps - 1 - step))) / deltaSliceIndex;
    u32 txSliceIdx = myRankIdx;
    u32 rxSliceIdx = (myRankIdx - (1 << (nSteps - 1 - step)) + rankSize) % rankSize;

    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;
    txSliceIdxs.reserve(nSlices);
    rxSliceIdxs.reserve(nSlices);
    for (u32 i = 0; i < nSlices; i++) {
        txSliceIdxs.push_back(txSliceIdx);
        rxSliceIdxs.push_back(rxSliceIdx);
        txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
    }
    return {txSliceIdxs, rxSliceIdxs};
}

HcclResult RunNhrReduceScatter(const TemplateDataParams &tempAlgParams,
                               TemplateResource &templateResource,
                               EngineType engineType)
{
    (void)engineType;
    u32 rankSize = static_cast<u32>(tempAlgParams.allRankDispls.size());
    u32 myRankIdx = 0;
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    u32 nSteps = CalcNhrStepNum(rankSize);
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];

    for (u32 step = 0; step < nSteps; ++step) {
        u32 delta = 1 << step;
        u32 sendToIdx = (myRankIdx + rankSize - delta) % rankSize;
        u32 recvFromIdx = (myRankIdx + delta) % rankSize;
        auto [txIdxs, rxIdxs] = GenNhrReduceScatterSliceIdxs(rankSize, myRankIdx, step);

        const ChannelInfo &linkSend = templateResource.channels.at(sendToIdx)[0];
        const ChannelInfo &linkRecv = templateResource.channels.at(recvFromIdx)[0];

        std::vector<DataSlice> txSrc, txDst, rxSrc, rxDst;
        for (u32 i = 0; i < txIdxs.size(); ++i) {
            u64 sz = tempAlgParams.sliceSize;
            u64 txOff = tempAlgParams.buffInfo.hcclBuffBaseOff + tempAlgParams.sliceSize * txIdxs[i];
            u64 rxOff = tempAlgParams.buffInfo.hcclBuffBaseOff + tempAlgParams.sliceSize * rxIdxs[i];
            txSrc.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, txOff, sz, sz / dataTypeSize);
            txDst.emplace_back(linkSend.remoteCclMem.addr, txOff, sz, sz / dataTypeSize);
            rxSrc.emplace_back(linkRecv.remoteCclMem.addr, rxOff, sz, sz / dataTypeSize);
            rxDst.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, rxOff, sz, sz / dataTypeSize);
        }

        SendRecvReduceInfo info{{linkSend, linkRecv}, {{txSrc, txDst}, {rxSrc, rxDst}},
                               tempAlgParams.dataType, {}};
        if (isDmaRead) {
            CHK_RET(SendRecvReadReduce(info, templateResource.threads[0]));
        } else {
            CHK_RET(SendRecvBatchWriteReduce(info, templateResource.threads[0]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunNhrAllGather(const TemplateDataParams &tempAlgParams,
                           TemplateResource &templateResource,
                           EngineType engineType)
{
    (void)engineType;
    u32 rankSize = static_cast<u32>(tempAlgParams.allRankDispls.size());
    u32 myRankIdx = 0;
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    u32 nSteps = CalcNhrStepNum(rankSize);
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];

    for (u32 step = 0; step < nSteps; ++step) {
        u32 delta = 1 << (nSteps - 1 - step);
        u32 sendToIdx = (myRankIdx + delta) % rankSize;
        u32 recvFromIdx = (myRankIdx + rankSize - delta) % rankSize;
        auto [txIdxs, rxIdxs] = GenNhrAllGatherSliceIdxs(rankSize, myRankIdx, step, nSteps);

        const ChannelInfo &linkSend = templateResource.channels.at(sendToIdx)[0];
        const ChannelInfo &linkRecv = templateResource.channels.at(recvFromIdx)[0];

        std::vector<DataSlice> txSrc, txDst, rxSrc, rxDst;
        for (u32 i = 0; i < txIdxs.size(); ++i) {
            u64 sz = tempAlgParams.sliceSize;
            u64 txOff = tempAlgParams.buffInfo.hcclBuffBaseOff + tempAlgParams.sliceSize * txIdxs[i];
            u64 rxOff = tempAlgParams.buffInfo.hcclBuffBaseOff + tempAlgParams.sliceSize * rxIdxs[i];
            txSrc.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, txOff, sz, sz / dataTypeSize);
            txDst.emplace_back(linkSend.remoteCclMem.addr, txOff, sz, sz / dataTypeSize);
            rxSrc.emplace_back(linkRecv.remoteCclMem.addr, rxOff, sz, sz / dataTypeSize);
            rxDst.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, rxOff, sz, sz / dataTypeSize);
        }

        SendRecvInfo info{{linkSend, linkRecv}, {{txSrc, txDst}, {rxSrc, rxDst}}, tempAlgParams.dataType};
        if (isDmaRead) {
            CHK_RET(SendRecvRead(info, templateResource.threads[0]));
        } else {
            CHK_RET(SendRecvBatchWrite(info, templateResource.threads[0]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunNhrScatter(const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource,
                         EngineType engineType)
{
    (void)engineType;
    u32 rankSize = static_cast<u32>(tempAlgParams.allRankDispls.size());
    u32 myRankIdx = 0;
    u32 root = tempAlgParams.root;
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    u32 nSteps = CalcNhrStepNum(rankSize);
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    u32 myRel = (myRankIdx + rankSize - root) % rankSize;

    for (u32 step = 0; step < nSteps; ++step) {
        u32 delta = 1 << (nSteps - 1 - step);
        u32 sendToIdx = ((myRel + delta) % rankSize + root) % rankSize;
        u32 recvFromIdx = ((myRel + rankSize - delta) % rankSize + root) % rankSize;

        const ChannelInfo &linkSend = templateResource.channels.at(sendToIdx)[0];
        const ChannelInfo &linkRecv = templateResource.channels.at(recvFromIdx)[0];

        u64 sz = tempAlgParams.sliceSize;
        u64 off = tempAlgParams.buffInfo.hcclBuffBaseOff;
        std::vector<DataSlice> txSrc{{tempAlgParams.buffInfo.hcclBuff.addr, off, sz, sz / dataTypeSize}};
        std::vector<DataSlice> txDst{{linkSend.remoteCclMem.addr, off, sz, sz / dataTypeSize}};
        std::vector<DataSlice> rxSrc{{linkRecv.remoteCclMem.addr, off, sz, sz / dataTypeSize}};
        std::vector<DataSlice> rxDst{{tempAlgParams.buffInfo.hcclBuff.addr, off, sz, sz / dataTypeSize}};

        SendRecvInfo info{{linkSend, linkRecv}, {{txSrc, txDst}, {rxSrc, rxDst}}, tempAlgParams.dataType};
        if (isDmaRead) {
            CHK_RET(SendRecvRead(info, templateResource.threads[0]));
        } else {
            CHK_RET(SendRecvBatchWrite(info, templateResource.threads[0]));
        }
    }
    return HCCL_SUCCESS;
}

}
