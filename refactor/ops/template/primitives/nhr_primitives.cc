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

HcclResult RunNhrReduceScatter(const TemplateDataParams &tempAlgParams,
                               TemplateResource &templateResource, EngineType engineType)
{
    (void)engineType;
    u32 rankSize = static_cast<u32>(tempAlgParams.allRankDispls.size());
    u32 myRankIdx = 0;
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    u64 sz = tempAlgParams.sliceSize;
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }

    for (u32 step = 0; step < nSteps; ++step) {
        u32 delta = 1 << step;
        u32 sendToIdx = (myRankIdx + rankSize - delta) % rankSize;
        u32 recvFromIdx = (myRankIdx + delta) % rankSize;
        u32 deltaSliceIndex = 1 << (step + 1);
        u32 nSlices = (rankSize - 1 + delta) / deltaSliceIndex;
        u32 txSliceIdx = sendToIdx;
        u32 rxSliceIdx = myRankIdx;

        const ChannelInfo &linkSend = templateResource.channels.at(sendToIdx)[0];
        const ChannelInfo &linkRecv = templateResource.channels.at(recvFromIdx)[0];

        std::vector<DataSlice> txSrc, txDst, rxSrc, rxDst;
        for (u32 i = 0; i < nSlices; ++i) {
            u64 txOff = tempAlgParams.buffInfo.hcclBuffBaseOff + sz * txSliceIdx;
            u64 rxOff = tempAlgParams.buffInfo.hcclBuffBaseOff + sz * rxSliceIdx;
            txSrc.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, txOff, sz, sz / dataTypeSize);
            txDst.emplace_back(linkSend.remoteCclMem.addr, txOff, sz, sz / dataTypeSize);
            rxSrc.emplace_back(linkRecv.remoteCclMem.addr, rxOff, sz, sz / dataTypeSize);
            rxDst.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, rxOff, sz, sz / dataTypeSize);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
            rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
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
                           TemplateResource &templateResource, EngineType engineType)
{
    (void)engineType;
    u32 rankSize = static_cast<u32>(tempAlgParams.allRankDispls.size());
    u32 myRankIdx = 0;
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    u64 sz = tempAlgParams.sliceSize;
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }

    for (u32 step = 0; step < nSteps; ++step) {
        u32 delta = 1 << (nSteps - 1 - step);
        u32 sendToIdx = (myRankIdx + delta) % rankSize;
        u32 recvFromIdx = (myRankIdx + rankSize - delta) % rankSize;
        u32 deltaSliceIndex = 1 << (nSteps - step);
        u32 nSlices = (rankSize - 1 + delta) / deltaSliceIndex;
        u32 txSliceIdx = myRankIdx;
        u32 rxSliceIdx = (myRankIdx + rankSize - delta) % rankSize;

        const ChannelInfo &linkSend = templateResource.channels.at(sendToIdx)[0];
        const ChannelInfo &linkRecv = templateResource.channels.at(recvFromIdx)[0];

        std::vector<DataSlice> txSrc, txDst, rxSrc, rxDst;
        for (u32 i = 0; i < nSlices; ++i) {
            u64 txOff = tempAlgParams.buffInfo.hcclBuffBaseOff + sz * txSliceIdx;
            u64 rxOff = tempAlgParams.buffInfo.hcclBuffBaseOff + sz * rxSliceIdx;
            txSrc.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, txOff, sz, sz / dataTypeSize);
            txDst.emplace_back(linkSend.remoteCclMem.addr, txOff, sz, sz / dataTypeSize);
            rxSrc.emplace_back(linkRecv.remoteCclMem.addr, rxOff, sz, sz / dataTypeSize);
            rxDst.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, rxOff, sz, sz / dataTypeSize);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
            rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
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
                         TemplateResource &templateResource, EngineType engineType)
{
    (void)engineType;
    u32 rankSize = static_cast<u32>(tempAlgParams.allRankDispls.size());
    u32 myRankIdx = 0;
    u32 root = tempAlgParams.root;
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    u64 sz = tempAlgParams.sliceSize;
    u32 myRel = (myRankIdx + rankSize - root) % rankSize;
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }

    for (u32 step = 0; step < nSteps; ++step) {
        u32 delta = 1 << (nSteps - 1 - step);
        u32 sendToIdx = ((myRel + delta) % rankSize + root) % rankSize;
        u32 recvFromIdx = ((myRel + rankSize - delta) % rankSize + root) % rankSize;

        const ChannelInfo &linkSend = templateResource.channels.at(sendToIdx)[0];
        const ChannelInfo &linkRecv = templateResource.channels.at(recvFromIdx)[0];

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

HcclResult RunNhrBarrier(const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource, EngineType engineType)
{
    (void)engineType;
    u32 rankSize = static_cast<u32>(templateResource.channels.size()) + 1;
    u32 myRankIdx = 0;
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    std::vector<DataSlice> empty;

    for (u32 step = 0; step < nSteps; ++step) {
        u32 delta = 1 << step;
        u32 peer = (myRankIdx + delta) % rankSize;
        const ChannelInfo &link = templateResource.channels.at(peer)[0];
        SendRecvInfo info{{link, link}, {{empty, empty}, {empty, empty}}, tempAlgParams.dataType};
        if (isDmaRead) {
            CHK_RET(SendRecvRead(info, templateResource.threads[0]));
        } else {
            CHK_RET(SendRecvBatchWrite(info, templateResource.threads[0]));
        }
    }
    return HCCL_SUCCESS;
}

}
