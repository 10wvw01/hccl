/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mesh_primitives.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

HcclResult RunMeshAllGather(const TemplateDataParams &tempAlgParams,
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

    for (u32 i = 1; i < rankSize; ++i) {
        u32 peer = (myRankIdx + i) % rankSize;
        const ChannelInfo &link = templateResource.channels.at(peer)[0];

        u64 myOff = tempAlgParams.buffInfo.hcclBuffBaseOff + sz * myRankIdx;
        u64 peerOff = tempAlgParams.buffInfo.hcclBuffBaseOff + sz * peer;
        std::vector<DataSlice> txSrc{{tempAlgParams.buffInfo.hcclBuff.addr, myOff, sz, sz / dataTypeSize}};
        std::vector<DataSlice> txDst{{link.remoteCclMem.addr, myOff, sz, sz / dataTypeSize}};
        std::vector<DataSlice> rxSrc{{link.remoteCclMem.addr, peerOff, sz, sz / dataTypeSize}};
        std::vector<DataSlice> rxDst{{tempAlgParams.buffInfo.hcclBuff.addr, peerOff, sz, sz / dataTypeSize}};

        SendRecvInfo info{{link, link}, {{txSrc, txDst}, {rxSrc, rxDst}}, tempAlgParams.dataType};
        if (isDmaRead) {
            CHK_RET(SendRecvRead(info, templateResource.threads[i - 1]));
        } else {
            CHK_RET(SendRecvBatchWrite(info, templateResource.threads[i - 1]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshReduceScatter(const TemplateDataParams &tempAlgParams,
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

    for (u32 i = 1; i < rankSize; ++i) {
        u32 peer = (myRankIdx + i) % rankSize;
        const ChannelInfo &link = templateResource.channels.at(peer)[0];

        u64 peerOff = tempAlgParams.buffInfo.hcclBuffBaseOff + sz * peer;
        u64 myOff = tempAlgParams.buffInfo.hcclBuffBaseOff + sz * myRankIdx;
        std::vector<DataSlice> txSrc{{tempAlgParams.buffInfo.inputPtr, peerOff, sz, sz / dataTypeSize}};
        std::vector<DataSlice> txDst{{link.remoteCclMem.addr, myOff, sz, sz / dataTypeSize}};
        std::vector<DataSlice> rxSrc{{link.remoteCclMem.addr, myOff, sz, sz / dataTypeSize}};
        std::vector<DataSlice> rxDst{{tempAlgParams.buffInfo.hcclBuff.addr, myOff, sz, sz / dataTypeSize}};

        SendRecvReduceInfo info{{link, link}, {{txSrc, txDst}, {rxSrc, rxDst}}, tempAlgParams.dataType, {}};
        if (isDmaRead) {
            CHK_RET(SendRecvReadReduce(info, templateResource.threads[i - 1]));
        } else {
            CHK_RET(SendRecvBatchWriteReduce(info, templateResource.threads[i - 1]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshScatter(const TemplateDataParams &tempAlgParams,
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

    if (myRankIdx == root) {
        u32 t = 0;
        for (u32 r = 0; r < rankSize; ++r) {
            if (r == root) { continue; }
            const ChannelInfo &link = templateResource.channels.at(r)[0];
            u64 off = tempAlgParams.buffInfo.hcclBuffBaseOff + sz * r;
            std::vector<DataSlice> txSrc{{tempAlgParams.buffInfo.inputPtr, off, sz, sz / dataTypeSize}};
            std::vector<DataSlice> txDst{{link.remoteCclMem.addr, off, sz, sz / dataTypeSize}};
            std::vector<DataSlice> empty;
            SendRecvInfo info{{link, link}, {{txSrc, txDst}, {empty, empty}}, tempAlgParams.dataType};
            CHK_RET(isDmaRead ? SendRecvRead(info, templateResource.threads[t]) :
                                SendRecvBatchWrite(info, templateResource.threads[t]));
            t++;
        }
    } else {
        const ChannelInfo &link = templateResource.channels.at(root)[0];
        u64 off = tempAlgParams.buffInfo.hcclBuffBaseOff + sz * myRankIdx;
        std::vector<DataSlice> rxSrc{{link.remoteCclMem.addr, off, sz, sz / dataTypeSize}};
        std::vector<DataSlice> rxDst{{tempAlgParams.buffInfo.outputPtr, off, sz, sz / dataTypeSize}};
        std::vector<DataSlice> empty;
        SendRecvInfo info{{link, link}, {{empty, empty}, {rxSrc, rxDst}}, tempAlgParams.dataType};
        CHK_RET(isDmaRead ? SendRecvRead(info, templateResource.threads[0]) :
                            SendRecvBatchWrite(info, templateResource.threads[0]));
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshGather(const TemplateDataParams &tempAlgParams,
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

    if (myRankIdx == root) {
        u32 t = 0;
        for (u32 r = 0; r < rankSize; ++r) {
            if (r == root) { continue; }
            const ChannelInfo &link = templateResource.channels.at(r)[0];
            u64 off = tempAlgParams.buffInfo.hcclBuffBaseOff + sz * r;
            std::vector<DataSlice> rxSrc{{link.remoteCclMem.addr, off, sz, sz / dataTypeSize}};
            std::vector<DataSlice> rxDst{{tempAlgParams.buffInfo.outputPtr, off, sz, sz / dataTypeSize}};
            std::vector<DataSlice> empty;
            SendRecvInfo info{{link, link}, {{empty, empty}, {rxSrc, rxDst}}, tempAlgParams.dataType};
            CHK_RET(isDmaRead ? SendRecvRead(info, templateResource.threads[t]) :
                                SendRecvBatchWrite(info, templateResource.threads[t]));
            t++;
        }
    } else {
        const ChannelInfo &link = templateResource.channels.at(root)[0];
        u64 off = tempAlgParams.buffInfo.hcclBuffBaseOff + sz * myRankIdx;
        std::vector<DataSlice> txSrc{{tempAlgParams.buffInfo.inputPtr, off, sz, sz / dataTypeSize}};
        std::vector<DataSlice> txDst{{link.remoteCclMem.addr, off, sz, sz / dataTypeSize}};
        std::vector<DataSlice> empty;
        SendRecvInfo info{{link, link}, {{txSrc, txDst}, {empty, empty}}, tempAlgParams.dataType};
        CHK_RET(isDmaRead ? SendRecvRead(info, templateResource.threads[0]) :
                            SendRecvBatchWrite(info, templateResource.threads[0]));
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshAllToAll(const TemplateDataParams &tempAlgParams,
                           TemplateResource &templateResource, EngineType engineType)
{
    (void)engineType;
    u32 rankSize = static_cast<u32>(tempAlgParams.sendCounts.size());
    u32 myRankIdx = 0;
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];

    for (u32 i = 1; i < rankSize; ++i) {
        u32 peer = (myRankIdx + i) % rankSize;
        const ChannelInfo &link = templateResource.channels.at(peer)[0];

        u64 txSz = tempAlgParams.sendCounts[peer] * dataTypeSize;
        u64 txOff = tempAlgParams.sdispls[peer] * dataTypeSize;
        u64 rxSz = tempAlgParams.recvCounts[peer] * dataTypeSize;
        u64 rxOff = tempAlgParams.rdispls[peer] * dataTypeSize;

        std::vector<DataSlice> txSrc{{tempAlgParams.buffInfo.inputPtr, txOff, txSz, tempAlgParams.sendCounts[peer]}};
        std::vector<DataSlice> txDst{{link.remoteCclMem.addr, txOff, txSz, tempAlgParams.sendCounts[peer]}};
        std::vector<DataSlice> rxSrc{{link.remoteCclMem.addr, rxOff, rxSz, tempAlgParams.recvCounts[peer]}};
        std::vector<DataSlice> rxDst{{tempAlgParams.buffInfo.outputPtr, rxOff, rxSz, tempAlgParams.recvCounts[peer]}};

        SendRecvInfo info{{link, link}, {{txSrc, txDst}, {rxSrc, rxDst}}, tempAlgParams.dataType};
        if (isDmaRead) {
            CHK_RET(SendRecvRead(info, templateResource.threads[i - 1]));
        } else {
            CHK_RET(SendRecvBatchWrite(info, templateResource.threads[i - 1]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunMeshBarrier(const TemplateDataParams &tempAlgParams,
                          TemplateResource &templateResource, EngineType engineType)
{
    (void)engineType;
    u32 rankSize = static_cast<u32>(templateResource.channels.size()) + 1;
    u32 myRankIdx = 0;
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    bool isDmaRead = IsPcieProtocol(templateResource.channels);
    std::vector<DataSlice> empty;

    for (u32 i = 1; i < rankSize; ++i) {
        u32 peer = (myRankIdx + i) % rankSize;
        const ChannelInfo &link = templateResource.channels.at(peer)[0];
        SendRecvInfo info{{link, link}, {{empty, empty}, {empty, empty}}, tempAlgParams.dataType};
        if (isDmaRead) {
            CHK_RET(SendRecvRead(info, templateResource.threads[i - 1]));
        } else {
            CHK_RET(SendRecvBatchWrite(info, templateResource.threads[i - 1]));
        }
    }
    return HCCL_SUCCESS;
}

}
