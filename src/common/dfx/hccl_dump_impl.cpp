/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_dfx.h"

#include <algorithm>
#include <cstring>

#include "adapter_acl.h"
#include "log.h"

namespace {
constexpr uint32_t HCOMM_CHANNEL_MAGIC_WORD = 0x0fcf0f0fU;
constexpr uint32_t MAX_DUMP_ARRAY_NUM = 8U;
constexpr uint32_t MAX_DUMP_BYTES = 64U;

enum class CommEngineType : int32_t {
    COMM_ENGINE_RESERVED = -1,
    COMM_ENGINE_CPU = 0,
    COMM_ENGINE_CPU_TS = 1,
    COMM_ENGINE_AICPU = 2,
    COMM_ENGINE_AICPU_TS = 3,
    COMM_ENGINE_AIV = 4,
    COMM_ENGINE_CCU = 5,
};

enum class RegedBufferType : int32_t {
    REGED_BUFFER_INVALID = -1,
    REGED_BUFFER_IPC = 0,
    REGED_BUFFER_RMA = 1,
};

enum class SqContextType : int32_t {
    SQ_CONTEXT_TYPE_INVALID = -1,
    SQ_CONTEXT_TYPE_UB_JFS = 0,
    SQ_CONTEXT_TYPE_ROCE = 1,
};

enum class CqContextType : int32_t {
    CQ_CONTEXT_TYPE_INVALID = -1,
    CQ_CONTEXT_TYPE_UB_JFC = 0,
    CQ_CONTEXT_TYPE_ROCE = 1,
};

struct CommAbiHeader {
    uint32_t version;
    uint32_t magicWord;
    uint32_t size;
    uint32_t reserved;
};

struct ProtectionInfo {
    int32_t type;
    uint8_t memInfo[24];
};

struct SqContext {
    SqContextType type;
    union {
        struct {
            uint64_t sqVa;
            uint64_t headAddr;
            uint64_t tailAddr;
            uint64_t dbVa;
            uint32_t jfsID;
            uint32_t wqeSize;
            uint32_t sqDepth;
            uint32_t tpID;
            uint8_t remoteEID[16];
        } ubJfs;
        struct {
            uint64_t sqVa;
            uint64_t headAddr;
            uint64_t tailAddr;
            uint64_t dbVa;
            uint32_t qpn;
            uint32_t wqeSize;
            uint32_t depth;
            int8_t dbMode;
            uint8_t sl;
        } roceSq;
        uint8_t raws[120];
    } contextInfo;
};

struct CqContext {
    CqContextType type;
    union {
        struct {
            uint64_t scqVa;
            uint64_t headAddr;
            uint64_t tailAddr;
            uint64_t dbVa;
            uint32_t jfcID;
            uint32_t cqeSize;
            uint32_t cqDepth;
        } ubJfc;
        struct {
            uint64_t cqVa;
            uint64_t headAddr;
            uint64_t tailAddr;
            uint64_t dbVa;
            uint32_t cqn;
            uint32_t cqeSize;
            uint32_t cqDepth;
            int8_t dbMode;
        } roceCq;
        uint8_t raws[120];
    } contextInfo;
};

struct RegedBufferEntity {
    RegedBufferType type;
    union {
        struct {
            uint64_t addr;
            uint64_t size;
        } ipc;
        struct {
            uint64_t addr;
            uint64_t size;
            ProtectionInfo protectionInfo;
        } rma;
        uint8_t raws[56];
    } bufferInfo;
};

struct ChannelEntity {
    CommAbiHeader abiHeader;
    CommEngineType engine;
    int32_t protocol;
    uint32_t localNotifyNum;
    uint32_t remoteNotifyNum;
    uint32_t localBufferNum;
    uint32_t remoteBufferNum;
    uint32_t sqNum;
    uint32_t cqNum;
    uint64_t localNotifyAddr;
    uint64_t remoteNotifyAddr;
    uint64_t localBufferAddr;
    uint64_t remoteBufferAddr;
    uint64_t sqContextAddr;
    uint64_t cqContextAddr;
    uint32_t wqeCnt;
    uint8_t reserve[156];
};

static_assert(sizeof(ChannelEntity) == 256, "ChannelEntity size must keep aligned with hcomm");

template <typename T>
bool CopyFromDevice(uint64_t deviceAddr, T &hostValue, const char *name)
{
    if (deviceAddr == 0) {
        HCCL_RUN_WARNING("[HcommChannelInfoDump] %s addr is 0.", name);
        return false;
    }
    std::memset(&hostValue, 0, sizeof(T));
    HcclResult ret = ops_hccl::haclrtMemcpy(&hostValue, sizeof(T), reinterpret_cast<const void *>(deviceAddr),
        sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcommChannelInfoDump] copy %s failed, addr[0x%llx], size[%llu], ret[%d].",
            name, static_cast<unsigned long long>(deviceAddr),
            static_cast<unsigned long long>(sizeof(T)), ret);
        return false;
    }
    return true;
}

bool CopyBytesFromDevice(uint64_t deviceAddr, uint8_t *buffer, size_t size, const char *name)
{
    if (deviceAddr == 0) {
        HCCL_RUN_WARNING("[HcommChannelInfoDump] %s addr is 0.", name);
        return false;
    }
    HcclResult ret = ops_hccl::haclrtMemcpy(buffer, size, reinterpret_cast<const void *>(deviceAddr), size,
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcommChannelInfoDump] copy %s failed, addr[0x%llx], size[%llu], ret[%d].",
            name, static_cast<unsigned long long>(deviceAddr), static_cast<unsigned long long>(size), ret);
        return false;
    }
    return true;
}

uint64_t GetBufferAddr(const RegedBufferEntity &buffer)
{
    return buffer.type == RegedBufferType::REGED_BUFFER_IPC ? buffer.bufferInfo.ipc.addr : buffer.bufferInfo.rma.addr;
}

uint64_t GetBufferSize(const RegedBufferEntity &buffer)
{
    return buffer.type == RegedBufferType::REGED_BUFFER_IPC ? buffer.bufferInfo.ipc.size : buffer.bufferInfo.rma.size;
}

uint64_t GetSqVa(const SqContext &sqContext)
{
    if (sqContext.type == SqContextType::SQ_CONTEXT_TYPE_UB_JFS) {
        return sqContext.contextInfo.ubJfs.sqVa;
    }
    if (sqContext.type == SqContextType::SQ_CONTEXT_TYPE_ROCE) {
        return sqContext.contextInfo.roceSq.sqVa;
    }
    return 0;
}

uint32_t GetSqDepth(const SqContext &sqContext)
{
    if (sqContext.type == SqContextType::SQ_CONTEXT_TYPE_UB_JFS) {
        return sqContext.contextInfo.ubJfs.sqDepth;
    }
    if (sqContext.type == SqContextType::SQ_CONTEXT_TYPE_ROCE) {
        return sqContext.contextInfo.roceSq.depth;
    }
    return 0;
}

uint32_t GetSqEntrySize(const SqContext &sqContext)
{
    if (sqContext.type == SqContextType::SQ_CONTEXT_TYPE_UB_JFS) {
        return sqContext.contextInfo.ubJfs.wqeSize;
    }
    if (sqContext.type == SqContextType::SQ_CONTEXT_TYPE_ROCE) {
        return sqContext.contextInfo.roceSq.wqeSize;
    }
    return 0;
}

uint64_t GetCqVa(const CqContext &cqContext)
{
    if (cqContext.type == CqContextType::CQ_CONTEXT_TYPE_UB_JFC) {
        return cqContext.contextInfo.ubJfc.scqVa;
    }
    if (cqContext.type == CqContextType::CQ_CONTEXT_TYPE_ROCE) {
        return cqContext.contextInfo.roceCq.cqVa;
    }
    return 0;
}

uint32_t GetCqDepth(const CqContext &cqContext)
{
    if (cqContext.type == CqContextType::CQ_CONTEXT_TYPE_UB_JFC) {
        return cqContext.contextInfo.ubJfc.cqDepth;
    }
    if (cqContext.type == CqContextType::CQ_CONTEXT_TYPE_ROCE) {
        return cqContext.contextInfo.roceCq.cqDepth;
    }
    return 0;
}

uint32_t GetCqEntrySize(const CqContext &cqContext)
{
    if (cqContext.type == CqContextType::CQ_CONTEXT_TYPE_UB_JFC) {
        return cqContext.contextInfo.ubJfc.cqeSize;
    }
    if (cqContext.type == CqContextType::CQ_CONTEXT_TYPE_ROCE) {
        return cqContext.contextInfo.roceCq.cqeSize;
    }
    return 0;
}

void DumpBytes(const char *name, uint32_t idx, uint64_t addr)
{
    uint8_t data[MAX_DUMP_BYTES] = {0};
    if (!CopyBytesFromDevice(addr, data, sizeof(data), name)) {
        return;
    }
    HCCL_RUN_INFO("[HcommChannelInfoDump] %s[%u] va[0x%llx] first64 "
        "%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x "
        "%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x "
        "%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x "
        "%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x.",
        name, idx, static_cast<unsigned long long>(addr),
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
        data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
        data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
        data[40], data[41], data[42], data[43], data[44], data[45], data[46], data[47],
        data[48], data[49], data[50], data[51], data[52], data[53], data[54], data[55],
        data[56], data[57], data[58], data[59], data[60], data[61], data[62], data[63]);
}

void DumpBufferArray(const char *name, uint64_t deviceAddr, uint32_t bufferNum)
{
    if (bufferNum == 0) {
        return;
    }
    if (deviceAddr == 0) {
        HCCL_RUN_WARNING("[HcommChannelInfoDump] %s addr is 0, bufferNum[%u].", name, bufferNum);
        return;
    }
    uint32_t dumpNum = std::min(bufferNum, MAX_DUMP_ARRAY_NUM);
    for (uint32_t idx = 0; idx < dumpNum; ++idx) {
        RegedBufferEntity buffer;
        uint64_t addr = deviceAddr + static_cast<uint64_t>(idx) * sizeof(RegedBufferEntity);
        if (!CopyFromDevice(addr, buffer, name)) {
            continue;
        }
        HCCL_RUN_INFO("[HcommChannelInfoDump] %s[%u] type[%d] addr[0x%llx] size[%llu].",
            name, idx, static_cast<int32_t>(buffer.type),
            static_cast<unsigned long long>(GetBufferAddr(buffer)),
            static_cast<unsigned long long>(GetBufferSize(buffer)));
    }
    if (bufferNum > dumpNum) {
        HCCL_RUN_INFO("[HcommChannelInfoDump] %s total[%u], dumped[%u].", name, bufferNum, dumpNum);
    }
}

void DumpSqContext(uint64_t deviceAddr, uint32_t sqNum)
{
    if (sqNum == 0) {
        return;
    }
    if (deviceAddr == 0) {
        HCCL_RUN_WARNING("[HcommChannelInfoDump] SqContext addr is 0, sqNum[%u].", sqNum);
        return;
    }
    uint32_t dumpNum = std::min(sqNum, MAX_DUMP_ARRAY_NUM);
    for (uint32_t idx = 0; idx < dumpNum; ++idx) {
        SqContext sqContext;
        uint64_t addr = deviceAddr + static_cast<uint64_t>(idx) * sizeof(SqContext);
        if (!CopyFromDevice(addr, sqContext, "SqContext")) {
            continue;
        }
        uint64_t sqVa = GetSqVa(sqContext);
        HCCL_RUN_INFO("[HcommChannelInfoDump] SqContext[%u] type[%d] sqVa[0x%llx] depth[%u] entrySize[%u].",
            idx, static_cast<int32_t>(sqContext.type), static_cast<unsigned long long>(sqVa),
            GetSqDepth(sqContext), GetSqEntrySize(sqContext));
        if (sqVa != 0) {
            DumpBytes("sqVa", idx, sqVa);
        }
    }
    if (sqNum > dumpNum) {
        HCCL_RUN_INFO("[HcommChannelInfoDump] SqContext total[%u], dumped[%u].", sqNum, dumpNum);
    }
}

void DumpCqContext(uint64_t deviceAddr, uint32_t cqNum)
{
    if (cqNum == 0) {
        return;
    }
    if (deviceAddr == 0) {
        HCCL_RUN_WARNING("[HcommChannelInfoDump] CqContext addr is 0, cqNum[%u].", cqNum);
        return;
    }
    uint32_t dumpNum = std::min(cqNum, MAX_DUMP_ARRAY_NUM);
    for (uint32_t idx = 0; idx < dumpNum; ++idx) {
        CqContext cqContext;
        uint64_t addr = deviceAddr + static_cast<uint64_t>(idx) * sizeof(CqContext);
        if (!CopyFromDevice(addr, cqContext, "CqContext")) {
            continue;
        }
        uint64_t cqVa = GetCqVa(cqContext);
        HCCL_RUN_INFO("[HcommChannelInfoDump] CqContext[%u] type[%d] cqVa[0x%llx] depth[%u] entrySize[%u].",
            idx, static_cast<int32_t>(cqContext.type), static_cast<unsigned long long>(cqVa),
            GetCqDepth(cqContext), GetCqEntrySize(cqContext));
        if (cqVa != 0) {
            DumpBytes("scqVa", idx, cqVa);
        }
    }
    if (cqNum > dumpNum) {
        HCCL_RUN_INFO("[HcommChannelInfoDump] CqContext total[%u], dumped[%u].", cqNum, dumpNum);
    }
}

void DumpChannel(ChannelHandle channel)
{
    ChannelEntity channelEntity;
    if (!CopyFromDevice(channel, channelEntity, "ChannelEntity")) {
        return;
    }

    HCCL_RUN_INFO("[HcommChannelInfoDump] ChannelEntity magic[0x%x] version[%u] size[%u] engine[%d] protocol[%d] "
        "localBufferNum[%u] remoteBufferNum[%u] sqNum[%u] cqNum[%u] wqeCnt[%u].",
        channelEntity.abiHeader.magicWord, channelEntity.abiHeader.version, channelEntity.abiHeader.size,
        static_cast<int32_t>(channelEntity.engine), channelEntity.protocol, channelEntity.localBufferNum,
        channelEntity.remoteBufferNum, channelEntity.sqNum, channelEntity.cqNum, channelEntity.wqeCnt);

    if (channelEntity.abiHeader.magicWord != HCOMM_CHANNEL_MAGIC_WORD) {
        HCCL_RUN_WARNING("[HcommChannelInfoDump] unexpected ChannelEntity magic[0x%x], expected[0x%x].",
            channelEntity.abiHeader.magicWord, HCOMM_CHANNEL_MAGIC_WORD);
    }

    DumpBufferArray("localBuffer", channelEntity.localBufferAddr, channelEntity.localBufferNum);
    DumpBufferArray("remoteBuffer", channelEntity.remoteBufferAddr, channelEntity.remoteBufferNum);
    DumpSqContext(channelEntity.sqContextAddr, channelEntity.sqNum);
    DumpCqContext(channelEntity.cqContextAddr, channelEntity.cqNum);
}
} // namespace

extern "C" void HcommChannelInfoDump(uint32_t channelNum, ChannelHandle *channels)
{
    if (channelNum == 0) {
        HCCL_RUN_WARNING("[HcommChannelInfoDump] channelNum is 0, no channel to dump.");
        return;
    }
    if (channels == nullptr) {
        HCCL_ERROR("[HcommChannelInfoDump] channels is nullptr, channelNum[%u].", channelNum);
        return;
    }

    HCCL_RUN_INFO("[HcommChannelInfoDump] begin dump, channelNum[%u].", channelNum);
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        HCCL_RUN_INFO("[HcommChannelInfoDump] channel[%u] handle[0x%llx].",
            idx, static_cast<unsigned long long>(channels[idx]));
        if (channels[idx] == 0) {
            HCCL_RUN_WARNING("[HcommChannelInfoDump] channel[%u] handle is 0, skip.", idx);
            continue;
        }
        DumpChannel(channels[idx]);
    }
}
