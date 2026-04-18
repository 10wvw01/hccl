/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <sstream>
#include "securec.h"
#include "hccl/base.h"
#include "hccl_common.h"
#include "log.h"

namespace ops_hccl {

// 用于判断操作类型
enum OpType {
    OP_PRE_SYNC_INTER_THREADS = 1,
    OP_POST_SYNC_INTER_THREADS = 2,
    OP_LOCAL_COPY,
    OP_LOCAL_REDUCE,
    OP_SEND_RECV_WRITE,
    OP_SEND_WRITE,
    OP_RECV_WRITE,
    OP_SEND_RECV_WRITE_REDUCE,
    OP_SEND_WRITE_REDUCE,
    OP_RECV_WRITE_REDUCE,
    OP_SEND_RECV_READ,
    OP_SEND_READ,
    OP_RECV_READ,
    OP_SEND_RECV_READ_REDUCE,
    OP_SEND_READ_REDUCE,
    OP_RECV_READ_REDUCE,
    OP_GROUP_BROAD_CAST,
    OP_GROUP_REDUCE
};

// OMNI Slice信息结构
struct OmniSliceInfo {
    uint64_t sliceType;  // 0: input, 1: output, 2: cclbuf
    uint64_t sliceIdx;
    uint64_t remoteRank;
};

// OMNI信号信息结构
struct OmniSendRecvInfo {
    OpType optype;
    HcclDataType inputDataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceType;   // 0: sum, 1: max, 2: min
    uint64_t channelId;
    uint64_t sliceNum;
    std::vector<OmniSliceInfo> srcSliceInfo;
    std::vector<OmniSliceInfo> dstSliceInfo;
    uint64_t remoteRank;
};

struct OmniSyncInfo {
    OpType optype;
    uint64_t mainThreadIdx;
    uint64_t subThreadNum;
    std::vector<uint8_t> subThreadIds;
};

// OMNI通道信息结构
struct OmniChannelInfo {
    CommProtocol channelProtocol; ///< 通信协议
    uint64_t remoteRank;    ///< 远端rankId
    uint64_t channelId;
};

// 资源信息结构
struct ResInfo {
    uint32_t slaveThreadNum;
    uint32_t notifyNumOnMainThread;
    uint32_t notifyNumPerThread;
    uint32_t netLayerNum;
    std::vector<std::map<uint32_t, OmniChannelInfo>> mapchannelInfo; // netlayer<dstrankid, channelinfo>
};

// OMNI XML信息结构
struct XmlInfo {
    ResInfo resInfo; // calc计算需要的资源
    std::vector<OmniSyncInfo> vecSyncInfo; // kernel处理的数据通信
    std::vector<OmniSendRecvInfo> vecSendRecvInfo; // kernel处理的数据通信
};

HcclMem HcclMemRange(HcclMem inMem, u64 offset, u64 size);

static inline u64 RoundUpWithDivisor(u64 value, u64 divisor)
{
    if ((value == 0) || (divisor == 0)) {
        return divisor;
    }
    // divisor必须大于等于1, 返回value向上取divisor的整数倍的值
    return ((value + (divisor - 1)) / divisor) * divisor;
}

template <typename... Args> inline std::string StringFormat(const char *format, Args... args)
{
    using namespace std;
    constexpr size_t bufSize = BUFSIZ;
    char             buffer[bufSize];
    size_t           actualSize = snprintf_s(&buffer[0], bufSize, bufSize, format, args...);
    actualSize++;

    if (actualSize > bufSize) {
        std::vector<char> newbuffer(actualSize);
        snprintf_s(newbuffer.data(), actualSize, actualSize, format, args...);
        return newbuffer.data();
    }
    return buffer;
}
}

#endif