/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCCL_AIV_H
#define HCCL_AIV_H
 
#include <vector>
#include "string"
 
#include "hccl_types.h"
#include "runtime/kernel.h"
#include "hccl_common.h"
#include "common.h"
#include "mem_device_pub.h"
#include "alg_profiling.h"

namespace hccl {
constexpr u64 ATTR_POS_AIV_COMM_BUFFER = 0x00;
constexpr u64 ATTR_POS_AIV_COMM_INFO_BUFFER = 0x01;
constexpr u64 AIV_COMM_BUFFER_BITMASK = 0x01;
constexpr u64 AIV_COMM_INFO_BUFFER_BITMASK = 0x02;

constexpr u64 AIV_ALL_REDUCE_BIG_SIZE = 16 * 1024 * 1024;
constexpr u64 AIV_ALL_REDUCE_A3_ENTRY_SIZE = 1 * 1024 * 1024; // AllReduce单张卡数据量A3
constexpr u64 AIV_REDUCE_SCATTER_DETER_SMALL_SIZE = 1 * 1024 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_BIG_SIZE = 190 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_MID_SIZE = 2 * 1024 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_A3_ENTRY_SIZE = 1 * 1024 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_A3_GRAPH_ENTRY_SIZE = 4 * 1024 * 1024;
constexpr u64 AIV_ALL_GATHER_BIG_SIZE = 512 * 1024;
constexpr u64 AIV_ALL_GATHER_SMALL_SIZE = 700 * 1024;
constexpr u64 AIV_ALL_GATHER_A3_ENTRY_SIZE = 512 * 1024;
constexpr u64 AIV_ALL_GATHER_A3_GRAPH_ENTRY_SIZE = 4 * 1024 * 1024;
constexpr u64 AIV_ALL_TO_ALL_BIG_SIZE = 512 * 1024;
constexpr u64 AIV_ALL_TO_ALL_A3_ENTRY_SIZE = 512 * 1024;
constexpr u64 AIV_BIG_SIZE = 256 * 1024 * 1024;
constexpr u64 AIV_ALL_REDUCE_DETER_SIZE = 1 * 1024 * 1024; // AllReduce确定性计算

constexpr u64 AIV_A3_ALL_REDUCE_GRAPH_GUIYI_SIZE = 190 * 1024;
constexpr u64 AIV_A3_REDUCE_SCATTER_GRAPH_GUIYI_SIZE = 760 * 1024;
constexpr u64 AIV_A3_ALL_GATHER_GRAPH_GUIYI_SIZE = 760 * 1024;
constexpr u64 AIV_A3_ALL_TO_ALL_GRAPH_GUIYI_SIZE = 760 * 1024;

constexpr u64 AIV_REDUCE_SCATTER_A3_SMALL_RANKSIZE_ENTRY_SIZE = 1 * 1024 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_A3_MID_RANKSIZE_ENTRY_SIZE = 512 * 1024;
constexpr u64 AIV_REDUCE_SCATTER_A3_LARGE_RANKSIZE_ENTRY_SIZE = 128 * 1024;

constexpr u64 AIV_ALL_GATHER_A3_SMALL_RANKSIZE_ENTRY_SIZE = 1 * 1024 * 1024;
constexpr u64 AIV_ALL_GATHER_A3_MID_RANKSIZE_ENTRY_SIZE = 512 * 1024;
constexpr u64 AIV_ALL_GATHER_A3_LARGE_RANKSIZE_ENTRY_SIZE = 32 * 1024;

constexpr u64 AIV_A3_CROSSNODE_TINY_SIZE = 28 * 1024;
constexpr u64 AIV_A3_CROSSNODE_SMALL_SIZE = 112 * 1024;
constexpr u64 AIV_A3_CROSSNODE_MID_SIZE = 448 * 1024;

constexpr u32 MAX_RANK_SIZE = 16; // server内最大卡数
constexpr u32 MAX_RANK_SIZE_A3 = 768; // 超节点内最大卡数

constexpr u32 BLOCK_DIM_FACTOR_TWO = 2;
constexpr u32 BLOCK_DIM_FACTOR_THREE = 3;
constexpr u32 BLOCK_DIM_FACTOR_FOUR = 4;
constexpr u32 BLOCK_DIM_FACTOR_SIX = 6;
constexpr u32 BLOCK_DIM_FACTOR_EIGHT = 8;
constexpr u32 BLOCK_DIM_THREE_PER_RANK_A3 = 3;
constexpr u32 BLOCK_DIM_FOUR_PER_RANK_A3 = 4;
constexpr u32 MAX_BLOCK_DIM = 48;
constexpr u32 HALF_MAX_BLOCK_DIM = 24;
constexpr u32 ONE_THIRD_MAX_BLOCK_DIM = 16;
constexpr u32 ONE_FOURTH_MAX_BLOCK_DIM = 12;
constexpr u32 ONE_SIXTH_MAX_BLOCK_DIM = 8;
constexpr u32 ONE_EIGHTH_MAX_BLOCK_DIM = 6;

constexpr s32 TAG_INIT_VALUE = 1;
constexpr s32 TAG_RESET_COUNT = 1000;
constexpr s32 AIV_A2_ALL_REDUCE_RDMA_KERNEL_NUM = 2;

// 非均匀算子AlltoAllV/AlltoAllVC/AllGatherV/ReduceScatterV需要的额外参数信息，A2场景
using ExtraArgs = struct AlltoAllExtraArgs {
    u64 sendCountMatrix[MAX_RANK_SIZE * MAX_RANK_SIZE] = {};
    u64 sendCounts[MAX_RANK_SIZE] = {};
    u64 sendDispls[MAX_RANK_SIZE] = {};
    u64 recvCounts[MAX_RANK_SIZE] = {};
    u64 recvDispls[MAX_RANK_SIZE] = {};
    u64 maxCount = 0;
};

// 非均匀算子AlltoAllV/AlltoAllVC/AllGatherV/ReduceScatterV需要的额外参数信息，A3场景
struct ExtraArgsV2 {
    u64 sendCounts[MAX_RANK_SIZE_A3] = {};
    u64 sendDispls[MAX_RANK_SIZE_A3] = {};
    u64 recvCounts[MAX_RANK_SIZE_A3] = {};
    u64 recvDispls[MAX_RANK_SIZE_A3] = {};
};

// 表示算子属性的参数，相对固定
struct AivOpArgs {
    HcclCMDType cmdType;
    const void* input;
    const void* output; 
    u64 count;
    HcclDataType dataType;
    HcclReduceOp op;
    u32 root;
    bool isOpBase;
};

// 表示拓扑信息的参数
struct AivTopoArgs {
    u32 rank;
    u32 rankSize;
    u32 devId;
    u32 serverId;
    u32 serverNum;
    DevType devType;
    std::string identify;

    AivTopoArgs(u32 rank, u32 rankSize, u32 devId = MAX_RANK_SIZE, u32 serverId = 0, u32 serverNum = 1,
        DevType devType = DevType::DEV_TYPE_910B, std::string identify= "INVALID_COMM")
    : rank(rank), rankSize(rankSize), devId(devId), serverId(serverId), serverNum(serverNum), devType(devType), identify(identify)
    {
    }
};

// 表示AIV所需要的资源参数
struct AivResourceArgs {
    const std::string &commTag;
    rtStream_t stream;
    void** buffersIn; // 注册的CCLIN地址，所有卡可访问
    void** buffersOut; // 注册的CCLOUT地址，所有卡可访问
    u64 bufferSize;
    u32 blockDim;
    s32 aivTag;
};

// 表示AIV算法流程控制的参数
struct AivAlgArgs {
    s32 step;
    bool isSmallCount;
    u32 deterministic;

    explicit AivAlgArgs(s32 step = -1, bool isSmallCount = false, u32 deterministic = 0)
    : step(step), isSmallCount(isSmallCount), deterministic(deterministic)
    {
    }
};

// 表示AIVProfiling所需要的参数
struct AivProfilingInfo{
    uint64_t beginTime = 0;
    OpCounterInfo counter;
};

// 表示AIVSuperKernel所需要的参数
using AivSuperKernelArgs = struct AivSuperKernelArgsDef {
    void* buffersIn[MAX_RANK_SIZE] = {}; // 注册的CCLIN地址，所有卡可访问
    void* buffersOut[MAX_RANK_SIZE] = {}; // 注册的CCLOUT地址，所有卡可访问
    u64 rank;
    u64 rankSize;
    u64 len;
    u64 dataType;
    u64 unitSize;
    u64 reduceOp;
    u64 blockdim;
    s64 tag; // 第几次调用，定时重置成1
    s64 clearEnable;
 
    AivSuperKernelArgsDef(void** buffIn, void** buffOut, u32 rank,
        u32 rankSize, u64 len, u32 dataType, u32 unitSize, u32 reduceOp,u32 blockdim = 0, s32 tag = 0, bool clearEnable = true)
        : rank(rank), rankSize(rankSize), len(len), dataType(dataType), unitSize(unitSize), reduceOp(reduceOp), blockdim(blockdim),tag(tag), clearEnable(clearEnable)
    {
        for (u32 i = 0; i < MAX_RANK_SIZE; i++) {
            buffersIn[i] = (u8 *) buffIn[i];
            buffersOut[i] = (u8 *) buffOut[i];
        }
    }
    AivSuperKernelArgsDef() {}
};

HcclResult RegisterKernel(DevType deviceType);

HcclResult ClearAivSyncBuf(void** cclBuffersOut, rtStream_t stream, const AivTopoArgs &topoArgs);

inline s32 GetNextAivTag(s32 curTag, s32 tagIncre = 1) { return (curTag + tagIncre - 1) % TAG_RESET_COUNT + 1; }

HcclResult ExecuteKernelLaunchInner(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs, void* args, u32 argsSize, 
    AivProfilingInfo& aivProfilingInfo);

HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs, 
    AivProfilingInfo& aivProfilingInfo);

HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs, const ExtraArgs &extraArgs, 
    AivProfilingInfo& aivProfilingInfo);

HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs, const AivTopoArgs &topoArgs,
    const AivResourceArgs &resourceArgs, const AivAlgArgs &algArgs, const ExtraArgsV2 &extraArgs, 
    AivProfilingInfo& aivProfilingInfo);

HcclResult ReadBinFile(const std::string& fileName, std::string& buffer);

void SetAivProfilingInfoBeginTime(AivProfilingInfo& aivProfilingInfo);
void SetAivProfilingInfoBeginTime(uint64_t& oneTime);
}

#endif // HCCL_AIV_H