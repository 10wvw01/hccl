/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_COMMON_H
#define OPS_HCCL_COMMON_H

#include <vector>
#include <unordered_map>

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>
#include <hccl/hcomm_primitives.h>
#include <acl/acl_rt.h>

#include "binary_stream.h"

constexpr uint32_t NOTIFY_IDX_ACK = 0;
constexpr uint32_t NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr uint32_t CUSTOM_TIMEOUT = 1800;

constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH;
constexpr uint32_t INVALID_VALUE_RANKID = 0xFFFFFFFF;

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread;          ///< AICPU_TS通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< AICPU_TS通信引擎上的thread资源
    std::vector<ChannelInfo> channels; ///< AICPU_TS通信引擎上的channel资源

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> channels;
    }
};

struct OpParam {
    char tag[TAG_LENGTH];
    void *inputPtr = nullptr;
    void *outputPtr = nullptr;
    uint64_t count = 0;
    uint32_t root = 0;
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    ThreadHandle cpuThread;
    ThreadHandle cpuThreadOnAicpu;
    ThreadHandle aicpuThreadOnCpu; ///< AICPU_TS通信引擎上的thread资源
    void *resCtx = nullptr;        ///< 通信引擎上下文中的资源信息，存放 AlgResourceCtx 序列化后的内容
    uint64_t ctxSize = 0;
};

const std::unordered_map<HcclDataType, uint32_t> SIZE_TABLE = {{HCCL_DATA_TYPE_INT8, sizeof(int8_t)},
    {HCCL_DATA_TYPE_INT16, sizeof(int16_t)}, {HCCL_DATA_TYPE_INT32, sizeof(int32_t)}, {HCCL_DATA_TYPE_FP16, 2},
    {HCCL_DATA_TYPE_FP32, sizeof(float)}, {HCCL_DATA_TYPE_INT64, sizeof(int64_t)},
    {HCCL_DATA_TYPE_UINT64, sizeof(uint64_t)}, {HCCL_DATA_TYPE_UINT8, sizeof(uint8_t)},
    {HCCL_DATA_TYPE_UINT16, sizeof(uint16_t)}, {HCCL_DATA_TYPE_UINT32, sizeof(uint32_t)}, {HCCL_DATA_TYPE_FP64, 8},
    {HCCL_DATA_TYPE_BFP16, 2}, {HCCL_DATA_TYPE_INT128, 16}, {HCCL_DATA_TYPE_HIF8, 2}, {HCCL_DATA_TYPE_FP8E4M3, 1},
    {HCCL_DATA_TYPE_FP8E5M2, 1}, {HCCL_DATA_TYPE_FP8E8M0, 1}};

#endif // OPS_HCCL_COMMON_H
