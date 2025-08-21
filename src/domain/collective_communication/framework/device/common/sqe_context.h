/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SQE_AICPU_CONTEXT_H__
#define __SQE_AICPU_CONTEXT_H__

#include "common/aicpu_hccl_def.h"

constexpr uint32_t AC_SQE_MAX_CNT = 2048U; // 一次mc2算子一个流上最大下发sqe数量

struct SqeLocalRingBuffer {
    uint8_t localBuff[AC_SQE_SIZE * AC_SQE_MAX_CNT]; // local buffer
    uint8_t sqeType[AC_SQE_MAX_CNT];                 // 记录SQE类型,用于后续解析
    uint32_t addInfo[AC_SQE_MAX_CNT];                // 记录额外信息
    uint64_t profTimestap[AC_SQE_MAX_CNT] {0};       // profiling上报
    uint16_t tailSqeTaskId;                          // 最后一个sqe对应的taskId
    uint16_t tailSqeIdx;                             // 最后一个sqe对应的数组idx
    uint16_t sqeCnt;                                 // 当前轮保存的sqe数量(下发后重置)
    uint32_t sqHead;
    uint32_t sqTail;
    uint16_t filpNum;
};

struct SqeInfo {
    uint32_t addr1High = 0; // write value addr/sdma src addr/cond op addr
    uint32_t addr1Low = 0;
    uint32_t addr2High = 0; // sdma dst addr/cond op value addr
    uint32_t addr2Low = 0;
    uint32_t sqeHeadIdx = 0; // 当前sqe或当前sqe组的起始idx
    uint32_t notifyId = 0;
    uint32_t length = 0; // sdma length
    uint32_t partId = 0; // sdma
    uint32_t remoteRank = 0;
    uint32_t dataType = 0;
    uint16_t streamId = 0;
    uint16_t eventId = 0;
    uint16_t taskId = 0;
    uint16_t condValue = 0; // cond op value
    uint8_t isLast = 0;     // cond op isLast
    uint8_t opCode = 0;     // sdma 类型
    uint8_t sqeNum = 0;     // 当前sqe组长度 默认0表示单个sqe
    uint8_t type = 0;
    uint8_t subType = 0;
    uint8_t valid = 0; // 是否有效标记位
    union TaskRelatedType{
        uint8_t rdmaType; // rdma类型 是payload还是notify
        uint8_t linkType; // 链路类型 SIO
    } taskRelated;
    uint8_t reverse[9] = {0};
}; // 64B

struct SqeContext {
    SqeLocalRingBuffer *buffPtr; // SqeLocalRingBuffer[AC_MAX_RANK_NUM]
    int32_t clusterId;
};
namespace dfx {
class ProfilingManager;
}
class SqeContextUtils {
    friend class dfx::ProfilingManager;

public:
    static void InitSqeContext();
    static void SyncVariable();
    static void SaveVariable();
    static HcclResult GetNextSqeBufferAddr(uint32_t streamId, uint8_t *&sqeBufferAddr, uint8_t *&sqeTypeAddr,
        uint16_t &taskId);
    static HcclResult RecordAddInfo(uint32_t streamId, uint32_t addInfo);
    static HcclResult QuerySqeInfoByHead(uint32_t streamId, uint32_t sqHead, SqeInfo *info);
    static HcclResult QuerySqeInfoByTaskId(uint32_t streamId, uint16_t taskId, SqeInfo *info);
    static HcclResult ClearLocalBuff();
    static HcclResult ClearCurBuff(uint32_t streamid, uint32_t leftBound = 0);
    static HcclResult ModifyBuffer(uint32_t streamid);
    static std::string GetString(const SqeInfo &sqeInfo);
    static std::string RtsqTaskTypeToStr(uint8_t type);
    static HcclResult QuerySqeInfo(const uint8_t *sqeLocal, uint8_t sqeType, uint32_t addInfo, SqeInfo *info);
    static HcclResult AddFlipTask(uint32_t streamId);
};

extern SqeContext *GetSqeContext();

#endif // __MC2_AICPU_CONTEXT_HPP__