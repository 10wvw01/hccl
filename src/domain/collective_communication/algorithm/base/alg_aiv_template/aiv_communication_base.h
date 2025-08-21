/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_COMMUNICATION_BASE_H
#define AIV_COMMUNICATION_BASE_H

#include "kernel_operator.h"
#include "sync_interface.h"

using namespace AscendC;

constexpr uint32_t MAX_RANK_SIZE = 16; // server内最大卡数
constexpr uint32_t MAX_RANK_SIZE_A3 = 768; // 超节点内最大卡数
constexpr uint32_t MAX_TARGET_NUM = 20; // 最大轮数

struct ExtraArgs {
    uint64_t sendCountMatrix[MAX_RANK_SIZE * MAX_RANK_SIZE] = {};
    uint64_t sendCounts[MAX_RANK_SIZE] = {};
    uint64_t sendDispls[MAX_RANK_SIZE] = {};
    uint64_t recvCounts[MAX_RANK_SIZE] = {};
    uint64_t recvDispls[MAX_RANK_SIZE] = {};
    uint64_t maxCount = 0;
};

struct ExtraArgsV2 {
    uint64_t sendCounts[MAX_RANK_SIZE_A3] = {};
    uint64_t sendDispls[MAX_RANK_SIZE_A3] = {};
    uint64_t recvCounts[MAX_RANK_SIZE_A3] = {};
    uint64_t recvDispls[MAX_RANK_SIZE_A3] = {};
};

using AivSuperKernelArgs = struct AivSuperKernelArgsDef {
    GM_ADDR buffersIn[MAX_RANK_SIZE] = {}; // 注册的CCLIN地址，所有卡可访问
    GM_ADDR buffersOut[MAX_RANK_SIZE] = {}; // 注册的CCLOUT地址，所有卡可访问
    uint64_t rank;
    uint64_t rankSize;
    uint64_t len;
    uint64_t dataType;
    uint64_t unitSize;
    uint64_t reduceOp;
    int64_t blockdim;
    int64_t tag; // 第几次调用，定时重置成1
    int64_t clearEnable;
};
enum class AivNotifyType {
    ACK,
    DataSignal,
    Done
};

enum class CommPattern {
    //server间
    interRank,
    //server内
    intraRank
};

#define KERNEL_ARGS_DEF \
GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffIn2, GM_ADDR buffIn3, \
GM_ADDR buffIn4, GM_ADDR buffIn5, GM_ADDR buffIn6, GM_ADDR buffIn7, \
GM_ADDR buffIn8, GM_ADDR buffIn9, GM_ADDR buffIn10, GM_ADDR buffIn11, \
GM_ADDR buffIn12, GM_ADDR buffIn13, GM_ADDR buffIn14, GM_ADDR buffIn15, \
GM_ADDR buffOut0, GM_ADDR buffOut1, GM_ADDR buffOut2, GM_ADDR buffOut3, \
GM_ADDR buffOut4, GM_ADDR buffOut5, GM_ADDR buffOut6, GM_ADDR buffOut7, \
GM_ADDR buffOut8, GM_ADDR buffOut9, GM_ADDR buffOut10, GM_ADDR buffOut11, \
GM_ADDR buffOut12, GM_ADDR buffOut13, GM_ADDR buffOut14, GM_ADDR buffOut15, \
GM_ADDR input, GM_ADDR output, uint32_t rank, uint32_t rankSize, uint64_t len, \
uint32_t dataType, uint32_t reduceOp, uint32_t root, int32_t tag, bool isOpBase, uint64_t bufferSize, \
int32_t aivRdmaStep, bool useAivRdmaSmall, int32_t serverNum, uint32_t devType, GM_ADDR headCountMem, \
GM_ADDR tailCountMem, GM_ADDR addOneMem, uint32_t counterMemSize, bool isEnableCounter, uint32_t deterministic

#define KERNEL_ARGS_CALL \
buffIn0, buffIn1, buffIn2, buffIn3, buffIn4, buffIn5, buffIn6, buffIn7, \
buffIn8, buffIn9, buffIn10, buffIn11, buffIn12, buffIn13, buffIn14, buffIn15, \
buffOut0, buffOut1, buffOut2, buffOut3, buffOut4, buffOut5, buffOut6, buffOut7, \
buffOut8, buffOut9, buffOut10, buffOut11, buffOut12, buffOut13, buffOut14, buffOut15, \
input, output, rank, rankSize, len, dataType, reduceOp, root, tag, isOpBase, bufferSize, aivRdmaStep, useAivRdmaSmall, \
serverNum, devType, headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter, deterministic

#define KERNEL_CLASS_INIT \
buffIn0, buffIn1, buffIn2, buffIn3, buffIn4, buffIn5, buffIn6, buffIn7, \
buffIn8, buffIn9, buffIn10, buffIn11, buffIn12, buffIn13, buffIn14, buffIn15, \
buffOut0, buffOut1, buffOut2, buffOut3, buffOut4, buffOut5, buffOut6, buffOut7, \
buffOut8, buffOut9, buffOut10, buffOut11, buffOut12, buffOut13, buffOut14, buffOut15, \
rank, rankSize, dataType, reduceOp, root, headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter

#define EXTERN_KERNEL_ARGS_DEF \
KERNEL_ARGS_DEF, ExtraArgs extraArgs

#define EXTERN_KERNEL_ARGS_DEF_V2 \
KERNEL_ARGS_DEF, ExtraArgsV2 extraArgs

#define EXTERN_KERNEL_ARGS_CALL \
KERNEL_ARGS_CALL, extraArgs

#define SUPERKERNEL_LITE_ARGS_DEF \
uint64_t args_offset
 
#define SUPERKERNEL_LITE_ARGS_EXTRACT \
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();\
    GM_ADDR hiddenInput = param_base[args_offset++];\
    GM_ADDR input = param_base[args_offset++];\
    GM_ADDR output = param_base[args_offset++]

#define SUPERKERNEL_ARGS_DEF \
GM_ADDR hiddenInput, GM_ADDR input, GM_ADDR output
 
#define SUPERKERNEL_ARGS_CALL \
hiddenInput, input, output
 
#define SUPERKERNEL_CLASS_INIT \
hiddenInput

constexpr uint64_t AIV_FLAG_BUFFER_SIZE = 3 * 1024 * 1024; // aiv算子的flag区域大小
constexpr uint64_t CLEAR_BUFFER_OFFSET = 1024 * 1024; // 用于清空的aiv buffer的偏移
constexpr uint64_t SYNC_BUFFER_OFFSET = 2 * 1024 * 1024; // 用于sync的aiv buffer的偏移
constexpr uint64_t BUFFER_AREA = 1024 * 1024; // aiv算子的单独功能flag区域大小
constexpr uint64_t GM_TMP_ARGS_OFFSET = 64 * 1024;

constexpr uint64_t AIV_ALL_REDUCE_BIG_SIZE = 16 * 1024 * 1024;
constexpr uint64_t AIV_ALL_REDUCE_SMALL_SIZE = 64 * 1024;
constexpr uint64_t AIV_INIT_OFFSET = 0;
constexpr uint64_t AIV_PING_PONG_SIZE = 16 * 1024 * 1024;
constexpr uint64_t AIV_PING_PONG_FACTOR_TWO = 2;

constexpr uint64_t AIV_ALL_GATHER_SMALL_SIZE = 700 * 1024;
constexpr uint64_t AIV_REDUCE_SCATTER_MID_SIZE = 2 * 1024 * 1024;
constexpr uint64_t AIV_REDUCE_SCATTER_V_MID_SIZE = 2 * 1024 * 1024;
constexpr uint64_t AIV_ALL_TO_ALL_BIG_SIZE = 512 * 1024;

constexpr uint64_t AIV_A3_ALL_REDUCE_GRAPH_GUIYI_SIZE = 190 * 1024;
constexpr uint64_t AIV_A3_REDUCE_SCATTER_GRAPH_GUIYI_SIZE = 760 * 1024;
constexpr uint64_t AIV_A3_ALL_GATHER_GRAPH_GUIYI_SIZE = 760 * 1024;
constexpr uint64_t AIV_A3_ALL_TO_ALL_GRAPH_GUIYI_SIZE = 760 * 1024;
constexpr uint64_t AIV_ALL_REDUCE_DETER_MID_SIZE = 1 * 1024 * 1024;
constexpr uint64_t AIV_REDUCE_SCATTER_DETER_SMALL_SIZE = 1 * 1024 * 1024;
constexpr uint32_t AIV_A3_CROSSNODE_TINY_SIZE = 28 * 1024;
constexpr uint32_t AIV_A3_CROSSNODE_SMALL_SIZE = 112 * 1024;
constexpr uint32_t AIV_A3_CROSSNODE_MID_SIZE = 448 * 1024;
constexpr uint32_t BLOCK_DIM_THREE_PER_RANK_A3 = 3;
constexpr uint32_t BLOCK_DIM_FOUR_PER_RANK_A3 = 4;
constexpr uint32_t MAX_BLOCK_DIM = 48;

constexpr uint32_t TAG_MOVE_LEFT_BITS = 15;

constexpr uint64_t UB_ALIGN_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE_4 = UB_FLAG_SIZE * 4;
constexpr uint64_t UB_FLAG_SIZE_8 = UB_FLAG_SIZE * 8;
constexpr uint64_t UB_MAX_DATA_SIZE = 190 * 1024;
constexpr uint64_t UB_DB_DATA_BATCH_SIZE = UB_MAX_DATA_SIZE / 2;
constexpr uint32_t MaxBufferSize = 200 * 1024 * 1024;

constexpr uint64_t FLAG_SIZE = 32;
constexpr uint64_t ATOMIC_FLAG_SIZE = 512;
constexpr uint64_t FLAG_INTERVAL = FLAG_SIZE * 2;
constexpr uint64_t FLAG_ONE_OFFSET = 0;
constexpr uint64_t FLAG_TWO_OFFSET = FLAG_SIZE;
constexpr uint64_t FLAG_THREE_OFFSET = FLAG_SIZE * 2;
constexpr uint64_t FLAG_FOUR_OFFSET = FLAG_SIZE * 3;
constexpr uint32_t HALF_MAX_BLOCK_DIM = 24;
constexpr uint32_t ONE_THIRD_MAX_BLOCK_DIM = 16;
constexpr uint32_t ONE_FOURTH_MAX_BLOCK_DIM = 12;
constexpr uint32_t ONE_SIXTH_MAX_BLOCK_DIM = 8;
constexpr uint32_t ONE_EIGHTH_MAX_BLOCK_DIM = 6;
constexpr uint64_t DETERMINISTIC_RANKSIZE = 4;

constexpr uint64_t IDX_0 = 0;
constexpr uint64_t IDX_1 = 1;
constexpr uint64_t IDX_2 = 2;
constexpr uint64_t IDX_3 = 3;
constexpr uint64_t IDX_4 = 4;
constexpr uint64_t IDX_5 = 5;
constexpr uint64_t IDX_6 = 6;
constexpr uint64_t IDX_7 = 7;
constexpr uint64_t IDX_8 = 8;
constexpr uint64_t IDX_9 = 9;
constexpr uint64_t IDX_10 = 10;
constexpr uint64_t IDX_11 = 11;
constexpr uint64_t IDX_12 = 12;
constexpr uint64_t IDX_13 = 13;
constexpr uint64_t IDX_14 = 14;
constexpr uint64_t IDX_15 = 15;

constexpr uint64_t DOUBLE = 2;
constexpr uint64_t FLAG_BUF_NUM = 3;

// 当前每个kernel最多使用4组同步标记，这里预留6组
constexpr uint32_t MAX_FLAG_SIZE_PER_KERNEL = 6 * MAX_RANK_SIZE * FLAG_SIZE;

// 将__COUNTER__改为固定偏移，新执行器需添加新偏移
#define AIV_ALL_REDUCE_DETER_910B_SMALLDATA 0
#define AIV_REDUCE_SCATTER_DETER_910B_SMALLDATA 1
#define AIV_ALL_REDUCE_DETER_910B_MIDDATA 2
#define AIV_REDUCE_SCATTER_DETER_910B_MIDDATA 3
#define AIV_ALL_REDUCE_DETER_910B_BIGDATA 4
#define AIV_REDUCE_SCATTER_DETER_910B_BIGDATA 5


#define BASE_FLAG_OFFSET (MAX_FLAG_SIZE_PER_KERNEL)

#define DEV_TYPE_910_93 4

class AivCommBase {
public:
    __aicore__ inline AivCommBase() {}

    __aicore__ inline void Init(GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffIn2, GM_ADDR buffIn3, GM_ADDR buffIn4,
                                GM_ADDR buffIn5, GM_ADDR buffIn6, GM_ADDR buffIn7, GM_ADDR buffIn8, GM_ADDR buffIn9,
                                GM_ADDR buffIn10, GM_ADDR buffIn11, GM_ADDR buffIn12, GM_ADDR buffIn13,
                                GM_ADDR buffIn14, GM_ADDR buffIn15, GM_ADDR buffOut0, GM_ADDR buffOut1,
                                GM_ADDR buffOut2, GM_ADDR buffOut3, GM_ADDR buffOut4, GM_ADDR buffOut5,
                                GM_ADDR buffOut6, GM_ADDR buffOut7, GM_ADDR buffOut8, GM_ADDR buffOut9,
                                GM_ADDR buffOut10, GM_ADDR buffOut11, GM_ADDR buffOut12, GM_ADDR buffOut13,
                                GM_ADDR buffOut14, GM_ADDR buffOut15, uint32_t rank, uint32_t rankSize,
                                uint32_t dataType, uint32_t reduceOp, uint32_t root, GM_ADDR headCountMem,
                                GM_ADDR tailCountMem, GM_ADDR addOneMem, uint32_t counterMemSize, bool isEnableCounter,
                                bool useDoubleBuffer)
    {
        InitBuffArray(buffIn0, buffIn1, buffIn2, buffIn3, buffIn4,
                buffIn5, buffIn6, buffIn7, buffIn8, buffIn9,
                buffIn10, buffIn11, buffIn12, buffIn13,
                buffIn14, buffIn15, buffOut0, buffOut1,
                buffOut2, buffOut3, buffOut4, buffOut5,
                buffOut6, buffOut7, buffOut8, buffOut9,
                buffOut10, buffOut11, buffOut12, buffOut13,
                buffOut14, buffOut15);

        rank_ = rank;
        rankSize_ = rankSize;
        reduceOp_ = reduceOp;

        useDoubleBuffer_ = useDoubleBuffer;
        blockdim_ = block_num;

        localOffset = (rankSize_ * BLOCK_DIM_FOUR_PER_RANK_A3 * FLAG_BUF_NUM) * FLAG_SIZE;
        multiOffset = MAX_BLOCK_DIM * DOUBLE * FLAG_SIZE+ localOffset;
        pingpongOffset = multiOffset + DOUBLE * DOUBLE * BLOCK_DIM_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE * DOUBLE;
        countOffset = DOUBLE * pingpongOffset;
        seperateOffset = countOffset + BLOCK_DIM_FOUR_PER_RANK_A3 * rankSize_ * FLAG_SIZE;

        pipe.InitBuffer(localFlagBuf, UB_FLAG_SIZE_4);
        localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_ONE_OFFSET);
        localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_TWO_OFFSET);
        localCheckGETensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_THREE_OFFSET);
        localGetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FOUR_OFFSET);

        pipe.InitBuffer(flagBatchSetQue, 1, UB_FLAG_SIZE_8); // 最多支持同时set8个flag值，256B可存放32个u64，最多2组16rank
        pipe.InitBuffer(flagBatchCheckQue, 1, UB_FLAG_SIZE_8); // 最多支持同时check8个flag值

        if (useDoubleBuffer) {
            pipe.InitBuffer(inOutQue, DOUBLE, UB_DB_DATA_BATCH_SIZE); // double buffer
        } else {
            pipe.InitBuffer(inOutQue, 1, UB_MAX_DATA_SIZE);
        }

        pipe.InitBuffer(flagInQue, AIV_PING_PONG_FACTOR_TWO, UB_FLAG_SIZE);
        InitOpCounter(headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter);
    }

    __aicore__ inline void Init(GM_ADDR hiddenInput, uint64_t threshold)
    {
        __gm__ AivSuperKernelArgs* args = reinterpret_cast<__gm__ AivSuperKernelArgs*>(hiddenInput);
        
        for (int32_t i = 0; i < MAX_RANK_SIZE; i++) {
           GM_IN[i] = args->buffersIn[i];
           GM_OUT[i] = args->buffersOut[i];
        }
        rank_ = args->rank;
        rankSize_ = args->rankSize;
        reduceOp_ = args->reduceOp;
        len_ = args->len;
        tag_ = args->tag;
        dataType_ = args->dataType;
        blockdim_ = args->blockdim;
 
        pipe.InitBuffer(localFlagBuf, UB_FLAG_SIZE_4);
        localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_ONE_OFFSET);
        localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_TWO_OFFSET);
        localCheckGETensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_THREE_OFFSET);
        localGetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FOUR_OFFSET);

        localOffset = (rankSize_ * BLOCK_DIM_FOUR_PER_RANK_A3 * FLAG_BUF_NUM) * FLAG_SIZE;
        multiOffset = MAX_BLOCK_DIM * DOUBLE * FLAG_SIZE+ localOffset;
        pingpongOffset = multiOffset + DOUBLE * DOUBLE * BLOCK_DIM_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE * DOUBLE;
        countOffset = DOUBLE * pingpongOffset;
        seperateOffset = countOffset + BLOCK_DIM_FOUR_PER_RANK_A3 * rankSize_ * FLAG_SIZE;
        
        if (len_ * (args->unitSize) > threshold) {
            useDoubleBuffer_ = true;
            pipe.InitBuffer(inOutQue, DOUBLE, UB_DB_DATA_BATCH_SIZE); // double buffer
        } else {
            useDoubleBuffer_ = false;
            pipe.InitBuffer(inOutQue, 1, UB_MAX_DATA_SIZE);
        }
 
        if (args->clearEnable == 1) {
            ClearSyncBuf();
        }
    }

    __aicore__ inline void InitBuffArray(GM_ADDR buffIn0, GM_ADDR buffIn1, GM_ADDR buffIn2, GM_ADDR buffIn3, GM_ADDR buffIn4,
                                GM_ADDR buffIn5, GM_ADDR buffIn6, GM_ADDR buffIn7, GM_ADDR buffIn8, GM_ADDR buffIn9,
                                GM_ADDR buffIn10, GM_ADDR buffIn11, GM_ADDR buffIn12, GM_ADDR buffIn13,
                                GM_ADDR buffIn14, GM_ADDR buffIn15, GM_ADDR buffOut0, GM_ADDR buffOut1,
                                GM_ADDR buffOut2, GM_ADDR buffOut3, GM_ADDR buffOut4, GM_ADDR buffOut5,
                                GM_ADDR buffOut6, GM_ADDR buffOut7, GM_ADDR buffOut8, GM_ADDR buffOut9,
                                GM_ADDR buffOut10, GM_ADDR buffOut11, GM_ADDR buffOut12, GM_ADDR buffOut13,
                                GM_ADDR buffOut14, GM_ADDR buffOut15)
    {
        GM_IN[IDX_0] = buffIn0;
        GM_IN[IDX_1] = buffIn1;
        GM_IN[IDX_2] = buffIn2;
        GM_IN[IDX_3] = buffIn3;
        GM_IN[IDX_4] = buffIn4;
        GM_IN[IDX_5] = buffIn5;
        GM_IN[IDX_6] = buffIn6;
        GM_IN[IDX_7] = buffIn7;
        GM_IN[IDX_8] = buffIn8;
        GM_IN[IDX_9] = buffIn9;
        GM_IN[IDX_10] = buffIn10;
        GM_IN[IDX_11] = buffIn11;
        GM_IN[IDX_12] = buffIn12;
        GM_IN[IDX_13] = buffIn13;
        GM_IN[IDX_14] = buffIn14;
        GM_IN[IDX_15] = buffIn15;

        GM_OUT[IDX_0] = buffOut0;
        GM_OUT[IDX_1] = buffOut1;
        GM_OUT[IDX_2] = buffOut2;
        GM_OUT[IDX_3] = buffOut3;
        GM_OUT[IDX_4] = buffOut4;
        GM_OUT[IDX_5] = buffOut5;
        GM_OUT[IDX_6] = buffOut6;
        GM_OUT[IDX_7] = buffOut7;
        GM_OUT[IDX_8] = buffOut8;
        GM_OUT[IDX_9] = buffOut9;
        GM_OUT[IDX_10] = buffOut10;
        GM_OUT[IDX_11] = buffOut11;
        GM_OUT[IDX_12] = buffOut12;
        GM_OUT[IDX_13] = buffOut13;
        GM_OUT[IDX_14] = buffOut14;
        GM_OUT[IDX_15] = buffOut15;
    }

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag) {}

    __aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b);

    __aicore__ inline uint64_t CalActualCount(uint32_t sliceIdx, uint64_t sliceCount, uint64_t avgLengthPerSlice,
        uint64_t tailLength);

    __aicore__ inline void CalBlockCountAndOffset(uint64_t len, uint32_t blockNumPerGroup, uint32_t blockIdxInGroup,
        uint32_t padCount, uint64_t &count, uint64_t &blockOffset);

    template<typename T>
    __aicore__ inline void SetAtomicOp(uint32_t atomicOp);

    template<typename T>
    __aicore__ inline void DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
        const uint32_t calCount);

    template<typename T>
    __aicore__ inline void DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
        const uint32_t calCount);

    template<typename T>
    __aicore__ inline void CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count, bool atomic = false,
        uint32_t atomicOp = 0);

    template<typename T>
    __aicore__ inline void CpGM2GMWithFlagWrap(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count,
        int32_t rank, uint64_t flushFrequency = 8, int32_t tag = 0);

    template<typename T>
    __aicore__ inline void CpGM2GMWithFlagWrap(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count,
        __gm__ int32_t* ctrlFlagGM, uint64_t flushFrequency = 8, int32_t tag = 0);

    __aicore__ inline void Record(uint32_t tag, int32_t waitRank, AivNotifyType notifyType, int32_t blockGroup = 0, bool ifPingpong = false);

    __aicore__ inline void LocalRecord(uint32_t tag, int32_t blockIdx, AivNotifyType notifyType, bool ifPingpong = false);

    __aicore__ inline void Record1vN(uint32_t tag, CommPattern pattern, AivNotifyType notifyType = AivNotifyType::ACK, int32_t block = 0, 
        bool ifPingpong = false);

    __aicore__ inline void RecordNv1(uint32_t tag, int32_t waitRank, AivNotifyType notifyType = AivNotifyType::ACK, int32_t block = 0,
        bool ifPingpong = false);

    __aicore__ inline void CountRecord(uint32_t tag, int64_t count, int32_t waitRank);

    __aicore__ inline void Wait(uint32_t tag, int32_t recordRank, AivNotifyType notifyType, int32_t blockGroup = 0, bool ifpingpong = false);

    __aicore__ inline void LocalWait(uint32_t tag, int32_t blockIdx, AivNotifyType notifyType, bool ifpingpong = false);
    
    __aicore__ inline void WaitNv1(uint32_t tag, int32_t recordRank, AivNotifyType notifyType = AivNotifyType::ACK,
        int32_t block = 0, bool ifPingpong = false);

    __aicore__ inline void Wait1vN(uint32_t tag, CommPattern pattern, bool ifClear = true, AivNotifyType notifyType = AivNotifyType::ACK,
        int32_t block = 0, bool ifPingpong = false);

    __aicore__ inline int32_t CountWait(int32_t recordRank, int32_t index);

    __aicore__ inline void Barrier(uint32_t step);
 
    __aicore__ inline void ClearFlag();
 
    __aicore__ inline void BlockSync();
 
    __aicore__ inline void ClearSyncBuf();

    __aicore__ inline void InitOpCounter(GM_ADDR headCountMem, GM_ADDR tailCountMem, GM_ADDR addOneMem, uint32_t counterMemSize,
        bool isEnableCounter)
    {
        headCountMem_ = headCountMem;
        tailCountMem_ = tailCountMem;
        addOneMem_ = addOneMem;
        counterMemSize_ = counterMemSize;
        isEnableCounter_ = isEnableCounter;
    }

    __aicore__ inline void HeadCounter()
    {
        if (block_idx == 0 && isEnableCounter_) {
            CpGM2GM((__gm__ int32_t*)headCountMem_, (__gm__ int32_t*)addOneMem_, counterMemSize_ / sizeof(int32_t), true,
                HcclReduceOp::HCCL_REDUCE_SUM);
        }
    }

    __aicore__ inline void TailCounter()
    {
        if (block_idx == 0 && isEnableCounter_) {
            CpGM2GM((__gm__ int32_t*)tailCountMem_, (__gm__ int32_t*)addOneMem_, counterMemSize_ / sizeof(int32_t), true,
                HcclReduceOp::HCCL_REDUCE_SUM);
        }
    }
//protected:
    GM_ADDR GM_IN[MAX_RANK_SIZE];
    GM_ADDR GM_OUT[MAX_RANK_SIZE];

    uint32_t rank_;
    uint32_t rankSize_;
    uint32_t reduceOp_;
    uint32_t dataType_;
 
    uint64_t len_;
    int32_t tag_;
    int32_t blockdim_;

    bool useDoubleBuffer_;

    TPipe pipe;
    TBuf<> localFlagBuf;
    LocalTensor<int32_t> localSetTensor;
    LocalTensor<int32_t> localCheckTensor;
    LocalTensor<int32_t> localCheckGETensor;
    LocalTensor<int32_t> localGetTensor;

    TQue<QuePosition::VECOUT, 1> flagBatchSetQue;
    TQue<QuePosition::VECIN, 1> flagBatchCheckQue;

    TQue<QuePosition::VECIN, 1> flagInQue;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutQue;

    GM_ADDR headCountMem_;
    GM_ADDR tailCountMem_;
    GM_ADDR addOneMem_;
    uint32_t counterMemSize_;
    bool isEnableCounter_;

    uint32_t localOffset;
    uint32_t multiOffset;
    uint32_t pingpongOffset;
    uint32_t countOffset;
    uint32_t seperateOffset;
};

__aicore__ inline void AivCommBase::Barrier(uint32_t step)
{
    // 用10个flag
    uint32_t flagOffset = 2 * 1024 * 1024 + (step % 2) * FLAG_SIZE * rankSize_;
    __gm__ int32_t *ctrlFlagsGM;
    if (GetBlockIdx() == 0) {
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < rankSize_; i++) {
            uint32_t targetRank = (rank_ + i) % rankSize_; 
            ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[targetRank] + flagOffset + rank_ * FLAG_SIZE);
            SetSignalValue(ctrlFlagsGM, localSetTensor, 1);
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < rankSize_; i++) {
            uint32_t targetRank = (rank_ + i) % rankSize_; 
            ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + targetRank * FLAG_SIZE);
            WaitSignalValue(ctrlFlagsGM, localCheckTensor, 1);
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < rankSize_; i++) {
            uint32_t targetRank = (rank_ + i) % rankSize_; 
            ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffset + targetRank * FLAG_SIZE);
            SetSignalValue(ctrlFlagsGM, localSetTensor, 0);
        }
    }
}
 
__aicore__ inline void AivCommBase::ClearFlag()
{
    // 用10个flag
    __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_]);
    __gm__ int32_t *emtpyGM = (__gm__ int32_t *)(GM_OUT[rank_] + CLEAR_BUFFER_OFFSET);
    CpGM2GM(ctrlFlagsGM, emtpyGM, BUFFER_AREA / sizeof(int32_t));
}
 
__aicore__ inline void AivCommBase::BlockSync()
{
    uint32_t flagOffset = SYNC_BUFFER_OFFSET + 2 * FLAG_SIZE * blockdim_;
    __gm__ int32_t *ctrlFlagsGM = (__gm__ int32_t *)(GM_OUT[rank_] + flagOffset);
    if (GetBlockIdx() == 0) {
        //通知其他核
        pipe_barrier(PIPE_ALL);
        for (int i = 1; i < blockdim_; i++) {
            SetSignalValue(ctrlFlagsGM + i * FLAG_SIZE, localSetTensor, 1);
        }
        pipe_barrier(PIPE_ALL);
    } else {
        //接收通知并清零
        WaitSignalValue(ctrlFlagsGM + GetBlockIdx() * FLAG_SIZE, localCheckTensor, 1);
        SetSignalValue(ctrlFlagsGM +  GetBlockIdx() * FLAG_SIZE, localSetTensor, 0);
        pipe_barrier(PIPE_ALL);
    }
}
 
__aicore__ inline void AivCommBase::ClearSyncBuf()
{
    // 用10个flag
    Barrier(1);
    ClearFlag();
    Barrier(DOUBLE);
    BlockSync();
}

__aicore__ inline uint64_t AivCommBase::CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t AivCommBase::CalActualCount(uint32_t sliceIdx, uint64_t sliceCount,
    uint64_t avgLengthPerSlice, uint64_t tailLength)
{
    if (sliceIdx == sliceCount - 1) {
        return tailLength;
    } else if (sliceIdx < sliceCount - 1) {
        return avgLengthPerSlice;
    } else {
        return 0;
    }
}

__aicore__ inline void AivCommBase::CalBlockCountAndOffset(uint64_t len, uint32_t blockNumPerGroup,
uint32_t blockIdxInGroup, uint32_t padCount, uint64_t &count, uint64_t &blockOffset)
{
    uint64_t avgLengthPerBlock = CeilDiv(len, blockNumPerGroup);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerBlock, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;
 
    count = CalActualCount(blockIdxInGroup, sliceCount, avgLengthPerSlice, tailLength);
    blockOffset = blockIdxInGroup * avgLengthPerSlice;
}

template<typename T>
__aicore__ inline void AivCommBase::SetAtomicOp(uint32_t atomicOp)
{
    switch (atomicOp) {
        case HcclReduceOp::HCCL_REDUCE_SUM:
            SetAtomicAdd<T>(); break;
        case HcclReduceOp::HCCL_REDUCE_MAX:
            SetAtomicMax<T>(); break;
        case HcclReduceOp::HCCL_REDUCE_MIN:
            SetAtomicMin<T>(); break;
        default:
            SetAtomicNone(); break;
    }
}

template<typename T>
__aicore__ inline void AivCommBase::DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
    const uint32_t calCount)
{
    if ((calCount * sizeof(T)) % UB_ALIGN_SIZE == 0) {
        DataCopy(dstLocal, srcGlobal, calCount);
    } else {
        // 结构体DataCopyExtParams最后一个参数是rsv保留位
        DataCopyExtParams copyParams{1, calCount * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 1, 0};
        DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
    }
}

template<typename T>
__aicore__ inline void AivCommBase::DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const uint32_t calCount)
{
    if ((calCount * sizeof(T)) % UB_ALIGN_SIZE == 0) {
        DataCopy(dstGlobal, srcLocal, calCount);
    } else {
        DataCopyExtParams copyParams{1, calCount * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPad(dstGlobal, srcLocal, copyParams);
    }
}

template<typename T>
__aicore__ inline void AivCommBase::CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count, bool atomic,
    uint32_t atomicOp)
{
    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);

    if (atomic) {
        SetAtomicOp<T>(atomicOp);
    }

    uint64_t maxCountPerLoop = UB_MAX_DATA_SIZE / sizeof(T);
    if (useDoubleBuffer_) {
        maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
    }

    uint64_t curOffset = 0;
    while (count > 0) {
        uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
        inOutQue.FreeTensor(localOut);

        count -= curCount;
        curOffset += curCount;
    }

    if (atomic) {
        SetAtomicNone();
    }
    return;
}

template<typename T>
__aicore__ inline void AivCommBase::CpGM2GMWithFlagWrap(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count,
    int32_t index, uint64_t flushFrequency, int32_t tag)
{
    uint64_t curBatchCount = 0;

    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);

    uint64_t maxCountPerLoop = UB_MAX_DATA_SIZE / sizeof(T);
    if (useDoubleBuffer_) {
        maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
    }

    uint64_t curOffset = 0;
    while (count > 0) {
        uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
        inOutQue.FreeTensor(localOut);

        count -= curCount;
        curOffset += curCount;

        curBatchCount += 1;

        if (curBatchCount % flushFrequency == 0 || count == 0) {
            SyncFunc<HardEvent::MTE3_S>();
            CountRecord(tag, curBatchCount, index);
        }
    }
}

template<typename T>
__aicore__ inline void AivCommBase::CpGM2GMWithFlagWrap(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count,
    __gm__ int32_t* ctrlFlagGM, uint64_t flushFrequency, int32_t tag)
{
    uint64_t curBatchCount = 0;

    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);

    uint64_t maxCountPerLoop = UB_MAX_DATA_SIZE / sizeof(T);
    if (useDoubleBuffer_) {
        maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
    }

    uint64_t curOffset = 0;
    while (count > 0) {
        uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
        inOutQue.FreeTensor(localOut);

        count -= curCount;
        curOffset += curCount;

        curBatchCount += 1;

        if (curBatchCount % flushFrequency == 0 || count == 0) {
            SyncFunc<HardEvent::MTE3_S>();
            SetSignalValue(ctrlFlagGM, localSetTensor, curBatchCount + tag);
        }
    }
}

__aicore__ inline void AivCommBase::Record(uint32_t tag, int32_t waitRank, AivNotifyType notifyType, int32_t blockGroup, bool ifPingpong)
{
    int32_t OffSet = ifPingpong ? pingpongOffset : 0;
    int32_t recordOffset = (blockGroup * rankSize_ * FLAG_BUF_NUM + int32_t(notifyType) * rankSize_ + rank_ ) * FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(GM_OUT[waitRank]+ OffSet + recordOffset);
    SetSignalValue(ctrlFlagGM, localSetTensor, tag);
}

__aicore__ inline void AivCommBase::LocalRecord(uint32_t tag, int32_t blockIdx, AivNotifyType notifyType, bool ifPingpong)
{
    int32_t OffSet = ifPingpong ? pingpongOffset : 0;
    int32_t recordOffset = localOffset + (int32_t(notifyType) * MAX_BLOCK_DIM + blockIdx) * FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(GM_OUT[rank_]+ OffSet + recordOffset);
    SetSignalValue(ctrlFlagGM, localSetTensor, tag);
}

__aicore__ inline void AivCommBase::Record1vN(uint32_t tag, CommPattern pattern, AivNotifyType notifyType, int32_t block, bool ifPingpong)
{
    int32_t OffSet = ifPingpong ? pingpongOffset : 0;
    int32_t recordOffset = multiOffset + (int32_t(pattern) * 2 * BLOCK_DIM_FOUR_PER_RANK_A3 +
        int32_t(notifyType) * BLOCK_DIM_FOUR_PER_RANK_A3 + block) * ATOMIC_FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(GM_OUT[rank_]+ OffSet + recordOffset);
    SetSignalValue(ctrlFlagGM, localSetTensor, tag);
}

__aicore__ inline void AivCommBase::RecordNv1(uint32_t tag, int32_t waitRank, AivNotifyType notifyType, int32_t block, bool ifPingpong)
{
    int32_t OffSet = ifPingpong ? pingpongOffset : 0;
    int32_t recordOffset = multiOffset + 2 * 2 * BLOCK_DIM_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE +
        (int32_t(waitRank == rank_) * BLOCK_DIM_FOUR_PER_RANK_A3 * 2 + int32_t(notifyType) * BLOCK_DIM_FOUR_PER_RANK_A3
        + block) * ATOMIC_FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(GM_OUT[waitRank]+ OffSet + recordOffset);
    AddSignalValue(ctrlFlagGM, localSetTensor, tag);
}

__aicore__ inline void AivCommBase::CountRecord(uint32_t tag, int64_t count, int32_t index)
{
    int32_t OffSet = countOffset;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(GM_OUT[rank_]+ OffSet + index * FLAG_SIZE);
    SetSignalValue(ctrlFlagGM, localSetTensor, tag + count);
}

__aicore__ inline void AivCommBase::Wait(uint32_t tag, int32_t recordRank, AivNotifyType notifyType, int32_t blockGroup, bool ifpingpong)
{
    int32_t OffSet = ifpingpong ? pingpongOffset : 0;
    int32_t waitOffset = (blockGroup * rankSize_ * FLAG_BUF_NUM + int32_t(notifyType) * rankSize_ + recordRank) * FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(GM_OUT[rank_] + OffSet + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
}

__aicore__ inline void AivCommBase::LocalWait(uint32_t tag, int32_t blockIdx, AivNotifyType notifyType, bool ifpingpong)
{
    int32_t OffSet = ifpingpong ? pingpongOffset : 0;
    int32_t waitOffset = localOffset + (int32_t(notifyType) * MAX_BLOCK_DIM + blockIdx) * FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(GM_OUT[rank_] + OffSet + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
}

__aicore__ inline void AivCommBase::WaitNv1(uint32_t tag, int32_t recordRank, AivNotifyType notifyType, int32_t block, bool ifPingpong)
{
    int32_t OffSet = ifPingpong ? pingpongOffset : 0;
    int32_t waitOffset = multiOffset + (int32_t(recordRank == rank_) * BLOCK_DIM_FOUR_PER_RANK_A3 * 2 +
        int32_t(notifyType) * BLOCK_DIM_FOUR_PER_RANK_A3 + block) * ATOMIC_FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(GM_OUT[recordRank]+ OffSet + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
}
//是否直接清零
__aicore__ inline void AivCommBase::Wait1vN(uint32_t tag, CommPattern pattern, bool ifClear, AivNotifyType notifyType, int32_t block, bool ifPingpong)
{
    int32_t OffSet = ifPingpong ? pingpongOffset : 0;
    int32_t waitOffset = multiOffset + 2 * 2 * BLOCK_DIM_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE +
        (int32_t(pattern) * BLOCK_DIM_FOUR_PER_RANK_A3 * 2 +
        int32_t(notifyType) * BLOCK_DIM_FOUR_PER_RANK_A3 + block) * ATOMIC_FLAG_SIZE;
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(GM_OUT[rank_]+ OffSet + waitOffset);
    WaitSignalValue(ctrlFlagGM, localCheckTensor, tag);
    PipeBarrier<PIPE_ALL>();
    if (ifClear) {
        SetSignalValue(ctrlFlagGM, localSetTensor, 0);
    }
}

__aicore__ inline int32_t AivCommBase::CountWait(int32_t recordRank, int32_t index)
{
    __gm__ int32_t *ctrlFlagGM = (__gm__ int32_t *)(GM_OUT[recordRank]+ countOffset + index * FLAG_SIZE);
    LocalTensor<int32_t> flag = flagInQue.AllocTensor<int32_t>();
    int32_t flagValue = GetSignalValue(ctrlFlagGM, flag);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    flagInQue.FreeTensor(flag);
    return flagValue;
}

#endif  /* AIV_COMMUNICATION_BASE_H */
