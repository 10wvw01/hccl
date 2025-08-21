/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_HCCL_DEF_H__
#define __AICPU_HCCL_DEF_H__

#include <cstdint>
#include "hccl_common.h"
#include "aicpu_operator_pub.h"
#include "aicpu_hccl_sqcq.h"
#include "aicpu_hccl_sqcqv1.h"
#include "aicpu_hccl_sqcqv2.h"
#include "dfx_extend_info.h"
#include "profiling_extend_info.h"
#include "common.h"

constexpr int32_t AICPU_CNT = 8;
constexpr int32_t CLUSTER_CNT = 2;
constexpr u64 HCCL_COPY_ALIGN = 16 * 1024;
constexpr u64 HCCL_MIN_SLICE_ALIGN = 128;
constexpr u32 AC_DEFAULT_ONE_SHOT_SIZE = 100 * 1024;  // 缺省小于100K使用oneshot算法
constexpr s32 AC_ERROR_INVALID_PARAM = 0x011088;
constexpr s32 KFC_ERROR_RES_INVALID = 0x011099;
constexpr s32 KFC_ERROR_RTSQ_FULL = 0x011098;
constexpr u32 AC_MAX_RANK_NUM = 32U;
constexpr u32 AICPU_OP_NOTIFY_NUM = 2U;
constexpr u32 AC_SQE_SIZE = 64U;
constexpr u32 AC_MAX_PROF_LOOP = 30U;
constexpr u32 AC_MAX_PROF_COMM_CNT = 1024U;
constexpr u32 KFC_AIC_NUM = 32;  // 最多有32
constexpr u32 AC_DEFAULT_WINDOW_DIM = 2;  // 小数据量时分2片做swap window使用，可以提前把下一次内容拷贝到window中
constexpr u32 HCCL_COMM_DOMAIN_KEY_MAX_LEN = 128;
constexpr u32 TILING_TURN_MAX = 32;
constexpr u32 AC_DEFAULT_RANK_GROUP = 2;
constexpr u32 HCCL_SMALL_COUNT_1_M = 1024 * 1024;
constexpr u32 HCCL_SMALL_COUNT_256K = 256 * 1024; // 256KB: 256 * 1024
constexpr u64 ALL_REDUCE_THRESHOLD = 1024 * 1024; // allreduce确定性计算，小于该值选择AL算法，否则选择RLA算法
constexpr u64 ALL_REDUCE_THRESHOLD_UT = 1024; // allreduce确定性计算ut测试值
constexpr uint8_t MC2_DEBUG_ONLY_CUBE = 1; // 只计算不通信
constexpr uint8_t MC2_DEBUG_PRINT_MSG = 2;
constexpr uint8_t MC2_DEBUG_PRINT_BUFF = 3;
constexpr uint8_t MC2_DEBUG_TIME_TAKEN = 4; // KFC算子自己统计各阶段耗时
constexpr uint8_t MC2_DEBUG_WAIT_COMM = 8; // KFC算子等待通信结束
constexpr uint8_t MC2_DEBUG_PREPARE_TIMEOUT = 250;
constexpr uint8_t MC2_DEBUG_COMMIT_TIMEOUT = 251;
constexpr uint8_t MC2_DEBUG_NOTIFY_WAIT_TIMEOUT = 252;
constexpr uint8_t MC2_DEBUG_AICORE_WAIT_TIMEOUT = 253;
constexpr uint8_t MC2_DEBUG_FINALIZE_TIMEOUT = 254;
constexpr uint8_t MC2_DEBUG_SDMA_ERROR = 255;
constexpr uint32_t AC_MSG_VALID_MASK = 0x5CDF123A;

constexpr uint32_t AC_MSG_CNT = 64U;  // 可以创建64个消息
constexpr int32_t MC2_API_NO_COMM_ALG = -1;
constexpr uint64_t MC2_API_MSG_TIMEOUT = 20UL;
constexpr uint32_t MC2_RES_CTX_MAX = 3;
constexpr int8_t MC2_HCCL_MAX_HANDLE_ID = 63;
constexpr uint32_t AC_MAX_RANK_NUM_V2 = 256;
constexpr uint32_t MC2_API_XORCHECK_LEN = 15;
constexpr uint32_t MC2_API_XORCHECK_PRINT_NUM = 10000;
constexpr uint32_t MC2_API_TURNCNT_PRINT_NUM = 8;
constexpr uint8_t MC2_MSG_VERSION_FOR_TILING_API = 1U;
constexpr uint8_t MC2_MSG_VERSION_FOR_STATIC_GRAPH = 2U;
constexpr u64 MC2_API_TurnCnt_VALID = 1'234'567'899'999'999'999;
constexpr const uint32_t BYTE_PER_KB = 1024U;
constexpr const uint32_t BYTE_PER_MB = BYTE_PER_KB * BYTE_PER_KB;
constexpr uint32_t HCCL_MSG_EXT_CHECK_LEN = 8;
const uint16_t TAIL_TASK = 1;
const uint16_t HEAD_TASK = 0;
constexpr int32_t MAX_COMM_CTX_NUM = 3;
constexpr int32_t LOCAL = 0;
constexpr int32_t REMOTE = 1;
constexpr uint16_t TS_ERROR_RETRY_CONSTRAINT = 1000; // 重执行失败，约束是算子不一致或者inplace算子不支持
constexpr uint16_t MAX_BATCH_WRITE_THREAD_NUM = 2;

using HcclHandle = int8_t;

enum AicpuCcOpExeMode {
    AICPU_CC_OP_EXE_MODE_GRAPH = 0,
    AICPU_CC_OP_EXE_MODE_RPC_SERVER = 1,
    AICPU_CC_OP_EXE_MODE_PREPARE,
    AICPU_CC_OP_EXE_MODE_DELIVER,
    AICPU_CC_OP_EXE_MODE_HCCL_INVOKE,
    AICPU_CC_OP_EXE_MODE_MAX
};

enum AicpuCcOpFinishMode {
    AICPUSUSPENDING_ERROR = 301U,
};

enum AicpuComType {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_GATHER,
    HCCL_CMD_SCATTER,
    HCCL_CMD_BATCH_SEND_RECV,
    HCCL_CMD_BATCH_PUT,
    HCCL_CMD_BATCH_GET,
    HCCL_CMD_ALLGATHER_V,
    HCCL_CMD_REDUCE_SCATTER_V,
    HCCL_CMD_BATCH_WRITE,
    HCCL_CMD_ALL,
    HCCL_CMD_FINALIZE = 100,
    HCCL_CMD_INTER_GROUP_SYNC,
    HCCL_CMD_INIT,
    HCCL_CMD_BARRIER,
    HCCL_CMD_MAX
};

enum CommAlgType {
    COMM_ALG_DEFAULT = 0,
    COMM_ALG_FULL_MESH = 1,
    COMM_ALG_DOUBLE_RING = 2,
    COMM_ALG_SWITCH_WING = 3,
    COMM_ALG_RESERVED
};

// preparePosition标记
enum TASK_PREPARE_POSITION {
    TASK_PREPARE_HOST = 0,        // host模式，通信由host下发
    TASK_PREPARE_KERNEL = 1,      // kernel模式，通信由kernel下发
    TASK_PREPARE_RESERVED = 100
};

enum MC2_BUFFER_TYPE {
    MC2_BUFFER_TYPE_DEFAULT = 0,
    MC2_BUFFER_TYPE_OUTPUT,
    MC2_BUFFER_TYPE_WINDOW_IN,
    MC2_BUFFER_TYPE_WINDOW_OUT,
    MC2_BUFFER_TYPE_WORKSPACE,
    MC2_BUFFER_TYPE_INPUT,
    MC2_BUFFER_TYPE_COMMOUT,
    MC2_BUFFER_TYPE_END
};

enum AicpuCCExecOp {
    CC_EXE_TWO_SHOT_8_STREAM = 0, /**< two shot算法，适用于大数据量通信场景*/
    CC_EXE_ONE_SHOT_8_STREAM = 1, /**< one shot算法，适用于小数据量通信场景*/
    CC_EXE_ONE_SHOT_4_STREAM = 2, /**< one shot算法，4条流，需要执行两轮通信 */
    CC_EXE_ONE_SHOT_1_STREAM = 3, /**< one shot算法，1条流，需要执行三轮通信 */
    CC_EXE_TWO_SHOT_1_STREAM = 4,
    CC_EXE_ONE_SHOT_HD = 5,
    CC_EXE_ONE_SHOT_SINGLE_RING = 6,
};

constexpr u32 HCCL_AICPU_WAIT_HOST_BASE_TIME_MS = 200 * 1000;
// 重执行状态
enum class HcclOpExecFSM {
    HCCL_OP_EXEC_FSM_INIT = 0,
    HCCL_OP_EXEC_FSM_LAUNCH,
    HCCL_OP_EXEC_FSM_WAIT_END,
    HCCL_OP_EXEC_FSM_STOPPING,
    HCCL_OP_EXEC_FSM_STOPPED,
    HCCL_OP_EXEC_FSM_CHANGE_LINK,
    HCCL_OP_EXEC_FSM_WAIT_RETRY,
    HCCL_OP_EXEC_FSM_RETRY,
    HCCL_OP_EXEC_FSM_END,
    HCCL_OP_EXEC_FSM_ERROR,
    HCCL_OP_EXEC_STOP_LAUNCH,
};

enum AicpuTilingVer {
    TILING_DATA_VER_OLD_FOR_HOST = 0,
    TILING_DATA_VER_OLD_FOR_KERNEL,
    TILING_DATA_VER_OLD_FOR_KERNEL_V2,
    TILING_DATA_VER_OLD_FOR_TILING_API = 100
};
// GE图模式，走RunCpuKernl时用此结构
struct HcclAicpuOpParam {
    // TD后续改成AicpuCcExeParam
    AicpuComType commType;
    HcclReduceOp opType;
    u64 sendBuffer;
    u64 recvBuffer;
    u64 count;
    HcclDataType hcclDataType;

    // CcContext param
    u32 rankId;
    u32 rankNum;
    std::vector<s64> windowsIn;
    std::vector<s64> windowsOut;
    std::vector<s64> noIpcNotifyIds;
    std::vector<s64> ipcNotifyIds;
    std::vector<s64> streamIds;
    std::vector<s64> sqIds;
    // 正式方案启用std::vector<s64> notifyWrAddr;
};

// 和AIC/AIV协同实现集合通信时HCCL传入此结构
struct AicpuRpcServerModeLaunchOpAPI {
    AicpuComType commType;  // 不同通信类型需要的资源可能不同，下面都是以最大需要数量定义
    HcclReduceOp opType;
    u32 rankId;
    u32 rankNum;
    u64 windowsIn[AC_MAX_RANK_NUM];
    u64 noIpcNotifyIds[AC_MAX_RANK_NUM * 2];
    u64 ipcNotifyIds[AC_MAX_RANK_NUM * 4];
    u64 streamIds[AC_MAX_RANK_NUM];
    u64 sqIds[AC_MAX_RANK_NUM];
    u64 workSpaceAddr;
};

struct AicpuComProfCommLoop {
    // addr消息处理耗时
    u64 hccExecStartTime;
    u64 sendTaskStartTime;
    u64 sendSqeFinishTime;

    u64 dataLen;
};

struct AicpuComProf {
    // 初始化耗时
    u64 launchEntryTime;
    u64 commInitEndTime;

    // sqe填写发送统计数据
    u32 fillSqeCnt;
    u64 fillSqeTimes;
    u32 sendSqeBatch;
    u64 sendSqeTimes;

    // 记录具体每轮通信信息
    u32 workCnt;
    AicpuComProfCommLoop commLoop[AC_MAX_PROF_COMM_CNT];
    u64 tid;
    s32 clusterId;
    u32 rankId;

    // 记录trace上报耗时
    u64 traceSubmitTime;
    u64 traceCtxTime;
    u64 traceSqeTime;

    // 尾处理耗时
    u64 receiveFinalizeTime;
    u64 endTime;
};

using HccCommResParamTask = HcclCombinOpParam;

struct KFCGroupTilingDataAuto {  // for grouped_mat_mul_all_reduce op
    HcclKFCTilingData msg[64];  // 64: same as the tiling data size
    uint32_t groupNum;
    uint32_t groupTilingMagicNum;
};

struct KFCGroupTilingData {  // for grouped_mat_mul_all_reduce op
    uint32_t groupNum;
    uint32_t reserve;
    HcclKFCTilingData msg[64];  // 64: same as the tiling data size
};

struct HcclCommParamDesc {
    uint64_t version : 4;    // 版本号，当前是1
    uint64_t groupNum : 4;   // groupMatmul的输入数量，每个group对应一个输入和一个输出地址
    uint64_t hasFfts : 1;    // 910下是否是ffts融合算子（多一个ffts_addr参数
    uint64_t tilingOff : 7;  // tilingdata指针所在的参数索引
    uint64_t isDyn : 48;     // 输入参数是否是动态输入，从IR输入开始计算，不包含前面的参数，is_dyn是一个bitmap，每个bit对应一个IR输入，如果是动态输入则为1，否则是0
};

struct KFCTask {
    u64 inputA;      // A矩阵地址，通信在前时为sendbuffer
    u64 outputC;     // 输出C矩阵地址
    u64 commOut;     // 双输出时，通信输出地址
    u64 context;     // HCCL通信context
    u64 workSpace;   // 通信结果不直接输出时，放到workspace中
    u64 tilingData;  // 通信
};
struct KFCTaskComm {
    u64 context;     // HCCL通信context
    u64 tilingData;  // 通信
};

struct KFCTaskV2 {
    u64 inputA;      // A矩阵地址，通信在前时为sendbuffer
    u64 outputC;     // 输出C矩阵地址
    u64 commOut;     // 双输出时，通信输出地址
    u64 ctxNum;
    u64 context[MC2_RES_CTX_MAX];     // HCCL通信context
    u64 workSpace;   // 通信结果不直接输出时，放到workspace中
    u64 tilingData;  // 通信
};

struct KFCResInitTask {
    u64 context;  // A矩阵地址，通信在前时为sendbuffer
    bool isCustom;
};

struct AicpuComRankInfo {
    u32 rankId;
    u64 window;  // default size 200*1024*1024
    u64 windowOut;
};

struct AicpuComSignalInfo {
    u64 address;  // notify地址
    s32 actualNotifyId;
};

// record the communication global info.
struct AicpuComContext {
    char hcomId[HCCL_COMM_DOMAIN_KEY_MAX_LEN];
    u32 devId;
    u32 ssid;
    u32 rankId;   // self rank id
    u32 rankNum;  // total rank, include self

    AicpuComType commType;    // AllReduce, scatter..
    HcclReduceOp reducekind;  // ADD,MAX,MIN,EQUAL

    AicpuCCExecOp commOpType;  // twoshot.onshot...
    u32 unitSize;
    u64 commLen;
    u64 totalCnt; // 发送总数据个数
    u64 windowSize;

    u64 workSpaceAddr;
    u32 notifyOff;       // device notify write/read value偏移
    u16 notifyBeginCnt;  // notift write value的使用个数
    u16 notifyEndCnt;    // notift read value的使用个数
    u8 useBufferType;   // 使用recvbuf类型
    u64 winOffset;

    u64 kfcNotifyId;  // 用于主流全部任务最后添加record，激活kfc流(即内部创建用于launch
                      // kfc算子的流)wait，kfc流开始执行下一个算子
    u32 eventIds[AC_MAX_RANK_NUM];

    u32 curTurnCnt;  // 当前算法执行通信轮次
    u32 turnValue[TILING_TURN_MAX * AC_MAX_RANK_NUM];
    u32 totalTurnCnt; // 需要通信总轮次

    // profiling
    AicpuComProf acprof[AC_MAX_PROF_LOOP];

    AicpuComRankInfo rankInfo[AC_MAX_RANK_NUM];
    HcclComStreamInfo streamInfo[AC_MAX_RANK_NUM];
    int logLevel;

    AicpuComSignalInfo noIpcPreNotify[AC_MAX_RANK_NUM];
    AicpuComSignalInfo noIpcPostNotify[AC_MAX_RANK_NUM];
    AicpuComSignalInfo ipcPreRecordNotify[AC_MAX_RANK_NUM];
    AicpuComSignalInfo ipcPreWaitNotify[AC_MAX_RANK_NUM];
    AicpuComSignalInfo ipcPostRecordNotify[AC_MAX_RANK_NUM];
    AicpuComSignalInfo ipcPostWaitNotify[AC_MAX_RANK_NUM];
    AicpuComSignalInfo aicpuOpNotify[2]; // 集合通信AICPU展开资源

    uint8_t commAlg; // 指定通信算法
    bool directlySendMainSteramSqe;  // 消除头开销，第一需要直接执行时先下发主流
    bool alreadyInit;
    bool determinism;
    bool retryEnable;
    s32 clusterId;
    DevType devType = DevType::DEV_TYPE_COUNT;
    uint8_t debugMode = 0;
    u64 overflowAddr;
    TASK_PREPARE_POSITION preparePosition; // mc2高阶api，记录当前的模式
    u32 msgPosForKernel;     // mc2高阶api，记录当前处理的msg的位置
    u8 curTurnCntForKernel;  // mc2高阶api，记录当前msg的通信轮次
    u8 onlyRead;  // 只使用 SDMA 读进行拷贝
    dfx::DfxExtendInfo dfxExtendInfo;

    int sendCntRecord[4]; // 4 记录aicpu过程中的sendCnt,最多记录4次
    int recvCntRecord[4]; // 4 记录aicpu过程中的recvCnt,最多记录4次
    u32 retryHoldTime;
    u32 retryIntervalTime;
    u32 opIndex;
    bool isOpLaunch = false; //每个算子下发记录 （kNull/KRunning）
    bool isStopLaunch = false;  //MC2测试用例下，主线程是否实现停止算子展开
    bool endStopLaunch = false;  //NsStopLaunch是否处理 （背景线程/主线程）
    bool commOpenStatus = false; //通信域使用情况
    bool isRunning = false;
    std::shared_ptr<hccl::HDCommunicate> kfcControlTransferH2D{nullptr};
    std::shared_ptr<hccl::HDCommunicate> kfcStatusTransferD2H{nullptr};
    dfx::ProfilingExtendInfo profilingExtendInfo;
    u8 totalTurnCntForKernel; // mc2高阶api，记录当前msg的总通信轮次
    bool skipLocalDataCopy; // 通信算法是否拷贝本卡数据，根据isCommOut配置
                            // isCommOut = 1时，该项为0；isCommOut = 0时，该项为1
    u64 gatherOut;
    std::vector<struct hccl::TransportDeviceNormalData> ibversData;
    bool multiServerFlag;
    int64_t chipId;
};

struct Mc2InitTilingInner {        // 这个必须放到mc2tiling的最前面
    uint32_t version;           // tiling结构体版本号,外部不可配置, 100开始
    uint32_t mc2HcommCnt;       // 通信的次数
    uint32_t offset[8];         // 每个通信的偏移
    uint8_t debugMode;          // 调测模式, 0表示关闭,1表示开启,外部可配置
    uint8_t preparePosition;    // prepare消息发送的位置，0表示device，1表示host,外部可配置
    uint16_t queueNum;
    uint16_t commBlockNum;
};

struct Mc2CcTilingInner {
    uint8_t skipLocalRankCopy;    // 跳过本卡拷贝，在通信结果只需要给MC2内部计算使用或者本卡拷贝由aicore完成时,
    uint8_t skipBufferWindowCopy; // 跳过hbm到window间搬运 0不跳过，1跳过snd-window, 2跳过window-rcv
    uint8_t stepSize;             // 通信步长，粗粒度融合时填0,
    uint8_t version;              // 版本号
    char reserved[12];            // 保留字段
    char groupName[128];          // groupName
    char algConfig[128];          // 算法配置
    int32_t opType;               // tiling结构体版本号
    int32_t reduceType;           // reduce类型
};

struct HcclMsgV1 {
    AicpuComType commType;      // 通信原语类型，AllReduce/AllGather.../Finalize/InterHcclGroupSync
    HcclReduceOp opType;        // reduce操作类型，sum/prod/max/min
    uint64_t sendBuffer;        // 源数据buffer地址。
    uint64_t recvBuffer;        // 目的数据buffer地址
    uint64_t dataCnt;           // 参与操作的数据个数
    uint64_t strideCount;       // 完整的数据结果一般是连续的，切分多轮后会导致需要加上stride，例如AllGather的stride是每个卡上的完整数据量
    uint64_t ccOpTilingData;    // 消息的tiling信息
    uint32_t valid;             // 检查消息有效性
    HcclDataType hcclDataType;  // 参与操作的数据类型
    uint8_t repeatCnt;          // 本消息需要重复的次数，默认是1
    HcclHandle selfHandleID;    // 通信消息对应的handleId值
    uint8_t seqNum;             // 消息序号
    uint8_t version;            // 消息的版本信息，version=1使用hcclMsgV1
    uint32_t xorCheck;          // xor checksum
    void PrintMsg(const std::string &desc, bool isEventLog = false) {
        const s32 modelId = (isEventLog ? (HCCL | RUN_LOG_MASK) : HCCL);
        LOG_FUNC(modelId, HCCL_LOG_INFO,
                 "[%s:%d] [%u]%s: Msg[commType %u, opType %u, sendBuffer %p, recvBuffer %p, dataCnt %lu, "
                 "strideCount %lu, ccOpTilingData %#llx, valid %u, hcclDataType %u, repeatCnt %u, selfHandleID %d, "
                 "seqNum %u, version %u, xorCheck %u]", __FILE__, __LINE__, syscall(SYS_gettid), desc.c_str(),
                 static_cast<uint32_t>(commType), static_cast<uint32_t>(opType), sendBuffer, recvBuffer, dataCnt,
                 strideCount, ccOpTilingData, valid, static_cast<uint32_t>(hcclDataType), repeatCnt,
                 static_cast<int32_t>(selfHandleID), seqNum, version, xorCheck);
        if (ccOpTilingData != 0UL && version == MC2_MSG_VERSION_FOR_TILING_API) {
            Mc2CcTilingInner *tilingData = reinterpret_cast<Mc2CcTilingInner *>(ccOpTilingData);
            LOG_FUNC(modelId, HCCL_LOG_INFO,
                     "[%s:%d] [%u]%s: Mc2CcTilingInner[skipLocalRankCopy %u, skipBufferWindowCopy %u, stepSize %u, "
                     "version %u, groupName %s, algConfig %s, opType %d, reduceType %d]", __FILE__, __LINE__,
                     syscall(SYS_gettid), desc.c_str(), tilingData->skipLocalRankCopy, tilingData->skipBufferWindowCopy,
                     tilingData->stepSize, tilingData->version, std::string(tilingData->groupName).c_str(),
                     std::string(tilingData->algConfig).c_str(), tilingData->opType, tilingData->reduceType);
        }
    }

    std::string MsgToStr() {
        std::stringstream ss;
        ss << std::to_string(static_cast<u32>(commType)) << ",";
        ss << std::to_string(static_cast<u32>(opType)) << ",";
        ss << "0x" << std::hex << sendBuffer << ",";
        ss << "0x" << std::hex << recvBuffer << ",";
        ss << std::to_string(dataCnt) << ",";
        ss << std::to_string(strideCount) << ",";
        ss << "0x" << std::hex << ccOpTilingData << ",";
        ss << "0x" << std::hex << valid << ",";
        ss << std::to_string(static_cast<u32>(hcclDataType)) << ",";
        ss << std::to_string(repeatCnt) << ",";
        ss << std::to_string(static_cast<s32>(selfHandleID)) << ",";
        ss << std::to_string(static_cast<u32>(seqNum)) << ",";
        ss << std::to_string(static_cast<u32>(version)) << ",";
        ss << "0x" << std::hex << xorCheck << ".";
        return ss.str();
    }
};

struct HcclMsg {
    AicpuComType commType;          // 通信原语类型，AllReduce/AllGather.../Finalize/InterHcclGroupSync
    HcclReduceOp opType;            // reduce操作类型，sum/prod/max/min
    uint64_t sendBuffer;            // 源数据buffer地址。
    uint64_t recvBuffer;            // 目的数据buffer地址
    uint64_t dataCnt;               // 参与操作的数据个数
    uint64_t strideCount;           // 完整的数据结果一般是连续的，切分多轮后会导致需要加上stride，例如AllGather的stride是每个卡上的完整数据量
    HcclDataType hcclDataType;      // 参与操作的数据类型
    uint32_t p2pSrcDestRankId;      // 点对点通信send/recv对端的rankId，send中的destRank, recv中的srcRank

    uint32_t valid;                 // 检查消息有效性
    uint8_t repeatCnt;              // 本消息需要重复的次数，默认是1
    uint8_t everyTurnRsp;           // 每轮都需要等待执行结束发送响应，再执行下一轮
    uint8_t everyTurnWait;          // 每轮都需要等待work消息再执行
    HcclHandle commDepGroupID;      // 本消息执行需要等待的通信域组id，默认是-1，表示不需要等待，用于设置notify监听的通信域组id
    HcclHandle commDepHandleID;     // 本消息执行需要等待的通信域轮次，默认是-1，表示不需要等待，用于设置notify监听的地址
    HcclHandle selfHandleID;        // 通信消息对应的handleId值
    uint8_t seqNum;                 // 消息序号
    uint8_t version;                // 消息的版本信息，version=0使用hcclMsg
    uint32_t xorCheck;              // xor checksum
    void PrintMsg(const std::string &desc, bool isEventLog = false) {
        if (version >= MC2_MSG_VERSION_FOR_TILING_API) {
            // 当前存在两个结构体，通过version来区分具体是哪一种
            reinterpret_cast<HcclMsgV1 *>(this)->PrintMsg(desc, isEventLog);
        } else {
            LOG_FUNC((isEventLog ? (HCCL | RUN_LOG_MASK) : HCCL), HCCL_LOG_INFO,
                     "[%s:%d] [%u]%s: Msg[commType %u, opType %u, sendBuffer %p, recvBuffer %p, dataCnt %lu, "
                     "strideCount %lu, hcclDataType %u, p2pSrcDestRankId %u, valid %u, repeatCnt %u, everyTurnRsp %u, "
                     "everyTurnWait %u, commDepGroupID %d, commDepHandleID %d, selfHandleID %d, seqNum %u, version %u, "
                     "xorCheck %u]", __FILE__, __LINE__, syscall(SYS_gettid), desc.c_str(),
                     static_cast<uint32_t>(commType), static_cast<uint32_t>(opType), sendBuffer, recvBuffer, dataCnt,
                     strideCount, static_cast<uint32_t>(hcclDataType), p2pSrcDestRankId, valid, repeatCnt, everyTurnRsp,
                     everyTurnWait, commDepGroupID, commDepHandleID, selfHandleID, seqNum, version, xorCheck);
        }
    }
    std::string MsgToStr()
    {
        if (version >= MC2_MSG_VERSION_FOR_TILING_API) {
            return reinterpret_cast<HcclMsgV1 *>(this)->MsgToStr();
        }
        std::stringstream ss;
        ss << std::to_string(static_cast<u32>(commType)) << ",";
        ss << std::to_string(static_cast<u32>(opType)) << ",";
        ss << "0x" << std::hex << sendBuffer << ",";
        ss << "0x" << std::hex << recvBuffer << ",";
        ss << std::to_string(dataCnt) << ",";
        ss << std::to_string(strideCount) << ",";
        ss << std::to_string(static_cast<u32>(hcclDataType)) << ",";
        ss << std::to_string(p2pSrcDestRankId) << ",";
        ss << "0x" << std::hex << valid << ",";
        ss << std::to_string(static_cast<u32>(repeatCnt)) << ",";
        ss << std::to_string(static_cast<u32>(everyTurnRsp)) << ",";
        ss << std::to_string(static_cast<u32>(everyTurnWait)) << ",";
        ss << std::to_string(static_cast<s32>(commDepGroupID)) << ",";
        ss << std::to_string(static_cast<s32>(commDepHandleID)) << ",";
        ss << std::to_string(static_cast<s32>(selfHandleID)) << ",";
        ss << std::to_string(static_cast<u32>(seqNum)) << ",";
        ss << std::to_string(static_cast<u32>(version)) << ",";
        ss << "0x" << std::hex << xorCheck << ",";
        return ss.str();
    }
};

struct CommonHcclMsg {
    AicpuComType commType;          // 通信原语类型，AllReduce/AllGather.../Finalize/InterHcclGroupSync
    HcclReduceOp opType;            // reduce操作类型，sum/prod/max/min
    uint64_t sendBuffer;            // 源数据buffer地址。
    uint64_t recvBuffer;            // 目的数据buffer地址
    uint64_t dataCnt;               // 参与操作的数据个数
    uint64_t strideCount;           // 完整的数据结果一般是连续的，切分多轮后会导致需要加上stride，例如AllGather的stride是每个卡上的完整数据量
    HcclDataType hcclDataType;      // 参与操作的数据类型
    uint32_t p2pSrcDestRankId;      // 点对点通信send/recv对端的rankId，send中的destRank, recv中的srcRank
    uint32_t valid;                 // 检查消息有效性
    uint8_t repeatCnt;              // 本消息需要重复的次数，默认是1
    uint8_t everyTurnRsp;           // 每轮都需要等待执行结束发送响应，再执行下一轮
    uint8_t everyTurnWait;          // 每轮都需要等待work消息再执行
    HcclHandle commDepGroupID;      // 本消息执行需要等待的通信域组id，默认是-1，表示不需要等待，用于设置notify监听的通信域组id
    HcclHandle commDepHandleID;     // 本消息执行需要等待的通信域轮次，默认是-1，表示不需要等待，用于设置notify监听的地址
    HcclHandle selfHandleID;        // 通信消息对应的handleId值
    uint8_t seqNum;                 // 消息序号
    uint8_t version;                // 消息的版本信息，version=0使用hcclMsg
    uint32_t xorCheck;              // xor checksum
    uint64_t ccOpTilingData;        // 消息的tiling信息
    void PrintMsg(const std::string &desc) {
        HCCL_INFO("%s Msg[version %u, commType %u, opType %u, sendBuffer %p, recvBuffer %p, dataCnt %lu, strideLen %lu,"
            " hcclDataType %u, p2pSrcDestRankId %u, valid %u"
            " repeatCnt %u, everyTurnRsp %u, everyTurnWait %u, commDepGroupID %d,"
            " commDepHandleID %d, selfHandleID %d, seqNum %u, ccOpTilingData %#llx]",
            desc.c_str(), static_cast<uint32_t>(version), static_cast<uint32_t>(commType),
            static_cast<uint32_t>(opType), sendBuffer, recvBuffer, dataCnt, strideCount,
            static_cast<uint32_t>(hcclDataType), p2pSrcDestRankId, valid, repeatCnt, everyTurnRsp, everyTurnWait,
            commDepGroupID, commDepHandleID, selfHandleID, seqNum, ccOpTilingData);
    }
};

struct WqeSendSharedContect {
    volatile u32 startedThreadNum = 0;
    volatile u32 workedThreadNum = 0;
    volatile bool taskFinishFlag = false;
    volatile u32 curThreadIdsOnCpu[AICPU_CNT];
    volatile u32 sendWqeNum[MAX_BATCH_WRITE_THREAD_NUM];
};

struct DataBlock {
	uint32_t data[16];
};

struct HcclMsgExt {
    uint64_t sendCounts[AC_MAX_RANK_NUM_V2];  // sendCounts[i]表示本rank发给rank i的数据量。以sendType为基本单位。
    uint64_t sendOffset[AC_MAX_RANK_NUM_V2];  // sendOffset[i]表示本rank发给rank i的数据相对起始位置的偏移量
    uint64_t recvCounts[AC_MAX_RANK_NUM_V2];  // recvCounts[i]表示本rank从rank i收到的数据量为n
    uint64_t recvOffset[AC_MAX_RANK_NUM_V2];  // recvOffset[i]表示本rank从rank i收到的数据相对起始位置的偏移量
    uint64_t reserved[6];
    uint64_t valid;
    uint64_t xorCheck;
    std::string MsgToStr(u32 rankSize)
    {
        std::stringstream ss;
        ss << "sendCounts:";
        for (uint32_t i = 0; i < rankSize; ++i) {
            ss << sendCounts[i] << ",";
        }
        ss << " sendOffset:";
        for (uint32_t i = 0; i < rankSize; ++i) {
            ss << sendOffset[i] << ",";
        }
        ss << " recvCounts:";
        for (uint32_t i = 0; i < rankSize; ++i) {
            ss << recvCounts[i] << ",";
        }
        ss << " recvOffset:";
        for (uint32_t i = 0; i < rankSize; ++i) {
            ss << recvOffset[i] << ",";
        }
        return ss.str();
    }
};

struct TurnCnt {
    uint64_t valid; //客户端写一个魔鬼数字，服务端不做修改
    uint64_t cnt;
    uint64_t reserved[6];
};

struct ControlHcclMsg {
    uint8_t restart;
    uint8_t restarting;
    uint8_t restartCnt;
    uint8_t reserved[61];
};

struct HcclMsgArea {
    HcclMsg sendMsgList[AC_MSG_CNT];  // 发送消息队列
    HcclMsg recvMsgList[AC_MSG_CNT];  // 接收消息队列，为了避免服务端和客户端同时写
    uint8_t reserved0[8 * BYTE_PER_KB];    // for abi compatibility
    TurnCnt commitTurnCnt[AC_MSG_CNT];
    TurnCnt finishedTurnCnt[AC_MSG_CNT];
    uint8_t reserved1[BYTE_PER_MB];
    HcclMsgExt paramExtMsgList[AC_MSG_CNT];
    ControlHcclMsg controlMsg;
};

constexpr uint32_t MAX_QUE_NUM = 48U;
constexpr uint32_t MAX_AICPU_BLOCK_DIM = 6U;
constexpr uint32_t PADDING_SIZE = 0xF7000;
struct HcclMsgAreaForMultiQue {
    HcclMsg sendMsgList[MAX_QUE_NUM][AC_MSG_CNT];
    TurnCnt commitTurnCnt[MAX_QUE_NUM][AC_MSG_CNT];
    TurnCnt finishedTurnCnt[MAX_QUE_NUM][AC_MSG_CNT];
    uint8_t pad[PADDING_SIZE];
    ControlHcclMsg controlMsg;
};

// HCCL 代码直调时直接传此结构：
struct AivAicpuOpParam {
    AicpuComType commType;  // 32b
    HcclReduceOp opType;    // 32b
    u64 sendBuffer;
    u64 recvBuffer;
    u64 count;
    u64 strideLen;

    // offset 32B
    HcclDataType hcclDataType;

    uint32_t valid;   // 检查消息有效性
    uint8_t isLast;   // 是否最后一个下
    uint8_t funID;    // 功能ID，1地址消息；  2开始工作
    uint8_t sendCnt;  // 发送计数
    uint8_t rcvCnt;   // 执行结束轮次技术
    uint8_t everyTurnRsp;  // 每轮都需要等待执行结束发送响应，再执行下一轮
    uint8_t everyTurnWait;  // 每轮都需要等待work消息再执行
    uint8_t totalTurnCnt;  // 总轮次
    uint8_t useBufferType;
    uint64_t winOffset; // 发送数据偏移地址

    HcclOpIdentifier opId;
    uint8_t res[2];      // 整体消息64字节
    void PrintMsg(const std::string &desc) {
        HCCL_INFO("%s Msg[commType %u, opType %u, sendBuffer %p, recvBuffer %p, count %lu, strideLen %lu,"
            " hcclDataType %s, valid %u, isLast %u, funID %u, sendCnt %u, rcvCnt %u,"
            " everyTurnRsp %u, everyTurnWait %u, totalTurnCnt %u, winOffset %lu]",
            desc.c_str(), static_cast<uint32_t>(commType), static_cast<uint32_t>(opType), sendBuffer,
            recvBuffer, count, strideLen, GetDataTypeEnumStr(hcclDataType).c_str(), valid, isLast,
            funID, sendCnt, rcvCnt, everyTurnRsp, everyTurnWait, totalTurnCnt, winOffset);
    }
};

struct AicDebugInfo {
    uint32_t opIdx; // 整网中第几个mc2算子
    uint32_t rankM;
    uint32_t rankK;
    uint32_t rankN;
    uint8_t opType;
    uint8_t isTransB;
    uint8_t isBias;
    uint8_t isGatherOut;
    uint8_t tileNum;
    uint8_t tailNum;
    uint8_t reverse[42]; // 42 扩展到64字节
}; // 64

constexpr uint32_t MAX_DEBUG_CNT = 128U;
struct AicDebugCntInfo {
    uint8_t cnt[MAX_DEBUG_CNT];
}; // 128

using AicpuAddOneNotifyWaitSqe = void (*)(uint16_t, uint16_t, u64, const uint8_t *, uint8_t *,
    const dfx::DfxTimeOutConfig &);
using AicpuAddOneRecordSqe = void(*)(uint16_t, uint16_t, u64, const uint8_t *, uint8_t *);
using AicpuAddOneWriteValueRecordSqe = void(*)(uint16_t, uint16_t, u64, const uint8_t *, uint8_t *);
using AicpuAddOneMemcpySqe = void(*)(uint16_t, uint16_t, const void *, uint32_t, const rtDataType_t,
    rtRecudeKind_t, const void *, uint32_t, uint32_t, uint32_t, u64, uint8_t, const uint8_t *, uint8_t *);
using AicpuAddOneEventResetSqe = void(*)(uint16_t, int32_t, uint16_t, int64_t, int64_t,
    u64, const uint8_t *, uint8_t *);
using AicpuAddOneEventRecordSqe = void(*)(uint16_t, int32_t, uint16_t, const uint8_t *, uint8_t *);
using AicpuAddOneEventWaitSqe = void(*)(uint16_t, int32_t, uint16_t, const uint8_t *, uint8_t *);
using AicpuAddOneRdmaDbSendSqe = void(*)(uint16_t, uint16_t, uint64_t, uint64_t,
    uint32_t, uint8_t, const uint8_t *, uint8_t *);
using AicpuAddOneFlipPlaceHolderSqe = void(*)(uint16_t, uint16_t, uint16_t, const uint8_t *, uint8_t *);

extern u32 g_proxLoopCnt;

extern AicpuComContext *AicpuGetComContext();
extern void AicpuGetAllComContext(AicpuComContext *&contextBase, uint32_t &contextNum);
extern AicpuAddOneNotifyWaitSqe AicpuGetAddOneNotifyWaitSqe();
extern AicpuAddOneRecordSqe AicpuGetAddOneRecordSqe();
extern AicpuAddOneWriteValueRecordSqe AicpuGetAddOneWriteValueRecordSqe();
extern AicpuAddOneMemcpySqe AicpuGetAddOneMemcpySqe();
extern AicpuAddOneEventResetSqe AicpuGetAddOneEventResetSqe();
extern AicpuAddOneEventRecordSqe AicpuGetAddOneEventRecordSqe();
extern AicpuAddOneEventWaitSqe AicpuGetAddOneEventWaitSqe();
extern AicpuAddOneRdmaDbSendSqe AicpuGetAddOneRdmaDbSendSqe();
extern AicpuAddOneFlipPlaceHolderSqe AicpuGetAddOneFlipPlaceHolderSqe();

template <typename T> void AicpuUpdatComContextMumber(u64 offset, T value)
{
    AicpuComContext *contextBase = nullptr;
    uint32_t contextNum = 0;
    AicpuGetAllComContext(contextBase, contextNum);
    for (uint32_t i = 0; i < contextNum; i++) {
        T *dst = reinterpret_cast<T *>(reinterpret_cast<u64>(&contextBase[i]) + offset);
        *dst = value;
    }
    return;
};

struct KfcState {
    KfcState() {
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, isRunning), true);
    }
    ~KfcState() {
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, isRunning), false);
    }
};

struct RestartParam {
    // 重执行标记，表示是否发生异常需要重执行
    bool restartFlag = false;
    // 重执行次数
    uint32_t restartCnt = 0;
    // 是否所有通信域都重执行协商完成
    uint32_t consultationAllEnd = 0;
    // 通信域重执行协商情况
    bool consultationResult[MAX_COMM_CTX_NUM] = {false, false, false};
    // 通信域是否执行过changeLink
    bool linkChanged[MAX_COMM_CTX_NUM] = {false, false, false};
    // 通信域重执行协商初始化state
    HcclOpExecFSM fsmState[MAX_COMM_CTX_NUM] = {HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END,
        HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END,
        HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END};
    KfcError errorCode[MAX_COMM_CTX_NUM] = {KfcError::kNone, KfcError::kNone, KfcError::kNone};
    std::chrono::time_point<std::chrono::steady_clock> startTime[MAX_COMM_CTX_NUM];
};

enum class BarrierStatus: u8 {
    NO_BARRIER = 0U,
    SELF_BARRIER,
    INTER_BARRIER
};

struct BarrierInfo {
    BarrierStatus status;
    u64 lastTimeStamp;
};

enum class AicpuServerRole {
    MASTER = 0,
    SLAVE = 1,
    INVALID = 2
};

#endif
