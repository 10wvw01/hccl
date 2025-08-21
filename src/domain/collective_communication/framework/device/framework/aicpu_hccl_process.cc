/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include <string>
#include <slog.h>
#include <hccl/hccl_types.h>
#include "log.h"
#include "log_control.h"
#include "securec.h"
#include "aicpu_communicator.h"
#include "aicpu_hccl_sqcqv1.h"
#include "aicpu_hccl_sqcqv2.h"
#include "algorithm/aicpu_hccl_allgather.h"
#include "algorithm/aicpu_hccl_dispatcher.h"
#include "algorithm/aicpu_hccl_reduce_scatter.h"
#include "algorithm/aicpu_hccl_dmy_cal_allreduce.h"
#include "algorithm/aicpu_hccl_allreduce.h"
#include "algorithm/aicpu_hccl_alltoall.h"
#include "common/aicpu_hccl_common.h"
#include "common/hccl_msg_ring_buffer.h"
#include "common/sqe_context.h"
#include "profiling_manager_device.h"
#include "profiling_extend_info.h"
#include "executor_tracer.h"
#include "mc2_trace_utils.h"
#include "utils/mc2_aicpu_utils.h"
#include "utils/aicpu_hdc_utils.h"
#include "hccl_types.h"
#include "framework/aicpu_hccl_process.h"
#include "framework/aicpu_rpc_serverv2.h"
#include "framework/aicpu_prof.h"
#include "utils/mc2_tiling_utils.h"
#include "dtype_common.h"
#include "coll_batch_write_executor.h"

using namespace hccl;

u32 g_proxLoopCnt = 0;
u32 g_profTotalCnt = 0;
struct hcclCommAicpuInfo {
    std::mutex commAicpuMapMutex;
    std::unordered_map<std::string, std::pair<std::shared_ptr<hccl::HcclCommAicpu>, bool>> commMap;
};
hcclCommAicpuInfo g_commAicpuInfo;
WqeSendSharedContect g_sharedCtx;
CommonHcclMsgRingBuffer g_hcclMsgQueue;

AicpuAddOneNotifyWaitSqe g_addOneNotifyWaitSqe = nullptr;
AicpuAddOneRecordSqe g_addOneRecordSqe = nullptr;
AicpuAddOneWriteValueRecordSqe g_addOneWriteValueRecordSqe = nullptr;
AicpuAddOneMemcpySqe g_addOneMemcpySqe = nullptr;
AicpuAddOneEventResetSqe g_addOneEventResetSqe = nullptr;
AicpuAddOneEventRecordSqe g_addOneEventRecordSqe = nullptr;
AicpuAddOneEventWaitSqe g_addOneEventWaitSqe = nullptr;
AicpuAddOneRdmaDbSendSqe g_addOneRdmaDbSendSqe = nullptr;
AicpuAddOneFlipPlaceHolderSqe g_addOneFlipPlaceHolderSqe = nullptr;

AicpuAddOneNotifyWaitSqe AicpuGetAddOneNotifyWaitSqe()
{
    return g_addOneNotifyWaitSqe;
}
AicpuAddOneRecordSqe AicpuGetAddOneRecordSqe()
{
    return g_addOneRecordSqe;
}
AicpuAddOneWriteValueRecordSqe AicpuGetAddOneWriteValueRecordSqe()
{
    return g_addOneWriteValueRecordSqe;
}
AicpuAddOneMemcpySqe AicpuGetAddOneMemcpySqe()
{
    return g_addOneMemcpySqe;
}
AicpuAddOneEventResetSqe AicpuGetAddOneEventResetSqe()
{
    return g_addOneEventResetSqe;
}
AicpuAddOneEventRecordSqe AicpuGetAddOneEventRecordSqe()
{
    return g_addOneEventRecordSqe;
}
AicpuAddOneEventWaitSqe AicpuGetAddOneEventWaitSqe()
{
    return g_addOneEventWaitSqe;
}
AicpuAddOneRdmaDbSendSqe AicpuGetAddOneRdmaDbSendSqe()
{
    return g_addOneRdmaDbSendSqe;
}
AicpuAddOneFlipPlaceHolderSqe AicpuGetAddOneFlipPlaceHolderSqe()
{
    return g_addOneFlipPlaceHolderSqe;
}

namespace {
AicpuComContext g_comContext[CLUSTER_CNT];
std::unordered_map<int32_t, uint32_t> g_streamIdMap;
DevType g_devType = DevType::DEV_TYPE_COUNT;

u32 GetActiveSqId(AicpuComContext *ctx)
{
    return ctx->rankId;
}
 
void InitRankInfo(HccCommResParamTask *commParam, AicpuComContext *ctx)
{
    for (u32 i = 0; i < ctx->rankNum; i++) {
        ctx->rankInfo[i].rankId = i;
        ctx->rankInfo[i].window = commParam->windowsIn[i];
        ctx->rankInfo[i].windowOut = commParam->windowsOut[i];
    }
}

void CopyCxtInfo(AicpuComContext *ctx)
{
    auto otherClusterId = CLUSTER_CNT - ctx->clusterId - 1;
    auto otherCluster = &g_comContext[otherClusterId];
    *otherCluster = *ctx;
    otherCluster->clusterId = otherClusterId;
    HCCL_DEBUG("curClusterId = %d, otherClusterId = %d, copy finished", ctx->clusterId, otherCluster->clusterId);
}

DevType GetDevType(AicpuComContext *ctx)
{
    return ctx->devType;
}

void InitSqCqFun(AicpuComContext *ctx)
{
    DevType devType = GetDevType(ctx);
    if (devType == DevType::DEV_TYPE_310P1 || devType == DevType::DEV_TYPE_310P3) {
        g_addOneNotifyWaitSqe = AddOneNotifyWaitSqeV2;
        g_addOneRecordSqe = AddOneRecordSqeV2;
        g_addOneWriteValueRecordSqe = AddOneWriteValueRecordSqeV2;
        g_addOneMemcpySqe = AddOneMemcpySqeV2;
        g_addOneEventResetSqe = AddOneEventResetSqeV2;
        g_addOneEventRecordSqe = AddOneEventRecordSqeV2;
        g_addOneEventWaitSqe = AddOneEventWaitSqeV2;
    } else {
        g_addOneNotifyWaitSqe = AddOneNotifyWaitSqeV1;
        g_addOneRecordSqe = AddOneRecordSqeV1;
        g_addOneWriteValueRecordSqe = AddOneWriteValueRecordSqeV1;
        g_addOneMemcpySqe = AddOneMemcpySqeV1;
        g_addOneEventResetSqe = AddOneEventResetSqeV1;
        g_addOneEventRecordSqe = AddOneEventRecordSqeV1;
        g_addOneEventWaitSqe = AddOneEventWaitSqeV1;
        g_addOneFlipPlaceHolderSqe = AddOneFlipPlaceHolderSqeV1;
        g_addOneRdmaDbSendSqe = AddOneRdmaDbSendSqeV1;
    }
}

HcclResult InitChipType(AicpuComContext *ctx)
{
    CHK_RET(hrtHalGetDeviceType(ctx->devId, ctx->devType));
    CHK_RET(hrtHalGetDeviceInfo(ctx->devId, MODULE_TYPE_SYSTEM, INFO_TYPE_PHY_CHIP_ID, &ctx->chipId));
    if (ctx->devType == DevType::DEV_TYPE_910 || ctx->devType == DevType::DEV_TYPE_NOSOC ||
        ctx->devType == DevType::DEV_TYPE_COUNT) {
        HCCL_ERROR("Get devtype [%d] is invalid", ctx->devType);
        return HCCL_E_DRV;
    }
    if (ctx->devType == DevType::DEV_TYPE_310P3 || ctx->devType == DevType::DEV_TYPE_310P1) {
        uint32_t ssid;
        const HcclResult ret = hrtDrvMemSmmuQuery(ctx->devId, &ssid);
        HCCL_DEBUG("ssid %u", ssid);
        ctx->ssid = ssid;
        ctx->determinism = false;
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("hrtDrvMemSmmuQuery error"), HCCL_E_DRV);
    }
    InitSqCqFun(ctx);
    return HCCL_SUCCESS;
}

AicpuComProf *GetAicpuComProf(const u32 index, AicpuComContext *&ctx)
{
    AicpuComProf *acprof = &ctx->acprof[index];
    if (acprof->workCnt <= 0) {
        ctx = &(g_comContext[1]);
        acprof = &ctx->acprof[index];
    }
    return acprof;
}

HcclResult IsSupportRDMAReduce(AicpuComType commType, HcclDataType dataType, HcclReduceOp op)
{
    static const std::set<AicpuComType> multiThreadComTypeWhiteList = {
        HCCL_CMD_BATCH_WRITE,
    };
    if (MC2AicpuUtils::GetBlockNum() > 1U &&
        multiThreadComTypeWhiteList.find(commType) == multiThreadComTypeWhiteList.end()) {
        HCCL_ERROR("Unsupported comm type %u with multi threads.", commType);
        return HCCL_E_PARA;
    }

    if (commType != HCCL_CMD_ALLREDUCE && commType != HCCL_CMD_REDUCE_SCATTER) {
        return HCCL_SUCCESS;
    }
    static const std::set<HcclDataType> dtypeWhiteList = {
        HCCL_DATA_TYPE_FP32,
        HCCL_DATA_TYPE_FP16,
        HCCL_DATA_TYPE_INT8,
        HCCL_DATA_TYPE_INT16,
        HCCL_DATA_TYPE_INT32,
        HCCL_DATA_TYPE_BFP16
    };
    if (dtypeWhiteList.find(dataType) == dtypeWhiteList.end()) {
        HCCL_ERROR("Unsupported datatype %s for comm type %u.", GetDataTypeEnumStr(dataType).c_str(), commType);
        return HCCL_E_PARA;
    }

    static const std::set<HcclReduceOp> reduceTypeWhiteList = {
        HCCL_REDUCE_SUM,
        HCCL_REDUCE_MAX,
        HCCL_REDUCE_MIN
    };
    if (reduceTypeWhiteList.find(op) == reduceTypeWhiteList.end()) {
        HCCL_ERROR("Unsupported reduce op %s.", GetReduceOpEnumStr(op).c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RunConcreteAlgorithm(AivAicpuOpParam *commParam, AivAicpuOpParam *commParamNext,
    AicpuComContext *ctx)
{
    void *src = reinterpret_cast<void *>(static_cast<const uintptr_t>(commParam->sendBuffer));
    void *dst = reinterpret_cast<void *>(static_cast<const uintptr_t>(commParam->recvBuffer));
    RECORD_PROF_TIME(hccExecStartTime);

    const bool waitFlag = ((ctx->devType != DevType::DEV_TYPE_310P1 && ctx->devType != DevType::DEV_TYPE_310P3) &&
        ctx->commAlg == COMM_ALG_FULL_MESH);
    HCCL_DEBUG("startRunAlg src:%p, dst:%p, ctx commType:%d, commParam commType:%d, waitFlag:%u.",
        src, dst, ctx->commType, commParam->commType, static_cast<u32>(waitFlag));
    if (waitFlag) {
        CHK_RET(DispatcherAicpu::AddWaitStartTaskOnMainStream(GetActiveSqId(ctx)));
    }

    HcclResult result = HCCL_SUCCESS;
    switch (ctx->commType) {
        case HCCL_CMD_REDUCE_SCATTER: {
            CHK_RET(IsSupportRDMAReduce(commParam->commType, commParam->hcclDataType, commParam->opType));
            u64 strideLen = (commParam->strideLen != 0) ? commParam->strideLen : commParam->count / ctx->rankNum;
            AicpuHcclReduceScatter reduceScatter(ctx);
            result = reduceScatter.RunAlgorithm(
                commParam->opType, src, dst, commParam->count, commParam->hcclDataType, strideLen);
            break;
        }
        case HCCL_CMD_ALLGATHER: {
            u64 strideLen = (commParam->strideLen != 0) ? commParam->strideLen : commParam->count;
            AicpuHcclAllgather allgather(ctx);
            result = allgather.RunAlgorithm(commParam->opType, src, dst, commParam->count, commParam->hcclDataType,
                strideLen, commParamNext);
            break;
        }
        case HCCL_CMD_ALLREDUCE: {
            CHK_RET(IsSupportRDMAReduce(commParam->commType, commParam->hcclDataType, commParam->opType));
            if (ctx->determinism) {
                u64 strideLen = (commParam->strideLen != 0) ? commParam->strideLen : commParam->count;
                AicpuHcclDmyCalAllreduce dmyCalAllreduce(ctx);
                result = dmyCalAllreduce.RunAlgorithm(commParam->opType, src, dst, commParam->count,
                    commParam->hcclDataType, strideLen, commParamNext);
            } else {
                AicpuHcclAllreduce allreduce(ctx);
                result = allreduce.RunAlgorithm(commParam->opType, src, dst, commParam->count, commParam->hcclDataType);
            }
            break;
        }
        case HCCL_CMD_ALLTOALL: {
            u64 strideLen = (commParam->strideLen != 0) ? commParam->strideLen : commParam->count;
            AicpuHcclAllToAll allToAll(ctx);
            result = allToAll.RunAlgorithm(commParam->opType, src, dst, commParam->count,
                                           commParam->hcclDataType, strideLen);
            break;
        }
        default: {
            HCCL_ERROR("commType [%d] is not supported.", commParam->commType);
            result = HCCL_E_PARA;
            break;
        }
    }

    ctx->curTurnCnt++;
    HCCL_DEBUG("addEndTask, curTurnCnt:%u, totalTurnCnt:%u", ctx->curTurnCnt, ctx->totalTurnCnt);
    if (waitFlag) {
        CHK_RET(DispatcherAicpu::AddExecEndTaskOnMainStream(GetActiveSqId(ctx)));
    }
    return result;
}

HcclResult SetMsgWinOffset(AicpuComContext *ctx, AivAicpuOpParam *msg)
{
    if (msg->useBufferType == MC2_BUFFER_TYPE_WINDOW_IN &&
            ((msg->commType == HCCL_CMD_ALLREDUCE && !ctx->determinism) || msg->commType == HCCL_CMD_ALLTOALL)) {
        // sendBuffer 减去本卡的winIn
        AicpuComRankInfo *selfRankInfo = &ctx->rankInfo[ctx->rankId];
        if (msg->sendBuffer < selfRankInfo->window) {
            HCCL_ERROR("sendBuffer addr[%p] must bigger than window addr[%p].", msg->sendBuffer,
                selfRankInfo->window);
            return HCCL_E_PARA;
        }
        msg->winOffset = msg->sendBuffer - selfRankInfo->window;
    }
    HCCL_INFO("Offsetting winOffset %lu", msg->winOffset);
    return HCCL_SUCCESS;
}

void GetNextMsgFromMsg(AivAicpuOpParam *msg, AivAicpuOpParam *nextMsg, u64 dataLen, u32 rankNum)
{
    *(nextMsg) = *(msg);
    // nextMsg的偏移同UpdateMsg
    if (nextMsg->commType == HCCL_CMD_REDUCE_SCATTER) {
        nextMsg->sendBuffer = nextMsg->sendBuffer + dataLen / rankNum;
        nextMsg->recvBuffer = nextMsg->recvBuffer + dataLen / rankNum;
    } else {
        nextMsg->sendBuffer = nextMsg->sendBuffer + dataLen;
        nextMsg->recvBuffer = nextMsg->recvBuffer + dataLen;
    }
    nextMsg->PrintMsg("GetNextMsgFromMsg nextMsg");
}

void UpdateMsg(AivAicpuOpParam *msg, u64 dataLen, u32 rankNum)
{
    // 如果是reduceScatter算法，sendBuffer和recvBuffer的偏移为recvCnt，即sendCnt/rankNum
    // allgather和allreduce算法，sendBuffer和recvBuffer的偏移为recvCnt=sendCnt
    // all2all算法，sendBuffer和recvBuffer的偏移为 sendCnt / rankNum
    // 如果recvBuffer是非连续存储的，则recvBuffer的偏移将变更为 sendCnt
    if (msg->commType == HCCL_CMD_REDUCE_SCATTER) {
        msg->sendBuffer = msg->sendBuffer + dataLen / rankNum;
        msg->recvBuffer = msg->recvBuffer + dataLen / rankNum;
    } else {
        msg->sendBuffer = msg->sendBuffer + dataLen;
        msg->recvBuffer = msg->recvBuffer + dataLen;
    }
    if (msg->commType == HCCL_CMD_ALLREDUCE || msg->commType == HCCL_CMD_ALLTOALL) {
        msg->winOffset = msg->winOffset + dataLen;
    }
    msg->PrintMsg("UpdateMsg msg");
}

void UseHardTimeOut(AicpuComContext *ctx)
{
    // ctx外部保证合法
    ctx->dfxExtendInfo.dfxTimeOutConfig.useCredit = true;
}

static uint8_t g_expectPrepareId[MAX_QUE_NUM];
void SetExpectPrepareId(uint8_t queueId, uint8_t msgId)
{
    g_expectPrepareId[queueId] = msgId;
}

uint8_t GetExpectPrepareId(uint8_t queueId)
{
    return g_expectPrepareId[queueId];
}
}

static constexpr uint32_t ALLTOALLV_INFO_INDEX_2 = 2;
static constexpr uint32_t ALLTOALLV_INFO_INDEX_3 = 3;
static constexpr uint32_t ALLTOALLV_INFO_INDEX_4 = 4;
static constexpr uint64_t MSG_TIMEOUT = 20;
static constexpr uint64_t KERNEL_TIMEOUT = 16 * 60;
static constexpr uint64_t LOGCOUNT_PRINT_TIMEOUT = 10000;
static constexpr uint64_t WQE_SEND_TIMEOUT = 60;
static constexpr uint64_t SLAVE_EXECUTE_TIMEOUT = 5 * 60;
struct TimeOutCheckInfo {
    bool msgFlag[MAX_COMM_CTX_NUM];
    u64 kernelStartTime;
    u64 msgStartTime[MAX_COMM_CTX_NUM];
    uint32_t invalidMsgCount[MAX_COMM_CTX_NUM];
};

const std::unordered_map<std::string, std::string> g_algName = {
    {"AllGather=level0:ring", "AllGatherRingFor91093Executor"},
    {"AllGather=level0:fullmesh", "AllGatherMeshOpbaseExecutor"},
    {"AllGather=level0:doublering", "AlignedAllGatherDoubleRingFor91093Executor"},
    {"ReduceScatter=level0:ring", "ReduceScatterRingFor91093Executor"},
    {"ReduceScatter=level0:fullmesh", "ReduceScatterMeshDmaEliminationExecutor"},
    {"ReduceScatter=level0:doublering", "AlignedReduceScatterDoubleRingFor91093Executor"},
    {"AllReduce=level0:ring", "AllReduceRingFor91093Executor"},
    {"AllReduce=level0:fullmesh", "AllReduceMeshOpbaseLoopExecutor"},
    {"AllReduce=level0:doublering", "AlignedAllReduceDoubleRingFor91093Executor"},
    {"AlltoAll=level0:pairwise", "RunAlltoAllVStaged"},
    {"AlltoAll=level0:fullmesh", "RunAlltoAllDirectFullmesh"},
    {"BatchWrite=level0:fullmesh", BATCH_WRITE_ALG_NAME}
};

thread_local TimeOutCheckInfo g_timeOutInfoInst = {{false, false, false}, 0, {0, 0, 0}, {0, 0, 0}};

AicpuComContext *AicpuGetComContext()
{
    auto clusterId = MC2AicpuUtils::GetCurClusterId();
    return &g_comContext[clusterId];
}

DevType AicpuHcclProcess::AicpuGetInnerDevType()
{
    return g_devType;
}

void AicpuGetAllComContext(AicpuComContext *&contextBase, uint32_t &contextNum)
{
    contextBase = &g_comContext[0];
    contextNum = CLUSTER_CNT;
    return;
}

struct CommInstMgr {
    HcclOpResParam *resParam;
    hccl::HcclCommAicpu *hcclCommAicpu;
    AicpuRpcServerV2 rpcServer;
};

static std::unordered_map<std::string, int32_t> g_commIdMap;
static CommInstMgr g_commInst[MAX_COMM_CTX_NUM];

void AicpuHcclProcess::InsertComIdMap(uint32_t groupIdx, const std::string &hcomId) {
    g_commIdMap[hcomId] = groupIdx;
}

int32_t AicpuHcclProcess::GetComGroupIdx(const std::string &hcomId) {
    if (g_commIdMap.find(hcomId) == g_commIdMap.end()) {
        return -1;
    }
    return g_commIdMap[hcomId];
}

HcclResult AicpuHcclProcess::InsertCommInst(uint32_t idx, hccl::HcclCommAicpu *comm, HcclOpResParam *resParam) {
    if (idx >= MAX_COMM_CTX_NUM) {
        return HCCL_E_DRV;
    }
    g_commInst[idx].resParam = resParam;
    g_commInst[idx].hcclCommAicpu = comm;
    return HCCL_SUCCESS;
}

hccl::HcclCommAicpu *AicpuHcclProcess::GetCommAicpuCommInst(uint32_t idx) {
    if (idx >= MAX_COMM_CTX_NUM) {
        return nullptr;
    }
    return g_commInst[idx].hcclCommAicpu;
}

HcclOpResParam *AicpuHcclProcess::GetCommAicpuResInst(uint32_t idx) {
    if (idx >= MAX_COMM_CTX_NUM) {
        return nullptr;
    }
    return g_commInst[idx].resParam;
}

AicpuRpcServerV2 *AicpuHcclProcess::GetCommRpcServer(uint32_t idx) {
    if (idx >= MAX_COMM_CTX_NUM) {
        return nullptr;
    }
    return &g_commInst[idx].rpcServer;
}

bool AicpuHcclProcess::CheckMsgEnableFlag(int groupIdx) {
    return g_timeOutInfoInst.msgFlag[groupIdx];
}

void AicpuHcclProcess::SetMsgEnableFlag(int groupIdx, bool flag) {
    g_timeOutInfoInst.msgFlag[groupIdx] = flag;
}

void AicpuHcclProcess::SetMsgStartTime(int groupIdx) {
    g_timeOutInfoInst.msgStartTime[groupIdx] = GetCurCpuTimestamp();
}

void AicpuHcclProcess::SetKernelStartTime(void) {
    g_timeOutInfoInst.kernelStartTime = GetCurCpuTimestamp();
}

void AicpuHcclProcess::AddMsgInValidCount(uint32_t idx) {
    g_timeOutInfoInst.invalidMsgCount[idx]++;
}

void AicpuHcclProcess::ClearMsgInValidCount(uint32_t idx) {
    g_timeOutInfoInst.invalidMsgCount[idx] = 0;
}

uint32_t AicpuHcclProcess::GetMsgInValidCount(uint32_t idx) {
    return g_timeOutInfoInst.invalidMsgCount[idx];
}

bool AicpuHcclProcess::CheckMsgTimeOut(void) {
    int timeoutFlag = 0;
    for (int idx = 0; idx < MAX_COMM_CTX_NUM; idx++) {
        if (AicpuHcclProcess::CheckMsgEnableFlag(idx) &&
            (GetCurCpuTimestamp() - g_timeOutInfoInst.msgStartTime[idx]) >
            static_cast<unsigned long long>(NSEC_PER_SEC * KERNEL_TIMEOUT)) {
            HCCL_ERROR("comm group idx %d ReadValidMsg timeout %lus... ", idx, KERNEL_TIMEOUT);
            timeoutFlag++;
        }
    }
    if (timeoutFlag) {
        return true;
    }
    return false;
}

bool AicpuHcclProcess::CheckKernelTimeOut(void) {
    if ((GetCurCpuTimestamp() - g_timeOutInfoInst.kernelStartTime) >
        static_cast<unsigned long long>(NSEC_PER_SEC * KERNEL_TIMEOUT)) {
        HCCL_ERROR("Kernel Execute TimeOut %lus...", KERNEL_TIMEOUT);
        return true;
    }
    return false;
}

bool AicpuHcclProcess::CheckTimeOut(u64 startTimeStamp, u64 timeOutTime) {
    if ((GetCurCpuTimestamp() - startTimeStamp) > static_cast<unsigned long long>(NSEC_PER_SEC * timeOutTime)) {
        HCCL_ERROR("Execution TimeOut %lus...", timeOutTime);
        return true;
    }
    return false;
}

AicpuComContext *AicpuHcclProcess::AicpuGetInnerComContext(std::string &key)
{
    AicpuComContext *ctx = AicpuGetComContext();
    if (ctx->alreadyInit && !strncmp(key.c_str(), ctx->hcomId, HCCL_COMM_DOMAIN_KEY_MAX_LEN)) {
        return ctx;
    }
    return nullptr;
}

AicpuCCExecOp AicpuHcclProcess::GetCcOpType(u64 comDataLen, u64 rankNum)
{
    AicpuCCExecOp ccType;
    AicpuComContext *ctx = AicpuGetComContext();
    if (ctx->devType == DevType::DEV_TYPE_310P1 || ctx->devType == DevType::DEV_TYPE_310P3) {
        if (ctx->onlyRead > 0) {
            HCCL_DEBUG("Only read mode enabled");
            ccType = CC_EXE_ONE_SHOT_SINGLE_RING;
        } else if (rankNum == 2) { // 2 卡
            if (comDataLen < HCCL_SMALL_COUNT_1_M) {
                ccType = CC_EXE_ONE_SHOT_1_STREAM;
            } else {
                ccType = CC_EXE_TWO_SHOT_1_STREAM;
            }
        } else { // 2 卡以上
            if (comDataLen < HCCL_SMALL_COUNT_256K && (rankNum & (rankNum - 1)) == 0) {
                ccType = CC_EXE_ONE_SHOT_HD;
            } else {
                ccType = CC_EXE_ONE_SHOT_SINGLE_RING;
            }
        }
    } else {
        if ((comDataLen < AC_DEFAULT_ONE_SHOT_SIZE) && ((rankNum % AC_DEFAULT_RANK_GROUP) == 0)) {
            ccType = CC_EXE_ONE_SHOT_8_STREAM;
        } else {
            ccType = CC_EXE_TWO_SHOT_8_STREAM;
        }
    }
    HCCL_DEBUG("GetCcOpType len:%lu, ccType:%d", comDataLen, ccType);
    return ccType;
}

template <typename T>
HcclResult AicpuHcclProcess::InitAndVerifySignal(const HcclSignalInfo &signalInfo, std::shared_ptr<T> &notify,
    u64 &addr)
{
    if (signalInfo.resId == INVALID_U64) {
        // 无效值不做校验
        HCCL_INFO("[HcclCommAicpu][InitAndVerifySignal] resId is invalid, need not check");
        return HCCL_SUCCESS;
    }

    EXECEPTION_CATCH((notify = std::make_shared<T>()), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(notify);
    CHK_RET(notify->Init(signalInfo, NotifyLoadType::DEVICE_NOTIFY));
    HcclSignalInfo notifyInfo;
    CHK_RET(notify->GetNotifyData(notifyInfo));
    addr = notifyInfo.addr;
    HCCL_INFO("[HcclCommAicpu][InitAndVerifySignal] success, resId[%u], tsId:%d, devId[%u]",
        signalInfo.resId,
        signalInfo.tsId,
        signalInfo.devId);

    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::InitSignalInfo(HccCommResParamTask *commParam, AicpuComContext *ctx)
{
    for (u32 i = 0; i < ctx->rankNum; i++) {
        // 跨片notify只用在其它rank上，本片位置未填写有效值
        if (ctx->rankId != i) {
            // no ipc pre sync
            u64 address = 0;
            std::shared_ptr<LocalNotify> localNotify;
            HcclSignalInfo *sigInfo = &commParam->signalInfo.noIpcNotifys[i];
            CHK_RET(InitAndVerifySignal(*sigInfo, localNotify, address));
            ctx->noIpcPreNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);

            if (sigInfo->rankId != ctx->rankInfo[i].rankId) {
                HCCL_DEBUG("rankId mismatch. current process rank:%d, sigInfo rank:%d", ctx->rankInfo[i].rankId,
                    sigInfo->rankId);
                return HCCL_E_INTERNAL;
            }

            // no ipc post sync
            sigInfo = &commParam->signalInfo.noIpcNotifys[ctx->rankNum + i];
            CHK_RET(InitAndVerifySignal(*sigInfo, localNotify, address));
            ctx->noIpcPostNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);

            // ipc pre record
            sigInfo = &commParam->signalInfo.ipcNotifys[i];
            std::shared_ptr<RemoteNotify> remoteNotify;
            CHK_RET(InitAndVerifySignal(*sigInfo, remoteNotify, ctx->ipcPreRecordNotify[i].address));
            ctx->ipcPreRecordNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);

            // ipc pre wait
            sigInfo = &commParam->signalInfo.ipcNotifys[ctx->rankNum + i];
            CHK_RET(InitAndVerifySignal(*sigInfo, localNotify, ctx->ipcPreWaitNotify[i].address));
            ctx->ipcPreWaitNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);

            // ipc post record
            sigInfo = &commParam->signalInfo.ipcNotifys[2 * ctx->rankNum + i]; // 2 is ipc post record(8-15)
            CHK_RET(InitAndVerifySignal(*sigInfo, remoteNotify, ctx->ipcPostRecordNotify[i].address));
            ctx->ipcPostRecordNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);

            // ipc post wait
            sigInfo = &commParam->signalInfo.ipcNotifys[3 * ctx->rankNum + i]; // 3 is ipc post wait(16-23)
            CHK_RET(InitAndVerifySignal(*sigInfo, localNotify, ctx->ipcPostWaitNotify[i].address));
            ctx->ipcPostWaitNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::InitStreamInfo(HccCommResParamTask *commParam, AicpuComContext *ctx)
{
    g_streamIdMap.clear();
    u32 streamNum =  (ctx->multiServerFlag) ? 1 : ctx->rankNum;
    for (u32 i = 0; i < streamNum; i++) {
        auto &streamInfo = ctx->streamInfo[i];

        // sqId, logicCqids无需校验，若非法，下发时会报错。
        streamInfo.sqId = commParam->streamInfo[i].sqIds;
        streamInfo.logicCqId = commParam->streamInfo[i].logicCqids;
        streamInfo.actualStreamId = commParam->streamInfo[i].streamIds;
        HCCL_INFO("streamInfo.sqId :%d, streamId:%d", streamInfo.sqId, streamInfo.actualStreamId);
        u64 sq_addr = 0;
        CHK_RET(QuerySqBaseAddr(ctx->devId, streamInfo.sqId, sq_addr));
        streamInfo.sqBaseAddr = reinterpret_cast<void *>(sq_addr);

        CHK_RET(QuerySqStatusByType(ctx->devId, streamInfo.sqId, DRV_SQCQ_PROP_SQ_DEPTH, streamInfo.sqDepth));
        // actual streamid->idx
        g_streamIdMap[streamInfo.actualStreamId] = i;
    }
    CHK_RET(ResetSqBuff(ctx));
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::InitEventId(HccCommResParamTask *commParam, AicpuComContext *ctx)
{
    for (u32 i = 0; i < ctx->rankNum; i++) {
        // eventid只用在片内，放全局
        HcclSignalInfo *sigInfo = &commParam->signalInfo.noIpcEvents[i];
        if (sigInfo->rankId == ctx->rankId) {
            // 盘古230B入图场景连续跑第二次会出现eventId校验失败，当前不使用event，删除KfcResIsInvalid校验
            ctx->eventIds[i] = sigInfo->resId;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::InitIbversData(HccCommResParamTask *commParam, AicpuComContext *ctx)
{
    HCCL_INFO("commParam->ibverbsData:%llu", commParam->ibverbsData);
    if (commParam->ibverbsDataSize != static_cast<u64>(ctx->rankNum) * sizeof(TransportDeviceNormalData)) {
        HCCL_ERROR("ibverbsData size[%llu] is not valid, expect size[%llu]",
            commParam->ibverbsDataSize, static_cast<u64>(ctx->rankNum) * sizeof(TransportDeviceNormalData));
        return HCCL_E_PARA;
    }
    ctx->ibversData.resize(ctx->rankNum);
    for (u32 i = 0; i < ctx->rankNum; i++) {
        void* memPtr = reinterpret_cast<void*>(commParam->ibverbsData + sizeof(TransportDeviceNormalData) * i);
        ctx->ibversData[i] = *(static_cast<TransportDeviceNormalData*>(memPtr));
        ctx->ibversData[i].Print();
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::InitAicpuOpNotify(HccCommResParamTask *commParam, AicpuComContext *ctx)
{
    for (u32 i = 0; i < sizeof(ctx->aicpuOpNotify) / sizeof(ctx->aicpuOpNotify[0]); i++) {
        HcclSignalInfo *sigInfo = &commParam->signalInfo.aicpuOpNotify[i];
        std::shared_ptr<LocalNotify> localNitfy;
        EXECEPTION_CATCH((localNitfy = std::make_shared<LocalNotify>()), return HCCL_E_PTR);
        CHK_RET(localNitfy->Init(*sigInfo, NotifyLoadType::DEVICE_NOTIFY));
        HcclSignalInfo signalInfo;
        CHK_RET(localNitfy->GetNotifyData(signalInfo));
        ctx->aicpuOpNotify[i].actualNotifyId = static_cast<s32>(sigInfo->resId);
        ctx->aicpuOpNotify[i].address = signalInfo.addr;
    }
    return HCCL_SUCCESS;
}
HcclResult AicpuHcclProcess::InitTimeOutConfig(HccCommResParamTask *commParam, AicpuComContext *ctx)
{
    CHK_PTR_NULL(commParam);
    CHK_PTR_NULL(ctx);
    ctx->dfxExtendInfo.dfxTimeOutConfig.sqeTimeOutTimeOut = commParam->config.notifyWaitTime;
    ctx->dfxExtendInfo.dfxTimeOutConfig.sqeCreditTimeOut = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;
    ctx->dfxExtendInfo.dfxTimeOutConfig.sqeWaitTimeOut = dfx::kKfcTimeOut;
    ctx->dfxExtendInfo.dfxTimeOutConfig.sqFullWaitTimeOut = dfx::kSqFullWaitTimeOut;
    HCCL_INFO("DFX timeout config init successfully with details: [%s]",
              ctx->dfxExtendInfo.dfxTimeOutConfig.ToString().c_str());
    return HCCL_SUCCESS;
}

u32 AicpuHcclProcess::AicpuRpcResInit(HccCommResParamTask *commParam)
{
    MC2AicpuUtils::PrintHcclCombinOpParam(*commParam);

    AicpuComContext *ctx = AicpuGetComContext();
    if (ctx->alreadyInit) {
        if (strncmp(ctx->hcomId, commParam->hcomId, HCCL_COMM_DOMAIN_KEY_MAX_LEN)) {
            char oldGrp[HCCL_COMM_DOMAIN_KEY_MAX_LEN];
            char newGrp[HCCL_COMM_DOMAIN_KEY_MAX_LEN];
            strcpy_s(oldGrp, HCCL_COMM_DOMAIN_KEY_MAX_LEN, ctx->hcomId);
            strcpy_s(newGrp, HCCL_COMM_DOMAIN_KEY_MAX_LEN, commParam->hcomId);
            oldGrp[HCCL_COMM_DOMAIN_KEY_MAX_LEN - 1] = 0;
            newGrp[HCCL_COMM_DOMAIN_KEY_MAX_LEN - 1] = 0;
            HCCL_ERROR("the comm domain is not valid old [%s] != new[%s].", oldGrp, newGrp);
            return AC_ERROR_INVALID_PARAM;
        }
        HCCL_INFO("The ctx was already inited, end AicpuRpcResInit");
        return 0;
    }
    SqeContextUtils::InitSqeContext();
    ctx->kfcControlTransferH2D = nullptr;
    ctx->kfcStatusTransferD2H = nullptr;
    memset_s(ctx, sizeof(AicpuComContext), 0, sizeof(AicpuComContext));
    s32 enableEvent = 0;
    ctx->logLevel = dlog_getlevel(HCCL, &enableEvent);
    ctx->rankId = commParam->rankId;
    ctx->rankNum = commParam->rankNum;
    ctx->windowSize = commParam->winSize;
    ctx->workSpaceAddr = commParam->mc2WorkSpace.workSpace;
    ctx->curTurnCnt = 0;
    ctx->commAlg = 0;
    ctx->multiServerFlag = commParam->multiServerFlag;
    std::iota(ctx->turnValue, ctx->turnValue + TILING_TURN_MAX * AC_MAX_RANK_NUM, 0);
    HcclSignalInfo *sigInfo = &commParam->signalInfo.aicpuNotify;
    std::shared_ptr<LocalNotify> localNitfy;
    EXECEPTION_CATCH((localNitfy = std::make_shared<LocalNotify>()), return HCCL_E_PTR);
    CHK_RET(localNitfy->Init(*sigInfo, NotifyLoadType::DEVICE_NOTIFY));
    ctx->kfcNotifyId = sigInfo->resId;

    CHK_RET(hrtDrvGetLocalDevIDByHostDevID(sigInfo->devId, &(ctx->devId)));

    if (ctx->multiServerFlag) {
        CHK_RET(InitIbversData(commParam, ctx));
    } else {
        InitRankInfo(commParam, ctx);
        CHK_RET(InitSignalInfo(commParam, ctx));
        CHK_RET(InitEventId(commParam, ctx));
    }

    CHK_RET(InitStreamInfo(commParam, ctx));
    CHK_RET(InitAicpuOpNotify(commParam, ctx));
    CHK_RET(InitTimeOutConfig(commParam, ctx));
    HCCL_INFO("remote_udevid: %u, local_devid: %u, ssid: %u", sigInfo->devId, ctx->devId, ctx->ssid);
    ctx->directlySendMainSteramSqe = false;
    ctx->clusterId = MC2AicpuUtils::GetCurClusterId();
    auto ret = strcpy_s(ctx->hcomId, HCCL_COMM_DOMAIN_KEY_MAX_LEN, commParam->hcomId);
    ctx->hcomId[HCCL_COMM_DOMAIN_KEY_MAX_LEN - 1] = 0;
    ctx->determinism = (commParam->config.deterministic != 0) ? true : false;
    ctx->retryEnable = (commParam->config.retryEnable == 1) ? true : false;
    ctx->retryHoldTime = commParam->config.retryHoldTime;
    ctx->retryIntervalTime = commParam->config.retryIntervalTime;
    HCCL_DEBUG("[AicpuRpcResInit] ctx->retryEnable [%d], ctx->retryHoldTime [%u], ctx->retryIntervalTime [%u]",
        ctx->retryEnable, ctx->retryHoldTime, ctx->retryIntervalTime);
    CHK_RET(InitChipType(ctx));
    ctx->overflowAddr = commParam->overFlowAddr;
    HCCL_DEBUG("Init hcom group [%s] strcpy ret %d", ctx->hcomId, static_cast<int>(ret));
    ctx->onlyRead = commParam->onlyRead;
    UseHardTimeOut(ctx);
    dfx::ProfilingExtendInfoHelper::Init(ctx);
    ctx->alreadyInit = true;
    ctx->commOpenStatus = true;
    ctx->opIndex = 0;
    if (commParam->kfcControlTransferH2DParams.buffLen != 0) {
        EXECEPTION_CATCH((ctx->kfcControlTransferH2D = std::make_shared<hccl::HDCommunicate>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(ctx->kfcControlTransferH2D);
        CHK_RET(ctx->kfcControlTransferH2D->InitDevice(commParam->kfcControlTransferH2DParams));
    }
    if (commParam->kfcStatusTransferD2HParams.buffLen != 0) {
        EXECEPTION_CATCH((ctx->kfcStatusTransferD2H = std::make_shared<hccl::HDCommunicate>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(ctx->kfcStatusTransferD2H);
        CHK_RET(ctx->kfcStatusTransferD2H->InitDevice(commParam->kfcStatusTransferD2HParams));
    }
    CopyCxtInfo(ctx);
    CallMC2MaintenanceThread(ctx);
    if (MC2TraceUtils::Init() != HCCL_SUCCESS) {
        HCCL_ERROR("Init trace failed.");
        return static_cast<u32>(HCCL_E_INTERNAL);
    }
    HCCL_RUN_INFO("End AicpuRpcResInit");
    return 0;
}

void AicpuHcclProcess::OutputProfLog(const AicpuComContext &ctx)
{
    if (!MC2AicpuUtils::IsDebugModeEquals(ctx, MC2_DEBUG_TIME_TAKEN)) {
        return;
    }
    u32 threshold = AC_MAX_PROF_LOOP - 1;
    // 数组中存满之后，一次性输出之前存的`AC_MAX_PROF_LOOP`个算子的信息
    if (g_proxLoopCnt != threshold) {
        return;
    }
    for (u32 i = 0; i <= threshold; i++) {
        AicpuComContext *ctx = &(g_comContext[0]);
        AicpuComProf *acprof = GetAicpuComProf(i, ctx);
        u32 workRcdCnt = acprof->workCnt > AC_MAX_PROF_COMM_CNT ? AC_MAX_PROF_COMM_CNT : acprof->workCnt;

        HCCL_RUN_INFO("OP %u: clusterID %u, tid %lu, rankId %u, workCnt %u, serverStartTime %lu, waitMsgStartTime %lu, "
                      "rtsqExeEndTime %lu, serverEndTime %lu, StartServer %lu, Finalize %lu, E2E %lu",
                      g_profTotalCnt + i, acprof->clusterId, acprof->tid, ctx->rankId, acprof->workCnt,
                      acprof->launchEntryTime, acprof->commInitEndTime, acprof->receiveFinalizeTime, acprof->endTime,
                      acprof->commInitEndTime - acprof->launchEntryTime, acprof->endTime - acprof->receiveFinalizeTime,
                      acprof->endTime - acprof->launchEntryTime);

        for (u32 j = 0; j < workRcdCnt; j++) {
            AicpuComProfCommLoop *commRcd = &acprof->commLoop[j];
            HCCL_RUN_INFO("Turn %u: kfcAlgExeStartTime %lu, sendTaskStartTime %lu, sendSqeFinishTime %lu, "
                          "TaskWaitRequest %lu, TaskOrchestration %lu, TaskLaunch %lu, TaskExecute %lu",
                          j + 1, commRcd->hccExecStartTime, commRcd->sendTaskStartTime, commRcd->sendSqeFinishTime,
                          commRcd->hccExecStartTime - acprof->commInitEndTime,
                          commRcd->sendTaskStartTime - commRcd->hccExecStartTime,
                          commRcd->sendSqeFinishTime - commRcd->sendTaskStartTime,
                          acprof->receiveFinalizeTime - commRcd->sendSqeFinishTime);
        }

        acprof->fillSqeCnt = 0;
        acprof->fillSqeTimes = 0;
        acprof->sendSqeBatch = 0;
        acprof->sendSqeTimes = 0;
        acprof->workCnt = 0;
        acprof->traceSubmitTime = 0;
        acprof->traceCtxTime = 0;
        acprof->traceSqeTime = 0;
    }
    g_profTotalCnt += AC_MAX_PROF_LOOP;
}

HcclResult AicpuHcclProcess::AddTaskForGroupSyncMsg(hccl::HcclCommAicpu *commAicpu, CommonHcclMsg *hcclMsg,
                                                    AicpuRpcServerV2* rpcServer, uint32_t groupIdx)
{
    if (static_cast<uint32_t>(hcclMsg->commDepGroupID) == groupIdx) {
        HCCL_ERROR("InterHcclGroupSync must be used for cross-domain synchronization, group id %d",
                   hcclMsg->commDepGroupID);
        return HCCL_E_INTERNAL;
    }
    AicpuRpcServerV2* rpcServerDep = AicpuHcclProcess::GetCommRpcServer(hcclMsg->commDepGroupID);
    if (rpcServerDep == nullptr) {
        HCCL_ERROR("get rpc server failed, group id %d", hcclMsg->commDepGroupID);
        return HCCL_E_INTERNAL;
    }
    uint64_t waitAddr = rpcServerDep->GetFinishAddrByHandleId(hcclMsg->commDepHandleID);
    if (waitAddr == 0) {
        HCCL_INFO("AddTaskForGroupSyncMsg waitAddr is not ready, group id %d", hcclMsg->commDepGroupID);
        return HCCL_E_UNAVAIL;
    }
    int32_t turnNum = rpcServerDep->GetMsgRepeatCnt(hcclMsg->commDepHandleID);
    if (turnNum < 0) {
        HCCL_INFO("AddTaskForGroupSyncMsg comm group %d idx %d is not ready",
            hcclMsg->commDepGroupID, hcclMsg->commDepHandleID);
        return HCCL_E_UNAVAIL;
    }
    hccl::Stream curStream = commAicpu->GetMainStrem();
    rpcServer->SetNeedRetryFlag(false);
    CHK_RET(rpcServer->AddCcoreWait(commAicpu->GetDispatcher(), waitAddr, static_cast<uint32_t>(turnNum), &curStream, false));
    return HCCL_SUCCESS;
}

void AicpuHcclProcess::PrepareOpParam(hccl::OpParam *opParam, CommonHcclMsg *hcclMsg, AicpuRpcServerV2 &rpc, hccl::HcclCommAicpu *commAicpu)
{
    if (HcclCommProf::IsDebugModeEquals(MC2_DEBUG_SDMA_ERROR)) {
        opParam->inputPtr = reinterpret_cast<void *>(0xdeadbeef);
        opParam->outputPtr = reinterpret_cast<void *>(0xdeadbeef);
    } else {
        opParam->inputPtr = reinterpret_cast<void *>(hcclMsg->sendBuffer);
        opParam->outputPtr = reinterpret_cast<void *>(hcclMsg->recvBuffer);
    }
    opParam->reduceType = hcclMsg->opType;
    opParam->stream = commAicpu->GetMainStrem();
    opParam->syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam->opBaseAtraceInfo = nullptr;
    opParam->opType = static_cast<HcclCMDType>(hcclMsg->commType);
    if (hcclMsg->commType == HCCL_CMD_ALLTOALLV || hcclMsg->commType == HCCL_CMD_ALLTOALL) {
        HcclMsgExt *hcclMsgExt = rpc.GetHcclMsgExtPtr();
        opParam->All2AllDataDes.sendType = opParam->All2AllDataDes.recvType = hcclMsg->hcclDataType;
        opParam->All2AllDataDes.sendCount = hcclMsg->dataCnt;
        if (hcclMsg->commType == HCCL_CMD_ALLTOALL && hcclMsg->strideCount > 0UL) {
            for (uint32_t i = 0U; i < commAicpu->GetRankSize(); ++i) {
                hcclMsgExt->sendCounts[i] = hcclMsgExt->recvCounts[i] = hcclMsg->dataCnt;
                hcclMsgExt->sendOffset[i] = hcclMsgExt->recvOffset[i] = hcclMsg->strideCount * i;
            }
            opParam->opType = static_cast<HcclCMDType>(HCCL_CMD_ALLTOALLV);
        }
        if (opParam->opType == static_cast<HcclCMDType>(HCCL_CMD_ALLTOALLV)) {
            opParam->All2AllDataDes.sendCounts = static_cast<void *>(hcclMsgExt->sendCounts);
            opParam->All2AllDataDes.recvCounts = static_cast<void *>(hcclMsgExt->recvCounts);
            opParam->All2AllDataDes.sdispls = static_cast<void *>(hcclMsgExt->sendOffset);
            opParam->All2AllDataDes.rdispls = static_cast<void *>(hcclMsgExt->recvOffset);
        }
    } else if (hcclMsg->commType == HCCL_CMD_BATCH_WRITE) {
        opParam->BatchWriteDataDes.itemNum = hcclMsg->dataCnt;
        opParam->BatchWriteDataDes.queueNum = rpc.GetTotalQueueNum();
        opParam->BatchWriteDataDes.queueIdx = static_cast<u32>(hcclMsg->opType);
        HCCL_DEBUG("[Sdma-BatchWrite]Queue size %u, global queue id %u, item number %u.",
            opParam->BatchWriteDataDes.queueNum, opParam->BatchWriteDataDes.queueIdx,
            opParam->BatchWriteDataDes.itemNum);
    } else {
        const u64 totalSize = hcclMsg->dataCnt * DataUnitSize(hcclMsg->hcclDataType);
        opParam->DataDes.count = hcclMsg->dataCnt;
        opParam->DataDes.dataType = hcclMsg->hcclDataType;
        opParam->DataDes.strideCount = hcclMsg->strideCount;
        opParam->inputSize = totalSize;
        opParam->outputSize = totalSize;
    }
}

void AicpuHcclProcess::RepeatUpdatepOpParam(hccl::OpParam &opParam, CommonHcclMsg *hcclMsg, HcclMsgExt *hcclMsgExt, hccl::HcclCommAicpu *commAicpu)
{
    uint64_t dataLen = hcclMsg->dataCnt * DataUnitSize(hcclMsg->hcclDataType);
    if (hcclMsg->commType == HCCL_CMD_ALLTOALLV || (hcclMsg->commType == HCCL_CMD_ALLTOALL && hcclMsg->strideCount > 0)) {
        for (uint32_t i = 0; i < commAicpu->GetRankSize(); i++) {
            hcclMsgExt->sendOffset[i] += hcclMsgExt->sendCounts[i];
            hcclMsgExt->recvOffset[i] += hcclMsgExt->recvCounts[i];
        }
    } else {
        opParam.outputPtr = reinterpret_cast<void *>(reinterpret_cast<int8_t *>(opParam.outputPtr) + dataLen);
        opParam.inputPtr = reinterpret_cast<void *>(reinterpret_cast<int8_t *>(opParam.inputPtr) + dataLen);
    }
}

HcclResult AicpuHcclProcess::AddTaskForHcclMsgV2(hccl::HcclCommAicpu *comm, AicpuRpcServerV2 *rpc, CommonHcclMsg *hcclMsg,
                                                 const HcclOpResParam *commParam)
{
    uint32_t curTurnCntForKernel = 0;
    rpc->SetMsgPosForKernel(0);
    CommInfoCtx curCtx;
    HcclResult ret = comm->GetCommInfoCtx(hcclMsg->commType, curCtx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to get comm info from aicpu instance.");
        return HCCL_E_INTERNAL;
    }
    hccl::OpParam opParam;
    std::string algName = curCtx.algName;
    opParam.tag = curCtx.tag;
    std::string newTag = opParam.tag + "_mc2" + algName + "_device";

    u32 aicpuAlgType =  (static_cast<u32>(curCtx.algType.algoLevel2) << (HCCL_LEVEL_ALGO_WIDTH + HCCL_LEVEL_ALGO_WIDTH)) +
        (static_cast<u32>(curCtx.algType.algoLevel1) << HCCL_LEVEL_ALGO_WIDTH) +
        static_cast<u32>(curCtx.algType.algoLevel0);
    comm->SetAlgType(static_cast<u64>(aicpuAlgType));
    PrepareOpParam(&opParam, hcclMsg, *rpc, comm);
    hccl::AlgResourceResponse *algResResponse;
    std::unique_ptr<hccl::CollExecutorBase> executor;
    while (curTurnCntForKernel < hcclMsg->repeatCnt) {
        HCCL_INFO("Orchestrate curTurnCntForKernel %u, hcclMsg->repeatCnt %u", curTurnCntForKernel, hcclMsg->repeatCnt);
        curTurnCntForKernel++;
        rpc->SetMsgPosForKernel(curTurnCntForKernel);
        CHK_RET(comm->GetAlgResponseRes(newTag, algName, opParam, commParam, executor, algResResponse));
        HcclResult hcclRet = comm->Orchestrate(newTag, algName, opParam, executor, *algResResponse, commParam);
        HcclCommProf::GetCurrentAicpuProf()->workCnt++;
        CHK_PRT_RET(hcclRet != HCCL_SUCCESS,
            HCCL_ERROR("Executor op fail, opParam.tag[%s], algName[%s]",
            newTag.c_str(), algName.c_str()), hcclRet);
        RepeatUpdatepOpParam(opParam, hcclMsg, rpc->GetHcclMsgExtPtr(), comm);
    }
    return HCCL_SUCCESS;
}

// 通信算法执行
HcclResult AicpuHcclProcess::AicpuCcOpExe(AivAicpuOpParam *commParam, AivAicpuOpParam *commParamNext,
    AicpuComContext *ctx)
{
    HCCL_DEBUG("----------start AicpuCcOpExe -------");
    if (commParam == nullptr) {
        HCCL_ERROR("AicpuCcOpExe commParam is null.");
        return HCCL_E_PARA;
    }

    if (ctx == nullptr) {
        HCCL_ERROR("AicpuCcOpExe ctx is null.");
        return HCCL_E_PARA;
    }

    // 1. process global resource, update context.
    ctx->unitSize = DataUnitSize(commParam->hcclDataType);
    CHK_PRT_RET(ctx->unitSize == 0, HCCL_ERROR("[AicpuCcOpExe]ctx->unitSize is zero."), HCCL_E_PARA);
    ctx->commLen = ctx->unitSize * commParam->count;
    ctx->commType = commParam->commType;
    ctx->reducekind = commParam->opType;
    ctx->commOpType = GetCcOpType(ctx->commLen, ctx->rankNum); // twoshot.onshot...
    ctx->totalTurnCnt = commParam->totalTurnCnt;
    ctx->useBufferType = commParam->useBufferType;
    ctx->winOffset = commParam->winOffset;

    if (MC2AicpuUtils::NeedRecordTimeTaken(*ctx)) {
        u32 index = ctx->acprof[g_proxLoopCnt].workCnt;
        index = (index >= AC_MAX_PROF_COMM_CNT) ? (AC_MAX_PROF_COMM_CNT - 1) : index;
        ctx->acprof[g_proxLoopCnt].commLoop[index].dataLen = ctx->commLen;
    }

    HcclResult result = RunConcreteAlgorithm(commParam, commParamNext, ctx);
    if (result != HCCL_SUCCESS) {
        HCCL_ERROR("Run comm alg failed, rankId:%d, result:%u.", ctx->rankId, result);
        return result;
    }
    ctx->acprof[g_proxLoopCnt].workCnt = ctx->curTurnCnt;
    // 所有轮次执行完毕后通知aclnn
    if (ctx->curTurnCnt == ctx->totalTurnCnt &&
        (ctx->devType != DevType::DEV_TYPE_310P1 && ctx->devType != DevType::DEV_TYPE_310P3) &&
        ctx->preparePosition != TASK_PREPARE_KERNEL) {
        CHK_RET(DispatcherAicpu::AddAllEndTaskOnMainStream(GetActiveSqId(ctx)));
    }

    return HCCL_SUCCESS;
}
HcclResult AicpuHcclProcess::TryRunRpcServerOneStageWait(AicpuComContext *ctx, AicpuRpcServer &rpc)
{
    HCCL_INFO("Start to run, round %u", ctx->dfxExtendInfo.kfcRestartConfig.tryRestartTimes);
    if (dfx::DfxExtendInfoHelper::TryRestartTooManyTimes(ctx->dfxExtendInfo)) {
        HCCL_ERROR("Restart too many times, max try count is %u",
                   ctx->dfxExtendInfo.kfcRestartConfig.maxRestartTimes);
        return HCCL_E_INTERNAL;
    }
    const auto ret = RunRpcServerOneStageWait(ctx, rpc);
    if (ret == HCCL_SUCCESS) {
        dfx::DfxExtendInfoHelper::ResetTryRestartTimes(ctx->dfxExtendInfo);
        CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kEnd, KfcError::kNone, 0));
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), false);
        return HCCL_SUCCESS;
    }
    if (ctx->dfxExtendInfo.commandToKfc == dfx::CommandToKfc::kRestart) {
        dfx::DfxExtendInfoHelper::TryRestartOnceMore(ctx->dfxExtendInfo);
        return TryRunRpcServerOneStageWait(ctx, rpc);
    }
    dfx::DfxExtendInfoHelper::ResetTryRestartTimes(ctx->dfxExtendInfo);
    if (ctx->isStopLaunch) {
        CopyCtxForBackGroundDfx(ctx);
        CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kStoplaunch, KfcError::kNone, 0));
    } else {
        CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError, KfcError::kInner, 0));
    }
    return ret;
}

void AicpuHcclProcess::CopyCtxForBackGroundDfx(const AicpuComContext *ctx)
{
    auto otherClusterId = CLUSTER_CNT - ctx->clusterId - 1;
    AicpuComContext *otherCluster = &g_comContext[otherClusterId];
    otherCluster->workSpaceAddr = ctx->workSpaceAddr;
    otherCluster->notifyOff = ctx->notifyOff;
    otherCluster->notifyBeginCnt = ctx->notifyBeginCnt;
    otherCluster->totalTurnCnt = ctx->totalTurnCnt;
    SqeContextUtils::SaveVariable();
}

struct BatchWriteItem {
    uint64_t localBuf;
    uint64_t remoteBuf;
    uint64_t count;
    uint32_t dataType;
    uint32_t remoteRankId;
};

// mock GetCpuId 多个线程需要放回不同的值，mock组件在多线程时不安全，会放回错误。所以在跑llt时加锁。
#ifdef CCL_LLT
std::mutex g_mtxForLLT; 
#endif
HcclResult AicpuHcclProcess::ConcurrentPostSendWqe(const CommonHcclMsg &commonHcclMsg, const AicpuComContext *ctx, u8 *needSendTotalNum) {
    const BatchWriteItem *item = reinterpret_cast<BatchWriteItem *>(static_cast<uintptr_t>(commonHcclMsg.sendBuffer));
    std::vector<Transport::Buffer> remoteList = {{}};
    std::vector<Transport::Buffer> local = {{}};
    int32_t cpuId = 0;
    {
        #ifdef CCL_LLT
        std::lock_guard<std::mutex> lock(g_mtxForLLT);
        #endif
        cpuId = MC2AicpuUtils::GetCpuId();
    }
    u32 threadId = g_sharedCtx.curThreadIdsOnCpu[cpuId];
    u32 sendWqeNum = 0;
    for (u64 i = 0; i < commonHcclMsg.dataCnt; ++i) {
        if (item->remoteRankId != ctx->rankId) {
            (*needSendTotalNum)++;
            if (item->remoteRankId % g_sharedCtx.workedThreadNum == threadId) {
                remoteList[0].addr = reinterpret_cast<void *>(item->remoteBuf);
                local[0].addr = reinterpret_cast<void *>(item->localBuf);
                remoteList[0].size = local[0].size =
                    item->count * DataUnitSize(static_cast<HcclDataType>(item->dataType));
                HCCL_INFO(
                    "Batch write item[%u]: context rankId [%u], remoteRankId[%u], sendThreadId[%ld], remoteBuf[%#llx],"
                    " localBuf[%#llx], dataType[%u], count[%lu]",
                    i,
                    ctx->rankId,
                    item->remoteRankId,
                    threadId,
                    item->remoteBuf,
                    item->localBuf,
                    item->dataType,
                    item->count);
                CHK_RET(MC2AicpuUtils::PostSend(*ctx, item->remoteRankId, remoteList, local, true));
                sendWqeNum++;
            }
        }
        ++item;
    }
    g_sharedCtx.sendWqeNum[threadId] = sendWqeNum;
    HCCL_DEBUG("thread %u send %u wqe success.", threadId, sendWqeNum);
    return HCCL_SUCCESS;
}

// 真正处理BatchWrite master 从commonHcclMsg中取消息，更新工作线程数，放到队列中。 从队列中取数据进行发送。
HcclResult AicpuHcclProcess::HandleBatchWriteOperation(const CommonHcclMsg &commonHcclMsg,const AicpuComContext *ctx) {
    if (commonHcclMsg.dataCnt == 0UL || commonHcclMsg.sendBuffer == 0UL) {
        HCCL_ERROR("Get msg send buffer is nullptr or dataCnt is zero. "
            "Msg[commType %u, opType %u, sendBuffer %p, dataCnt %lu]", static_cast<uint32_t>(commonHcclMsg.commType),
            static_cast<uint32_t>(commonHcclMsg.opType), commonHcclMsg.sendBuffer, commonHcclMsg.dataCnt);
        return HCCL_E_PARA;
    }

    g_sharedCtx.workedThreadNum = g_sharedCtx.startedThreadNum;
    if (g_sharedCtx.workedThreadNum > 1) {
        bool success = false;
        while(!success) {
		    success = g_hcclMsgQueue.Enqueue(&commonHcclMsg);
	    }
    }
    u8 needSendTotalNum = 0;
    CHK_RET(ConcurrentPostSendWqe(commonHcclMsg, ctx, &needSendTotalNum));
    HCCL_DEBUG("total need send wqe num is %u", needSendTotalNum);
    CHK_RET(WaitForSlaveCompletion(ctx, needSendTotalNum));
    g_hcclMsgQueue.Dequeue();
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::WaitForSlaveCompletion(const AicpuComContext *ctx, u8 needSendTotalNum) {
    HCCL_DEBUG("needsendTotalNum is %ld.", needSendTotalNum);
    u64 startTimeStamp = GetCurCpuTimestamp();
    while (true) {
        uint32_t sendNum = 0;
        for (uint32_t i = 0; i < g_sharedCtx.workedThreadNum; ++i) {
            HCCL_DEBUG("wait thread %ld send %ld wqe success.", i, g_sharedCtx.sendWqeNum[i]);
            sendNum += g_sharedCtx.sendWqeNum[i];
        }
        if (needSendTotalNum <= sendNum) {
            HCCL_DEBUG("needsendTotalNum is %ld, already send %ld", needSendTotalNum, sendNum);
            return HCCL_SUCCESS;
        }
        if (CheckTimeOut(startTimeStamp, WQE_SEND_TIMEOUT)) {
            g_sharedCtx.taskFinishFlag = true;
            HCCL_ERROR("slave thread send wqe timeout.");
            return HCCL_E_TIMEOUT;
        }
    }
}

void AicpuHcclProcess::RecordReportStatus(uint32_t groupNum, dfx::ReportStatus status) {
    for (uint32_t i = 0; i < groupNum; i++) {
        hccl::HcclCommAicpu *comm = AicpuHcclProcess::GetCommAicpuCommInst(i);
        if (comm != nullptr) {
            comm->RecordReportStatus(status);
        }
    }
}

HcclResult AicpuHcclProcess::AllEndCcOpExe(AicpuComContext *ctx)
{
    return DispatcherAicpu::AddAllEndTaskOnMainStream(GetActiveSqId(ctx));
}

HcclResult AicpuHcclProcess::AddTaskForHcclMsg(AicpuComContext *ctx, AicpuRpcServer &rpc, CommonHcclMsg *hcclMsg,
    AivAicpuOpParam *msg)
{
    // reduce scatter:在strideLen使能的情况下，如果recvCount * repeat > strideLen 则偏移越界，报错
    if (hcclMsg->commType == HCCL_CMD_REDUCE_SCATTER && hcclMsg->strideCount != 0 &&
        hcclMsg->dataCnt * hcclMsg->repeatCnt > hcclMsg->strideCount) {
        HCCL_ERROR("In reduce scatter algorithm, when stride Count is not zero, repeatCnt * dataCnt"
            " should not be greater than strideCount.");
        hcclMsg->PrintMsg("");
        return HCCL_E_PARA;
    }

    AivAicpuOpParam *tmpptr = nullptr;
    AivAicpuOpParam nextMsg;
    u64 dataLen = DataUnitSize(msg->hcclDataType) * msg->count;
    ctx->curTurnCntForKernel = 0;
    ctx->totalTurnCntForKernel = hcclMsg->repeatCnt;
    while (ctx->curTurnCntForKernel < hcclMsg->repeatCnt) {
        HCCL_INFO("RunRpcServerApi ctx->curTurnCntForKernel %u, hcclMsg->repeatCnt %u",
            ctx->curTurnCntForKernel,
            hcclMsg->repeatCnt);
        // 当前msg预取仅支持当前及下一条msg都为allgather
        if (hcclMsg->commType == HCCL_CMD_ALLGATHER &&
            (hcclMsg->hcclDataType == HCCL_DATA_TYPE_FP16 || hcclMsg->hcclDataType == HCCL_DATA_TYPE_BFP16)) {
            HCCL_INFO("Try get allgather next msg");
            HcclMsg tmpMsg;
            if (ctx->curTurnCntForKernel < (hcclMsg->repeatCnt - 1)) {
                GetNextMsgFromMsg(msg, &nextMsg, dataLen, ctx->rankNum);
                tmpptr = &nextMsg;
            } else if (rpc.CheckRcvAddrMsg(&tmpMsg, ctx->msgPosForKernel + 1)) {
                CommonHcclMsg commonHcclMsg;
                GetCommonHcclMsg(&tmpMsg, &commonHcclMsg);
                rpc.HcclMsg2AicAicpuOpParam(&commonHcclMsg, &nextMsg);
                tmpptr = &nextMsg;
            } else {
                HCCL_INFO("nextMsg is not ready. msgPos %u", ctx->msgPosForKernel + 1);
                tmpptr = nullptr;
            }
            // 如果nextMsg和hcclMsg不同commtype或datatype，nextMsg要置为nullptr
            if (tmpptr != nullptr && (tmpptr->commType != hcclMsg->commType ||
                tmpptr->hcclDataType != hcclMsg->hcclDataType)) {
                HCCL_INFO("Set nextMsg nullptr");
                tmpptr = nullptr;
            }
        }
        ctx->curTurnCntForKernel++;
        CHK_RET(AicpuCcOpExe(msg, tmpptr, ctx));
        TaskOrchestrator::ActiveRecordMain(GetActiveSqId(ctx));
        // 更新msg
        UpdateMsg(msg, dataLen, ctx->rankNum);
    }
    return HCCL_SUCCESS;
}

// 针对 HCCL_CMD_BATCH_WRITE 的batch write的消息，master循环访问queue0，如果有消息，读取发送到WQE队列中。
HcclResult AicpuHcclProcess::RunRpcServerApi(AicpuComContext *ctx, AicpuRpcServer &rpc)
{
    HCCL_INFO("----------start RunRpcServerApi -------");
    if (ctx->devType != DevType::DEV_TYPE_910B) {
        HCCL_ERROR("Platform not support, please use 910B platform.");
        return HCCL_E_PARA;
    }
    HcclMsg hcclMsg;
    CommonHcclMsg commonHcclMsg;
    AivAicpuOpParam msg;
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), dfx::KfcStatus::kOneStart);
    CallMC2MaintenanceThread(ctx);
    ctx->directlySendMainSteramSqe = true;
    ctx->msgPosForKernel = 0;

    msg.opId.index = ctx->opIndex + 1;
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, opIndex), msg.opId.index);
    if (ctx->endStopLaunch) {
        HCCL_WARNING("the op should not be launched in suspending status");
        return HCCL_E_SUSPENDING;
    }
    CHK_RET(AicpuHdcUtils::InitOpExecStatus(ctx->kfcStatusTransferD2H, msg.opId));
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), true);

    while (true) {
        HCCL_INFO("start to read the [%u] msg", ctx->msgPosForKernel);
        if (!rpc.ReadAddrMsg(&hcclMsg, ctx->msgPosForKernel)) {
            HCCL_ERROR("fail to get addr msg, msgPos %u", ctx->msgPosForKernel);
            rpc.PrintAllHcclMsgArea();
            TaskOrchestrator::PrintTimeOutSqInfo(ctx, ctx->dfxExtendInfo.dfxTimeOutConfig.sqeWaitTimeOut);
            return HCCL_E_TIMEOUT;
        }
        GetCommonHcclMsg(&hcclMsg, &commonHcclMsg);
        // 处理finalzie消息
        if (commonHcclMsg.commType == HCCL_CMD_FINALIZE) {
            ctx->acprof[g_proxLoopCnt].receiveFinalizeTime = GetCurCpuTimestamp(true);
            if (ctx->debugMode == MC2_DEBUG_PRINT_BUFF) {
                rpc.PrintAllHcclMsgAreaData();
            }
            break;
        } else if (commonHcclMsg.commType == HCCL_CMD_INIT) {
            continue;
        } else if (commonHcclMsg.commType == HCCL_CMD_INTER_GROUP_SYNC ||
            commonHcclMsg.commType == HCCL_CMD_BARRIER) {
            HCCL_ERROR("Msg %u is not supported.", static_cast<uint32_t>(commonHcclMsg.commType));
            return HCCL_E_PARA;
        } else if (commonHcclMsg.commType == HCCL_CMD_BATCH_WRITE) {
            // 校验多机场景，multiServerFlag必须为true
            if (!ctx->multiServerFlag) {
                HCCL_ERROR("Batch write is only support in multi server.");
                return HCCL_E_PARA;
            }
			// 处理BatchWrite的操作从直接发送->队列发送。
            CHK_RET(AicpuHcclProcess::HandleBatchWriteOperation(commonHcclMsg, ctx));
            // 刷一下标记内存 commitTUrnCnt=0, finsihTurnCnt++
            rpc.WriteTurnCnt(ctx->msgPosForKernel);
        } else {
            rpc.HcclMsg2AicAicpuOpParam(&commonHcclMsg, &msg);
            if (msg.sendBuffer == 0UL || msg.recvBuffer == 0UL) {
                HCCL_ERROR("Get msg buffer is nullptr.");
                msg.PrintMsg("Invalid msg buffer");
                rpc.PrintAllHcclMsgArea();
                return HCCL_E_PARA;
            }
            CHK_RET(SetMsgWinOffset(ctx, &msg));
            CHK_RET(AddTaskForHcclMsg(ctx, rpc, &commonHcclMsg, &msg));
        }
        // 切换到下一个msg
        ctx->msgPosForKernel = (ctx->msgPosForKernel + 1) % AC_MSG_CNT;
    }
    // 添加结束任务
    if (!ctx->multiServerFlag) {
        CHK_RET(AllEndCcOpExe(ctx));
        TaskOrchestrator::ActiveRecordMain(GetActiveSqId(ctx));
        ctx->directlySendMainSteramSqe = false;
        CHK_RET(MC2AicpuUtils::WaitTaskFinish(ctx));
    } else {
        HCCL_DEBUG("master over task is finish.");
        g_sharedCtx.taskFinishFlag = true;
    }
    rpc.WriteFinishWhenAllFinalize(ctx->msgPosForKernel);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), dfx::KfcStatus::kOneFinished);
    return HCCL_SUCCESS;
}

void AicpuHcclProcess::CallMC2MaintenanceThread(AicpuComContext *ctx)
{
    if (!IsSupportStartMC2MaintenanceThread()) {
        return;
    }

    if (ctx->commOpenStatus && (ctx->devType == DevType::DEV_TYPE_310P1 || ctx->devType == DevType::DEV_TYPE_310P3)) {
        return;
    }

    HCCL_INFO("Register back ground func on device type %u, status %u.", static_cast<u32>(ctx->devType),
        static_cast<u32>(ctx->commOpenStatus));
    hrtHalStartMC2MaintenanceThread(
        dfx_tracer::ExecutorTracer::BackGroundDfx, ctx, dfx_tracer::ExecutorTracer::StopBackGroundDfx, ctx);
}

HcclResult AicpuHcclProcess::RunRpcServerOneStageWait(AicpuComContext *ctx, AicpuRpcServer &rpc)
{
    AivAicpuOpParam g_msg[3];
    AivAicpuOpParam *msg = &g_msg[0];
    AivAicpuOpParam *preMsg = &g_msg[1];
    AivAicpuOpParam *nextMsg = &g_msg[2];
    AivAicpuOpParam *tmpptr = nullptr;
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), dfx::KfcStatus::kOneStart);
    CallMC2MaintenanceThread(ctx);
    // 读取首轮任务，并准备，直接先下主流(直接激活)，后续还是先下从流，激活时下主流
    rpc.CheckRcvAddrMsg(msg, 0);
    if (!rpc.CheckAivIsEnd(0)) {
        rpc.ReadAddrMsg(nextMsg, 0);
        tmpptr = nextMsg;
    }
    HcclUpdateOpIndex(msg->commType, ctx);
    msg->opId.index = ctx->opIndex;
    if(ctx->endStopLaunch){
        HCCL_WARNING("the op should not be launched in suspending status");
        return HCCL_E_SUSPENDING;
    }
    auto ret = AicpuHdcUtils::InitOpExecStatus(ctx->kfcStatusTransferD2H, msg->opId);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), true);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("InitOpExecStatus failed, ret:%u", ret);
        return ret;
    }
    ctx->directlySendMainSteramSqe = true;
    CHK_RET(AicpuCcOpExe(msg, tmpptr, ctx));

    while (!rpc.CheckAivIsEnd(0)) {
        tmpptr = msg;
        msg = preMsg;
        preMsg = tmpptr; // msg <-> preMsg
        tmpptr = nullptr;

        // 读取下一次任务，并编排
        rpc.CheckRcvAddrMsg(msg, 0);
        if (!rpc.CheckAivIsEnd(0)) {
            rpc.ReadAddrMsg(nextMsg, 0);
            tmpptr = nextMsg;
        }
        CHK_RET(AicpuCcOpExe(msg, tmpptr, ctx));

        // 激活下一次任务执行
        CHK_RET(TaskOrchestrator::ActiveRecordMain(GetActiveSqId(ctx)));
    }

    // 激活下一次任务执行
    CHK_RET(TaskOrchestrator::ActiveRecordMain(GetActiveSqId(ctx)));
    ctx->directlySendMainSteramSqe = false;
    CHK_RET(MC2AicpuUtils::WaitTaskFinish(ctx));
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), dfx::KfcStatus::kOneFinished);
    MC2AicpuUtils::PrintBuffer(ctx, *msg);
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::RunRpcServerTwoStageWait(AicpuComContext *ctx, AicpuRpcServer &rpc)
{
    HCCL_DEBUG("----------start RunRpcServerTwoStageWait -------");
    AivAicpuOpParam gMsg[3];
    AivAicpuOpParam *msg = &gMsg[0];
    AivAicpuOpParam *msgWork = &gMsg[1];
    AivAicpuOpParam *nextMsg = &gMsg[2];
    AivAicpuOpParam *tmpptr = nullptr;

    // 读取首轮任务，并准备，直接先下主流(直接激活)，后续还是先下从流，激活时下主流
    // 1.1、首轮读地址（需要自动产生）
    rpc.CheckRcvAddrMsg(msg, 0);
    if (!rpc.CheckAivIsEnd(0)) {
        // 读取下一轮地址
        rpc.ReadAddrMsg(nextMsg, 0);
        tmpptr = nextMsg;
    }

    // 1.2 首轮提前读看是否需要提前下主流，即判断sendcnt是否大于等于当前轮次
    if (rpc.ReadWorkMsg(msgWork, 0, (ctx->curTurnCnt + 1)) && rpc.GetWaitPolicy() != 0) {
        ctx->directlySendMainSteramSqe = true;
    }

    // 1.3 首轮编排开始
    CHK_RET(AicpuCcOpExe(msg, tmpptr, ctx));
    ctx->directlySendMainSteramSqe = false;
    // 2、等待激活任务执行，如果前面已经激活，则ActiveRecordMain会空转一圈
    if (rpc.GetWaitPolicy() != 0) {
        rpc.CheckRcvWorkMsg(msgWork, 0, ctx->curTurnCnt);
    }

    MC2AicpuUtils::PrintBuffer(ctx, *msg);
    CHK_RET(TaskOrchestrator::ActiveRecordMain(GetActiveSqId(ctx)));
    while (!rpc.CheckAivIsEnd(0)) {
        HCCL_DEBUG("RunRpcServerTwoStageWait CheckAivIsEnd");
        tmpptr = nullptr;
        // 3.1、读取下一轮任务
        rpc.CheckRcvAddrMsg(msg, 0);
        if (!rpc.CheckAivIsEnd(0)) {
            rpc.ReadAddrMsg(nextMsg, 0);
            tmpptr = nextMsg;
        }

        // 3.2 开始编排下一轮
        CHK_RET(AicpuCcOpExe(msg, tmpptr, ctx));

        // 5.1 等待上一轮执行结束
        TaskOrchestrator::WaitMainStreamFinish(ctx);

        // 6.1 激活下一轮
        rpc.CheckRcvWorkMsg(msgWork, 0, ctx->curTurnCnt);
        CHK_RET(TaskOrchestrator::ActiveRecordMain(GetActiveSqId(ctx)));

        // 7.1 发送上一轮消息
        rpc.PostMsg(ctx->curTurnCnt - 1);
    }

    if (rpc.GetRspPolicy() != 0) {
        // 8.1 等待执行结束
        TaskOrchestrator::WaitMainStreamFinish(ctx);
        rpc.ClearWorkMsg();
        HCCL_INFO("[commType:%d, opType:%s, sendBuffer:%p, recvBuffer:%p, count:%d, data_type:%s, "
            "sendCnt:%d, rcvCnt:%d, funID:%d, valid:%d, everyTurnRsp:%d, strideLen:%d, isLast:%d",
            msg->commType, GetReduceOpEnumStr(msg->opType).c_str(), msg->sendBuffer, msg->recvBuffer, msg->count,
            GetDataTypeEnumStr(msg->hcclDataType).c_str(), msg->sendCnt,
            msg->rcvCnt, msg->funID, msg->valid, msg->everyTurnRsp, msg->strideLen, msg->isLast);
        // 9.1 发送最后一轮消息
        rpc.PostMsg(ctx->curTurnCnt);
    }
    MC2AicpuUtils::PrintBuffer(ctx, *msg);
    HCCL_DEBUG("----------end RunRpcServerTwoStageWait -------");
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::AICPU_RpcServerUnfoldStageWait(AicpuComContext *ctx, AicpuRpcServer &rpc)
{
    AivAicpuOpParam opParams;

    auto waitStopExecCmdTimeoutMs = HcclGetWaitStopExecCmdTimeout();
    auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);

    auto startTime = std::chrono::steady_clock::now();

    KfcError errorCode = KfcError::kNone;
    uint32_t retryCnt = 0;
    uint32_t beginSqePos = INVALID_UINT;
    uint32_t endSqePos = INVALID_UINT;
    HcclOpExecFSM state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_INIT;
    HcclResult ret = HCCL_SUCCESS;
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), dfx::KfcStatus::kOneStart);
    CallMC2MaintenanceThread(ctx);
    while (true) {
        switch (state) {
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_INIT:
                ret = HcclOpExecFsmInitProcess(ctx, state, errorCode, rpc, opParams);
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_LAUNCH:
                ret = HcclOpExecFsmLaunchProcess(ctx, state, errorCode, opParams, beginSqePos, endSqePos);
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END:
                ret = HcclOpExecFsmWaitEndProcess(ctx, state, errorCode, retryCnt);
                if (state == HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING) {
                    startTime = std::chrono::steady_clock::now();
                }
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING:
                if ((std::chrono::steady_clock::now() - startTime) >= waitStopExecCmdTimeout) {
                    HCCL_ERROR("hccl aicpu wait stop exec timeout[%u ms].", waitStopExecCmdTimeoutMs);
                    errorCode = KfcError::kTimeout;
                    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
                } else {
                    ret = HcclOpExecFsmStoppingProcess(ctx, state, errorCode, retryCnt);
                }
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED:
                ret = HcclOpExecFsmStoppedProcess(ctx, state, errorCode, retryCnt, opParams, beginSqePos, endSqePos);
                if (state == HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY) {
                    startTime = std::chrono::steady_clock::now();
                }
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY:
                {
                    auto waitRetryCmdTimeoutMs = HcclGetWaitRetryCmdTimeout(ctx, retryCnt);
                    auto waitRetryCmdTimeout = std::chrono::milliseconds(waitRetryCmdTimeoutMs);
                    if ((std::chrono::steady_clock::now() - startTime) >= waitRetryCmdTimeout) {
                        HCCL_ERROR("hccl aicpu wait retry timeout[%u ms].", waitRetryCmdTimeoutMs);
                        errorCode = KfcError::kTimeout;
                        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
                    } else {
                        ret = HcclOpExecFsmWaitRetryProcess(ctx, state, errorCode, retryCnt);
                    }
                }
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_RETRY:
                ret = HcclOpExecFsmRetryProcess(ctx, state, errorCode, retryCnt, opParams, endSqePos);
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_END:
                return HcclOpExecFsmEndProcess(ctx, retryCnt, opParams);
            case HcclOpExecFSM::HCCL_OP_EXEC_STOP_LAUNCH:
                HCCL_DEBUG("[NsTest][AICPU] stop the kernel");
                if (!ctx->isStopLaunch) {
                        return HCCL_E_SUSPENDING;
                } else {
                        HCCL_RUN_INFO("[NsTest][AICPU] stop the kernel for stop command");
                        CopyCtxForBackGroundDfx(ctx);
                        if (UpdateOpExecStatus(ctx, state, KfcStatus::kStoplaunch, errorCode, 0) == HCCL_SUCCESS) {
                            return HCCL_E_SUSPENDING;
                        } else {
                            break;
                        }
                }
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR:
            default: {
                UpdateOpExecStatus(ctx, state, KfcStatus::kError, errorCode, retryCnt);
                return (ret == HCCL_SUCCESS) ? HCCL_E_INTERNAL : ret;
            }
        }
    }
    return HCCL_SUCCESS;
}

u32 AicpuHcclProcess::HcclGetWaitStopExecCmdTimeout()
{
    // NOTE：超时时间暂定10s
    return HCCL_AICPU_WAIT_HOST_BASE_TIME_MS;
}

u32 AicpuHcclProcess::HcclGetWaitRetryCmdTimeout(AicpuComContext *ctx, uint32_t retryCnt)
{
    if (retryCnt == 0) {
        return HCCL_AICPU_WAIT_HOST_BASE_TIME_MS + ctx->retryHoldTime;
    } else {
        return HCCL_AICPU_WAIT_HOST_BASE_TIME_MS + ctx->retryIntervalTime;
    }
}

HcclResult AicpuHcclProcess::UpdateOpExecStatus(AicpuComContext *ctx, HcclOpExecFSM &fsmState, KfcStatus state,
    KfcError &errorCode, uint32_t retryCnt)
{
    auto ret = AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, state, errorCode, retryCnt);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("SetOpExecStatus failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return ret;
}

bool AicpuHcclProcess::HcclOpCheckInplace(const AivAicpuOpParam &opParams)
{
    if (opParams.sendBuffer != opParams.recvBuffer) {
        return false;
    }

    const std::set<AicpuComType> HcclInplaceOpSet = { HCCL_CMD_ALLREDUCE,      HCCL_CMD_REDUCE,    HCCL_CMD_ALLGATHER,
                                                      HCCL_CMD_REDUCE_SCATTER, HCCL_CMD_ALLTOALLV, HCCL_CMD_ALLTOALLVC,
                                                      HCCL_CMD_ALLTOALL,       HCCL_CMD_GATHER,    HCCL_CMD_SCATTER };
    if (HcclInplaceOpSet.find(opParams.commType) != HcclInplaceOpSet.end()) {
        return true;
    }
    return false;
}

bool AicpuHcclProcess::HcclOpCheckSupportRetry(AicpuComType opType)
{
    const std::set<AicpuComType> HcclSupportRetryOpSet = {
        HCCL_CMD_BROADCAST, HCCL_CMD_ALLREDUCE,  HCCL_CMD_REDUCE,   HCCL_CMD_ALLGATHER, HCCL_CMD_REDUCE_SCATTER,
        HCCL_CMD_ALLTOALLV, HCCL_CMD_ALLTOALLVC, HCCL_CMD_ALLTOALL, HCCL_CMD_GATHER,    HCCL_CMD_SCATTER
    };
    return (HcclSupportRetryOpSet.find(opType) != HcclSupportRetryOpSet.end());
}

bool AicpuHcclProcess::HcclOpSupportRetry(AicpuComContext *ctx, AivAicpuOpParam &opParams)
{
    if (!ctx->retryEnable) {
        HCCL_INFO("hccl aicpu can not retry, enable[%u].", ctx->retryEnable);
        return false;
    }

    // 不支持inplace的通信算子重执行
    if (HcclOpCheckInplace(opParams)) {
        HCCL_INFO("hccl aicpu can not retry, opType[%u], sendBuffer[0x%016lx], recvBuffer[0x%016lx].",
            opParams.commType, opParams.sendBuffer, opParams.recvBuffer);
        return false;
    }

    if (HcclOpCheckSupportRetry(opParams.commType)) {
        return true;
    }
    return false;
}

HcclResult AicpuHcclProcess::CalcDataSize(HcclCMDType op, HcclDataType type, u64 count,
    u32 rankSize, u64 &inputSize, u64 &outputSize)
{
    u32 perDataSize = DataUnitSize(type);
    if (perDataSize == 0) {
        HCCL_ERROR("[AicpuHcclProcess][CalcDataSize] type [%u] DataUnitSize is 0", type);
        return HCCL_E_PARA;
    }

    switch (op) {
        case HcclCMDType::HCCL_CMD_ALLGATHER:
            inputSize = count * perDataSize;
            outputSize = rankSize * count * perDataSize;
            break;
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
        case HcclCMDType::HCCL_CMD_SCATTER:
            inputSize = rankSize * count * perDataSize;
            outputSize = count * perDataSize;
            break;
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
        case HcclCMDType::HCCL_CMD_BROADCAST:
        case HcclCMDType::HCCL_CMD_REDUCE:
        case HcclCMDType::HCCL_CMD_SEND:
        case HcclCMDType::HCCL_CMD_RECEIVE:
        default:
            inputSize = count * perDataSize;
            outputSize = count * perDataSize;
            break;
    }

    HCCL_DEBUG("[AicpuHcclProcess][CalcDataSize] perDataSize %u count %lu rankSize %u input %lu output %lu",
        perDataSize, count, rankSize, inputSize, outputSize);
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::CalcDataSizeV(hccl::OpParam &param, u32 rankSize)
{
    const HcclDataType type = param.GetDataType();
    u32 perDataSize = DataUnitSize(type);
    if (perDataSize == 0) {
        HCCL_ERROR("[AicpuHcclProcess][CalcDataSizeV] type[%u] DataUnitSize is 0", type);
        return HCCL_E_PARA;
    }

    const u32 rankId = param.srcRank;
    if (rankId >= rankSize) {
        HCCL_ERROR("[AicpuHcclProcess][CalcDataSizeV] rankId[%u] should not be bigger than rankSize[%u]", rankId,
            rankSize);
        return HCCL_E_PARA;
    }

    const HcclCMDType op = param.opType;
    switch (op) {
        case HcclCMDType::HCCL_CMD_ALLGATHER_V:
            {
                const u64 *counts = static_cast<u64 *>(param.VDataDes.counts);
                const u64 totalSize = std::accumulate(counts, counts + rankSize, 0) * perDataSize;
                param.inputSize = param.GetDataCount(rankId) * perDataSize;
                param.outputSize = totalSize;
                break;
            }
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V:
            {
                const u64 *counts = static_cast<u64 *>(param.VDataDes.counts);
                const u64 totalSize = std::accumulate(counts, counts + rankSize, 0) * perDataSize;
                param.inputSize = totalSize;
                param.outputSize = param.GetDataCount(rankId) * perDataSize;
                break;
            }
        default:
            HCCL_ERROR("[AicpuHcclProcess][CalcDataSizeV] op[%u] is not supported", op);
            return HCCL_E_PARA;
    }

    HCCL_DEBUG("[AicpuHcclProcess][CalcDataSizeV] opType[%u] perDataSize[%u] rankId[%u] rankSize[%u] input[%llu] "
        "output[%llu]", op, perDataSize, rankId, rankSize, param.inputSize, param.outputSize);
    return HCCL_SUCCESS;
}

u64 AicpuHcclProcess::CalcOpTilingVDataDesVDataLen(u32 rankSize)
{
    const u32 vFactor = 2;  // counts和displs 2个变长数组
    return vFactor * rankSize * sizeof(u64);
}

void AicpuHcclProcess::HcclUpdateOpIndex(AicpuComType opType, AicpuComContext *ctx)
{
    if (HcclOpCheckSupportRetry(opType)) {
        auto opIndex = ctx->opIndex + 1;
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, opIndex), opIndex);
    } else {
        // NOTE: send / recv / batchsendrecv 算子不是通信域内所有卡都参与，opIndex需要另行处理；重执行暂不支持该类算子
    }
    return;
}

HcclResult AicpuHcclProcess::HcclOpExecFsmInitProcess(AicpuComContext *ctx, HcclOpExecFSM &state,
    KfcError &errorCode, AicpuRpcServer &rpc, AivAicpuOpParam &opParams)
{
    HCCL_DEBUG("----------start AICPU_RpcServerUnfoldStageWait -------");
    rpc.CheckRcvAddrMsg(&opParams, 0);
    ctx->directlySendMainSteramSqe = true;

    HcclUpdateOpIndex(opParams.commType, ctx);
    opParams.opId.index = ctx->opIndex;
    if(ctx->endStopLaunch){
        HCCL_WARNING("[NsRecovery] Suspending status should not launch task");
        state = HcclOpExecFSM::HCCL_OP_EXEC_STOP_LAUNCH;
        return HCCL_SUCCESS;
    }
    auto ret = AicpuHdcUtils::InitOpExecStatus(ctx->kfcStatusTransferD2H, opParams.opId);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), true);
    if (ret == HCCL_SUCCESS) {
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_LAUNCH;
    } else {
        HCCL_ERROR("InitOpExecStatus failed, ret:%u", ret);
        errorCode = KfcError::kInner;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return ret;
}

HcclResult AicpuHcclProcess::HcclOpExecFsmLaunchProcess(AicpuComContext *ctx, HcclOpExecFSM &state,
    KfcError &errorCode, AivAicpuOpParam &opParams, uint32_t &beginSqePos, uint32_t &endSqePos)
{
    HCCL_DEBUG("hccl aicpu start launch task");
    auto ret = LaunchHcclOp(ctx, &opParams, beginSqePos, endSqePos);
    if (ret == HCCL_SUCCESS) {
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END;
    } else if (ret == HCCL_E_SUSPENDING) {
        HCCL_RUN_INFO("[NsRecovery][AICPU]hccl aicpu force stop in launch process");
        state = HcclOpExecFSM::HCCL_OP_EXEC_STOP_LAUNCH;
    } else {
        HCCL_ERROR("LaunchHcclOp failed, ret:%u", ret);
        errorCode = KfcError::kInner;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return ret;
}

HcclResult AicpuHcclProcess::HcclOpExecFsmWaitEndProcess(AicpuComContext *ctx, HcclOpExecFSM &state,
    KfcError &errorCode, uint32_t retryCnt)
{
    HCCL_DEBUG("hccl aicpu wait task finish.");
    bool isWaitTask = (ctx->debugMode == MC2_DEBUG_WAIT_COMM);
    auto ret = MC2AicpuUtils::WaitTaskFinish(ctx, isWaitTask);
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("hccl aicpu exec complete.");
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_END;
    } else if (ret == HCCL_E_SUSPENDING) {
        HCCL_RUN_INFO("[NsRecovery][AICPU]hccl aicpu force stop in launch loop");
        if (ctx->isStopLaunch == true) {
            state = HcclOpExecFSM::HCCL_OP_EXEC_STOP_LAUNCH;
        } else {
            CHK_RET(UpdateOpExecStatus(ctx, state, KfcStatus::kStoplaunch, errorCode, retryCnt));
            state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING;
        }
    } else {
        HCCL_ERROR("WaitTaskFinish failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return ret;
}

HcclResult AicpuHcclProcess::HcclOpExecFsmStoppingProcess(AicpuComContext *ctx, HcclOpExecFSM &state,
    KfcError &errorCode, uint32_t retryCnt)
{
    HCCL_DEBUG("hccl aicpu stopping.");
    if (TaskOrchestrator::IsTaskExceptionForHccs(ctx)) {
        HCCL_INFO("hccl aicpu recoverable task exception accurs.");
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED;
        return HCCL_SUCCESS;
    }

    KfcCommand cmd = KfcCommand::kNone;
    auto ret = AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("GetOpExecCtrlCmd failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }
    if (cmd == KfcCommand::kExit) {
        HCCL_WARNING("hccl aicpu exec fsm stop by exit cmd.");
        errorCode = KfcError::kExit;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    } else if ((cmd == KfcCommand::kStopExec)) {
        HCCL_INFO("hccl aicpu get stop exec cmd.");
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED;
    } else if ((cmd == KfcCommand::kNone) || (cmd == KfcCommand::kStopLaunch)) {
        HCCL_DEBUG("hccl aicpu wait for stop exec cmd.");
        // do nothing
    } else {
        HCCL_ERROR("GetOpExecCtrlCmd failed, invalid cmd[%u]", cmd);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::HcclOpExecFsmStoppedProcess(AicpuComContext *ctx, HcclOpExecFSM &state,
    KfcError &errorCode, uint32_t retryCnt, AivAicpuOpParam &opParams, uint32_t beginSqePos, uint32_t endSqePos)
{
    HCCL_DEBUG("hccl aicpu stop exec.");
    KfcCommand cmd = KfcCommand::kNone;
    auto ret = AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("GetOpExecCtrlCmd failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }

    if (cmd == KfcCommand::kExit) {
        HCCL_ERROR("hccl aicpu exec fsm stop by exit cmd.");
        errorCode = KfcError::kExit;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return HCCL_SUCCESS;
    }

    if (!HcclOpSupportRetry(ctx, opParams)) {
        HCCL_ERROR("hccl aicpu not support retry, enable[%u], commType[%u].", ctx->retryEnable, opParams.commType);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return HCCL_SUCCESS;
    }

    uint32_t sqHead = 0xFFFFFFFF;
    CHK_RET(QuerySqStatusByType(ctx->devId, ctx->streamInfo[ctx->rankId].sqId, DRV_SQCQ_PROP_SQ_HEAD, sqHead));
    if (sqHead == endSqePos) {
        HCCL_INFO("hccl aicpu record complete task is complete, can not retry. params: sqHead %u, beginSqePos %u "
                  "endSqePos %u",
            sqHead, beginSqePos, endSqePos);
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_END;
    } else if (sqHead == beginSqePos) {
        HCCL_ERROR(
            "hccl aicpu wait start task is not complete, can not retry. params: sqHead %u, beginSqePos %u endSqePos %u",
            sqHead, beginSqePos, endSqePos);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    } else {
        HCCL_INFO("hccl aicpu op is runing, can retry. params: sqHead %u, beginSqePos %u endSqePos %u", sqHead,
            beginSqePos, endSqePos);
        if (TaskOrchestrator::IsTaskExceptionForHccs(ctx)) {
            HCCL_INFO("hccl aicpu stop by sdma/write task exception, can retry.");
            errorCode = KfcError::kSdma;
        }
        CHK_RET(UpdateOpExecStatus(ctx, state, KfcStatus::kStopExec, errorCode, retryCnt));
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::HcclOpExecFsmWaitRetryProcess(AicpuComContext *ctx, HcclOpExecFSM &state,
    KfcError &errorCode, uint32_t retryCnt)
{
    HCCL_DEBUG("hccl aicpu wait for retry cmd.");
    KfcCommand cmd = KfcCommand::kNone;
    auto ret = AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("GetOpExecCtrlCmd failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }
    if (cmd == KfcCommand::kRetry) {
        HCCL_INFO("hccl aicpu recv retrey cmd from host.");

        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.pollStatus), dfx::PollStatus::kDefault);
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.cqeStatus), dfx::CqeStatus::kDefault);
        ret = ResetSqBuff(ctx);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("ResetSqBuff failed, ret:%u", ret);
            errorCode = KfcError::kInner;
            state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
            return ret;
        }
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_RETRY;
    } else if (cmd == KfcCommand::kExit) {
        errorCode = KfcError::kExit;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    } else {
        // do nothing
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::HcclOpExecFsmRetryProcess(AicpuComContext *ctx, HcclOpExecFSM &state,
    KfcError &errorCode, uint32_t &retryCnt, AivAicpuOpParam &opParams, uint32_t &endSqePos)
{
    HCCL_DEBUG("hccl retry launch task");
    retryCnt++;
    auto ret = RetryLaunchHcclOp(ctx, &opParams, endSqePos);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("RetryLaunchHcclOp failed, ret:%u", ret);
        errorCode = KfcError::kInner;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }
    errorCode = KfcError::kNone;
    CHK_RET(UpdateOpExecStatus(ctx, state, KfcStatus::kRuning, errorCode, retryCnt));
    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END;
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::HcclOpExecFsmEndProcess(AicpuComContext *ctx, uint32_t retryCnt, AivAicpuOpParam &opParams)
{
    auto ret =
        AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kEnd, KfcError::kNone, retryCnt);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), false);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, dfxExtendInfo.kfcStatus), dfx::KfcStatus::kOneFinished);
    MC2AicpuUtils::PrintBuffer(ctx, opParams);
    ctx->directlySendMainSteramSqe = false;
    HCCL_DEBUG("----------end AICPU_RpcServerUnfoldStageWait -------");

    return ret;
}

HcclResult AicpuHcclProcess::ResetSqBuff(AicpuComContext *ctx)
{
    CHK_RET(SqeContextUtils::ClearLocalBuff());
    SqeContext *sqeContext = GetSqeContext();
    u32 streamNum =  (ctx->multiServerFlag) ? 1 : ctx->rankNum;
    for (u32 i = 0; i < streamNum; i++) {
        auto &buff = sqeContext->buffPtr[i];
        CHK_RET(QuerySqStatusByType(ctx->devId, ctx->streamInfo[i].sqId, DRV_SQCQ_PROP_SQ_TAIL, buff.sqTail));
        CHK_RET(QuerySqStatusByType(ctx->devId, ctx->streamInfo[i].sqId, DRV_SQCQ_PROP_SQ_HEAD, buff.sqHead));
        HCCL_INFO("hccl aicpu reset stream buffer, sqid:%d head:%u tail:%u.", ctx->streamInfo[i].sqId, buff.sqHead, buff.sqTail);
    }
    HCCL_INFO("reset stream sq buffer success.");
    return HCCL_SUCCESS;
}


HcclResult AicpuHcclProcess::LaunchHcclOp(AicpuComContext *ctx, AivAicpuOpParam *commParam, uint32_t &beginSqePos,
    uint32_t &endSqePos)
{
    // 获取通信stream上首次下发的notify wait
    // task的尾指针，已便重执行stop时判断是否已执行该task，如果该task已执行完成则可支持通信重执行
    CHK_RET(QuerySqStatusByType(ctx->devId, ctx->streamInfo[ctx->rankId].sqId, DRV_SQCQ_PROP_SQ_TAIL, beginSqePos));

    // STARS调度执行到该通信算子时，会触发一次本地notify record触发通信算子在AICPU上展开、执行

    CHK_RET(DispatcherAicpu::AicpuUnfoldSignalWait(ctx->rankId, 0, DispatcherAicpu::IPC));
    CHK_RET(AicpuCcOpExe(commParam, nullptr, ctx));

    // AICPU上通信task下发完成后，在通信stream上紧跟着下发一个notify record，以通知通信主stream通信算子执行完成
    CHK_RET(DispatcherAicpu::AicpuUnfoldSignalRecord(ctx->rankId, 1, DispatcherAicpu::IPC));

    KfcCommand cmd = KfcCommand::kNone;
    if ((ctx->endStopLaunch == false) && (ctx->commOpenStatus == true)) {
        CHK_RET(AicpuHdcUtils::GetOpExecCtrlCmd(ctx->kfcControlTransferH2D, cmd));
        if (cmd == KfcCommand::NsStopLaunch) {
            AicpuUpdatComContextMumber(offsetof(AicpuComContext, endStopLaunch), true);
            AicpuUpdatComContextMumber(offsetof(AicpuComContext, isStopLaunch), true);
            return HCCL_E_SUSPENDING;
        }
    }
    // 启动通信task执行
    CHK_RET(TaskOrchestrator::ActiveRecordMain(GetActiveSqId(ctx)));

    CHK_RET(QuerySqStatusByType(ctx->devId, ctx->streamInfo[ctx->rankId].sqId, DRV_SQCQ_PROP_SQ_TAIL, endSqePos));

    HCCL_INFO("hccl aicpu launch hccl op task success. stream sqid:%d bigen:%u end:%u",
        ctx->streamInfo[ctx->rankId].sqId, beginSqePos, endSqePos);

    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::RetryLaunchHcclOp(AicpuComContext *ctx, AivAicpuOpParam *commParam, uint32_t &endSqePos)
{
    CHK_RET(AicpuCcOpExe(commParam, nullptr, ctx));

    // AICPU上通信task下发完成后，在通信stream上紧跟着下发一个notify record，以通知通信主stream通信算子执行完成
    CHK_RET(DispatcherAicpu::AicpuUnfoldSignalRecord(ctx->rankId, 1, DispatcherAicpu::IPC));

    // 启动通信task执行
    CHK_RET(TaskOrchestrator::ActiveRecordMain(GetActiveSqId(ctx)));

    CHK_RET(QuerySqStatusByType(ctx->devId, ctx->streamInfo[ctx->rankId].sqId, DRV_SQCQ_PROP_SQ_TAIL, endSqePos));

    HCCL_INFO("hccl aicpu retry launch hccl op task success. stream sqid:%d end:%u",
        ctx->streamInfo[ctx->rankId].sqId, endSqePos);
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::DealReturnValue(const AicpuComContext *ctx, const HcclResult ret) {
    if (ctx->isStopLaunch) {
        CopyCtxForBackGroundDfx(ctx);
        CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kStoplaunch,
                KfcError::kNone, 0));
        return HCCL_E_SUSPENDING;
    } else if (ctx->endStopLaunch) {
        return HCCL_E_SUSPENDING;
    } else {
        HCCL_ERROR("RunRpcServerApi faild");
        CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kError,
                KfcError::kInner, 0));
        return ret;
    }
}

HcclResult AicpuHcclProcess::RunSlaveRpcServerForApi(AicpuComContext *ctx)
{
    HCCL_INFO("----------start Slave Rpc Server For Api Hccl, ctx:%p ----------", ctx);
    if (ctx->devType != DevType::DEV_TYPE_910B) {
        HCCL_WARNING("Platform not support multi thread handle batch write, please use 910B platform.");
        return HCCL_SUCCESS;
    }
    CommonHcclMsg commonHcclMsg;
    int32_t sendSeqNum = -1;
    u32 threadId = g_sharedCtx.curThreadIdsOnCpu[MC2AicpuUtils::GetCpuId()];
    u64 startTimeStamp = GetCurCpuTimestamp();
    while (true) {
#if defined(__aarch64__) || defined(__amd64__)
        __asm__ __volatile__("nop");
#endif

        if (g_sharedCtx.taskFinishFlag) {
            HCCL_INFO("task is finish, salve process exit");
            break;
        }
        u8 needSendTotalNum = 0;
        if (threadId < g_sharedCtx.workedThreadNum && g_hcclMsgQueue.Peek(&commonHcclMsg) && commonHcclMsg.seqNum != sendSeqNum) {
            if (commonHcclMsg.commType == HCCL_CMD_BATCH_WRITE) {
                CHK_RET(ConcurrentPostSendWqe(commonHcclMsg, ctx, &needSendTotalNum));
                sendSeqNum = commonHcclMsg.seqNum;
            }
        }

        if (CheckTimeOut(startTimeStamp, SLAVE_EXECUTE_TIMEOUT)) {
            HCCL_ERROR("slave thread timeout.");
            return HCCL_E_TIMEOUT;
        }
    }
    return HCCL_SUCCESS;
}

// master的入口
HcclResult AicpuHcclProcess::AicpuRunRpcServerForApi(AicpuComContext *ctx) {
    static AicpuRpcServer rpc;
    rpc.Init(ctx->workSpaceAddr);
    ctx->acprof[g_proxLoopCnt].commInitEndTime = GetCurCpuTimestamp(true);
    const HcclResult ret = RunRpcServerApi(ctx, rpc);
    AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), false);
    if (ret != HCCL_SUCCESS) {
        return DealReturnValue(ctx, ret);
    } else {
        CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kEnd, KfcError::kNone, 0));
        return ret;
    }
}

HcclResult AicpuHcclProcess::AicpuRunRpcServer(AicpuComContext *ctx, KFCTask *taskInfo)
{
    HCCL_INFO("----------start AicpuRunRpcServer -------");
    // 启动RPC服务
    static AicpuRpcServer rpc;
    rpc.Init(ctx->workSpaceAddr, ctx->notifyOff, ctx->notifyBeginCnt, taskInfo);
    ctx->acprof[g_proxLoopCnt].commInitEndTime = GetCurCpuTimestamp(true);
    if (rpc.GetPreparePosition() == TASK_PREPARE_KERNEL) {
        auto ret = RunRpcServerApi(ctx, rpc);
        AicpuUpdatComContextMumber(offsetof(AicpuComContext, isOpLaunch), false);
        if (ret != HCCL_SUCCESS) {
            return DealReturnValue(ctx, ret);
        } else {
            CHK_RET(AicpuHdcUtils::SetOpExecStatus(ctx->kfcStatusTransferD2H, KfcStatus::kEnd, KfcError::kNone, 0));
            return ret;
        }
    }

    if (rpc.GetTaskType() == HCCL_KFC_TASK_HCCL_ONLY_EXE) {
        auto ret = AICPU_RpcServerUnfoldStageWait(ctx, rpc);
        if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_SUSPENDING)) {
            HCCL_ERROR("AicpuRpcStageWait faild, commType:%d, reducekind:%d, totalCnt:%lu, totalTurnCnt:%u",
                ctx->commType, ctx->reducekind, ctx->totalCnt, ctx->totalTurnCnt);
            return ret;
        }
        if (ret == HCCL_E_SUSPENDING) {
            HCCL_RUN_INFO("[NsRecovery][AICPU] Suspending");
            return ret;
        }
    } else if (ctx->devType == DevType::DEV_TYPE_310P1 || ctx->devType == DevType::DEV_TYPE_310P3) {
        auto ret = RunRpcServerTwoStageWait(ctx, rpc);
        if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_SUSPENDING)) {
            HCCL_ERROR("RunRpcServerTwoStageWait faild, commType:%d, reducekind:%d, totalCnt:%lu, totalTurnCnt:%u",
                ctx->commType, ctx->reducekind, ctx->totalCnt, ctx->totalTurnCnt);
            return ret;
        }
        if (ret == HCCL_E_SUSPENDING) {
            HCCL_RUN_INFO("[NsRecovery][MC2] Suspending");
            return ret;
        }
    } else {
        auto ret = TryRunRpcServerOneStageWait(ctx, rpc);
        if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_SUSPENDING)) {
            HCCL_ERROR("TryRunRpcServerOneStageWait faild, commType:%d, reducekind:%d, totalCnt:%lu, totalTurnCnt:%u",
                ctx->commType, ctx->reducekind, ctx->totalCnt, ctx->totalTurnCnt);
            return ret;
        }
        if (ret == HCCL_E_SUSPENDING) {
            HCCL_RUN_INFO("[NsRecovery][MC2] Suspending");
            return ret;
        }
    }

    HCCL_INFO("----------end AicpuRunRpcServer -------");
    return HCCL_SUCCESS;
}

u32 AicpuHcclProcess::AicpuRpcResInitV2(HcclOpResParam *commParam, bool isCustom)
{
    HCCL_DEBUG("[AicpuHcclProcess][AicpuRpcResInitV2]Entry AicpuRpcResInitV2 process-------");
    hccl::HcclCommAicpu *commAicpu = nullptr;
    HcclResult ret = HCCL_SUCCESS;
    std::string group = commParam->hcomId;
    CHK_RET(AicpuCreateCommbyGroup(group, &commAicpu));
    if (commAicpu == nullptr) {
        HCCL_ERROR("[AicpuHcclProcess][AicpuRpcResInitV2]commAicpu is null group[%s]", group.c_str());
        return 1U;
    }
    ret = commAicpu->Init(commParam, isCustom);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AicpuHcclProcess][AicpuRpcResInitV2]errNo[0x%016llx] Failed to init comm resource group[%s]",
            HCCL_ERROR_CODE(ret),
            group.c_str()),
        ret);
    HCCL_DEBUG("[AicpuHcclProcess][AicpuRpcResInitV2]AicpuRpcResInitV2 process end-------");
    CHK_RET(hrtHalGetDeviceType(commAicpu->GetDevId(), g_devType));
    HCCL_INFO("[AicpuHcclProcess][AicpuRpcResInitV2] get PlatformVersion %u, %u", static_cast<u32>(g_devType),
        commAicpu->GetDevId());
    AicpuComContext *ctx = AicpuGetComContext();
    CallMC2MaintenanceThread(ctx);

    return 0;
}

HcclResult AicpuHcclProcess::AicpuCreateCommbyGroup(const std::string &group, hccl::HcclCommAicpu **aicpuCommPtr)
{
    std::unique_lock<std::mutex> lock(g_commAicpuInfo.commAicpuMapMutex);
    auto iter = g_commAicpuInfo.commMap.find(group);
    if (iter == g_commAicpuInfo.commMap.end()) {
        std::shared_ptr<hccl::HcclCommAicpu> aicpuComm;
        EXECEPTION_CATCH((aicpuComm = std::make_shared<hccl::HcclCommAicpu>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(aicpuComm);
        g_commAicpuInfo.commMap[group] = {aicpuComm, false};
        HCCL_INFO("[AicpuCreateCommbyGroup]Create new comm group [%s]", group.c_str());
        *aicpuCommPtr = aicpuComm.get();
        return HCCL_SUCCESS;
    }

    HCCL_ERROR(
        "[AicpuHcclProcess][AicpuCreateCommbyGroup]errNo[0x%016llx] Repeated initialization comm resource group[%s]",
        HCCL_ERROR_CODE(HCCL_E_INTERNAL), group.c_str());
    return HCCL_E_INTERNAL;
}

std::mutex& AicpuHcclProcess::AicpuGetCommMutex()
{
    return g_commAicpuInfo.commAicpuMapMutex;
}

hccl::HcclCommAicpu *AicpuHcclProcess::AicpuGetCommbyGroup(const std::string &group)
{
    auto startTime = std::chrono::steady_clock::now();
    constexpr u32 pollIntervalUs = 10; // 轮询间隔10us
    constexpr u32 pollTimeoutMs = 10; // 轮询超时时间10ms
    auto waitPollTimeOutMs = std::chrono::milliseconds(pollTimeoutMs);

    while (true) {
        std::unique_lock<std::mutex> lock(g_commAicpuInfo.commAicpuMapMutex);
        auto iter = g_commAicpuInfo.commMap.find(group);
        if (iter == g_commAicpuInfo.commMap.end()) {
            HCCL_ERROR("[AicpuHcclProcess] exist group size is [%u]", g_commAicpuInfo.commMap.size());
            auto curIter = g_commAicpuInfo.commMap.begin();
            int i = 0;
            while (curIter != g_commAicpuInfo.commMap.end()) {
                HCCL_ERROR("[AicpuHcclProcess] exist group idx is [%d] key[%s] value", i, curIter->first.c_str());
                curIter++;
            }
            return nullptr;
        }
        if (iter->second.second) {
            if ((std::chrono::steady_clock::now() - startTime) >= waitPollTimeOutMs) {
                HCCL_ERROR("[AicpuGetCommbyGroup]poll timeout, comm group [%s] has been used, last executed op: %s",
                    group.c_str(), iter->second.first->GetExcuteOp().c_str());
                return nullptr;
            }

            HCCL_WARNING("[AicpuGetCommbyGroup]comm group [%s] has been used, last executed op: %s",
                group.c_str(), iter->second.first->GetExcuteOp().c_str());
            lock.unlock();

            usleep(pollIntervalUs);
            continue;
        }
        iter->second.second = true;
        return iter->second.first.get();
    }
    return nullptr;
}

void AicpuHcclProcess::AicpuReleaseCommbyGroup(const std::string &group)
{
    std::unique_lock<std::mutex> lock(g_commAicpuInfo.commAicpuMapMutex);
    auto iter = g_commAicpuInfo.commMap.find(group);
    if (iter == g_commAicpuInfo.commMap.end()) {
        return;
    }
    iter->second.second = false;
}

HcclResult AicpuHcclProcess::AicpuGetCommAll(std::vector<std::pair<std::string, HcclCommAicpu *>> &aicpuCommInfo)
{
    for (auto &kv : g_commAicpuInfo.commMap) {
        aicpuCommInfo.push_back({kv.first, kv.second.first.get()});
    }
    return HCCL_SUCCESS;
}

void AicpuHcclProcess::AicpuDestoryCommbyGroup(const std::string &group)
{
    auto iter = g_commAicpuInfo.commMap.find(group);
    if (iter == g_commAicpuInfo.commMap.end()) {
        HCCL_ERROR("[AicpuHcclProcess][%s]Group[%s] is not exist", __func__, group.c_str());
        return;
    }
    if (iter->second.second == true) {
        HCCL_WARNING("[AicpuHcclProcess][%s]comm group [%s] has been used.", __func__, group.c_str());
        return;
    }
    g_commAicpuInfo.commMap.erase(group);
    HCCL_INFO("[AicpuHcclProcess][%s]Destory comm group [%s] success.", __func__, group.c_str());
}

void AicpuHcclProcess::GetCommonHcclMsg(HcclMsg *hcclMsg, CommonHcclMsg *commonHcclMsg, u64 tilingBase)
{
    if (hcclMsg->version >= MC2_MSG_VERSION_FOR_TILING_API) {
        HcclMsgV1 *hcclMsgV1 = reinterpret_cast<HcclMsgV1 *>(hcclMsg);
        const size_t copyOffset = offsetof(HcclMsgV1, hcclDataType);
        (void)memcpy_s(commonHcclMsg, copyOffset, hcclMsgV1, copyOffset);
        if (hcclMsg->version == MC2_MSG_VERSION_FOR_STATIC_GRAPH && tilingBase != 0UL) {
            commonHcclMsg->ccOpTilingData = hcclMsgV1->ccOpTilingData + tilingBase;
        } else {
            commonHcclMsg->ccOpTilingData = hcclMsgV1->ccOpTilingData;
        }
        commonHcclMsg->valid = hcclMsgV1->valid;
        commonHcclMsg->hcclDataType = hcclMsgV1->hcclDataType;
        commonHcclMsg->repeatCnt = hcclMsgV1->repeatCnt;
        commonHcclMsg->selfHandleID = hcclMsgV1->selfHandleID;
        commonHcclMsg->seqNum = hcclMsgV1->seqNum;
        commonHcclMsg->version = hcclMsgV1->version;
        commonHcclMsg->xorCheck = hcclMsgV1->xorCheck;
    } else {
        (void)memcpy_s(commonHcclMsg, sizeof(CommonHcclMsg), hcclMsg, sizeof(HcclMsg));
        commonHcclMsg->ccOpTilingData = 0UL;
    }
}

const std::unordered_set<std::string> STEP_SIZE_SUPPORT_LIST = {
    "AlltoAll=level0:fullmesh;level1:pairwise"
};

HcclResult AicpuHcclProcess::ParseCcOpTilingData(CommonHcclMsg *commonHcclMsg, int32_t groupIdx)
{
    const u8 version = commonHcclMsg->version;
    HCCL_INFO("Hccl client message version %u", version);
    AicpuRpcServerV2 *rpc = AicpuHcclProcess::GetCommRpcServer(groupIdx);
    rpc->SetStepSize(0U);
    rpc->SetTotalStep((0U));
    if (version < MC2_MSG_VERSION_FOR_TILING_API) {
        return HCCL_SUCCESS;
    }

    Mc2CcTilingInner *mc2CcTiling = reinterpret_cast<Mc2CcTilingInner *>(commonHcclMsg->ccOpTilingData);
    if (mc2CcTiling == nullptr) {
        HCCL_ERROR("Tiling is nullptr.");
        return HCCL_E_PARA;
    }

    // 校验tiling的groupName与当前接收数据的group 的index是否一致
    int32_t tilingGroupIdx = GetComGroupIdx(std::string(mc2CcTiling->groupName));
    if (tilingGroupIdx != groupIdx) {
        HCCL_ERROR("Failed to check groupName %s, groupIdx %d, tiling GroupIdx %d",
            mc2CcTiling->groupName, groupIdx, tilingGroupIdx);
        return HCCL_E_PARA;
    }

    HcclOpResParam *commParam = AicpuHcclProcess::GetCommAicpuResInst(groupIdx);
    std::string curAlgName;
    CHK_PRT_RET(!SelectAlgName(mc2CcTiling->algConfig, commParam->topoInfo.topoType, curAlgName),
                HCCL_ERROR("Failed to select algname."), HCCL_E_PARA);
    AlgType algType;
    HcclCommAicpu *commAicpu = AicpuHcclProcess::GetCommAicpuCommInst(groupIdx);
    SelectAlgType(commAicpu, mc2CcTiling->algConfig, commParam->topoInfo.moduleNum, algType);
    std::string curTag = std::string(mc2CcTiling->groupName) + std::to_string(mc2CcTiling->opType);
    commAicpu->SetCommInfoCtx(static_cast<u8>(mc2CcTiling->opType), curAlgName, curTag, algType);

    if (mc2CcTiling->stepSize > 0U) {
        CHK_PRT_RET(STEP_SIZE_SUPPORT_LIST.find(mc2CcTiling->algConfig) == STEP_SIZE_SUPPORT_LIST.end(),
            HCCL_ERROR("Alg %s is not supported when step size is %u.", mc2CcTiling->algConfig, mc2CcTiling->stepSize),
            HCCL_E_PARA);
        rpc->SetStepSize(mc2CcTiling->stepSize);
        rpc->SetTotalStep(commParam->rankSize);
    }
    return HCCL_SUCCESS;
}

bool AicpuHcclProcess::CheckNsCommand(hccl::HcclCommAicpu *comm) {
    KfcCommand cmd;
    if (comm->BackGroundGetCmd(cmd) != HCCL_SUCCESS || cmd != KfcCommand::NsStopLaunch ||
        comm->BackGroundSetStatus(KfcStatus::kStoplaunch) != HCCL_SUCCESS) {
        return false;
    }
    comm->SetCommRecoveryFlag(true);
    comm->SetNsStopLaunchStatus(true);
    HCCL_WARNING("N second stop Launch for recv stop launch cmd.");
    return true;
}

HcclResult AicpuHcclProcess::CheckRestartError(hccl::HcclCommAicpu *comm) {
    // 支持重执行时，检测是否有可重执行的sdma异常, 或者kStopLaunch命令
    if (comm->GetOpRetryEnable()) {
        if (comm->IsTaskExceptionForHccs()) {
            HCCL_WARNING("MC2 restart Sdma error happened.");
            return HCCL_E_SUSPENDING;
        }

        KfcCommand cmd = KfcCommand::kNone;
        CHK_RET(comm->BackGroundGetCmd(cmd));
        if (cmd == KfcCommand::kStopLaunch) {
            HCCL_WARNING("MC2 restart receive kfc command stop launch.");
            return HCCL_E_SUSPENDING;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::RpcServerPreCheck(AicpuRpcServerV2 *rpc, hccl::HcclCommAicpu *comm, bool &finalizeFlag)
{
    if (CheckNsCommand(comm)) {
        finalizeFlag = true;
        rpc->SetNeedRetryFlag(false);
        return HCCL_E_AGAIN;
    }
    if (CheckRestartError(comm) == HCCL_E_SUSPENDING) {
        return HCCL_E_SUSPENDING;
    }
    if (comm->GetDfxExtendInfo()->pollStatus == dfx::PollStatus::kStopAsException) {
        if (comm->GetOpRetryEnable() && comm->IsTaskExceptionForHccs()) {
            HCCL_WARNING("MC2 restart Sdma error happened.");
            return HCCL_E_SUSPENDING;
        }
        HCCL_ERROR("MC2 hccl aicpu exec failed, for task exception.");
        return HCCL_E_INTERNAL;
    }
    if (rpc->GetIsFinalize()) {
        if (comm->CheckCommAllFinish()) {
            finalizeFlag = true;
            rpc->WriteFinishWhenAllFinalize();
        }
        return HCCL_E_AGAIN;
    }
    return HCCL_SUCCESS;
}

static constexpr u64 BARRIER_TIMEOUT = static_cast<u64>(NSEC_PER_SEC) * 60UL;
HcclResult AicpuHcclProcess::BarrierProcess(u32 groupIdx, u32 queueId, BarrierStatus &status)
{
    static std::atomic<u32> threadFinishCnt;
    AicpuRpcServerV2 *rpc = AicpuHcclProcess::GetCommRpcServer(groupIdx);
    BarrierInfo *barrierInfos = rpc->GetBarrierInfoByGroupIdx(groupIdx);
    BarrierStatus &selfFlag = barrierInfos[queueId].status;
    if (selfFlag == BarrierStatus::NO_BARRIER) {
        barrierInfos[queueId].lastTimeStamp = GetCurCpuTimestamp();
        status = BarrierStatus::NO_BARRIER;
        return HCCL_SUCCESS;
    }

    u32 &barrierFinishCnt = rpc->GetBarrierFinishCnts()[MC2AicpuUtils::GetBlockIdx()];
    if (selfFlag == BarrierStatus::SELF_BARRIER) {
        HcclCommAicpu *comm = AicpuHcclProcess::GetCommAicpuCommInst(groupIdx);
        if (comm->CheckCommSlaveStreamFinish(queueId)) {
            barrierInfos[queueId].lastTimeStamp = GetCurCpuTimestamp();
            selfFlag = BarrierStatus::INTER_BARRIER;
            ++barrierFinishCnt;
            HCCL_INFO("[%s][Queue %u]All tasks in queue are finished, local count is %u.",
                      __func__, queueId, barrierFinishCnt);
        }
    }

    const u32 queNum = rpc->GetTotalQueueNum();
    if (selfFlag == BarrierStatus::INTER_BARRIER) {
        u32 start = 0U;
        u32 end = 0U;
        rpc->GetLocalQueueRange(start, end);
        if (barrierFinishCnt == end + 1U - start) {
            barrierFinishCnt = 0U;
            if (threadFinishCnt.fetch_add(1U, std::memory_order_relaxed) == MC2AicpuUtils::GetBlockNum() - 1U) {
                threadFinishCnt.store(0U);
                rpc->ClearBarrierStatus();
                HCCL_INFO("[%s]Barrier finishes in all threads and all infos are cleared.", __func__);
                return HCCL_SUCCESS;
            }
            barrierInfos[queueId].lastTimeStamp = GetCurCpuTimestamp();
        }
    }

    status = selfFlag;
    const u64 ts = GetCurCpuTimestamp();
    if (barrierInfos[queueId].lastTimeStamp != 0UL && ts - barrierInfos[queueId].lastTimeStamp > BARRIER_TIMEOUT) {
        HCCL_ERROR("[%s]Timeout when checking barrier status for channel %u, from %llu to %llu.",
                   __func__, queueId, barrierInfos[queueId].lastTimeStamp, ts);
        for (u32 i = 0U; i < queNum; ++i) {
            HCCL_ERROR("[%s][Queue %u]Barrier status %u.", __func__, i, static_cast<u32>(barrierInfos[i].status));
        }
        return HCCL_E_AGAIN;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::BatchWriteProcess(CommonHcclMsg &msg, AicpuRpcServerV2 &rpc, hccl::HcclCommAicpu &comm,
                                               HcclOpResParam &param)
{
    hccl::OpParam opParam;
    PrepareOpParam(&opParam, &msg, rpc, &comm);
    static hccl::AlgResourceResponse *algResResponse = nullptr;
    if (UNLIKELY(algResResponse == nullptr)) {
        const std::string tag = comm.GetCommId() + std::to_string(HCCL_CMD_BATCH_WRITE) +
                std::string("_mc2") + std::string(BATCH_WRITE_ALG_NAME) + std::string("_device");
        std::unique_ptr<hccl::CollExecutorBase> executor;
        CHK_RET(comm.GetAlgResponseRes(tag, BATCH_WRITE_ALG_NAME, opParam, &param, executor, algResResponse));
    }
    const u64 ts = GetCurCpuTimestamp();
    while (algResResponse->slaveStreams.empty()) {
        if (UNLIKELY(GetCurCpuTimestamp() - ts > static_cast<u64>(NSEC_PER_SEC))) {
            HCCL_ERROR("[%s]Timeout during batchwrite initialization.", __func__);
        }
    }
    HcclResult ret = comm.OrchestrateSdmaSqe(opParam);
    HcclCommProf::GetCurrentAicpuProf()->workCnt++;
    return ret;
}

HcclResult AicpuHcclProcess::RunRpcServerLoopProcess(u32 groupIdx, bool &finalizeFlag)
{
    HcclMsg hcclMsg;
    CommonHcclMsg commonHcclMsg;
    AicpuRpcServerV2 *rpc = AicpuHcclProcess::GetCommRpcServer(groupIdx);
    HcclCommAicpu *comm = AicpuHcclProcess::GetCommAicpuCommInst(groupIdx);
    HcclOpResParam *commParam = AicpuHcclProcess::GetCommAicpuResInst(groupIdx);
    u32 start = 0U;
    u32 end = 0U;
    rpc->GetLocalQueueRange(start, end);
    const u64 tilingBase = rpc->GetTilingBaseAddr();
    HcclResult ret;
    do {
        ret = RpcServerPreCheck(rpc, comm, finalizeFlag);
        if (ret == HCCL_E_AGAIN) {
            return HCCL_SUCCESS;
        } else if (ret != HCCL_SUCCESS) {
            return ret;
        }

        HcclMsg *msgLists[MAX_QUE_NUM];
        u32 msgListNum;
        rpc->GetMsgWorkSpace(msgLists, msgListNum);
        SetMsgEnableFlag(groupIdx, false);
        for (u32 i = 0U; i < msgListNum; ++i) {
            const u32 queIdx = start + i;
            if (rpc->GetIsFinalize(queIdx)) {
                continue;
            }

            BarrierStatus status;
            CHK_PRT_RET(BarrierProcess(groupIdx, queIdx, status) != HCCL_SUCCESS,
                rpc->PrintAllHcclMsgArea(commParam->rankSize), HCCL_E_INTERNAL);

            if (status != BarrierStatus::NO_BARRIER) {
                SetMsgEnableFlag(groupIdx, true);
                continue;
            }

            uint32_t currMsgPos = rpc->GetMsgPos(queIdx);
            if (!rpc->ReadAddrMsg(&hcclMsg, msgLists[i], queIdx, currMsgPos, commParam->rankSize)) {
                if (rpc->IsExceedLimit(hcclMsg.commType, commParam->rankSize)) {
                    HCCL_ERROR("The number[%u] of ranks exceeds the 256p limit supported by "
                               "the ALLTOALL/ALLTOALLV algorithm.", commParam->rankSize);
                    return HCCL_E_INTERNAL;
                }
                AddMsgInValidCount(groupIdx);
                if (GetMsgInValidCount(groupIdx) % LOGCOUNT_PRINT_TIMEOUT == 0) {
                    HCCL_DEBUG("Fail to get msg, addr is %p, queue %u, msgPos %u, group %s",
                                msgLists[i], queIdx, currMsgPos, comm->GetCommId().c_str());
                }
                if (rpc->IsPrintLog()) {
                    LogControl logControl(false, true);
                    comm->PrintTaskExceptionAllComm();
                }
                continue;
            }

            SetMsgStartTime(groupIdx);
            ClearMsgInValidCount(groupIdx);
            SetMsgEnableFlag(groupIdx, true);

            GetCommonHcclMsg(&hcclMsg, &commonHcclMsg, tilingBase);
            HCCL_INFO("Process message queue %u pos %u seq num %u type %u group %s.", queIdx, currMsgPos,
                commonHcclMsg.seqNum, commonHcclMsg.commType, comm->GetCommId().c_str());
            if (commonHcclMsg.commType == HCCL_CMD_FINALIZE) {
                if (HcclCommProf::IsDebugModeEquals(MC2_DEBUG_PRINT_BUFF)) {
                    rpc->PrintAllHcclMsgAreaData();
                }
                rpc->SetIsFinalize(queIdx, true);
                if (rpc->GetTotalQueueNum() == 0U) {
                    Stream curStream = comm->GetMainStrem();
                    rpc->ResetCommitTaskAdd(comm->GetDispatcher(), &curStream);
                    comm->ActiveMainStreamTask();
                }
                SetExpectPrepareId(queIdx, 0U);
                continue;
            } else if (commonHcclMsg.commType == HCCL_CMD_INTER_GROUP_SYNC) {
                ret = AddTaskForGroupSyncMsg(comm, &commonHcclMsg, rpc, groupIdx);
                if (ret == HCCL_E_UNAVAIL) {
                    SetMsgEnableFlag(groupIdx, false);
                    rpc->SetNeedRetryFlag(true);
                    continue;
                } else if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            } else if (commonHcclMsg.commType == HCCL_CMD_BARRIER) {
                rpc->GetBarrierInfoByGroupIdx(groupIdx)[queIdx].status = BarrierStatus::SELF_BARRIER;
            } else {
                ret = rpc->ProcessExpectPrepareMsg(commonHcclMsg.seqNum, GetExpectPrepareId(queIdx));
                if (ret == HCCL_E_UNAVAIL) {
                    SetMsgEnableFlag(groupIdx, false);
                    rpc->SetNeedRetryFlag(true);
                    continue;
                } else if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                rpc->SetNeedRetryFlag(false);
                rpc->SetMsgRepeatCnt(commonHcclMsg.repeatCnt);
                rpc->SetMsgHandlePos(currMsgPos, commonHcclMsg.selfHandleID);
                if (commonHcclMsg.commType == HCCL_CMD_BATCH_WRITE) {
                    CHK_RET(BatchWriteProcess(commonHcclMsg, *rpc, *comm, *commParam));
                } else {
                    CHK_RET(ParseCcOpTilingData(&commonHcclMsg, groupIdx));
                    CHK_RET(IsSupportRDMAReduce(commonHcclMsg.commType, commonHcclMsg.hcclDataType,
                                                commonHcclMsg.opType));
                    CHK_RET(AddTaskForHcclMsgV2(comm, rpc, &commonHcclMsg, commParam));
                }
                SetExpectPrepareId(queIdx, commonHcclMsg.seqNum + 1U);
            }
            rpc->SetMsgPos(queIdx, (currMsgPos + 1) % AC_MSG_CNT);
        }
    } while (CheckMsgEnableFlag(groupIdx));
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::SetNsOpStatus(uint32_t groupNum, bool state)
{
    for (uint32_t i = 0; i < groupNum; i++) {
        hccl::HcclCommAicpu *comm = AicpuHcclProcess::GetCommAicpuCommInst(i);
        if (comm != nullptr) {
            comm->SetNsOpStatus(state);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::CheckNsStopLaunchStatus(uint32_t groupNum)
{
    for (uint32_t i = 0; i < groupNum; i++) {
        hccl::HcclCommAicpu *comm = AicpuHcclProcess::GetCommAicpuCommInst(i);
        if (comm != nullptr && comm->GetNsStopLaunchStatus()) {
            return HCCL_E_SUSPENDING;
        }
    }
    return HCCL_SUCCESS;
}

void AicpuHcclProcess::ResetRestartParam(RestartParam &restartParam)
{
    restartParam.restartCnt++;
    restartParam.restartFlag = false;
    restartParam.consultationAllEnd = 0;
    for (uint32_t i = 0; i < MAX_COMM_CTX_NUM; i++) {
        restartParam.consultationResult[i] = false;
        restartParam.linkChanged[i] = false;
        restartParam.fsmState[i] = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END;
        restartParam.errorCode[i] = KfcError::kNone;
    }
}

bool AicpuHcclProcess::GetOpRetryEnable(uint32_t groupNum)
{
    for (uint32_t i = 0; i < groupNum; i++) {
        hccl::HcclCommAicpu *comm = AicpuHcclProcess::GetCommAicpuCommInst(i);
        if (comm == nullptr || !comm->GetOpRetryEnable()) {
            return false;
        }
    }
    return true;
}

std::string AicpuHcclProcess::GetNewTag(uint32_t groupIdx)
{
    hccl::HcclCommAicpu *comm = AicpuHcclProcess::GetCommAicpuCommInst(groupIdx);
    AicpuRpcServerV2 *rpc = AicpuHcclProcess::GetCommRpcServer(groupIdx);
    uint32_t currMsgPos = rpc->GetMsgPos();
    currMsgPos = currMsgPos > 0 ? currMsgPos - 1 : currMsgPos;
    HcclMsg *msgLists[MAX_QUE_NUM];
    u32 msgListNum;
    rpc->GetMsgWorkSpace(msgLists, msgListNum);
    CommInfoCtx curCtx;
    comm->GetCommInfoCtx(msgLists[0U][currMsgPos].commType, curCtx);
    return curCtx.tag + "_mc2" + curCtx.algName + "_device";;
}

HcclResult AicpuHcclProcess::RestartProcessConsulation(RestartParam &restartParam, bool &finalizeAllEnd, bool *finalizeMask, uint32_t groupNum)
{
    for (uint32_t i = 0; i < groupNum; i++) {
        if (restartParam.consultationResult[i]) {
            continue;
        }
        hccl::HcclCommAicpu *comm = AicpuHcclProcess::GetCommAicpuCommInst(i);
        if (comm == nullptr) {
            HCCL_ERROR("Failed to obtain the AICPU communication domain pointer."
                "Check whether the parameters are correct.");
            return HCCL_E_PARA;
        }
        std::string newTag = GetNewTag(i);
        HcclResult ret = comm->Mc2RetryProcess(restartParam, i);
        if (ret == HCCL_SUCCESS) {
            if (restartParam.consultationResult[i]) {
                HCCL_RUN_INFO("[MC2][AICPU]MC2 restart process success, groupIdx %u , tag %s", i, newTag.c_str());
                restartParam.consultationAllEnd++;
            }
        } else {
            // 重执行协商流程失败，直接返回错误
            HCCL_ERROR("[MC2][AICPU]MC2 restart process groupIdx %u failed at state %u ret is %u tag is %s", i, restartParam.fsmState[i], ret, newTag.c_str());
            return ret;
        }
    }

    // 全部协商重执行完成
    if (restartParam.consultationAllEnd >= groupNum) {
        HCCL_RUN_INFO("[MC2][AICPU]MC2 restart process all group success, reset param and write restart");
        SetExpectPrepareId(0U, 0U);
        ResetRestartParam(restartParam);
        finalizeAllEnd = false;
        for (uint32_t i = 0; i < groupNum; i++) {
            // 重置结束标志
            finalizeMask[i] = false;
            // 重置rpc
            AicpuRpcServerV2 *rpc = AicpuHcclProcess::GetCommRpcServer(i);
            rpc->Reset();
            rpc->WriteRestartFlag();
            AicpuHcclProcess::SetMsgStartTime(i);
            HCCL_INFO("MC2 restart process reset rpc param end. groupIndex = %u", i);
        }
        AicpuHcclProcess::SetKernelStartTime();
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::RunRpcServerInnerProcessV2(uint32_t groupNum)
{
    const bool retryEnable = GetOpRetryEnable(groupNum);
    RestartParam restartParam;
    auto opStartTime = std::chrono::steady_clock::now();
    bool finalizeMask[MAX_COMM_CTX_NUM] = {false, false, false};
    AicpuHcclProcess::SetKernelStartTime();
    HcclCommProf::GetCurrentAicpuProf()->commInitEndTime = GetCurCpuTimestamp(true);
    if (CheckNsStopLaunchStatus(groupNum) != HCCL_SUCCESS) {
        HCCL_WARNING("the op should not be launched in the suspending status");
        return HCCL_E_SUSPENDING;
    }
    CHK_RET(SetNsOpStatus(groupNum, true));
    while (true) {
        bool finishFlag = true;
        for (uint32_t i = 0; i < groupNum; i++) {
            if (finalizeMask[i]) {
                continue;
            }
            finishFlag = false;
            if (restartParam.restartFlag) {
                continue;
            }
            HcclResult res = RunRpcServerLoopProcess(i, finalizeMask[i]);
            if (res == HCCL_E_SUSPENDING) {
                if (retryEnable) {
                    restartParam.restartFlag = true;
                    break;
                }
                HCCL_RUN_INFO("[MC2][Restart]Mc2 can not retry, not all comm retryEnable are true");
            } else if (res != HCCL_SUCCESS) {
                HCCL_ERROR("RPC server failed to run.");
                return res;
            }
        }

        if (restartParam.restartFlag && MC2AicpuUtils::GetBlockIdx() == 0U) {
            HcclResult res = RestartProcessConsulation(restartParam, finishFlag, finalizeMask, groupNum);
            if (res != HCCL_SUCCESS) {
                HCCL_ERROR("[MC2][AICPU]MC2 restart process failed, restartCnt = %u, res = %u",
                           restartParam.restartCnt, res);
                RecordReportStatus(groupNum, dfx::ReportStatus::kRetryFail);
                return res;
            }
        }
        // 全部结束
        if (finishFlag) {
            HCCL_INFO("RPC server process ends.");
            HcclCommProf::GetCurrentAicpuProf()->receiveFinalizeTime = GetCurCpuTimestamp(true);
            CHK_RET(SetNsOpStatus(groupNum, false));
            if (restartParam.restartCnt > 0) {
                auto opEndTime = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(opEndTime - opStartTime).count();
                HCCL_RUN_INFO("[MC2][AICPU]MC2 restart exec success, restartCnt = %u, take time = %ld s", restartParam.restartCnt, duration);
                RecordReportStatus(groupNum, dfx::ReportStatus::kRetrySuccess);
            }
            return HCCL_SUCCESS;
        }
        // 消息超时或总执行时间超时
        if (CheckMsgTimeOut() || CheckKernelTimeOut()) {
            HCCL_ERROR("RPC server process Timeout.");
            for (uint32_t i = 0; i < groupNum; i++) {
                AicpuRpcServerV2 *rpc = AicpuHcclProcess::GetCommRpcServer(i);
                HcclOpResParam *commParam = AicpuHcclProcess::GetCommAicpuResInst(i);
                if (rpc != nullptr && commParam != nullptr) {
                    rpc->PrintAllHcclMsgArea(commParam->rankSize);
                }
            }
            return HCCL_E_TIMEOUT;
        }
    }
    return HCCL_SUCCESS;
}

bool AicpuHcclProcess::SetAlgTypeLevel1(HcclAlgoType algoConfig, AlgTypeLevel1 &algType, uint32_t moduleNum)
{
    switch (algoConfig) {
        case HcclAlgoType::HCCL_ALGO_TYPE_HDR:
            algType = AlgTypeLevel1::ALG_LEVEL1_HD;
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_RING:
            algType = AlgTypeLevel1::ALG_LEVEL1_RING;
            HCCL_INFO("server num[%u]: level1:ring algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
            HCCL_INFO("server num[%u]: level1:nhr algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1:
            algType = AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
            HCCL_INFO("server num[%u]: level1:nhr_v1 algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_NB:
            algType = AlgTypeLevel1::ALG_LEVEL1_NB;
            HCCL_INFO("server num[%u]: level1:nb algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE:
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            HCCL_INFO("server num[%u]: level1:pipeline algo is set.", moduleNum);
            break;
        case HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH:
        case HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE:
            HCCL_WARNING("level1:fullmesh algo is not suported. the config is ignored.");
        default:
            HCCL_WARNING("algo is not suported. the config is ignored.");
            return false;
    }
    return true;
}

void AicpuHcclProcess::SetAlgoLevel1(hccl::HcclCommAicpu *commAicpu, HcclAlgoType algoConfig,
    uint32_t moduleNum, AlgTypeLevel1 &algType, bool isDefault)
{
    if ((isDefault == false) && (SetAlgTypeLevel1(algoConfig, algType, moduleNum))) {
        // 不使用default配置
        HCCL_INFO("[AicpuHcclProcess][SetAlgoLevel1] algType[%u], moduleNum[%u]", algType, moduleNum);
        return;
    }
    if (moduleNum >=  HCCL_INTER_SERVER_RING_ALGO_MAX_SUPPORT_SERVER_NUM) {
        // server 数为 8 以上：使用 HD 算法
        algType = AlgTypeLevel1::ALG_LEVEL1_HD;
    } else {
        // server 数为 2 的非整数次幂：使用 RING 算法
        // server 数为 2 的整数次幂：使用 HD 算法
        algType = (((moduleNum & (moduleNum - 1)) != 0) || (moduleNum == 1)) ?
            AlgTypeLevel1::ALG_LEVEL1_RING :
            AlgTypeLevel1::ALG_LEVEL1_HD;
    }
    DevType devType = commAicpu->GetDevType();
    if (algType == AlgTypeLevel1::ALG_LEVEL1_HD && devType == DevType::DEV_TYPE_910_93) {
        algType = AlgTypeLevel1::ALG_LEVEL1_NHR;
    }
    HCCL_INFO("[AicpuHcclProcess][SetAlgoLevel1] algType[%u], moduleNum[%u]", algType, moduleNum);
}

bool AicpuHcclProcess::SplitHcclAlgoGetLevel1Res(std::string &algoConfig, std::string &algos)
{
    std::string remainAlgoConfig;
    std::size_t found = algoConfig.find(";");
    if ((found == 0) || (found == (algoConfig.length() - 1)) || (found == std::string::npos)) {
        HCCL_INFO("algoConfig %s thereis no level1 algo config", algoConfig.c_str());
        return true;
    }
    remainAlgoConfig = algoConfig.substr(found + 1);
    found = remainAlgoConfig.find(";");
    std::size_t msgPos = 0;
    if (found != std::string::npos) {
        msgPos = found;
        HCCL_WARNING("[AicpuHcclProcess] algo level is more than 1, not supported !");
    } else {
        msgPos = remainAlgoConfig.size();
    }
    algos = (remainAlgoConfig.substr(0, msgPos));
    return false;
}

HcclResult AicpuHcclProcess::ParserHcclAlgoLevel1(std::string &algoLevel, uint32_t &level, HcclAlgoType &algoType)
{
    std::size_t found = algoLevel.find(":");
    if ((found == 0) || (found == (algoLevel.length() - 1))) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid.");
        return HCCL_E_PARA;
    }

    std::string orginalLevel = algoLevel.substr(0, found);
    std::string orginalAlgo = algoLevel.substr(found + 1);

    const std::map<std::string, HcclAlgoType> hcclAlgoTypeMap = {
        {"null", HcclAlgoType::HCCL_ALGO_TYPE_NULL},
        {"ring", HcclAlgoType::HCCL_ALGO_TYPE_RING},
        {"pipeline", HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE},
        {"fullmesh", HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH},
        {"H-D_R", HcclAlgoType::HCCL_ALGO_TYPE_HDR},
        {"pairwise", HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE},
        {"NHR", HcclAlgoType::HCCL_ALGO_TYPE_NHR},
        {"NHR_V1", HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1},
        {"NB", HcclAlgoType::HCCL_ALGO_TYPE_NB},
        {"NA", HcclAlgoType::HCCL_ALGO_TYPE_NA},
    };

    auto iterAlgoType = hcclAlgoTypeMap.find(orginalAlgo);
    if (iterAlgoType == hcclAlgoTypeMap.end()) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid, algo %s is not supported.", orginalAlgo.c_str());
        return HCCL_E_PARA;
    }
    level = HCCL_ALGO_LEVEL_1;
    algoType = iterAlgoType->second;

    return HCCL_SUCCESS;
}

bool AicpuHcclProcess::SelectAlgName(const std::string &algConfig, u32 topoType, std::string &algName)
{
    std::string curConfig;
    std::size_t found = algConfig.find(";");
    if (found == 0) {
        return false;
    } else if (found == std::string::npos) {
        curConfig = algConfig;
    } else {
        curConfig = algConfig.substr(0, found);
    }
    if (static_cast<TopoType>(topoType) == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
        if (curConfig == "AllGather=level0:doublering" || curConfig == "ReduceScatter=level0:doublering" ||
            curConfig == "AllReduce=level0:doublering") {
            std::size_t pos = curConfig.find(":");
            std::string algConfigTmp = curConfig.substr(0, pos + 1) + "ring";
            algName = g_algName.at(algConfigTmp);
            return true;
        }
    }
    auto res = g_algName.find(curConfig);
    if (res != g_algName.end()) {
        algName = res->second;
        return true;
    }
    HCCL_ERROR("[AicpuHcclProcess][SelectAlgName] algo_name is not exist, algConfig %s is no.", algConfig.c_str());
    return false;
}

void AicpuHcclProcess::SelectAlgType(hccl::HcclCommAicpu *commAicpu, const std::string &algConfig,
    uint32_t moduleNum, AlgType &algType)
{
    // 当前默认只会穿入0 1两层算法配置，多余层数穿入不做解析.
    // 0层算法 当前先写死
    // 1层算法 按默认值取
    AlgTypeLevel0 algType0 =  AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    // 构造 1层 algoType, 未填写则取默认值
    HcclAlgoType level1AlgoConfig;
    std::string algos;
    uint32_t level = 0;
    AlgTypeLevel1 algType1 = AlgTypeLevel1::ALG_LEVEL1_RESERVED;

    std::size_t found = algConfig.find("=");
    std::string curAlgConfig = algConfig.substr(found + 1);
    bool useDefault = SplitHcclAlgoGetLevel1Res(curAlgConfig, algos);
    if (useDefault == false) {
        ParserHcclAlgoLevel1(algos, level, level1AlgoConfig);
    }
    SetAlgoLevel1(commAicpu, level1AlgoConfig, moduleNum, algType1, useDefault);
    algType.algoLevel0 = algType0;
    algType.algoLevel1 = algType1;
}

HcclResult AicpuHcclProcess::RunRpcServerApiV2(void *tilingData, uint32_t groupNum)
{
    // 待适配 startthread DFX
    uint32_t commNum = MC2TilingGetHcommCnt(tilingData);
    for (uint32_t i = 0; i < commNum; i++) {
        Mc2HcommCfg *cfg = MC2TilingGetHcommCfg(tilingData, i);
        int32_t groupIdx = GetComGroupIdx(std::string(cfg->groupName));
        if (groupIdx < 0) {
            HCCL_ERROR("RunRpcServerApiV2 idx %d cannot get group by hcomId %s", i, cfg->groupName);
            return HCCL_E_INTERNAL;
        }
        hccl::HcclCommAicpu *comm = AicpuHcclProcess::GetCommAicpuCommInst(groupIdx);
        if (comm == nullptr) {
            HCCL_ERROR("RunRpcServerApiV2 cannot get CommAicpu by groupIdx %d", groupIdx);
            return HCCL_E_INTERNAL;
        }
        HcclOpResParam *commParam = AicpuHcclProcess::GetCommAicpuResInst(groupIdx);
        std::string curAlgName;
        if (!SelectAlgName(cfg->algConfig, commParam->topoInfo.topoType, curAlgName)) {
            return HCCL_E_INTERNAL;
        }
        std::string curTag = std::string(cfg->groupName) + std::to_string(cfg->opType);
        uint32_t moduleNum = commParam->topoInfo.moduleNum;
        AlgType algType;
        SelectAlgType(comm, cfg->algConfig, moduleNum, algType);
        comm->SetCommInfoCtx(static_cast<u8>(cfg->opType), curAlgName, curTag, algType);
    }
    CHK_RET(AicpuHcclProcess::RunRpcServerInnerProcessV2(groupNum));
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::PrepareHcommInstance(u32 idx, HcclOpResParam *commParam, const Mc2InitTilingInner *tiling)
{
    const std::string &group = commParam->hcomId;
    hccl::HcclCommAicpu *hcclCommAicpu = AicpuHcclProcess::AicpuGetCommbyGroup(group);
    CHK_PRT_RET(hcclCommAicpu == nullptr,
        HCCL_ERROR("RunAicpuRpcSrvLaunchV2 get Hcclcomm error group [%s]", group.c_str()), HCCL_E_INTERNAL);

    DevType devType = hcclCommAicpu->GetDevType();
    CHK_PRT_RET(devType != DevType::DEV_TYPE_910_93,
        HCCL_ERROR("Platform %u not support, please use 910_93 platform.", static_cast<u32>(devType)),
        HCCL_E_INTERNAL);

    const dfx::DfxExtendInfo *dfxInfo = hcclCommAicpu->GetDfxExtendInfo();
    CHK_PRT_RET(dfxInfo->cqeStatus != dfx::CqeStatus::kDefault ||
        dfxInfo->pollStatus == dfx::PollStatus::kStopAsException,
        HCCL_ERROR("Exist errors before, cqeStatus:%d, pollStatus:%d, group[%s]",
        dfxInfo->cqeStatus, dfxInfo->pollStatus, group.c_str()), HCCL_E_INTERNAL);

    AicpuRpcServerV2 *rpcServer = AicpuHcclProcess::GetCommRpcServer(idx);
    CHK_PRT_RET(rpcServer == nullptr,
        HCCL_ERROR("RunAicpuRpcSrvLaunchV2 get rpc inst error idx %d group [%s]", idx, group.c_str()),
        HCCL_E_INTERNAL);

    HcclResult ret = rpcServer->Init(commParam->mc2WorkSpace, tiling);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Failed to init for group [%s]", group.c_str()), HCCL_E_INTERNAL);

    hcclCommAicpu->SetAicpuRpcServer(rpcServer);
    hcclCommAicpu->SetIsDeviceMode(true);
    ret = AicpuHcclProcess::InsertCommInst(idx, hcclCommAicpu, commParam);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("Failed to insert comm inst."), HCCL_E_INTERNAL);

    AicpuHcclProcess::InsertComIdMap(idx, group);
    HCCL_INFO("Insert group %s at index %u.", group.c_str(), idx);
    return HCCL_SUCCESS;
}

HcclResult AicpuHcclProcess::AicpuRunRpcServerForMC2V2(KFCTaskV2 *task, const Mc2InitTilingInner *tilingData)
{
    static std::atomic<s32> activeThreads;
    static std::atomic<bool> initialized;

    s32 prevCount = activeThreads.fetch_add(1, std::memory_order_relaxed);
    if (prevCount == 0) {
        for (u64 i = 0; i < task->ctxNum; i++) {
            HcclOpResParam *ctx = reinterpret_cast<HcclOpResParam *>(task->context[i]);
            MC2AicpuUtils::PrintMC2HcclOpResParam(ctx);
            CHK_PRT_RET(AicpuHcclProcess::PrepareHcommInstance(i, ctx, tilingData) != HCCL_SUCCESS,
                        AicpuHcclProcess::AicpuReleaseCommbyGroup(ctx->hcomId), HCCL_E_INTERNAL);
        }
        initialized = true;
    } else {
        const u64 ts = GetCurCpuTimestamp();
        while (!initialized.load(std::memory_order_acquire)) {
            if (UNLIKELY(GetCurCpuTimestamp() - ts > static_cast<u64>(NSEC_PER_SEC))) {
                HCCL_ERROR("[%s]Timeout during instance preparation.", __func__);
                return HCCL_E_INTERNAL;
            }
        }
    }

    HcclResult ret = AicpuHcclProcess::RunRpcServerInnerProcessV2(static_cast<u32>(task->ctxNum));

    prevCount = activeThreads.fetch_sub(1, std::memory_order_acq_rel);
    if (prevCount == 1) {
        for (u64 i = 0; i < task->ctxNum; i++) {
            HcclOpResParam *ctx = reinterpret_cast<HcclOpResParam *>(task->context[i]);
            AicpuHcclProcess::AicpuReleaseCommbyGroup(ctx->hcomId);
        }
        initialized = false;
        activeThreads.store(0, std::memory_order_release);
    } else {
        const u64 ts = GetCurCpuTimestamp();
        while (activeThreads.load(std::memory_order_acquire) != 0) {
            if (UNLIKELY(GetCurCpuTimestamp() - ts > static_cast<u64>(NSEC_PER_SEC))) {
                HCCL_ERROR("[%s]Timeout during instance finalize.", __func__);
                return HCCL_E_INTERNAL;
            }
        }
    }
    return ret;
}

HcclResult AicpuHcclProcess::AicpuRunRpcServerForMC2(KFCTaskV2 *task)
{
    HcclOpResParam *commParam[MAX_COMM_CTX_NUM] = { 0 };
    for (int i = 0; i < static_cast<int>(task->ctxNum); i++) {
        commParam[i] = reinterpret_cast<HcclOpResParam *>(task->context[i]);
        CHK_RET(AicpuHcclProcess::PrepareHcommInstance(i, commParam[i]));
    }
    HcclResult ret = AicpuHcclProcess::RunRpcServerApiV2(reinterpret_cast<void *>(task->tilingData),
                                                         static_cast<uint32_t>(task->ctxNum));
    for (int i = 0; i < static_cast<int>(task->ctxNum); i++) {
        std::string group = commParam[i]->hcomId;
        AicpuHcclProcess::AicpuReleaseCommbyGroup(group);
    }
    return ret;
}

HcclResult AicpuHcclProcess::AicpuRunRpcServerV2(
    hccl::HcclCommAicpu *hcclCommAicpu, OpTilingData *tilingData, HcclOpResParam *commParam)
{
    std::string algName = tilingData->algName;
    std::string tag = reinterpret_cast<char *>(tilingData->tag);
    std::string newTag = reinterpret_cast<char *>(tilingData->newTag);

    HCCL_DEBUG("[AicpuHcclProcess][AicpuRunRpcServerV2]Entry AicpuRunRpcServerV2, group[%s], tag[%s], newTag[%s]",
        hcclCommAicpu->GetGroupName().c_str(), tag.c_str(), newTag.c_str());

    HCCL_DEBUG("[AicpuHcclProcess][AicpuRunRpcServerV2]Entry AicpuRunRpcServerV2, algName[%s], algtype[%llu],"\
        "floatOverflowMode[%u], dumpDebug[%u], debugMode[%u], inputPtr[%p], outputPtr[%p].", algName.c_str(),
        tilingData->algType, tilingData->floatOverflowMode, tilingData->dumpDebug,
        tilingData->debugMode, tilingData->inputPtr, tilingData->outputPtr);

    HCCL_DEBUG("[AicpuHcclProcess][AicpuRunRpcServerV2]Entry AicpuRunRpcServerV2, reduceType[%u], syncMode[%u],"\
        "root[%u], dstRank[%u], srcRank[%u], opType[%u], index[%u], length[%llu].", tilingData->reduceType,
        tilingData->syncMode, tilingData->root, tilingData->dstRank, tilingData->srcRank,
        tilingData->opType, tilingData->index, tilingData->length);
    hccl::OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = reinterpret_cast<void *>(tilingData->inputPtr);
    opParam.outputPtr = reinterpret_cast<void *>(tilingData->outputPtr);
    opParam.reduceType = static_cast<HcclReduceOp>(tilingData->reduceType);
    opParam.stream = hcclCommAicpu->GetMainStrem();
    opParam.syncMode = static_cast<SyncMode>(tilingData->syncMode);

    hcclCommAicpu->UpdateNotifyWaitTimeOut(opParam.syncMode, commParam->config.notifyWaitTime);

    opParam.opBaseAtraceInfo = nullptr;
    opParam.root = tilingData->root;
    opParam.dstRank = tilingData->dstRank;
    opParam.srcRank = tilingData->srcRank;
    opParam.opType = static_cast<HcclCMDType>(tilingData->opType);
    opParam.isZeroCopy = tilingData->isZeroCopy;
    opParam.index = tilingData->index;
    hcclCommAicpu->PrepareOpRetryHandler(tilingData->inplaceSupportRetry,
        tilingData->retryEnable, tilingData->inPlaceSupportRetryStatus,
        tilingData->isInplacePreSync, tilingData->isPostSync);
    u8* dynamicDataPtr = reinterpret_cast<u8*>(tilingData) + sizeof(struct OpTilingData);
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (opParam.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        struct OpTilingBatchSendRecvDataDes* batchSendRecvDataPtr =
            reinterpret_cast<struct OpTilingBatchSendRecvDataDes*>(dynamicDataPtr);
        opParam.BatchSendRecvDataDes.itemNum = batchSendRecvDataPtr->itemNum;
        opParam.BatchSendRecvDataDes.sendRecvItemsPtr = batchSendRecvDataPtr->batchSendRecvItem;
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag:%s", opParam.tag.c_str());
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", opParam.tag.c_str()));
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        struct OpTilingAllToAllDataDes* a2ADataPtr =
            reinterpret_cast<struct OpTilingAllToAllDataDes*>(dynamicDataPtr);
        opParam.All2AllDataDes.sendType =  static_cast<HcclDataType>(a2ADataPtr->sendType);
        opParam.All2AllDataDes.recvType =  static_cast<HcclDataType>(a2ADataPtr->recvType);
        opParam.All2AllDataDes.sendCount = a2ADataPtr->sendCount;
        HCCL_DEBUG("[AicpuHcclProcess][AicpuRunRpcServerV2] alltoall aicpu, sendCounts[%llu] rankSize[%u]",
            opParam.All2AllDataDes.sendCount, commParam->rankSize);
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag:%s,ct:%llu,dt:%u",
            opParam.tag.c_str(), opParam.All2AllDataDes.sendCount, opParam.All2AllDataDes.sendType);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", opParam.tag.c_str()));
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        struct OpTilingAlltoallvDataDes* alltoallvDataPtr =
            reinterpret_cast<struct OpTilingAlltoallvDataDes*>(dynamicDataPtr);
        opParam.All2AllDataDes.sendType =  static_cast<HcclDataType>(alltoallvDataPtr->sendType);
        opParam.All2AllDataDes.recvType =  static_cast<HcclDataType>(alltoallvDataPtr->recvType);
        u64 rankSize =  commParam->rankSize;
        opParam.All2AllDataDes.sendCounts =  static_cast<void *>(alltoallvDataPtr->sendRecvInfos);
        opParam.All2AllDataDes.recvCounts =  static_cast<void *>(static_cast<u64 *>(alltoallvDataPtr->sendRecvInfos) + rankSize);
        opParam.All2AllDataDes.sdispls =  static_cast<void *>(static_cast<u64 *>(alltoallvDataPtr->sendRecvInfos) + ALLTOALLV_INFO_INDEX_2 * rankSize);
        opParam.All2AllDataDes.rdispls =  static_cast<void *>(static_cast<u64 *>(alltoallvDataPtr->sendRecvInfos) + ALLTOALLV_INFO_INDEX_3 * rankSize);
        if (algName == "RunAlltoAllVTwoLevelPipeline") {
            hcclCommAicpu->SetSendRecvInfoPtr(
                static_cast<void *>(static_cast<u64 *>(alltoallvDataPtr->sendRecvInfos) + ALLTOALLV_INFO_INDEX_4 * rankSize));
        }
        HCCL_DEBUG("[AicpuHcclProcess][AicpuRunRpcServerV2] sendCountsPtr[%p], recvCountsPtr[%p], sdisplsPtr[%p], rdisplsPtr[%p].",
            opParam.All2AllDataDes.sendCounts, opParam.All2AllDataDes.recvCounts, opParam.All2AllDataDes.sdispls, opParam.All2AllDataDes.rdispls);
        for(u32 i= 0; i < rankSize; i++) {
            HCCL_DEBUG("[AicpuHcclProcess][AicpuRunRpcServerV2] sendCounts[%llu], recvCounts[%llu].",
                *(static_cast<const u64 *>(opParam.All2AllDataDes.sendCounts) + i), *(static_cast<const u64 *>(opParam.All2AllDataDes.recvCounts) + i));
        }
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag:%s,ct:%llu,dt:%u",
            opParam.tag.c_str(), opParam.All2AllDataDes.sendCounts, opParam.All2AllDataDes.sendType);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", opParam.tag.c_str()));
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        struct OpTilingAlltoallvcDataDes* alltoallvcDataPtr =
            reinterpret_cast<struct OpTilingAlltoallvcDataDes*>(dynamicDataPtr);
        opParam.All2AllDataDes.sendType =  static_cast<HcclDataType>(alltoallvcDataPtr->sendType);
        opParam.All2AllDataDes.recvType =  static_cast<HcclDataType>(alltoallvcDataPtr->recvType);
        opParam.All2AllDataDes.sendCountMatrix = static_cast<void *>(alltoallvcDataPtr->sendCountMatrix);
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag:%s,ct:%llu,dt:%u",
            opParam.tag.c_str(), opParam.All2AllDataDes.sendCountMatrix, opParam.All2AllDataDes.sendType);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", opParam.tag.c_str()));
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V ||
        opParam.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        const u64 vStructSize = sizeof(struct OpTilingVDataDes);
        CHK_PRT_RET(tilingData->length < vStructSize, HCCL_ERROR("[AicpuHcclProcess][AicpuRunRpcServerV2] TilingData "
            "corrupt, length[%llu], expect[%llu]", tilingData->length, vStructSize), HCCL_E_PARA);
        const u32 rankSize = commParam->rankSize;
        struct OpTilingVDataDes* vDataPtr = reinterpret_cast<struct OpTilingVDataDes*>(dynamicDataPtr);
        const u64 vDataLen = CalcOpTilingVDataDesVDataLen(rankSize);
        CHK_PRT_RET(vDataPtr->vDataLen != vDataLen, HCCL_ERROR("[AicpuHcclProcess][AicpuRunRpcServerV2] TilingVDataDes "
            "corrupt, length[%llu], expect[%llu], rankSize[%u]", vDataPtr->vDataLen, vDataLen, rankSize), HCCL_E_PARA);
        opParam.VDataDes.dataType = static_cast<HcclDataType>(vDataPtr->dataType);
        opParam.VDataDes.counts = static_cast<void *>(static_cast<u64 *>(vDataPtr->vData));
        opParam.VDataDes.displs = static_cast<void *>(static_cast<u64 *>(vDataPtr->vData) + rankSize);
        CHK_RET(CalcDataSizeV(opParam, rankSize));
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag:%s,ct:%llu,ds:%llu,dt:%u",
            opParam.tag.c_str(), opParam.VDataDes.counts, opParam.VDataDes.displs, opParam.VDataDes.dataType);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", opParam.tag.c_str()));
    } else {
        struct OpTilingDataDes* opDataDesPtr = reinterpret_cast<struct OpTilingDataDes*>(dynamicDataPtr);
        opParam.DataDes.count = opDataDesPtr->count;
        opParam.DataDes.dataType = static_cast<HcclDataType>(opDataDesPtr->dataType);
        CHK_RET(CalcDataSize(opParam.opType, static_cast<HcclDataType>(opDataDesPtr->dataType), opDataDesPtr->count,
            hcclCommAicpu->GetRankSize(), opParam.inputSize, opParam.outputSize));
        HCCL_DEBUG("[AicpuHcclProcess][AicpuRunRpcServerV2] Entry AicpuRunRpcServerV2, "
            "count[%llu], dataType[%u] inputSize[%lu] outputSize[%lu].",
            opDataDesPtr->count, opDataDesPtr->dataType, opParam.inputSize, opParam.outputSize);
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U, "tag:%s,ct:%llu,dt:%u",
            opParam.tag.c_str(), opParam.DataDes.count, opParam.DataDes.dataType);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", opParam.tag.c_str()));
    }

    CHK_RET(hrtSetLocalDeviceSatMode(static_cast<rtFloatOverflowMode_t>(tilingData->floatOverflowMode)));
    hcclCommAicpu->SetDumpDebug(tilingData->dumpDebug);
    hcclCommAicpu->SetAlgType(tilingData->algType);
    hcclCommAicpu->SetDebugMode(tilingData->debugMode);
    hcclCommAicpu->SetIsDeviceMode(false);
    hcclCommAicpu->SetUserStreamId(tilingData->userStreamId);
    /* 接口交互信息日志 */
    std::string logInfo = std::string(stackLogBuffer);
    CHK_RET_AND_PRINT_IDE(hcclCommAicpu->SaveTraceInfo(logInfo), opParam.tag.c_str());

    HcclUs startut = TIME_NOW();
    CHK_RET(hcclCommAicpu->ExecOp(newTag, algName, opParam, commParam));
    HcclUs endut = TIME_NOW();
    /* 关键状态记录 */
    std::string endInfo = "AicpuRunRpcServerV2:success,take time: " +
        std::to_string(DURATION_US(endut - startut).count()) + " us";
    CHK_RET_AND_PRINT_IDE(hcclCommAicpu->SaveTraceInfo(endInfo), opParam.tag.c_str());

    HCCL_INFO("[AicpuHcclProcess][AicpuRunRpcServerV2]AicpuRunRpcServerV2 process end-------");
    return HCCL_SUCCESS;
}

void AicpuHcclProcess::IncProfCnt(const AicpuComContext &ctx)
{
    g_proxLoopCnt++;
    // 数组中最多记录AC_MAX_PROF_LOOP个算子的prof信息, 超过之后从头开始复写
    g_proxLoopCnt %= AC_MAX_PROF_LOOP;
    HCCL_INFO("g_proxLoopCnt set to %u", g_proxLoopCnt);
}


uint32_t AicpuHcclProcess::GetStreamRankIdx(int32_t actualStreamId)
{
    auto it = g_streamIdMap.find(actualStreamId);
    return it == g_streamIdMap.cend() ? UINT32_MAX : it->second;
}

constexpr s32 PREFER_CLUSTER_ID = 0;
constexpr u32 DELAY_TIME_IN_US = 15U;
std::mutex g_mtxForCpuCheck;
std::array<s64, AICPU_CNT> g_curTaskIdsOnCpu = {-1, -1, -1, -1, -1, -1, -1, -1};
AicpuServerRole AicpuHcclProcess::GetVerifiedServerRole(const AicpuComContext &ctx)
{
    if (!ctx.multiServerFlag) {
        HCCL_INFO("Skip server start check for non-multi server scene.");
        return AicpuServerRole::MASTER;
    }
    if (MC2AicpuUtils::GetCurClusterId() != PREFER_CLUSTER_ID) {
        usleep(DELAY_TIME_IN_US);
    }
    std::lock_guard<std::mutex> lock(g_mtxForCpuCheck);
    uint64_t taskId = UINT64_MAX;
    uint32_t streamId;
    if (aicpu::GetTaskAndStreamId == nullptr ||
        aicpu::GetTaskAndStreamId(taskId, streamId) != aicpu::status_t::AICPU_ERROR_NONE) {
        HCCL_ERROR("Failed to get task ID.");
        return AicpuServerRole::INVALID;
    }
    int32_t cpuId = MC2AicpuUtils::GetCpuId();
    const auto it = std::find(g_curTaskIdsOnCpu.begin(), g_curTaskIdsOnCpu.end(), (s64)taskId);
    if (it == g_curTaskIdsOnCpu.end()) {
        InitMultiThreadSharedCtx(cpuId, taskId);
        HCCL_INFO(
            "Task ID %u not start master thread, start it on cpu %d, clusterID %d",
            taskId, cpuId, MC2AicpuUtils::GetCurClusterId());
        return AicpuServerRole::MASTER;
    } else {
        if (g_sharedCtx.startedThreadNum >= MAX_BATCH_WRITE_THREAD_NUM) {
            HCCL_INFO("already start %ld thread, exceed the system limit %ld",
                g_sharedCtx.startedThreadNum,
                MAX_BATCH_WRITE_THREAD_NUM);
            return AicpuServerRole::INVALID;
        }
        g_sharedCtx.sendWqeNum[g_sharedCtx.startedThreadNum] = 0;
        g_sharedCtx.curThreadIdsOnCpu[cpuId] = g_sharedCtx.startedThreadNum++;
        HCCL_INFO("Task ID %u start slave thread, threadId %ld on cpu %d. clusterID %d",
            taskId,
            g_sharedCtx.curThreadIdsOnCpu[cpuId],
            cpuId,
            MC2AicpuUtils::GetCurClusterId());
        return AicpuServerRole::SLAVE;
    }
}

void AicpuHcclProcess::InitMultiThreadSharedCtx(int32_t cpuId, uint64_t taskId)
{
    g_sharedCtx.startedThreadNum = 1;
    g_hcclMsgQueue.Clear();
    g_sharedCtx.taskFinishFlag = false;
    g_sharedCtx.curThreadIdsOnCpu[cpuId] = 0;
    g_sharedCtx.sendWqeNum[0] = 0;
    for (s32 i = 0; i < AICPU_CNT; ++i) {
        g_curTaskIdsOnCpu[i] = (i == cpuId ? taskId : -1);
        g_sharedCtx.curThreadIdsOnCpu[i] = 0;
    }
}