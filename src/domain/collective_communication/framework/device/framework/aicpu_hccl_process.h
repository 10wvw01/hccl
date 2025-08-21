/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_HCCL_PROCESS_H__
#define __AICPU_HCCL_PROCESS_H__

#include <memory>
#include <vector>
#include <string>
#include "common.h"
#include "aicpu_communicator.h"
#include "common/aicpu_hccl_def.h"
#include "framework/aicpu_rpc_server.h"
#include "framework/aicpu_rpc_serverv2.h"
#include "hdc_pub.h"
#include "stream_pub.h"

class AicpuHcclProcess {
public:
    ~AicpuHcclProcess() = default;
    u32 AicpuRpcResInit(HccCommResParamTask *commParam);
    static AicpuComContext *AicpuGetInnerComContext(std::string &key);
    static u32 AicpuRpcResInitV2(HcclOpResParam *commParam, bool isCustom);
    static std::mutex& AicpuGetCommMutex();
    static hccl::HcclCommAicpu *AicpuGetCommbyGroup(const std::string &group);
    static HcclResult AicpuGetCommAll(std::vector<std::pair<std::string, hccl::HcclCommAicpu *>> &aicpuCommInfo);
    static void AicpuDestoryCommbyGroup(const std::string &group);
    static void AicpuReleaseCommbyGroup(const std::string &group);
    static uint32_t GetStreamRankIdx(int32_t actualStreamId);

    /**
     * 数组存满后，触发一下打印，打印数组中所有的元素信息
     */
    static void OutputProfLog(const AicpuComContext &ctx);
    /**
     * 管理prof数组index的计数
     */
    static void IncProfCnt(const AicpuComContext &ctx);

    static HcclResult AicpuCcOpExe(AivAicpuOpParam *args, AivAicpuOpParam *commParamNext, AicpuComContext *ctx);
    static HcclResult AllEndCcOpExe(AicpuComContext *ctx);
    static HcclResult AddTaskForHcclMsg(AicpuComContext *ctx, AicpuRpcServer &rpc, CommonHcclMsg *hcclMsg,
        AivAicpuOpParam *msg);
    static HcclResult AicpuRunRpcServer(AicpuComContext *ctx, KFCTask *taskInfo);
    static HcclResult AicpuRunRpcServerForApi(AicpuComContext *ctx);
    static HcclResult RunSlaveRpcServerForApi(AicpuComContext *ctx);
    virtual HcclResult InitTimeOutConfig(HccCommResParamTask *commParam, AicpuComContext *ctx);
    static HcclResult AicpuRunRpcServerV2(
        hccl::HcclCommAicpu *hcclCommAicpu, OpTilingData *tilingData, HcclOpResParam *commParam);
    static DevType AicpuGetInnerDevType();
    static HcclResult AicpuRunRpcServerForMC2V2(KFCTaskV2 *task, const Mc2InitTilingInner *tilingData);
    static HcclResult AicpuRunRpcServerForMC2(KFCTaskV2 *task);
    static void CallMC2MaintenanceThread(AicpuComContext *ctx);
    static HcclResult CheckNsStopLaunchStatus(uint32_t groupNum);
    static AicpuServerRole GetVerifiedServerRole(const AicpuComContext &ctx);

private:
    static AicpuCCExecOp GetCcOpType(u64 comDataLen, u64 rankNum);
    static HcclResult InitSignalInfo(HccCommResParamTask *commParam, AicpuComContext *ctx);
    static HcclResult InitStreamInfo(HccCommResParamTask *commParam, AicpuComContext *ctx);
    static HcclResult InitEventId(HccCommResParamTask *commParam, AicpuComContext *ctx);
    static HcclResult InitAicpuOpNotify(HccCommResParamTask *commParam, AicpuComContext *ctx);
    static void ClearMsgInValidCount(uint32_t idx);
    static void AddMsgInValidCount(uint32_t idx);
    static uint32_t GetMsgInValidCount(uint32_t idx);
    static bool HcclOpSupportRetry(AicpuComContext *ctx, AivAicpuOpParam &opParams);
    static HcclResult CalcDataSize(HcclCMDType op, HcclDataType type, u64 count, u32 rankSize,
        u64 &inputSize, u64 &outputSize);
    static HcclResult CalcDataSizeV(hccl::OpParam &param, u32 rankSize);
    static u64 CalcOpTilingVDataDesVDataLen(u32 rankSize);
    static HcclResult ResetSqBuff(AicpuComContext *ctx);
    static HcclResult LaunchHcclOp(AicpuComContext *ctx, AivAicpuOpParam *commParam, uint32_t &beginSqePos, uint32_t &endSqePos);
    static HcclResult RetryLaunchHcclOp(AicpuComContext *ctx, AivAicpuOpParam *commParam, uint32_t &endSqePos);
    static HcclResult HcclOpExecFsmInitProcess(AicpuComContext *ctx, HcclOpExecFSM &state, KfcError &errorCode,
        AicpuRpcServer &rpc, AivAicpuOpParam &opParams);
    static HcclResult HcclOpExecFsmLaunchProcess(AicpuComContext *ctx, HcclOpExecFSM &state,
        KfcError &errorCode, AivAicpuOpParam &opParams, uint32_t &beginSqePos, uint32_t &endSqePos);
    static HcclResult HcclOpExecFsmWaitEndProcess(AicpuComContext *ctx, HcclOpExecFSM &state,
        KfcError &errorCode, uint32_t retryCnt);
    static HcclResult HcclOpExecFsmStoppingProcess(AicpuComContext *ctx, HcclOpExecFSM &state,
        KfcError &errorCode, uint32_t retryCnt);
    static HcclResult HcclOpExecFsmStoppedProcess(AicpuComContext *ctx, HcclOpExecFSM &state,
        KfcError &errorCode, uint32_t retryCnt, AivAicpuOpParam &opParams, uint32_t beginSqePos, uint32_t endSqePos);
    static HcclResult HcclOpExecFsmWaitRetryProcess(AicpuComContext *ctx, HcclOpExecFSM &state,
        KfcError &errorCode, uint32_t retryCnt);
    static HcclResult HcclOpExecFsmRetryProcess(AicpuComContext *ctx, HcclOpExecFSM &state, KfcError &errorCode,
        uint32_t &retryCnt, AivAicpuOpParam &opParams, uint32_t &endSqePos);
    static HcclResult HcclOpExecFsmEndProcess(AicpuComContext *ctx, uint32_t retryCnt, AivAicpuOpParam &opParams);
    static HcclResult UpdateOpExecStatus(AicpuComContext *ctx, HcclOpExecFSM &fsmState, KfcStatus state,
        KfcError &errorCode, uint32_t retryCnt);
    static bool HcclOpCheckSupportRetry(AicpuComType opType);
    static bool HcclOpCheckInplace(const AivAicpuOpParam &opParams);
    static void HcclUpdateOpIndex(AicpuComType opType, AicpuComContext *ctx);
    static u32 HcclGetWaitStopExecCmdTimeout();
    static u32 HcclGetWaitRetryCmdTimeout(AicpuComContext *ctx, uint32_t retryCnt);
    static HcclResult InsertCommInst(uint32_t idx, hccl::HcclCommAicpu *comm, HcclOpResParam *resParam);
    static hccl::HcclCommAicpu *GetCommAicpuCommInst(uint32_t idx);
    static HcclOpResParam *GetCommAicpuResInst(uint32_t idx);
    static AicpuRpcServerV2 *GetCommRpcServer(uint32_t idx);
    static void InsertComIdMap(uint32_t groupIdx, const std::string &hcomId);
    static int32_t GetComGroupIdx(const std::string &hcomId);
    static void SetMsgStartTime(int groupIdx);
    static bool CheckMsgEnableFlag(int groupIdx);
    static void SetMsgEnableFlag(int groupIdx, bool flag);
    static void SetKernelStartTime(void);
    static bool CheckMsgTimeOut(void);
    static bool CheckKernelTimeOut(void);
    static void CopyCtxForBackGroundDfx(const AicpuComContext *ctx);
    static HcclResult DealReturnValue(const AicpuComContext *ctx, const HcclResult ret);
    static bool CheckNsCommand(hccl::HcclCommAicpu *comm);
    static HcclResult CheckRestartError(hccl::HcclCommAicpu *comm);
    static void ResetRestartParam(RestartParam &restartParam);
    static bool GetOpRetryEnable(uint32_t groupNum);
    static HcclResult RestartProcessConsulation(RestartParam &restartParam, bool &finalizeAllEnd, bool *finalizeMask, uint32_t groupNum);
    static HcclResult SetNsOpStatus(uint32_t groupNum, bool state);
    static HcclResult RunRpcServerInnerProcessV2(uint32_t groupNum);
    static HcclResult RunRpcServerLoopProcess(u32 groupIdx, bool &finalizeFlag);
    static bool SetAlgTypeLevel1(HcclAlgoType algoConfig, AlgTypeLevel1 &algType, uint32_t moduleNum);
    static bool SelectAlgName(const std::string &algConfig, u32 topoType, std::string &algName);
    static void SelectAlgType(hccl::HcclCommAicpu *commAicpu, const std::string &curAlgName,
        uint32_t moduleNum, AlgType &algType);
    static bool SplitHcclAlgoGetLevel1Res(std::string &algoConfig, std::string &algos);
    static HcclResult ParserHcclAlgoLevel1(std::string &algoLevel, uint32_t &level, HcclAlgoType &algoType);
    static void SetAlgoLevel1(hccl::HcclCommAicpu *commAicpu, HcclAlgoType algoConfig,
        uint32_t moduleNum, AlgTypeLevel1 &algType, bool isDefault);
    static HcclResult AddTaskForGroupSyncMsg(hccl::HcclCommAicpu *commAicpu, CommonHcclMsg *hcclMsg,
                                             AicpuRpcServerV2* rpcServer, uint32_t groupIdx);
    static void RepeatUpdatepOpParam(hccl::OpParam &opParam, CommonHcclMsg *hcclMsg, HcclMsgExt *hcclMsgExt, hccl::HcclCommAicpu *commAicpu);
    static void PrepareOpParam(hccl::OpParam *opParam, CommonHcclMsg *hcclMsg, AicpuRpcServerV2 &rpc, hccl::HcclCommAicpu *commAicpu);
    static HcclResult AddTaskForHcclMsgV2(hccl::HcclCommAicpu *comm, AicpuRpcServerV2 *rpc, CommonHcclMsg *hcclMsg,
                                          const HcclOpResParam *commParam);
    static HcclResult HandleBatchWriteOperation(const CommonHcclMsg &commonHcclMsg, const AicpuComContext *ctx);
    static void RecordReportStatus(uint32_t groupNum, dfx::ReportStatus status);
    static std::string GetNewTag(uint32_t groupIdx);
    static HcclResult RpcServerPreCheck(AicpuRpcServerV2 *rpc, hccl::HcclCommAicpu *comm, bool &finalizeFlag);
    static HcclResult BarrierProcess(u32 groupIdx, u32 queueId, BarrierStatus &status);
    static HcclResult RunRpcServerApi(AicpuComContext *ctx, AicpuRpcServer &rpc);
    static HcclResult RunRpcServerApiV2(void *tilingData, uint32_t groupNum);
    static HcclResult RunRpcServerOneStageWait(AicpuComContext *ctx, AicpuRpcServer &rpc);
    static HcclResult TryRunRpcServerOneStageWait(AicpuComContext *ctx, AicpuRpcServer &rpc);
    static HcclResult RunRpcServerTwoStageWait(AicpuComContext *ctx, AicpuRpcServer &rpc);
    static HcclResult AICPU_RpcServerUnfoldStageWait(AicpuComContext *ctx, AicpuRpcServer &rpc);
    static HcclResult PrepareHcommInstance(u32 idx, HcclOpResParam *commParam,
        const Mc2InitTilingInner *tiling = nullptr);
    static void GetCommonHcclMsg(HcclMsg *hcclMsg, CommonHcclMsg *commonHcclMsg, u64 tilingBase = 0UL);
    static HcclResult ParseCcOpTilingData(CommonHcclMsg *commonHcclMsg, int32_t groupIdx);
    template <typename T>
    static HcclResult InitAndVerifySignal(const HcclSignalInfo &signalInfo, std::shared_ptr<T> &notify, u64 &addr);
    static HcclResult AicpuCreateCommbyGroup(const std::string &group, hccl::HcclCommAicpu **aicpuCommPtr);
    static HcclResult InitIbversData(HccCommResParamTask *commParam, AicpuComContext *ctx);
    static HcclResult ConcurrentPostSendWqe(const CommonHcclMsg &commonHcclMsg, const AicpuComContext *ctx, u8 *needSendTotalNum);
    static HcclResult WaitForSlaveCompletion(const AicpuComContext *ctx, u8 needSendTotalNum);
    static void InitMultiThreadSharedCtx(int32_t cpuId, uint64_t taskId);
    static bool CheckTimeOut(u64 startTimeStamp, u64 timeOutTime);
    static HcclResult BatchWriteProcess(CommonHcclMsg &msg, AicpuRpcServerV2 &rpc, hccl::HcclCommAicpu &comm,
                                        HcclOpResParam &param);
};

#endif // __AICPU_HCCL_PROCESS_HPP__