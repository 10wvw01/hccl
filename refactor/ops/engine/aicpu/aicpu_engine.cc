/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_engine.h"

#include "load_kernel.h"
#include "hccl_algorithm.h"
#include "binary_stream.h"
#include "utils/utils.h"
#include "log.h"

#include "hcomm_primitives_dl.h"
#include "hccl_res_dl.h"
#include "hcomm_host_profiling_dl.h"
#include "exec_timeout_manager.h"
#include "aicpu_timeout.h"
#include "dlhcomm_function.h"
#include "dpu/kernel_launch.h"

#include <hccl/hccl_res.h>
#include <acl/acl_rt.h>
#include <atomic>
#include <vector>

namespace ops_hccl {

// 对应原始 op_common.cc 中的常量
constexpr u32 HOST_WAIT_AICPU_NOTIFYIDX = 0; // host主流wait aicpu流的notify idx
constexpr u32 HOST_NOTIFY_TIMEOUT_OFFSET = 27; // host等待Device通知的超时时间偏移量
constexpr u32 KERNEL_TIMEOUT_OFFSET = 25; // kernel启动超时时间偏移量


// ───────────── 静态辅助函数 (被调用者在前) ─────────────
// NOTIFY_IDX_ACK / NOTIFY_IDX_DATA_SIGNAL 已在 alg_param.h 中定义为 constexpr

static void *GetSliceAddr(const DataSlice &slice)
{
    return static_cast<void *>(static_cast<s8 *>(slice.addr_) + slice.offset_);
}

static void TraceDataSlice(const char *funcName, const char *transType, u32 sliceIdx, u32 sliceNum,
    const DataSlice &srcSlice, const DataSlice &dstSlice, const void *src, const void *dst, u64 len,
    HcclDataType dataType, HcclReduceOp reduceOp)
{
    HCCL_DEBUG("[AiCpuEngine][%s][%s] sliceIdx[%u], sliceNum[%u], src[%p], dst[%p], len[%llu].",
        funcName, transType, sliceIdx, sliceNum, src, dst,
        static_cast<unsigned long long>(len));
}

static const ChannelInfo* LookupChannel(const std::map<u32, std::vector<ChannelInfo>> &channels, u32 rank)
{
    auto it = channels.find(rank);
    if (it == channels.end() || it->second.empty()) {
        return nullptr;
    }
    return &it->second[0];
}

static bool IsPcieProtocol(const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    for (auto it = channels.begin(); it != channels.end(); ++it) {
        if (!it->second.empty() && it->second[0].protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
            return true;
        }
    }
    return false;
}


// ═══════════════════════════════════════════════════════════════════
// CreateRes / LaunchKernel
// ═══════════════════════════════════════════════════════════════════

HcclResult AiCpuEngine::CreateRes(HcclComm comm, const OpParam &param, HcclAlgorithm &alg,
                                   AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resReq)
{
    // 1. 将 algHierarchyInfo 和序列化的 HcclAlgorithm 存入 resCtx_
    resCtx_.algHierarchyInfo = algHierarchyInfo;
    resCtx_.commInfoPtr = static_cast<void*>(comm);
    {
        BinaryStream algoBs;
        alg.SerializeTo(algoBs);
        std::vector<char> algoSerialData;
        algoBs.Dump(algoSerialData);
        resCtx_.algoSerialData = std::move(algoSerialData);
    }

    // 2. 从通信域获取 CCL buffer（对应 HcclAllocAlgResourceAICPU）
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtx_.cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};

    // 3. 填充 notify/thread 信息
    resCtx_.notifyNumOnMainThread = resReq.notifyNumOnMainThread;
    resCtx_.slaveThreadNum = resReq.slaveThreadNum;
    resCtx_.notifyNumPerThread = resReq.notifyNumPerThread;

    // 4. 创建线程（对应 HcclGetThread）
    CHK_RET(HcclGetThreadInternal(comm, param, resReq));

    // 5. 创建 host CPU TS 线程并导出，设置到 resCtx_（对应原始 HcclExecOp 中的 cpuTsThread 逻辑）
    if (param.engine == COMM_ENGINE_AICPU_TS || param.engine == COMM_ENGINE_CPU) {
        ThreadHandle cpuTsThread{0};
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, 1, &cpuTsThread));
        ThreadHandle exportedAicpuTsThread{0};
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &cpuTsThread, COMM_ENGINE_AICPU_TS, &exportedAicpuTsThread));
        // 导出给 AICPU_TS 的 cpuTsThread，设到 param 供 device 侧使用
        const_cast<OpParam &>(param).opThread = exportedAicpuTsThread;

        // 获取主流信息并导出为 exportedCpuTsThread（对应原始 GetMainThreadInfo + HcclThreadExportToCommEngine）
        ThreadHandle mainThread = resCtx_.threads.empty() ? 0 : resCtx_.threads[0];
        ThreadHandle exportedCpuTsThread{0};
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &mainThread, COMM_ENGINE_CPU_TS, &exportedCpuTsThread));
        resCtx_.cpuTsThread = cpuTsThread;
        resCtx_.exportedCpuTsThread = exportedCpuTsThread;
    }

    // 6. 按层级申请 channel（对应 HcclGetChannel + HcclGetChannelImpl）
    resCtx_.channels.resize(resReq.channels.size());
    for (u32 level = 0; level < resReq.channels.size(); ++level) {
        std::vector<HcclChannelDesc> &levelNChannelRequest = resReq.channels[level];
        // 区分 device / host 链路，分别用不同 CommEngine 建链
        std::vector<HcclChannelDesc> deviceChannelRequest;
        std::vector<HcclChannelDesc> hostChannelRequest;
        for (auto &channelRequest : levelNChannelRequest) {
            if (channelRequest.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
                deviceChannelRequest.emplace_back(channelRequest);
            } else if (channelRequest.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
                hostChannelRequest.emplace_back(channelRequest);
            }
        }
        // device 建链
        CHK_RET(HcclGetChannelImplInternal(level, comm, param, deviceChannelRequest, CommEngine::COMM_ENGINE_AICPU_TS));
        // host 建链
        CHK_RET(HcclGetChannelImplInternal(level, comm, param, hostChannelRequest, CommEngine::COMM_ENGINE_CPU));
    }

    HCCL_INFO("[AiCpuEngine][CreateRes] success, slaveThreadNum[%u], channelLevel[%zu]",
              resCtx_.slaveThreadNum, resCtx_.channels.size());
    return HCCL_SUCCESS;
}

// 对应原始 HcclGetThread：区分 AICPU_TS/CPU 和 host 模式创建线程
HcclResult AiCpuEngine::HcclGetThreadInternal(HcclComm comm, const OpParam &param, AlgResourceRequest &resReq)
{
    if (param.engine == COMM_ENGINE_AICPU_TS || param.engine == COMM_ENGINE_CPU) {
        u32 maxNotifyNum = resReq.notifyNumOnMainThread;
        for (u32 i = 0; i < resReq.notifyNumPerThread.size(); i++) {
            if (resReq.notifyNumPerThread[i] > maxNotifyNum) {
                maxNotifyNum = resReq.notifyNumPerThread[i];
            }
        }
        u32 threadNum = resReq.slaveThreadNum + 1;
        std::vector<ThreadHandle> threads(threadNum);
        // maxNotifyNum 需要再增加一个用于 host-device 同步
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, threadNum, maxNotifyNum + 1, threads.data()));
        CHK_RET(SaveMainThreadInfoInternal(comm, param, threads[0], maxNotifyNum + 1));
        // 申请展开流对应的 Thread
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CPU, 1, 0, &resCtx_.unfoldThread));
        CHK_RET(SaveUnfoldThreadInfoInternal(comm, param, resCtx_.unfoldThread));
        HCCL_INFO("[HcclGetThread] unfoldThread [%lu]", resCtx_.unfoldThread);
        for (u32 i = 0; i < threadNum; i++) {
            resCtx_.threads.push_back(threads[i]);
        }
    } else {
        // host 模式：将主流封装为 thread，并创建主流上的 notify
        ThreadHandle thread;
        CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, param.stream,
            resReq.notifyNumOnMainThread, &thread));
        resCtx_.threads.push_back(thread);
        u32 maxNotifyNum = 0;
        for (u32 i = 0; i < resReq.notifyNumPerThread.size(); i++) {
            if (resReq.notifyNumPerThread[i] > maxNotifyNum) {
                maxNotifyNum = resReq.notifyNumPerThread[i];
            }
        }
        // host 模式下从线程通过 stream 创建
        u32 threadNum = resReq.slaveThreadNum;
        if (threadNum > 0) {
            std::vector<ThreadHandle> slaveThreads(threadNum);
            CHK_RET(HcclThreadAcquire(comm, param.engine, threadNum, maxNotifyNum, slaveThreads.data()));
            for (u32 i = 0; i < threadNum; i++) {
                resCtx_.threads.push_back(slaveThreads[i]);
            }
        }
    }
    return HCCL_SUCCESS;
}

// 对应原始 SaveMainThreadInfo：保存主流信息到 host 内存
HcclResult AiCpuEngine::SaveMainThreadInfoInternal(HcclComm comm, const OpParam &param,
    ThreadHandle thread, u32 notifyNum)
{
    uint64_t size = sizeof(ThreadHandle) + sizeof(u32);
    void *ctx = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle *>(ctx);
    *threadPtr = thread;
    char* curPtr = reinterpret_cast<char *>(ctx);
    curPtr += sizeof(ThreadHandle);
    u32 *notifyNumPtr = reinterpret_cast<u32 *>(curPtr);
    *notifyNumPtr = notifyNum;
    HCCL_INFO("[SaveMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]",
        threadPtr, thread, notifyNumPtr, notifyNum);
    return HCCL_SUCCESS;
}

// 对应原始 SaveUnfoldThreadInfo：保存展开流信息到 host 内存
HcclResult AiCpuEngine::SaveUnfoldThreadInfoInternal(HcclComm comm, const OpParam &param,
    ThreadHandle unfoldThread)
{
    uint64_t size = sizeof(ThreadHandle);
    void *ctx = nullptr;
    char unfoldAlgTag[ALG_TAG_LENGTH] = {0};
    int ret = snprintf_s(unfoldAlgTag, sizeof(unfoldAlgTag), sizeof(unfoldAlgTag) - 1, "%s_unfold", param.algTag);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[%s] failed to fill unfoldAlgTag", __func__), HCCL_E_INTERNAL);
    CHK_RET(HcclEngineCtxCreate(comm, unfoldAlgTag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle *>(ctx);
    *threadPtr = unfoldThread;
    HCCL_INFO("[SaveUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]",
        unfoldAlgTag, threadPtr, unfoldThread);
    return HCCL_SUCCESS;
}

// 对应原始 HcclGetChannelImpl：按 CommEngine 建链并填充 ChannelInfo
HcclResult AiCpuEngine::HcclGetChannelImplInternal(u32 level, HcclComm comm, const OpParam &param,
    std::vector<HcclChannelDesc>& channelRequest, CommEngine commEngine)
{
    if (channelRequest.empty()) {
        HCCL_INFO("[HcclGetChannelImpl] channelRequest is empty");
        return HCCL_SUCCESS;
    }
    u32 channelNum = static_cast<u32>(channelRequest.size());
    std::vector<ChannelHandle> levelNChannels(channelNum);
    if (channelNum > 0) {
        // 参数一致性校验信息注册到通信域，HcclChannelAcquire 内部存在读清动作，每次调用前均需注册
        CHK_RET(AddExchangeInfoInternal(comm, param));
        CHK_RET(HcclChannelAcquire(comm, commEngine, channelRequest.data(),
            channelNum, levelNChannels.data()));
    }

    for (u32 idx = 0; idx < channelNum; idx++) {
        ChannelInfo channel;
        const HcclChannelDesc &channelDesc = channelRequest[idx];
        channel.isValid = true;
        channel.remoteRank = channelDesc.remoteRank;
        channel.protocol = channelDesc.channelProtocol;
        channel.locationType = channelDesc.remoteEndpoint.loc.locType;
        channel.notifyNum = channelDesc.notifyNum;
        channel.handle = levelNChannels[idx];
        // 获取远端 CCL buffer
        void* remoteCclBufferAddr = nullptr;
        uint64_t remoteCclBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteCclBufferAddr, &remoteCclBufferSize));
        channel.remoteCclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteCclBufferAddr, remoteCclBufferSize};
        HCCL_INFO("[HcclGetChannelImpl]remoteRank[%u] protocol[%u] remoteCclBufferAddr[0x%llx] remoteCclBufferSize[%u]",
            channelDesc.remoteRank, channelDesc.channelProtocol, remoteCclBufferAddr, remoteCclBufferSize);
        resCtx_.channels[level].push_back(channel);
    }
    return HCCL_SUCCESS;
}

// 对应原始 AddExchangeInfo：参数一致性校验信息注册
HcclResult AiCpuEngine::AddExchangeInfoInternal(HcclComm comm, const OpParam &param)
{
    CHK_PTR_NULL(comm);
    // 简化：仅注册交换信息，NeedInconsistentCheck 在重构中暂未引入
    return HCCL_SUCCESS;
}

HcclResult AiCpuEngine::LaunchKernel(const OpParam &param)
{
    HCCL_INFO("[AiCpuEngine][LaunchKernel] start, commName[%s], tag[%s], algTag[%s]",
              param.commName, param.tag, param.algTag);

    // 对应原始 HcclAicpuKernelEntranceLaunch + AicpuKernelLaunch
    CHK_RET(AicpuKernelEntranceLaunchInternal(param));

    HCCL_INFO("[AiCpuEngine][LaunchKernel] end, tag[%s], algTag[%s], commName[%s]",
              param.tag, param.algTag, param.commName);
    return HCCL_SUCCESS;
}

// 对应原始 HcclAicpuKernelEntranceLaunch：host 侧下发 AICPU kernel
HcclResult AiCpuEngine::AicpuKernelEntranceLaunchInternal(const OpParam &param)
{
    HCCL_DEBUG("[AicpuKernelEntranceLaunch]start to run aicpu kernel");
    // 当前aicpu launch接口只能有一个输入参数，将Context指针放在param参数中
    const_cast<OpParam &>(param).aicpuRecordCpuIdx = HOST_WAIT_AICPU_NOTIFYIDX;
    const_cast<OpParam &>(param).resCtx = &resCtx_;

    if (param.engine == COMM_ENGINE_CPU) {
        // 注册dpu回调函数
        CHK_RET(static_cast<HcclResult>(HcclTaskRegister(resCtx_.commInfoPtr, param.algTag, HcclLaunchDPUKernel)));
    }

    // Host stream通知Device主thread，使用主流上idx最大的notify
    ThreadHandle cpuTsThread = resCtx_.cpuTsThread;
    ThreadHandle exportedCpuTsThread = resCtx_.exportedCpuTsThread;
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(cpuTsThread, exportedCpuTsThread,
        resCtx_.notifyNumOnMainThread - 1)));
    // AicpuKernel report
    uint64_t beginTime = HcommGetProfilingSysCycleTime();
    CHK_RET(AicpuKernelLaunchInternal(param, resCtx_.unfoldThread));
    CHK_PTR_NULL(resCtx_.commInfoPtr);
    std::string kernelName = "HcclLaunchAicpuKernel";
    char* kernelNameCStr = const_cast<char*>(kernelName.c_str());
    HcclResult ret = HcclReportAicpuKernel(resCtx_.commInfoPtr, beginTime, kernelNameCStr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[AicpuKernelEntranceLaunch] HcclReportAicpuKernel failed, beginTime %lu, kernelNameCStr %s, ret %d ",
            beginTime, kernelNameCStr, ret);
        return ret;
    }
    // Host stream等待Device的通知
    AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
    u32 hostNotifyWaitTime = IsHcommDefaultTimeoutSupported() ? timeout.hostNotifyTimeout :
        AddAicpuTimeoutOffset(param.opConfig.execTimeout, HOST_NOTIFY_TIMEOUT_OFFSET);
    if (HcommIsSupportHcommSetNotifyWaitTimeOut()) {
        CHK_RET(HcclSetNotifyWaitTimeOut(hostNotifyWaitTime));
    }
    CHK_RET(HcclThreadNotifyWaitOnThreadDefault(cpuTsThread, param.aicpuRecordCpuIdx, hostNotifyWaitTime));

    return HCCL_SUCCESS;
}

// 对应原始 AicpuKernelLaunch：通过 aclrt API 下发 AICPU kernel binary
HcclResult AiCpuEngine::AicpuKernelLaunchInternal(const OpParam &param, ThreadHandle unfoldThread)
{
    std::string kernelName = "HcclLaunchAicpuKernel";
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    // 注意，目前开源HCCL加载AICPU kernel使用的是从json文件加载
    // 详见load_kernel.cc中的LoadAICPUKernel函数，且只实现了scatter的，先共用scatter的
    aclError ret = aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, "
        "kernelName:%s", ret, kernelName.c_str()), HCCL_E_RUNTIME);
    ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, "
        "kernelName:%s", ret, kernelName.c_str()), HCCL_E_RUNTIME);
    aclrtParamHandle paraHandle;
    size_t paramSize = sizeof(OpParam) + param.varMemSize;
    ret = aclrtKernelArgsAppend(argsHandle, const_cast<OpParam*>(&param), paramSize, &paraHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append "
        "size %u, kernelName:%s", ret, paramSize, kernelName.c_str()), HCCL_E_RUNTIME);
    ret = aclrtKernelArgsFinalize(argsHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, "
        "kernelName:%s", ret, kernelName.c_str()), HCCL_E_RUNTIME);

    AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
    u16 kernelLaunchTimeout = IsHcommDefaultTimeoutSupported() ? timeout.kernelLaunchTimeout :
        ToKernelLaunchTimeout(AddAicpuTimeoutOffset(param.opConfig.execTimeout, KERNEL_TIMEOUT_OFFSET));
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = kernelLaunchTimeout;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr u32 numBlocks = 1;
    HCCL_INFO("[AicpuKernelLaunch] unfoldThread [%lu]", unfoldThread);
    void* unfoldStream = nullptr;
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo || param.opMode == OpMode::OFFLOAD) { // 不走提前展开
        ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
    } else {
        HcclResult ret1 = HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(
            resCtx_.commInfoPtr, unfoldThread, 0, sizeof(void*), &unfoldStream);
        if (ret1 == HCCL_E_NOT_SUPPORT) {
            ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
        } else if (ret1 != HCCL_SUCCESS) {
            return ret1;
        } else {
            ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, unfoldStream, &cfg, argsHandle, nullptr);
        }
    }
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[AicpuKernelLaunch][aclrtLaunchKernelWithConfig]"
        "errNo[0x%016llx] launch kernel failed", ret), HCCL_E_OPEN_FILE_FAILURE);
    return HCCL_SUCCESS;
}

HcclResult AiCpuEngine::Send(const TransferContext &ctx) {
    // Step 1: 方向判断
    //   enableRemoteMemAccess && buffType==OUTPUT → READ (从远端拉取到本地 output)
    //   其他场景 → WRITE (推送数据到远端)
    TransferDirection direction =
        (ctx.enableRemoteMemAccess && ctx.buffType == BufferType::OUTPUT)
            ? TransferDirection::READ : TransferDirection::WRITE;

    // Step 2: 获取线程
    if (ctx.templateRes.threads.empty()) {
        HCCL_ERROR("[AiCpuEngine][Send] threads is empty");
        return HCCL_E_INTERNAL;
    }
    const ThreadHandle &thread = ctx.templateRes.threads[0];

    // Step 3: Rank → Channel 映射
    u32 dstRank = ctx.txRxSlicesList.dstRankId_;
    u32 srcRank = ctx.txRxSlicesList.srcRankId_;
    const auto &channels = ctx.templateRes.channels;
    const ChannelInfo *txCh = LookupChannel(channels, dstRank);
    const ChannelInfo *rxCh = LookupChannel(channels, srcRank);

    // Step 4: 判断发送/接收/双向 (不再判断 hasReduce)
    bool hasTx = !ctx.txRxSlicesList.txSlicesList_.srcSlices_.empty();
    bool hasRx = !ctx.txRxSlicesList.rxSlicesList_.srcSlices_.empty();

    // 空操作
    if (!hasTx && !hasRx) {
        return HCCL_SUCCESS;
    }

    // Step 5: 路由 (direction × hasTx/hasRx → 6 个统一 wrapper, reduceOp 透传)
    if (hasTx && hasRx && txCh != nullptr && rxCh != nullptr) {
        // ── 双向: SendRecv* 系列 ──
        TxRxChannels channelPair(*txCh, *rxCh);
        TxRxSlicesList slicesList(ctx.txRxSlicesList.txSlicesList_,
                                  ctx.txRxSlicesList.rxSlicesList_);
        SendRecvInfo info(channelPair, slicesList, ctx.dataType);
        if (direction == TransferDirection::READ) {
            return SendRecvRead(info, thread, ctx.reduceOp);
        }
        return SendRecvWrite(info, thread, ctx.reduceOp);
    }
    if (hasTx && txCh != nullptr) {
        // ── 只发送: Send* 系列 ──
        SlicesList slices = ctx.txRxSlicesList.txSlicesList_;
        DataInfo info(*txCh, slices, ctx.dataType);
        if (direction == TransferDirection::READ) {
            return SendRead(info, thread, ctx.reduceOp);
        }
        return SendWrite(info, thread, ctx.reduceOp);
    }
    if (hasRx && rxCh != nullptr) {
        // ── 只接收: Recv* 系列 ──
        SlicesList slices = ctx.txRxSlicesList.rxSlicesList_;
        DataInfo info(*rxCh, slices, ctx.dataType);
        if (direction == TransferDirection::READ) {
            return RecvRead(info, thread, ctx.reduceOp);
        }
        return RecvWrite(info, thread, ctx.reduceOp);
    }

    HCCL_ERROR("[AiCpuEngine][Send] channel not found, dstRank[%u], srcRank[%u]", dstRank, srcRank);
    return HCCL_E_INTERNAL;
}

// ───────────── Write 系列 (推送) ─────────────
HcclResult AiCpuEngine::SendWrite(const DataInfo &sendInfo, const ThreadHandle &thread, HcclReduceOp reduceOp)
{
    const std::vector<DataSlice> srcSlices = sendInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendInfo.slices_.dstSlices_;
    const ChannelInfo &sendChannel = sendInfo.channel_;
    u32 sliceNum = srcSlices.size();
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    bool isReduce = (reduceOp != HCCL_REDUCE_RESERVED);
    const char *transType = isReduce ? "WRITE_REDUCE" : "WRITE";
    for (u32 i = 0; i < sliceNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) { continue; }
        void *dst = GetSliceAddr(dstSlice);
        void *src = GetSliceAddr(srcSlice);
        TraceDataSlice("SendWrite", transType, i, sliceNum, srcSlice, dstSlice, src, dst,
            isReduce ? srcSlice.count_ : srcSlice.size_, sendInfo.dataType_, reduceOp);
        if (isReduce) {
            CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread, sendChannel.handle, dst, src, srcSlice.count_,
                static_cast<HcommDataType>(sendInfo.dataType_), static_cast<HcommReduceOp>(reduceOp))));
        } else {
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, sendChannel.handle, dst, src, srcSlice.size_)));
        }
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuEngine::RecvWrite(const DataInfo &recvInfo, const ThreadHandle &thread, HcclReduceOp reduceOp)
{
    // Write 模式下接收方只做 notify 同步, reduce 发生在对端(写方), reduceOp 此处不影响。
    (void)reduceOp;
    const ChannelInfo &recvChannel = recvInfo.channel_;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuEngine::SendRecvWrite(const SendRecvInfo &sendRecvInfo, const ThreadHandle &thread, HcclReduceOp reduceOp)
{
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_;
    const ChannelInfo &sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo &recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    u32 repeatNum = srcSlices.size();
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    bool isReduce = (reduceOp != HCCL_REDUCE_RESERVED);
    const char *transType = isReduce ? "WRITE_REDUCE" : "WRITE";
    for (u32 i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) { continue; }
        void *dst = GetSliceAddr(dstSlice);
        void *src = GetSliceAddr(srcSlice);
        TraceDataSlice("SendRecvWrite", transType, i, repeatNum, srcSlice, dstSlice, src, dst,
            isReduce ? srcSlice.count_ : srcSlice.size_, sendRecvInfo.dataType_, reduceOp);
        if (isReduce) {
            CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread, sendChannel.handle, dst, src, srcSlice.count_,
                static_cast<HcommDataType>(sendRecvInfo.dataType_), static_cast<HcommReduceOp>(reduceOp))));
        } else {
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, sendChannel.handle, dst, src, srcSlice.size_)));
        }
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

// ───────────── Read 系列 (拉取) ─────────────

HcclResult AiCpuEngine::SendRead(const DataInfo &sendInfo, const ThreadHandle &thread, HcclReduceOp reduceOp)
{
    // Read 模式下被读方只做 notify 同步, 实际搬移由对端(读方)完成, reduceOp 此处不影响。
    (void)reduceOp;
    const ChannelInfo &sendChannel = sendInfo.channel_;
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuEngine::RecvRead(const DataInfo &recvInfo, const ThreadHandle &thread, HcclReduceOp reduceOp)
{
    const std::vector<DataSlice> srcSlices = recvInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = recvInfo.slices_.dstSlices_;
    const ChannelInfo &recvChannel = recvInfo.channel_;
    u32 repeatNum = srcSlices.size();
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    bool isReduce = (reduceOp != HCCL_REDUCE_RESERVED);
    const char *transType = isReduce ? "READ_REDUCE" : "READ";
    for (u32 i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) { continue; }
        void *dst = GetSliceAddr(dstSlice);
        void *src = GetSliceAddr(srcSlice);
        TraceDataSlice("RecvRead", transType, i, repeatNum, srcSlice, dstSlice, src, dst,
            isReduce ? srcSlice.count_ : srcSlice.size_, recvInfo.dataType_, reduceOp);
        if (isReduce) {
            CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, recvChannel.handle, dst, src, srcSlice.count_,
                static_cast<HcommDataType>(recvInfo.dataType_), static_cast<HcommReduceOp>(reduceOp))));
        } else {
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, recvChannel.handle, dst, src, srcSlice.size_)));
        }
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuEngine::SendRecvRead(const SendRecvInfo &sendRecvInfo, const ThreadHandle &thread,
                                       HcclReduceOp reduceOp)
{
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_;
    const ChannelInfo &sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo &recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    u32 repeatNum = srcSlices.size();
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    bool isReduce = (reduceOp != HCCL_REDUCE_RESERVED);
    const char *transType = isReduce ? "READ_REDUCE" : "READ";
    for (u32 i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) { continue; }
        void *dst = GetSliceAddr(dstSlice);
        void *src = GetSliceAddr(srcSlice);
        TraceDataSlice("SendRecvRead", transType, i, repeatNum, srcSlice, dstSlice, src, dst,
            isReduce ? srcSlice.count_ : srcSlice.size_, sendRecvInfo.dataType_, reduceOp);
        if (isReduce) {
            CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, recvChannel.handle, dst, src, srcSlice.count_,
                static_cast<HcommDataType>(sendRecvInfo.dataType_), static_cast<HcommReduceOp>(reduceOp))));
        } else {
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, recvChannel.handle, dst, src, srcSlice.size_)));
        }
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
