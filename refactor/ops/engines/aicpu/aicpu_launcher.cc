/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_launcher.h"

#include "load_kernel.h"
#include "ops_executor.h"
#include "utils/utils.h"
#include "log.h"

#include "hcomm_primitives_dl.h"
#include "hccl_res_dl.h"
#include "exec_timeout_manager.h"

#include <atomic>
#include <vector>

namespace ops_hccl {


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
    HCCL_DEBUG("[AicpuLauncher][%s][%s] sliceIdx[%u], sliceNum[%u], src[%p], dst[%p], len[%llu].",
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

// ───────────── 自由函数：供 template 层调用的 SendRecv ─────────────

HcclResult SendRecvRead(const SendRecvInfo &info, const ThreadHandle &thread)
{
    const ChannelInfo &sendChannel = info.sendRecvChannels_.txChannel_;
    const ChannelInfo &recvChannel = info.sendRecvChannels_.rxChannel_;
    const std::vector<DataSlice> &srcSlices = info.sendRecvSlices_.rxSlicesList_.srcSlices_;
    const std::vector<DataSlice> &dstSlices = info.sendRecvSlices_.rxSlicesList_.dstSlices_;
    const u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (u32 j = 0; j < srcSlices.size(); ++j) {
        if (srcSlices[j].size_ == 0) { continue; }
        void *src = GetSliceAddr(srcSlices[j]);
        void *dst = GetSliceAddr(dstSlices[j]);
        CHK_RET(static_cast<HcclResult>(
            HcommReadOnThread(thread, recvChannel.handle, dst, src, srcSlices[j].size_)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

HcclResult SendRecvWrite(const SendRecvInfo &info, const ThreadHandle &thread)
{
    const ChannelInfo &sendChannel = info.sendRecvChannels_.txChannel_;
    const ChannelInfo &recvChannel = info.sendRecvChannels_.rxChannel_;
    const std::vector<DataSlice> &srcSlices = info.sendRecvSlices_.txSlicesList_.srcSlices_;
    const std::vector<DataSlice> &dstSlices = info.sendRecvSlices_.txSlicesList_.dstSlices_;
    const u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (u32 j = 0; j < srcSlices.size(); ++j) {
        if (srcSlices[j].size_ == 0) { continue; }
        void *src = GetSliceAddr(srcSlices[j]);
        void *dst = GetSliceAddr(dstSlices[j]);
        CHK_RET(static_cast<HcclResult>(
            HcommWriteOnThread(thread, sendChannel.handle, dst, src, srcSlices[j].size_)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}


// ═══════════════════════════════════════════════════════════════════
// CreateRes / LaunchKernel / Send
// ═══════════════════════════════════════════════════════════════════

HcclResult AiCpuLauncher::CreateRes(AlgResourceRequest &res)
{
    resCtx_.notifyNumOnMainThread = res.notifyNumOnMainThread;
    resCtx_.slaveThreadNum = res.slaveThreadNum;
    resCtx_.notifyNumPerThread = res.notifyNumPerThread;

    void *cclBufferAddr = nullptr;
    u64 cclBufferSize = 0;
    resCtx_.cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};

    resCtx_.threads.resize(res.slaveThreadNum + 1);

    // 按层级申请 channel（迁移自 op_common.cc:HcclGetChannelImpl）。
    // AICPU 引擎使用 CommEngine::COMM_ENGINE_CPU。
    for (size_t level = 0; level < res.channels.size(); ++level) {
        std::vector<HcclChannelDesc> &channelRequest = res.channels[level];
        u32 channelNum = static_cast<u32>(channelRequest.size());
        std::vector<ChannelHandle> levelNChannels(channelNum);
        if (channelNum > 0) {
            CHK_RET(HcclChannelAcquire(comm_, CommEngine::COMM_ENGINE_CPU, channelRequest.data(),
                channelNum, levelNChannels.data()));
        }
        std::vector<ChannelInfo> levelChannels;
        levelChannels.reserve(channelNum);
        for (u32 idx = 0; idx < channelNum; ++idx) {
            const HcclChannelDesc &channelDesc = channelRequest[idx];
            ChannelInfo channelInfo;
            channelInfo.isValid = true;
            channelInfo.remoteRank = channelDesc.remoteRank;
            channelInfo.protocol = channelDesc.channelProtocol;
            channelInfo.locationType = channelDesc.remoteEndpoint.loc.locType;
            channelInfo.notifyNum = channelDesc.notifyNum;
            channelInfo.handle = levelNChannels[idx];
            levelChannels.emplace_back(std::move(channelInfo));
        }
        resCtx_.channels.emplace_back(std::move(levelChannels));
    }

    HCCL_INFO("[AiCpuLauncher][CreateRes] success, slaveThreadNum[%u], channelLevel[%zu]",
              resCtx_.slaveThreadNum, resCtx_.channels.size());
    return HCCL_SUCCESS;
}

HcclResult AiCpuLauncher::LaunchKernel(const OpParam &param, OpsExecutor &executor)
{
    HCCL_INFO("[AiCpuLauncher][LaunchKernel] start, commName[%s], tag[%s], algTag[%s]",
              param.commName, param.tag, param.algTag);
    CHK_RET(LoadAICPUKernel());
    CHK_RET(HcclLaunchAicpuKernel(param, executor, resCtx_));
    HCCL_INFO("[AiCpuLauncher][LaunchKernel] end, tag[%s], algTag[%s], commName[%s]",
              param.tag, param.algTag, param.commName);
    return HCCL_SUCCESS;
}

HcclResult AiCpuLauncher::Send(const TransferContext &ctx) {
    // Step 1: 方向判断
    //   enableRemoteMemAccess && buffType==OUTPUT → READ (从远端拉取到本地 output)
    //   其他场景 → WRITE (推送数据到远端)
    TransferDirection direction =
        (ctx.enableRemoteMemAccess && ctx.buffType == BufferType::OUTPUT)
            ? TransferDirection::READ : TransferDirection::WRITE;

    // isDmaRead: 记录 PCIe 链路状态 (供日志/调试使用)
    bool isDmaRead = IsPcieProtocol(ctx.templateRes.channels);

    // Step 2: 获取线程
    if (ctx.templateRes.threads.empty()) {
        HCCL_ERROR("[AiCpuLauncher][Send] threads is empty");
        return HCCL_E_INTERNAL;
    }
    const ThreadHandle &thread = ctx.templateRes.threads[0];

    // Step 3: Rank → Channel 映射
    u32 dstRank = ctx.txRxSlicesList.dstRankId_;
    u32 srcRank = ctx.txRxSlicesList.srcRankId_;
    const auto &channels = ctx.templateRes.channels;
    const ChannelInfo *txCh = LookupChannel(channels, dstRank);
    const ChannelInfo *rxCh = LookupChannel(channels, srcRank);

    // Step 4: 判断发送/接收/双向
    bool hasTx = !ctx.txRxSlicesList.txSlicesList_.srcSlices_.empty();
    bool hasRx = !ctx.txRxSlicesList.rxSlicesList_.srcSlices_.empty();
    bool hasReduce = (ctx.reduceOp != HCCL_REDUCE_RESERVED);

    // 空操作
    if (!hasTx && !hasRx) {
        return HCCL_SUCCESS;
    }

    HCCL_DEBUG("[AiCpuLauncher][Send] direction[%d], isDmaRead[%d], hasTx[%d], hasRx[%d], hasReduce[%d]",
        static_cast<int>(direction), static_cast<int>(isDmaRead),
        static_cast<int>(hasTx), static_cast<int>(hasRx), static_cast<int>(hasReduce));

    // Step 5: 路由 (direction × tx/rx × reduce → 12 条路径)
    if (hasTx && hasRx && txCh != nullptr && rxCh != nullptr) {
        // ── 双向: SendRecv* 系列 ──
        TxRxChannels channelPair(*txCh, *rxCh);
        TxRxSlicesList slicesList(ctx.txRxSlicesList.txSlicesList_,
                                  ctx.txRxSlicesList.rxSlicesList_);
        if (direction == TransferDirection::READ) {
            if (hasReduce) {
                SendRecvReduceInfo reduceInfo(channelPair, slicesList, ctx.dataType, ctx.reduceOp);
                return SendRecvReadReduce(reduceInfo, thread);
            }
            SendRecvInfo info(channelPair, slicesList, ctx.dataType);
            return SendRecvRead(info, thread);
        } else {
            if (hasReduce) {
                SendRecvReduceInfo reduceInfo(channelPair, slicesList, ctx.dataType, ctx.reduceOp);
                return SendRecvWriteReduce(reduceInfo, thread);
            }
            SendRecvInfo info(channelPair, slicesList, ctx.dataType);
            return SendRecvWrite(info, thread);
        }
    } else if (hasTx && txCh != nullptr) {
        // ── 只发送: Send* 系列 ──
        SlicesList slices = ctx.txRxSlicesList.txSlicesList_;
        if (direction == TransferDirection::READ) {
            if (hasReduce) {
                DataReduceInfo reduceInfo(*txCh, slices, ctx.dataType, ctx.reduceOp);
                return SendReadReduce(reduceInfo, thread);
            }
            DataInfo info(*txCh, slices, ctx.dataType);
            return SendRead(info, thread);
        } else {
            if (hasReduce) {
                DataReduceInfo reduceInfo(*txCh, slices, ctx.dataType, ctx.reduceOp);
                return SendWriteReduce(reduceInfo, thread);
            }
            DataInfo info(*txCh, slices, ctx.dataType);
            return SendWrite(info, thread);
        }
    } else if (hasRx && rxCh != nullptr) {
        // ── 只接收: Recv* 系列 ──
        SlicesList slices = ctx.txRxSlicesList.rxSlicesList_;
        if (direction == TransferDirection::READ) {
            if (hasReduce) {
                DataReduceInfo reduceInfo(*rxCh, slices, ctx.dataType, ctx.reduceOp);
                return RecvReadReduce(reduceInfo, thread);
            }
            DataInfo info(*rxCh, slices, ctx.dataType);
            return RecvRead(info, thread);
        } else {
            if (hasReduce) {
                DataReduceInfo reduceInfo(*rxCh, slices, ctx.dataType, ctx.reduceOp);
                return RecvWriteReduce(reduceInfo, thread);
            }
            DataInfo info(*rxCh, slices, ctx.dataType);
            return RecvWrite(info, thread);
        }
    }

    HCCL_ERROR("[AiCpuLauncher][Send] channel not found, dstRank[%u], srcRank[%u]", dstRank, srcRank);
    return HCCL_E_INTERNAL;
}

// ═══════════════════════════════════════════════════════════════════
// AICPU 数据传输 wrapper 成员函数实现 (共 12 个)
// 复制自 alg_data_trans_wrapper.cc, 原文件保留不动。
// Write 系列: HcommWriteOnThread (本端推送)
// Read 系列: HcommReadOnThread (本端拉取)
// Send* = 发起方, Recv* = 接收方, SendRecv* = 双向
// ═══════════════════════════════════════════════════════════════════

// ───────────── Write 系列 (推送) ─────────────

HcclResult AiCpuLauncher::SendWrite(const DataInfo &sendInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = sendInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendInfo.slices_.dstSlices_;
    const ChannelInfo &sendChannel = sendInfo.channel_;
    u32 sliceNum = srcSlices.size();
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (u32 i = 0; i < sliceNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) { continue; }
        void *dst = GetSliceAddr(dstSlice);
        void *src = GetSliceAddr(srcSlice);
        TraceDataSlice("SendWrite", "WRITE", i, sliceNum, srcSlice, dstSlice, src, dst,
            srcSlice.size_, sendInfo.dataType_, HcclReduceOp::HCCL_REDUCE_RESERVED);
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, sendChannel.handle, dst, src, srcSlice.size_)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuLauncher::RecvWrite(const DataInfo &recvInfo, const ThreadHandle &thread)
{
    const ChannelInfo &recvChannel = recvInfo.channel_;
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuLauncher::SendRecvWrite(const SendRecvInfo &sendRecvInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_;
    const ChannelInfo &sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo &recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    u32 repeatNum = srcSlices.size();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (u32 i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) { continue; }
        void *dst = GetSliceAddr(dstSlice);
        void *src = GetSliceAddr(srcSlice);
        TraceDataSlice("SendRecvWrite", "WRITE", i, repeatNum, srcSlice, dstSlice, src, dst,
            srcSlice.size_, sendRecvInfo.dataType_, HcclReduceOp::HCCL_REDUCE_RESERVED);
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, sendChannel.handle, dst, src, srcSlice.size_)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuLauncher::SendWriteReduce(const DataReduceInfo &sendInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = sendInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendInfo.slices_.dstSlices_;
    const ChannelInfo &sendChannel = sendInfo.channel_;
    u32 repeatNum = srcSlices.size();
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (u32 i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) { continue; }
        void *dst = GetSliceAddr(dstSlice);
        void *src = GetSliceAddr(srcSlice);
        TraceDataSlice("SendWriteReduce", "WRITE_REDUCE", i, repeatNum, srcSlice, dstSlice, src, dst,
            srcSlice.count_, sendInfo.dataType_, sendInfo.reduceType_);
        CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread,
            sendChannel.handle, dst, src, srcSlice.count_,
            static_cast<HcommDataType>(sendInfo.dataType_),
            static_cast<HcommReduceOp>(sendInfo.reduceType_))));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuLauncher::RecvWriteReduce(const DataReduceInfo &recvInfo, const ThreadHandle &thread)
{
    const ChannelInfo &recvChannel = recvInfo.channel_;
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuLauncher::SendRecvWriteReduce(const SendRecvReduceInfo &sendRecvInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_;
    const ChannelInfo &sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo &recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    u32 repeatNum = srcSlices.size();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK)));
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (u32 i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) { continue; }
        void *dst = GetSliceAddr(dstSlice);
        void *src = GetSliceAddr(srcSlice);
        TraceDataSlice("SendRecvWriteReduce", "WRITE_REDUCE", i, repeatNum, srcSlice, dstSlice, src, dst,
            srcSlice.count_, sendRecvInfo.dataType_, sendRecvInfo.reduceType_);
        CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread,
            sendChannel.handle, dst, src, srcSlice.count_,
            static_cast<HcommDataType>(sendRecvInfo.dataType_),
            static_cast<HcommReduceOp>(sendRecvInfo.reduceType_))));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

// ───────────── Read 系列 (拉取) ─────────────

HcclResult AiCpuLauncher::SendRead(const DataInfo &sendInfo, const ThreadHandle &thread)
{
    const ChannelInfo &sendChannel = sendInfo.channel_;
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuLauncher::RecvRead(const DataInfo &recvInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = recvInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = recvInfo.slices_.dstSlices_;
    const ChannelInfo &recvChannel = recvInfo.channel_;
    u32 repeatNum = srcSlices.size();
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (u32 i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) { continue; }
        void *dst = GetSliceAddr(dstSlice);
        void *src = GetSliceAddr(srcSlice);
        TraceDataSlice("RecvRead", "READ", i, repeatNum, srcSlice, dstSlice, src, dst,
            srcSlice.size_, recvInfo.dataType_, HcclReduceOp::HCCL_REDUCE_RESERVED);
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, recvChannel.handle, dst, src, srcSlice.size_)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuLauncher::SendRecvRead(const SendRecvInfo &sendRecvInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_;
    const ChannelInfo &sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo &recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    u32 repeatNum = srcSlices.size();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (u32 i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) { continue; }
        void *dst = GetSliceAddr(dstSlice);
        void *src = GetSliceAddr(srcSlice);
        TraceDataSlice("SendRecvRead", "READ", i, repeatNum, srcSlice, dstSlice, src, dst,
            srcSlice.size_, sendRecvInfo.dataType_, HcclReduceOp::HCCL_REDUCE_RESERVED);
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, recvChannel.handle, dst, src, srcSlice.size_)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuLauncher::SendReadReduce(const DataReduceInfo &sendInfo, const ThreadHandle &thread)
{
    const ChannelInfo &sendChannel = sendInfo.channel_;
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuLauncher::RecvReadReduce(const DataReduceInfo &recvInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = recvInfo.slices_.srcSlices_;
    const std::vector<DataSlice> dstSlices = recvInfo.slices_.dstSlices_;
    const ChannelInfo &recvChannel = recvInfo.channel_;
    u32 repeatNum = srcSlices.size();
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (u32 i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) { continue; }
        void *dst = GetSliceAddr(dstSlice);
        void *src = GetSliceAddr(srcSlice);
        TraceDataSlice("RecvReadReduce", "READ_REDUCE", i, repeatNum, srcSlice, dstSlice, src, dst,
            srcSlice.count_, recvInfo.dataType_, recvInfo.reduceType_);
        CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread,
            recvChannel.handle, dst, src, srcSlice.count_,
            static_cast<HcommDataType>(recvInfo.dataType_),
            static_cast<HcommReduceOp>(recvInfo.reduceType_))));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult AiCpuLauncher::SendRecvReadReduce(const SendRecvReduceInfo &sendRecvInfo, const ThreadHandle &thread)
{
    const std::vector<DataSlice> srcSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_;
    const std::vector<DataSlice> dstSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_;
    const ChannelInfo &sendChannel = sendRecvInfo.sendRecvChannels_.txChannel_;
    const ChannelInfo &recvChannel = sendRecvInfo.sendRecvChannels_.rxChannel_;
    u32 repeatNum = srcSlices.size();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, sendChannel.handle, NOTIFY_IDX_ACK)));
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, recvChannel.handle, NOTIFY_IDX_ACK, execTimeout)));
    for (u32 i = 0; i < repeatNum; i++) {
        const DataSlice srcSlice = srcSlices[i];
        const DataSlice dstSlice = dstSlices[i];
        if (srcSlice.size_ == 0) { continue; }
        void *dst = GetSliceAddr(dstSlice);
        void *src = GetSliceAddr(srcSlice);
        TraceDataSlice("SendRecvReadReduce", "READ_REDUCE", i, repeatNum, srcSlice, dstSlice, src, dst,
            srcSlice.count_, sendRecvInfo.dataType_, sendRecvInfo.reduceType_);
        CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread,
            recvChannel.handle, dst, src, srcSlice.count_,
            static_cast<HcommDataType>(sendRecvInfo.dataType_),
            static_cast<HcommReduceOp>(sendRecvInfo.reduceType_))));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, sendChannel.handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
