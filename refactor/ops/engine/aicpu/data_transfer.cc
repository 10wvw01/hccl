/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_transfer.h"

#include "alg_param.h"
#include "log.h"
#include "hcomm_primitives_dl.h"
#include "exec_timeout_manager.h"

namespace ops_hccl {

// ───────────── 静态辅助函数 ─────────────

static void *GetSliceAddr(const DataSlice &slice, void *fallbackBase = nullptr)
{
    void *base = (slice.addr_ == nullptr) ? fallbackBase : slice.addr_;
    if (base == nullptr) {
        return nullptr;
    }
    return static_cast<void *>(static_cast<s8 *>(base) + slice.offset_);
}

static void TraceDataSlice(const char *funcName, const char *transType, u32 sliceIdx, u32 sliceNum,
    const DataSlice &srcSlice, const DataSlice &dstSlice, const void *src, const void *dst, u64 len,
    HcclDataType dataType, HcclReduceOp reduceOp)
{
    HCCL_DEBUG("[DataTransfer][%s][%s] sliceIdx[%u], sliceNum[%u], src[%p], dst[%p], len[%llu].",
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

// ───────────── Write 系列 (非PCIe: 本端主动推送) ─────────────

static HcclResult SendWrite(const DataInfo &sendInfo, const ThreadHandle &thread, HcclReduceOp reduceOp)
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
        void *dst = GetSliceAddr(dstSlice, sendChannel.remoteCclMem.addr);
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

static HcclResult RecvWrite(const DataInfo &recvInfo, const ThreadHandle &thread, HcclReduceOp reduceOp)
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

static HcclResult SendRecvWrite(const SendRecvInfo &sendRecvInfo, const ThreadHandle &thread, HcclReduceOp reduceOp)
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
        void *dst = GetSliceAddr(dstSlice, sendChannel.remoteCclMem.addr);
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

// ───────────── Read 系列 (PCIe: 本端主动拉取) ─────────────

static HcclResult SendRead(const DataInfo &sendInfo, const ThreadHandle &thread, HcclReduceOp reduceOp)
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

static HcclResult RecvRead(const DataInfo &recvInfo, const ThreadHandle &thread, HcclReduceOp reduceOp)
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
        void *src = GetSliceAddr(srcSlice, recvChannel.remoteCclMem.addr);
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

static HcclResult SendRecvRead(const SendRecvInfo &sendRecvInfo, const ThreadHandle &thread,
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
        void *src = GetSliceAddr(srcSlice, recvChannel.remoteCclMem.addr);
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

// ═══════════════════════════════════════════════════════════════════
// DataTransferSend：数据传输入口
// ═══════════════════════════════════════════════════════════════════

HcclResult DataTransferSend(const TransferContext &ctx) {
    // Step 1: 方向判断
    //   enableRemoteMemAccess && buffType==OUTPUT → READ (从远端拉取到本地 output)
    //   其他场景 → WRITE (推送数据到远端)
    TransferDirection direction =
        (ctx.enableRemoteMemAccess && ctx.buffType == BufferType::OUTPUT)
            ? TransferDirection::READ : TransferDirection::WRITE;

    // Step 2: 获取线程
    if (ctx.templateRes.threads.empty()) {
        HCCL_ERROR("[DataTransfer][Send] threads is empty");
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

    HCCL_ERROR("[DataTransfer][Send] channel not found, dstRank[%u], srcRank[%u]", dstRank, srcRank);
    return HCCL_E_INTERNAL;
}

}  // namespace ops_hccl
