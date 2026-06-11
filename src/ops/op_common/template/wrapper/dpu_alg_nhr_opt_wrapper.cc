/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpu_alg_nhr_opt_wrapper.h"
#include "hcomm_primitives.h"

namespace ops_hccl {

// ========== 内部辅助：构造 tx/rx slice 列表 ==========
namespace {

void BuildTxSlices(
    const AicpuNHRStepInfo &stepInfo,
    const TemplateDataParams &tempAlgParam,
    u32 repeat,
    u32 templateRankSize,
    void *sendCclBuffAddr,
    std::vector<DataSlice> &txSrcSlices,
    std::vector<DataSlice> &txDstSlices)
{
    for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
        u32 txId = stepInfo.txSliceIdxs.at(i);
        u64 sliceSize = tempAlgParam.allRankSliceSize.at(txId);
        u64 sliceCount = tempAlgParam.allRankProcessedDataCount.at(txId);
        u64 sliceOffset = tempAlgParam.allRankDispls.at(txId);
        u64 srcDstOffset = repeat * templateRankSize * sliceSize +
            tempAlgParam.buffInfo.hcclBuffBaseOff + sliceOffset;

        if (sliceSize != 0) {
            txSrcSlices.push_back(
                DataSlice(tempAlgParam.buffInfo.hcclBuff.addr, srcDstOffset, sliceSize, sliceCount));
            txDstSlices.push_back(
                DataSlice(sendCclBuffAddr, srcDstOffset, sliceSize, sliceCount));
        }
    }
}

void BuildRxSlices(
    const AicpuNHRStepInfo &stepInfo,
    const TemplateDataParams &tempAlgParam,
    u32 repeat,
    u32 templateRankSize,
    void *recvCclBuffAddr,
    std::vector<DataSlice> &rxSrcSlices,
    std::vector<DataSlice> &rxDstSlices)
{
    for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
        u32 rxId = stepInfo.rxSliceIdxs.at(i);
        u64 sliceSize = tempAlgParam.allRankSliceSize.at(rxId);
        u64 sliceCount = tempAlgParam.allRankProcessedDataCount.at(rxId);
        u64 sliceOffset = tempAlgParam.allRankDispls.at(rxId);
        u64 srcDstOffset = repeat * templateRankSize * sliceSize +
            tempAlgParam.buffInfo.hcclBuffBaseOff + sliceOffset;

        if (sliceSize != 0) {
            rxSrcSlices.push_back(
                DataSlice(recvCclBuffAddr, srcDstOffset, sliceSize, sliceCount));
            rxDstSlices.push_back(
                DataSlice(tempAlgParam.buffInfo.hcclBuff.addr, srcDstOffset, sliceSize, sliceCount));
        }
    }
}

}  // anonymous namespace

// ========== 统一 BatchSR：覆盖只发 / 只收 / 同对端 / 不同对端四种场景 ==========

HcclResult BatchSRUnified(
    const AicpuNHRStepInfo &stepInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const TemplateDataParams &tempAlgParam,
    u32 repeat,
    u32 myRank,
    u32 templateRankSize)
{
#ifndef AICPU_COMPILE
    bool hasTx = stepInfo.txSliceIdxs.size() > 0;
    bool hasRx = stepInfo.rxSliceIdxs.size() > 0;

    // 空闲 rank，直接跳过
    if (!hasTx && !hasRx) {
        return HCCL_SUCCESS;
    }

    const ChannelInfo *txCh = hasTx ? &channels.at(stepInfo.toRank)[0] : nullptr;
    const ChannelInfo *rxCh = hasRx ? &channels.at(stepInfo.fromRank)[0] : nullptr;
    bool samePeer = hasTx && hasRx && (stepInfo.toRank == stepInfo.fromRank);

    // ====== Phase 1: Step 同步（按对端 rank 大小排序，避免死锁）======
    // 同对端只需同步 txCh；不同对端按 rank 从小到大，先 Record 再批量 Wait
    if (hasTx && hasRx && !samePeer && stepInfo.toRank > stepInfo.fromRank) {
        // 先 rx（fromRank 较小），后 tx（toRank 较大）
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(0, rxCh->handle, NOTIFY_IDX_STEP_SYNC)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(0, rxCh->handle, NOTIFY_IDX_STEP_SYNC, STEP_SYNC_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(0, txCh->handle, NOTIFY_IDX_STEP_SYNC)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(0, txCh->handle, NOTIFY_IDX_STEP_SYNC, STEP_SYNC_TIMEOUT)));
    } else {
        // 先 tx 后 rx（samePeer 时只同步 txCh）
        if (hasTx) {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(0, txCh->handle, NOTIFY_IDX_STEP_SYNC)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(0, txCh->handle, NOTIFY_IDX_STEP_SYNC, STEP_SYNC_TIMEOUT)));
        }
        if (hasRx && !samePeer) {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(0, rxCh->handle, NOTIFY_IDX_STEP_SYNC)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(0, rxCh->handle, NOTIFY_IDX_STEP_SYNC, STEP_SYNC_TIMEOUT)));
        }
    }

    // ====== Phase 2: 构造 slice 数据 ======
    void *sendCclBuffAddr = hasTx ? txCh->remoteCclMem.addr : nullptr;
    void *recvCclBuffAddr = hasRx ? rxCh->remoteCclMem.addr : nullptr;

    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    if (hasTx) {
        BuildTxSlices(stepInfo, tempAlgParam, repeat, templateRankSize,
            sendCclBuffAddr, txSrcSlices, txDstSlices);
    }
    if (hasRx) {
        BuildRxSlices(stepInfo, tempAlgParam, repeat, templateRankSize,
            recvCclBuffAddr, rxSrcSlices, rxDstSlices);
    }

    // ====== Phase 3: 双向 DATA 握手（发方 Record+Wait，收方 Record+Wait） + FIN_ACK + Fence ======

    if (hasTx && hasRx) {
        if (samePeer) {
            // ---- 同对端：SendRecv 模式 ----
            for (u32 i = 0; i < txSrcSlices.size(); i++) {
                void *dst = static_cast<s8 *>(txDstSlices[i].addr_) + txDstSlices[i].offset_;
                void *src = static_cast<s8 *>(txSrcSlices[i].addr_) + txSrcSlices[i].offset_;
                CHK_RET(static_cast<HcclResult>(HcommWriteNbiOnThread(
                    0, txCh->handle, dst, src, txSrcSlices[i].size_)));
            }
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(0, txCh->handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(0, rxCh->handle, NOTIFY_IDX_DATA_SIGNAL, STEP_SYNC_TIMEOUT)));
            // CHK_RET(static_cast<HcclResult>(
            //     HcommChannelNotifyRecordOnThread(0, txCh->handle, NOTIFY_IDX_FIN_ACK)));
            // CHK_RET(static_cast<HcclResult>(
            //     HcommChannelNotifyWaitOnThread(0, rxCh->handle, NOTIFY_IDX_FIN_ACK, STEP_SYNC_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, txCh->handle)));
            CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
        } else if (stepInfo.toRank < stepInfo.fromRank) {
            // ---- 不同对端，先 Send 后 Recv ----
            if (txSrcSlices.size() != 0) {
                for (u32 i = 0; i < txSrcSlices.size(); i++) {
                    void *dst = static_cast<s8 *>(txDstSlices[i].addr_) + txDstSlices[i].offset_;
                    void *src = static_cast<s8 *>(txSrcSlices[i].addr_) + txSrcSlices[i].offset_;
                    CHK_RET(static_cast<HcclResult>(HcommWriteNbiOnThread(
                        0, txCh->handle, dst, src, txSrcSlices[i].size_)));
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(0, txCh->handle, NOTIFY_IDX_DATA_SIGNAL)));
                // CHK_RET(static_cast<HcclResult>(
                //     HcommChannelNotifyRecordOnThread(0, txCh->handle, NOTIFY_IDX_FIN_ACK)));
                // CHK_RET(static_cast<HcclResult>(
                //     HcommChannelNotifyWaitOnThread(0, txCh->handle, NOTIFY_IDX_FIN_ACK, STEP_SYNC_TIMEOUT)));
            }
            if (rxSrcSlices.size() != 0) {
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(0, rxCh->handle, NOTIFY_IDX_DATA_SIGNAL, STEP_SYNC_TIMEOUT)));
                // CHK_RET(static_cast<HcclResult>(
                //     HcommChannelNotifyRecordOnThread(0, rxCh->handle, NOTIFY_IDX_FIN_ACK)));
                // CHK_RET(static_cast<HcclResult>(
                //     HcommChannelNotifyWaitOnThread(0, rxCh->handle, NOTIFY_IDX_FIN_ACK, STEP_SYNC_TIMEOUT)));
            }
            if (txSrcSlices.size() != 0) {
                CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, txCh->handle)));
            }
            if (rxSrcSlices.size() != 0) {
                CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, rxCh->handle)));
            }
            CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
        } else {
            // ---- 不同对端，先 Recv 后 Send ----
            if (rxSrcSlices.size() != 0) {
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(0, rxCh->handle, NOTIFY_IDX_DATA_SIGNAL, STEP_SYNC_TIMEOUT)));
                // CHK_RET(static_cast<HcclResult>(
                //     HcommChannelNotifyRecordOnThread(0, rxCh->handle, NOTIFY_IDX_FIN_ACK)));
                // CHK_RET(static_cast<HcclResult>(
                //     HcommChannelNotifyWaitOnThread(0, rxCh->handle, NOTIFY_IDX_FIN_ACK, STEP_SYNC_TIMEOUT)));
            }
            if (txSrcSlices.size() != 0) {
                for (u32 i = 0; i < txSrcSlices.size(); i++) {
                    void *dst = static_cast<s8 *>(txDstSlices[i].addr_) + txDstSlices[i].offset_;
                    void *src = static_cast<s8 *>(txSrcSlices[i].addr_) + txSrcSlices[i].offset_;
                    CHK_RET(static_cast<HcclResult>(HcommWriteNbiOnThread(
                        0, txCh->handle, dst, src, txSrcSlices[i].size_)));
                }
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(0, txCh->handle, NOTIFY_IDX_DATA_SIGNAL)));
                // CHK_RET(static_cast<HcclResult>(
                //     HcommChannelNotifyRecordOnThread(0, txCh->handle, NOTIFY_IDX_FIN_ACK)));
                // CHK_RET(static_cast<HcclResult>(
                //     HcommChannelNotifyWaitOnThread(0, txCh->handle, NOTIFY_IDX_FIN_ACK, STEP_SYNC_TIMEOUT)));
            }
            if (txSrcSlices.size() != 0) {
                CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, txCh->handle)));
            }
            if (rxSrcSlices.size() != 0) {
                CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, rxCh->handle)));
            }
            CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
        }
    } else if (hasTx) {
        // ---- 只发 ----
        if (txSrcSlices.size() != 0) {
            for (u32 i = 0; i < txSrcSlices.size(); i++) {
                void *dst = static_cast<s8 *>(txDstSlices[i].addr_) + txDstSlices[i].offset_;
                void *src = static_cast<s8 *>(txSrcSlices[i].addr_) + txSrcSlices[i].offset_;
                CHK_RET(static_cast<HcclResult>(HcommWriteNbiOnThread(
                    0, txCh->handle, dst, src, txSrcSlices[i].size_)));
            }
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(0, txCh->handle, NOTIFY_IDX_DATA_SIGNAL)));
            // CHK_RET(static_cast<HcclResult>(
            //     HcommChannelNotifyRecordOnThread(0, txCh->handle, NOTIFY_IDX_FIN_ACK)));
            // CHK_RET(static_cast<HcclResult>(
            //     HcommChannelNotifyWaitOnThread(0, txCh->handle, NOTIFY_IDX_FIN_ACK, STEP_SYNC_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, txCh->handle)));
            CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
        }
    } else {
        // ---- 只收 ----
        if (rxSrcSlices.size() != 0) {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(0, rxCh->handle, NOTIFY_IDX_DATA_SIGNAL, STEP_SYNC_TIMEOUT)));
            // CHK_RET(static_cast<HcclResult>(
            //     HcommChannelNotifyRecordOnThread(0, rxCh->handle, NOTIFY_IDX_FIN_ACK)));
            // CHK_RET(static_cast<HcclResult>(
            //     HcommChannelNotifyWaitOnThread(0, rxCh->handle, NOTIFY_IDX_FIN_ACK, STEP_SYNC_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, rxCh->handle)));
            CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
        }
    }
#endif
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
