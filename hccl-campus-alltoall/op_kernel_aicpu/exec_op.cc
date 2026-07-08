/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exec_op.h"

namespace ops_hccl {
// 单次通信允许的最大字节数：256 MB
constexpr uint64_t MAX_DATA_SIZE = 256ULL * 1024 * 1024;

// AllToAll 全连接直发算法（串行）：
//   - 自块（j == myRank）：本地拷贝，不走 channel
//   - 其余 peer（j != myRank）：双向同时收发，本地 ccl buffer 单槽复用，
//     notify 节拍沿用 broadcast ExecDoubling（初始就绪屏障 + 每段 Write/Record/Wait/LocalCopy/Record/Wait）
//   - 语义：param.count 为每块元素数（= sendCount），sendbuf/recvbuf 第 j 块偏移 = j * count * dataTypeSize
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    // 无数据可交换
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    uint32_t dataTypeSize = SIZE_TABLE.at(param.dataType);
    uint64_t perBlock = param.count; // 每块元素数
    ThreadHandle thread = resCtx.aicpuThread;
    void *cclBuffAddr = resCtx.localBuffer.addr;
    uint64_t maxDataSizePerLoop = std::min(MAX_DATA_SIZE, resCtx.localBuffer.size);
    uint64_t maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
    CHK_PRT_RET(maxDataCountPerLoop == 0,
        HCCL_ERROR("ExecOp: ccl buffer too small for one element, maxDataSizePerLoop[%llu] dataTypeSize[%u]",
            static_cast<unsigned long long>(maxDataSizePerLoop), dataTypeSize),
        HCCL_E_INTERNAL);

    // 构建 remoteRank -> ChannelInfo 映射
    std::unordered_map<uint32_t, ChannelInfo> channelMap;
    for (const auto &ch : resCtx.channels) {
        channelMap.emplace(ch.remoteRank, ch);
    }

    // 自块：本地拷贝 sendbuf 第 myRank 块 -> recvbuf 第 myRank 块（不走 channel）
    uint64_t selfOffset = static_cast<uint64_t>(param.myRank) * perBlock * dataTypeSize;
    void *selfSrc = static_cast<void *>(static_cast<uint8_t *>(param.inputPtr) + selfOffset);
    void *selfDst = static_cast<void *>(static_cast<uint8_t *>(param.outputPtr) + selfOffset);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, selfDst, selfSrc, perBlock * dataTypeSize)));

    // 其余 peer 串行
    for (uint32_t j = 0; j < param.rankSize; j++) {
        if (j == param.myRank) {
            continue;
        }
        auto it = channelMap.find(j);
        CHK_PRT_RET(it == channelMap.end(), HCCL_ERROR("ExecOp: channel to rank[%u] not found", j), HCCL_E_INTERNAL);
        const ChannelInfo &channel = it->second;

        uint8_t *sendBase = static_cast<uint8_t *>(param.inputPtr) + j * perBlock * dataTypeSize;
        uint8_t *recvBase = static_cast<uint8_t *>(param.outputPtr) + j * perBlock * dataTypeSize;

        // 初始就绪屏障：双方互报到达交换点（对端 ccl buffer 已就位）
        CHK_RET(
            static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

        // 按 maxDataCountPerLoop 分段，本地 ccl buffer 单槽复用
        uint64_t loopCount = perBlock / maxDataCountPerLoop + (perBlock % maxDataCountPerLoop != 0 ? 1 : 0);
        for (uint64_t k = 0; k < loopCount; k++) {
            uint64_t subCount = std::min(maxDataCountPerLoop, perBlock - k * maxDataCountPerLoop);
            uint64_t subBytes = subCount * dataTypeSize;
            void *sendAddr = static_cast<void *>(sendBase + k * maxDataSizePerLoop);
            void *recvAddr = static_cast<void *>(recvBase + k * maxDataSizePerLoop);

            // 写本地 sendbuf 第 j 块(本段) -> 对端 ccl buffer（偏移 0，单槽复用）
            CHK_RET(static_cast<HcclResult>(
                HcommWriteOnThread(thread, channel.handle, channel.remoteCclMem.addr, sendAddr, subBytes)));
            // 通知对端已写完本段；等对端把本段写到我的 ccl buffer
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            // 从本地 ccl buffer 拷贝到 recvbuf 第 j 块(本段)
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, recvAddr, cclBuffAddr, subBytes)));
            // 通知对端我已读完本段(可向我 ccl 写下一段)；等对端读完(我可向下一段)
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
