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
#include "log.h"

#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {

/**
 * 创建 AICPU 运行时资源。
 * 参照原 src 中 HcclAllocAlgResourceAICPU 的实现：
 *   1. 从通信域获取 CCL buffer 作为跨 Rank 缓存（cclMem）；
 *   2. 将资源请求中的标量字段（notifyNumOnMainThread/slaveThreadNum/notifyNumPerThread）回填到 resCtx_；
 *   3. 根据 slaveThreadNum 创建主线程与从线程（HcommThreadCreate）；
 *   4. 遍历每层级 channels 创建通信通道（HcommChannelCreate）。
 * resCtx_ 作为成员保存，供后续 LaunchKernel 直接使用。
 *
 * 注：当前 CreateRes 接口仅传入 AlgResourceRequest，comm/param/resPack 的获取方式待重构确定后补全。
 */
HcclResult AiCpuLauncher::CreateRes(AlgResourceRequest &res)
{
    // 1. 回填标量资源需求字段
    resCtx_.notifyNumOnMainThread = res.notifyNumOnMainThread;
    resCtx_.slaveThreadNum = res.slaveThreadNum;
    resCtx_.notifyNumPerThread = res.notifyNumPerThread;

    // 2. 获取 CCL buffer 作为 scratch buffer（跨 Rank 缓存）
    // Todo: comm 需通过通信域句柄获取，待 CreateRes 接口或 Launcher 构造补全 comm 传递
    void *cclBufferAddr = nullptr;
    u64 cclBufferSize = 0;
    // CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtx_.cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};

    // 3. 创建 thread：主线程 + slaveThreadNum 个从线程
    // 线程布局参见 OpsExecutor::CalcRes 注释：
    //   thread[0]                              = main thread
    //   thread[1]                              = intra main
    //   threads[2..maxIntra+1]                 = intra slaves
    //   thread[maxIntra+2]                     = inter main
    //   threads[maxIntra+3..maxIntra+maxInter] = inter slaves
    // Todo: 调用 HcommThreadCreate 创建主线程与从线程，回填到 resCtx_.threads
    resCtx_.threads.resize(res.slaveThreadNum + 1);

    // 4. 创建 channel：遍历每层级的 HcclChannelDesc 创建通信通道
    for (size_t level = 0; level < res.channels.size(); ++level) {
        std::vector<ChannelInfo> levelChannels;
        levelChannels.reserve(res.channels[level].size());
        for (size_t idx = 0; idx < res.channels[level].size(); ++idx) {
            HcclChannelDesc &channelDesc = res.channels[level][idx];
            ChannelInfo channelInfo;
            // Todo: 调用 HcommChannelCreate(comm, channelDesc, channelInfo) 创建实际通信通道
            CHK_RET(HcommChannelCreate(channelDesc, channelInfo));
            levelChannels.emplace_back(channelInfo);
        }
        resCtx_.channels.emplace_back(std::move(levelChannels));
    }

    HCCL_INFO("[AiCpuLauncher][CreateRes] success, slaveThreadNum[%u], notifyNumOnMainThread[%u], channelLevel[%zu]",
              resCtx_.slaveThreadNum, resCtx_.notifyNumOnMainThread, resCtx_.channels.size());
    return HCCL_SUCCESS;
}

/**
 * 下发 AICPU kernel 到设备侧执行。
 * 工作流程：
 *   1. 加载 AICPU kernel 二进制文件（LoadAICPUKernel），确保 kernel 已加载到设备；
 *   2. 调用 HcclLaunchAicpuKernel 完成环境准备、算法编排与 profiling 上报：
 *      a. 获取通信域句柄（HcommAcquireComm）；
 *      b. 根据 opType 还原变长数据（如 AllGatherV 的 counts/displs）；
 *      c. 设置 batch mode，注册 DFX op 信息与 profiling；
 *      d. 主 thread 等待 Host stream 的 notify 通知；
 *      e. 调用 executor.Orchestrate 驱动算法编排；
 *      f. 上报 profiling，通知 Host stream 完成，结束 batch mode；
 *   3. 释放通信域句柄（HcommReleaseComm）。
 * resCtx_ 由 CreateRes 回填，直接传递给 HcclLaunchAicpuKernel，无需重新反序列化。
 */
HcclResult AiCpuLauncher::LaunchKernel(const OpParam &param, OpsExecutor &executor)
{
    HCCL_INFO("[AiCpuLauncher][LaunchKernel] start, commName[%s], tag[%s], algTag[%s]",
              param.commName, param.tag, param.algTag);

    // 步骤1：加载 AICPU kernel 二进制
    CHK_RET(LoadAICPUKernel());

    // 步骤2：通过 HcclLaunchAicpuKernel 入口完成环境准备、算法编排与 profiling 上报
    // 传入 resCtx_（CreateRes 已回填），内部直接使用，无需从 param->resCtx 反序列化
    CHK_RET(HcclLaunchAicpuKernel(param, executor, resCtx_));

    HCCL_INFO("[AiCpuLauncher][LaunchKernel] end, tag[%s], algTag[%s], commName[%s]",
              param.tag, param.algTag, param.commName);
    return HCCL_SUCCESS;
}


// ───────────── 辅助: Rank → Channel 映射 ─────────────
static const ChannelInfo* LookupChannel(const std::map<u32, std::vector<ChannelInfo>> &channels, u32 rank)
{
    auto it = channels.find(rank);
    if (it == channels.end() || it->second.empty()) {
        return nullptr;
    }
    return &it->second[0];
}

// ───────────── 主 Send ─────────────
HcclResult AiCpuLauncher::Send(const TransferContext &ctx) {
    // Step 1: 方向判断
    //   enableRemoteMemAccess && buffType==OUTPUT → READ (从远端拉取到本地 output)
    //   其他场景 → WRITE (推送数据到远端)
    TransferDirection direction =
        (ctx.enableRemoteMemAccess && ctx.buffType == BufferType::OUTPUT)
            ? TransferDirection::READ : TransferDirection::WRITE;

    // Step 2: 获取线程 (调用者控制线程分配, 置于 threads[0])
    if (ctx.templateRes.threads.empty()) {
        HCCL_ERROR("[AiCpuLauncher][Send] threads is empty");
        return HCCL_E_INTERNAL;
    }
    const ThreadHandle &thread = ctx.templateRes.threads[0];

    // Step 3: Rank → Channel 映射
    u32 dstRank = ctx.txRxSlicesList.dstRankId_;
    u32 srcRank = ctx.txRxSlicesList.srcRankId_;
    const auto &channels = ctx.templateRes.channels;

    const ChannelInfo *txCh = LookupChannel(channels, dstRank);  // 发送目标 channel
    const ChannelInfo *rxCh = LookupChannel(channels, srcRank);  // 接收来源 channel

    // Step 4: 判断双向 / 单向
    bool hasTx = !ctx.txRxSlicesList.txSlicesList_.srcSlices_.empty();
    bool hasRx = !ctx.txRxSlicesList.rxSlicesList_.srcSlices_.empty();
    bool isBidirectional = hasTx && hasRx && txCh != nullptr && rxCh != nullptr;

    bool hasReduce = (ctx.reduceOp != HCCL_REDUCE_RESERVED);

    // Step 5: 路由到底层 wrapper 函数
    if (isBidirectional) {
        // 双向: 构造 SendRecvInfo, 使用 SendRecv* 系列
        TxRxChannels channelPair(*txCh, *rxCh);
        TxRxSlicesList slicesList(ctx.txRxSlicesList.txSlicesList_,
                                  ctx.txRxSlicesList.rxSlicesList_);

        if (direction == TransferDirection::WRITE) {
            if (hasReduce) {
                SendRecvReduceInfo reduceInfo(channelPair, slicesList, ctx.dataType, ctx.reduceOp);
                return SendRecvWriteReduce(reduceInfo, thread);
            }
            SendRecvInfo info(channelPair, slicesList, ctx.dataType);
            return SendRecvWrite(info, thread);
        } else {
            if (hasReduce) {
                SendRecvReduceInfo reduceInfo(channelPair, slicesList, ctx.dataType, ctx.reduceOp);
                return SendRecvReadReduce(reduceInfo, thread);
            }
            SendRecvInfo info(channelPair, slicesList, ctx.dataType);
            return SendRecvRead(info, thread);
        }
    } else {
        // 单向: 构造 DataInfo, 使用 Send*/Recv* 系列
        u32 rank = (direction == TransferDirection::WRITE) ? dstRank : srcRank;
        const ChannelInfo *ch = LookupChannel(channels, rank);
        if (ch == nullptr) {
            HCCL_ERROR("[AiCpuLauncher][Send] channel not found for rank[%u]", rank);
            return HCCL_E_INTERNAL;
        }
        const SlicesList &slices = (direction == TransferDirection::WRITE)
            ? ctx.txRxSlicesList.txSlicesList_
            : ctx.txRxSlicesList.rxSlicesList_;

        if (direction == TransferDirection::WRITE) {
            if (hasReduce) {
                DataReduceInfo reduceInfo(*ch, slices, ctx.dataType, ctx.reduceOp);
                return SendWriteReduce(reduceInfo, thread);
            }
            DataInfo info(*ch, slices, ctx.dataType);
            return SendWrite(info, thread);
        } else {
            if (hasReduce) {
                DataReduceInfo reduceInfo(*ch, slices, ctx.dataType, ctx.reduceOp);
                return SendReadReduce(reduceInfo, thread);
            }
            DataInfo info(*ch, slices, ctx.dataType);
            return SendRead(info, thread);
        }
    }
}

}  // namespace ops_hccl
