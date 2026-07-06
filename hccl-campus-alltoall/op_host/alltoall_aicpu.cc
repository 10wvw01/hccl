/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "alltoall_aicpu.h"
#include "launch_aicpu_kernel.h"

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

namespace {
// 获取本 rank 到 dstRank 的 layer-0 直连链路并构造 channel 描述符。
// 仅使用 netLayer=0（框内），不回退 layer-1；无链路则视为拓扑异常。
HcclResult AcquireDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc *desc)
{
    constexpr uint32_t netLayer = 0;
    uint32_t listSize = 0;
    CommLink *linkList = nullptr;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize));
    CHK_PRT_RET(listSize == 0,
        HCCL_ERROR("AcquireDesc: no layer-0 link between rank[%u] and rank[%u]", srcRank, dstRank),
        HCCL_E_INTERNAL);

    CHK_RET(HcclChannelDescInit(desc, 1));
    CommLink link = linkList[0];
    desc->remoteRank = dstRank;
    desc->notifyNum = CHANNEL_NOTIFY_NUM;
    desc->channelProtocol = link.linkAttr.linkProtocol;
    desc->localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc->localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc->localEndpoint.loc = link.srcEndpointDesc.loc;
    desc->remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc->remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc->remoteEndpoint.loc = link.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

// 全连接直发：对框内其余每个 rank 各建一条 channel（共 rankSize-1 条），仅 layer-0。
HcclResult HcclGetChannelAICPU(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    uint32_t channelNum = param.rankSize - 1;
    std::vector<HcclChannelDesc> desc(channelNum);
    std::vector<ChannelHandle> channels(channelNum);
    uint32_t idx = 0;
    for (uint32_t j = 0; j < param.rankSize; j++) {
        if (j == param.myRank) {
            continue;
        }
        CHK_RET(AcquireDesc(comm, param.myRank, j, &desc[idx]));
        idx++;
    }
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, desc.data(), channelNum, channels.data()));
    for (uint32_t i = 0; i < channelNum; i++) {
        ChannelInfo channel;
        channel.remoteRank = desc[i].remoteRank;
        channel.handle = channels[i];
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        void *cclBuf = nullptr;
        uint64_t cclBufSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channels[i], &cclBuf, &cclBufSize));
        channel.remoteCclMem = CommBuffer{cclBuf, cclBufSize};
        resCtxHost.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAllocAlgResourceAICPU(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    // 从通信域获取 CCL buffer（本地暂存，对端写入此 buffer 后本地 LocalCopy 到 output）
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
    if (param.rankSize > 1) {
        CHK_RET(HcclGetChannelAICPU(comm, param, resCtxHost));
    }
    return HCCL_SUCCESS;
}
}

namespace ops_aicpu {
HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    /*
     * 适用场景：
     *   - 单框 8 卡全连接拓扑（框内任意两 rank 间存在 layer-0 直连链路）
     *   - 均匀 alltoall：sendCount == recvCount 且 sendType == recvType
     *   - 仅 layer-0（框内）通信，不处理跨框（layer-1）与非均匀（sendCount != recvCount 或
     *     sendType != recvType）情形
     * 语义：sendCount/recvCount 为「发给/来自每个 rank 的元素数」（每块大小），
     *       sendbuf/recvbuf 总量 = rankSize × sendCount；rank i 的 sendbuf 第 j 块发给 rank j，
     *       recvbuf 第 j 块存放来自 rank j 的数据。
     */
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", "hccl_custom_alltoall");
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("Failed to fill param.tag, ret[%d]", ret), HCCL_E_INTERNAL);
    CHK_RET(HcclGetCommName(comm, param.commName));
    param.inputPtr = const_cast<void *>(sendBuf);
    param.outputPtr = const_cast<void *>(recvBuf);
    param.count = sendCount;
    param.dataType = sendType;
    param.opType = HcclCMDType::HCCL_CMD_ALLTOALL;

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine aicpuEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuEngine = CommEngine::COMM_ENGINE_CPU_TS;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 thread，并申请 Notify；同时导出为 AICPU 上可用的 thread
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = sizeof(AlgResourceCtx);
    if (HcclEngineCtxGet(comm, param.tag, aicpuEngine, &ctx, &size) == HCCL_SUCCESS) {
        // AICPU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = static_cast<AlgResourceCtx *>(ctx);
        param.ctxSize = size;

        // CPU 资源已经存在，复用资源
 	    void *hostCtx = nullptr;
 	    uint64_t hostCtxSize = 0;
 	    CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuEngine, &hostCtx, &hostCtxSize));
 	    auto *aicpuThreadCache = static_cast<ThreadHandle *>(hostCtx);
 	    CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThreadCache, cpuEngine, &param.aicpuThreadOnCpu));
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 创建一个 AICPU_TS 类型的 thread，并导出为 CPU 上可用的 thread
        CHK_RET(HcclThreadAcquire(comm, aicpuEngine, 1, 1, &resCtxHost.aicpuThread));
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuEngine, &param.aicpuThreadOnCpu));

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        HcclDfxOpInfo dfxInfo;
        CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast<void *>(&dfxInfo)));

        // STEP 2.2: 根据通信算法申请通信资源（全连接直发：rankSize-1 条 layer-0 channel + 本地 CCL buffer）
        CHK_RET(HcclAllocAlgResourceAICPU(comm, param, resCtxHost));
        
        // ==============================================
        // STEP 2.3: 在 AICPU_TS 引擎上申请ctx，并将resCtxHost拷贝至 device侧
        // ==============================================
        // 序列化 resCtxHost 并拷贝到 device 侧
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuEngine, param.tag, seq.data(), seqSize, 0));
        // 将aicpu主线程句柄缓存到host内存的context上，供后续复用
        void *hostCtx = nullptr;
 	    uint64_t hostCtxSize = sizeof(ThreadHandle);
 	    CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuEngine, hostCtxSize, &hostCtx));
        auto *aicpuThreadCache = static_cast<ThreadHandle*>(hostCtx);
        *aicpuThreadCache = resCtxHost.aicpuThread;
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl_ag::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
}
