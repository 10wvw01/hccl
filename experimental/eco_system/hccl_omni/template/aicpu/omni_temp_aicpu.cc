/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/omni_temp_aicpu.h"
#include "log.h"
#include "param_check.h"
#include "utils.h"

namespace ops_hccl {

// Helper: compute per-slice process count (in elements) for variable-length AlltoAllV
// When sendCounts/recvCounts are available, returns counts[sliceIdx]; else falls back to sliceSize/dtypeSize
// Note: For multi-loop scenarios, counts[] represents the total count per slice, not the remaining count.
// The single-loop case (data fits in CCL buffer) works correctly. Multi-loop unequal AlltoAllV requires
// additional tracking of processed counts per slice (TODO).
static u64 GetSliceProcessCount(
    omni::BufferTypeTmp sliceType, const TemplateDataParams &tempAlgParams, uint64_t sliceIdx, uint64_t dtypeSize)
{
    if (sliceType == omni::INPUT && !tempAlgParams.sendCounts.empty() && sliceIdx < tempAlgParams.sendCounts.size()) {
        return tempAlgParams.sendCounts[sliceIdx];
    }
    if (sliceType == omni::OUTPUT && !tempAlgParams.recvCounts.empty() && sliceIdx < tempAlgParams.recvCounts.size()) {
        return tempAlgParams.recvCounts[sliceIdx];
    }
    // HCCL_BUFFER or fallback: uniform sliceSize
    return (dtypeSize > 0) ? tempAlgParams.sliceSize / dtypeSize : 0;
}

// Helper function to get buffer address based on slice type
// When sdispls/rdispls are available (unequal AlltoAllV), uses displs-based addressing;
// otherwise falls back to fixed-stride model (equal partition).
static void *GetBufferAddrBySliceType(
    omni::BufferTypeTmp sliceType, const TemplateDataParams &tempAlgParams, uint64_t sliceIdx, uint64_t sliceNum)
{
    u64 dtypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    switch (sliceType) {
        case omni::INPUT: {
            void *addr;
            if (!tempAlgParams.sdispls.empty() && sliceIdx < tempAlgParams.sdispls.size()) {
                // Unequal AlltoAllV: sdispls[sliceIdx] is in elements, convert to bytes + inBuffBaseOff for loop
                // advancement
                u64 elemOff = tempAlgParams.sdispls[sliceIdx];
                addr = (void *)((char *)tempAlgParams.buffInfo.inputPtr + elemOff * dtypeSize
                                + tempAlgParams.buffInfo.inBuffBaseOff);
                HCCL_INFO("YHB-CHECKER: [GetBufferAddrBySliceType] input-displs(%p = %p + %llu*%llu + %llu), "
                          "sdispls[%llu]=%llu",
                    addr, (void *)tempAlgParams.buffInfo.inputPtr, elemOff, dtypeSize,
                    tempAlgParams.buffInfo.inBuffBaseOff, sliceIdx, elemOff);
            } else {
                // Equal partition: fixed stride
                uint64_t sliceSize = tempAlgParams.buffInfo.inputSize;
                addr = (void *)((char *)tempAlgParams.buffInfo.inputPtr + tempAlgParams.buffInfo.inBuffBaseOff
                                + sliceIdx * sliceSize);
                HCCL_INFO("YHB-CHECKER: [GetBufferAddrBySliceType] input-stride(%p = %p + %llu + %llu * %llu)", addr,
                    (void *)tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, sliceIdx, sliceSize);
            }
            return addr;
        }
        case omni::OUTPUT: {
            void *addr;
            if (!tempAlgParams.rdispls.empty() && sliceIdx < tempAlgParams.rdispls.size()) {
                // Unequal AlltoAllV: rdispls[sliceIdx] is in elements, convert to bytes + outBuffBaseOff for loop
                // advancement
                u64 elemOff = tempAlgParams.rdispls[sliceIdx];
                addr = (void *)((char *)tempAlgParams.buffInfo.outputPtr + elemOff * dtypeSize
                                + tempAlgParams.buffInfo.outBuffBaseOff);
                HCCL_INFO("YHB-CHECKER: [GetBufferAddrBySliceType] output-displs(%p = %p + %llu*%llu + %llu), "
                          "rdispls[%llu]=%llu",
                    addr, (void *)tempAlgParams.buffInfo.outputPtr, elemOff, dtypeSize,
                    tempAlgParams.buffInfo.outBuffBaseOff, sliceIdx, elemOff);
            } else {
                // Equal partition: fixed stride
                uint64_t sliceSize = tempAlgParams.buffInfo.outputSize;
                addr = (void *)((char *)tempAlgParams.buffInfo.outputPtr + tempAlgParams.buffInfo.outBuffBaseOff
                                + sliceIdx * sliceSize);
                HCCL_INFO("YHB-CHECKER: [GetBufferAddrBySliceType] output-stride(%p = %p + %llu + %llu * %llu)", addr,
                    (void *)tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, sliceIdx,
                    sliceSize);
            }
            return addr;
        }
        case omni::HCCL_BUFFER: {
            // HCCL_BUFFER 始终用 fixed stride（slot 等分），不受 displs 影响
            uint64_t sliceSize = tempAlgParams.buffInfo.hcclBuffSize;
            void *addr = (void *)((char *)tempAlgParams.buffInfo.hcclBuff.addr + tempAlgParams.buffInfo.hcclBuffBaseOff
                                  + sliceIdx * sliceSize);
            HCCL_INFO("YHB-CHECKER: [GetBufferAddrBySliceType] hccl(%p = %p + %llu + %llu * %llu)", addr,
                (void *)tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuffBaseOff, sliceIdx,
                sliceSize);
            return addr;
        }
        default:
            return nullptr;
    }
}

// Helper function to get remote address based on slice type
// When sdispls/rdispls are available (unequal AlltoAllV), uses displs-based addressing;
// otherwise falls back to fixed-stride model (equal partition).
static void *GetRemoteAddrBySliceType(omni::BufferTypeTmp sliceType, const TemplateDataParams &tempAlgParams,
    const ChannelInfo &channel, uint64_t sliceIdx, uint64_t sliceNum)
{
    u64 dtypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    switch (sliceType) {
        case omni::INPUT: {
            void *addr;
            if (!tempAlgParams.sdispls.empty() && sliceIdx < tempAlgParams.sdispls.size()) {
                u64 elemOff = tempAlgParams.sdispls[sliceIdx];
                addr = (void *)((char *)channel.remoteInput.addr + elemOff * dtypeSize
                                + tempAlgParams.buffInfo.inBuffBaseOff);
                HCCL_INFO("YHB-CHECKER: [GetRemoteAddrBySliceType] input-displs(%p = %p + %llu*%llu + %llu)", addr,
                    (void *)channel.remoteInput.addr, elemOff, dtypeSize, tempAlgParams.buffInfo.inBuffBaseOff);
            } else {
                uint64_t sliceSize = tempAlgParams.buffInfo.inputSize;
                addr = (void *)((char *)channel.remoteInput.addr + tempAlgParams.buffInfo.inBuffBaseOff
                                + sliceIdx * sliceSize);
                HCCL_INFO("YHB-CHECKER: [GetRemoteAddrBySliceType] input-stride(%p = %p + %llu + %llu * %llu)", addr,
                    (void *)channel.remoteInput.addr, tempAlgParams.buffInfo.inBuffBaseOff, sliceIdx, sliceSize);
            }
            return addr;
        }
        case omni::OUTPUT: {
            void *addr;
            if (!tempAlgParams.rdispls.empty() && sliceIdx < tempAlgParams.rdispls.size()) {
                u64 elemOff = tempAlgParams.rdispls[sliceIdx];
                addr = (void *)((char *)channel.remoteOutput.addr + elemOff * dtypeSize
                                + tempAlgParams.buffInfo.outBuffBaseOff);
                HCCL_INFO("YHB-CHECKER: [GetRemoteAddrBySliceType] output-displs(%p = %p + %llu*%llu + %llu)", addr,
                    (void *)channel.remoteOutput.addr, elemOff, dtypeSize, tempAlgParams.buffInfo.outBuffBaseOff);
            } else {
                uint64_t sliceSize = tempAlgParams.buffInfo.outputSize;
                addr = (void *)((char *)channel.remoteOutput.addr + tempAlgParams.buffInfo.outBuffBaseOff
                                + sliceIdx * sliceSize);
                HCCL_INFO("YHB-CHECKER: [GetRemoteAddrBySliceType] output-stride(%p = %p + %llu + %llu * %llu)", addr,
                    (void *)channel.remoteOutput.addr, tempAlgParams.buffInfo.outBuffBaseOff, sliceIdx, sliceSize);
            }
            return addr;
        }
        case omni::HCCL_BUFFER: {
            // HCCL_BUFFER 始终用 fixed stride
            uint64_t sliceSize = tempAlgParams.buffInfo.hcclBuffSize;
            void *addr = (void *)((char *)channel.remoteCclMem.addr + tempAlgParams.buffInfo.hcclBuffBaseOff
                                  + sliceIdx * sliceSize);
            HCCL_INFO("YHB-CHECKER: [GetRemoteAddrBySliceType] hccl(%p = %p + %llu + %llu * %llu)", addr,
                (void *)channel.remoteCclMem.addr, tempAlgParams.buffInfo.hcclBuffBaseOff, sliceIdx, sliceSize);
            return addr;
        }
        default:
            return nullptr;
    }
}

OmniTempAicpu::OmniTempAicpu(const OpParam &param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

OmniTempAicpu::~OmniTempAicpu()
{
}

HcclResult OmniTempAicpu::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest, const omni::XmlInfo &xmlInfo)
{
    HCCL_INFO("[OmniTempAicpu][CalcRes] Start to calc resource.");

    resourceRequest.slaveThreadNum = topoInfo->xmlInfo.resInfo.slaveThreadNum;
    resourceRequest.notifyNumOnMainThread = topoInfo->xmlInfo.resInfo.notifyNumOnMainThread;
    resourceRequest.notifyNumPerThread.assign(topoInfo->xmlInfo.resInfo.notifyNumPerThread, 1);

    // 计算通道资源
    std::vector<HcclChannelDesc> level0Channels;
    // CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    // CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level0Channels));
    CHK_RET(CalcChannelRequestOmni(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);

    HCCL_INFO("[OmniTempAicpu][CalcRes] Calc resource success.");
    return HCCL_SUCCESS;
}

HcclResult OmniTempAicpu::CalcChannelRequestOmni(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, const std::vector<std::vector<u32>> &subcommInfo,
    std::vector<HcclChannelDesc> &channels)
{
#ifndef AICPU_COMPILE
    channels.clear();
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    CHK_PRT_RET((it == subcommInfo[COMM_LEVEL0].end()),
        HCCL_ERROR("[AivTempOmni][CalcChannelRequestOmni] Rank [%d] is not in commInfo.", topoInfo->userRank),
        HcclResult::HCCL_E_PARA);

    const u32 myRank = topoInfo->userRank;

    for (const auto &channelInfo : topoInfo->xmlInfo.resInfo.vecChannelInfo) {
        const u64 remoteRank = channelInfo.remoteRank;

        uint32_t *netLayers = nullptr;
        uint32_t netLayerNum = 0;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);

        HCCL_DEBUG("CalcChannelRequestOmni netLayerNum %u", netLayerNum);

        for (auto netLayer : netLayersVector) {
            CommLink *linkList = nullptr;
            u32 listSize = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, static_cast<u32>(remoteRank), &linkList, &listSize));

            for (u32 idx = 0; idx < listSize; idx++) {
                HCCL_DEBUG(
                    "CalcChannelRequestOmni HcclRankGraphGetLinks myrank %u to remoteRank %u, %u, %u, linkProtocol %u",
                    myRank, remoteRank, netLayer, listSize, linkList[idx].linkAttr.linkProtocol);

                if (linkList[idx].linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                    continue;
                }

                HcclChannelDesc channelDesc;
                HcclChannelDescInit(&channelDesc, 1);
                channelDesc.remoteRank = static_cast<u32>(remoteRank);
                channelDesc.localEndpoint.protocol = linkList[idx].srcEndpointDesc.protocol;
                channelDesc.localEndpoint.commAddr = linkList[idx].srcEndpointDesc.commAddr;
                channelDesc.localEndpoint.loc = linkList[idx].srcEndpointDesc.loc;
                channelDesc.remoteEndpoint.protocol = linkList[idx].dstEndpointDesc.protocol;
                channelDesc.remoteEndpoint.commAddr = linkList[idx].dstEndpointDesc.commAddr;
                channelDesc.remoteEndpoint.loc = linkList[idx].dstEndpointDesc.loc;
                channelDesc.channelProtocol = linkList[idx].linkAttr.linkProtocol;
                channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
                for (u32 i = 0; i < 8; ++i) {
                    channels.push_back(channelDesc);
                }

                HCCL_INFO("CalcChannelRequestOmni Add channel request between %zu and %zu, netLayerIdx %u, "
                          "linkListIdx %u, protocol %zu",
                    myRank, channelDesc.remoteRank, netLayer, idx, channelDesc.remoteEndpoint.protocol);

                break;
            }
        }
    }
#endif
    return HCCL_SUCCESS;
}

u64 OmniTempAicpu::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // usrIn和cclBuffer大小相同
    return 1;
}

HcclResult OmniTempAicpu::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
    TemplateResource &templateResource, const omni::XmlInfo &xmlInfo)
{
    HCCL_INFO("[OmniTempAicpu] Run Start");

    // 检查XML信息是否已传递
    if (xmlInfo.vecNormalInstruction.empty()) {
        HCCL_WARNING("[OmniTempAicpu] XML information is empty, OMNI operations may not execute correctly");
    } else {
        HCCL_INFO("[OmniTempAicpu] XML information contains %lu signals", xmlInfo.vecNormalInstruction.size());
    }

    // 执行OMNI算法
    CHK_RET(RunOmni(templateResource.channels, templateResource.threads, tempAlgParams, xmlInfo));

    HCCL_INFO("[OmniTempAicpu] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult OmniTempAicpu::RunOmni(const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams, const omni::XmlInfo &xmlInfo)
{
    HCCL_INFO("[OmniTempAicpu][RunOmni] Start processing OMNI signals");

    // 遍历XML中的所有信号信息
    for (const auto &signalInfo : xmlInfo.vecNormalInstruction) {
        HcclResult ret = HCCL_SUCCESS;
        HCCL_INFO("YHB-CHECKER: =====%s start=====", omni::OpTypeToString(signalInfo.opType).c_str());

        switch (signalInfo.opType) {
            case omni::OP_PRE_SYNC_INTER_THREADS:
                ret = HandlePreSyncInterThreads(signalInfo, threads);
                break;
            case omni::OP_POST_SYNC_INTER_THREADS:
                ret = HandlePostSyncInterThreads(signalInfo, threads);
                break;
            case omni::OP_LOCAL_COPY:
            case omni::OP_LOCAL_REDUCE:
                ret = HandleLocalCopy(signalInfo, threads, tempAlgParams);
                break;
            case omni::OP_SEND_RECV_WRITE:
            case omni::OP_SEND_RECV_WRITE_REDUCE:
                ret = HandleSendRecvWrite(signalInfo, channels, threads, tempAlgParams);
                break;
            case omni::OP_SEND_WRITE:
            case omni::OP_SEND_WRITE_REDUCE:
                ret = HandleSendWrite(signalInfo, channels, threads, tempAlgParams);
                break;
            case omni::OP_RECV_WRITE:
            case omni::OP_RECV_WRITE_REDUCE:
                ret = HandleRecvWrite(signalInfo, channels, threads, tempAlgParams);
                break;
            case omni::OP_SEND_RECV_READ:
            case omni::OP_SEND_RECV_READ_REDUCE:
                ret = HandleSendRecvRead(signalInfo, channels, threads, tempAlgParams);
                break;
            case omni::OP_SEND_READ:
            case omni::OP_SEND_READ_REDUCE:
                ret = HandleSendRead(signalInfo, channels, threads, tempAlgParams);
                break;
            case omni::OP_RECV_READ:
            case omni::OP_RECV_READ_REDUCE:
                ret = HandleRecvRead(signalInfo, channels, threads, tempAlgParams);
                break;
            case omni::OP_GROUP_BROAD_CAST:
                ret = HandleGroupBroadcast(signalInfo, channels, threads, tempAlgParams);
                break;
            case omni::OP_GROUP_REDUCE:
                ret = HandleGroupReduce(signalInfo, channels, threads, tempAlgParams);
                break;
            case omni::OP_SEND_RECV_WRITE_DPU:
                ret = HandleSendRecvWriteDPU(signalInfo, channels, threads, tempAlgParams);
                break;
            case omni::OP_SEND_WRITE_DPU:
                ret = HandleSendWriteDPU(signalInfo, channels, threads, tempAlgParams);
                break;
            case omni::OP_RECV_WRITE_DPU:
                ret = HandleRecvWriteDPU(signalInfo, channels, threads, tempAlgParams);
                break;
            default:
                HCCL_ERROR("[RunOmni] Unsupported operation type: %s", omni::OpTypeToString(signalInfo.opType).c_str());
                ret = HCCL_E_INTERNAL;
                break;
        }
        HCCL_INFO("YHB-CHECKER: =====%s end=====", omni::OpTypeToString(signalInfo.opType).c_str());

        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[RunOmni] Failed to handle operation type: %d, error: 0x%016llx", signalInfo.opType,
                HCCL_ERROR_CODE(ret));
            return ret;
        }
    }

    HCCL_INFO("[OmniTempAicpu][RunOmni] All OMNI signals processed successfully");
    return HCCL_SUCCESS;
}

void OmniTempAicpu::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = templateRankSize_;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void OmniTempAicpu::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult OmniTempAicpu::HandlePreSyncInterThreads(
    const omni::OmniNormalInstruction &signalInfo, const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[HandlePreSyncInterThreads] Start pre-sync inter threads, mainThreadIdx: %lu, subThreadNum: %lu",
        signalInfo.syncInfo.mainThreadIdx, signalInfo.syncInfo.subThreadNum);

    // 验证主线程索引
    if (signalInfo.syncInfo.mainThreadIdx >= threads.size()) {
        HCCL_ERROR("[HandlePreSyncInterThreads] Invalid mainThreadIdx: %lu, threads size: %lu",
            signalInfo.syncInfo.mainThreadIdx, threads.size());
        return HCCL_E_INTERNAL;
    }

    // 验证子线程数量
    if (signalInfo.syncInfo.subThreadNum == 0 || signalInfo.syncInfo.subThreadNum >= threads.size()) {
        HCCL_ERROR("[HandlePreSyncInterThreads] Invalid subThreadNum: %lu, threads size: %lu",
            signalInfo.syncInfo.subThreadNum, threads.size());
        return HCCL_E_INTERNAL;
    }

    // 准备子线程句柄
    std::vector<ThreadHandle> subThreads;
    if (signalInfo.syncInfo.subThreadIds.empty()) {
        // 如果没有指定子线程ID，使用默认顺序（排除主线程）
        for (size_t i = 0; i < threads.size(); i++) {
            if (i != signalInfo.syncInfo.mainThreadIdx) {
                subThreads.push_back(threads[i]);
            }
        }
        // 确保子线程数量匹配
        if (subThreads.size() != signalInfo.syncInfo.subThreadNum) {
            HCCL_WARNING("[HandlePreSyncInterThreads] subThreadNum mismatch: expected %lu, got %lu",
                signalInfo.syncInfo.subThreadNum, subThreads.size());
        }
    } else {
        // 使用指定的子线程ID
        for (const auto &threadId : signalInfo.syncInfo.subThreadIds) {
            if (threadId < threads.size()) {
                subThreads.push_back(threads[threadId]);
            } else {
                HCCL_WARNING("[HandlePreSyncInterThreads] Invalid threadId in subThreadIds: %u", threadId);
            }
        }
    }

    // 准备通知索引
    std::vector<u32> notifyIdxMainToSub;
    for (size_t i = 0; i < subThreads.size(); i++) {
        notifyIdxMainToSub.push_back(0); // 使用默认通知索引
    }

    // 执行前同步
    CHK_RET(PreSyncInterThreads(threads[signalInfo.syncInfo.mainThreadIdx], subThreads, notifyIdxMainToSub));

    HCCL_INFO("[HandlePreSyncInterThreads] Pre-sync inter threads completed successfully");
    return HCCL_SUCCESS;
}

HcclResult OmniTempAicpu::HandlePostSyncInterThreads(
    const omni::OmniNormalInstruction &signalInfo, const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[HandlePostSyncInterThreads] Start post-sync inter threads, mainThreadIdx: %lu, subThreadNum: %lu",
        signalInfo.syncInfo.mainThreadIdx, signalInfo.syncInfo.subThreadNum);

    // 验证主线程索引
    if (signalInfo.syncInfo.mainThreadIdx >= threads.size()) {
        HCCL_ERROR("[HandlePostSyncInterThreads] Invalid mainThreadIdx: %lu, threads size: %lu",
            signalInfo.syncInfo.mainThreadIdx, threads.size());
        return HCCL_E_INTERNAL;
    }

    // 验证子线程数量
    if (signalInfo.syncInfo.subThreadNum == 0 || signalInfo.syncInfo.subThreadNum >= threads.size()) {
        HCCL_ERROR("[HandlePostSyncInterThreads] Invalid subThreadNum: %lu, threads size: %lu",
            signalInfo.syncInfo.subThreadNum, threads.size());
        return HCCL_E_INTERNAL;
    }

    // 准备子线程句柄
    std::vector<ThreadHandle> subThreads;
    if (signalInfo.syncInfo.subThreadIds.empty()) {
        // 如果没有指定子线程ID，使用默认顺序（排除主线程）
        for (size_t i = 0; i < threads.size(); i++) {
            if (i != signalInfo.syncInfo.mainThreadIdx) {
                subThreads.push_back(threads[i]);
            }
        }
        // 确保子线程数量匹配
        if (subThreads.size() != signalInfo.syncInfo.subThreadNum) {
            HCCL_WARNING("[HandlePostSyncInterThreads] subThreadNum mismatch: expected %lu, got %lu",
                signalInfo.syncInfo.subThreadNum, subThreads.size());
        }
    } else {
        // 使用指定的子线程ID
        for (const auto &threadId : signalInfo.syncInfo.subThreadIds) {
            if (threadId < threads.size()) {
                subThreads.push_back(threads[threadId]);
            } else {
                HCCL_WARNING("[HandlePostSyncInterThreads] Invalid threadId in subThreadIds: %u", threadId);
            }
        }
    }

    // 准备通知索引
    std::vector<u32> notifyIdxSubToMain;
    for (size_t i = 0; i < subThreads.size(); i++) {
        notifyIdxSubToMain.push_back(i); // 使用递增通知索引
    }

    // 执行后同步
    CHK_RET(PostSyncInterThreads(threads[signalInfo.syncInfo.mainThreadIdx], subThreads, notifyIdxSubToMain));

    HCCL_INFO("[HandlePostSyncInterThreads] Post-sync inter threads completed successfully");
    return HCCL_SUCCESS;
}

HcclResult OmniTempAicpu::HandleLocalCopy(const omni::OmniNormalInstruction &signalInfo,
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams)
{
    if (signalInfo.sendRecvInfo.srcSliceInfo.empty() || signalInfo.sendRecvInfo.dstSliceInfo.empty()) {
        HCCL_ERROR("[HandleLocalCopy] Invalid slice info");
        return HCCL_E_INTERNAL;
    }

    uint64_t dtypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];

    HCCL_INFO("YHB-CHECKER: [HandleLocalCopy] count(%u), dataType[%u] dtypeSize[%llu]",
        signalInfo.sendRecvInfo.srcSliceInfo.size(), tempAlgParams.dataType, dtypeSize);
    for (uint32_t i = 0; i < signalInfo.sendRecvInfo.srcSliceInfo.size(); ++i) {
        // 计算源地址
        const auto &srcSlice = signalInfo.sendRecvInfo.srcSliceInfo[i];
        void *srcAddr = GetBufferAddrBySliceType(
            srcSlice.sliceType, tempAlgParams, srcSlice.sliceIdx, signalInfo.sendRecvInfo.sliceNum);
        if (!srcAddr) {
            HCCL_ERROR("[HandleLocalCopy] Invalid source address");
            return HCCL_E_INTERNAL;
        }

        // 计算目标地址
        const auto &dstSlice = signalInfo.sendRecvInfo.dstSliceInfo[i];
        void *dstAddr = GetBufferAddrBySliceType(
            dstSlice.sliceType, tempAlgParams, dstSlice.sliceIdx, signalInfo.sendRecvInfo.sliceNum);
        if (!dstAddr) {
            HCCL_ERROR("[HandleLocalCopy] Invalid destination address");
            return HCCL_E_INTERNAL;
        }

        // 计算传输大小：使用 per-slice process count 支持不等分 AlltoAllV
        uint64_t processCount = GetSliceProcessCount(srcSlice.sliceType, tempAlgParams, srcSlice.sliceIdx, dtypeSize);
        // 对于 copy/reduce，src 和 dst 的 processCount 应相同，取较小值保证安全
        uint64_t dstProcessCount
            = GetSliceProcessCount(dstSlice.sliceType, tempAlgParams, dstSlice.sliceIdx, dtypeSize);
        if (dstProcessCount < processCount) {
            processCount = dstProcessCount;
        }
        uint64_t processSize = processCount * dtypeSize;

        HCCL_DEBUG("HandleLocalCopy i[%u] processCount[%llu] processSize[%llu]", i, processCount, processSize);

        // 执行本地拷贝
        DataSlice srcSliceObj(srcAddr, 0, processSize, processCount);
        DataSlice dstSliceObj(dstAddr, 0, processSize, processCount);

        if (signalInfo.opType == omni::OP_LOCAL_COPY) {
            CHK_RET(static_cast<HcclResult>(
                LocalCopy(threads[signalInfo.sendRecvInfo.threadIdx], srcSliceObj, dstSliceObj)));
        } else if (signalInfo.opType == omni::OP_LOCAL_REDUCE) {
            CHK_RET(static_cast<HcclResult>(LocalReduce(threads[signalInfo.sendRecvInfo.threadIdx], srcSliceObj,
                dstSliceObj, tempAlgParams.dataType, signalInfo.sendRecvInfo.reduceType)));
        } else {
            HCCL_ERROR("[HandleLocalCopy] Invalid opcode %s", omni::OpTypeToString(signalInfo.opType).c_str());
        }
    }
    return HCCL_SUCCESS;
}

HcclResult OmniTempAicpu::HandleSendRecvWrite(const omni::OmniNormalInstruction &signalInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    uint64_t remoteRank = signalInfo.sendRecvInfo.dstSliceInfo[0].remoteRank;
    uint64_t remoteRecvRank = signalInfo.sendRecvInfo.dstSliceInfo[0].remoteRecvRank;

    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRecvWrite] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }
    auto recvIt = channels.find(remoteRecvRank);
    if (recvIt == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRecvWrite] Channel not found for remote recv rank: %lu", remoteRecvRank);
        return HCCL_E_INTERNAL;
    }

    if (++roundRobinIndex_ >= it->second.size()) {
        roundRobinIndex_ = 0;
    }
    roundRobinIndex_ = 0;
    const ChannelInfo &channel = it->second[roundRobinIndex_];
    const ChannelInfo &recvChannel = recvIt->second[roundRobinIndex_];
    uint64_t sliceNum = signalInfo.sendRecvInfo.sliceNum;
    uint64_t dtypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];

    HCCL_DEBUG("YHB: HandleSendRecvWrite thread %u myRank_ %u remoteRank %u remoteRecvRank %u sliceNum %u rrIndex %u "
               "rrSize %u",
        signalInfo.sendRecvInfo.threadIdx, myRank_, remoteRank, remoteRecvRank, sliceNum, roundRobinIndex_,
        it->second.size());

    // 准备发送数据切片
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;

    HCCL_INFO("YHB-CHECKER: [HandleSendRecvWrite] count(%u)", signalInfo.sendRecvInfo.srcSliceInfo.size());
    for (uint32_t i = 0; i < signalInfo.sendRecvInfo.srcSliceInfo.size(); ++i) {
        auto &srcSlice = signalInfo.sendRecvInfo.srcSliceInfo[i];
        void *srcAddr = GetBufferAddrBySliceType(srcSlice.sliceType, tempAlgParams, srcSlice.sliceIdx, sliceNum);
        if (!srcAddr) {
            HCCL_ERROR("[HandleSendRecvWrite] Invalid source address");
            return HCCL_E_INTERNAL;
        }
        // Per-slice process count 支持不等分 AlltoAllV
        uint64_t processCount = GetSliceProcessCount(srcSlice.sliceType, tempAlgParams, srcSlice.sliceIdx, dtypeSize);
        uint64_t processSize = processCount * dtypeSize;
        txSrcSlices.push_back(DataSlice(srcAddr, 0, processSize, processCount));

        auto &dstSlice = signalInfo.sendRecvInfo.dstSliceInfo[i];
        void *dstAddr
            = GetRemoteAddrBySliceType(dstSlice.sliceType, tempAlgParams, channel, dstSlice.sliceIdx, sliceNum);
        if (!dstAddr) {
            HCCL_ERROR("[HandleSendRecvWrite] Invalid destination address");
            return HCCL_E_INTERNAL;
        }
        txDstSlices.push_back(DataSlice(dstAddr, 0, processSize, processCount));
    }

    HCCL_DEBUG("myRank_[%u] remoteRank[%u]", myRank_, remoteRank);

    // 准备接收数据切片
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    // 执行发送接收写操作
    TxRxChannels txRxChannels(channel, recvChannel);

    TxRxSlicesList txRxSlicesList(SlicesList(std::move(txSrcSlices), std::move(txDstSlices)),
        SlicesList(std::move(rxSrcSlices), std::move(rxDstSlices)));

    if (signalInfo.opType == omni::OP_SEND_RECV_WRITE) {
        SendRecvInfo sendRecvInfo(std::move(txRxChannels), std::move(txRxSlicesList));
        CHK_RET(static_cast<HcclResult>(SendRecvWrite(sendRecvInfo, threads[signalInfo.sendRecvInfo.threadIdx])));
    } else if (signalInfo.opType == omni::OP_SEND_RECV_WRITE_REDUCE) {
        SendRecvReduceInfo sendRecvInfo(std::move(txRxChannels), std::move(txRxSlicesList), tempAlgParams.dataType,
            signalInfo.sendRecvInfo.reduceType);
        CHK_RET(static_cast<HcclResult>(SendRecvWriteReduce(sendRecvInfo, threads[signalInfo.sendRecvInfo.threadIdx])));
    } else {
        HCCL_ERROR("[HandleSendRecvWrite] Invalid opcode %s", omni::OpTypeToString(signalInfo.opType).c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult OmniTempAicpu::HandleSendWrite(const omni::OmniNormalInstruction &signalInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    uint64_t remoteRank = signalInfo.sendRecvInfo.dstSliceInfo[0].remoteRank;

    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendWrite] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }

    if (++roundRobinIndex_ >= it->second.size()) {
        roundRobinIndex_ = 0;
    }
    roundRobinIndex_ = 0;
    const ChannelInfo &channel = it->second[roundRobinIndex_];
    uint64_t sliceNum = signalInfo.sendRecvInfo.sliceNum;
    uint64_t dtypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];

    // 准备发送数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto &srcSlice : signalInfo.sendRecvInfo.srcSliceInfo) {
        void *srcAddr = GetBufferAddrBySliceType(srcSlice.sliceType, tempAlgParams, srcSlice.sliceIdx, sliceNum);
        if (!srcAddr) {
            HCCL_ERROR("[HandleSendWrite] Invalid source address");
            return HCCL_E_INTERNAL;
        }
        uint64_t processCount = GetSliceProcessCount(srcSlice.sliceType, tempAlgParams, srcSlice.sliceIdx, dtypeSize);
        uint64_t processSize = processCount * dtypeSize;
        srcSlices.push_back(DataSlice(srcAddr, 0, processSize, processCount));
    }

    for (uint32_t i = 0; i < signalInfo.sendRecvInfo.dstSliceInfo.size(); ++i) {
        const auto &dstSlice = signalInfo.sendRecvInfo.dstSliceInfo[i];
        void *dstAddr
            = GetRemoteAddrBySliceType(dstSlice.sliceType, tempAlgParams, channel, dstSlice.sliceIdx, sliceNum);
        if (!dstAddr) {
            HCCL_ERROR("[HandleSendWrite] Invalid destination address");
            return HCCL_E_INTERNAL;
        }
        // dstSlice processCount 应与对应 srcSlice 一致，确保发送和写入数据量匹配
        uint64_t processCount = (i < signalInfo.sendRecvInfo.srcSliceInfo.size())
                                    ? GetSliceProcessCount(signalInfo.sendRecvInfo.srcSliceInfo[i].sliceType,
                                          tempAlgParams, signalInfo.sendRecvInfo.srcSliceInfo[i].sliceIdx, dtypeSize)
                                    : tempAlgParams.sliceSize / dtypeSize;
        uint64_t processSize = processCount * dtypeSize;
        dstSlices.push_back(DataSlice(dstAddr, 0, processSize, processCount));
    }

    // 执行发送写操作
    SlicesList sendSliceList(std::move(srcSlices), std::move(dstSlices));
    if (signalInfo.opType == omni::OP_SEND_WRITE) {
        DataInfo sendInfo(channel, std::move(sendSliceList));
        CHK_RET(static_cast<HcclResult>(SendWrite(sendInfo, threads[signalInfo.sendRecvInfo.threadIdx])));
    } else if (signalInfo.opType == omni::OP_SEND_WRITE_REDUCE) {
        DataReduceInfo sendInfo(
            channel, std::move(sendSliceList), tempAlgParams.dataType, signalInfo.sendRecvInfo.reduceType);
        CHK_RET(static_cast<HcclResult>(SendWriteReduce(sendInfo, threads[signalInfo.sendRecvInfo.threadIdx])));
    } else {
        HCCL_ERROR("[HandleSendWrite] Invalid opcode %s", omni::OpTypeToString(signalInfo.opType).c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult OmniTempAicpu::HandleRecvWrite(const omni::OmniNormalInstruction &signalInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    uint64_t remoteRank = signalInfo.sendRecvInfo.srcSliceInfo[0].remoteRank;

    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleRecvWrite] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }

    if (++roundRobinIndex_ >= it->second.size()) {
        roundRobinIndex_ = 0;
    }
    roundRobinIndex_ = 0;
    const ChannelInfo &channel = it->second[roundRobinIndex_];
    uint64_t dtypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];

    // 准备接收数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    // RecvWrite 当前不使用 src/dst slices（接收端由远端写入），保留固定 sliceSize 逻辑
    uint64_t processCount = tempAlgParams.sliceSize / dtypeSize;
    uint64_t processSize = processCount * dtypeSize;

    // 执行接收写操作
    SlicesList recvSliceList(std::move(srcSlices), std::move(dstSlices));
    if (signalInfo.opType == omni::OP_RECV_WRITE) {
        DataInfo recvInfo(channel, std::move(recvSliceList));
        CHK_RET(static_cast<HcclResult>(RecvWrite(recvInfo, threads[signalInfo.sendRecvInfo.threadIdx])));
    } else if (signalInfo.opType == omni::OP_RECV_WRITE_REDUCE) {
        DataReduceInfo recvInfo(
            channel, std::move(recvSliceList), tempAlgParams.dataType, signalInfo.sendRecvInfo.reduceType);
        CHK_RET(static_cast<HcclResult>(RecvWriteReduce(recvInfo, threads[signalInfo.sendRecvInfo.threadIdx])));
    } else {
        HCCL_ERROR("[HandleRecvWrite] Invalid opcode %s", omni::OpTypeToString(signalInfo.opType).c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult OmniTempAicpu::HandleSendRecvRead(const omni::OmniNormalInstruction &signalInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    uint64_t remoteRank = signalInfo.sendRecvInfo.srcSliceInfo[0].remoteRank;
    uint64_t remoteRecvRank = signalInfo.sendRecvInfo.srcSliceInfo[0].remoteRecvRank;

    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRecvRead] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }
    auto recvIt = channels.find(remoteRecvRank);
    if (recvIt == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRecvRead] Channel not found for remote recv rank: %lu", remoteRecvRank);
        return HCCL_E_INTERNAL;
    }

    if (++roundRobinIndex_ >= it->second.size()) {
        roundRobinIndex_ = 0;
    }
    roundRobinIndex_ = 0;
    const ChannelInfo &channel = it->second[roundRobinIndex_];
    const ChannelInfo &recvChannel = recvIt->second[roundRobinIndex_];
    uint64_t sliceNum = signalInfo.sendRecvInfo.sliceNum;
    uint64_t dtypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];

    // 准备发送数据切片
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;

    // 准备接收数据切片
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    for (uint32_t i = 0; i < signalInfo.sendRecvInfo.srcSliceInfo.size(); ++i) {
        auto &srcSlice = signalInfo.sendRecvInfo.srcSliceInfo[i];
        void *srcAddr
            = GetRemoteAddrBySliceType(srcSlice.sliceType, tempAlgParams, channel, srcSlice.sliceIdx, sliceNum);
        if (!srcAddr) {
            HCCL_ERROR("[HandleSendRecvRead] Invalid source address");
            return HCCL_E_INTERNAL;
        }
        uint64_t processCount = GetSliceProcessCount(srcSlice.sliceType, tempAlgParams, srcSlice.sliceIdx, dtypeSize);
        uint64_t processSize = processCount * dtypeSize;
        rxSrcSlices.push_back(DataSlice(srcAddr, 0, processSize, processCount));

        auto &dstSlice = signalInfo.sendRecvInfo.dstSliceInfo[i];
        void *dstAddr = GetBufferAddrBySliceType(dstSlice.sliceType, tempAlgParams, dstSlice.sliceIdx, sliceNum);
        if (!dstAddr) {
            HCCL_ERROR("[HandleSendRecvRead] Invalid destination address");
            return HCCL_E_INTERNAL;
        }
        uint64_t dstProcessCount
            = GetSliceProcessCount(dstSlice.sliceType, tempAlgParams, dstSlice.sliceIdx, dtypeSize);
        uint64_t dstProcessSize = dstProcessCount * dtypeSize;
        rxDstSlices.push_back(DataSlice(dstAddr, 0, dstProcessSize, dstProcessCount));
    }

    // 执行发送接收读操作
    TxRxChannels txRxChannels(channel, recvChannel);
    TxRxSlicesList txRxSlicesList(SlicesList(std::move(txSrcSlices), std::move(txDstSlices)),
        SlicesList(std::move(rxSrcSlices), std::move(rxDstSlices)));

    if (signalInfo.opType == omni::OP_SEND_RECV_READ) {
        SendRecvInfo sendRecvInfo(std::move(txRxChannels), std::move(txRxSlicesList));
        CHK_RET(static_cast<HcclResult>(SendRecvRead(sendRecvInfo, threads[signalInfo.sendRecvInfo.threadIdx])));
    } else if (signalInfo.opType == omni::OP_SEND_RECV_READ_REDUCE) {
        SendRecvReduceInfo sendRecvInfo(std::move(txRxChannels), std::move(txRxSlicesList), tempAlgParams.dataType,
            signalInfo.sendRecvInfo.reduceType);
        CHK_RET(static_cast<HcclResult>(SendRecvReadReduce(sendRecvInfo, threads[signalInfo.sendRecvInfo.threadIdx])));
    } else {
        HCCL_ERROR("[HandleSendRecvRead] Invalid opcode %s", omni::OpTypeToString(signalInfo.opType).c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult OmniTempAicpu::HandleSendRead(const omni::OmniNormalInstruction &signalInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    uint64_t remoteRank = signalInfo.sendRecvInfo.dstSliceInfo[0].remoteRank;

    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRead] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }

    if (++roundRobinIndex_ >= it->second.size()) {
        roundRobinIndex_ = 0;
    }
    roundRobinIndex_ = 0;
    const ChannelInfo &channel = it->second[roundRobinIndex_];
    uint64_t sliceNum = signalInfo.sendRecvInfo.sliceNum;
    uint64_t dtypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];

    // 准备发送数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    // SendRead 当前不填充 slices，保留固定 sliceSize 逻辑
    uint64_t processCount = tempAlgParams.sliceSize / dtypeSize;
    uint64_t processSize = processCount * dtypeSize;

    // 执行发送读操作
    SlicesList sendSliceList(std::move(srcSlices), std::move(dstSlices));

    if (signalInfo.opType == omni::OP_SEND_READ) {
        DataInfo sendInfo(channel, std::move(sendSliceList));
        CHK_RET(static_cast<HcclResult>(SendRead(sendInfo, threads[signalInfo.sendRecvInfo.threadIdx])));
    } else if (signalInfo.opType == omni::OP_SEND_READ_REDUCE) {
        DataReduceInfo sendInfo(
            channel, std::move(sendSliceList), tempAlgParams.dataType, signalInfo.sendRecvInfo.reduceType);
        CHK_RET(static_cast<HcclResult>(SendReadReduce(sendInfo, threads[signalInfo.sendRecvInfo.threadIdx])));
    } else {
        HCCL_ERROR("[HandleSendRead] Invalid opcode %s", omni::OpTypeToString(signalInfo.opType).c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult OmniTempAicpu::HandleRecvRead(const omni::OmniNormalInstruction &signalInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    uint64_t remoteRank = signalInfo.sendRecvInfo.srcSliceInfo[0].remoteRank;

    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleRecvRead] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }

    if (++roundRobinIndex_ >= it->second.size()) {
        roundRobinIndex_ = 0;
    }
    roundRobinIndex_ = 0;
    const ChannelInfo &channel = it->second[roundRobinIndex_];
    uint64_t sliceNum = signalInfo.sendRecvInfo.sliceNum;
    uint64_t dtypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];

    // 准备接收数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto &srcSlice : signalInfo.sendRecvInfo.srcSliceInfo) {
        void *srcAddr
            = GetRemoteAddrBySliceType(srcSlice.sliceType, tempAlgParams, channel, srcSlice.sliceIdx, sliceNum);
        if (!srcAddr) {
            HCCL_ERROR("[HandleRecvRead] Invalid source address");
            return HCCL_E_INTERNAL;
        }
        uint64_t processCount = GetSliceProcessCount(srcSlice.sliceType, tempAlgParams, srcSlice.sliceIdx, dtypeSize);
        uint64_t processSize = processCount * dtypeSize;
        srcSlices.push_back(DataSlice(srcAddr, 0, processSize, processCount));
    }

    for (const auto &dstSlice : signalInfo.sendRecvInfo.dstSliceInfo) {
        void *dstAddr = GetBufferAddrBySliceType(dstSlice.sliceType, tempAlgParams, dstSlice.sliceIdx, sliceNum);
        if (!dstAddr) {
            HCCL_ERROR("[HandleRecvRead] Invalid destination address");
            return HCCL_E_INTERNAL;
        }
        uint64_t processCount = GetSliceProcessCount(dstSlice.sliceType, tempAlgParams, dstSlice.sliceIdx, dtypeSize);
        uint64_t processSize = processCount * dtypeSize;
        dstSlices.push_back(DataSlice(dstAddr, 0, processSize, processCount));
    }

    // 执行接收读操作
    SlicesList recvSliceList(std::move(srcSlices), std::move(dstSlices));

    if (signalInfo.opType == omni::OP_RECV_READ) {
        DataInfo recvInfo(channel, std::move(recvSliceList));
        CHK_RET(static_cast<HcclResult>(RecvRead(recvInfo, threads[signalInfo.sendRecvInfo.threadIdx])));
    } else if (signalInfo.opType == omni::OP_RECV_READ_REDUCE) {
        DataReduceInfo recvInfo(
            channel, std::move(recvSliceList), tempAlgParams.dataType, signalInfo.sendRecvInfo.reduceType);
        CHK_RET(static_cast<HcclResult>(RecvReadReduce(recvInfo, threads[signalInfo.sendRecvInfo.threadIdx])));
    } else {
        HCCL_ERROR("[HandleRecvRead] Invalid opcode %s", omni::OpTypeToString(signalInfo.opType).c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult OmniTempAicpu::HandleGroupBroadcast(const omni::OmniNormalInstruction &signalInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    uint64_t sliceNum = signalInfo.sendRecvInfo.sliceNum;
    uint64_t dtypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];

    // 准备源数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto &srcSlice : signalInfo.sendRecvInfo.srcSliceInfo) {
        void *srcAddr = GetBufferAddrBySliceType(srcSlice.sliceType, tempAlgParams, srcSlice.sliceIdx, sliceNum);
        if (!srcAddr) {
            HCCL_ERROR("[HandleGroupBroadcast] Invalid source address");
            return HCCL_E_INTERNAL;
        }
        uint64_t processCount = GetSliceProcessCount(srcSlice.sliceType, tempAlgParams, srcSlice.sliceIdx, dtypeSize);
        uint64_t processSize = processCount * dtypeSize;
        srcSlices.push_back(DataSlice(srcAddr, 0, processSize, processCount));
    }

    // 准备目标数据切片
    std::vector<ChannelInfo> channelInfos;
    for (const auto &dstSlice : signalInfo.sendRecvInfo.dstSliceInfo) {
        void *dstAddr = GetBufferAddrBySliceType(dstSlice.sliceType, tempAlgParams, dstSlice.sliceIdx, sliceNum);
        if (!dstAddr) {
            HCCL_ERROR("[HandleGroupBroadcast] Invalid destination address");
            return HCCL_E_INTERNAL;
        }
        uint64_t processCount = GetSliceProcessCount(dstSlice.sliceType, tempAlgParams, dstSlice.sliceIdx, dtypeSize);
        uint64_t processSize = processCount * dtypeSize;
        dstSlices.push_back(DataSlice(dstAddr, 0, processSize, processCount));

        // 获取对应远程rank的channel
        auto it = channels.find(dstSlice.remoteRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[HandleGroupBroadcast] Channel not found for remote rank: %lu", dstSlice.remoteRank);
            return HCCL_E_INTERNAL;
        }
        channelInfos.push_back(it->second[0]);
    }

    // 执行组广播操作 - AICPU不支持GroupBroadcast，使用循环发送代替
    HCCL_WARNING("[HandleGroupBroadcast] GroupBroadcast not supported in AICPU, using loop send instead");

    // 为每个目标rank执行发送操作
    for (size_t i = 0; i < channelInfos.size(); i++) {
        SlicesList sendSliceList(srcSlices, {dstSlices[i]});
        DataInfo sendInfo(channelInfos[i], std::move(sendSliceList));
        CHK_RET(static_cast<HcclResult>(SendWrite(sendInfo, threads[0])));
    }
    return HCCL_SUCCESS;
}

HcclResult OmniTempAicpu::HandleGroupReduce(const omni::OmniNormalInstruction &signalInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    uint64_t sliceNum = signalInfo.sendRecvInfo.sliceNum;
    uint64_t dtypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];

    // 准备源数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto &srcSlice : signalInfo.sendRecvInfo.srcSliceInfo) {
        void *srcAddr = GetBufferAddrBySliceType(srcSlice.sliceType, tempAlgParams, srcSlice.sliceIdx, sliceNum);
        if (!srcAddr) {
            HCCL_ERROR("[HandleGroupReduce] Invalid source address");
            return HCCL_E_INTERNAL;
        }
        uint64_t processCount = GetSliceProcessCount(srcSlice.sliceType, tempAlgParams, srcSlice.sliceIdx, dtypeSize);
        uint64_t processSize = processCount * dtypeSize;
        srcSlices.push_back(DataSlice(srcAddr, 0, processSize, processCount));
    }

    // 准备目标数据切片
    std::vector<ChannelInfo> channelInfos;
    for (const auto &dstSlice : signalInfo.sendRecvInfo.dstSliceInfo) {
        void *dstAddr = GetBufferAddrBySliceType(dstSlice.sliceType, tempAlgParams, dstSlice.sliceIdx, sliceNum);
        if (!dstAddr) {
            HCCL_ERROR("[HandleGroupReduce] Invalid destination address");
            return HCCL_E_INTERNAL;
        }
        uint64_t processCount = GetSliceProcessCount(dstSlice.sliceType, tempAlgParams, dstSlice.sliceIdx, dtypeSize);
        uint64_t processSize = processCount * dtypeSize;
        dstSlices.push_back(DataSlice(dstAddr, 0, processSize, processCount));

        // 获取对应远程rank的channel
        auto it = channels.find(dstSlice.remoteRank);
        if (it == channels.end() || it->second.empty()) {
            HCCL_ERROR("[HandleGroupReduce] Channel not found for remote rank: %lu", dstSlice.remoteRank);
            return HCCL_E_INTERNAL;
        }
        channelInfos.push_back(it->second[0]);
    }

    // 执行组归约操作 - AICPU不支持GroupReduce，使用循环接收归约代替
    HCCL_WARNING("[HandleGroupReduce] GroupReduce not supported in AICPU, using loop recv reduce instead");

    // 为每个源rank执行接收归约操作
    for (size_t i = 0; i < channelInfos.size(); i++) {
        SlicesList recvSliceList({srcSlices[i]}, dstSlices);
        DataReduceInfo recvInfo(
            channelInfos[i], std::move(recvSliceList), tempAlgParams.dataType, signalInfo.sendRecvInfo.reduceType);
        CHK_RET(static_cast<HcclResult>(RecvWriteReduce(recvInfo, threads[0])));
    }
    return HCCL_SUCCESS;
}

HcclResult OmniTempAicpu::HandleSendRecvWriteDPU(const omni::OmniNormalInstruction &signalInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    // AICPU模式不支持DPU操作，映射到普通的SendRecvWrite
    HCCL_WARNING("[HandleSendRecvWriteDPU] DPU operation not supported in AICPU mode, fallback to SendRecvWrite");
    return HandleSendRecvWrite(signalInfo, channels, threads, tempAlgParams);
}

HcclResult OmniTempAicpu::HandleSendWriteDPU(const omni::OmniNormalInstruction &signalInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    // AICPU模式不支持DPU操作，映射到普通的SendWrite
    HCCL_WARNING("[HandleSendWriteDPU] DPU operation not supported in AICPU mode, fallback to SendWrite");
    return HandleSendWrite(signalInfo, channels, threads, tempAlgParams);
}

HcclResult OmniTempAicpu::HandleRecvWriteDPU(const omni::OmniNormalInstruction &signalInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    // AICPU模式不支持DPU操作，映射到普通的RecvWrite
    HCCL_WARNING("[HandleRecvWriteDPU] DPU operation not supported in AICPU mode, fallback to RecvWrite");
    return HandleRecvWrite(signalInfo, channels, threads, tempAlgParams);
}

} // namespace ops_hccl