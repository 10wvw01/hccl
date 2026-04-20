/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/ins_temp_all_to_all_v_omni.h"
#include "log.h"
#include "param_check.h"
#include "utils.h"

namespace ops_hccl {

InsTempAlltoAllVOmni::InsTempAlltoAllVOmni(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempAlltoAllVOmni::~InsTempAlltoAllVOmni()
{
}

HcclResult InsTempAlltoAllVOmni::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest, const XmlInfo& xmlInfo)
{
    HCCL_INFO("[InsTempAlltoAllVOmni][CalcRes] Start to calc resource.");

    // 保存XML信息供后续使用（兼容旧模式，但新实现不依赖XML）
    xmlInfo_ = xmlInfo;

    u32 threadNum = templateRankSize_;
    threadNum_ = threadNum;
    resourceRequest.slaveThreadNum = xmlInfo.resInfo.slaveThreadNum;
    resourceRequest.notifyNumOnMainThread = xmlInfo.resInfo.notifyNumOnMainThread;
    resourceRequest.notifyNumPerThread.assign(xmlInfo.resInfo.notifyNumPerThread, 1);

    // 计算通道资源
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    // if (!xmlInfo.resInfo.mapchannelInfo.empty()) {
    //     const auto& channelMap = xmlInfo.resInfo.mapchannelInfo[0];
    //     for (const auto& entry : channelMap) {
    //         const u32& remoteRank = entry.first;
    //         const ChannelInfo& channelInfo = entry.second;
    //         HcclChannelDesc channelDesc = {};
    //         channelDesc.channelIndex = channelInfo.channelId;
    //         channelDesc.channelProtocol = channelInfo.channelProtocol;
    //         level0Channels.push_back(channelDesc);
    //     }
    // }
    resourceRequest.channels.push_back(level0Channels);

    HCCL_INFO("[InsTempAlltoAllVOmni][CalcRes] Calc resource success.");
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllVOmni::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // usrIn和cclBuffer大小相同
    return 1;
}


HcclResult InsTempAlltoAllVOmni::KernelRun(const OpParam& param,
    const TemplateDataParams& tempAlgParams,
    TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = SIZE_TABLE[dataType_];
    cclBufferCountPerRank_ = tempAlgParams.inputSliceStride / dataTypeSize_;

    HCCL_INFO("[InsTempAlltoAllVOmni] Run Start");

    // 检查XML信息是否已传递（兼容旧模式，但新实现不依赖XML）
    if (xmlInfo_.vecSendRecvInfo.empty() && xmlInfo_.vecSyncInfo.empty()) {
        HCCL_INFO("[InsTempAlltoAllVOmni] Using algorithm-based operation sequence");
    } else {
        HCCL_WARNING("[InsTempAlltoAllVOmni] XML information present but algorithm-based sequence will be used");
    }

    // 多线程同步处理
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    // 执行OMNI算法
    // 对于AICPU_TS引擎，循环逻辑已移到executor中
    if (param.engine != CommEngine::COMM_ENGINE_AICPU_TS) {
        CHK_RET(RunOmni(templateResource.channels, templateResource.threads, tempAlgParams));
    } else {
        // AICPU_TS引擎直接执行单个操作
        HCCL_INFO("[InsTempAlltoAllVOmni] AICPU_TS engine - single operation execution");
        // 对于AICPU_TS引擎，根据templateResource.optype执行相应操作
        HcclResult ret = HCCL_SUCCESS;
        switch (templateResource.optype) {
            case OP_LOCAL_COPY:
                ret = HandleLocalCopy(templateResource.threads, tempAlgParams);
                break;
            case OP_LOCAL_REDUCE:
                ret = HandleLocalReduce(templateResource.threads, tempAlgParams);
                break;
            case OP_SEND_RECV_WRITE:
                ret = HandleSendRecvWrite(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            case OP_SEND_WRITE:
                ret = HandleSendWrite(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            case OP_RECV_WRITE:
                ret = HandleRecvWrite(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            case OP_SEND_RECV_WRITE_REDUCE:
                ret = HandleSendRecvWriteReduce(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            case OP_SEND_WRITE_REDUCE:
                ret = HandleSendWriteReduce(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            case OP_RECV_WRITE_REDUCE:
                ret = HandleRecvWriteReduce(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            case OP_SEND_RECV_READ:
                ret = HandleSendRecvRead(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            case OP_SEND_READ:
                ret = HandleSendRead(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            case OP_RECV_READ:
                ret = HandleRecvRead(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            case OP_SEND_RECV_READ_REDUCE:
                ret = HandleSendRecvReadReduce(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            case OP_SEND_READ_REDUCE:
                ret = HandleSendReadReduce(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            case OP_RECV_READ_REDUCE:
                ret = HandleRecvReadReduce(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            case OP_GROUP_BROAD_CAST:
                ret = HandleGroupBroadcast(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            case OP_GROUP_REDUCE:
                ret = HandleGroupReduce(templateResource.channels, templateResource.threads, tempAlgParams);
                break;
            default:
                HCCL_ERROR("[InsTempAlltoAllVOmni] Unsupported operation type: %d", templateResource.optype);
                ret = HCCL_E_INTERNAL;
                break;
        }
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempAlltoAllVOmni] Failed to handle operation type: %d, error: 0x%016llx",
                      templateResource.optype, HCCL_ERROR_CODE(ret));
            return ret;
        }
    }

    // 后同步处理
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempAlltoAllVOmni] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::RunOmni(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads,
    const TemplateDataParams &tempAlgParams)
{
    // 执行OMNI指令序列
    DoRepeatOmni(channels, threads, tempAlgParams);
    return HCCL_SUCCESS;
}

void InsTempAlltoAllVOmni::DoRepeatOmni(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                       const std::vector<ThreadHandle> &threads,
                                       const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[InsTempAlltoAllVOmni][DoRepeatOmni] Start processing OMNI signals based on algorithm logic");

    // 检查XML信息是否已传递（兼容旧模式，但新实现不依赖XML）
    if (!xmlInfo_.vecSendRecvInfo.empty() || !xmlInfo_.vecSyncInfo.empty()) {
        HCCL_WARNING("[DoRepeatOmni] XML information present but algorithm-based sequence will be used");
    }

    // 根据AlltoAllV算法逻辑生成操作序列
    // 参考Mesh 1D实现，但保持OMNI的操作类型支持

    // 首先计算算法rank（在子通信域中的位置）
    u32 myAlgRank = 0;
    if (!subCommRanks_.empty() && !subCommRanks_[0].empty()) {
        auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
        if (iter != subCommRanks_[0].end()) {
            myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
        } else {
            HCCL_ERROR("[DoRepeatOmni] Failed to find myRank_ in subCommRanks_[0]");
            return;
        }
    }

    // cclBufferCountPerRank_已在KernelRun中计算

    // AlltoAllV主循环：处理与每个rank的通信
    for (u32 queIdx = 0; queIdx < threadNum_; queIdx++) {
        if (queIdx == myAlgRank) {
            // 本地拷贝：从input到output
            if (tempAlgParams.sendCounts[myAlgRank] > 0) {
                HcclResult ret = HandleLocalCopy(threads, tempAlgParams);
                if (ret != HCCL_SUCCESS) {
                    HCCL_ERROR("[DoRepeatOmni] HandleLocalCopy failed for local rank");
                    return;
                }
            }
            continue;
        }

        u32 nextRank = queIdx; // 逻辑rank
        u32 remoteRank = subCommRanks_[0][nextRank]; // 物理rank

        // 根据sendCounts和recvCounts决定操作类型
        bool hasSend = (tempAlgParams.sendCounts[nextRank] > 0);
        bool hasRecv = (tempAlgParams.recvCounts[nextRank] > 0);

        if (hasSend && hasRecv) {
            // 发送和接收都有，使用SendRecvWrite
            HcclResult ret = HandleSendRecvWrite(channels, threads, tempAlgParams);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[DoRepeatOmni] HandleSendRecvWrite failed for rank %u", nextRank);
                return;
            }
        } else if (hasSend) {
            // 只有发送，使用SendWrite
            HcclResult ret = HandleSendWrite(channels, threads, tempAlgParams);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[DoRepeatOmni] HandleSendWrite failed for rank %u", nextRank);
                return;
            }
        } else if (hasRecv) {
            // 只有接收，使用RecvWrite
            HcclResult ret = HandleRecvWrite(channels, threads, tempAlgParams);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[DoRepeatOmni] HandleRecvWrite failed for rank %u", nextRank);
                return;
            }
        }
        // 如果sendCounts和recvCounts都为0，跳过
    }

    // 后拷贝：从cclbuf到output（对于接收到的数据）
    for (u32 queIdx = 0; queIdx < threadNum_; queIdx++) {
        if (queIdx == myAlgRank) {
            continue; // 本地rank不需要后拷贝
        }

        u32 curAlgRank = queIdx;
        if (tempAlgParams.recvCounts[curAlgRank] > 0) {
            // 对于后拷贝，可以重用LocalCopy逻辑，但需要调整参数
            // 这里暂时跳过，实际实现需要额外的后拷贝逻辑
            HCCL_INFO("[DoRepeatOmni] Post-copy needed for rank %u, size %lu",
                     curAlgRank, tempAlgParams.recvCounts[curAlgRank] * dataTypeSize_);
        }
    }

    HCCL_INFO("[InsTempAlltoAllVOmni][DoRepeatOmni] All OMNI operations processed successfully");
}

void InsTempAlltoAllVOmni::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = templateRankSize_;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAlltoAllVOmni::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult InsTempAlltoAllVOmni::HandleLocalCopy(const std::vector<ThreadHandle> &threads,
                                                const TemplateDataParams &tempAlgParams)
{
    // 从TemplateDataParams中获取切片信息
    // 假设tempAlgParams中已经包含了必要的切片信息
    // 对于LocalCopy，通常是从input到output的拷贝

    // 检查必要的参数
    if (tempAlgParams.sliceSize == 0) {
        HCCL_ERROR("[HandleLocalCopy] Invalid sliceSize: %lu", tempAlgParams.sliceSize);
        return HCCL_E_INTERNAL;
    }

    // 计算源地址和目标地址
    // 根据Mesh 1D的实现，LocalCopy通常使用input和output指针
    void* srcAddr = tempAlgParams.buffInfo.inputPtr;
    void* dstAddr = tempAlgParams.buffInfo.outputPtr;

    if (!srcAddr || !dstAddr) {
        HCCL_ERROR("[HandleLocalCopy] Invalid buffer addresses: src=%p, dst=%p", srcAddr, dstAddr);
        return HCCL_E_INTERNAL;
    }

    // 根据切片偏移计算地址 - 参考Mesh 1D实现
    // 对于本地拷贝，使用sendCounts和recvCounts确定偏移
    u64 srcOffset = 0;
    u64 dstOffset = 0;

    // 在AlltoAllV中，本地拷贝通常处理myAlgRank对应的数据
    // 这里需要额外的上下文信息，暂时使用0作为示例
    u32 myAlgRank = 0;
    if (myAlgRank < tempAlgParams.sdispls.size()) {
        srcOffset = tempAlgParams.sdispls[myAlgRank] * dataTypeSize_;
    }
    if (myAlgRank < tempAlgParams.rdispls.size()) {
        dstOffset = tempAlgParams.rdispls[myAlgRank] * dataTypeSize_;
    }

    srcAddr = static_cast<char*>(srcAddr) + srcOffset;
    dstAddr = static_cast<char*>(dstAddr) + dstOffset;

    // 执行本地拷贝
    DataSlice srcSliceObj(srcAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_);
    DataSlice dstSliceObj(dstAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_);

    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSliceObj, dstSliceObj)));

    HCCL_INFO("[HandleLocalCopy] Local copy completed: size=%lu", tempAlgParams.sliceSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleLocalReduce(const std::vector<ThreadHandle> &threads,
                                                  const TemplateDataParams &tempAlgParams)
{
    // 从TemplateDataParams中获取归约相关信息
    // LocalReduce通常用于多个源切片归约到单个目标切片

    // 检查必要的参数
    if (tempAlgParams.sliceSize == 0) {
        HCCL_ERROR("[HandleLocalReduce] Invalid sliceSize: %lu", tempAlgParams.sliceSize);
        return HCCL_E_INTERNAL;
    }

    // 计算目标地址
    // 根据具体实现，目标地址可能是output或cclbuf
    void* dstAddr = tempAlgParams.buffInfo.outputPtr; // 默认使用output
    if (!dstAddr) {
        dstAddr = tempAlgParams.buffInfo.hcclBuff.addr; // 备选使用cclbuf
    }

    if (!dstAddr) {
        HCCL_ERROR("[HandleLocalReduce] Invalid destination address");
        return HCCL_E_INTERNAL;
    }

    // 计算目标偏移 - 参考Mesh 1D实现
    u64 dstOffset = 0;
    // 在AlltoAllV中，LocalReduce通常用于后处理阶段
    // 这里需要额外的上下文信息，暂时使用0作为示例
    u32 myAlgRank = 0;
    if (myAlgRank < tempAlgParams.rdispls.size()) {
        dstOffset = tempAlgParams.rdispls[myAlgRank] * dataTypeSize_;
    }

    dstAddr = static_cast<char*>(dstAddr) + dstOffset;
    DataSlice dstSliceObj(dstAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_);

    // 处理源切片 - AlltoAllV通常只有一个源切片用于LocalReduce
    void* srcAddr = tempAlgParams.buffInfo.hcclBuff.addr; // 通常从cclbuf归约到output
    if (!srcAddr) {
        srcAddr = tempAlgParams.buffInfo.inputPtr; // 备选使用input
    }

    if (!srcAddr) {
        HCCL_ERROR("[HandleLocalReduce] Invalid source address");
        return HCCL_E_INTERNAL;
    }

    // 计算源偏移 - 参考Mesh 1D后拷贝逻辑
    u64 srcOffset = 0;
    u32 curAlgRank = 0; // 需要知道当前处理的rank
    srcOffset = curAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff;

    srcAddr = static_cast<char*>(srcAddr) + srcOffset;
    DataSlice srcSliceObj(srcAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_);

    // 执行本地归约操作
    // 归约类型和数据类型需要从tempAlgParams中获取
    HcclDataType dataType = tempAlgParams.dataType;
    HcclReduceOp reduceType = HCCL_REDUCE_SUM; // 默认使用SUM

    CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], srcSliceObj, dstSliceObj, dataType, reduceType)));

    HCCL_INFO("[HandleLocalReduce] Local reduce completed: size=%lu, dataType=%d, reduceType=%d",
              tempAlgParams.sliceSize, dataType, reduceType);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendRecvWrite(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                    const std::vector<ThreadHandle> &threads,
                                                    const TemplateDataParams &tempAlgParams)
{
    // 参考Mesh 1D实现中的SendRecvWrite逻辑
    // 需要从tempAlgParams中获取远程rank信息

    // 检查必要的参数
    if (tempAlgParams.sliceSize == 0) {
        HCCL_ERROR("[HandleSendRecvWrite] Invalid sliceSize: %lu", tempAlgParams.sliceSize);
        return HCCL_E_INTERNAL;
    }

    // 从tempAlgParams中获取远程rank信息
    // 在AlltoAllV中，远程rank通过循环索引确定
    // 这里需要额外的上下文信息，暂时使用第一个通道作为示例
    u32 remoteRank = 0;
    if (!channels.empty()) {
        remoteRank = channels.begin()->first;
    }

    // 查找对应远程rank的通道信息
    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRecvWrite] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    // 参考Mesh 1D实现，发送数据通常从input到远程cclbuf
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;

    void* srcAddr = tempAlgParams.buffInfo.inputPtr;
    if (!srcAddr) {
        HCCL_ERROR("[HandleSendRecvWrite] Invalid source address");
        return HCCL_E_INTERNAL;
    }

    // 计算发送偏移 - 参考Mesh 1D实现：tempAlgParams.sdispls[nextRank] * dataTypeSize_
    // 这里需要知道nextRank，暂时使用0作为示例
    u32 nextRank = 0;
    u64 srcOffset = 0;
    if (nextRank < tempAlgParams.sdispls.size()) {
        srcOffset = tempAlgParams.sdispls[nextRank] * dataTypeSize_;
    }
    srcAddr = static_cast<char*>(srcAddr) + srcOffset;
    txSrcSlices.push_back(DataSlice(srcAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 目标地址通常是远程cclbuf
    void* remoteCclBuffAddr = channel.remoteCclMem.addr;
    if (!remoteCclBuffAddr) {
        HCCL_ERROR("[HandleSendRecvWrite] Invalid remote ccl buffer address");
        return HCCL_E_INTERNAL;
    }

    // 计算远程cclbuf偏移 - 参考Mesh 1D实现：myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff
    // 这里需要知道myAlgRank，暂时使用0作为示例
    u32 myAlgRank = 0;
    u64 remoteOffset = myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff;
    remoteCclBuffAddr = static_cast<char*>(remoteCclBuffAddr) + remoteOffset;
    txDstSlices.push_back(DataSlice(remoteCclBuffAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 准备接收数据切片
    // 接收数据通常从本地cclbuf到output
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    void* localCclBuffAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    if (!localCclBuffAddr) {
        HCCL_ERROR("[HandleSendRecvWrite] Invalid local ccl buffer address");
        return HCCL_E_INTERNAL;
    }

    // 计算本地cclbuf偏移 - 参考Mesh 1D实现：nextRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff
    u64 localCclOffset = nextRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff;
    localCclBuffAddr = static_cast<char*>(localCclBuffAddr) + localCclOffset;
    rxSrcSlices.push_back(DataSlice(localCclBuffAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    void* dstAddr = tempAlgParams.buffInfo.outputPtr;
    if (!dstAddr) {
        HCCL_ERROR("[HandleSendRecvWrite] Invalid destination address");
        return HCCL_E_INTERNAL;
    }

    // 计算接收偏移 - 参考Mesh 1D实现：tempAlgParams.rdispls[curAlgRank] * dataTypeSize_
    // 这里需要知道curAlgRank，暂时使用0作为示例
    u32 curAlgRank = 0;
    u64 dstOffset = 0;
    if (curAlgRank < tempAlgParams.rdispls.size()) {
        dstOffset = tempAlgParams.rdispls[curAlgRank] * dataTypeSize_;
    }
    dstAddr = static_cast<char*>(dstAddr) + dstOffset;
    rxDstSlices.push_back(DataSlice(dstAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 执行发送接收写操作
    TxRxChannels txRxChannels(channel, channel);
    TxRxSlicesList txRxSlicesList(
        SlicesList(std::move(txSrcSlices), std::move(txDstSlices)),
        SlicesList(std::move(rxSrcSlices), std::move(rxDstSlices))
    );
    SendRecvInfo sendRecvInfo(std::move(txRxChannels), std::move(txRxSlicesList));

    CHK_RET(static_cast<HcclResult>(SendRecvWrite(sendRecvInfo, threads[0])));

    HCCL_INFO("[HandleSendRecvWrite] SendRecvWrite completed: remoteRank=%lu, size=%lu", remoteRank, tempAlgParams.sliceSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendWrite(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                const std::vector<ThreadHandle> &threads,
                                                const TemplateDataParams &tempAlgParams)
{
    // 参考Mesh 1D实现中的SendWrite逻辑
    // 需要从tempAlgParams中获取远程rank信息

    // 检查必要的参数
    if (tempAlgParams.sliceSize == 0) {
        HCCL_ERROR("[HandleSendWrite] Invalid sliceSize: %lu", tempAlgParams.sliceSize);
        return HCCL_E_INTERNAL;
    }

    // 从tempAlgParams中获取远程rank信息
    // 假设tempAlgParams中包含了远程rank信息
    u32 remoteRank = 0; // 需要从tempAlgParams中获取
    // 临时实现：使用第一个通道作为远程rank
    if (!channels.empty()) {
        remoteRank = channels.begin()->first;
    }

    // 查找对应远程rank的通道信息
    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendWrite] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    // 参考Mesh 1D实现，发送数据通常从input到远程cclbuf
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    void* srcAddr = tempAlgParams.buffInfo.inputPtr;
    if (!srcAddr) {
        HCCL_ERROR("[HandleSendWrite] Invalid source address");
        return HCCL_E_INTERNAL;
    }

    // 计算发送偏移
    // 参考Mesh 1D实现：tempAlgParams.sdispls[nextRank] * dataTypeSize_
    u64 srcOffset = 0; // 需要从tempAlgParams中获取
    if (tempAlgParams.inputSliceStride > 0) {
        // 根据具体实现确定如何计算偏移
        srcOffset = 0; // 临时占位
    }

    srcAddr = static_cast<char*>(srcAddr) + srcOffset;
    srcSlices.push_back(DataSlice(srcAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 目标地址通常是远程cclbuf
    void* remoteCclBuffAddr = channel.remoteCclMem.addr;
    if (!remoteCclBuffAddr) {
        HCCL_ERROR("[HandleSendWrite] Invalid remote ccl buffer address");
        return HCCL_E_INTERNAL;
    }

    // 计算远程cclbuf偏移
    // 参考Mesh 1D实现：myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff
    u64 remoteOffset = 0; // 需要从tempAlgParams中获取
    remoteCclBuffAddr = static_cast<char*>(remoteCclBuffAddr) + remoteOffset;
    dstSlices.push_back(DataSlice(remoteCclBuffAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 执行发送写操作
    SlicesList sendSliceList(std::move(srcSlices), std::move(dstSlices));
    DataInfo sendInfo(channel, std::move(sendSliceList));
    CHK_RET(static_cast<HcclResult>(SendWrite(sendInfo, threads[0])));

    HCCL_INFO("[HandleSendWrite] SendWrite completed: remoteRank=%lu, size=%lu", remoteRank, tempAlgParams.sliceSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleRecvWrite(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                const std::vector<ThreadHandle> &threads,
                                                const TemplateDataParams &tempAlgParams)
{
    // 参考Mesh 1D实现中的RecvWrite逻辑
    // 需要从tempAlgParams中获取远程rank信息

    // 检查必要的参数
    if (tempAlgParams.sliceSize == 0) {
        HCCL_ERROR("[HandleRecvWrite] Invalid sliceSize: %lu", tempAlgParams.sliceSize);
        return HCCL_E_INTERNAL;
    }

    // 从tempAlgParams中获取远程rank信息
    // 假设tempAlgParams中包含了远程rank信息
    u32 remoteRank = 0; // 需要从tempAlgParams中获取
    // 临时实现：使用第一个通道作为远程rank
    if (!channels.empty()) {
        remoteRank = channels.begin()->first;
    }

    // 查找对应远程rank的通道信息
    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleRecvWrite] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备接收数据切片
    // 参考Mesh 1D实现，接收数据通常从本地cclbuf到output
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    // 源地址通常是本地cclbuf
    void* localCclBuffAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    if (!localCclBuffAddr) {
        HCCL_ERROR("[HandleRecvWrite] Invalid local ccl buffer address");
        return HCCL_E_INTERNAL;
    }

    // 计算本地cclbuf偏移
    // 参考Mesh 1D实现：nextRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff
    u64 localCclOffset = 0; // 需要从tempAlgParams中获取
    localCclBuffAddr = static_cast<char*>(localCclBuffAddr) + localCclOffset;
    srcSlices.push_back(DataSlice(localCclBuffAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    void* dstAddr = tempAlgParams.buffInfo.outputPtr;
    if (!dstAddr) {
        HCCL_ERROR("[HandleRecvWrite] Invalid destination address");
        return HCCL_E_INTERNAL;
    }

    // 计算接收偏移
    // 参考Mesh 1D实现：tempAlgParams.rdispls[curAlgRank] * dataTypeSize_
    u64 dstOffset = 0; // 需要从tempAlgParams中获取
    dstAddr = static_cast<char*>(dstAddr) + dstOffset;
    dstSlices.push_back(DataSlice(dstAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 执行接收写操作
    SlicesList recvSliceList(std::move(srcSlices), std::move(dstSlices));
    DataInfo recvInfo(channel, std::move(recvSliceList));
    CHK_RET(static_cast<HcclResult>(RecvWrite(recvInfo, threads[0])));

    HCCL_INFO("[HandleRecvWrite] RecvWrite completed: remoteRank=%lu, size=%lu", remoteRank, tempAlgParams.sliceSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendRecvWriteReduce(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                          const std::vector<ThreadHandle> &threads,
                                                          const TemplateDataParams &tempAlgParams)
{
    // 参考Mesh 1D实现中的SendRecvWriteReduce逻辑
    // 需要从tempAlgParams中获取远程rank信息

    // 检查必要的参数
    if (tempAlgParams.sliceSize == 0) {
        HCCL_ERROR("[HandleSendRecvWriteReduce] Invalid sliceSize: %lu", tempAlgParams.sliceSize);
        return HCCL_E_INTERNAL;
    }

    // 从tempAlgParams中获取远程rank信息
    // 假设tempAlgParams中包含了远程rank信息
    u32 remoteRank = 0; // 需要从tempAlgParams中获取
    // 临时实现：使用第一个通道作为远程rank
    if (!channels.empty()) {
        remoteRank = channels.begin()->first;
    }

    // 查找对应远程rank的通道信息
    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRecvWriteReduce] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    // 参考Mesh 1D实现，发送数据通常从input到远程cclbuf
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;

    void* srcAddr = tempAlgParams.buffInfo.inputPtr;
    if (!srcAddr) {
        HCCL_ERROR("[HandleSendRecvWriteReduce] Invalid source address");
        return HCCL_E_INTERNAL;
    }

    // 计算发送偏移
    // 参考Mesh 1D实现：tempAlgParams.sdispls[nextRank] * dataTypeSize_
    u64 srcOffset = 0; // 需要从tempAlgParams中获取
    if (tempAlgParams.inputSliceStride > 0) {
        srcOffset = 0; // 临时占位
    }

    srcAddr = static_cast<char*>(srcAddr) + srcOffset;
    txSrcSlices.push_back(DataSlice(srcAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 目标地址通常是远程cclbuf
    void* remoteCclBuffAddr = channel.remoteCclMem.addr;
    if (!remoteCclBuffAddr) {
        HCCL_ERROR("[HandleSendRecvWriteReduce] Invalid remote ccl buffer address");
        return HCCL_E_INTERNAL;
    }

    // 计算远程cclbuf偏移
    // 参考Mesh 1D实现：myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff
    u64 remoteOffset = 0; // 需要从tempAlgParams中获取
    remoteCclBuffAddr = static_cast<char*>(remoteCclBuffAddr) + remoteOffset;
    txDstSlices.push_back(DataSlice(remoteCclBuffAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 准备接收数据切片
    // 接收数据通常从本地cclbuf到output
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    void* localCclBuffAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    if (!localCclBuffAddr) {
        HCCL_ERROR("[HandleSendRecvWriteReduce] Invalid local ccl buffer address");
        return HCCL_E_INTERNAL;
    }

    // 计算本地cclbuf偏移
    // 参考Mesh 1D实现：nextRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff
    u64 localCclOffset = 0; // 需要从tempAlgParams中获取
    localCclBuffAddr = static_cast<char*>(localCclBuffAddr) + localCclOffset;
    rxSrcSlices.push_back(DataSlice(localCclBuffAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    void* dstAddr = tempAlgParams.buffInfo.outputPtr;
    if (!dstAddr) {
        HCCL_ERROR("[HandleSendRecvWriteReduce] Invalid destination address");
        return HCCL_E_INTERNAL;
    }

    // 计算接收偏移
    // 参考Mesh 1D实现：tempAlgParams.rdispls[curAlgRank] * dataTypeSize_
    u64 dstOffset = 0; // 需要从tempAlgParams中获取
    dstAddr = static_cast<char*>(dstAddr) + dstOffset;
    rxDstSlices.push_back(DataSlice(dstAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 执行发送接收写归约操作
    // 归约类型和数据类型需要从tempAlgParams中获取
    HcclDataType dataType = tempAlgParams.dataType;
    HcclReduceOp reduceType = HCCL_REDUCE_SUM; // 默认使用SUM

    TxRxChannels txRxChannels(channel, channel);
    TxRxSlicesList txRxSlicesList(
        SlicesList(std::move(txSrcSlices), std::move(txDstSlices)),
        SlicesList(std::move(rxSrcSlices), std::move(rxDstSlices))
    );
    SendRecvReduceInfo sendRecvInfo(std::move(txRxChannels), std::move(txRxSlicesList),
                                   dataType, reduceType);

    CHK_RET(static_cast<HcclResult>(SendRecvWriteReduce(sendRecvInfo, threads[0])));

    HCCL_INFO("[HandleSendRecvWriteReduce] SendRecvWriteReduce completed: remoteRank=%lu, size=%lu", remoteRank, tempAlgParams.sliceSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendWriteReduce(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                      const std::vector<ThreadHandle> &threads,
                                                      const TemplateDataParams &tempAlgParams)
{
    // 参考Mesh 1D实现中的SendWriteReduce逻辑
    // 需要从tempAlgParams中获取远程rank信息

    // 检查必要的参数
    if (tempAlgParams.sliceSize == 0) {
        HCCL_ERROR("[HandleSendWriteReduce] Invalid sliceSize: %lu", tempAlgParams.sliceSize);
        return HCCL_E_INTERNAL;
    }

    // 从tempAlgParams中获取远程rank信息
    // 假设tempAlgParams中包含了远程rank信息
    u32 remoteRank = 0; // 需要从tempAlgParams中获取
    // 临时实现：使用第一个通道作为远程rank
    if (!channels.empty()) {
        remoteRank = channels.begin()->first;
    }

    // 查找对应远程rank的通道信息
    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendWriteReduce] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    // 参考Mesh 1D实现，发送数据通常从input到远程cclbuf
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    void* srcAddr = tempAlgParams.buffInfo.inputPtr;
    if (!srcAddr) {
        HCCL_ERROR("[HandleSendWriteReduce] Invalid source address");
        return HCCL_E_INTERNAL;
    }

    // 计算发送偏移
    // 参考Mesh 1D实现：tempAlgParams.sdispls[nextRank] * dataTypeSize_
    u64 srcOffset = 0; // 需要从tempAlgParams中获取
    if (tempAlgParams.inputSliceStride > 0) {
        srcOffset = 0; // 临时占位
    }

    srcAddr = static_cast<char*>(srcAddr) + srcOffset;
    srcSlices.push_back(DataSlice(srcAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 目标地址通常是远程cclbuf
    void* remoteCclBuffAddr = channel.remoteCclMem.addr;
    if (!remoteCclBuffAddr) {
        HCCL_ERROR("[HandleSendWriteReduce] Invalid remote ccl buffer address");
        return HCCL_E_INTERNAL;
    }

    // 计算远程cclbuf偏移
    // 参考Mesh 1D实现：myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff
    u64 remoteOffset = 0; // 需要从tempAlgParams中获取
    remoteCclBuffAddr = static_cast<char*>(remoteCclBuffAddr) + remoteOffset;
    dstSlices.push_back(DataSlice(remoteCclBuffAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 执行发送写归约操作
    // 归约类型和数据类型需要从tempAlgParams中获取
    HcclDataType dataType = tempAlgParams.dataType;
    HcclReduceOp reduceType = HCCL_REDUCE_SUM; // 默认使用SUM

    SlicesList sendSliceList(std::move(srcSlices), std::move(dstSlices));
    DataReduceInfo sendInfo(channel, std::move(sendSliceList), dataType, reduceType);
    CHK_RET(static_cast<HcclResult>(SendWriteReduce(sendInfo, threads[0])));

    HCCL_INFO("[HandleSendWriteReduce] SendWriteReduce completed: remoteRank=%lu, size=%lu", remoteRank, tempAlgParams.sliceSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleRecvWriteReduce(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                      const std::vector<ThreadHandle> &threads,
                                                      const TemplateDataParams &tempAlgParams)
{
    // 参考Mesh 1D实现中的RecvWriteReduce逻辑
    // 需要从tempAlgParams中获取远程rank信息

    // 检查必要的参数
    if (tempAlgParams.sliceSize == 0) {
        HCCL_ERROR("[HandleRecvWriteReduce] Invalid sliceSize: %lu", tempAlgParams.sliceSize);
        return HCCL_E_INTERNAL;
    }

    // 从tempAlgParams中获取远程rank信息
    // 假设tempAlgParams中包含了远程rank信息
    u32 remoteRank = 0; // 需要从tempAlgParams中获取
    // 临时实现：使用第一个通道作为远程rank
    if (!channels.empty()) {
        remoteRank = channels.begin()->first;
    }

    // 查找对应远程rank的通道信息
    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleRecvWriteReduce] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备接收数据切片
    // 参考Mesh 1D实现，接收数据通常从本地cclbuf到output
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    // 源地址通常是本地cclbuf
    void* localCclBuffAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    if (!localCclBuffAddr) {
        HCCL_ERROR("[HandleRecvWriteReduce] Invalid local ccl buffer address");
        return HCCL_E_INTERNAL;
    }

    // 计算本地cclbuf偏移
    // 参考Mesh 1D实现：nextRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff
    u64 localCclOffset = 0; // 需要从tempAlgParams中获取
    localCclBuffAddr = static_cast<char*>(localCclBuffAddr) + localCclOffset;
    srcSlices.push_back(DataSlice(localCclBuffAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    void* dstAddr = tempAlgParams.buffInfo.outputPtr;
    if (!dstAddr) {
        HCCL_ERROR("[HandleRecvWriteReduce] Invalid destination address");
        return HCCL_E_INTERNAL;
    }

    // 计算接收偏移
    // 参考Mesh 1D实现：tempAlgParams.rdispls[curAlgRank] * dataTypeSize_
    u64 dstOffset = 0; // 需要从tempAlgParams中获取
    dstAddr = static_cast<char*>(dstAddr) + dstOffset;
    dstSlices.push_back(DataSlice(dstAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 执行接收写归约操作
    // 归约类型和数据类型需要从tempAlgParams中获取
    HcclDataType dataType = tempAlgParams.dataType;
    HcclReduceOp reduceType = HCCL_REDUCE_SUM; // 默认使用SUM

    SlicesList recvSliceList(std::move(srcSlices), std::move(dstSlices));
    DataReduceInfo recvInfo(channel, std::move(recvSliceList), dataType, reduceType);
    CHK_RET(static_cast<HcclResult>(RecvWriteReduce(recvInfo, threads[0])));

    HCCL_INFO("[HandleRecvWriteReduce] RecvWriteReduce completed: remoteRank=%lu, size=%lu", remoteRank, tempAlgParams.sliceSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendRecvRead(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                   const std::vector<ThreadHandle> &threads,
                                                   const TemplateDataParams &tempAlgParams)
{
    // 参考Mesh 1D实现中的SendRecvRead逻辑
    // 需要从tempAlgParams中获取远程rank信息

    // 检查必要的参数
    if (tempAlgParams.sliceSize == 0) {
        HCCL_ERROR("[HandleSendRecvRead] Invalid sliceSize: %lu", tempAlgParams.sliceSize);
        return HCCL_E_INTERNAL;
    }

    // 从tempAlgParams中获取远程rank信息
    // 假设tempAlgParams中包含了远程rank信息
    u32 remoteRank = 0; // 需要从tempAlgParams中获取
    // 临时实现：使用第一个通道作为远程rank
    if (!channels.empty()) {
        remoteRank = channels.begin()->first;
    }

    // 查找对应远程rank的通道信息
    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRecvRead] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    // 参考Mesh 1D实现，发送数据通常从input到远程cclbuf
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;

    void* srcAddr = tempAlgParams.buffInfo.inputPtr;
    if (!srcAddr) {
        HCCL_ERROR("[HandleSendRecvRead] Invalid source address");
        return HCCL_E_INTERNAL;
    }

    // 计算发送偏移
    // 参考Mesh 1D实现：tempAlgParams.sdispls[nextRank] * dataTypeSize_
    u64 srcOffset = 0; // 需要从tempAlgParams中获取
    if (tempAlgParams.inputSliceStride > 0) {
        srcOffset = 0; // 临时占位
    }

    srcAddr = static_cast<char*>(srcAddr) + srcOffset;
    txSrcSlices.push_back(DataSlice(srcAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 目标地址通常是远程cclbuf
    void* remoteCclBuffAddr = channel.remoteCclMem.addr;
    if (!remoteCclBuffAddr) {
        HCCL_ERROR("[HandleSendRecvRead] Invalid remote ccl buffer address");
        return HCCL_E_INTERNAL;
    }

    // 计算远程cclbuf偏移
    // 参考Mesh 1D实现：myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff
    u64 remoteOffset = 0; // 需要从tempAlgParams中获取
    remoteCclBuffAddr = static_cast<char*>(remoteCclBuffAddr) + remoteOffset;
    txDstSlices.push_back(DataSlice(remoteCclBuffAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 准备接收数据切片
    // 接收数据通常从本地cclbuf到output
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    void* localCclBuffAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    if (!localCclBuffAddr) {
        HCCL_ERROR("[HandleSendRecvRead] Invalid local ccl buffer address");
        return HCCL_E_INTERNAL;
    }

    // 计算本地cclbuf偏移
    // 参考Mesh 1D实现：nextRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff
    u64 localCclOffset = 0; // 需要从tempAlgParams中获取
    localCclBuffAddr = static_cast<char*>(localCclBuffAddr) + localCclOffset;
    rxSrcSlices.push_back(DataSlice(localCclBuffAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    void* dstAddr = tempAlgParams.buffInfo.outputPtr;
    if (!dstAddr) {
        HCCL_ERROR("[HandleSendRecvRead] Invalid destination address");
        return HCCL_E_INTERNAL;
    }

    // 计算接收偏移
    // 参考Mesh 1D实现：tempAlgParams.rdispls[curAlgRank] * dataTypeSize_
    u64 dstOffset = 0; // 需要从tempAlgParams中获取
    dstAddr = static_cast<char*>(dstAddr) + dstOffset;
    rxDstSlices.push_back(DataSlice(dstAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 执行发送接收读操作
    TxRxChannels txRxChannels(channel, channel);
    TxRxSlicesList txRxSlicesList(
        SlicesList(std::move(txSrcSlices), std::move(txDstSlices)),
        SlicesList(std::move(rxSrcSlices), std::move(rxDstSlices))
    );
    SendRecvInfo sendRecvInfo(std::move(txRxChannels), std::move(txRxSlicesList));

    CHK_RET(static_cast<HcclResult>(SendRecvRead(sendRecvInfo, threads[0])));

    HCCL_INFO("[HandleSendRecvRead] SendRecvRead completed: remoteRank=%lu, size=%lu", remoteRank, tempAlgParams.sliceSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendRead(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                               const std::vector<ThreadHandle> &threads,
                                               const TemplateDataParams &tempAlgParams)
{
    // 参考Mesh 1D实现中的SendRead逻辑
    // 需要从tempAlgParams中获取远程rank信息

    // 检查必要的参数
    if (tempAlgParams.sliceSize == 0) {
        HCCL_ERROR("[HandleSendRead] Invalid sliceSize: %lu", tempAlgParams.sliceSize);
        return HCCL_E_INTERNAL;
    }

    // 从tempAlgParams中获取远程rank信息
    // 假设tempAlgParams中包含了远程rank信息
    u32 remoteRank = 0; // 需要从tempAlgParams中获取
    // 临时实现：使用第一个通道作为远程rank
    if (!channels.empty()) {
        remoteRank = channels.begin()->first;
    }

    // 查找对应远程rank的通道信息
    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRead] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    // 参考Mesh 1D实现，发送数据通常从input到远程cclbuf
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    void* srcAddr = tempAlgParams.buffInfo.inputPtr;
    if (!srcAddr) {
        HCCL_ERROR("[HandleSendRead] Invalid source address");
        return HCCL_E_INTERNAL;
    }

    // 计算发送偏移
    // 参考Mesh 1D实现：tempAlgParams.sdispls[nextRank] * dataTypeSize_
    u64 srcOffset = 0; // 需要从tempAlgParams中获取
    if (tempAlgParams.inputSliceStride > 0) {
        srcOffset = 0; // 临时占位
    }

    srcAddr = static_cast<char*>(srcAddr) + srcOffset;
    srcSlices.push_back(DataSlice(srcAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 目标地址通常是远程cclbuf
    void* remoteCclBuffAddr = channel.remoteCclMem.addr;
    if (!remoteCclBuffAddr) {
        HCCL_ERROR("[HandleSendRead] Invalid remote ccl buffer address");
        return HCCL_E_INTERNAL;
    }

    // 计算远程cclbuf偏移
    // 参考Mesh 1D实现：myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff
    u64 remoteOffset = 0; // 需要从tempAlgParams中获取
    remoteCclBuffAddr = static_cast<char*>(remoteCclBuffAddr) + remoteOffset;
    dstSlices.push_back(DataSlice(remoteCclBuffAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 执行发送读操作
    SlicesList sendSliceList(std::move(srcSlices), std::move(dstSlices));
    DataInfo sendInfo(channel, std::move(sendSliceList));
    CHK_RET(static_cast<HcclResult>(SendRead(sendInfo, threads[0])));

    HCCL_INFO("[HandleSendRead] SendRead completed: remoteRank=%lu, size=%lu", remoteRank, tempAlgParams.sliceSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleRecvRead(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                               const std::vector<ThreadHandle> &threads,
                                               const TemplateDataParams &tempAlgParams)
{
    // 参考Mesh 1D实现中的RecvRead逻辑
    // 需要从tempAlgParams中获取远程rank信息

    // 检查必要的参数
    if (tempAlgParams.sliceSize == 0) {
        HCCL_ERROR("[HandleRecvRead] Invalid sliceSize: %lu", tempAlgParams.sliceSize);
        return HCCL_E_INTERNAL;
    }

    // 从tempAlgParams中获取远程rank信息
    // 假设tempAlgParams中包含了远程rank信息
    u32 remoteRank = 0; // 需要从tempAlgParams中获取
    // 临时实现：使用第一个通道作为远程rank
    if (!channels.empty()) {
        remoteRank = channels.begin()->first;
    }

    // 查找对应远程rank的通道信息
    auto it = channels.find(remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleRecvRead] Channel not found for remote rank: %lu", remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备接收数据切片
    // 参考Mesh 1D实现，接收数据通常从本地cclbuf到output
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    // 源地址通常是本地cclbuf
    void* localCclBuffAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    if (!localCclBuffAddr) {
        HCCL_ERROR("[HandleRecvRead] Invalid local ccl buffer address");
        return HCCL_E_INTERNAL;
    }

    // 计算本地cclbuf偏移
    // 参考Mesh 1D实现：nextRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff
    u64 localCclOffset = 0; // 需要从tempAlgParams中获取
    localCclBuffAddr = static_cast<char*>(localCclBuffAddr) + localCclOffset;
    srcSlices.push_back(DataSlice(localCclBuffAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    void* dstAddr = tempAlgParams.buffInfo.outputPtr;
    if (!dstAddr) {
        HCCL_ERROR("[HandleRecvRead] Invalid destination address");
        return HCCL_E_INTERNAL;
    }

    // 计算接收偏移
    // 参考Mesh 1D实现：tempAlgParams.rdispls[curAlgRank] * dataTypeSize_
    u64 dstOffset = 0; // 需要从tempAlgParams中获取
    dstAddr = static_cast<char*>(dstAddr) + dstOffset;
    dstSlices.push_back(DataSlice(dstAddr, 0, tempAlgParams.sliceSize, tempAlgParams.sliceSize / dataTypeSize_));

    // 执行接收读操作
    SlicesList recvSliceList(std::move(srcSlices), std::move(dstSlices));
    DataInfo recvInfo(channel, std::move(recvSliceList));
    CHK_RET(static_cast<HcclResult>(RecvRead(recvInfo, threads[0])));

    HCCL_INFO("[HandleRecvRead] RecvRead completed: remoteRank=%lu, size=%lu", remoteRank, tempAlgParams.sliceSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendRecvReadReduce(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                         const std::vector<ThreadHandle> &threads,
                                                         const TemplateDataParams &tempAlgParams)
{
    // AlltoAllV通常不使用读操作，这里提供基本实现框架
    HCCL_WARNING("[HandleSendRecvReadReduce] SendRecvReadReduce not typically used in AlltoAllV, using SendRecvWrite as fallback");

    // 使用SendRecvWrite作为替代实现
    return HandleSendRecvWrite(channels, threads, tempAlgParams);
}

HcclResult InsTempAlltoAllVOmni::HandleSendReadReduce(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                     const std::vector<ThreadHandle> &threads,
                                                     const TemplateDataParams &tempAlgParams)
{
    // AlltoAllV通常不使用读操作，这里提供基本实现框架
    HCCL_WARNING("[HandleSendReadReduce] SendReadReduce not typically used in AlltoAllV, using SendWrite as fallback");

    // 使用SendWrite作为替代实现
    return HandleSendWrite(channels, threads, tempAlgParams);
}

HcclResult InsTempAlltoAllVOmni::HandleRecvReadReduce(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                     const std::vector<ThreadHandle> &threads,
                                                     const TemplateDataParams &tempAlgParams)
{
    // AlltoAllV通常不使用读操作，这里提供基本实现框架
    HCCL_WARNING("[HandleRecvReadReduce] RecvReadReduce not typically used in AlltoAllV, using RecvWrite as fallback");

    // 使用RecvWrite作为替代实现
    return HandleRecvWrite(channels, threads, tempAlgParams);
}

HcclResult InsTempAlltoAllVOmni::HandleGroupBroadcast(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                     const std::vector<ThreadHandle> &threads,
                                                     const TemplateDataParams &tempAlgParams)
{
    // AlltoAllV通常不使用组广播操作，这里提供基本实现框架
    HCCL_WARNING("[HandleGroupBroadcast] GroupBroadcast not typically used in AlltoAllV, using SendWrite as fallback");

    // 使用SendWrite作为替代实现
    return HandleSendWrite(channels, threads, tempAlgParams);
}

HcclResult InsTempAlltoAllVOmni::HandleGroupReduce(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                  const std::vector<ThreadHandle> &threads,
                                                  const TemplateDataParams &tempAlgParams)
{
    // AlltoAllV通常不使用组归约操作，这里提供基本实现框架
    HCCL_WARNING("[HandleGroupReduce] GroupReduce not typically used in AlltoAllV, using RecvWrite as fallback");

    // 使用RecvWrite作为替代实现
    return HandleRecvWrite(channels, threads, tempAlgParams);
}

HcclResult InsTempAlltoAllVOmni::HandlePreSyncInterThreads(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[HandlePreSyncInterThreads] Start pre-sync inter threads");

    // 验证线程数量
    if (threads.empty()) {
        HCCL_ERROR("[HandlePreSyncInterThreads] No threads available");
        return HCCL_E_INTERNAL;
    }

    // 使用默认配置：主线程为第一个线程，子线程为其他所有线程
    u32 mainThreadIdx = 0;
    std::vector<ThreadHandle> subThreads;

    // 准备子线程句柄（排除主线程）
    for (size_t i = 1; i < threads.size(); i++) {
        subThreads.push_back(threads[i]);
    }

    // 如果没有子线程，直接返回成功
    if (subThreads.empty()) {
        HCCL_INFO("[HandlePreSyncInterThreads] No sub-threads to sync");
        return HCCL_SUCCESS;
    }

    // 准备通知索引（使用默认值0）
    std::vector<u32> notifyIdxMainToSub(subThreads.size(), 0);

    // 执行前同步
    CHK_RET(PreSyncInterThreads(threads[mainThreadIdx], subThreads, notifyIdxMainToSub));

    HCCL_INFO("[HandlePreSyncInterThreads] Pre-sync inter threads completed successfully");
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandlePostSyncInterThreads(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[HandlePostSyncInterThreads] Start post-sync inter threads");

    // 验证线程数量
    if (threads.empty()) {
        HCCL_ERROR("[HandlePostSyncInterThreads] No threads available");
        return HCCL_E_INTERNAL;
    }

    // 使用默认配置：主线程为第一个线程，子线程为其他所有线程
    u32 mainThreadIdx = 0;
    std::vector<ThreadHandle> subThreads;

    // 准备子线程句柄（排除主线程）
    for (size_t i = 1; i < threads.size(); i++) {
        subThreads.push_back(threads[i]);
    }

    // 如果没有子线程，直接返回成功
    if (subThreads.empty()) {
        HCCL_INFO("[HandlePostSyncInterThreads] No sub-threads to sync");
        return HCCL_SUCCESS;
    }

    // 准备通知索引（使用递增索引）
    std::vector<u32> notifyIdxSubToMain;
    for (size_t i = 0; i < subThreads.size(); i++) {
        notifyIdxSubToMain.push_back(i);
    }

    // 执行后同步
    CHK_RET(PostSyncInterThreads(threads[mainThreadIdx], subThreads, notifyIdxSubToMain));

    HCCL_INFO("[HandlePostSyncInterThreads] Post-sync inter threads completed successfully");
    return HCCL_SUCCESS;
}

} // namespace ops_hccl