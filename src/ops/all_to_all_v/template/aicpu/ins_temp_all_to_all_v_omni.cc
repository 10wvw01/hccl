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

    // 保存XML信息供后续使用
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

    HCCL_INFO("[InsTempAlltoAllVOmni] Run Start");

    // 检查XML信息是否已传递
    if (xmlInfo_.vecSendRecvInfo.empty() && xmlInfo_.vecSyncInfo.empty()) {
        HCCL_WARNING("[InsTempAlltoAllVOmni] XML information is empty, OMNI operations may not execute correctly");
    } else {
        HCCL_INFO("[InsTempAlltoAllVOmni] XML information contains %lu sync signals and %lu data signals",
                  xmlInfo_.vecSyncInfo.size(), xmlInfo_.vecSendRecvInfo.size());
    }

    // 多线程同步处理
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    // 执行OMNI算法
    CHK_RET(RunOmni(templateResource.channels, templateResource.threads, tempAlgParams));

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
    HCCL_INFO("[InsTempAlltoAllVOmni][DoRepeatOmni] Start processing OMNI signals");

    // 处理同步指令
    for (const auto& syncInfo : xmlInfo_.vecSyncInfo) {
        HcclResult ret = HCCL_SUCCESS;

        switch (syncInfo.optype) {
            case OP_PRE_SYNC_INTER_THREADS:
                ret = HandlePreSyncInterThreads(syncInfo, threads);
                break;
            case OP_POST_SYNC_INTER_THREADS:
                ret = HandlePostSyncInterThreads(syncInfo, threads);
                break;
            default:
                HCCL_ERROR("[DoRepeatOmni] Unsupported sync operation type: %d", syncInfo.optype);
                ret = HCCL_E_INTERNAL;
                break;
        }

        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[DoRepeatOmni] Failed to handle sync operation type: %d, error: 0x%016llx",
                      syncInfo.optype, HCCL_ERROR_CODE(ret));
            return;
        }
    }

    // 遍历XML中的所有信号信息
    for (const auto& signalInfo : xmlInfo_.vecSendRecvInfo) {
        HcclResult ret = HCCL_SUCCESS;

        switch (signalInfo.optype) {
            case OP_LOCAL_COPY:
                ret = HandleLocalCopy(signalInfo, threads, tempAlgParams);
                break;
            case OP_LOCAL_REDUCE:
                ret = HandleLocalReduce(signalInfo, threads, tempAlgParams);
                break;
            case OP_SEND_RECV_WRITE:
                ret = HandleSendRecvWrite(signalInfo, channels, threads, tempAlgParams);
                break;
            case OP_SEND_WRITE:
                ret = HandleSendWrite(signalInfo, channels, threads, tempAlgParams);
                break;
            case OP_RECV_WRITE:
                ret = HandleRecvWrite(signalInfo, channels, threads, tempAlgParams);
                break;
            case OP_SEND_RECV_WRITE_REDUCE:
                ret = HandleSendRecvWriteReduce(signalInfo, channels, threads, tempAlgParams);
                break;
            case OP_SEND_WRITE_REDUCE:
                ret = HandleSendWriteReduce(signalInfo, channels, threads, tempAlgParams);
                break;
            case OP_RECV_WRITE_REDUCE:
                ret = HandleRecvWriteReduce(signalInfo, channels, threads, tempAlgParams);
                break;
            case OP_SEND_RECV_READ:
                ret = HandleSendRecvRead(signalInfo, channels, threads, tempAlgParams);
                break;
            case OP_SEND_READ:
                ret = HandleSendRead(signalInfo, channels, threads, tempAlgParams);
                break;
            case OP_RECV_READ:
                ret = HandleRecvRead(signalInfo, channels, threads, tempAlgParams);
                break;
            case OP_SEND_RECV_READ_REDUCE:
                ret = HandleSendRecvReadReduce(signalInfo, channels, threads, tempAlgParams);
                break;
            case OP_SEND_READ_REDUCE:
                ret = HandleSendReadReduce(signalInfo, channels, threads, tempAlgParams);
                break;
            case OP_RECV_READ_REDUCE:
                ret = HandleRecvReadReduce(signalInfo, channels, threads, tempAlgParams);
                break;
            case OP_GROUP_BROAD_CAST:
                ret = HandleGroupBroadcast(signalInfo, channels, threads, tempAlgParams);
                break;
            case OP_GROUP_REDUCE:
                ret = HandleGroupReduce(signalInfo, channels, threads, tempAlgParams);
                break;
            default:
                HCCL_ERROR("[DoRepeatOmni] Unsupported operation type: %d", signalInfo.optype);
                ret = HCCL_E_INTERNAL;
                break;
        }

        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[DoRepeatOmni] Failed to handle operation type: %d, error: 0x%016llx",
                      signalInfo.optype, HCCL_ERROR_CODE(ret));
            return;
        }
    }

    HCCL_INFO("[InsTempAlltoAllVOmni][DoRepeatOmni] All OMNI signals processed successfully");
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

HcclResult InsTempAlltoAllVOmni::HandleLocalCopy(const OmniSendRecvInfo& signalInfo,
                                                const std::vector<ThreadHandle> &threads,
                                                const TemplateDataParams &tempAlgParams)
{
    if (signalInfo.srcSliceInfo.empty() || signalInfo.dstSliceInfo.empty()) {
        HCCL_ERROR("[HandleLocalCopy] Invalid slice info");
        return HCCL_E_INTERNAL;
    }

    // 计算源地址
    void* srcAddr = nullptr;
    const auto& srcSlice = signalInfo.srcSliceInfo[0];
    if (srcSlice.sliceType == 0) { // input
        srcAddr = tempAlgParams.buffInfo.inputPtr;
    } else if (srcSlice.sliceType == 1) { // output
        srcAddr = tempAlgParams.buffInfo.outputPtr;
    } else if (srcSlice.sliceType == 2) { // cclbuf
        srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    }

    if (!srcAddr) {
        HCCL_ERROR("[HandleLocalCopy] Invalid source address");
        return HCCL_E_INTERNAL;
    }

    srcAddr = static_cast<char*>(srcAddr) + srcSlice.sliceIdx * processSize_;

    // 计算目标地址
    void* dstAddr = nullptr;
    const auto& dstSlice = signalInfo.dstSliceInfo[0];
    if (dstSlice.sliceType == 0) { // input
        dstAddr = tempAlgParams.buffInfo.inputPtr;
    } else if (dstSlice.sliceType == 1) { // output
        dstAddr = tempAlgParams.buffInfo.outputPtr;
    } else if (dstSlice.sliceType == 2) { // cclbuf
        dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    }

    if (!dstAddr) {
        HCCL_ERROR("[HandleLocalCopy] Invalid destination address");
        return HCCL_E_INTERNAL;
    }

    dstAddr = static_cast<char*>(dstAddr) + dstSlice.sliceIdx * processSize_;

    // 执行本地拷贝
    DataSlice srcSliceObj(srcAddr, 0, processSize_, processSize_ / dataTypeSize_);
    DataSlice dstSliceObj(dstAddr, 0, processSize_, processSize_ / dataTypeSize_);

    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSliceObj, dstSliceObj)));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleLocalReduce(const OmniSendRecvInfo& signalInfo,
                                                  const std::vector<ThreadHandle> &threads,
                                                  const TemplateDataParams &tempAlgParams)
{
    if (signalInfo.srcSliceInfo.empty() || signalInfo.dstSliceInfo.empty()) {
        HCCL_ERROR("[HandleLocalReduce] Invalid slice info");
        return HCCL_E_INTERNAL;
    }

    // 计算目标地址
    void* dstAddr = nullptr;
    const auto& dstSlice = signalInfo.dstSliceInfo[0];
    if (dstSlice.sliceType == 0) { // input
        dstAddr = tempAlgParams.buffInfo.inputPtr;
    } else if (dstSlice.sliceType == 1) { // output
        dstAddr = tempAlgParams.buffInfo.outputPtr;
    } else if (dstSlice.sliceType == 2) { // cclbuf
        dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    }

    if (!dstAddr) {
        HCCL_ERROR("[HandleLocalReduce] Invalid destination address");
        return HCCL_E_INTERNAL;
    }

    dstAddr = static_cast<char*>(dstAddr) + dstSlice.sliceIdx * processSize_;
    DataSlice dstSliceObj(dstAddr, 0, processSize_, processSize_ / dataTypeSize_);

    // 处理所有源切片
    for (const auto& srcSlice : signalInfo.srcSliceInfo) {
        // 计算源地址
        void* srcAddr = nullptr;
        if (srcSlice.sliceType == 0) { // input
            srcAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (srcSlice.sliceType == 1) { // output
            srcAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (srcSlice.sliceType == 2) { // cclbuf
            srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!srcAddr) {
            HCCL_ERROR("[HandleLocalReduce] Invalid source address");
            return HCCL_E_INTERNAL;
        }

        srcAddr = static_cast<char*>(srcAddr) + srcSlice.sliceIdx * processSize_;
        DataSlice srcSliceObj(srcAddr, 0, processSize_, processSize_ / dataTypeSize_);

        // 执行本地归约操作
        CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], srcSliceObj, dstSliceObj,
                                                   signalInfo.inputDataType, signalInfo.reduceType)));
    }

    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendRecvWrite(const OmniSendRecvInfo& signalInfo,
                                                    const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                    const std::vector<ThreadHandle> &threads,
                                                    const TemplateDataParams &tempAlgParams)
{
    auto it = channels.find(signalInfo.remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRecvWrite] Channel not found for remote rank: %lu", signalInfo.remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;

    for (const auto& srcSlice : signalInfo.srcSliceInfo) {
        void* srcAddr = nullptr;
        if (srcSlice.sliceType == 0) { // input
            srcAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (srcSlice.sliceType == 1) { // output
            srcAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (srcSlice.sliceType == 2) { // cclbuf
            srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!srcAddr) {
            HCCL_ERROR("[HandleSendRecvWrite] Invalid source address");
            return HCCL_E_INTERNAL;
        }

        srcAddr = static_cast<char*>(srcAddr) + srcSlice.sliceIdx * processSize_;
        txSrcSlices.push_back(DataSlice(srcAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 准备接收数据切片
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    for (const auto& dstSlice : signalInfo.dstSliceInfo) {
        void* dstAddr = nullptr;
        if (dstSlice.sliceType == 0) { // input
            dstAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (dstSlice.sliceType == 1) { // output
            dstAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (dstSlice.sliceType == 2) { // cclbuf
            dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!dstAddr) {
            HCCL_ERROR("[HandleSendRecvWrite] Invalid destination address");
            return HCCL_E_INTERNAL;
        }

        dstAddr = static_cast<char*>(dstAddr) + dstSlice.sliceIdx * processSize_;
        rxDstSlices.push_back(DataSlice(dstAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 执行发送接收写操作
    TxRxChannels txRxChannels(channel, channel);
    TxRxSlicesList txRxSlicesList(
        SlicesList(std::move(txSrcSlices), std::move(txDstSlices)),
        SlicesList(std::move(rxSrcSlices), std::move(rxDstSlices))
    );
    SendRecvInfo sendRecvInfo(std::move(txRxChannels), std::move(txRxSlicesList));

    CHK_RET(static_cast<HcclResult>(SendRecvWrite(sendRecvInfo, threads[0])));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendWrite(const OmniSendRecvInfo& signalInfo,
                                                const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                const std::vector<ThreadHandle> &threads,
                                                const TemplateDataParams &tempAlgParams)
{
    auto it = channels.find(signalInfo.remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendWrite] Channel not found for remote rank: %lu", signalInfo.remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto& srcSlice : signalInfo.srcSliceInfo) {
        void* srcAddr = nullptr;
        if (srcSlice.sliceType == 0) { // input
            srcAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (srcSlice.sliceType == 1) { // output
            srcAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (srcSlice.sliceType == 2) { // cclbuf
            srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!srcAddr) {
            HCCL_ERROR("[HandleSendWrite] Invalid source address");
            return HCCL_E_INTERNAL;
        }

        srcAddr = static_cast<char*>(srcAddr) + srcSlice.sliceIdx * processSize_;
        srcSlices.push_back(DataSlice(srcAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 执行发送写操作
    SlicesList sendSliceList(std::move(srcSlices), std::move(dstSlices));
    DataInfo sendInfo(channel, std::move(sendSliceList));
    CHK_RET(static_cast<HcclResult>(SendWrite(sendInfo, threads[0])));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleRecvWrite(const OmniSendRecvInfo& signalInfo,
                                                const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                const std::vector<ThreadHandle> &threads,
                                                const TemplateDataParams &tempAlgParams)
{
    auto it = channels.find(signalInfo.remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleRecvWrite] Channel not found for remote rank: %lu", signalInfo.remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备接收数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto& dstSlice : signalInfo.dstSliceInfo) {
        void* dstAddr = nullptr;
        if (dstSlice.sliceType == 0) { // input
            dstAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (dstSlice.sliceType == 1) { // output
            dstAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (dstSlice.sliceType == 2) { // cclbuf
            dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!dstAddr) {
            HCCL_ERROR("[HandleRecvWrite] Invalid destination address");
            return HCCL_E_INTERNAL;
        }

        dstAddr = static_cast<char*>(dstAddr) + dstSlice.sliceIdx * processSize_;
        dstSlices.push_back(DataSlice(dstAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 执行接收写操作
    SlicesList recvSliceList(std::move(srcSlices), std::move(dstSlices));
    DataInfo recvInfo(channel, std::move(recvSliceList));
    CHK_RET(static_cast<HcclResult>(RecvWrite(recvInfo, threads[0])));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendRecvWriteReduce(const OmniSendRecvInfo& signalInfo,
                                                          const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                          const std::vector<ThreadHandle> &threads,
                                                          const TemplateDataParams &tempAlgParams)
{
    auto it = channels.find(signalInfo.remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRecvWriteReduce] Channel not found for remote rank: %lu", signalInfo.remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    std::vector<DataSlice> txSrcSlices; 
    std::vector<DataSlice> txDstSlices;

    for (const auto& srcSlice : signalInfo.srcSliceInfo) {
        void* srcAddr = nullptr;
        if (srcSlice.sliceType == 0) { // input
            srcAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (srcSlice.sliceType == 1) { // output
            srcAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (srcSlice.sliceType == 2) { // cclbuf
            srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!srcAddr) {
            HCCL_ERROR("[HandleSendRecvWriteReduce] Invalid source address");
            return HCCL_E_INTERNAL;
        }

        srcAddr = static_cast<char*>(srcAddr) + srcSlice.sliceIdx * processSize_;
        txSrcSlices.push_back(DataSlice(srcAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 准备接收数据切片
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    for (const auto& dstSlice : signalInfo.dstSliceInfo) {
        void* dstAddr = nullptr;
        if (dstSlice.sliceType == 0) { // input
            dstAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (dstSlice.sliceType == 1) { // output
            dstAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (dstSlice.sliceType == 2) { // cclbuf
            dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!dstAddr) {
            HCCL_ERROR("[HandleSendRecvWriteReduce] Invalid destination address");
            return HCCL_E_INTERNAL;
        }

        dstAddr = static_cast<char*>(dstAddr) + dstSlice.sliceIdx * processSize_;
        rxDstSlices.push_back(DataSlice(dstAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 执行发送接收写归约操作
    TxRxChannels txRxChannels(channel, channel);
    TxRxSlicesList txRxSlicesList(
        SlicesList(std::move(txSrcSlices), std::move(txDstSlices)),
        SlicesList(std::move(rxSrcSlices), std::move(rxDstSlices))
    );
    SendRecvReduceInfo sendRecvInfo(std::move(txRxChannels), std::move(txRxSlicesList),
                                   signalInfo.inputDataType, signalInfo.reduceType);

    CHK_RET(static_cast<HcclResult>(SendRecvWriteReduce(sendRecvInfo, threads[0])));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendWriteReduce(const OmniSendRecvInfo& signalInfo,
                                                      const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                      const std::vector<ThreadHandle> &threads,
                                                      const TemplateDataParams &tempAlgParams)
{
    auto it = channels.find(signalInfo.remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendWriteReduce] Channel not found for remote rank: %lu", signalInfo.remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto& srcSlice : signalInfo.srcSliceInfo) {
        void* srcAddr = nullptr;
        if (srcSlice.sliceType == 0) { // input
            srcAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (srcSlice.sliceType == 1) { // output
            srcAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (srcSlice.sliceType == 2) { // cclbuf
            srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!srcAddr) {
            HCCL_ERROR("[HandleSendWriteReduce] Invalid source address");
            return HCCL_E_INTERNAL;
        }

        srcAddr = static_cast<char*>(srcAddr) + srcSlice.sliceIdx * processSize_;
        srcSlices.push_back(DataSlice(srcAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 执行发送写归约操作
    SlicesList sendSliceList(std::move(srcSlices), std::move(dstSlices));
    DataReduceInfo sendInfo(channel, std::move(sendSliceList), signalInfo.inputDataType, signalInfo.reduceType);
    CHK_RET(static_cast<HcclResult>(SendWriteReduce(sendInfo, threads[0])));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleRecvWriteReduce(const OmniSendRecvInfo& signalInfo,
                                                      const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                      const std::vector<ThreadHandle> &threads,
                                                      const TemplateDataParams &tempAlgParams)
{
    auto it = channels.find(signalInfo.remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleRecvWriteReduce] Channel not found for remote rank: %lu", signalInfo.remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备接收数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto& dstSlice : signalInfo.dstSliceInfo) {
        void* dstAddr = nullptr;
        if (dstSlice.sliceType == 0) { // input
            dstAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (dstSlice.sliceType == 1) { // output
            dstAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (dstSlice.sliceType == 2) { // cclbuf
            dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!dstAddr) {
            HCCL_ERROR("[HandleRecvWriteReduce] Invalid destination address");
            return HCCL_E_INTERNAL;
        }

        dstAddr = static_cast<char*>(dstAddr) + dstSlice.sliceIdx * processSize_;
        dstSlices.push_back(DataSlice(dstAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 执行接收写归约操作
    SlicesList recvSliceList(std::move(srcSlices), std::move(dstSlices));
    DataReduceInfo recvInfo(channel, std::move(recvSliceList), signalInfo.inputDataType, signalInfo.reduceType);
    CHK_RET(static_cast<HcclResult>(RecvWriteReduce(recvInfo, threads[0])));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendRecvRead(const OmniSendRecvInfo& signalInfo,
                                                   const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                   const std::vector<ThreadHandle> &threads,
                                                   const TemplateDataParams &tempAlgParams)
{
    auto it = channels.find(signalInfo.remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRecvRead] Channel not found for remote rank: %lu", signalInfo.remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;

    for (const auto& srcSlice : signalInfo.srcSliceInfo) {
        void* srcAddr = nullptr;
        if (srcSlice.sliceType == 0) { // input
            srcAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (srcSlice.sliceType == 1) { // output
            srcAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (srcSlice.sliceType == 2) { // cclbuf
            srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!srcAddr) {
            HCCL_ERROR("[HandleSendRecvRead] Invalid source address");
            return HCCL_E_INTERNAL;
        }

        srcAddr = static_cast<char*>(srcAddr) + srcSlice.sliceIdx * processSize_;
        txSrcSlices.push_back(DataSlice(srcAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 准备接收数据切片
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    for (const auto& dstSlice : signalInfo.dstSliceInfo) {
        void* dstAddr = nullptr;
        if (dstSlice.sliceType == 0) { // input
            dstAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (dstSlice.sliceType == 1) { // output
            dstAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (dstSlice.sliceType == 2) { // cclbuf
            dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!dstAddr) {
            HCCL_ERROR("[HandleSendRecvRead] Invalid destination address");
            return HCCL_E_INTERNAL;
        }

        dstAddr = static_cast<char*>(dstAddr) + dstSlice.sliceIdx * processSize_;
        rxDstSlices.push_back(DataSlice(dstAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 执行发送接收读操作
    TxRxChannels txRxChannels(channel, channel);
    TxRxSlicesList txRxSlicesList(
        SlicesList(std::move(txSrcSlices), std::move(txDstSlices)),
        SlicesList(std::move(rxSrcSlices), std::move(rxDstSlices))
    );
    SendRecvInfo sendRecvInfo(std::move(txRxChannels), std::move(txRxSlicesList));

    CHK_RET(static_cast<HcclResult>(SendRecvRead(sendRecvInfo, threads[0])));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendRead(const OmniSendRecvInfo& signalInfo,
                                               const std::map<u32, std::vector<ChannelInfo>> &channels,
                                               const std::vector<ThreadHandle> &threads,
                                               const TemplateDataParams &tempAlgParams)
{
    auto it = channels.find(signalInfo.remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRead] Channel not found for remote rank: %lu", signalInfo.remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto& srcSlice : signalInfo.srcSliceInfo) {
        void* srcAddr = nullptr;
        if (srcSlice.sliceType == 0) { // input
            srcAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (srcSlice.sliceType == 1) { // output
            srcAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (srcSlice.sliceType == 2) { // cclbuf
            srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!srcAddr) {
            HCCL_ERROR("[HandleSendRead] Invalid source address");
            return HCCL_E_INTERNAL;
        }

        srcAddr = static_cast<char*>(srcAddr) + srcSlice.sliceIdx * processSize_;
        srcSlices.push_back(DataSlice(srcAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 执行发送读操作
    SlicesList sendSliceList(std::move(srcSlices), std::move(dstSlices));
    DataInfo sendInfo(channel, std::move(sendSliceList));
    CHK_RET(static_cast<HcclResult>(SendRead(sendInfo, threads[0])));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleRecvRead(const OmniSendRecvInfo& signalInfo,
                                               const std::map<u32, std::vector<ChannelInfo>> &channels,
                                               const std::vector<ThreadHandle> &threads,
                                               const TemplateDataParams &tempAlgParams)
{
    auto it = channels.find(signalInfo.remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleRecvRead] Channel not found for remote rank: %lu", signalInfo.remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备接收数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto& dstSlice : signalInfo.dstSliceInfo) {
        void* dstAddr = nullptr;
        if (dstSlice.sliceType == 0) { // input
            dstAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (dstSlice.sliceType == 1) { // output
            dstAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (dstSlice.sliceType == 2) { // cclbuf
            dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!dstAddr) {
            HCCL_ERROR("[HandleRecvRead] Invalid destination address");
            return HCCL_E_INTERNAL;
        }

        dstAddr = static_cast<char*>(dstAddr) + dstSlice.sliceIdx * processSize_;
        dstSlices.push_back(DataSlice(dstAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 执行接收读操作
    SlicesList recvSliceList(std::move(srcSlices), std::move(dstSlices));
    DataInfo recvInfo(channel, std::move(recvSliceList));
    CHK_RET(static_cast<HcclResult>(RecvRead(recvInfo, threads[0])));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendRecvReadReduce(const OmniSendRecvInfo& signalInfo,
                                                         const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                         const std::vector<ThreadHandle> &threads,
                                                         const TemplateDataParams &tempAlgParams)
{
    auto it = channels.find(signalInfo.remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendRecvReadReduce] Channel not found for remote rank: %lu", signalInfo.remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;

    for (const auto& srcSlice : signalInfo.srcSliceInfo) {
        void* srcAddr = nullptr;
        if (srcSlice.sliceType == 0) { // input
            srcAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (srcSlice.sliceType == 1) { // output
            srcAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (srcSlice.sliceType == 2) { // cclbuf
            srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!srcAddr) {
            HCCL_ERROR("[HandleSendRecvReadReduce] Invalid source address");
            return HCCL_E_INTERNAL;
        }

        srcAddr = static_cast<char*>(srcAddr) + srcSlice.sliceIdx * processSize_;
        txSrcSlices.push_back(DataSlice(srcAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 准备接收数据切片
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;

    for (const auto& dstSlice : signalInfo.dstSliceInfo) {
        void* dstAddr = nullptr;
        if (dstSlice.sliceType == 0) { // input
            dstAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (dstSlice.sliceType == 1) { // output
            dstAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (dstSlice.sliceType == 2) { // cclbuf
            dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!dstAddr) {
            HCCL_ERROR("[HandleSendRecvReadReduce] Invalid destination address");
            return HCCL_E_INTERNAL;
        }

        dstAddr = static_cast<char*>(dstAddr) + dstSlice.sliceIdx * processSize_;
        rxDstSlices.push_back(DataSlice(dstAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 执行发送接收读归约操作
    TxRxChannels txRxChannels(channel, channel);
    TxRxSlicesList txRxSlicesList(
        SlicesList(std::move(txSrcSlices), std::move(txDstSlices)),
        SlicesList(std::move(rxSrcSlices), std::move(rxDstSlices))
    );
    SendRecvReduceInfo sendRecvInfo(std::move(txRxChannels), std::move(txRxSlicesList),
                                   signalInfo.inputDataType, signalInfo.reduceType);

    CHK_RET(static_cast<HcclResult>(SendRecvReadReduce(sendRecvInfo, threads[0])));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleSendReadReduce(const OmniSendRecvInfo& signalInfo,
                                                     const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                     const std::vector<ThreadHandle> &threads,
                                                     const TemplateDataParams &tempAlgParams)
{
    auto it = channels.find(signalInfo.remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleSendReadReduce] Channel not found for remote rank: %lu", signalInfo.remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备发送数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto& srcSlice : signalInfo.srcSliceInfo) {
        void* srcAddr = nullptr;
        if (srcSlice.sliceType == 0) { // input
            srcAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (srcSlice.sliceType == 1) { // output
            srcAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (srcSlice.sliceType == 2) { // cclbuf
            srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!srcAddr) {
            HCCL_ERROR("[HandleSendReadReduce] Invalid source address");
            return HCCL_E_INTERNAL;
        }

        srcAddr = static_cast<char*>(srcAddr) + srcSlice.sliceIdx * processSize_;
        srcSlices.push_back(DataSlice(srcAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 执行发送读归约操作
    SlicesList sendSliceList(std::move(srcSlices), std::move(dstSlices));
    DataReduceInfo sendInfo(channel, std::move(sendSliceList), signalInfo.inputDataType, signalInfo.reduceType);
    CHK_RET(static_cast<HcclResult>(SendReadReduce(sendInfo, threads[0])));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleRecvReadReduce(const OmniSendRecvInfo& signalInfo,
                                                     const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                     const std::vector<ThreadHandle> &threads,
                                                     const TemplateDataParams &tempAlgParams)
{
    auto it = channels.find(signalInfo.remoteRank);
    if (it == channels.end() || it->second.empty()) {
        HCCL_ERROR("[HandleRecvReadReduce] Channel not found for remote rank: %lu", signalInfo.remoteRank);
        return HCCL_E_INTERNAL;
    }

    const ChannelInfo& channel = it->second[0];

    // 准备接收数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto& dstSlice : signalInfo.dstSliceInfo) {
        void* dstAddr = nullptr;
        if (dstSlice.sliceType == 0) { // input
            dstAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (dstSlice.sliceType == 1) { // output
            dstAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (dstSlice.sliceType == 2) { // cclbuf
            dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!dstAddr) {
            HCCL_ERROR("[HandleRecvReadReduce] Invalid destination address");
            return HCCL_E_INTERNAL;
        }

        dstAddr = static_cast<char*>(dstAddr) + dstSlice.sliceIdx * processSize_;
        dstSlices.push_back(DataSlice(dstAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 执行接收读归约操作
    SlicesList recvSliceList(std::move(srcSlices), std::move(dstSlices));
    DataReduceInfo recvInfo(channel, std::move(recvSliceList), signalInfo.inputDataType, signalInfo.reduceType);
    CHK_RET(static_cast<HcclResult>(RecvReadReduce(recvInfo, threads[0])));
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandleGroupBroadcast(const OmniSendRecvInfo& signalInfo,
                                                     const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                     const std::vector<ThreadHandle> &threads,
                                                     const TemplateDataParams &tempAlgParams)
{
    // 准备源数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto& srcSlice : signalInfo.srcSliceInfo) {
        void* srcAddr = nullptr;
        if (srcSlice.sliceType == 0) { // input
            srcAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (srcSlice.sliceType == 1) { // output
            srcAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (srcSlice.sliceType == 2) { // cclbuf
            srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!srcAddr) {
            HCCL_ERROR("[HandleGroupBroadcast] Invalid source address");
            return HCCL_E_INTERNAL;
        }

        srcAddr = static_cast<char*>(srcAddr) + srcSlice.sliceIdx * processSize_;
        srcSlices.push_back(DataSlice(srcAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 准备目标数据切片
    std::vector<ChannelInfo> channelInfos;
    for (const auto& dstSlice : signalInfo.dstSliceInfo) {
        void* dstAddr = nullptr;
        if (dstSlice.sliceType == 0) { // input
            dstAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (dstSlice.sliceType == 1) { // output
            dstAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (dstSlice.sliceType == 2) { // cclbuf
            dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!dstAddr) {
            HCCL_ERROR("[HandleGroupBroadcast] Invalid destination address");
            return HCCL_E_INTERNAL;
        }

        dstAddr = static_cast<char*>(dstAddr) + dstSlice.sliceIdx * processSize_;
        dstSlices.push_back(DataSlice(dstAddr, 0, processSize_, processSize_ / dataTypeSize_));

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

HcclResult InsTempAlltoAllVOmni::HandleGroupReduce(const OmniSendRecvInfo& signalInfo,
                                                  const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                  const std::vector<ThreadHandle> &threads,
                                                  const TemplateDataParams &tempAlgParams)
{
    // 准备源数据切片
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;

    for (const auto& srcSlice : signalInfo.srcSliceInfo) {
        void* srcAddr = nullptr;
        if (srcSlice.sliceType == 0) { // input
            srcAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (srcSlice.sliceType == 1) { // output
            srcAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (srcSlice.sliceType == 2) { // cclbuf
            srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!srcAddr) {
            HCCL_ERROR("[HandleGroupReduce] Invalid source address");
            return HCCL_E_INTERNAL;
        }

        srcAddr = static_cast<char*>(srcAddr) + srcSlice.sliceIdx * processSize_;
        srcSlices.push_back(DataSlice(srcAddr, 0, processSize_, processSize_ / dataTypeSize_));
    }

    // 准备目标数据切片
    std::vector<ChannelInfo> channelInfos;
    for (const auto& dstSlice : signalInfo.dstSliceInfo) {
        void* dstAddr = nullptr;
        if (dstSlice.sliceType == 0) { // input
            dstAddr = tempAlgParams.buffInfo.inputPtr;
        } else if (dstSlice.sliceType == 1) { // output
            dstAddr = tempAlgParams.buffInfo.outputPtr;
        } else if (dstSlice.sliceType == 2) { // cclbuf
            dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        }

        if (!dstAddr) {
            HCCL_ERROR("[HandleGroupReduce] Invalid destination address");
            return HCCL_E_INTERNAL;
        }

        dstAddr = static_cast<char*>(dstAddr) + dstSlice.sliceIdx * processSize_;
        dstSlices.push_back(DataSlice(dstAddr, 0, processSize_, processSize_ / dataTypeSize_));

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
        DataReduceInfo recvInfo(channelInfos[i], std::move(recvSliceList),
                               signalInfo.inputDataType, signalInfo.reduceType);
        CHK_RET(static_cast<HcclResult>(RecvWriteReduce(recvInfo, threads[0])));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandlePreSyncInterThreads(const OmniSyncInfo& syncInfo,
                                                          const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[HandlePreSyncInterThreads] Start pre-sync inter threads, mainThreadIdx: %lu, subThreadNum: %lu",
              syncInfo.mainThreadIdx, syncInfo.subThreadNum);

    // 验证主线程索引
    if (syncInfo.mainThreadIdx >= threads.size()) {
        HCCL_ERROR("[HandlePreSyncInterThreads] Invalid mainThreadIdx: %lu, threads size: %lu",
                  syncInfo.mainThreadIdx, threads.size());
        return HCCL_E_INTERNAL;
    }

    // 验证子线程数量
    if (syncInfo.subThreadNum == 0 || syncInfo.subThreadNum >= threads.size()) {
        HCCL_ERROR("[HandlePreSyncInterThreads] Invalid subThreadNum: %lu, threads size: %lu",
                  syncInfo.subThreadNum, threads.size());
        return HCCL_E_INTERNAL;
    }

    // 准备子线程句柄
    std::vector<ThreadHandle> subThreads;
    if (syncInfo.subThreadIds.empty()) {
        // 如果没有指定子线程ID，使用默认顺序（排除主线程）
        for (size_t i = 0; i < threads.size(); i++) {
            if (i != syncInfo.mainThreadIdx) {
                subThreads.push_back(threads[i]);
            }
        }
        // 确保子线程数量匹配
        if (subThreads.size() != syncInfo.subThreadNum) {
            HCCL_WARNING("[HandlePreSyncInterThreads] subThreadNum mismatch: expected %lu, got %lu",
                        syncInfo.subThreadNum, subThreads.size());
        }
    } else {
        // 使用指定的子线程ID
        for (const auto& threadId : syncInfo.subThreadIds) {
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
    CHK_RET(PreSyncInterThreads(threads[syncInfo.mainThreadIdx], subThreads, notifyIdxMainToSub));

    HCCL_INFO("[HandlePreSyncInterThreads] Pre-sync inter threads completed successfully");
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVOmni::HandlePostSyncInterThreads(const OmniSyncInfo& syncInfo,
                                                           const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[HandlePostSyncInterThreads] Start post-sync inter threads, mainThreadIdx: %lu, subThreadNum: %lu",
              syncInfo.mainThreadIdx, syncInfo.subThreadNum);

    // 验证主线程索引
    if (syncInfo.mainThreadIdx >= threads.size()) {
        HCCL_ERROR("[HandlePostSyncInterThreads] Invalid mainThreadIdx: %lu, threads size: %lu",
                  syncInfo.mainThreadIdx, threads.size());
        return HCCL_E_INTERNAL;
    }

    // 验证子线程数量
    if (syncInfo.subThreadNum == 0 || syncInfo.subThreadNum >= threads.size()) {
        HCCL_ERROR("[HandlePostSyncInterThreads] Invalid subThreadNum: %lu, threads size: %lu",
                  syncInfo.subThreadNum, threads.size());
        return HCCL_E_INTERNAL;
    }

    // 准备子线程句柄
    std::vector<ThreadHandle> subThreads;
    if (syncInfo.subThreadIds.empty()) {
        // 如果没有指定子线程ID，使用默认顺序（排除主线程）
        for (size_t i = 0; i < threads.size(); i++) {
            if (i != syncInfo.mainThreadIdx) {
                subThreads.push_back(threads[i]);
            }
        }
        // 确保子线程数量匹配
        if (subThreads.size() != syncInfo.subThreadNum) {
            HCCL_WARNING("[HandlePostSyncInterThreads] subThreadNum mismatch: expected %lu, got %lu",
                        syncInfo.subThreadNum, subThreads.size());
        }
    } else {
        // 使用指定的子线程ID
        for (const auto& threadId : syncInfo.subThreadIds) {
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
    CHK_RET(PostSyncInterThreads(threads[syncInfo.mainThreadIdx], subThreads, notifyIdxSubToMain));

    HCCL_INFO("[HandlePostSyncInterThreads] Post-sync inter threads completed successfully");
    return HCCL_SUCCESS;
}

} // namespace ops_hccl