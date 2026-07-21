/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/ins_temp_all_to_all_v_bsp.h"
#include <algorithm>

namespace ops_hccl {
namespace {
constexpr u32 BSP_SEND_THREAD_BASE = 1;
constexpr u32 BSP_NOTIFY_NUM_PER_THREAD = 1;
constexpr u32 NET_NUM = 2;
}

InsTempAlltoAllVBsp::InsTempAlltoAllVBsp(const OpParam &param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempAlltoAllVBsp::~InsTempAlltoAllVBsp()
{
}

std::string InsTempAlltoAllVBsp::Describe() const
{
    return "Template of alltoallv BSP UBX with topology-derived asymmetric tx/rx";
}

HcclResult InsTempAlltoAllVBsp::NormalizeSubCommRanks(const TopoInfoWithNetLayerDetails *topoInfo)
{
    CHK_PTR_NULL(topoInfo);
    if (subCommRanks_.size() == NET_NUM) {
        const u32 rowNum = static_cast<u32>(subCommRanks_[0].size());
        subCommRanks_ = {subCommRanks_[1]};
        templateRankSize_ = subCommRanks_[0].size();
        CHK_RET(InitBspShape(rowNum, templateRankSize_));
    } else {
        CHK_PRT_RET(subCommRanks_.empty() || subCommRanks_[0].empty(),
                    HCCL_ERROR("[InsTempAlltoAllVBsp][Normalize] invalid subCommRanks."),
                    HCCL_E_PARA);
        templateRankSize_ = subCommRanks_[0].size();
        CHK_RET(InitBspShape(templateRankSize_, templateRankSize_));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVBsp::InitBspShape(u32 rowNum, u32 rankNum)
{
    rowNum_ = rowNum;
    rankNum_ = rankNum;
    CHK_PRT_RET(rowNum_ == 0 || rankNum_ == 0,
                HCCL_ERROR("[InsTempAlltoAllVBsp][InitShape] invalid shape rowNum[%u] rankNum[%u].",
                           rowNum_, rankNum_),
                HCCL_E_PARA);
    CHK_PRT_RET(rankNum_ % rowNum_ != 0,
                HCCL_ERROR("[InsTempAlltoAllVBsp][InitShape] rankNum[%u] is not divisible by rowNum[%u].",
                           rankNum_, rowNum_),
                HCCL_E_NOT_SUPPORT);
    colNum_ = rankNum_ / rowNum_;
    HCCL_INFO("[InsTempAlltoAllVBsp][InitShape] rowNum[%u] colNum[%u] rankNum[%u].",
              rowNum_, colNum_, rankNum_);
    return CheckBspShape();
}

HcclResult InsTempAlltoAllVBsp::CheckBspShape() const
{
    CHK_PRT_RET(rowNum_ == 0 || colNum_ == 0 || rankNum_ == 0 || rowNum_ * colNum_ != rankNum_,
                HCCL_ERROR("[InsTempAlltoAllVBsp][CheckShape] invalid BSP shape rowNum[%u] colNum[%u] "
                           "rankNum[%u].", rowNum_, colNum_, rankNum_),
                HCCL_E_PARA);
    CHK_PRT_RET(templateRankSize_ != rankNum_,
                HCCL_ERROR("[InsTempAlltoAllVBsp][CheckShape] templateRankSize[%u] does not match "
                           "rankNum[%u].", templateRankSize_, rankNum_),
                HCCL_E_PARA);
    CHK_PRT_RET(myRank_ >= rankNum_,
                HCCL_ERROR("[InsTempAlltoAllVBsp][CheckShape] myRank[%u] out of BSP rank range[%u].",
                           myRank_, rankNum_),
                HCCL_E_PARA);
    return HCCL_SUCCESS;
}

u32 InsTempAlltoAllVBsp::GetRowNum() const
{
    return rowNum_;
}

u32 InsTempAlltoAllVBsp::GetColNum() const
{
    return colNum_;
}

u32 InsTempAlltoAllVBsp::GetBspThreadNum() const
{
    return GetRowNum() * 2;
}

u32 InsTempAlltoAllVBsp::SelectPlane(u32 deltaC, u32 deltaR) const
{
    (void)deltaC;
    return deltaR;
}

HcclResult InsTempAlltoAllVBsp::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    CHK_RET(NormalizeSubCommRanks(topoInfo));

    std::vector<HcclChannelDesc> level0Channels;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, myChannelDescs,
                                                         CommTopo::COMM_TOPO_1DMESH));
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                level0Channels.push_back(channel);
            }
        }
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }

    resourceRequest.channels.push_back(level0Channels);
    const u32 bspThreadNum = GetBspThreadNum();
    resourceRequest.slaveThreadNum = bspThreadNum;
    resourceRequest.notifyNumOnMainThread = bspThreadNum;
    for (u32 idx = 0; idx < bspThreadNum; idx++) {
        resourceRequest.notifyNumPerThread.push_back(BSP_NOTIFY_NUM_PER_THREAD);
    }
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllVBsp::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

void InsTempAlltoAllVBsp::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    for (u32 idx = 0; idx < GetBspThreadNum(); idx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAlltoAllVBsp::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    for (u32 idx = 0; idx < GetBspThreadNum(); idx++) {
        notifyIdxSubToMain.push_back(idx);
    }
}

HcclResult InsTempAlltoAllVBsp::KernelRun(const OpParam &param,
    const TemplateDataParams &tempAlgParams, TemplateResource &templateResource)
{
    HCCL_INFO("[InsTempAlltoAllVBsp][KernelRun] start.");
    threadNum_ = templateResource.threads.size();
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = SIZE_TABLE[dataType_];
    isDmaRead_ = IsPcieProtocol(templateResource.channels);
    if (subCommRanks_.size() == NET_NUM) {
        const u32 rowNum = static_cast<u32>(subCommRanks_[0].size());
        subCommRanks_ = {subCommRanks_[1]};
        templateRankSize_ = subCommRanks_[0].size();
        CHK_RET(InitBspShape(rowNum, templateRankSize_));
    }
    CHK_PRT_RET(isDmaRead_,
                HCCL_ERROR("[InsTempAlltoAllVBsp][KernelRun] pcie/read protocol is not supported."),
                HCCL_E_NOT_SUPPORT);
    CHK_RET(CheckBspShape());
    const u32 bspThreadNum = GetBspThreadNum();
    CHK_PRT_RET(threadNum_ < bspThreadNum + 1,
                HCCL_ERROR("[InsTempAlltoAllVBsp][KernelRun] threadNum[%u] is less than required[%u].",
                           threadNum_, bspThreadNum + 1),
                HCCL_E_PARA);

    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1,
                                         templateResource.threads.begin() + bspThreadNum + 1);
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));

    CHK_RET(LocalCopyForMyRank(tempAlgParams, templateResource.threads[0]));

    for (u32 deltaC = 1; deltaC < GetColNum(); deltaC++) {
        std::vector<BspSlot> slotPlans;
        CHK_RET(CalcBspRoundPlan(deltaC, slotPlans));
        for (const auto &slot : slotPlans) {
            const ThreadHandle &sendThread = templateResource.threads[BSP_SEND_THREAD_BASE + slot.plane];
            const ThreadHandle &recvThread = templateResource.threads[BSP_SEND_THREAD_BASE + GetRowNum() + slot.plane];
            CHK_RET(RunBspSlot(tempAlgParams, templateResource.channels, slot, sendThread, recvThread));
        }
    }

    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    HCCL_INFO("[InsTempAlltoAllVBsp][KernelRun] end.");
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVBsp::LocalCopyForMyRank(
    const TemplateDataParams &tempAlgParams, const ThreadHandle &thread) const
{
    CHK_PRT_RET(myRank_ >= tempAlgParams.sendCounts.size() || myRank_ >= tempAlgParams.recvCounts.size() ||
                    myRank_ >= tempAlgParams.sdispls.size() || myRank_ >= tempAlgParams.rdispls.size(),
                HCCL_ERROR("[InsTempAlltoAllVBsp][LocalCopy] local A2AV info missing. myRank[%u] "
                           "sendCounts[%zu] recvCounts[%zu] sdispls[%zu] rdispls[%zu].",
                           myRank_, tempAlgParams.sendCounts.size(), tempAlgParams.recvCounts.size(),
                           tempAlgParams.sdispls.size(), tempAlgParams.rdispls.size()),
                HCCL_E_INTERNAL);
    u64 sendSize = tempAlgParams.sendCounts[myRank_] * dataTypeSize_;
    u64 recvSize = tempAlgParams.recvCounts[myRank_] * dataTypeSize_;
    u64 copySize = std::min(sendSize, recvSize);
    u64 copyCount = std::min(tempAlgParams.sendCounts[myRank_], tempAlgParams.recvCounts[myRank_]);
    if (copySize == 0) {
        return HCCL_SUCCESS;
    }
    DataSlice srcSlice(tempAlgParams.buffInfo.inputPtr,
        tempAlgParams.sdispls[myRank_] * dataTypeSize_, copySize, copyCount);
    DataSlice dstSlice(tempAlgParams.buffInfo.outputPtr,
        tempAlgParams.rdispls[myRank_] * dataTypeSize_, copySize, copyCount);
    return static_cast<HcclResult>(LocalCopy(thread, srcSlice, dstSlice));
}

HcclResult InsTempAlltoAllVBsp::CalcBspRoundPlan(u32 deltaC, std::vector<BspSlot> &slotPlans) const
{
    CHK_RET(CheckBspShape());
    CHK_PRT_RET(deltaC == 0 || deltaC >= GetColNum(),
                HCCL_ERROR("[InsTempAlltoAllVBsp][CalcPlan] invalid deltaC[%u].", deltaC),
                HCCL_E_PARA);
    slotPlans.clear();
    u32 myCol = myRank_ / GetRowNum();
    u32 myRow = myRank_ % GetRowNum();
    for (u32 deltaR = 0; deltaR < GetRowNum(); deltaR++) {
        u32 txCol = (myCol + deltaC) % GetColNum();
        u32 txRow = (myRow + deltaR) % GetRowNum();
        u32 rxCol = (myCol + GetColNum() - deltaC) % GetColNum();
        u32 rxRow = (myRow + GetRowNum() - deltaR) % GetRowNum();
        BspSlot slot;
        slot.txRank = txCol * GetRowNum() + txRow;
        slot.rxRank = rxCol * GetRowNum() + rxRow;
        slot.deltaC = deltaC;
        slot.deltaR = deltaR;
        slot.plane = SelectPlane(deltaC, deltaR);
        slotPlans.push_back(slot);
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVBsp::SelectBspChannel(
    const std::map<u32, std::vector<ChannelInfo>> &channels, u32 remoteRank, u32 plane, ChannelInfo &channel) const
{
    auto iter = channels.find(remoteRank);
    CHK_PRT_RET(iter == channels.end(),
                HCCL_ERROR("[InsTempAlltoAllVBsp][SelectChannel] remoteRank[%u] not found.", remoteRank),
                HCCL_E_PARA);
    CHK_PRT_RET(plane >= GetRowNum() || iter->second.size() < GetRowNum(),
                HCCL_ERROR("[InsTempAlltoAllVBsp][SelectChannel] invalid plane/channelNum. plane[%u] "
                           "channelNum[%zu] rowNum[%u] remoteRank[%u].",
                           plane, iter->second.size(), GetRowNum(), remoteRank),
                HCCL_E_PARA);
    channel = iter->second[plane];
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllVBsp::RunBspSlot(const TemplateDataParams &tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const BspSlot &slot,
    const ThreadHandle &sendThread, const ThreadHandle &recvThread) const
{
    ChannelInfo txChannel;
    ChannelInfo rxChannel;
    CHK_RET(SelectBspChannel(channels, slot.txRank, slot.plane, txChannel));
    CHK_RET(SelectBspChannel(channels, slot.rxRank, slot.plane, rxChannel));

    u64 txSize = tempAlgParams.sendCounts[slot.txRank] * dataTypeSize_;
    u64 txCount = tempAlgParams.sendCounts[slot.txRank];
    u64 txSrcOffset = tempAlgParams.sdispls[slot.txRank] * dataTypeSize_;
    CHK_PRT_RET(!tempAlgParams.enableRemoteMemAccess || txChannel.remoteOutputGraphMode.addr == nullptr,
                HCCL_ERROR("[InsTempAlltoAllVBsp][RunSlot] remote output access is unavailable. "
                           "myRank[%u] txRank[%u] plane[%u] enableRemoteMemAccess[%d].",
                           myRank_, slot.txRank, slot.plane, tempAlgParams.enableRemoteMemAccess),
                HCCL_E_INTERNAL);
    CHK_PRT_RET(slot.txRank >= tempAlgParams.remoteRdispls.size() ||
                    slot.txRank >= tempAlgParams.remoteRecvCounts.size(),
                HCCL_ERROR("[InsTempAlltoAllVBsp][RunSlot] remote A2AV info missing. txRank[%u] "
                           "remoteRdispls[%zu] remoteRecvCounts[%zu].",
                           slot.txRank, tempAlgParams.remoteRdispls.size(),
                           tempAlgParams.remoteRecvCounts.size()),
                HCCL_E_INTERNAL);
    CHK_PRT_RET(tempAlgParams.remoteRecvCounts[slot.txRank] != tempAlgParams.sendCounts[slot.txRank],
                HCCL_ERROR("[InsTempAlltoAllVBsp][RunSlot] remote recv count mismatch. myRank[%u] txRank[%u] "
                           "sendCount[%llu] remoteRecvCount[%llu].",
                           myRank_, slot.txRank, tempAlgParams.sendCounts[slot.txRank],
                           tempAlgParams.remoteRecvCounts[slot.txRank]),
                HCCL_E_PARA);
    u64 txDstOffset = tempAlgParams.remoteRdispls[slot.txRank] * dataTypeSize_;

    u64 rxSize = tempAlgParams.recvCounts[slot.rxRank] * dataTypeSize_;
    u64 rxCount = tempAlgParams.recvCounts[slot.rxRank];
    u64 rxDstOffset = tempAlgParams.rdispls[slot.rxRank] * dataTypeSize_;

    CHK_PRT_RET(txSrcOffset + txSize > tempAlgParams.buffInfo.inputSize,
                HCCL_ERROR("[InsTempAlltoAllVBsp][RunSlot] tx input out of range. txRank[%u] off[%llu] size[%llu] "
                           "inputSize[%llu].", slot.txRank, txSrcOffset, txSize, tempAlgParams.buffInfo.inputSize),
                HCCL_E_INTERNAL);
    CHK_PRT_RET(txDstOffset + txSize > txChannel.remoteOutputGraphMode.size,
                HCCL_ERROR("[InsTempAlltoAllVBsp][RunSlot] tx remote output out of range. txRank[%u] plane[%u] "
                           "off[%llu] size[%llu] remoteSize[%llu].",
                           slot.txRank, slot.plane, txDstOffset, txSize, txChannel.remoteOutputGraphMode.size),
                HCCL_E_INTERNAL);
    CHK_PRT_RET(rxDstOffset + rxSize > tempAlgParams.buffInfo.outputSize,
                HCCL_ERROR("[InsTempAlltoAllVBsp][RunSlot] rx output out of range. rxRank[%u] plane[%u] "
                           "dstOff[%llu] size[%llu] outputSize[%llu].",
                           slot.rxRank, slot.plane, rxDstOffset, rxSize, tempAlgParams.buffInfo.outputSize),
                HCCL_E_INTERNAL);

    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    if (txSize > 0) {
        txSrcSlices.emplace_back(tempAlgParams.buffInfo.inputPtr, txSrcOffset, txSize, txCount);
        txDstSlices.emplace_back(txChannel.remoteOutputGraphMode.addr, txDstOffset, txSize, txCount);
    }
    if (rxSize > 0) {
        rxSrcSlices.emplace_back(tempAlgParams.buffInfo.outputPtr, rxDstOffset, rxSize, rxCount);
        rxDstSlices.emplace_back(tempAlgParams.buffInfo.outputPtr, rxDstOffset, rxSize, rxCount);
    }

    const bool samePeerChannel = slot.txRank == slot.rxRank;
    if (samePeerChannel && txSize > 0 && rxSize > 0) {
        SendRecvInfo sendRecvInfo{{txChannel, rxChannel},
            {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_};
        HcclResult ret = SendRecvWrite(sendRecvInfo, sendThread);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[InsTempAlltoAllVBsp][RunSlot] same peer sendrecv failed. myRank[%u] peer[%u] "
                               "deltaC[%u] deltaR[%u] plane[%u] ret[0x%x].",
                               myRank_, slot.txRank, slot.deltaC, slot.deltaR, slot.plane, ret),
                    ret);
        return HCCL_SUCCESS;
    }

    HcclResult recvResult = HCCL_SUCCESS;
    HcclResult sendResult = HCCL_SUCCESS;
    if (rxSize > 0) {
        DataInfo recvInfo(rxChannel, {rxSrcSlices, rxDstSlices}, dataType_);
        recvResult = RecvWrite(recvInfo, recvThread);
    }
    if (txSize > 0) {
        DataInfo sendInfo(txChannel, {txSrcSlices, txDstSlices}, dataType_);
        sendResult = SendBatchWrite(sendInfo, sendThread);
    }

    HcclResult ret = (recvResult != HCCL_SUCCESS) ? recvResult : sendResult;
    CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[InsTempAlltoAllVBsp][RunSlot] split send/recv failed. myRank[%u] txRank[%u] "
                           "rxRank[%u] deltaC[%u] deltaR[%u] plane[%u] recvRet[0x%x] sendRet[0x%x].",
                           myRank_, slot.txRank, slot.rxRank, slot.deltaC, slot.deltaR, slot.plane,
                           recvResult, sendResult),
                ret);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
