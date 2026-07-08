/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "hccl_ccu_res.h"
// #include "ccu_assist_pub.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

#include "ccu_launch_dl.h"

#include "ccu_temp_omni.h"
#include "kernel/ccu_kernel_omni.h"

namespace ops_hccl {

constexpr u32 DIE_NUM = 2;
constexpr u32 UDIE0 = 0;
constexpr u32 UDIE1 = 1;

CcuTempOmni::CcuTempOmni(const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    tempRankSize_ = subCommRanks[0].size();
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
}

CcuTempOmni::~CcuTempOmni()
{
}

HcclResult CcuTempOmni::CreateChannelFromLink(HcclComm comm, u32 myRank, u32 rank, uint32_t netLayer, u32 idx,
    const CommLink &link, const std::string &funcName, std::vector<HcclChannelDesc> &channels)
{
    (void)comm;
    HcclChannelDesc channelDesc;
    HcclChannelDescInit(&channelDesc, 1);
    channelDesc.remoteRank = rank;
    channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
    channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    HCCL_DEBUG("%s local device phyId: %u, remote device phyId: %u.", funcName.c_str(),
        channelDesc.localEndpoint.loc.device.devPhyId, channelDesc.remoteEndpoint.loc.device.devPhyId);
    HCCL_INFO("%s Add channel request between %zu and %zu, netLayerIdx %u, "
              "linkListIdx %u, protocol %zu",
        funcName.c_str(), myRank, channelDesc.remoteRank, netLayer, idx, channelDesc.remoteEndpoint.protocol);
    channelDesc.channelProtocol = link.linkAttr.linkProtocol;
    channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
    channels.push_back(channelDesc);
    return HCCL_SUCCESS;
}

HcclResult CcuTempOmni::ProcessLinkForProtocol(HcclComm comm, const std::vector<CommProtocol> &expectedProtocols,
    const std::vector<CommLink> &linkList, u32 myRank, u32 remoteRank, uint32_t netLayer,
    std::vector<HcclChannelDesc> &channels, bool &protocolFound, const std::string &funcName)
{
    protocolFound = false;
    HCCL_INFO("ProcessLinkForProtocol expectedProtocols size %u", expectedProtocols.size());
    for (auto expectedProtocol : expectedProtocols) {
        for (u32 idx = 0; idx < linkList.size(); idx++) {
            HCCL_INFO("linkProtocol %u, expectedProtocol %u", linkList[idx].linkAttr.linkProtocol, expectedProtocol);
            if (linkList[idx].linkAttr.linkProtocol == expectedProtocol) {
                CHK_RET(
                    CreateChannelFromLink(comm, myRank, remoteRank, netLayer, idx, linkList[idx], funcName, channels));
                protocolFound = true;
                break;
            }
        }
        if (protocolFound) {
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempOmni::CalcChannelRequestOmni(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, const std::vector<std::vector<u32>> &subcommInfo,
    std::vector<HcclChannelDesc> &channels)
{
    channels.clear();

    const u32 myRank = topoInfo->userRank;

    for (const auto &channelInfo : topoInfo->xmlInfo.resInfo.vecChannelInfo) {
        const u64 remoteRank = channelInfo.remoteRank;

        uint32_t *netLayers = nullptr;
        uint32_t netLayerNum = 0;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);

        HCCL_DEBUG("CalcChannelRequestOmni netLayerNum %u", netLayerNum);
        for (auto netLayer : netLayersVector) {
            HCCL_DEBUG("CalcChannelRequestOmni netLayer %u, xml netlayerid %u", netLayer, channelInfo.netlayerId);
            if (netLayer != channelInfo.netlayerId) {
                continue;
            }

            CommLink *linkList = nullptr;
            u32 listSize = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, static_cast<u32>(remoteRank), &linkList, &listSize));

            HCCL_DEBUG("CalcChannelRequestOmni netLayer %u, rank %u to remote rank %u linksize %u", netLayer, myRank,
                remoteRank, listSize);
            for (u32 idx = 0; idx < listSize; idx++) {
                HCCL_DEBUG("CalcChannelRequestOmni HcclRankGraphGetLinks myrank %u to remoteRank %u, netLayer %u, "
                           "listSize %u, linkProtocol %u",
                    myRank, remoteRank, netLayer, listSize, linkList[idx].linkAttr.linkProtocol);

                // if (linkList[idx].linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                //     continue;
                // }

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
                channels.push_back(channelDesc);

                HCCL_INFO("CalcChannelRequestOmni Add channel request between %zu and %zu, netLayerIdx %u, "
                          "linkListIdx %u, protocol %zu",
                    myRank, channelDesc.remoteRank, netLayer, idx, channelDesc.remoteEndpoint.protocol);

                break;
            }
        }
    }

    return HCCL_SUCCESS;
}

// 分别记录两个Die上的channel，构造rankGroup
HcclResult CcuTempOmni::PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs)
{
    for (const auto &channel : channelDescs) {
        const u32 remoteRank = channel.remoteRank;
        uint32_t dieId = 0;
        HcclResult ret = GetChannelDieId(comm, myRank_, channel, dieId);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CcuTempOmni][PartitionChannels] Rank[%d] channel to remoteRank[%d], Failed to "
                       "get dieId. errNo[0x%016llx]",
                myRank_, remoteRank, HCCL_ERROR_CODE(ret)),
            ret);
        CHK_PRT_RET(dieId >= DIE_NUM,
            HCCL_ERROR("[CcuTempOmni][PartitionChannels] Rank[%d] channel to remoteRank[%d], dieId[%u] is "
                       "invalid.",
                myRank_, remoteRank, dieId),
            HCCL_E_INTERNAL);
        HCCL_INFO("[CcuTempOmni][PartitionChannels] Rank[%d] channel to remoteRank[%d], insert to "
                  "channels at dieId[%u].",
            myRank_, remoteRank, dieId);
        channels_[dieId].emplace_back(channel);  // 记录此channel属于哪个die
        rankGroup_[dieId].push_back(remoteRank); // 记录此rank属于哪个die
    }

    HCCL_INFO(
        "PartitionChannels rankGroup_[0] size %u, rankGroup_[1] size %u", rankGroup_[0].size(), rankGroup_[1].size());

    HCCL_DEBUG("[CcuTempOmni][PartitionChannels] Rank[%d], die0 channels[%u], die1 channels[%u].", myRank_,
        channels_[0].size(), channels_[1].size());
    // keep myRank_ at last, sync with kernel
    if (channels_[0].size() == 0 && channels_[1].size() != 0) {
        rankGroup_[1].push_back(myRank_);
        return HcclResult::HCCL_SUCCESS;
    } else if (channels_[0].size() != 0 && channels_[1].size() == 0) {
        rankGroup_[0].push_back(myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    if (channels_[0].size() < channels_[1].size()) {
        rankGroup_[0].push_back(myRank_);
    } else {
        rankGroup_[1].push_back(myRank_);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempOmni::BuildOmniSliceMeta(const OpParam &param, u32 rankSize, u64 sliceNum, u64 dataCount,
    std::vector<u64> &sendSliceData, std::vector<u64> &recvSliceData, std::vector<u64> &sendSdispls,
    std::vector<u64> &localSdispls)
{
    HCCL_INFO("[BuildOmniSliceMeta] BEGIN");

    const u64 *scPtr = static_cast<const u64 *>(param.all2AllVDataDes.sendCounts);
    const u64 *rcPtr = static_cast<const u64 *>(param.all2AllVDataDes.recvCounts);

    sendSliceData.clear();
    recvSliceData.clear();
    sendSdispls.clear();
    localSdispls.clear();
    sendSliceData.reserve(sliceNum);
    recvSliceData.reserve(sliceNum);
    sendSdispls.reserve(sliceNum);
    localSdispls.reserve(sliceNum);

    const u32 ratio = static_cast<u32>(sliceNum / rankSize);
    if (rankSize == 0) {
        HCCL_ERROR("[BuildOmniSliceMeta] rankSize is 0");
        return HCCL_E_PARA;
    }

    if (dataCount != 0) { // 均分
        const u64 perSliceCount = dataCount / sliceNum;
        for (u32 i = 0; i < sliceNum; i++) {
            sendSliceData.push_back(perSliceCount);
            recvSliceData.push_back(perSliceCount);
            u32 rankIdx = i % ratio;
            sendSdispls.push_back(rankIdx * perSliceCount);
            localSdispls.push_back(i * perSliceCount);

            HCCL_ERROR("BuildOmniSliceMeta dataCount is not 0, sliceIdx %u, ratio %u, perSliceCount %u, rankIdx %u, "
                       "sendSliceData %u, "
                       "recvSliceData "
                       "%u, sendSdispls %u, localSdispls %u",
                i, ratio, perSliceCount, rankIdx, sendSliceData[i], recvSliceData[i], sendSdispls[i], localSdispls[i]);
        }
        return HCCL_SUCCESS;
    }

    if (sliceNum % rankSize != 0) {
        HCCL_ERROR("[BuildOmniSliceMeta] sliceNum[%llu] is not divisible by rankSize[%u]", sliceNum, rankSize);
        return HCCL_E_PARA;
    }

    for (u32 i = 0; i < sliceNum; i++) { // 非均分
        const u32 rankIdx = i / ratio;
        const u32 slotInRank = i % ratio;
        const u64 sendBase = scPtr[rankIdx] / ratio;
        const u64 sendRem = scPtr[rankIdx] % ratio;
        const u64 recvBase = rcPtr[rankIdx] / ratio;
        const u64 recvRem = rcPtr[rankIdx] % ratio;
        sendSliceData.push_back(sendBase + (slotInRank < sendRem ? 1 : 0));
        recvSliceData.push_back(recvBase + (slotInRank < recvRem ? 1 : 0));
        sendSdispls.push_back(sendBase * slotInRank);
        localSdispls.push_back(sendSdispls[rankIdx] + sendBase * slotInRank);

        HCCL_ERROR(
            "BuildOmniSliceMeta dataCount is 0, sliceIdx %u, ratio %u, rankIdx %u, sendSliceData %u, recvSliceData "
            "%u, sendSdispls %u, localSdispls %u",
            i, ratio, rankIdx, sendSliceData[i], recvSliceData[i], sendSdispls[i], localSdispls[i]);
    }

    return HCCL_SUCCESS;
}

HcclResult CcuTempOmni::BuildOmniTaskArgs(const BuffInfo &buffInfo, uint32_t syncIdx,
    const std::vector<u64> &sendCounts, const std::vector<u64> &recvCounts, const std::vector<u64> &sdispls,
    const std::vector<u64> &rdispls, const std::vector<u64> &sendSliceData, const std::vector<u64> &recvSliceData,
    const std::vector<u64> &sendSdispls, const std::vector<u64> &localSdispls, std::vector<uint64_t> &taskArgs) const
{
    taskArgs.clear();
    taskArgs.push_back(PointerToAddr(buffInfo.inputPtr) + buffInfo.inBuffBaseOff);
    taskArgs.push_back(PointerToAddr(buffInfo.outputPtr) + buffInfo.outBuffBaseOff);
    taskArgs.push_back(PointerToAddr(buffInfo.hcclBuff.addr));
    uint64_t token = 0;
    CHK_RET(GetToken(buffInfo, token));
    taskArgs.push_back(token);
    taskArgs.push_back(syncIdx);

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_INTERLEAVE;
    config.memSlice = CCU_MS_INTERLEAVE * CCU_MS_SIZE;

    uint64_t xnMaxTransportSize = UB_MAX_TRANS_SIZE;
    auto xnMaxTransportGoSize = CalGoSize(xnMaxTransportSize, config);
    for (auto val : xnMaxTransportGoSize) {
        taskArgs.push_back(val);
    }

    for (uint32_t i = 0; i < tempRankSize_; i++) {
        taskArgs.push_back(sendCounts[i]);
        taskArgs.push_back(recvCounts[i]);
        taskArgs.push_back(sdispls[i]);
        taskArgs.push_back(rdispls[i]);
    }

    for (uint64_t i = 0; i < sliceNum_; i++) {
        taskArgs.push_back(recvSliceData[i]);
        taskArgs.push_back(sendSdispls[i]);
        taskArgs.push_back(localSdispls[i]);

        uint64_t tailSize = sendSliceData[i] % UB_MAX_TRANS_SIZE;
        uint64_t loopNum = UINT64_MAX - 1 - (sendSliceData[i] / UB_MAX_TRANS_SIZE);
        taskArgs.push_back(tailSize);
        taskArgs.push_back(loopNum);
        auto tailGoSize = CalGoSize(tailSize, config);
        for (auto val : tailGoSize) {
            taskArgs.push_back(val);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult CcuTempOmni::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    HCCL_DEBUG("[CalcRes] rankid [%u] begin", mySubCommRank_);

    if (!topoInfo->xmlInfo.vecNormalInstruction.empty()) {
        sliceNum_ = topoInfo->xmlInfo.vecNormalInstruction[0].sendRecvInfo.sliceNum;
    }

    resourceRequest.slaveThreadNum = topoInfo->xmlInfo.resInfo.slaveThreadNum;
    resourceRequest.notifyNumOnMainThread = topoInfo->xmlInfo.resInfo.notifyNumOnMainThread;
    resourceRequest.notifyNumPerThread.assign(topoInfo->xmlInfo.resInfo.notifyNumPerThread, 1);

    // 计算channel信息
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestOmni(comm, param, topoInfo, subCommRanks_, channelDescs)); // 只支持两个rank之间1条链接
    // CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));

    HCCL_DEBUG("[CalcRes] rankid [%u] channelDescs size [%u]", mySubCommRank_, channelDescs.size());
    if (channelDescs.size() == 0) {
        HCCL_DEBUG("rankid [%u] channel size is 0", mySubCommRank_);
        return HCCL_E_NOT_FOUND;
    }

    CHK_RET(PartitionChannels(comm, channelDescs)); // 分别记录两个Die上的channel，构造rankGroup
    resourceRequest.channels.emplace_back(channelDescs);

    std::vector<std::vector<std::vector<OmniSendRecvInfo>>> tmpSendRecvInfo; // dieid:sync:opvec
    tmpSendRecvInfo.resize(DIE_NUM);

    syncNum_ = topoInfo->xmlInfo.syncNum;
    std::vector<OmniSendRecvInfo> tmpInfoDie0;
    std::vector<OmniSendRecvInfo> tmpInfoDie1;
    HCCL_DEBUG("[CalcRes] Instruction size %u", topoInfo->xmlInfo.vecNormalInstruction.size());
    for (auto &dieSendRecvInfo : topoInfo->xmlInfo.vecNormalInstruction) {
        if (dieSendRecvInfo.opType == omni::OP_RES_REQUEST
            || dieSendRecvInfo.opType == omni::OP_PRE_SYNC_INTER_THREADS) {
            continue;
        }

        if (dieSendRecvInfo.opType == omni::OP_POST_SYNC_INTER_THREADS) { // 此处编译每个同步中需要做的操作
            tmpSendRecvInfo[0].push_back(tmpInfoDie0);
            tmpSendRecvInfo[1].push_back(tmpInfoDie1);
            tmpInfoDie0.clear();
            tmpInfoDie1.clear();
            continue;
        }

        OmniSendRecvInfo sendRecvInfo;
        sendRecvInfo.opType = dieSendRecvInfo.opType;
        sendRecvInfo.inputDataType = dieSendRecvInfo.sendRecvInfo.inputDataType;
        sendRecvInfo.outputDataType = dieSendRecvInfo.sendRecvInfo.outputDataType;
        sendRecvInfo.reduceType = dieSendRecvInfo.sendRecvInfo.reduceType;
        sendRecvInfo.sliceNum = dieSendRecvInfo.sendRecvInfo.sliceNum;
        sendRecvInfo.threadIdx = dieSendRecvInfo.sendRecvInfo.threadIdx;
        sendRecvInfo.netlayerId = dieSendRecvInfo.sendRecvInfo.netlayerId;
        sendRecvInfo.srcSliceInfo = dieSendRecvInfo.sendRecvInfo.srcSliceInfo;
        sendRecvInfo.dstSliceInfo = dieSendRecvInfo.sendRecvInfo.dstSliceInfo;

        if (std::find(
                rankGroup_[0].begin(), rankGroup_[0].end(), dieSendRecvInfo.sendRecvInfo.dstSliceInfo[0].remoteRank)
            != rankGroup_[0].end()) {
            tmpInfoDie0.push_back(sendRecvInfo);
        } else if (std::find(rankGroup_[1].begin(), rankGroup_[1].end(),
                       dieSendRecvInfo.sendRecvInfo.dstSliceInfo[0].remoteRank)
                   != rankGroup_[1].end()) {
            tmpInfoDie1.push_back(sendRecvInfo);
        } else {
            HCCL_ERROR("[CalcRes] sendRecvInfo.remoteRank is %u not in rankGroup",
                dieSendRecvInfo.sendRecvInfo.dstSliceInfo[0].remoteRank);
        }
    }

    if (syncNum_ == 0) {
        tmpSendRecvInfo[0].push_back(tmpInfoDie0);
        tmpSendRecvInfo[1].push_back(tmpInfoDie1);
    }

    HCCL_DEBUG("[CalcRes] syncNum is %u, die0 sendRecvInfo size %u, die1 sendRecvInfo size %u", syncNum_,
        tmpSendRecvInfo[0].size(), tmpSendRecvInfo[1].size());

    HCCL_DEBUG("randgroup [0] size %u, [1] size %u", rankGroup_[0].size(), rankGroup_[1].size());

    uint32_t dieNum = 0;
    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) { // 2Die算法，需要执行两次
        if (rankGroup_[dieId].size() == 0)
            continue;
        dieNum++;

        // 创建每个kernel的kernelArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
        CcuKernelInfo kernelInfo;
        strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuOmniKernel");
        kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuOmniKernel);

        auto kernelArg = std::make_shared<CcuKernelArgOmni>();
        kernelArg->rankId = myRank_;
        kernelArg->opParam = param;
        kernelArg->subCommRanks = subCommRanks_;
        kernelArg->rankGroup = rankGroup_[dieId];
        kernelArg->sendRecvInfo = tmpSendRecvInfo[dieId];

        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = channels_[dieId];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);

        HCCL_DEBUG("[CcuTempOmni][CalcRes] dieId=%u, channels=%llu, ccuKernelInfos=%llu", dieId,
            channels_[dieId].size(), resourceRequest.ccuKernelInfos.size());
    }

    resourceRequest.ccuKernelNum.push_back(dieNum); // kernel数量

    HCCL_DEBUG("[CalcRes] rankid [%u] end", mySubCommRank_);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempOmni::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
    TemplateResource &templateResource, const omni::XmlInfo &xmlInfo, const std::vector<u64> &sendSliceData,
    const std::vector<u64> &recvSliceData, const std::vector<u64> &sendSdispls, const std::vector<u64> &localSdispls,
    uint32_t syncIdx)
{
    HCCL_INFO("[CcuTempOmni] rankid [%u] KernelRun begin", mySubCommRank_);

    buffInfo_ = templateDataParams.buffInfo;
    sliceNum_
        = xmlInfo.vecNormalInstruction.empty() ? sliceNum_ : xmlInfo.vecNormalInstruction[0].sendRecvInfo.sliceNum;

    std::vector<uint64_t> taskArgs;
    CHK_RET(BuildOmniTaskArgs(buffInfo_, syncIdx, templateDataParams.sendCounts, templateDataParams.recvCounts,
        templateDataParams.sdispls, templateDataParams.rdispls, sendSliceData, recvSliceData, sendSdispls, localSdispls,
        taskArgs));

    auto argSize = taskArgs.size();

    for (uint64_t dieIdx = 0; dieIdx < templateResource.ccuKernels.size(); dieIdx++) {
        CcuResult launchRet = HcommCcuKernelLaunch(
            templateResource.threads[dieIdx], templateResource.ccuKernels[dieIdx], taskArgs.data(), argSize);

        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempOmni::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }

        CcuKernelSubmitInfo submitInfo;
        submitInfo.kernelHandle = templateResource.ccuKernels[dieIdx];
        CHK_RET(FillCachedArgs(submitInfo, taskArgs[0], taskArgs[1], taskArgs[2], taskArgs[3], taskArgs[4], taskArgs[5],
            taskArgs[6], taskArgs[7], taskArgs[8], argSize, buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff));
        templateResource.submitInfos.push_back(submitInfo);
    }

    HCCL_INFO("[CcuTempOmni] rankid [%u] KernelRun end", mySubCommRank_);
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempOmni::GetThreadNum() const
{
    return tempRankSize_ > 1 ? tempRankSize_ - 1 : 1;
}

u64 CcuTempOmni::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return tempRankSize_;
}

HcclResult CcuTempOmni::FastLaunch(const OpParam &param, const TemplateFastLaunchCtx &tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempOmni::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempOmni::FastLaunch] start");

    u32 rankSize = 0;
    CHK_RET(HcclGetRankSize(static_cast<HcclComm>(param.hcclComm), &rankSize));
    tempRankSize_ = rankSize;

    const uint64_t *cachedArgs = tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs;
    constexpr u32 argSizeIdx = 9;
    constexpr u32 inputOffsetIdx = 10;
    constexpr u32 outputOffsetIdx = 11;
    const uint64_t argSize = cachedArgs[argSizeIdx];
    if (argSize < 9 + 4 * rankSize || (argSize - 9 - 4 * rankSize) % 9 != 0) {
        HCCL_ERROR("[CcuTempOmni::FastLaunch] invalid cached argSize[%llu] rankSize[%u]", argSize, rankSize);
        return HCCL_E_INTERNAL;
    }
    sliceNum_ = (argSize - 9 - 4 * rankSize) / 9;

    const u64 *scPtr = static_cast<const u64 *>(param.all2AllVDataDes.sendCounts);
    const u64 *rcPtr = static_cast<const u64 *>(param.all2AllVDataDes.recvCounts);
    const u64 *sdPtr = static_cast<const u64 *>(param.all2AllVDataDes.sdispls);
    const u64 *rdPtr = static_cast<const u64 *>(param.all2AllVDataDes.rdispls);

    std::vector<u64> sendCounts(tempRankSize_, 0);
    std::vector<u64> recvCounts(tempRankSize_, 0);
    std::vector<u64> sdispls(tempRankSize_, 0);
    std::vector<u64> rdispls(tempRankSize_, 0);
    for (uint32_t i = 0; i < tempRankSize_; i++) {
        sendCounts[i] = scPtr[i];
        recvCounts[i] = rcPtr[i];
        sdispls[i] = sdPtr[i];
        rdispls[i] = rdPtr[i];
    }

    std::vector<u64> sendSliceData;
    std::vector<u64> recvSliceData;
    std::vector<u64> sendSdispls;
    std::vector<u64> localSdispls;
    CHK_RET(BuildOmniSliceMeta(
        param, tempRankSize_, sliceNum_, param.dataCount, sendSliceData, recvSliceData, sendSdispls, localSdispls));

    BuffInfo buffInfo = tempFastLaunchCtx.buffInfo;
    buffInfo.inputPtr = param.inputPtr;
    buffInfo.outputPtr = param.outputPtr;
    buffInfo.hcclBuff = param.hcclBuff;
    buffInfo.inBuffBaseOff = cachedArgs[inputOffsetIdx];
    buffInfo.outBuffBaseOff = cachedArgs[outputOffsetIdx];

    std::vector<uint64_t> taskArgs;
    CHK_RET(BuildOmniTaskArgs(buffInfo, static_cast<uint32_t>(cachedArgs[4]), sendCounts, recvCounts, sdispls, rdispls,
        sendSliceData, recvSliceData, sendSdispls, localSdispls, taskArgs));

    HCCL_INFO("[CcuTempOmni][FastLaunch] rankid [%u] input[%llu] output[%llu] scratch[%llu] token[%llu] syncIdx[%llu] "
              "argSize[%llu]",
        mySubCommRank_, taskArgs[0], taskArgs[1], taskArgs[2], taskArgs[3], taskArgs[4], taskArgs.size());

    for (u32 i = 0; i < tempFastLaunchCtx.ccuKernelSubmitInfos.size(); i++) {
        CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[i],
            tempFastLaunchCtx.ccuKernelSubmitInfos[i].kernelHandle, taskArgs.data(), taskArgs.size());
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempOmni::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    HCCL_DEBUG("[CcuTempOmni::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl