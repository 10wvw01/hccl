/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "ccu_temp_all_to_all_v_mesh1d_2Die.h"
#include "kernel/ccu_kernel_all_to_all_v_mesh1d_2die.h"
#include "ccu_launch_dl.h"
#include "hccl_res_dl.h"

namespace ops_hccl {
CcuTempAlltoAllVMesh1D2Die::CcuTempAlltoAllVMesh1D2Die(const OpParam &param, RankId rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    for (u32 i = 0; i < subCommRanks_.size(); i++) {
        for (u32 j = 0; j < subCommRanks_[i].size(); j++) {
            HCCL_INFO("subCommRanks_[%u][%u]=%u", i, j, subCommRanks_[i][j]);
        }
    }

    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        myRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempAlltoAllVMesh1D2Die::~CcuTempAlltoAllVMesh1D2Die()
{
}

HcclResult CcuTempAlltoAllVMesh1D2Die::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(subCommRanks_.size() != 1 || subCommRanks_[0].empty(),
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][CalcRes] Invalid subCommRanks[%u] or subCommRanks empty.",
            subCommRanks_.size()), HcclResult::HCCL_E_INTERNAL);

    HCCL_DEBUG("[CcuTempAlltoAllVMesh1D2Die][CalcRes] rankSize[%u] subCommRanks0[%u].", templateRankSize_,
        subCommRanks_[0].size());

    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.push_back(1);

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));
    HCCL_INFO("channelDescs size[%u]", channelDescs.size());

    CHK_RET(PartitionChannels(comm, channelDescs, rankIdToChannelDesc_));
    resourceRequest.channels.emplace_back(channelDescs);
    HCCL_INFO("resourceRequest.channels[%d]", resourceRequest.channels.size());

    const uint32_t rankSize = subCommRanks_[0].size();
    resourceRequest.ccuKernelNum.push_back(DIE_NUM);

    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        CcuKernelInfo kernelInfo;
        CHK_SAFETY_FUNC_RET(strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllToAllVMesh1D2DieKernel"));
        kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllToAllVMesh1D2DieKernel);

        const bool withMyRank = channels_[dieId].size() < channels_[1 - dieId].size();
        auto kernelArg = std::make_shared<CcuKernelArgAllToAllVMesh1D2Die>();
        kernelArg->rankId = myRank_;
        kernelArg->opParam = param;
        kernelArg->subCommRanks = subCommRanks_;
        kernelArg->withMyRank = withMyRank;
        kernelArg->rankGroup = rankGroup_[dieId];
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = channels_[dieId];
        resourceRequest.ccuKernelInfos.emplace_back(kernelInfo);
        HCCL_DEBUG("[CcuTempAlltoAllVMesh1D2Die][CalcRes] dieId=%u, channels=%llu, withMyRank=%u, ccuKernelInfos=%llu",
            dieId, channels_[dieId].size(), withMyRank, resourceRequest.ccuKernelInfos.size());
    }

    CHK_RET(SaveCacheCtx(comm, param));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh1D2Die::PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs,
                                                        std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc)
{
    (void) channelDescs;
    for (auto& rankToChannels: rankIdToChannelDesc){
        u32 remoteRank = rankToChannels.first;
        std::vector<HcclChannelDesc>& channel_list = rankToChannels.second;

        using DieIdType = uint32_t;
        const uint32_t dieIdTypeSize = sizeof(DieIdType);
        bool isClos = channel_list.size() > 1;

        for (const auto &channel : channel_list) {
            DieIdType dieId = 0;
            EndpointDesc localEndpoint = channel.localEndpoint;
            CHK_RET(HcclRankGraphGetEndpointInfo(comm, myRank_, &localEndpoint, ENDPOINT_ATTR_DIE_ID,
                dieIdTypeSize, static_cast<void*>(&dieId)));
            channels_[dieId].emplace_back(channel);
            rankGroup_[dieId].push_back(channel.remoteRank);

            if (isClos) {
                closPeers_.insert(channel.remoteRank);
                if (closBwCoeff_[dieId] == 0) {
                    CHK_RET(GetChannelBwCoeff(comm, myRank_, channel, closBwCoeff_[dieId]));
                }
            }
        }
    }

    if (closBwCoeff_[0] <= closBwCoeff_[1]) {
        closMinorDieId_ = 0;
        closMajorDieId_ = 1;
    } else {
        closMinorDieId_ = 1;
        closMajorDieId_ = 0;
    }
    totalBwCoeff_ = closBwCoeff_[0] + closBwCoeff_[1];

    if (channels_[0].size() < channels_[1].size()) {
        rankGroup_[0].push_back(myRank_);
    } else {
        rankGroup_[1].push_back(myRank_);
    }

    HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][PartitionChannels] Rank[%d], channels size, "
        "die0 channels[%u], die1 channels[%u].", myRank_, channels_[0].size(), channels_[1].size());

    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAlltoAllVMesh1D2Die::SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
}

HcclResult CcuTempAlltoAllVMesh1D2Die::SaveCacheCtx(HcclComm comm, const OpParam &param)
{
    Mesh2DieCacheCtx cacheCtx;
    cacheCtx.dieNum = DIE_NUM;
    cacheCtx.rankGroup[0] = rankGroup_[0];
    cacheCtx.rankGroup[1] = rankGroup_[1];
    cacheCtx.closPeers = closPeers_;
    cacheCtx.closBwCoeff[0] = closBwCoeff_[0];
    cacheCtx.closBwCoeff[1] = closBwCoeff_[1];
    cacheCtx.totalBwCoeff = totalBwCoeff_;
    cacheCtx.closMinorDieId = closMinorDieId_;
    cacheCtx.closMajorDieId = closMajorDieId_;

    std::vector<char> buf = cacheCtx.Serialize();

    char cacheTag[ALG_TAG_LENGTH] = {0};
    int ret = snprintf_s(cacheTag, sizeof(cacheTag), sizeof(cacheTag) - 1, "%s_mesh2die", param.algTag);
    CHK_PRT_RET(ret <= 0,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][SaveCacheCtx] failed to fill cacheTag"), HCCL_E_INTERNAL);

    void *ctxPtr = nullptr;
    uint64_t ctxSize = 0;
    HcclResult getRet = HcclEngineCtxGet(comm, cacheTag, CommEngine::COMM_ENGINE_CPU_TS, &ctxPtr, &ctxSize);
    if (getRet == HCCL_SUCCESS && ctxPtr != nullptr && ctxSize == buf.size()) {
        errno_t memcpyRet = memcpy_s(ctxPtr, ctxSize, buf.data(), buf.size());
        CHK_PRT_RET(memcpyRet != EOK,
            HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][SaveCacheCtx] memcpy_s to existing ctx failed, ret=%d", memcpyRet),
            HcclResult::HCCL_E_INTERNAL);
        HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][SaveCacheCtx] updated existing cacheCtx, size[%zu]", buf.size());
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(HcclEngineCtxCreate(comm, cacheTag, CommEngine::COMM_ENGINE_CPU_TS, buf.size(), &ctxPtr));

    errno_t memcpyRet = memcpy_s(ctxPtr, buf.size(), buf.data(), buf.size());
    CHK_PRT_RET(memcpyRet != EOK,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][SaveCacheCtx] memcpy_s failed, ret=%d", memcpyRet),
        HcclResult::HCCL_E_INTERNAL);

    HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][SaveCacheCtx] created and saved cacheCtx, size[%zu], "
        "rankGroup0[%zu] rankGroup1[%zu] closPeers[%zu] closBwCoeff[%u/%u] totalBwCoeff[%u] "
        "closMinorDieId[%u] closMajorDieId[%u]",
        buf.size(), rankGroup_[0].size(), rankGroup_[1].size(), closPeers_.size(),
        closBwCoeff_[0], closBwCoeff_[1], totalBwCoeff_, closMinorDieId_, closMajorDieId_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh1D2Die::LoadCacheCtx(const OpParam &param, Mesh2DieCacheCtx &cacheCtx)
{
    char cacheTag[ALG_TAG_LENGTH] = {0};
    int ret = snprintf_s(cacheTag, sizeof(cacheTag), sizeof(cacheTag) - 1, "%s_mesh2die", param.algTag);
    CHK_PRT_RET(ret <= 0,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][LoadCacheCtx] failed to fill cacheTag"), HCCL_E_INTERNAL);

    void *ctxPtr = nullptr;
    uint64_t ctxSize = 0;
    HcclComm comm = static_cast<HcclComm>(param.hcclComm);
    HcclResult getRet = HcclEngineCtxGet(comm, cacheTag, CommEngine::COMM_ENGINE_CPU_TS, &ctxPtr, &ctxSize);
    CHK_PRT_RET(getRet != HCCL_SUCCESS,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][LoadCacheCtx] HcclEngineCtxGet failed, ret=%d", getRet),
        getRet);
    CHK_PRT_RET(ctxPtr == nullptr || ctxSize == 0,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][LoadCacheCtx] cache ctx is null or empty"),
        HcclResult::HCCL_E_INTERNAL);

    cacheCtx.Deserialize(static_cast<const char *>(ctxPtr), static_cast<size_t>(ctxSize));

    HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][LoadCacheCtx] loaded cacheCtx, size[%llu], "
        "rankGroup0[%zu] rankGroup1[%zu] closPeers[%zu] closBwCoeff[%u/%u] totalBwCoeff[%u] "
        "closMinorDieId[%u] closMajorDieId[%u]",
        ctxSize, cacheCtx.rankGroup[0].size(), cacheCtx.rankGroup[1].size(), cacheCtx.closPeers.size(),
        cacheCtx.closBwCoeff[0], cacheCtx.closBwCoeff[1], cacheCtx.totalBwCoeff,
        cacheCtx.closMinorDieId, cacheCtx.closMajorDieId);
    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAlltoAllVMesh1D2Die::FillRankGroupTaskArgs(uint32_t dieId, const Mesh2DieCacheCtx &cacheCtx,
    const LoopGroupConfig &config, std::vector<uint64_t> &taskArgs)
{
    for (auto peerId : cacheCtx.rankGroup[dieId]) {
        uint64_t sendSize = localSendRecvInfo_.sendLength[peerId];
        uint64_t sendOffset = localSendRecvInfo_.sendOffset[peerId];
        uint64_t recvOffset = localSendRecvInfo_.recvOffset[peerId];

        if (cacheCtx.closPeers.count(peerId) > 0 && cacheCtx.totalBwCoeff > 0) {
            uint64_t recvLength = localSendRecvInfo_.recvLength[peerId];
            uint32_t myBwCoeff = cacheCtx.closBwCoeff[dieId];
            if (dieId == cacheCtx.closMajorDieId) {
                sendOffset += sendSize * cacheCtx.closBwCoeff[cacheCtx.closMinorDieId] / cacheCtx.totalBwCoeff;
                recvOffset += recvLength * cacheCtx.closBwCoeff[cacheCtx.closMinorDieId] / cacheCtx.totalBwCoeff;
            }
            sendSize = sendSize * myBwCoeff / cacheCtx.totalBwCoeff;
        }

        const uint64_t floorLoopNum = sendSize / UB_MAX_TRANS_SIZE;
        uint64_t sendLoopNum = UINT64_MAX - 1 - floorLoopNum;
        uint64_t sendTailSize = sendSize - floorLoopNum * UB_MAX_TRANS_SIZE;
        auto sendTailGoSize = CalGoSize(sendTailSize, config);
        taskArgs.push_back(sendOffset);
        taskArgs.push_back(recvOffset);
        taskArgs.push_back(sendTailSize);
        for (auto val : sendTailGoSize) {
            taskArgs.push_back(val);
        }
        taskArgs.push_back(sendLoopNum);
    }
    return;
}

HcclResult CcuTempAlltoAllVMesh1D2Die::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
    TemplateResource& templateResource)
{
    CHK_PRT_RET(subCommRanks_.empty() || subCommRanks_[0].empty(),
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][KernelRun] subCommRanks empty."), HcclResult::HCCL_E_INTERNAL);

    buffInfo_ = templateDataParams.buffInfo;
    CHK_PRT_RET(buffInfo_.inputPtr == nullptr || buffInfo_.outputPtr == nullptr,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][KernelRun] Rank[%d] input[%#llx] or output[%#llx] is null",
            myRank_, buffInfo_.inputPtr, buffInfo_.outputPtr),
        HcclResult::HCCL_E_PTR);

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr);
    HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][KernelRun] begin. Rank[%d], input[%#llx/%#llx], output[%#llx/%#llx], "
        "sendType[%d], recvType[%d]", myRank_, inputAddr, param.inputPtr, outputAddr, param.outputPtr,
        param.all2AllVDataDes.sendType, param.all2AllVDataDes.recvType);

    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    Mesh2DieCacheCtx cacheCtx;
    CHK_RET(LoadCacheCtx(param, cacheCtx));

    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    std::vector<u32> notifyIdxMainToSub(1, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    for (uint32_t dieId = 0; dieId < DIE_NUM; dieId++) {
        std::vector<uint64_t> taskArgs;
        taskArgs.push_back(inputAddr);
        taskArgs.push_back(outputAddr);
        taskArgs.push_back(token);

        LoopGroupConfig config{};
        config.msInterleave = CCU_MS_INTERLEAVE;
        config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
        config.memSlice = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
        auto xnMaxTransportGoSize = CalGoSize(UB_MAX_TRANS_SIZE, config);
        for (auto val : xnMaxTransportGoSize) {
            taskArgs.push_back(val);
        }

        FillRankGroupTaskArgs(dieId, cacheCtx, config, taskArgs);

        uint64_t argSize = taskArgs.size();
        HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][KernelRun] dieId[%u] argSize[%llu] rankGroupSize[%zu]",
            dieId, argSize, cacheCtx.rankGroup[dieId].size());
        CcuResult launchRet = HcommCcuKernelLaunch(
            templateResource.threads[dieId], templateResource.ccuKernels[dieId],
            taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    std::vector<u32> notifyIdxSubToMain(1, 0);
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    HCCL_DEBUG("[CcuTempAlltoAllVMesh1D2Die][KernelRun] end. Rank[%d]", myRank_);

    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl
