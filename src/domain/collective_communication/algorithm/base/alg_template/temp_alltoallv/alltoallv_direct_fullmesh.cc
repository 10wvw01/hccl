/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_direct_fullmesh.h"

namespace hccl {
AlltoAllVDirectFullMesh::AlltoAllVDirectFullMesh(const HcclDispatcher dispatcher)
    : AlgTemplateBase(dispatcher)
{
}

AlltoAllVDirectFullMesh::~AlltoAllVDirectFullMesh() {}

HcclResult AlltoAllVDirectFullMesh::GenerateSubStreamInfo(const std::vector<Stream> &subStreams,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain)
{
    u32 totalSubstreamSize = (totalRdmaRankNum_ > 0) ?
        (sdmaConcurrentNum_ + rdmaConcurrentNum_ + 1) : (sdmaConcurrentNum_);
    if (subStreams.size() < totalSubstreamSize || meshSignalMainToSub.size() < totalSubstreamSize ||
        meshSignalSubToMain.size() < totalSubstreamSize) {
        HCCL_ERROR("[AlltoAllVDirectFullMesh][GenerateSubStreamInfo]subStreamsSize[%zu], meshSignalMainToSubSize[%zu]"\
            "meshSignalSubToMainSize[%zu] is smaller than totalSubstreamSize[%u]",subStreams.size(),
            meshSignalMainToSub.size(), meshSignalSubToMain.size(), totalSubstreamSize);
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(links_.size() < userRankSize_, HCCL_ERROR("[AlltoAllVDirectFullMesh][GenerateSubStreamInfo]"\
        "links_.size()[%zu] is smaller than userRankSize_[%u].", links_.size(), userRankSize_),
        HCCL_E_PARA);
    HCCL_DEBUG("subStreams.size[%zu], meshSignalMainToSub.size[%zu], links_.size[%zu]",
        subStreams.size(), meshSignalMainToSub.size(), links_.size());
    u32 index = 0;
    for (u32 sdmaIndex = 0; sdmaIndex < sdmaConcurrentNum_; sdmaIndex++) {
        sdmaSubStream_.push_back(subStreams[index]);
        sdmaMeshSignalMainToSub_.push_back(meshSignalMainToSub[index]);
        sdmaMeshSignalSubToMain_.push_back(meshSignalSubToMain[index]);
        index++;
    }
    for (u32 localIndex = 0; localIndex < sdmaConcurrentNum_; localIndex++) {
        localSubStream_.push_back(subStreams[index]);
        localSignalMainToSub_.push_back(meshSignalMainToSub[index]);
        localSignalSubToMain_.push_back(meshSignalSubToMain[index]);
        index++;
    }
    if (totalRdmaRankNum_ > 0) {
        rdmaSubStreams_.push_back(subStreams[index]);
        main2RdmaControlStreamNotify_ = meshSignalMainToSub[index];
        rdmaControl2MainStreamNotify_ = meshSignalSubToMain[index];
        index++;
        for (u32 rdmaIndex = 0; rdmaIndex < rdmaConcurrentNum_; rdmaIndex++) {
            rdmaSubStreams_.push_back(subStreams[index]);
            rdmaControl2SubNotifies_.push_back(meshSignalMainToSub[index]);
            rdmaSub2ControlNotifies_.push_back(meshSignalSubToMain[index]);
            index++;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::Prepare(PrepareData &param)
{
    mainStream_ = param.stream;
    userRank_ = param.userRank;
    userRankSize_ = param.userRankSize;
    links_ = *param.linksPtr;
    localSendRecvInfoPtr_ = param.localSendRecvInfoPtr;
    devNumInlocalPod_ = param.devNumInlocalPod;
    rankIdxInPod_ = param.rankIdxInPod;
    opType_ = param.opType;
    algOpContext_ = param.algOpContext;

    podStartRank_ = userRank_ - rankIdxInPod_;
    podEndRank_ = podStartRank_ + devNumInlocalPod_ - 1;
    sdmaConcurrentNum_ = (devNumInlocalPod_ > ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) ?
        (ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) : (devNumInlocalPod_);

    totalRdmaRankNum_ = userRankSize_ - devNumInlocalPod_;
    rdmaConcurrentNum_ = (totalRdmaRankNum_ > ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE) ?
        (ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE) : (totalRdmaRankNum_);
    HCCL_DEBUG("[AlltoAllVDirectFullMesh]devNumInlocalPod_[%u], userRankSize_[%u] podStartRank_[%u]" \
        "podEndRank_[%u], totalRdmaRankNum_[%u], sdmaConcurrentNum_[%u], rdmaConcurrentNum_[%u]",
        devNumInlocalPod_, userRankSize_, podStartRank_, podEndRank_, totalRdmaRankNum_,
        sdmaConcurrentNum_, rdmaConcurrentNum_);

    CHK_PRT_RET(userRankSize_ == 0, HCCL_ERROR("[AlltoAllVDirectFullMesh][Prepare]userRankSize_ is zero."),
        HCCL_E_PARA);

    userInput_ = param.inputMem;
    userOutput_ = param.outputMem;
    cclInMem_ = param.cclInMem;
    cclOutMem_ = param.cclOutMem;
    workMode_ = param.workMode;
    isSuPodAsym_ = param.isSuPodAsym;

    u64 maxSendLen = CalcMaxSendLen();
    isBigCount_ = (maxSendLen > ALLTOALLV_DIRECT_FULLMESH_BIG_SIZE) ? true : false;
    CHK_RET(GenerateSubStreamInfo(*param.subStreamsPtr, *param.signalPtr, *param.signalAuxPtr));

    if (algOpContext_.mc2Handler.stepSize > 0) {
        sdmaConcurrentNum_ = (devNumInlocalPod_ > 1) ? 1 : (devNumInlocalPod_);
        //MC2细粒度不需要本地并发处理
        isBigCount_ = false;
    }

    /* 考虑当group0 的rank 跟 group 1的所有rank通信时，每次都要收发，所以取sdmaConcurrentNum_块；
    跟group 0内的rank通信有一块儿浪费 */
    u32 blockGroup = (isBigCount_ || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLV || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLVC) ? 2 : 1;
    sdmaDataBlockSize_= (cclInMem_.size() / std::max(1u, sdmaConcurrentNum_ * blockGroup));
    // 向下对齐到16k Byte
    if (sdmaDataBlockSize_> HCCL_MIN_SLICE_ALIGN_910B) {
        sdmaDataBlockSize_= (sdmaDataBlockSize_/ HCCL_MIN_SLICE_ALIGN_910B) * HCCL_MIN_SLICE_ALIGN_910B;
    }
    CHK_PRT_RET(sdmaDataBlockSize_== 0, HCCL_ERROR("[AlltoAllVDirectFullMesh][Prepare]sdmaDataBlockSize_is zero."),
        HCCL_E_INTERNAL);
    HCCL_DEBUG("[AlltoAllVDirectFullMesh][Prepare] userRank [%u] total cclsize[%llu]," \
        "sdmaDataBlockSize_[%llu], BigCountFlag[%d], stepSize[%u]", userRank_, cclInMem_.size(), sdmaDataBlockSize_, isBigCount_,
        algOpContext_.mc2Handler.stepSize);

    // 一半的CCLOut用来发送RDMA数据，另一半用来接收RDMA数据，因此需要除以2
    rdmaDataBlockSize_ = cclOutMem_.size() / std::max(1u, rdmaConcurrentNum_) / 2;

    return HCCL_SUCCESS;
}

std::string AlltoAllVDirectFullMesh::GetStreamIndexString()
{
    std::string res = "";
    for (auto& info : subStreamReadInfo_) {
        u32 destRank = info.first;
        u32 streamIndex = destRank % sdmaConcurrentNum_;
        res += std::to_string(streamIndex) + ", ";
    }
    return res;
}

u64 AlltoAllVDirectFullMesh::CalcMaxSendLen()
{
    u64 maxSendLen = 0;
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;

    for (u32 dstRank = 0; dstRank < localSendRecvInfo.sendLength.size(); dstRank++) {
        maxSendLen = std::max(maxSendLen, localSendRecvInfo.sendLength[dstRank]);
    }

    HCCL_DEBUG("[AlltoAllVDirectFullMesh][CalcMaxSendLen] maxSendLen[%llu]", maxSendLen);
    return maxSendLen;
}

void AlltoAllVDirectFullMesh::UpdateCurrRankRecvInfo(u32 roundIdx, u32 side, u32 destRank,
    std::vector<ReadDataBlock>& readInfo, u32 maxRecvStep)
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;
    u64 remainRecvLen = localSendRecvInfo.recvLength[destRank];
    u64 scratchOffset = 0;
    u32 bufferIdx = 0;
    u32 pairNum = sdmaConcurrentNum_ / RANK_SET_COMPUTE_CONST;
    if (sdmaConcurrentNum_ == 1) { // 保证和当前rank距离一样时，send/recv用的是同一块buff
        bufferIdx = 0;
    } else if (side == 0) { // 在curRank左边
        u32 gap = (userRank_ - destRank + devNumInlocalPod_) % devNumInlocalPod_;
        bufferIdx = pairNum - (gap - roundIdx * pairNum);
    } else if (side == 1) { // 在curRank右边
        u32 gap = (destRank - userRank_ + devNumInlocalPod_) % devNumInlocalPod_;
        bufferIdx = pairNum - 1 + (gap - roundIdx * pairNum);
    } else { // 最后一个中间位置的rank
        bufferIdx = 0;
    }

    if ((isBigCount_ || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLV || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLVC) &&
        (roundIdx % RANK_SET_COMPUTE_CONST != 0)) { // 奇数轮，用下半Buffer
        bufferIdx += sdmaConcurrentNum_;
    }

    scratchOffset = bufferIdx * sdmaDataBlockSize_;

    u32 recvStepIdx = 0;
    u64 dataOffset = 0;
    HCCL_DEBUG("usrRank[%u] total recv localSendRecvInfo.recvLength[%llu] from dstRank[%u] bufferIdx[%u]",
        userRank_, remainRecvLen, destRank, bufferIdx);

    while(recvStepIdx < maxRecvStep && remainRecvLen > 0) {
        u64 currDataRemainLen = localSendRecvInfo.recvLength[destRank] - dataOffset;
        u64 recvLen = std::min(sdmaDataBlockSize_, currDataRemainLen);
        u64 userOutOffset = localSendRecvInfo.recvOffset[destRank] + dataOffset;
        HCCL_DEBUG("[AlltoAllVDirectFullMesh][UpdateCurrRankRecvInfo] usrRank[%u] recv from destRank [%u]"
            "sendStepIdx[%u] recvLen[%lu] userOutOffset[%llu] scratchOffset[%llu]",
            userRank_, destRank, recvStepIdx, recvLen, userOutOffset, scratchOffset);
        readInfo.push_back({recvLen, scratchOffset, userOutOffset});
        dataOffset += recvLen;
        recvStepIdx++;
        remainRecvLen -= recvLen;
    }
}

void AlltoAllVDirectFullMesh::UpdateCurrRankSendInfo(u32 roundIdx, u32 side, u32 destRank,
    std::vector<SendDataBlock>& sendInfo, u32 maxSendStep)
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;
    u64 remainSendLen = localSendRecvInfo.sendLength[destRank];

    u64 scratchOffset = 0;
    u32 bufferIdx = 0;
    u32 pairNum = sdmaConcurrentNum_ / RANK_SET_COMPUTE_CONST;
    if (sdmaConcurrentNum_ == 1) { // 保证和当前rank距离一样时，send/recv用的是同一块buff
        bufferIdx = 0;
    } else if (side == 0) { // 在curRank左边
        u32 gap = (userRank_ - destRank + devNumInlocalPod_) % devNumInlocalPod_;
        bufferIdx = pairNum - 1 + (gap - roundIdx * pairNum);
    } else if (side == 1) { // 在curRank右边
        u32 gap = (destRank - userRank_ + devNumInlocalPod_) % devNumInlocalPod_;
        bufferIdx = pairNum - (gap - roundIdx * pairNum);
    } else { // 最后一个中间位置的rank
        bufferIdx = 0;
    }

    if ((isBigCount_ || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLV || opType_ == HcclCMDType::HCCL_CMD_ALLTOALLVC) &&
        (roundIdx % RANK_SET_COMPUTE_CONST != 0)) { // 奇数轮，用下半Buffer
        bufferIdx += sdmaConcurrentNum_;
    }
    scratchOffset = bufferIdx * sdmaDataBlockSize_;

    u32 sendStepIdx = 0;
    u64 dataOffset = 0;
    HCCL_DEBUG("usrRank[%u] total send localSendRecvInfo.sendLength[%llu] to dstRank[%u] bufferIdx[%u]",
        userRank_, remainSendLen, destRank, bufferIdx);

    while (sendStepIdx < maxSendStep && remainSendLen > 0) {
        u64 currDataRemainLen = localSendRecvInfo.sendLength[destRank] - dataOffset;
        u64 sendLen = std::min(sdmaDataBlockSize_, currDataRemainLen);
        u64 userInOffset = localSendRecvInfo.sendOffset[destRank] + dataOffset;
        HCCL_DEBUG("[AlltoAllVDirectFullMesh][UpdateCurrRankSendInfo] usrRank[%u] send to destRank [%u]"
            " sendStepIdx[%u] sendLen[%lu] userInOffset[%llu] scratchOffset[%llu]",
            userRank_, destRank, sendStepIdx, sendLen, userInOffset, scratchOffset);
        sendInfo.push_back({sendLen, userInOffset, scratchOffset});
        dataOffset += sendLen;
        sendStepIdx++;
        remainSendLen -= sendLen;
    }
}

void AlltoAllVDirectFullMesh::UpdateSendRecvInfo(u32 roundIdx,
    std::unordered_map<u32, std::vector<ReadDataBlock>> &subStreamReadInfo,
    std::unordered_map<u32, std::vector<SendDataBlock>> &subStreamSendInfo,
    const std::vector<std::vector<std::pair<u32,u32>>> &partialCommRankSet)
{
    for (u32 side = 0; side < partialCommRankSet.size(); side++) {
        for (u32 j = 0; j < partialCommRankSet[side].size(); j++) {
            u32 readRemoteRank = partialCommRankSet[side][j].first;
            if (readRemoteRank == userRank_) {
                continue;
            }
            u32 currDestRecvStep = recvNumSubStep_[readRemoteRank];
            std::vector<ReadDataBlock> readInfo;
            UpdateCurrRankRecvInfo(roundIdx, side, readRemoteRank, readInfo, currDestRecvStep);

            subStreamReadInfo[readRemoteRank] = readInfo;
        }
    }

    for (u32 side = 0; side < partialCommRankSet.size(); side++) {
        for (u32 j = 0; j < partialCommRankSet[side].size(); j++) {
            u32 sendRemoteRank = partialCommRankSet[side][j].second;
            if (sendRemoteRank == userRank_) {
                continue;
            }
            u32 currDestSendStep = sendNumSubStep_[sendRemoteRank];
            std::vector<SendDataBlock> sendInfo;
            UpdateCurrRankSendInfo(roundIdx, side, sendRemoteRank, sendInfo, currDestSendStep);

            subStreamSendInfo[sendRemoteRank] = sendInfo;
        }
    }
}

void AlltoAllVDirectFullMesh::UpdateOpBaseSubStreamInfo(u32 roundIdx)
{
    if (roundIdx == 0 || !isBigCount_) {
        subStreamReadInfo_.clear();
        subStreamSendInfo_.clear();
        UpdateSendRecvInfo(roundIdx, subStreamReadInfo_, subStreamSendInfo_, partialCommRankSet_);
    }
    if (isBigCount_ && (roundIdx < commRounds_ - 1)) {
        nextSubStreamReadInfo_.clear();
        nextSubStreamSendInfo_.clear();
        UpdateSendRecvInfo(roundIdx + 1, nextSubStreamReadInfo_, nextSubStreamSendInfo_, nextPartialCommRankSet_);
    }
}

HcclResult AlltoAllVDirectFullMesh::PrepareIntraData(u32 step,
    std::unordered_map<u32,std::vector<SendDataBlock>> &subStreamSendInfo)
{
    u32 sendDataIndex = 0;
    for (auto& sdmaInfo : subStreamSendInfo) {
        const std::vector<SendDataBlock>& sendInfo = sdmaInfo.second;
        if (step < sendNumSubStep_[sdmaInfo.first]) {
            DeviceMem src = userInput_.range(sendInfo[step].userInOffset, sendInfo[step].sendLen);
            DeviceMem dst = cclInMem_.range(sendInfo[step].scratchOffset, sendInfo[step].sendLen);
            HCCL_DEBUG("[AlltoAllVDirectFullMesh][PrepareIntraData]userRank [%u] copy from userInOffset[%llu]"
                "len[%u] to scratchOffset [%llu]", userRank_, sendInfo[step].userInOffset, sendInfo[step].sendLen,
                sendInfo[step].scratchOffset);
            if (isBigCount_) {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, localSubStream_[sendDataIndex]));
            } else {
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
            }
        }
        sendDataIndex++;
    }
    return HCCL_SUCCESS;
}

void AlltoAllVDirectFullMesh::UpdateRemoteRankSet(u32 roundIdx, u32 groupRankSize)
{
    if (sdmaConcurrentNum_ == 1) {
        UpdatePartialCommunicationRankSetPairWise(roundIdx, groupRankSize);
    } else {
        UpdatePartialCommunicationRankSet(roundIdx, groupRankSize, partialCommRankSet_);
    }
}

void AlltoAllVDirectFullMesh::UpdatePartialCommunicationRankSetPairWise(u32 roundIdx, u32 groupRankSize)
{
    partialCommRankSet_.clear();
    partialCommRankSet_.resize(1);
    for (u32 i = roundIdx * sdmaConcurrentNum_; i < (roundIdx * sdmaConcurrentNum_ + groupRankSize); i++) {
        u32 readRemoteRank = podStartRank_ + (rankIdxInPod_ + devNumInlocalPod_ - i) % devNumInlocalPod_;
        u32 sendRemoteRank = podStartRank_ + (rankIdxInPod_ + i) % devNumInlocalPod_;
        partialCommRankSet_[0].push_back(std::make_pair(readRemoteRank, sendRemoteRank));
        HCCL_DEBUG("[AlltoAllVDirectFullMesh][UpdatePartialCommunicationRankSetPairWise] userRank [%u] i[%u]" \
            "readRemoteRank[%u] writeRemoteRank[%u]", userRank_, i, readRemoteRank, sendRemoteRank);
    }
    HCCL_DEBUG("[AlltoAllVDirectFullMesh][UpdatePartialCommunicationRankSetPairWise] partialCommRankSet_ size[%zu]",
        partialCommRankSet_[0].size());
}

void AlltoAllVDirectFullMesh::UpdatePartialCommunicationRankSet(u32 roundIdx, u32 groupRankSize,
    std::vector<std::vector<std::pair<u32,u32>>> &partialCommRankSet)
{
    partialCommRankSet.clear();
    partialCommRankSet.resize(RANK_SET_COMPUTE_CONST + 1);
    u32 pairNumPerRound = sdmaConcurrentNum_ / RANK_SET_COMPUTE_CONST;
    u32 pairSize = (groupRankSize < sdmaConcurrentNum_) ?
        (groupRankSize + RANK_SET_COMPUTE_CONST - 1) / RANK_SET_COMPUTE_CONST: pairNumPerRound;
    for (u32 i = roundIdx * pairNumPerRound + 1;
         i < (roundIdx * pairNumPerRound + pairSize + 1); i++) {
        u32 leftRemoteRank = podStartRank_ + (rankIdxInPod_ + devNumInlocalPod_ - i) % devNumInlocalPod_;
        u32 rightRemoteRank = podStartRank_ + (rankIdxInPod_ + i) % devNumInlocalPod_;
        if (leftRemoteRank == rightRemoteRank) {
            partialCommRankSet[2].push_back(std::make_pair(leftRemoteRank, leftRemoteRank));
        } else {
            partialCommRankSet[0].push_back(std::make_pair(leftRemoteRank, leftRemoteRank));
            partialCommRankSet[1].push_back(std::make_pair(rightRemoteRank, rightRemoteRank));
        }
        HCCL_DEBUG("[AlltoAllVDirectFullMesh][UpdatePartialCommunicationRankSet] round[%u] userRank [%u] i[%u]" \
            "read/write leftRemoteRank[%u] rightRemoteRank[%u]", roundIdx, userRank_, i, leftRemoteRank, rightRemoteRank);
    }
    HCCL_DEBUG("[AlltoAllVDirectFullMesh][UpdatePartialCommunicationRankSet] round[%u] partialCommRankSet_ total size[%zu]",
        roundIdx, partialCommRankSet[0].size() + partialCommRankSet[1].size() + partialCommRankSet[2].size());
}

// 主流只需要通知当前子步骤需要收发数据的 SDMA 流，减少同步开销
HcclResult AlltoAllVDirectFullMesh::NotifySubStreamStart()
{
    for (u32 streamIndex = 0; streamIndex < subStreamReadInfo_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, sdmaMeshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(sdmaSubStream_[streamIndex], dispatcher_, sdmaMeshSignalSubToMain_[streamIndex],
            INVALID_VALUE_STAGE));
    }
    for (u32 streamIndex = 0; streamIndex < subStreamReadInfo_.size(); streamIndex++) {
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, sdmaSubStream_[streamIndex], dispatcher_));
    }
    HCCL_DEBUG("[AlltoAllVDirectFullMesh][NotifySubStreamStart] userRank [%u] main stream notify sdma stream [%s]",
        userRank_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::WaitSubStreamFinish()
{
    for (u32 streamIndex = 0; streamIndex < subStreamReadInfo_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(sdmaSubStream_[streamIndex], dispatcher_, sdmaMeshSignalMainToSub_[streamIndex],
            INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, sdmaMeshSignalMainToSub_[streamIndex],
            INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG("[AlltoAllVDirectFullMesh][WaitSubStreamFinish] userRank [%u] main stream wait sdma stream [%s]",
        userRank_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::NotifyLocalSubStreamStart()
{
    for (u32 streamIndex = 0; streamIndex < subStreamSendInfo_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, localSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(localSubStream_[streamIndex], dispatcher_, localSignalSubToMain_[streamIndex],
            INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::WaitLocalSubStreamFinish()
{
    for (u32 streamIndex = 0; streamIndex < subStreamSendInfo_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(localSubStream_[streamIndex], dispatcher_, localSignalMainToSub_[streamIndex],
            INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, localSignalMainToSub_[streamIndex],
            INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

u32 AlltoAllVDirectFullMesh::CalcNumSubStep()
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;

    sendNumSubStep_.clear();
    recvNumSubStep_.clear();
    u32 numSubStep = 0;

    for (u32 destRank = podStartRank_; destRank < podStartRank_ + devNumInlocalPod_; destRank++) {
        if (destRank == userRank_) {
            continue;
        }

        u32 currRankSendSubStep = ((localSendRecvInfo.sendLength[destRank] + sdmaDataBlockSize_- 1) / sdmaDataBlockSize_);
        sendNumSubStep_[destRank] = currRankSendSubStep;

        u32 currRankRecvSubStep = ((localSendRecvInfo.recvLength[destRank] + sdmaDataBlockSize_- 1) / sdmaDataBlockSize_);
        recvNumSubStep_[destRank] = currRankRecvSubStep;
        HCCL_DEBUG("[AlltoAllVDirectFullMesh][CalcNumSubStep] userRank [%u] currRankSendSubStep[%u]" \
        "currRankRecvSubStep[%u]", userRank_, currRankSendSubStep, currRankRecvSubStep);
        numSubStep = std::max(numSubStep, std::max(currRankSendSubStep, currRankRecvSubStep));
    }
    HCCL_DEBUG("[AlltoAllVDirectFullMesh][CalcNumSubStep] userRank [%u] max communication step[%u]",
        userRank_, numSubStep);
    return numSubStep;
}

HcclResult AlltoAllVDirectFullMesh::NotifyRemoteRankStart(u32 step)
{
    u32 streamIndex = 0;
    for (auto& sendRecvSide : partialCommRankSet_) {
        for (auto& sendRecvPair : sendRecvSide) {
            u32 recvRank = sendRecvPair.first;
            u32 sendRank = sendRecvPair.second;
            if (sendRank == userRank_) {
                continue;
            }
            const std::vector<ReadDataBlock>& readInfo = subStreamReadInfo_[recvRank];
            const std::vector<SendDataBlock>& sendInfo = subStreamSendInfo_[sendRank];
            Stream& currStream = sdmaSubStream_[streamIndex];
            const LINK& readTransport = links_[recvRank];
            const LINK& sendTransport = links_[sendRank];
            if (step < sendInfo.size()) {
                CHK_RET(sendTransport->TxAck(currStream));
            }

            if (step < readInfo.size()) {
                CHK_RET(readTransport->RxAck(currStream));
            }
            streamIndex ++;
        }
    }
    HCCL_INFO("[AlltoAllVDirectFullMesh][NotifyRemoteRankStart] done");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::SDMAwithRemoteRankAndNotifyEnd(u32 step)
{
    u32 streamIndex = 0;
    for (auto& sendRecvSide : partialCommRankSet_) {
        for (auto& sendRecvPair : sendRecvSide) {
            u32 recvRank = sendRecvPair.first;
            u32 sendRank = sendRecvPair.second;
            if (sendRank == userRank_) {
                continue;
            }
            const std::vector<ReadDataBlock>& readInfo = subStreamReadInfo_[recvRank];
            const std::vector<SendDataBlock>& sendInfo = subStreamSendInfo_[sendRank];
            Stream& currStream = sdmaSubStream_[streamIndex];
            const LINK& readTransport = links_[recvRank];
            const LINK& sendTransport = links_[sendRank];
            if (step < readInfo.size()) {
                const LINK& intraNeighboorTransport = links_[recvRank];
                CHK_PTR_NULL(intraNeighboorTransport);
                void* remDMAMemPtr = nullptr;
                CHK_RET(intraNeighboorTransport->GetRemoteMem(UserMemType::INPUT_MEM, &remDMAMemPtr));
                DeviceMem remoteCCLInMem = DeviceMem::create(static_cast<u8 *>(remDMAMemPtr), cclInMem_.size());
                DeviceMem srcMem = remoteCCLInMem.range(readInfo[step].remoteOffset, readInfo[step].recvLen);
                DeviceMem dstMem = userOutput_.range(readInfo[step].recvOffset, readInfo[step].recvLen);
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, currStream,
                    readTransport->GetRemoteRank(), readTransport->GetLinkType()));
                HCCL_DEBUG("[AlltoAllVDirectFullMesh][SendRecvData] userRank [%u], recvRank[%u], sendRank[%u]," \
                    "sdma stream [%llu] read data from remote offset [%llu] len [%llu] to local [%llu]",
                    userRank_,  recvRank, sendRank, streamIndex, readInfo[step].remoteOffset,
                    readInfo[step].recvLen, readInfo[step].recvOffset); 
                CHK_RET(readTransport->TxDataSignal(currStream));
            }
            if (step < sendInfo.size()) {
                CHK_RET(sendTransport->RxDataSignal(currStream));
            }
            streamIndex ++;
        }
    }
    HCCL_INFO("[AlltoAllVDirectFullMesh][SDMAwithRemoteRankAndNotifyEnd] done");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::SendRecvData(u32 step, u32 roundIdx)
{
    HCCL_DEBUG("[AlltoAllVDirectFullMesh][SendRecvData] userRank [%u] sdma stream [%s] wait main stream",
        userRank_, GetStreamIndexString().c_str());
    CHK_RET(NotifyRemoteRankStart(step));
    CHK_RET(WaitSubStreamFinish());
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(NotifySubStreamStart());
    if (isBigCount_ && (roundIdx < commRounds_ - 1)) {
        CHK_RET(NotifyLocalSubStreamStart());
        CHK_RET(PrepareIntraData(step, nextSubStreamSendInfo_));
    }
    CHK_RET(SDMAwithRemoteRankAndNotifyEnd(step));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::LocalCopy()
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;
    DeviceMem src = userInput_.range(localSendRecvInfo.sendOffset[userRank_],
        localSendRecvInfo.sendLength[userRank_]);
    DeviceMem dst = userOutput_.range(localSendRecvInfo.recvOffset[userRank_],
        localSendRecvInfo.recvLength[userRank_]);
    HCCL_DEBUG("[AlltoAllVDirectFullMesh][LocalCopy]userRank [%u] copy from userInput [%llu] len [%llu]" \
        "to userOutput [%llu] dstLen[%llu]", userRank_, localSendRecvInfo.sendOffset[userRank_],
        localSendRecvInfo.sendLength[userRank_],
        localSendRecvInfo.recvOffset[userRank_],
        localSendRecvInfo.recvLength[userRank_]);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::RunGroupFullMeshAlltoall(u32 roundIdx, u32 step)
{
    UpdateOpBaseSubStreamInfo(roundIdx);
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    if (isBigCount_ && (roundIdx == 0) ) {
        CHK_RET(NotifyLocalSubStreamStart());
        CHK_RET(PrepareIntraData(step, subStreamSendInfo_));
        CHK_RET(WaitLocalSubStreamFinish());
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    } else if (!isBigCount_) {
        CHK_RET(PrepareIntraData(step, subStreamSendInfo_));
    }
    CHK_RET(NotifySubStreamStart());
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(SendRecvData(step, roundIdx));
    if (step == 0 && !islocalCpyDone_) {
        CHK_RET(LocalCopy());
        islocalCpyDone_ = true;
    }
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(WaitSubStreamFinish());
    if (isBigCount_ && (roundIdx < commRounds_ - 1)) {
        CHK_RET(WaitLocalSubStreamFinish());
    }
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    return HCCL_SUCCESS;
}

// 主流通知RDMA控制流启动
HcclResult AlltoAllVDirectFullMesh::MainNotifyRdmaControlStart()
{
    CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, rdmaControl2MainStreamNotify_, INVALID_VALUE_STAGE));
    CHK_RET(LocalNotify::Wait(rdmaSubStreams_[0], dispatcher_, rdmaControl2MainStreamNotify_, INVALID_VALUE_STAGE));
    return HCCL_SUCCESS;
}

// RDMA控制流通知主流任务完成
HcclResult AlltoAllVDirectFullMesh::RdmaControlNotifyMainFinish()
{
    CHK_RET(LocalNotify::Post(rdmaSubStreams_[0], dispatcher_, main2RdmaControlStreamNotify_, INVALID_VALUE_STAGE));
    CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, main2RdmaControlStreamNotify_, INVALID_VALUE_STAGE));
    return HCCL_SUCCESS;
}

// RDMA控制流通知从流启动任务
HcclResult AlltoAllVDirectFullMesh::RdmaControlNotifySubStart()
{
    for (u32 i = 1; i < rdmaSubStreams_.size(); i++) {
        CHK_RET(LocalNotify::Post(rdmaSubStreams_[0], dispatcher_, rdmaSub2ControlNotifies_[i-1], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(rdmaSubStreams_[i], dispatcher_, rdmaSub2ControlNotifies_[i-1], INVALID_VALUE_STAGE));
    }

    return HCCL_SUCCESS;
}

// 从流通知RDMA控制流任务结束
HcclResult AlltoAllVDirectFullMesh::SubNotifyRdmaControlFinish()
{
    for (u32 i = 1; i < rdmaSubStreams_.size(); i++) {
        CHK_RET(LocalNotify::Post(rdmaSubStreams_[i], dispatcher_, rdmaControl2SubNotifies_[i-1], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(rdmaSubStreams_[0], dispatcher_, rdmaControl2SubNotifies_[i-1], INVALID_VALUE_STAGE));
    }

    return HCCL_SUCCESS;
}

u32 AlltoAllVDirectFullMesh::GetNextDstRank(u32& curDstRank)
{
    if (curDstRank >= userRankSize_) {
        curDstRank = curDstRank % userRankSize_;
    }
    if (curDstRank == podStartRank_) {
        curDstRank += devNumInlocalPod_;
    }
    curDstRank = curDstRank % userRankSize_;
    return curDstRank++;
}

u32 AlltoAllVDirectFullMesh::GetPreSrcRank(u32& curDstRank)
{
    if (curDstRank == podStartRank_ + devNumInlocalPod_ - 1) {
        curDstRank = (curDstRank + userRankSize_ - devNumInlocalPod_) % userRankSize_;
    }

    if (curDstRank == 0) {
        curDstRank = userRankSize_ - 1;
        return 0;
    }
    return curDstRank--;
}

void AlltoAllVDirectFullMesh::GenRdmaSendInfo(u32 dstRank, std::vector<SendDataBlock>& sendInfo)
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;
    u64 sendOffset = localSendRecvInfo.sendOffset[dstRank];
    u64 sendLength = localSendRecvInfo.sendLength[dstRank];
    while (sendLength > 0) {
        u64 curSendLength = std::min(sendLength, rdmaDataBlockSize_);
        SendDataBlock sendData;
        sendData.userInOffset = sendOffset;
        sendData.sendLen = curSendLength;
        u32 index = dstRank % rdmaConcurrentNum_;
        sendData.scratchOffset = rdmaDataBlockSize_ * index;
        sendInfo.push_back(sendData);
        sendOffset += curSendLength;
        sendLength -= curSendLength;
        HCCL_DEBUG("[GenRdmaSendInfo] userRank[%u], dstRank[%u], sendData.userInOffset[%llu]," \
            "sendData.sendLen[%llu], sendData.scratchOffset[%llu]", userRank_, dstRank,
            sendData.userInOffset, sendData.sendLen, sendData.scratchOffset);
    }
    return;
}

void AlltoAllVDirectFullMesh::GenRdmaRecvInfo(u32 srcRank, std::vector<RecvDataBlock>& recvInfo)
{
    const SendRecvInfo& localSendRecvInfo = *localSendRecvInfoPtr_;
    u64 recvOffset = localSendRecvInfo.recvOffset[srcRank];
    u64 recvLength = localSendRecvInfo.recvLength[srcRank];
    while (recvLength > 0) {
        u64 curRecvLength = std::min(recvLength, rdmaDataBlockSize_);
        RecvDataBlock recvData;
        recvData.recvOffset = recvOffset;
        recvData.recvLen = curRecvLength;
        u32 index = srcRank % rdmaConcurrentNum_;
        recvData.scratchOffset = rdmaDataBlockSize_ * index + rdmaDataBlockSize_ * rdmaConcurrentNum_;
        recvInfo.push_back(recvData);
        recvOffset += curRecvLength;
        recvLength -= curRecvLength;
        HCCL_DEBUG("[GenRdmaRecvInfo] userRank[%llu], srcRank[%u], recvData.recvOffset[%llu]," \
            "recvData.recvLen[%llu], recvData.scratchOffset[%llu]", userRank_, srcRank,
            recvData.recvOffset, recvData.recvLen, recvData.scratchOffset);
    }
    return;
}

// 将数据从userIn拷贝到CCL out
HcclResult AlltoAllVDirectFullMesh::CopyDataForSend(u32 dstRank, std::vector<SendDataBlock>& sendInfo, u32 curStep, Stream strem)
{
    if (curStep >= sendInfo.size()) {
        return HCCL_SUCCESS;
    }
    DeviceMem src = userInput_.range(sendInfo[curStep].userInOffset, sendInfo[curStep].sendLen);
    DeviceMem dst = cclOutMem_.range(sendInfo[curStep].scratchOffset, sendInfo[curStep].sendLen);
    HCCL_DEBUG("[CopyDataForSend] userRank[%u], dstRank[%u], userInOffset[%llu], sendLen[%llu], scratchOffset[%llu]",
        userRank_, dstRank, sendInfo[curStep].userInOffset, sendInfo[curStep].sendLen, sendInfo[curStep].scratchOffset);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, strem));
    return HCCL_SUCCESS;
}

// 从流完成RDMA数据的收发
HcclResult AlltoAllVDirectFullMesh::SendRecvRdmaData(u32 dstRank, u32 srcRank, std::vector<SendDataBlock>& sendInfo,
    std::vector<RecvDataBlock>& recvInfo, u32 curStep, Stream strem)
{
    const LINK& sendTransport = links_[dstRank];
    const LINK& recvTransport = links_[srcRank];
    HCCL_DEBUG("[AlltoAllVDirectFullMesh][SendRecvRdmaData] userRank[%u], dstRank[%u], srcRank[%u]",
        userRank_, dstRank, srcRank);
    u32 minStep = std::min(sendInfo.size(), recvInfo.size());
    CHK_PTR_NULL(sendTransport);
    CHK_PTR_NULL(recvTransport);
    if (curStep < minStep) {
        CHK_RET(recvTransport->TxAck(strem));
        CHK_RET(sendTransport->RxAck(strem));
        u64 sendSrcOffset = (dstRank % rdmaConcurrentNum_) * rdmaDataBlockSize_;
        void* srcPtr = static_cast<u8 *>(cclOutMem_.ptr()) + sendSrcOffset;
        u32 dstIndex = userRank_ % rdmaConcurrentNum_;
        u64 sendDstOffset = (dstIndex + rdmaConcurrentNum_) * rdmaDataBlockSize_;
        CHK_RET(sendTransport->TxAsync(UserMemType::OUTPUT_MEM, sendDstOffset, srcPtr,
            sendInfo[curStep].sendLen, strem));

        u64 recvDstOffset = (srcRank % rdmaConcurrentNum_ + rdmaConcurrentNum_) * rdmaDataBlockSize_;
        void* dstPtr = static_cast<u8 *>(cclOutMem_.ptr()) + recvDstOffset;
        u64 recvSrcOffset = (userRank_ % rdmaConcurrentNum_) * rdmaDataBlockSize_;
        CHK_RET(recvTransport->RxAsync(UserMemType::OUTPUT_MEM, recvSrcOffset, dstPtr,
            recvInfo[curStep].recvLen, strem));
        CHK_RET(recvTransport->PostFinAck(strem));
        CHK_RET(sendTransport->WaitFinAck(strem));
        HCCL_DEBUG("[AlltoAllVDirectFullMesh][SendRecvRdmaData] sendSrcOffset[%llu], sendDstOffset[%llu]," \
            "recvDstOffset[%llu], recvSrcOffset[%llu], srcPtr[%p], dstPtr[%p]",sendSrcOffset,
            sendDstOffset, recvDstOffset, recvSrcOffset, srcPtr, dstPtr);
    } else if (curStep < sendInfo.size()) {
        CHK_RET(sendTransport->RxAck(strem));
        u64 sendSrcOffset = (dstRank % rdmaConcurrentNum_) * rdmaDataBlockSize_;
        void* srcPtr = static_cast<u8 *>(cclOutMem_.ptr()) + sendSrcOffset;
        u32 dstIndex = userRank_ % rdmaConcurrentNum_;
        u64 sendDstOffset = (dstIndex + rdmaConcurrentNum_) * rdmaDataBlockSize_;
        CHK_RET(sendTransport->TxAsync(UserMemType::OUTPUT_MEM, sendDstOffset, srcPtr,
            sendInfo[curStep].sendLen, strem));
        CHK_RET(sendTransport->WaitFinAck(strem));
    } else {
        CHK_RET(recvTransport->TxAck(strem));
        u64 recvDstOffset = (srcRank % rdmaConcurrentNum_ + rdmaConcurrentNum_) * rdmaDataBlockSize_;
        void* dstPtr = static_cast<u8 *>(cclOutMem_.ptr()) + recvDstOffset;
        u64 recvSrcOffset = (userRank_ % rdmaConcurrentNum_) * rdmaDataBlockSize_;
        CHK_RET(recvTransport->RxAsync(UserMemType::OUTPUT_MEM, recvSrcOffset, dstPtr,
            recvInfo[curStep].recvLen, strem));
        CHK_RET(recvTransport->PostFinAck(strem));
    }
    return HCCL_SUCCESS;
}

// 从流将接收到的数据拷贝到输出
HcclResult AlltoAllVDirectFullMesh::CopyRecvDataToOutput(u32 srcRank, std::vector<RecvDataBlock>& recvInfo,
    u32 curStep, Stream strem)
{
    if (curStep >= recvInfo.size()) {
        return HCCL_SUCCESS;
    }
    u64 srcOffset = (srcRank % rdmaConcurrentNum_ + rdmaConcurrentNum_) * rdmaDataBlockSize_;
    DeviceMem src = cclOutMem_.range(srcOffset, recvInfo[curStep].recvLen);
    DeviceMem dst = userOutput_.range(recvInfo[curStep].recvOffset, recvInfo[curStep].recvLen);
    HCCL_DEBUG("[AlltoAllVDirectFullMesh][CopyRecvDataToOutput] userRank[%u], srcRank[%u], srcOffset[%llu]," \
        "recvInfo[curStep].recvOffset[%llu], recvLen[%llu]", userRank_, srcRank, srcOffset,
        recvInfo[curStep].recvOffset, recvInfo[curStep].recvLen);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, strem));
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::ProcessSingleGroupRdmaData(std::vector<u32>& dstRanks, std::vector<u32>& srcRanks)
{
    for (u32 index = 0; index < dstRanks.size(); index++) {
        u32 dstRank = dstRanks[index];
        u32 srcRank = srcRanks[index];
        Stream strem = rdmaSubStreams_[index + 1];

        std::vector<SendDataBlock> sendInfo;
        std::vector<RecvDataBlock> recvInfo;
        GenRdmaSendInfo(dstRank, sendInfo);
        GenRdmaRecvInfo(srcRank, recvInfo);
        u32 totalStep = std::max(sendInfo.size(), recvInfo.size());

        for (u32 curStep = 0; curStep < totalStep; curStep++) {
            CHK_RET(CopyDataForSend(dstRank, sendInfo, curStep, strem));
            CHK_RET(SendRecvRdmaData(dstRank, srcRank, sendInfo, recvInfo, curStep, strem));
            CHK_RET(CopyRecvDataToOutput(srcRank, recvInfo, curStep, strem));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::ProcessRdmaData()
{
    // RDMA通信轮次
    u32 rdmaRoundNum = (totalRdmaRankNum_ + rdmaConcurrentNum_ - 1) / rdmaConcurrentNum_;

    u32 leftRankNum = totalRdmaRankNum_;
    u32 curSrcRank = INVALID_VALUE_RANKID;
    u32 curDstRank = INVALID_VALUE_RANKID;
    if (isSuPodAsym_) {
        for (u32 i = 0; i < userRankSize_; i++) {
            if (i < podStartRank_ || i > podEndRank_) {
                curSrcRank = i;
                curDstRank = i;
                break;
            }
        }
    } else {
        curDstRank = (userRank_ + devNumInlocalPod_) % userRankSize_;
        curSrcRank = (userRank_ + userRankSize_ - devNumInlocalPod_) % userRankSize_;
    }

    for (u32 round = 0; round < rdmaRoundNum; round++) {
        u32 curProcessRankNum = leftRankNum >= rdmaConcurrentNum_ ? rdmaConcurrentNum_ : leftRankNum;
        leftRankNum -= curProcessRankNum;

        std::vector<u32> dstRanks;
        std::vector<u32> srcRanks;
        for (u32 i = 0; i < curProcessRankNum; i++) {
            dstRanks.push_back(GetNextDstRank(curDstRank));
            if (isSuPodAsym_) {
                srcRanks.push_back(dstRanks.back());
            } else {
                srcRanks.push_back(GetPreSrcRank(curSrcRank));
            }
        }
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, rdmaSubStreams_[0], dispatcher_));
        CHK_RET(RdmaControlNotifySubStart());
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, rdmaSubStreams_[0], dispatcher_));
        CHK_RET(ProcessSingleGroupRdmaData(dstRanks, srcRanks));
        CHK_RET(SubNotifyRdmaControlFinish());
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, rdmaSubStreams_[0], dispatcher_));
    }
    HCCL_INFO("[AlltoAllVDirectFullMesh][ProcessRdmaData] done");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::RunRDMA()
{
    // 先启动RDMA通信
    CHK_RET(MainNotifyRdmaControlStart());
    CHK_RET(ProcessRdmaData());
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(LocalCopy());
    islocalCpyDone_ = true;
    HCCL_INFO("[AlltoAllVDirectFullMesh][RunRDMA] finished.");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::RunSDMATasks(u32 roundIdx, u32 step, u32 groupRankSize, u32 leftRankSize)
{
    if (isBigCount_) {
        if (roundIdx == 0) {
            UpdatePartialCommunicationRankSet(roundIdx, groupRankSize, partialCommRankSet_);
        }
        if (roundIdx < commRounds_ - 1) {
            u32 nextgroupRankSize = (leftRankSize - groupRankSize > sdmaConcurrentNum_) ?
                sdmaConcurrentNum_ : leftRankSize - groupRankSize;
            UpdatePartialCommunicationRankSet(roundIdx + 1, nextgroupRankSize, nextPartialCommRankSet_);
        }
        CHK_RET(RunGroupFullMeshAlltoall(roundIdx, step));

        if (roundIdx < commRounds_ - 1) {
            partialCommRankSet_ = nextPartialCommRankSet_;
            subStreamSendInfo_ = nextSubStreamSendInfo_;
            subStreamReadInfo_ = nextSubStreamReadInfo_;
        }
        CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, localSubStream_));
    } else {
        UpdatePartialCommunicationRankSet(roundIdx, groupRankSize, partialCommRankSet_);
        CHK_RET(RunGroupFullMeshAlltoall(roundIdx, step));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::RunSDMAFineGrained(u32 totalStep, HcclOpMetaInfoDef &opMeta){
    if (totalStep > 1){
        //细粒度场景不支持切分
        HCCL_ERROR("[AlltoAllVDirectFullMesh][RunSDMAFineGrained] AlltoAllV is not supported when totalStep[%u] > 1, "\
            "HCCL buffer is insufficient. stepSize : %u ", totalStep, algOpContext_.mc2Handler.stepSize);
        return HCCL_E_NOT_SUPPORT;
    } else if (totalStep == 0){
        //totalStep不需要通信，但是需要适配高阶API wait/write
        for (u32 roundIdx = 0; roundIdx < commRounds_; roundIdx++) {
            CHK_RET(mc2HandlerPub.Mc2WaitValue(dispatcher_, mainStream_, &(algOpContext_.mc2Handler), roundIdx));
            CHK_RET(mc2HandlerPub.Mc2WriteValue(dispatcher_, mainStream_, &(algOpContext_.mc2Handler)));
            HCCL_INFO("[AlltoAllVDirectFullMesh][RunSDMAFineGrained] step is 0 finished.");
        }
    } else {
        // totalStep == 1 细粒度修改
        u32 leftRankSize = devNumInlocalPod_; // leftRankSize中去掉本卡
        for (u32 roundIdx = 0; roundIdx < commRounds_ && leftRankSize > 0; roundIdx++) {
            CHK_RET(mc2HandlerPub.Mc2WaitValue(dispatcher_, mainStream_, &(algOpContext_.mc2Handler), roundIdx));
            CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
            u32 groupRankSize = (leftRankSize > sdmaConcurrentNum_) ? sdmaConcurrentNum_ : leftRankSize;
            UpdateRemoteRankSet(roundIdx, groupRankSize);
            CHK_RET(RunGroupFullMeshAlltoall(roundIdx, 0));
            leftRankSize -= groupRankSize;
            CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, sdmaSubStream_));
            CHK_RET(mc2HandlerPub.Mc2WriteValue(dispatcher_, mainStream_, &(algOpContext_.mc2Handler)));
        }
        HCCL_INFO("[AlltoAllVDirectFullMesh][RunSDMAFineGrained] fine-grained finished.");
        return HCCL_SUCCESS;
    }

    if (totalStep == 0 && !islocalCpyDone_) {
        CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
        CHK_RET(LocalCopy());
        islocalCpyDone_ = true;
        CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, sdmaSubStream_));
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[AlltoAllVDirectFullMesh][RunSDMAFineGrained] finished.");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::RunSDMA(HcclOpMetaInfoDef &opMeta)
{
    u32 totalStep = CalcNumSubStep();
 
    // 计算每个rank分组fullmesh后需要通信的轮次，向上取整
    commRounds_ = (devNumInlocalPod_ + sdmaConcurrentNum_ - 1) / sdmaConcurrentNum_;
    HCCL_DEBUG("[AlltoAllVDirectFullMesh][RunSDMA] userRank [%u] communication rounds[%llu] totalStep %u stepSize %u",
        userRank_, commRounds_, totalStep, algOpContext_.mc2Handler.stepSize);

    if (UNLIKELY(algOpContext_.mc2Handler.stepSize > 0)){
        CHK_RET(RunSDMAFineGrained(totalStep, opMeta));
    } else {
        if (totalStep == 0 && !islocalCpyDone_) {
            CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
            CHK_RET(LocalCopy());
            islocalCpyDone_ = true;
            CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, sdmaSubStream_));
            return HCCL_SUCCESS;
        }

        for (u32 step = 0; step < totalStep; step++) {
            u32 leftRankSize = devNumInlocalPod_ - 1; // leftRankSize中去掉本卡
            for (u32 roundIdx = 0; roundIdx < commRounds_ && leftRankSize > 0; roundIdx++) {
                CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
                u32 groupRankSize = (leftRankSize > sdmaConcurrentNum_) ? sdmaConcurrentNum_ : leftRankSize;
                CHK_RET(RunSDMATasks(roundIdx, step, groupRankSize, leftRankSize));
                leftRankSize -= groupRankSize;
                CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, sdmaSubStream_));
            }
        }
    }
    
    HCCL_INFO("[AlltoAllVDirectFullMesh][RunSDMA] finished.");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::RunAsync()
{   
    HcclOpMetaInfoDef opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::ZCOPY, cclInMem_.size(), true);
    CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));

    if (algOpContext_.mc2Handler.stepSize > 0){
        if(algOpContext_.mc2Handler.stepSize > userRankSize_ || userRankSize_ % algOpContext_.mc2Handler.stepSize != 0){
            HCCL_ERROR("[AlltoAllVDirectFullMesh][RunAsync] Step size should be less than or equal to the rank size, "\
                        "and the rank size should be a multiple of the step size, but the step size is [%u] and the rank size is [%u].",
                        algOpContext_.mc2Handler.stepSize, userRankSize_);
            return HCCL_E_PARA;
        }
        if(userRankSize_ == 1){
            HCCL_INFO("[AlltoAllVDirectFullMesh][RunAsync] AlltoAllV do localcopy with 1 rank");
            CHK_RET(mc2HandlerPub.Mc2WaitValue(dispatcher_, mainStream_, &(algOpContext_.mc2Handler), 0));
            CHK_RET(LocalCopy());
            CHK_RET(mc2HandlerPub.Mc2WriteValue(dispatcher_, mainStream_, &(algOpContext_.mc2Handler)));
            return HCCL_SUCCESS;
        }
    }

    if (userRankSize_ == 1) {
        HCCL_INFO("[AlltoAllVDirectFullMesh][RunAsync] do localcopy with 1 rank");
        CHK_RET(LocalCopy());
        return HCCL_SUCCESS;
    }

    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    if (totalRdmaRankNum_ > 0) {
        CHK_RET(RunRDMA());
    }

    CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, rdmaSubStreams_));

    if (devNumInlocalPod_ > 1) {
        CHK_RET(RunSDMA(opMeta));
    }

    if (totalRdmaRankNum_ > 0) {
        // 等待RDMA通信结束
        CHK_RET(InitTask(dispatcher_, mainStream_, opMeta.isEnableCache, opMeta.GetCacheKey()));
        CHK_RET(RdmaControlNotifyMainFinish());
        CHK_RET(LaunchTaskExtend(dispatcher_, mainStream_, rdmaSubStreams_));
    }

    HCCL_INFO("[AlltoAllVDirectFullMesh][RunAsync] finished.");
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVDirectFullMesh::GetNslbAdjInfo(const u32 rank, const u32 rankSize,
                                                   const std::vector<LINK> &links, AdjInfo& nslbAdjInfo)
{
    if (rankSize == 1) {
        return HCCL_SUCCESS;
    }

    u32 devNumInlocalPod = nslbAdjInfo.dstRankNum;
    u32 totalRdmaRankNum = rankSize - devNumInlocalPod;

    u32 rdmaConcurrentNum = (totalRdmaRankNum > ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE) ?
        (ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE) : (totalRdmaRankNum);
    if (rdmaConcurrentNum == 0) {
        return HCCL_SUCCESS;
    }
    // RDMA通信轮次
    u32 rdmaRoundNum = (totalRdmaRankNum + rdmaConcurrentNum - 1) / rdmaConcurrentNum;
    if (rdmaRoundNum == 0) {
        return HCCL_SUCCESS;
    }
    u32 currStage = rank / devNumInlocalPod;

    for (u32 step = 0; step < rdmaRoundNum; step++) {
        u32 sendTo =(rank + devNumInlocalPod + step) % rankSize;
        u32 sendToStag = sendTo / devNumInlocalPod;
        if(currStage == sendToStag) {
            //此时认为时同一个超节点内通讯
            sendTo += devNumInlocalPod;
        }
        NslbDpAdjInfo adjInfoStep = {0};
        adjInfoStep.dstLocalRankId = sendTo;
        adjInfoStep.phaseId = step + 1;
        adjInfoStep.rev = 0;
        nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
    }
    nslbAdjInfo.dstRankNum = nslbAdjInfo.nsAdjInfo.size();
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_2_ALL_V_DIRECT_FULL_MESH, AlltoAllVDirectFullMesh);
} // namespace hccl