/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_reduce_mesh_1D_two_shot.h"

namespace ops_hccl {

InsTempAllReduceMesh1DTwoShot::InsTempAllReduceMesh1DTwoShot(const OpParam& param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks) : InsAlgTemplateBase(param, rankId, subCommRanks){}

InsTempAllReduceMesh1DTwoShot::~InsTempAllReduceMesh1DTwoShot(){}

u64 InsTempAllReduceMesh1DTwoShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void) inBuffType;
    (void) outBuffType;
    u64 multiple = 2;  // multiple=1且数据非均衡切分时，hcclBuffer会不足，因此用2
    HCCL_INFO("[InsTempAllReduceMesh1DTwoShot] Ccl Buffer multiple is [%llu].", multiple);
    return multiple;
}

HcclResult InsTempAllReduceMesh1DTwoShot::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    CHK_RET(GetRes(resourceRequest));

    std::vector<HcclChannelDesc> channelReq;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelReq));
    resourceRequest.channels.push_back(channelReq);

    HCCL_INFO("[InsTempAllReduceMesh1DTwoShot] Calculate resource finished."
        "resource request: threadNum[%u], main thread notifyNum[%u], channelNum[%u]",
        resourceRequest.slaveThreadNum + 1, resourceRequest.notifyNumOnMainThread,
        resourceRequest.channels.at(0).size());
    return HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DTwoShot::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = templateRankSize_ - 1;

    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;

    return HCCL_SUCCESS;
}

u64 InsTempAllReduceMesh1DTwoShot::GetThreadNum() const
{
    // 需要rankSize个线程并行
    return templateRankSize_;
}

void InsTempAllReduceMesh1DTwoShot::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 slaveThreadNum = threadNum_ - 1;
    notifyIdxMainToSub.assign(slaveThreadNum, 0);  // 从线程全部用第0个Notify等待主线程的同步信号
}

void InsTempAllReduceMesh1DTwoShot::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 slaveThreadNum = threadNum_ - 1;
    for (u32 notifyIdx = 0; notifyIdx < slaveThreadNum; ++notifyIdx) {
        notifyIdxSubToMain.emplace_back(notifyIdx);
    }
}

HcclResult InsTempAllReduceMesh1DTwoShot::KernelRun(const OpParam& param,
    const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("========== [MiniTask] KernelRun START ==========");

    threadNum_ = templateResource.threads.size();
    CHK_PRT_RET(threadNum_ != templateRankSize_,
        HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][KernelRun] thread num is invalid, need[%u], actual[%u].",
            templateRankSize_, threadNum_), HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(subCommRanks_.size() == 0,
        HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][KernelRun] subCommRanks is empty."),
        HcclResult::HCCL_E_INTERNAL);
    rankList_ = subCommRanks_.at(0);

    CHK_PRT_RET(rankList_.size() != templateRankSize_,
        HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][KernelRun] rank[%u] count is invalid in rank list.", myRank_),
        HcclResult::HCCL_E_INTERNAL);

    // 获取当前rank在rank列表中的序号
    CHK_RET(GetAlgRank(myRank_, rankList_, myRankIdx_));

    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    needAicpuReduce_ =
        dataType_ == HcclDataType::HCCL_DATA_TYPE_INT64 || dataType_ == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        dataType_ == HcclDataType::HCCL_DATA_TYPE_FP64 || param.reduceType == HcclReduceOp::HCCL_REDUCE_PROD;

    // --- 日志：初始参数 ---
    HCCL_INFO("[MiniTask][KernelRun] myRank=%u, myRankIdx=%u, rankSize=%u, threadNum=%u",
        myRank_, myRankIdx_, templateRankSize_, threadNum_);
    HCCL_INFO("[MiniTask][KernelRun] dataType=%d, dataTypeSize=%u, count=%llu, processSize=%llu",
        (int)dataType_, dataTypeSize_, count_, processSize_);
    HCCL_INFO("[MiniTask][KernelRun] reduceOp=%d, needAicpuReduce=%d",
        (int)param.reduceType, (int)needAicpuReduce_);
    HCCL_INFO("[MiniTask][KernelRun] inputPtr=%p, outputPtr=%p, hcclBuff=%p",
        tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.hcclBuff.addr);

    if (count_ == 0) {
        HCCL_WARNING("[InsTempAllReduceMesh1DTwoShot][KernelRun] data count is 0.");
        return HcclResult::HCCL_SUCCESS;
    }

    // 数据切片
    CHK_RET(SplitData());
    CHK_PRT_RET(sliceInfoList_.size() != templateRankSize_,
        HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][KernelRun] slice num[%u] is not equal to rank size[%u].",
            sliceInfoList_.size(), templateRankSize_), HcclResult::HCCL_E_INTERNAL);

    HCCL_INFO("[MiniTask][KernelRun] --- Phase 1/2: RunReduceScatter BEGIN ---");
    // TwoShot算法，第一步ReduceScatter
    CHK_RET(RunReduceScatter(param, tempAlgParams, templateResource.channels, templateResource.threads));
    HCCL_INFO("[MiniTask][KernelRun] --- Phase 1/2: RunReduceScatter DONE ---");

    HCCL_INFO("[MiniTask][KernelRun] --- Phase 2/2: RunAllGather BEGIN ---");
    // TwoShot算法，第二步AllGather
    CHK_RET(RunAllGather(tempAlgParams, templateResource.channels, templateResource.threads));
    HCCL_INFO("[MiniTask][KernelRun] --- Phase 2/2: RunAllGather DONE ---");

    HCCL_INFO("========== [MiniTask] KernelRun FINISHED (rank=%u) ==========", myRank_);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DTwoShot::SplitData()
{
    u32 sliceNum = templateRankSize_;
    sliceInfoList_.clear();
    sliceInfoList_.reserve(sliceNum);

    u64 sliceCount = RoundUp(count_, sliceNum);
    u64 sliceSize = sliceCount * dataTypeSize_;

    HCCL_INFO("[MiniTask][SplitData] sliceNum=%u, sliceCount=%llu, sliceSize=%llu, count=%llu",
        sliceNum, sliceCount, sliceSize, count_);

    u64 offsetCount = 0;
    u64 offsetSize = 0;
    for (u32 sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
        if (count_ - offsetCount > sliceCount) {
            sliceInfoList_.emplace_back(offsetSize, sliceSize, sliceCount);
            offsetCount += sliceCount;
            offsetSize = offsetCount * dataTypeSize_;
        } else {
            u64 curSliceCount = count_ - offsetCount;
            u64 curSliceSize = curSliceCount * dataTypeSize_;
            sliceInfoList_.emplace_back(offsetSize, curSliceSize, curSliceCount);
            offsetCount = count_;
            offsetSize = offsetCount * dataTypeSize_;
        }
    }

    for (u32 i = 0; i < sliceInfoList_.size(); ++i) {
        HCCL_INFO("[MiniTask][SplitData]   slice[%u]: offset=%llu, size=%llu, count=%llu",
            i, sliceInfoList_.at(i).offset, sliceInfoList_.at(i).size, sliceInfoList_.at(i).count);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DTwoShot::RunReduceScatter(const OpParam& param,
                                                           const TemplateDataParams &tempAlgParams,
                                                           const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                           const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[MiniTask][RunReduceScatter] rank=%u BEGIN, threads=%zu", myRank_, threads.size());

    // 主线程向从线程发送启动信号
    PreSync(threads);
    HCCL_INFO("[MiniTask][RunReduceScatter] rank=%u PreSync done, entering ScatterData", myRank_);

    CHK_RET(ScatterData(tempAlgParams, channels, threads));
    HCCL_INFO("[MiniTask][RunReduceScatter] rank=%u ScatterData done, entering PostSync", myRank_);

    // 从线程往主线程返回结束信号
    PostSync(threads);
    HCCL_INFO("[MiniTask][RunReduceScatter] rank=%u PostSync done", myRank_);

    // 增加thread synchronize以支持64类数据类型
    if (needAicpuReduce_) {
        HCCL_INFO("[MiniTask][RunReduceScatter] rank=%u needAicpuReduce, batch sync start", myRank_);
        // 启动任务并等待所有threads任务执行完成
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        for (const auto &thread : threads) {
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
        }
        HCCL_INFO("[MiniTask][RunReduceScatter] rank=%u needAicpuReduce batch sync done", myRank_);
    }

    // 将数据reduce到第0片数据的位置
    HCCL_INFO("[MiniTask][RunReduceScatter] rank=%u entering ReduceData", myRank_);
    CHK_RET(ReduceData(tempAlgParams, threads));
    HCCL_INFO("[MiniTask][RunReduceScatter] rank=%u ReduceData done", myRank_);

    HCCL_INFO("[MiniTask][RunReduceScatter] rank=%u END", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DTwoShot::ScatterData(const TemplateDataParams &tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    void* localInBuffPtr = tempAlgParams.buffInfo.inputPtr;
    void* localHcclBuffPtr = tempAlgParams.buffInfo.hcclBuff.addr;
    u64 inBuffBaseOffset = tempAlgParams.buffInfo.inBuffBaseOff;
    u64 hcclBuffBaseOffset = tempAlgParams.buffInfo.hcclBuffBaseOff;

    u64 recvSize = sliceInfoList_.at(myRankIdx_).size;
    u64 recvCount = sliceInfoList_.at(myRankIdx_).count;
    u64 recvOffset = sliceInfoList_.at(myRankIdx_).offset;

    HCCL_INFO("[MiniTask][ScatterData] rank=%u(myRankIdx=%u) BEGIN, localInBuff=%p, localHcclBuff=%p, "
        "inBuffBaseOff=%llu, hcclBuffBaseOff=%llu",
        myRank_, myRankIdx_, localInBuffPtr, localHcclBuffPtr, inBuffBaseOffset, hcclBuffBaseOffset);
    HCCL_INFO("[MiniTask][ScatterData] rank=%u my recv: offset=%llu size=%llu count=%llu",
        myRank_, recvOffset, recvSize, recvCount);

    for (u32 remoteIdx = 0; remoteIdx < templateRankSize_; ++remoteIdx) {
        u64 sendSize = sliceInfoList_.at(remoteIdx).size;
        u64 sendCount = sliceInfoList_.at(remoteIdx).count;
        u64 sendOffset = sliceInfoList_.at(remoteIdx).offset;

        // 发送和接收数据量都为0的时候，既不发送也不接收
        if (sendSize == 0 && recvSize == 0) {
            HCCL_INFO("[MiniTask][ScatterData] rank=%u -> remoteIdx=%u: both zero, skip", myRank_, remoteIdx);
            continue;
        }

        // 数据片序号等于自身rank序号时，本地拷贝数据
        if (remoteIdx == myRankIdx_) {
            HCCL_INFO("[MiniTask][ScatterData] rank=%u -> remoteIdx=%u: LocalCopy myself, "
                "src(off=%llu size=%llu) -> dst(hcclOff=%llu size=%llu)",
                myRank_, remoteIdx, inBuffBaseOffset + sendOffset, sendSize,
                hcclBuffBaseOffset + remoteIdx * recvSize, recvSize);
            DataSlice copySrcSlice(localInBuffPtr, inBuffBaseOffset + sendOffset, sendSize, sendCount);
            DataSlice copyDstSlice(localHcclBuffPtr, hcclBuffBaseOffset + remoteIdx * recvSize, recvSize, recvCount);
            CHK_PRT_RET(LocalCopy(threads.at(remoteIdx), copySrcSlice, copyDstSlice),
                HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][ScatterData] LocalCopy failed."),
                HcclResult::HCCL_E_INTERNAL);
            continue;
        }

        // 数据片序号不等于自身rank序号时，跨rank发送和接收数据
        u32 remoteRank = rankList_.at(remoteIdx);
        const ChannelInfo &sendRecvChannel = channels.at(remoteRank).at(0);

        void* remoteInBuffPtr = sendRecvChannel.remoteCclMem.addr;
        void* remoteHcclBuffPtr = sendRecvChannel.remoteCclMem.addr;

        DataSlice sendSrcSlice(localInBuffPtr, inBuffBaseOffset + sendOffset, sendSize, sendCount);
        DataSlice sendDstSlice(remoteHcclBuffPtr, hcclBuffBaseOffset + myRankIdx_ * sendSize, sendSize, sendCount);
        std::vector<DataSlice> sendSrcSlicesList{sendSrcSlice};
        std::vector<DataSlice> sendDstSlicesList{sendDstSlice};

        DataSlice recvSrcSlice(remoteInBuffPtr, inBuffBaseOffset + recvOffset, recvSize, recvCount);
        DataSlice recvDstSlice(localHcclBuffPtr, hcclBuffBaseOffset + remoteIdx * recvSize, recvSize, recvCount);
        std::vector<DataSlice> recvSrcSlicesList{recvSrcSlice};
        std::vector<DataSlice> recvDstSlicesList{recvDstSlice};

        if (sendSize == 0) {
            // 发送数据片为0时，只接收数据
            HCCL_INFO("[MiniTask][ScatterData] rank=%u -> remoteIdx=%u(rank=%u): RecvWrite ONLY, "
                "recv remote(off=%llu size=%llu) -> local(hcclOff=%llu size=%llu)",
                myRank_, remoteIdx, remoteRank, inBuffBaseOffset + recvOffset, recvSize,
                hcclBuffBaseOffset + remoteIdx * recvSize, recvSize);
            SlicesList recvSlicesList(recvSrcSlicesList, recvDstSlicesList);
            DataInfo recvInfo(sendRecvChannel, recvSlicesList);
            CHK_PRT_RET(RecvWrite(recvInfo, threads.at(remoteIdx)),
                HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][ScatterData] Recv failed."),
                HcclResult::HCCL_E_INTERNAL);
        } else if (recvSize == 0) {
            // 接收数据片为0时，只发送数据
            HCCL_INFO("[MiniTask][ScatterData] rank=%u -> remoteIdx=%u(rank=%u): SendWrite ONLY, "
                "send local(off=%llu size=%llu) -> remote(hcclOff=%llu size=%llu)",
                myRank_, remoteIdx, remoteRank, inBuffBaseOffset + sendOffset, sendSize,
                hcclBuffBaseOffset + myRankIdx_ * sendSize, sendSize);
            SlicesList sendSlicesList(sendSrcSlicesList, sendDstSlicesList);
            DataInfo sendInfo(sendRecvChannel, sendSlicesList, dataType_);
            CHK_PRT_RET(SendBatchWrite(sendInfo, threads.at(remoteIdx)),
                HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][ScatterData] Send failed."),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            TxRxChannels sendRecvChannels(sendRecvChannel, sendRecvChannel);  // 收发双向用同一个Channel
            TxRxSlicesList sendRecvSlicesList({sendSrcSlicesList, sendDstSlicesList},
                {recvSrcSlicesList, recvDstSlicesList});
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);
            HCCL_INFO("[MiniTask][ScatterData] rank=%u -> remoteIdx=%u(rank=%u): SendRecv BOTH, "
                "send(off=%llu size=%llu) -> remote(hcclOff=%llu), "
                "recv remote(off=%llu size=%llu) -> local(hcclOff=%llu)",
                myRank_, remoteIdx, remoteRank,
                inBuffBaseOffset + sendOffset, sendSize, hcclBuffBaseOffset + myRankIdx_ * sendSize,
                inBuffBaseOffset + recvOffset, recvSize, hcclBuffBaseOffset + remoteIdx * recvSize);
            CHK_PRT_RET(SendRecvBatchWrite(sendRecvInfo, threads.at(remoteIdx)),
                HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][ScatterData] SendRecv failed."),
                HcclResult::HCCL_E_INTERNAL);
        }
    }

    HCCL_INFO("[MiniTask][ScatterData] rank=%u END", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DTwoShot::ReduceData(const TemplateDataParams &tempAlgParams,
    const std::vector<ThreadHandle> &threads)
{
    void* localHcclBuffPtr = tempAlgParams.buffInfo.hcclBuff.addr;
    u64 hcclBuffBaseOffset = tempAlgParams.buffInfo.hcclBuffBaseOff;

    u64 sliceSize = sliceInfoList_.at(myRankIdx_).size;
    u64 sliceCount = sliceInfoList_.at(myRankIdx_).count;

    HCCL_INFO("[MiniTask][ReduceData] rank=%u BEGIN, mySlice: size=%llu count=%llu, reduceOp=%d, dataType=%d",
        myRank_, sliceSize, sliceCount, (int)reduceOp_, (int)dataType_);

    // 数据量为0的数据片无需Reduce
    if (sliceSize == 0) {
        HCCL_INFO("[MiniTask][ReduceData] rank=%u sliceSize=0, skip reduce", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    // 每片数据Reduce到第0片数据的位置
    DataSlice reduceDstSlice(localHcclBuffPtr, hcclBuffBaseOffset, sliceSize, sliceCount);
    HCCL_INFO("[MiniTask][ReduceData] rank=%u dst: hcclBuff=%p offset=%llu (the 0th slice position)",
        myRank_, localHcclBuffPtr, hcclBuffBaseOffset);

    for (u32 sliceIdx = 1; sliceIdx < templateRankSize_; ++sliceIdx) {
        DataSlice reduceSrcSlice(localHcclBuffPtr, hcclBuffBaseOffset + sliceIdx * sliceSize, sliceSize, sliceCount);
        HCCL_INFO("[MiniTask][ReduceData] rank=%u reduce step %u/%u: src(offset=%llu) + dst(offset=%llu) -> dst",
            myRank_, sliceIdx, templateRankSize_ - 1,
            hcclBuffBaseOffset + sliceIdx * sliceSize, hcclBuffBaseOffset);
        // 确定性计算，顺序Reduce
        CHK_PRT_RET(LocalReduce(threads[0], reduceSrcSlice, reduceDstSlice, dataType_, reduceOp_),
            HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][ReduceData] LocalReduce failed"),
            HcclResult::HCCL_E_INTERNAL);
    }

    HCCL_INFO("[MiniTask][ReduceData] rank=%u END, all %u slices reduced into slice[0]", myRank_, templateRankSize_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DTwoShot::RunAllGather(const TemplateDataParams &tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[MiniTask][RunAllGather] rank=%u BEGIN, threads=%zu", myRank_, threads.size());

    // 主线程向从线程发送启动信号
    PreSync(threads);
    HCCL_INFO("[MiniTask][RunAllGather] rank=%u PreSync done, entering GatherData", myRank_);

    CHK_RET(GatherData(tempAlgParams, channels, threads));
    HCCL_INFO("[MiniTask][RunAllGather] rank=%u GatherData done, entering PostSync", myRank_);

    // 从线程往主线程返回结束信号
    PostSync(threads);
    HCCL_INFO("[MiniTask][RunAllGather] rank=%u PostSync done", myRank_);

    HCCL_INFO("[MiniTask][RunAllGather] rank=%u END", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DTwoShot::GatherData(const TemplateDataParams &tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    void* localHcclBuffPtr = tempAlgParams.buffInfo.hcclBuff.addr;
    void* localOutBuffPtr = tempAlgParams.buffInfo.outputPtr;
    u64 hcclBuffBaseOffset = tempAlgParams.buffInfo.hcclBuffBaseOff;
    u64 outBuffBaseOffset = tempAlgParams.buffInfo.outBuffBaseOff;

    u64 sendSize = sliceInfoList_.at(myRankIdx_).size;
    u64 sendCount = sliceInfoList_.at(myRankIdx_).count;
    u64 sendOffset = sliceInfoList_.at(myRankIdx_).offset;

    HCCL_INFO("[MiniTask][GatherData] rank=%u(myRankIdx=%u) BEGIN, localHcclBuff=%p, localOutBuff=%p",
        myRank_, myRankIdx_, localHcclBuffPtr, localOutBuffPtr);
    HCCL_INFO("[MiniTask][GatherData] rank=%u my data (reduced result at hcclBuff[0]): size=%llu count=%llu",
        myRank_, sendSize, sendCount);

    for (u32 remoteIdx = 0; remoteIdx < sliceInfoList_.size(); ++remoteIdx) {
        u64 recvSize = sliceInfoList_.at(remoteIdx).size;
        u64 recvCount = sliceInfoList_.at(remoteIdx).count;
        u64 recvOffset = sliceInfoList_.at(remoteIdx).offset;

        // 发送和接收数据量都为0的时候，既不发送也不接收
        if (sendSize == 0 && recvSize == 0) {
            HCCL_INFO("[MiniTask][GatherData] rank=%u -> remoteIdx=%u: both zero, skip", myRank_, remoteIdx);
            continue;
        }

        // 数据片序号等于自身rank序号时，本地拷贝数据
        if (remoteIdx == myRankIdx_) {
            HCCL_INFO("[MiniTask][GatherData] rank=%u -> remoteIdx=%u: LocalCopy myself, "
                "src(hcclBuff[0] size=%llu) -> dst(outBuff off=%llu size=%llu)",
                myRank_, remoteIdx, sendSize, outBuffBaseOffset + recvOffset, recvSize);
            DataSlice copySrcSlice(localHcclBuffPtr, hcclBuffBaseOffset, sendSize, sendCount);
            DataSlice copyDstSlice(localOutBuffPtr, outBuffBaseOffset + recvOffset, recvSize, recvCount);
            CHK_PRT_RET(LocalCopy(threads.at(remoteIdx), copySrcSlice, copyDstSlice),
                HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][ScatterData] LocalCopy failed."),
                HcclResult::HCCL_E_INTERNAL);
            continue;
        }

        // 数据片序号不等于自身rank序号时，跨rank发送和接收数据
        u32 remoteRank = rankList_.at(remoteIdx);
        const ChannelInfo &sendRecvChannel = channels.at(remoteRank).at(0);

        void* remoteHcclBuffPtr = sendRecvChannel.remoteCclMem.addr;

        DataSlice sendSrcSlice(localHcclBuffPtr, hcclBuffBaseOffset, sendSize, sendCount);
        DataSlice sendDstSlice(remoteHcclBuffPtr, outBuffBaseOffset + sendOffset, sendSize, sendCount);
        std::vector<DataSlice> sendSrcSlicesList{sendSrcSlice};
        std::vector<DataSlice> sendDstSlicesList{sendDstSlice};

        DataSlice recvSrcSlice(remoteHcclBuffPtr, hcclBuffBaseOffset, recvSize, recvCount);
        DataSlice recvDstSlice(localOutBuffPtr, outBuffBaseOffset + recvOffset, recvSize, recvCount);
        std::vector<DataSlice> recvSrcSlicesList{recvSrcSlice};
        std::vector<DataSlice> recvDstSlicesList{recvDstSlice};

        if (sendSize == 0) {
            // 发送数据片为0时，只接收数据
            HCCL_INFO("[MiniTask][GatherData] rank=%u -> remoteIdx=%u(rank=%u): RecvRead ONLY, "
                "recv remote(hcclBuff[0] size=%llu) -> local(outBuff off=%llu size=%llu)",
                myRank_, remoteIdx, remoteRank, recvSize, outBuffBaseOffset + recvOffset, recvSize);
            SlicesList recvSlicesList(recvSrcSlicesList, recvDstSlicesList);
            DataInfo recvInfo(sendRecvChannel, recvSlicesList, dataType_);
            CHK_PRT_RET(RecvBatchRead(recvInfo, threads.at(remoteIdx)),
                HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][ScatterData] Recv failed."),
                HcclResult::HCCL_E_INTERNAL);
        } else if (recvSize == 0) {
            // 接收数据片为0时，只发送数据
            HCCL_INFO("[MiniTask][GatherData] rank=%u -> remoteIdx=%u(rank=%u): SendRead ONLY, "
                "send local(hcclBuff[0] size=%llu) -> remote(outBuff off=%llu size=%llu)",
                myRank_, remoteIdx, remoteRank, sendSize, outBuffBaseOffset + sendOffset, sendSize);
            SlicesList sendSlicesList(sendSrcSlicesList, sendDstSlicesList);
            DataInfo sendInfo(sendRecvChannel, sendSlicesList, dataType_);
            CHK_PRT_RET(SendRead(sendInfo, threads.at(remoteIdx)),
                HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][ScatterData] Send failed."),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            TxRxChannels sendRecvChannels(sendRecvChannel, sendRecvChannel);  // 收发双向用同一个Channel
            TxRxSlicesList sendRecvSlicesList({sendSrcSlicesList, sendDstSlicesList},
                {recvSrcSlicesList, recvDstSlicesList});
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);
            HCCL_INFO("[MiniTask][GatherData] rank=%u -> remoteIdx=%u(rank=%u): SendRecvRead BOTH, "
                "send local(hcclBuff[0] size=%llu) -> remote(outBuff off=%llu), "
                "recv remote(hcclBuff[0] size=%llu) -> local(outBuff off=%llu)",
                myRank_, remoteIdx, remoteRank,
                sendSize, outBuffBaseOffset + sendOffset,
                recvSize, outBuffBaseOffset + recvOffset);
            CHK_PRT_RET(SendRecvBatchRead(sendRecvInfo, threads.at(remoteIdx)),
                HCCL_ERROR("[InsTempAllReduceMesh1DTwoShot][ScatterData] SendRecv failed."),
                HcclResult::HCCL_E_INTERNAL);
        }
    }

    HCCL_INFO("[MiniTask][GatherData] rank=%u END", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DTwoShot::PreSync(const std::vector<ThreadHandle> &threads)
{
    if (threads.size() > 1) {
        std::vector<ThreadHandle> slaveThreads(threads.begin() + 1, threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(threads.at(0), slaveThreads, notifyIdxMainToSub_));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DTwoShot::PostSync(const std::vector<ThreadHandle> &threads)
{
    if (threads.size() > 1) {
        std::vector<ThreadHandle> slaveThreads(threads.begin() + 1, threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(threads.at(0), slaveThreads, notifyIdxSubToMain_));
    }
    return HcclResult::HCCL_SUCCESS;
}

}  // ops_hccl