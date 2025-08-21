/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "reduce_scatter_plant_local_reduce.h"
#include "alg_template_register.h"

namespace hccl {
ReduceScatterPlantLocalReduce::ReduceScatterPlantLocalReduce(const HcclDispatcher dispatcher)
    : AlgTemplateBase(dispatcher)
{}

ReduceScatterPlantLocalReduce::~ReduceScatterPlantLocalReduce()
{}

HcclResult ReduceScatterPlantLocalReduce::Prepare(void *inputMemPtr, DeviceMem &cclInMem, DeviceMem &outputMem,
    const Stream &stream, std::vector<Stream> &subStreams, std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
    std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux, GroupSlicesInfo &grouSlicesInfo,
    const HcclReduceOp reductionOp, u32 all2allOffset, const HcclDataType dataType)
{
    inputMemPtr_ = inputMemPtr;       // UserInPtr，All2All使用
    inputMem_ = cclInMem;             // 空拷贝 & 存放最后一块数据（Allreduce非整除场景）
    outputMem_ = outputMem;           // 单算子CclOut 图模式Scrach/UserOut，LocalReduce使用
    stream_ = stream;
    subStreams_ = subStreams;
    meshSignalPtr_ = &meshSignal;
    meshSignalAuxPtr_ = &meshSignalAux;
    groupSlicesInfo_ = std::move(grouSlicesInfo);
    reductionOp_ = reductionOp;
    all2allOffset_ = all2allOffset;
    dataType_ = dataType;
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduce::MainRecordSub(Stream &mainStream, u32 firstSubStreamIndex,
    u32 totalTask)
{
    for (u32 streamIndex = firstSubStreamIndex; streamIndex < totalTask; streamIndex++) {
        CHK_RET(LocalNotify::Post(mainStream, dispatcher_, (*meshSignalAuxPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduce::SubWaitMain(u32 firstSubStreamIndex, u32 totalTask)
{
    for (u32 streamIndex = firstSubStreamIndex; streamIndex < totalTask; streamIndex++) {
        CHK_RET(LocalNotify::Wait(subStreams_[streamIndex], dispatcher_,
            (*meshSignalAuxPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduce::MainWaitSub(Stream &mainStream, u32 firstSubStreamIndex, u32 totalTask)
{
    for (u32 streamIndex = firstSubStreamIndex; streamIndex < totalTask; streamIndex++) {
        CHK_RET(LocalNotify::Wait(mainStream, dispatcher_, (*meshSignalPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduce::SubRecordMain(u32 firstSubStreamIndex, u32 totalTask)
{
    for (u32 streamIndex = firstSubStreamIndex; streamIndex < totalTask; streamIndex++) {
        CHK_RET(LocalNotify::Post(subStreams_[streamIndex], dispatcher_, (*meshSignalPtr_)[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduce::MainRecordLocalReduceWait(u32 lRMainStreamIndex)
{
    CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAuxPtr_)[lRMainStreamIndex], profilerInput_.stage));
    CHK_RET(LocalNotify::Wait(subStreams_[lRMainStreamIndex], dispatcher_, (*meshSignalAuxPtr_)[lRMainStreamIndex],
        profilerInput_.stage));
    return HCCL_SUCCESS;
}

u32 ReduceScatterPlantLocalReduce::CalcOutputIndex(const u32 round)
{
    return (all2allOffset_ + round + localRank_) % rankSize_;
}

bool ReduceScatterPlantLocalReduce::isLastGroup(const u32 groupId)
{
    return groupId == groupSlicesInfo_.size() - 1;
}

bool ReduceScatterPlantLocalReduce::isLastRank(const u32 rankId)
{
    return rankId == rankSize_ - 1;
}

bool ReduceScatterPlantLocalReduce::isLastBlockData(const u32 outputIndex)
{
    return outputIndex == rankSize_ - 1;
}

HcclResult ReduceScatterPlantLocalReduce::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    HCCL_INFO("ReduceScatterPlantLocalReduce run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p].",
        rank, rankSize, inputMem_.ptr(), outputMem_.ptr());
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    CHK_PRT_RET(links.size() < rankSize, HCCL_ERROR("[%s]rank[%u] linksize[%llu] is less than rankSize[%u]",
        __func__, rank, links.size(), rankSize), HCCL_E_INTERNAL);

    rankSize_ = rankSize;
    localRank_ = rank;

    // All2All主流（主流）通知LocalReduce主流开始准备执行,
    // All2All需要rankSize条流，其中主流完成LocalCopy&第一个A2A任务，因此主从同步需要rankSize-2个任务。lRMainStreamId_需要-2
    all2allSubStreamNum_ = rankSize - 2;
    lRMainStreamId_ = all2allSubStreamNum_;

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    CHK_RET(MainRecordLocalReduceWait(lRMainStreamId_));
    // 额外一次LocalReduce主流通知All2All主流准备好接受信息（通知第一次执行完的All2AllWait）
    CHK_RET(LocalNotify::Post(subStreams_[lRMainStreamId_], dispatcher_, (*meshSignalPtr_)[lRMainStreamId_],
        profilerInput_.stage));

    HcclResult ret = HCCL_SUCCESS;
    for (u32 groupId = 0; groupId < groupSlicesInfo_.size(); groupId++) {
        const MemBlockInfo& memBlockInfo = groupSlicesInfo_[groupId];
        ret = RunAlltoAll(links, groupId, memBlockInfo);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]RunAlltoAll failed, localRank[%u], groupId[%u]",
            __func__, localRank_, groupId), ret);
        
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignalPtr_)[lRMainStreamId_], profilerInput_.stage));
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
        CHK_RET(MainRecordLocalReduceWait(lRMainStreamId_));
        CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[lRMainStreamId_], dispatcher_));

        ret = RunLocalReduce(groupId, memBlockInfo);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]LocalReduce failed, localRank[%u], groupId[%u]",
            __func__, localRank_, groupId), ret);
        
        // LocalReduce主流通知All2All主流执行完成，可以下发下一次LocalReduce操作
        CHK_RET(LocalNotify::Post(subStreams_[lRMainStreamId_], dispatcher_, (*meshSignalPtr_)[lRMainStreamId_],
            profilerInput_.stage));
    }

    // All2All主流等待最后一次LocalReduce执行完成
    CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignalPtr_)[lRMainStreamId_], profilerInput_.stage));
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    HCCL_INFO("ReduceScatterPlantLocalReduce finished: localRank[%u] ranksize[%u]", localRank_, rankSize_);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduce::LocalCopy(u32 groupId, const MemBlockInfo& memBlockInfo)
{
    u64 sliceSize = memBlockInfo.size[localRank_];
    if (sliceSize == 0) {
        return HCCL_SUCCESS;
    }
    
    DeviceMem src = DeviceMem::create(static_cast<u8 *>(inputMemPtr_) + 
        memBlockInfo.userInputOffsets[localRank_], sliceSize);
    
    // 当非最后一组最后一卡且outputIndex是最后一块时，Copy至CclIn/UserIn预留位
    DeviceMem dst;
    u32 outputIndex = CalcOutputIndex(localRank_);
    if (isLastBlockData(outputIndex) && !(isLastRank(localRank_) && isLastGroup(groupId))) {
        dst = inputMem_.range(memBlockInfo.outputOffsets[localRank_], sliceSize);
    } else {
        dst = outputMem_.range(memBlockInfo.outputOffsets[outputIndex], sliceSize);
    }
    
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduce::RunAlltoAll(const std::vector<LINK> &links, u32 groupId,
    const MemBlockInfo& memBlockInfo)
{
    // 本卡优先拷贝同号位数据
    CHK_RET(LocalCopy(groupId, memBlockInfo));
    
    // 主流通知从流可以开始接受数据
    u32 all2allfirstSubStreamId = 0;
    CHK_RET(MainRecordSub(stream_, all2allfirstSubStreamId, all2allSubStreamNum_));
    CHK_RET(SubWaitMain(all2allfirstSubStreamId, all2allSubStreamNum_));

    // 开始数据拷贝
    u32 streamIndex = 0;
    for (u32 round = 0; round < rankSize_; round++) {
        if (round == localRank_) {
            continue;
        }
        Stream &subStream = (streamIndex == 0) ? stream_ : subStreams_[streamIndex - 1];
        CHK_SMART_PTR_NULL(links[round]);
        CHK_RET(links[round]->TxAck(subStream));
        CHK_RET(links[round]->RxAck(subStream));
        streamIndex++;
    }

    CHK_RET(SubRecordMain(all2allfirstSubStreamId, all2allSubStreamNum_));
    CHK_RET(MainWaitSub(stream_, all2allfirstSubStreamId, all2allSubStreamNum_));
    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[lRMainStreamId_], dispatcher_));

    CHK_RET(MainRecordSub(stream_, all2allfirstSubStreamId, all2allSubStreamNum_));
    CHK_RET(SubWaitMain(all2allfirstSubStreamId, all2allSubStreamNum_));
    streamIndex = 0;
    for (u32 round = 0; round < rankSize_; round++) {
        if (round == localRank_) {
            continue;
        }
        Stream &subStream = (streamIndex == 0) ? stream_ : subStreams_[streamIndex - 1];
        CHK_SMART_PTR_NULL(links[round]);

        u64 sliceSize = memBlockInfo.size[round];
        if (sliceSize != 0) {
            u64 userMemInOffset = memBlockInfo.userInputOffsets[round];
            DeviceMem src = DeviceMem::create(static_cast<u8 *>(inputMemPtr_) + userMemInOffset, sliceSize);

            u32 outputIndex = CalcOutputIndex(round);
            u64 dstOffset = 0;
            void *remMemPtr = nullptr;
            if (isLastBlockData(outputIndex) && !(isLastRank(round) && isLastGroup(groupId))) {
                CHK_RET(links[round]->GetRemoteMem(UserMemType::INPUT_MEM, &remMemPtr));
                dstOffset = memBlockInfo.outputOffsets[round];
            } else {
                CHK_RET(links[round]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
                dstOffset = memBlockInfo.outputOffsets[outputIndex];
            }
            DeviceMem dst = DeviceMem::create(static_cast<u8 *>(remMemPtr) + dstOffset, sliceSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStream, links[round]->GetRemoteRank(),
                    links[round]->GetLinkType()));
        }

        CHK_RET(links[round]->TxDataSignal(subStream));
        CHK_RET(links[round]->RxDataSignal(subStream));
        streamIndex++;
    }

    // 从流通知主流完成拷贝
    CHK_RET(SubRecordMain(all2allfirstSubStreamId, all2allSubStreamNum_));
    CHK_RET(MainWaitSub(stream_, all2allfirstSubStreamId, all2allSubStreamNum_));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterPlantLocalReduce::RunLocalReduce(u32 groupId, const MemBlockInfo& memBlockInfo)
{
    u32 reduceStep = static_cast<u32>(std::ceil(log2(rankSize_)));
    u64 srcOffset = memBlockInfo.inputOffsets[localRank_];
    u64 sliceSize = memBlockInfo.size[localRank_];
    u64 count = sliceSize / DataUnitSize(dataType_);

    for (u32 round = 0; round < reduceStep; round++) {
        u32 tailIndex = std::min(rankSize_, static_cast<u32>(1 << static_cast<int>(reduceStep - round))) - 1;
        u32 headIndex = static_cast<u32>(1 << static_cast<int>((reduceStep - round - 1)));

        // LR主流通知从流可以开始接受数据
        for (u32 offset = 0; offset < tailIndex - headIndex; offset++) {
            u32 streamId = lRMainStreamId_ + offset + 1;
            // 只有reduce任务 > 1时才需要主从流同步: LR主流通知从流, 从流Wait LR主流
            CHK_RET(LocalNotify::Post(subStreams_[lRMainStreamId_], dispatcher_, (*meshSignalAuxPtr_)[streamId],
                    profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(subStreams_[streamId], dispatcher_, (*meshSignalAuxPtr_)[streamId],
                profilerInput_.stage));
        }
        
        // LocalReduce操作
        for (u32 offset = 0; offset <= tailIndex - headIndex; offset++) {   
            u32 inputIndex = CalcOutputIndex(headIndex + offset); // reduce的源数据offset
            u32 outputIndex = CalcOutputIndex(offset);            // reduce的目标offset

            Stream &subStream = subStreams_[lRMainStreamId_ + offset];

            if (sliceSize == 0) {
                continue;
            }

            void *srcPtr;
            void *dstPtr;
            if (!(isLastRank(localRank_) && isLastGroup(groupId)) && isLastBlockData(inputIndex)) {
                srcPtr = static_cast<u8 *>(inputMem_.ptr()) + srcOffset;
            } else {
                srcPtr = static_cast<u8 *>(outputMem_.ptr()) + memBlockInfo.outputOffsets[inputIndex];
            }

            if (!(isLastRank(localRank_) && isLastGroup(groupId)) && isLastBlockData(outputIndex)) {
                dstPtr = static_cast<u8 *>(inputMem_.ptr()) + srcOffset;
            } else {
                dstPtr = static_cast<u8 *>(outputMem_.ptr()) + memBlockInfo.outputOffsets[outputIndex];
            }
            
            CHK_RET(HcclReduceAsync(dispatcher_, srcPtr, count, dataType_, reductionOp_, subStream, dstPtr,
                INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP, INLINE_REDUCE_BIT));
        }

        // 从流通知LR主流可以开始下一轮
        for (u32 offset = 0; offset < tailIndex - headIndex; offset++) {
            u32 streamId = lRMainStreamId_ + offset + 1;
            // 只有reduce任务 > 1时才需要主从流同步: LR主流通知从流, 从流Wait LR主流
            CHK_RET(LocalNotify::Post(subStreams_[streamId], dispatcher_, (*meshSignalPtr_)[streamId],
                profilerInput_.stage));
            CHK_RET(LocalNotify::Wait(subStreams_[lRMainStreamId_], dispatcher_, (*meshSignalPtr_)[streamId],
                profilerInput_.stage));
        }

        CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, subStreams_[lRMainStreamId_], dispatcher_));
    }

    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE, ReduceScatterPlantLocalReduce);
} // namespace hccl