/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "aiv_communication_base_v2.h"
 
using namespace AscendC;
 
// todo 简化参数
 
class AivBroadcastMesh1D : public AivCommBase {
    constexpr static uint64_t CORE_NUMS_ALL = 2;

public:
    __aicore__ inline AivBroadcastMesh1D() {}

    template<typename T>
    __aicore__ inline void Process(uint64_t curCount, uint64_t sliceId, uint64_t stride);

    template<typename T>
    __aicore__ inline void ProcessBigData(uint64_t curCount, uint64_t sliceId);

    template<typename T>
    __aicore__ inline void ProcessBigDataOpt(uint64_t curCount, uint64_t sliceId);

private:
    __aicore__ inline void CalculateOffsetAndCount(uint64_t totalData, uint64_t index, 
                                                   uint64_t totalParts, uint64_t &offset, uint64_t &count);
};

__aicore__ inline void AivBroadcastMesh1D::CalculateOffsetAndCount(uint64_t totalData, uint64_t index, 
                                               uint64_t totalParts, uint64_t &offset, uint64_t &count)
{
    if (totalParts == 0) {
        offset = 0;
        count = 0;
        return;
    }
    uint64_t dataPerPart = totalData / totalParts;
    uint64_t remainder = totalData % totalParts;
    if (index < remainder) {
        offset = index * dataPerPart + index;
        count = dataPerPart + 1;
    } else {
        offset = index * dataPerPart + remainder;
        count = dataPerPart;
    }
}
 
template<typename T>
__aicore__ inline void AivBroadcastMesh1D::Process(uint64_t curCount, uint64_t sliceId, uint64_t stride)
{
    curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
    uint64_t dataTypeSize = sizeof(T);
    if (block_idx >= rankSize_) {
        return;
    }
    uint32_t peerRank = block_idx / (rankSize_ / rankSize_);
    uint64_t offsetPerCore = curCount / rankSize_ * dataTypeSize;
    uint64_t dataOffset = offsetPerCore * block_idx;
    uint64_t countPerCore = block_idx == rankSize_ - 1 ? curCount - (rankSize_ - 1) * (curCount / rankSize_)
                                    : curCount / rankSize_;
    uint64_t flag_offset = block_idx;
    __gm__ T *inputGM = (__gm__ T *)(input_ + dataOffset);
    __gm__ T *cclGM = (__gm__ T *)(GM_IN[peerRank] + dataOffset);
    // scatter
    if (rank_ == root_) {
        CpGM2GM(cclGM, inputGM, countPerCore);
        PipeBarrier<PIPE_ALL>();
        // 避免多核同时访问一个flag
        for (uint32_t i = 0; i < rankSize_; i++) {
            Record(i, flag_offset, curTag_);
        }
    }
 
    // allgather
    WaitFlag(rank_, flag_offset, curTag_);
    CpGM2GM(inputGM, cclGM, countPerCore);
    PipeBarrier<PIPE_ALL>();
}

template<typename T>
__aicore__ inline void AivBroadcastMesh1D::ProcessBigData(uint64_t curCount, uint64_t sliceId)
{
    curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
    // root节点先用全量核去写本卡cclBuffer，这里的每个核要对应多个flag，让其他卡可以用全量核来读
    // 然后其他卡用全量核去读数据
    // 最后做allgather
    uint64_t curStageCoreNum = numBlocks_ / rankSize_ * rankSize_;
    if (block_idx >= curStageCoreNum) {
        return;
    }

    uint64_t coreNumPerRank = curStageCoreNum / rankSize_;
    uint64_t targetRank = block_idx / coreNumPerRank;
    uint64_t coreIndex = (block_idx - (targetRank * coreNumPerRank)) % coreNumPerRank;
    uint64_t flag_offset = 0;

    // 先把数据按照rankSize 切分
    uint64_t rankInnerDispls = 0;
    uint64_t targetRankCurCount = 0;
    CalculateOffsetAndCount(curCount, targetRank, rankSize_, rankInnerDispls, targetRankCurCount);

    // 给每个核划分数据
    uint64_t innerDispls = 0;
    uint64_t sendCurCount = 0;
    CalculateOffsetAndCount(targetRankCurCount, coreIndex, coreNumPerRank, innerDispls, sendCurCount);

    // root 开始本卡搬运数据:这里是全量卡都去搬比较好，还是就用rankSize的卡去搬
    uint64_t sendInputOffset = input_ + (rankInnerDispls + innerDispls) * sizeof(T);
    uint64_t sendCclInOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + (rankInnerDispls + innerDispls) * sizeof(T);
    if (rank_ == root_) {
        CpGM2GM((__gm__ T *)sendCclInOffset, (__gm__ T *)sendInputOffset, sendCurCount);
        PipeBarrier<PIPE_ALL>();
        // targetRankCurCount这么多的数据量，有coreNumPerRank去写，但是有coreNumPerRank * rankSize的核去读，所以一个核要写rankSize个flag
        for (uint64_t i = 0; i < rankSize_; i++) {
            Record(root_, block_idx * rankSize_ + i, curTag_);
        }
    }

    // 现在除了root节点，其他卡要用全量核去拿root卡上的数据
    uint64_t rankInnerDisplsStage1 = 0;
    uint64_t targetRankCurCountStage1 = 0;
    CalculateOffsetAndCount(curCount, rank_, rankSize_, rankInnerDisplsStage1, targetRankCurCountStage1);

    // 给每rankSize个核划分数据
    uint64_t rankSizeCoreDataIndex = block_idx / rankSize_;
    uint64_t rankSizeCoreInnerDispls = 0;
    uint64_t rankSizeCoreSendCurCount = 0;
    CalculateOffsetAndCount(targetRankCurCountStage1, rankSizeCoreDataIndex, 
                            coreNumPerRank, rankSizeCoreInnerDispls, rankSizeCoreSendCurCount);

    // 给每个核划分数据
    uint64_t coreIndexStage1 = (block_idx - (rankSizeCoreDataIndex * rankSize_)) % rankSize_;
    uint64_t innerDisplsStage1 = 0;
    uint64_t sendCurCountStage1 = 0;
    CalculateOffsetAndCount(rankSizeCoreSendCurCount, coreIndexStage1, 
                            rankSize_, innerDisplsStage1, sendCurCountStage1);

    // 每个核开始去读数据
    uint64_t recvCclInOffset = reinterpret_cast<uint64_t>(GM_IN[root_]) + (rankInnerDisplsStage1 + rankSizeCoreInnerDispls + innerDisplsStage1) * sizeof(T);
    uint64_t recvCclOutOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + (rankInnerDisplsStage1 + rankSizeCoreInnerDispls + innerDisplsStage1) * sizeof(T);
    uint64_t flagTotal = rankSize_ * curStageCoreNum;
    flag_offset = rank_ * coreNumPerRank * rankSize_ + rankSizeCoreDataIndex * rankSize_ + coreIndexStage1;
    WaitFlag(root_, flag_offset, curTag_);
    if (rank_ != root_) {
        CpGM2GM((__gm__ T *)recvCclOutOffset, (__gm__ T *)recvCclInOffset, sendCurCountStage1);
        PipeBarrier<PIPE_ALL>();
        Record(rank_, flag_offset, curTag_);
    }
    if (coreIndexStage1 == 0) {
        for (uint64_t i = 0; i < rankSize_; i++) {
            uint64_t flag_offset_w = rank_ * coreNumPerRank * rankSize_ + rankSizeCoreDataIndex * rankSize_ + i;
            WaitFlag(rank_, flag_offset_w, curTag_);
        }
        for (uint64_t i = 0; i < rankSize_; i++) {
            Record(i, flagTotal + rank_ + rankSizeCoreDataIndex * rankSize_, curTag_);
        }
    }

    // 最后所有的卡去做allgather,每个卡要读对面rankSize个flag
    uint64_t gatherSrcOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + (rankInnerDispls + innerDispls) * sizeof(T);
    uint64_t ouputOffset = input_ + (rankInnerDispls + innerDispls) * sizeof(T);
    if ((rank_ != root_) && (sendCurCount > 0)) {
        // 每块数据要去等rankSize个flag
        WaitFlag(rank_, flagTotal + targetRank + coreIndex * rankSize_, curTag_);
        CpGM2GM((__gm__ T *)ouputOffset, (__gm__ T *)gatherSrcOffset, sendCurCount);
        PipeBarrier<PIPE_ALL>();
    }
}

/**
 * 优化版大数据量广播实现
 * 优化策略：
 * 1. 保持原有的三阶段架构，确保正确性
 * 2. 优化flag管理，减少同步开销
 * 3. 提高核利用率，让所有核参与数据传输
 * 4. 优化数据切分策略，减少重复计算
 */
template<typename T>
__aicore__ inline void AivBroadcastMesh1D::ProcessBigDataOpt(uint64_t curCount, uint64_t sliceId)
{
    curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
    
    uint64_t curStageCoreNum = numBlocks_ / rankSize_ * rankSize_;
    if (block_idx >= curStageCoreNum) {
        return;
    }
    
    uint64_t coreNumPerRank = curStageCoreNum / rankSize_;
    uint64_t targetRank = block_idx / coreNumPerRank;
    uint64_t coreIndex = block_idx % coreNumPerRank;
    
    // Stage1: root节点所有核并行写本地CCL Buffer
    uint64_t rankInnerDispls = 0;
    uint64_t targetRankCurCount = 0;
    CalculateOffsetAndCount(curCount, targetRank, rankSize_, rankInnerDispls, targetRankCurCount);
    
    uint64_t innerDispls = 0;
    uint64_t sendCurCount = 0;
    CalculateOffsetAndCount(targetRankCurCount, coreIndex, coreNumPerRank, innerDispls, sendCurCount);
    
    uint64_t sendInputOffset = input_ + (rankInnerDispls + innerDispls) * sizeof(T);
    uint64_t sendCclInOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + (rankInnerDispls + innerDispls) * sizeof(T);
    
    if (rank_ == root_) {
        // root节点并行写本地CCL Buffer
        CpGM2GM((__gm__ T *)sendCclInOffset, (__gm__ T *)sendInputOffset, sendCurCount);
        PipeBarrier<PIPE_ALL>();
        // 每个核写rankSize个flag，让其他卡的对应核可以开始读取
        for (uint32_t i = 0; i < rankSize_; i++) {
            Record(root_, block_idx * rankSize_ + i, curTag_);
        }
    }
    
    // Stage2: 所有节点的所有核并行从root读取数据到本地CCL Buffer
    // 计算本rank需要读取的数据偏移
    uint64_t localRankInnerDispls = 0;
    uint64_t localRankCurCount = 0;
    CalculateOffsetAndCount(curCount, rank_, rankSize_, localRankInnerDispls, localRankCurCount);
    
    // 按rankSize分组，每组有coreNumPerRank个核
    uint64_t groupIndex = block_idx / rankSize_;
    uint64_t groupInnerDispls = 0;
    uint64_t groupCurCount = 0;
    CalculateOffsetAndCount(localRankCurCount, groupIndex, coreNumPerRank, groupInnerDispls, groupCurCount);
    
    // 组内每个核再切分
    uint64_t innerCoreIndex = block_idx % rankSize_;
    uint64_t innerCoreDispls = 0;
    uint64_t innerCoreCount = 0;
    CalculateOffsetAndCount(groupCurCount, innerCoreIndex, rankSize_, innerCoreDispls, innerCoreCount);
    
    uint64_t recvOffset = localRankInnerDispls + groupInnerDispls + innerCoreDispls;
    uint64_t recvCclInOffset = reinterpret_cast<uint64_t>(GM_IN[root_]) + recvOffset * sizeof(T);
    uint64_t recvCclOutOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + recvOffset * sizeof(T);
    
    // 等待root节点对应数据块完成
    uint64_t flag_offset = targetRank * coreNumPerRank * rankSize_ + groupIndex * rankSize_ + innerCoreIndex;
    WaitFlag(root_, flag_offset, curTag_);
    
    if (rank_ != root_ && innerCoreCount > 0) {
        CpGM2GM((__gm__ T *)recvCclOutOffset, (__gm__ T *)recvCclInOffset, innerCoreCount);
        PipeBarrier<PIPE_ALL>();
        Record(rank_, flag_offset, curTag_);
    }
    
    // Stage2完成后的同步 - 只有特定核负责等待和通知
    if (innerCoreIndex == 0) {
        uint64_t flagTotal = rankSize_ * curStageCoreNum;
        for (uint32_t i = 0; i < rankSize_; i++) {
            uint64_t flag_offset_w = rank_ * coreNumPerRank * rankSize_ + groupIndex * rankSize_ + i;
            WaitFlag(rank_, flag_offset_w, curTag_);
        }
        for (uint32_t i = 0; i < rankSize_; i++) {
            Record(i, flagTotal + rank_ + groupIndex * rankSize_, curTag_);
        }
    }
    
    // Stage3: 所有节点将本地CCL Buffer的数据拷贝到input（完成最终广播）
    uint64_t gatherSrcOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + (rankInnerDispls + innerDispls) * sizeof(T);
    uint64_t outputOffset = input_ + (rankInnerDispls + innerDispls) * sizeof(T);
    
    if ((rank_ != root_) && (sendCurCount > 0)) {
        uint64_t flagTotal = rankSize_ * curStageCoreNum;
        WaitFlag(rank_, flagTotal + targetRank + coreIndex * rankSize_, curTag_);
        CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)gatherSrcOffset, sendCurCount);
        PipeBarrier<PIPE_ALL>();
    }
}
 
template<typename T>
__aicore__ inline void AivBroadcastV2Mesh1D(KERNEL_ARGS_DEF)
{
    AivBroadcastMesh1D op;
    op.Init(KERNEL_CLASS_INIT, true);
    SyncAll<true>();
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();
    if (len * sizeof(T) >= DATA_LIMIT) {
        // 使用优化版大数据量处理函数
        op.ProcessBigDataOpt<T>(len, sliceId);
    } else {
        op.Process<T>(len, sliceId, inputSliceStride);
    }
    op.BarrierAll();
}
