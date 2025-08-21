/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_HCCL_ALLREDUCE_H__
#define __AICPU_HCCL_ALLREDUCE_H__

#include "aicpu_hccl_algorithm.h"

class AicpuHcclAllreduce : public AicpuHcclAlgorithm {
public:
    explicit AicpuHcclAllreduce(AicpuComContext *ctx) : AicpuHcclAlgorithm(ctx) {}
    ~AicpuHcclAllreduce() override = default;

    HcclResult RunAlgorithm(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataCount,
        HcclDataType dataType, u64 strideLen = 0, AivAicpuOpParam *nextTask = nullptr) override;

private:
    int64_t RoundUpWithDivisor(u64 value, u64 divisor) const;
    HcclResult PrepareSlice(u64 dataCount, HcclDataType dataType, u32 sliceNum, std::vector<Slice>& dataSlice) const;
    void GetDataSizes16K(std::vector<u64>& dataSizes, u64 allDataSize) const;
    void RunAllReduceSlice(u8 *curOutputPtr, u8 *curInputPtr, u64 *sliceSize, u64 *dataSlice,
        HcclReduceOp opType, HcclDataType dataType) const;
    void RunAllReduceSliceWin2Win(u8 *curOutputPtr, u64 *sliceSize, u64 *dataSlice,
        HcclReduceOp opType, HcclDataType dataType) const;
    HcclResult RunAllReduceAlign(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataCount,
        HcclDataType dataType) const;
    HcclResult RunAllReduceAlignWin2Win(HcclReduceOp opType, void *recvBuffer,
        u64 dataCount, HcclDataType dataType) const;
    HcclResult RunAllReduce(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataCount,
        HcclDataType dataType) const;
    HcclResult RunAllReduceOneShot4Stream(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataSize,
        HcclDataType dataType) const;
    HcclResult RunAllReduceReduceBcast(HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataSize,
        HcclDataType dataType) const;
    HcclResult RunReduceBcastOnMainSq(u32 mainRankId, u32 maxStreamNum, u32 /* startRank */, u32 /* end_rank */,
        void *sendBuffer, void *recvBuffer, u64 dataSize, HcclDataType dataType) const;
    HcclResult RunReduceBcastOnOtherSq(HcclReduceOp opType, u32 mainRankId, u32 maxStreamNum,
        void *sendBuffer, void *recvBuffer, u64 dataSize, HcclDataType dataType) const;
    HcclResult RunAllReduceOneShot1Stream(HcclReduceOp opType, void *sendBuffer, void *recvBuffer,
        u64 dataSize, HcclDataType dataType) const;
    HcclResult RunAllReduceTwoShot1Stream(HcclReduceOp opType, void *sendBuffer, void *recvBuffer,
        u64 dataCount, HcclDataType dataType) const;
    u32 GetHdPeer(const u32 hdRound, const u32 curRank) const;
    HcclResult RunAllReduceOneshotHD(HcclReduceOp opType, void *sendBuffer, void *recvBuffer,
        u64 dataSize, HcclDataType dataType) const;
    HcclResult RunAllReduceRing(HcclReduceOp opType, void *sendBuffer, void *recvBuffer,
        u64 dataCount, HcclDataType dataType) const;
    u64 AlignWith(u64 oriValue, u64 alignValue) const;
    HcclResult GetBurstDataCounts(u64 windowSize, u64 dataCount, std::vector<u64>& burstDataCounts) const;
    std::vector<std::vector<u32>> GetRingOrders() const;
    HcclResult reorderRingSlice(const std::vector<u32> &ringOrder, const std::vector<Slice> &ringSlices,
        std::vector<Slice> &orderedRingSlices) const;
    HcclResult PrepareRingSlice(
        const std::vector<std::vector<u32>> &ringOrders, u64 dataCount,
        HcclDataType dataType,
        std::vector<std::vector<Slice>> &orderedAllRingSlice) const;
    HcclResult RingIPCPreSync(const u32 stream, const u32 prevRank, const u32 nextRank) const;
    HcclResult RingIPCPostSync(const u32 stream, const u32 prevRank, const u32 nextRank) const;
    HcclResult GetPrevRankList(const std::vector<u32> &ringOrder, std::vector<u32> &previousRankList) const;
    size_t FindNextRank(const std::vector<u32> &previousRankList, const u32 localRank) const;
    Slice* GetNextRingSlice(const std::vector<u32> &previousRankList,
        std::vector<Slice> &orderedRingSlices, u32 &curSliceIdx) const;
    HcclResult RunAllReduceRingSingleBurst(
        HcclReduceOp opType, void *sendBuffer, void *recvBuffer, u64 dataCount,
        HcclDataType dataType, std::vector<std::vector<u32>> &ringOrders) const;
    HcclResult RunAllReduceRingAlg(HcclReduceOp opType, void *sendBuffer, void *recvBuffer,
        std::vector<Slice> &orderedRingSlices, std::vector<u32> &ringOrder, HcclDataType dataType) const;
};

#endif
