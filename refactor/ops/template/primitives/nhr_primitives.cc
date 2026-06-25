/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * NHR 算法原语 - 实现
 *
 * 实现三种核心原语：AllGather、ReduceScatter、Scatter
 * 其他算子（如 AllReduce = ReduceScatter + AllGather）通过编排层组合实现
 */

#include "nhr_primitives.h"

namespace ops_hccl {

// ===== 辅助函数实现 =====

u32 CalcNhrStepNum(u32 rankSize)
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    return nSteps;
}

std::pair<std::vector<u32>, std::vector<u32>> GenNhrReduceScatterSliceIdxs(
    u32 rankSize, u32 myRankIdx, u32 step)
{
    u32 deltaRank = 1 << step;
    u32 txSliceIdx = (myRankIdx + rankSize - deltaRank) % rankSize;
    u32 rxSliceIdx = myRankIdx;

    u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);

    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;
    txSliceIdxs.reserve(nSlices);
    rxSliceIdxs.reserve(nSlices);

    for (u32 i = 0; i < nSlices; i++) {
        txSliceIdxs.push_back(txSliceIdx);
        rxSliceIdxs.push_back(rxSliceIdx);
        txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
    }
    return {txSliceIdxs, rxSliceIdxs};
}

std::pair<std::vector<u32>, std::vector<u32>> GenNhrAllGatherSliceIdxs(
    u32 rankSize, u32 myRankIdx, u32 step, u32 nSteps)
{
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 txSliceIdx = myRankIdx;
    u32 rxSliceIdx = (myRankIdx - (1 << (nSteps - 1 - step)) + rankSize) % rankSize;

    u32 nSlices = (rankSize - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);

    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;
    txSliceIdxs.reserve(nSlices);
    rxSliceIdxs.reserve(nSlices);

    for (u32 i = 0; i < nSlices; i++) {
        txSliceIdxs.push_back(txSliceIdx);
        rxSliceIdxs.push_back(rxSliceIdx);
        txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
    }
    return {txSliceIdxs, rxSliceIdxs};
}

std::vector<SliceOffset> GenNhrDataSlices(const NhrAlgParam& param)
{
    u32 sliceNum = param.rankSize;
    std::vector<SliceOffset> slices;
    slices.reserve(sliceNum);

    // 使用统一的尾块计算工具
    u64 baseSliceSize = CalcBaseSliceSize(param.count, param.dataTypeSize, sliceNum);
    u64 tailSize = CalcTailSize(param.count, param.dataTypeSize, sliceNum);
    u64 baseSliceCount = param.count / sliceNum;

    u64 offsetSize = 0;
    for (u32 sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
        u64 curSliceSize = CalcSliceSizeWithTail(sliceIdx, sliceNum, baseSliceSize, tailSize);
        u64 curSliceCount = (sliceIdx == sliceNum - 1 && tailSize > 0) ?
            (curSliceSize / param.dataTypeSize) : baseSliceCount;
        slices.emplace_back(offsetSize, curSliceSize, curSliceCount);
        offsetSize += curSliceSize;
    }
    return slices;
}

// ===== NHR ReduceScatter 原语实现 =====


HCCLResult RunNhrReduceScatter(const NhrAlgParam& param,bool isDmaRead)
{
    SliceInfoList steps;
    u32 nSteps = CalcNhrStepNum(param.rankSize);
    steps.reserve(nSteps);

    for (u32 step = 0; step < nSteps; ++step) {
        u32 deltaRank = 1 << step;
        u32 sendToIdx = (param.myRankIdx + param.rankSize - deltaRank) % param.rankSize;
        u32 recvFromIdx = (param.myRankIdx + deltaRank) % param.rankSize;

        auto [txSliceIdxs, rxSliceIdxs] = GenNhrReduceScatterSliceIdxs(
            param.rankSize, param.myRankIdx, step);

        TransferStep transferStep;
        transferStep.srcRank = recvFromIdx;
        transferStep.dstRank = sendToIdx;
        transferStep.threadIdx = 0;

        for (u32 i = 0; i < txSliceIdxs.size(); ++i) {
            u32 txIdx = txSliceIdxs[i];
            if (txIdx < dataSlices.size()) {
                transferStep.txSlices.push_back(dataSlices[txIdx]);
            }
        }
        for (u32 i = 0; i < rxSliceIdxs.size(); ++i) {
            u32 rxIdx = rxSliceIdxs[i];
            if (rxIdx < dataSlices.size()) {
                transferStep.rxSlices.push_back(dataSlices[rxIdx]);
            }
        }

        transferStep.op = SelectSendRecvReduceOp(isDmaRead);
        steps.push_back(std::move(transferStep));
    }
    return steps;
}

// ===== NHR AllGather 原语实现 =====
// 计算AllGather NHR每一步的数据切片，并调用合适的传输方式下发task
HCCLResult RunNhrAllGather(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource， EngineType engineType)
{
    SliceInfoList steps;
    u32 nSteps = CalcNhrStepNum(param.rankSize);
    steps.reserve(nSteps);

    for (u32 step = 0; step < nSteps; ++step) {
        u32 deltaRank = 1 << (nSteps - 1 - step);
        u32 sendToIdx = (param.myRankIdx + deltaRank) % param.rankSize;
        u32 recvFromIdx = (param.myRankIdx + param.rankSize - deltaRank) % param.rankSize;

        auto [txSliceIdxs, rxSliceIdxs] = GenNhrAllGatherSliceIdxs(
            param.rankSize, param.myRankIdx, step, nSteps);

        TransferStep transferStep;
        transferStep.srcRank = recvFromIdx;
        transferStep.dstRank = sendToIdx;
        transferStep.threadIdx = 0;

        for (u32 i = 0; i < txSliceIdxs.size(); ++i) {
            u32 txIdx = txSliceIdxs[i];
            if (txIdx < dataSlices.size()) {
                transferStep.txSlices.push_back(dataSlices[txIdx]);
            }
        }
        for (u32 i = 0; i < rxSliceIdxs.size(); ++i) {
            u32 rxIdx = rxSliceIdxs[i];
            if (rxIdx < dataSlices.size()) {
                transferStep.rxSlices.push_back(dataSlices[rxIdx]);
            }
        }

        SendRecv(tempAlgParams, templateResource, engineType);
    }
    return steps;
}

// ===== NHR Scatter 原语实现 =====

HCCLResult RunNhrScatter(const TemplateDataParams &tempAlgParams, TemplateResource &templateResource， EngineType engineType)
{
    SliceInfoList steps;
    u32 nSteps = CalcNhrStepNum(param.rankSize);
    steps.reserve(nSteps);

    // Scatter 使用 NHR 的递归半数通信模式
    // root rank 持有全部数据，逐步分发给其他 rank
    u32 myRelativeIdx = (param.myRankIdx + param.rankSize - root) % param.rankSize;

    for (u32 step = 0; step < nSteps; ++step) {
        u32 deltaRank = 1 << (nSteps - 1 - step);
        u32 sendToRelative = (myRelativeIdx + deltaRank) % param.rankSize;
        u32 recvFromRelative = (myRelativeIdx + param.rankSize - deltaRank) % param.rankSize;

        // 转换回实际 rank 索引
        u32 sendToIdx = (sendToRelative + root) % param.rankSize;
        u32 recvFromIdx = (recvFromRelative + root) % param.rankSize;

        TransferStep transferStep;
        transferStep.srcRank = recvFromIdx;
        transferStep.dstRank = sendToIdx;
        transferStep.threadIdx = 0;

        // Scatter 阶段：发送自己持有的部分数据，接收新的部分数据
        // 在 step 中，发送的是 sendToIdx 对应的数据片
        if (sendToIdx < dataSlices.size()) {
            transferStep.txSlices.push_back(dataSlices[sendToIdx]);
        }
        if (recvFromIdx < dataSlices.size()) {
            transferStep.rxSlices.push_back(dataSlices[recvFromIdx]);
        }

        // Scatter 不做归约，使用普通 SendRecv
        SelectSendRecvOp(tempAlgParams, templateResource, engineType);
    }
    return steps;
}

}  // namespace ops_hccl
