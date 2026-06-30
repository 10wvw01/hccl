/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
 */

#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

#include "mesh_allgather_planner.h"

namespace {

using namespace ops_hccl;

constexpr u32 RANK_SIZE = 4;
constexpr u32 CHANNELS_PER_RANK = 4;
constexpr u64 DATA_TYPE_SIZE = sizeof(int32_t);

void *FakePtr(uint64_t value)
{
    return reinterpret_cast<void *>(static_cast<uintptr_t>(value));
}

HcclMem FakeMem(uint64_t addr, uint64_t size)
{
    HcclMem mem;
    mem.addr = FakePtr(addr);
    mem.size = size;
    return mem;
}

ChannelInfo MakeChannelInfo(u32 connectedRank, u32 channelIdx)
{
    ChannelInfo channel;
    channel.isValid = true;
    channel.remoteRank = connectedRank;
    channel.protocol = CommProtocol::COMM_PROTOCOL_HCCS;
    channel.locationType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    channel.portGroupSize = CHANNELS_PER_RANK;
    channel.remoteCclMem = FakeMem(0x20000000 + connectedRank * 0x100000 + channelIdx * 0x10000, 0x1000000);
    channel.remoteOutputGraphMode = FakeMem(0x30000000 + connectedRank * 0x100000 + channelIdx * 0x10000, 0x1000000);
    return channel;
}

TemplateResource MakeTemplateResource(const std::vector<u32> &ranks)
{
    TemplateResource resource;
    for (u32 connectedRank : ranks) {
        std::vector<ChannelInfo> channels;
        for (u32 channelIdx = 0; channelIdx < CHANNELS_PER_RANK; ++channelIdx) {
            channels.emplace_back(MakeChannelInfo(connectedRank, channelIdx));
        }
        resource.channels[connectedRank] = channels;
    }
    return resource;
}

TemplateDataParams MakeTemplateDataParams(BufferType inBuffType = BufferType::INPUT,
                                          BufferType outBuffType = BufferType::OUTPUT,
                                          bool enableRemoteMemAccess = false,
                                          u32 repeatNum = 1)
{
    TemplateDataParams params;
    params.dataType = HCCL_DATA_TYPE_INT32;
    params.count = 256;
    params.sliceSize = params.count * DATA_TYPE_SIZE;
    params.tailSize = 128 * DATA_TYPE_SIZE;
    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = params.sliceSize;
    params.repeatNum = repeatNum;
    params.inputRepeatStride = params.sliceSize * RANK_SIZE + 128;
    params.outputRepeatStride = params.sliceSize * RANK_SIZE + 256;
    params.enableRemoteMemAccess = enableRemoteMemAccess;

    params.buffInfo.inputPtr = FakePtr(0x10000000);
    params.buffInfo.outputPtr = FakePtr(0x11000000);
    params.buffInfo.hcclBuff = FakeMem(0x12000000, 0x1000000);
    params.buffInfo.inBuffType = inBuffType;
    params.buffInfo.outBuffType = outBuffType;
    params.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    params.buffInfo.inputSize = params.sliceSize * RANK_SIZE;
    params.buffInfo.outputSize = params.sliceSize * RANK_SIZE;
    params.buffInfo.hcclBuffSize = params.sliceSize * RANK_SIZE;
    params.buffInfo.inBuffBaseOff = 0;
    params.buffInfo.outBuffBaseOff = 0;
    params.buffInfo.hcclBuffBaseOff = 0;
    return params;
}

const char *SendRecvModeName(MeshAllGatherSendRecvMode mode)
{
    switch (mode) {
        case MeshAllGatherSendRecvMode::DMA_READ:
            return "DMA_READ";
        case MeshAllGatherSendRecvMode::WRITE:
            return "WRITE";
        case MeshAllGatherSendRecvMode::BATCH_WRITE:
            return "BATCH_WRITE";
        default:
            return "UNKNOWN";
    }
}

void DumpSlices(const char *tag, const std::vector<DataSlice> &slices)
{
    for (u32 i = 0; i < slices.size(); ++i) {
        const DataSlice &slice = slices[i];
        std::cout << "    " << tag << "[" << i << "]"
                  << " addr=" << slice.addr_
                  << " offset=" << slice.offset_
                  << " size=" << slice.size_
                  << " count=" << slice.count_
                  << std::endl;
    }
}

void DumpPlan(const MeshAllGatherSlicePlan &plan)
{
    std::cout << "sendRecvMode=" << SendRecvModeName(plan.sendRecvMode)
              << " taskNum=" << plan.tasks.size() << std::endl;
    for (u32 taskIdx = 0; taskIdx < plan.tasks.size(); ++taskIdx) {
        const MeshAllGatherPeerChannelPlan &task = plan.tasks[taskIdx];
        std::cout << "task[" << taskIdx << "]"
                  << " threadIdx=" << task.threadIdx
                  << " connectedRank=" << task.connectedRank
                  << " connectedAlgRank=" << task.connectedAlgRank
                  << " channelIdx=" << task.channelIdx
                  << " linkRemote=" << task.linkRemote
                  << std::endl;
        DumpSlices("txSrc", task.txSrcSlices);
        DumpSlices("txDst", task.txDstSlices);
        DumpSlices("rxSrc", task.rxSrcSlices);
        DumpSlices("rxDst", task.rxDstSlices);
    }
}

MeshAllGatherPrimitiveOptions MakeZAxisOptions()
{
    MeshAllGatherPrimitiveOptions options;
    options.sliceMode = MeshAllGatherSliceMode::Z_AXIS_DETOUR;
    options.hasZAxisDetourConfig = true;
    options.zAxis.level0ChannelNumPerRank = 2;
    options.zAxis.level1ChannelNumPerRank = 2;
    options.zAxis.level0DataRatio = 0.5;
    return options;
}

bool SameSlice(const DataSlice &lhs, const DataSlice &rhs, const char *tag,
               u32 taskIdx, u32 sliceIdx, std::string &reason)
{
    if (lhs.addr_ == rhs.addr_ && lhs.offset_ == rhs.offset_ &&
        lhs.size_ == rhs.size_ && lhs.count_ == rhs.count_) {
        return true;
    }
    std::ostringstream oss;
    oss << "task[" << taskIdx << "] " << tag << "[" << sliceIdx << "] mismatch"
        << " new(addr=" << lhs.addr_ << ",off=" << lhs.offset_ << ",size=" << lhs.size_
        << ",count=" << lhs.count_ << ")"
        << " old(addr=" << rhs.addr_ << ",off=" << rhs.offset_ << ",size=" << rhs.size_
        << ",count=" << rhs.count_ << ")";
    reason = oss.str();
    return false;
}

bool SameSliceList(const std::vector<DataSlice> &lhs, const std::vector<DataSlice> &rhs,
                   const char *tag, u32 taskIdx, std::string &reason)
{
    if (lhs.size() != rhs.size()) {
        std::ostringstream oss;
        oss << "task[" << taskIdx << "] " << tag << " size mismatch new="
            << lhs.size() << " old=" << rhs.size();
        reason = oss.str();
        return false;
    }
    for (u32 i = 0; i < lhs.size(); ++i) {
        if (!SameSlice(lhs[i], rhs[i], tag, taskIdx, i, reason)) {
            return false;
        }
    }
    return true;
}

bool SamePlan(const MeshAllGatherSlicePlan &newPlan, const MeshAllGatherSlicePlan &oldPlan, std::string &reason)
{
    if (newPlan.sendRecvMode != oldPlan.sendRecvMode) {
        reason = "sendRecvMode mismatch";
        return false;
    }
    if (newPlan.tasks.size() != oldPlan.tasks.size()) {
        std::ostringstream oss;
        oss << "taskNum mismatch new=" << newPlan.tasks.size() << " old=" << oldPlan.tasks.size();
        reason = oss.str();
        return false;
    }
    for (u32 i = 0; i < newPlan.tasks.size(); ++i) {
        const auto &lhs = newPlan.tasks[i];
        const auto &rhs = oldPlan.tasks[i];
        if (lhs.connectedRank != rhs.connectedRank ||
            lhs.connectedAlgRank != rhs.connectedAlgRank ||
            lhs.channelIdx != rhs.channelIdx ||
            lhs.threadIdx != rhs.threadIdx ||
            lhs.linkRemote != rhs.linkRemote) {
            std::ostringstream oss;
            oss << "task[" << i << "] meta mismatch"
                << " new(rank=" << lhs.connectedRank << ",algRank=" << lhs.connectedAlgRank
                << ",channel=" << lhs.channelIdx << ",thread=" << lhs.threadIdx
                << ",link=" << lhs.linkRemote << ")"
                << " old(rank=" << rhs.connectedRank << ",algRank=" << rhs.connectedAlgRank
                << ",channel=" << rhs.channelIdx << ",thread=" << rhs.threadIdx
                << ",link=" << rhs.linkRemote << ")";
            reason = oss.str();
            return false;
        }
        if (!SameSliceList(lhs.txSrcSlices, rhs.txSrcSlices, "txSrc", i, reason) ||
            !SameSliceList(lhs.txDstSlices, rhs.txDstSlices, "txDst", i, reason) ||
            !SameSliceList(lhs.rxSrcSlices, rhs.rxSrcSlices, "rxSrc", i, reason) ||
            !SameSliceList(lhs.rxDstSlices, rhs.rxDstSlices, "rxDst", i, reason)) {
            return false;
        }
    }
    return true;
}

HcclResult GetAlgRank(const std::vector<u32> &ranks, u32 rank, u32 &algRank)
{
    for (u32 i = 0; i < ranks.size(); ++i) {
        if (ranks[i] == rank) {
            algRank = i;
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_PARA;
}

void AppendOldZAxisSlices(const TemplateDataParams &params, const MeshAllGatherPrimitiveOptions &options,
                          const std::vector<u32> &ranks, u32 myAlgRank, u32 connectedAlgRank,
                          const ChannelInfo &linkRemote, u32 idx, const std::vector<u64> &elemCountOut,
                          const std::vector<u64> &sizeOut, const std::vector<u64> &elemOffset,
                          MeshAllGatherPeerChannelPlan &task)
{
    for (u32 rpt = 0; rpt < params.repeatNum; ++rpt) {
        const u64 outBaseOff = params.buffInfo.outBuffBaseOff + rpt * params.outputRepeatStride;
        const u64 scratchRepeatStride = params.sliceSize * static_cast<u32>(ranks.size());
        u64 scratchBase = params.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
        if (params.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
            scratchBase = params.buffInfo.hcclBuffBaseOff + rpt * params.inputRepeatStride;
        }
        const u64 txOutOffset = params.outputSliceStride * myAlgRank + outBaseOff + elemOffset[idx];
        const u64 txScratchOffset = scratchBase + params.sliceSize * myAlgRank + elemOffset[idx];
        const u64 txDstOffset = (!params.enableRemoteMemAccess) ? txScratchOffset : txOutOffset;
        const u64 rxOutOffset = params.outputSliceStride * connectedAlgRank + outBaseOff + elemOffset[idx];
        const u64 rxScratchOffset = scratchBase + params.inputSliceStride * connectedAlgRank + elemOffset[idx];
        const u64 rxSrcOffset = (!params.enableRemoteMemAccess) ? rxScratchOffset : rxOutOffset;
        void *txDstPtr = (!params.enableRemoteMemAccess) ?
            linkRemote.remoteCclMem.addr : linkRemote.remoteOutputGraphMode.addr;
        void *rxSrcPtr = (!params.enableRemoteMemAccess) ?
            linkRemote.remoteCclMem.addr : linkRemote.remoteOutputGraphMode.addr;

        (void)options;
        task.txSrcSlices.emplace_back(params.buffInfo.outputPtr, txOutOffset, sizeOut[idx], elemCountOut[idx]);
        task.txDstSlices.emplace_back(txDstPtr, txDstOffset, sizeOut[idx], elemCountOut[idx]);
        task.rxDstSlices.emplace_back(params.buffInfo.outputPtr, rxOutOffset, sizeOut[idx], elemCountOut[idx]);
        task.rxSrcSlices.emplace_back(rxSrcPtr, rxSrcOffset, sizeOut[idx], elemCountOut[idx]);
    }
}

HcclResult BuildOldZAxisOraclePlan(const TemplateDataParams &params, const TemplateResource &resource,
                                   const std::vector<u32> &ranks, u32 myRank,
                                   const MeshAllGatherPrimitiveOptions &options,
                                   MeshAllGatherSlicePlan &plan)
{
    plan.tasks.clear();
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(ranks, myRank, myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[params.dataType];
    const bool dmaRead = (params.buffInfo.inBuffType == BufferType::HCCL_BUFFER &&
                          params.buffInfo.outBuffType != BufferType::HCCL_BUFFER);
    plan.sendRecvMode = dmaRead ? MeshAllGatherSendRecvMode::DMA_READ : MeshAllGatherSendRecvMode::WRITE;

    std::vector<ChannelInfo> mergedChannels;
    for (u32 rank : ranks) {
        if (rank == myRank) {
            continue;
        }
        const auto &rankChannels = resource.channels.at(rank);
        mergedChannels.insert(mergedChannels.end(), rankChannels.begin(), rankChannels.end());
    }

    for (u32 threadIdx = 0; threadIdx < mergedChannels.size(); ++threadIdx) {
        const u32 connectedRank = mergedChannels[threadIdx].remoteRank;
        const auto &curChannels = resource.channels.at(connectedRank);
        const u32 channelsPerRank = static_cast<u32>(curChannels.size());
        const u32 idx = (channelsPerRank == 0) ? 0 : (threadIdx % channelsPerRank);
        u32 connectedAlgRank = 0;
        CHK_RET(GetAlgRank(ranks, connectedRank, connectedAlgRank));

        u64 sliceSize = params.sliceSize;
        if (dmaRead) {
            if (params.tailSize != 0 && connectedAlgRank == ranks.size() - 1) {
                sliceSize = params.tailSize;
            }
        } else if (params.tailSize != 0 && myAlgRank == ranks.size() - 1) {
            sliceSize = params.tailSize;
        }
        const u64 sliceCount = sliceSize / dataTypeSize;
        std::vector<u64> elemCountOut;
        std::vector<u64> sizeOut;
        std::vector<u64> elemOffset;
        CHK_RET(CalcDataSplitByPortGroupZAxisDetour(sliceCount, dataTypeSize, curChannels,
                                                    elemCountOut, sizeOut, elemOffset,
                                                    options.zAxis.level0ChannelNumPerRank,
                                                    options.zAxis.level1ChannelNumPerRank,
                                                    static_cast<float>(options.zAxis.level0DataRatio)));

        const ChannelInfo &linkRemote = resource.channels.at(connectedRank)[idx];
        MeshAllGatherPeerChannelPlan task;
        task.connectedRank = connectedRank;
        task.connectedAlgRank = connectedAlgRank;
        task.channelIdx = idx;
        task.threadIdx = threadIdx;
        task.linkRemote = &linkRemote;
        AppendOldZAxisSlices(params, options, ranks, myAlgRank, connectedAlgRank, linkRemote, idx,
                             elemCountOut, sizeOut, elemOffset, task);
        plan.tasks.emplace_back(task);
    }
    return HCCL_SUCCESS;
}

bool RunScenario(const char *name, const TemplateDataParams &params, const TemplateResource &resource,
                 const std::vector<u32> &ranks, u32 myRank, const MeshAllGatherPrimitiveOptions &options,
                 bool dumpPlan)
{
    MeshAllGatherSlicePlan plan;
    HcclResult result = BuildMeshAllGatherSlicePlan(params, resource, ranks, myRank, options, plan);
    if (result != HCCL_SUCCESS) {
        std::cerr << name << ": BuildMeshAllGatherSlicePlan failed, result=" << result << std::endl;
        return false;
    }

    if (dumpPlan) {
        DumpPlan(plan);
    }

    MeshAllGatherSlicePlan oldPlan;
    result = BuildOldZAxisOraclePlan(params, resource, ranks, myRank, options, oldPlan);
    if (result != HCCL_SUCCESS) {
        std::cerr << name << ": BuildOldZAxisOraclePlan failed, result=" << result << std::endl;
        return false;
    }
    std::string reason;
    if (!SamePlan(plan, oldPlan, reason)) {
        std::cerr << name << ": Z-axis oracle compare failed: " << reason << std::endl;
        return false;
    }
    std::cout << name << ": Z-axis oracle compare passed. taskNum=" << plan.tasks.size() << std::endl;
    return true;
}

} // namespace

int main()
{
    const std::vector<u32> ranks{0, 1, 2, 3};
    TemplateResource resource = MakeTemplateResource(ranks);
    MeshAllGatherPrimitiveOptions options = MakeZAxisOptions();

    bool ok = true;
    ok = RunScenario("write_non_tail_rank", MakeTemplateDataParams(), resource, ranks, 1, options, true) && ok;
    ok = RunScenario("write_tail_rank", MakeTemplateDataParams(), resource, ranks, 3, options, false) && ok;
    ok = RunScenario("dma_read_tail_peer",
                     MakeTemplateDataParams(BufferType::HCCL_BUFFER, BufferType::OUTPUT),
                     resource, ranks, 1, options, false) && ok;
    ok = RunScenario("remote_mem_access",
                     MakeTemplateDataParams(BufferType::INPUT, BufferType::OUTPUT, true),
                     resource, ranks, 1, options, false) && ok;
    ok = RunScenario("repeat_hccl_input",
                     MakeTemplateDataParams(BufferType::HCCL_BUFFER, BufferType::OUTPUT, false, 2),
                     resource, ranks, 1, options, false) && ok;

    if (!ok) {
        return 1;
    }
    std::cout << "All Z-axis oracle scenarios passed." << std::endl;
    return 0;
}
