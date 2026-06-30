/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include "mesh_primitives.h"

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

TemplateDataParams MakeTemplateDataParams()
{
    TemplateDataParams params;
    params.dataType = HCCL_DATA_TYPE_INT32;
    params.count = 256;
    params.sliceSize = params.count * DATA_TYPE_SIZE;
    params.tailSize = 128 * DATA_TYPE_SIZE;
    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = params.sliceSize;
    params.repeatNum = 1;
    params.inputRepeatStride = params.sliceSize * RANK_SIZE;
    params.outputRepeatStride = params.sliceSize * RANK_SIZE;
    params.enableRemoteMemAccess = false;

    params.buffInfo.inputPtr = FakePtr(0x10000000);
    params.buffInfo.outputPtr = FakePtr(0x11000000);
    params.buffInfo.hcclBuff = FakeMem(0x12000000, 0x1000000);
    params.buffInfo.inBuffType = BufferType::INPUT;
    params.buffInfo.outBuffType = BufferType::OUTPUT;
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

} // namespace

int main()
{
    const std::vector<u32> ranks{0, 1, 2, 3};
    const u32 myRank = 1;

    TemplateDataParams params = MakeTemplateDataParams();
    TemplateResource resource = MakeTemplateResource(ranks);
    MeshAllGatherPrimitiveOptions options = MakeZAxisOptions();

    MeshAllGatherSlicePlan plan;
    HcclResult result = BuildMeshAllGatherSlicePlan(params, resource, ranks, myRank, options, plan);
    if (result != HCCL_SUCCESS) {
        std::cerr << "BuildMeshAllGatherSlicePlan failed, result=" << result << std::endl;
        return static_cast<int>(result);
    }

    DumpPlan(plan);
    return 0;
}
