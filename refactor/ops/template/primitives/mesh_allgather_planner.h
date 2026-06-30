/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
 */

#ifndef MESH_ALLGATHER_PLANNER_H
#define MESH_ALLGATHER_PLANNER_H

#include "template_utils.h"

namespace ops_hccl {

// 描述上层算法描述选中的数据切分语义。
// 这个维度故意比 Mesh/NHR 或 EngineType 更细：不同 Mesh AllGather 变体即使运行在同一类
// engine 上，也可能使用不同的 offset 计算公式。
enum class MeshAllGatherSliceMode {
    NORMAL_FIXED,
    VARIABLE_COUNT,
    OMNIPIPE_STEP,
    COMMON_CHANNEL_SPLIT,
    Z_AXIS_DETOUR,
    MESH_CHUNK,
};

// Z-axis detour 会按固定边界/比例，把同一个 peer 的数据切到 level0 和 level1 两组 channel 上。
// 这个边界只在规划 channel 资源时明确；level0/level1 channel 合并后，不能再可靠地反推出原始边界。
struct ZAxisDetourConfig {
    u32 level0ChannelNumPerRank{0};
    u32 level1ChannelNumPerRank{0};
    double level0DataRatio{0.0};
};

// Primitive options 只携带无法从 TemplateDataParams 或 TemplateResource 推导出的切分信息。
// 在最终的直接函数调用链路中，sliceMode 应来自 TemplateDesc.variant，zAxis 应来自资源规划阶段记录的元数据。
struct MeshAllGatherPrimitiveOptions {
    // sliceMode 是旧 template 类型/算法变体在新 primitive 入口里的显式表达。
    // RunMeshAllGather 不再从 EngineType 或 channel 数量反推自己是哪种 Mesh AllGather。
    MeshAllGatherSliceMode sliceMode{MeshAllGatherSliceMode::NORMAL_FIXED};
    // Z-axis 需要 level0/level1 的 channel 边界；普通 Mesh AllGather 不需要这组配置。
    bool hasZAxisDetourConfig{false};
    ZAxisDetourConfig zAxis;
};

// SendRecv mode 是可执行计划的一部分，这样 RunMeshAllGather 执行每个 task 时不需要再检查各变体规则。
enum class MeshAllGatherSendRecvMode {
    DMA_READ,
    WRITE,
    BATCH_WRITE,
};

// 一个 peer/channel 对应一个可执行通信 task。
// planner 直接产出具体 DataSlice 列表，执行路径不需要关心 offset 来自固定切片、变长位移、
// OmniPipe step，还是 Z-axis channel 切分。
struct MeshAllGatherPeerChannelPlan {
    // connectedRank/connectedAlgRank 仍对应旧 RunAllGatherMesh 中“当前通信对端”的两个 rank 表达。
    u32 connectedRank{0};
    u32 connectedAlgRank{0};
    // channelIdx 表示 connectedRank 的第几个 channel；普通 Mesh 只有 0，channel split/Z-axis 会有多个。
    u32 channelIdx{0};
    // threadIdx 对应执行这个 connectedRank/channel 通信任务的 thread。
    u32 threadIdx{0};
    // linkRemote 对应旧代码里的 channels.at(connectedRank)[channelIdx]。
    const ChannelInfo *linkRemote{nullptr};
    // 四组 DataSlice 保留旧 SendRecvInfo 的组织方式：tx 是本 rank 发给对端，rx 是本 rank 从对端收。
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
};

// RunMeshAllGather 消费的完整计划。
// 这里是“变体相关的数据布局逻辑”和“公共 SendRecv 执行循环”的边界。
struct MeshAllGatherSlicePlan {
    // sendRecvMode 是本轮 AllGather 使用 SendRecvRead、SendRecvWrite 还是 SendRecvBatchWrite。
    MeshAllGatherSendRecvMode sendRecvMode{MeshAllGatherSendRecvMode::DMA_READ};
    // tasks 是已经展开好的 connectedRank/channel 通信任务列表，执行层只顺序消费它。
    std::vector<MeshAllGatherPeerChannelPlan> tasks;
};

// 只构造切片计划，不发起通信。
// 这个函数是旧 template 中 offset/channel 相关逻辑的预期抽取点。
HcclResult BuildMeshAllGatherSlicePlan(const TemplateDataParams &tempAlgParams,
                                       const TemplateResource &templateResource,
                                       const std::vector<u32> &ranks, u32 myRank,
                                       const MeshAllGatherPrimitiveOptions &options,
                                       MeshAllGatherSlicePlan &plan);

} // namespace ops_hccl

#endif
