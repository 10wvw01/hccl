/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * NHR 算法原语 - 接口定义
 *
 * 职责：以纯函数形式实现 NHR 算法的三种核心原语
 *   - AllGather：每个 rank 提供一片数据，最终所有 rank 获得全部数据
 *   - ReduceScatter：每个 rank 提供全部数据，归约后每个 rank 获得一片
 *   - Scatter：root rank 将数据分片发送给所有 rank
 *
 * 设计原则：无类结构、纯函数、输入输出明确
 */

#ifndef NHR_PRIMITIVES_H
#define NHR_PRIMITIVES_H

#include "alg_primitive_types.h"

namespace ops_hccl {

// ===== NHR 辅助函数 =====

/** 计算 NHR 算法通信步数：ceil(log2(rankSize)) */
u32 CalcNhrStepNum(u32 rankSize);

/** 生成 NHR ReduceScatter 单步的 slice 索引 */
std::pair<std::vector<u32>, std::vector<u32>> GenNhrReduceScatterSliceIdxs(
    u32 rankSize, u32 myRankIdx, u32 step);

/** 生成 NHR AllGather 单步的 slice 索引 */
std::pair<std::vector<u32>, std::vector<u32>> GenNhrAllGatherSliceIdxs(
    u32 rankSize, u32 myRankIdx, u32 step, u32 nSteps);

/** 将数据按 rank 数量均衡切分为 SliceOffset 列表 */
std::vector<SliceOffset> GenNhrDataSlices(const NhrAlgParam& param);

// ===== NHR 三种核心原语 =====

/**
 * NHR ReduceScatter 原语
 * 每步包含：srcRank（rx来源）、dstRank（tx目标）、txSlices、rxSlices
 * 传输操作类型：SendRecvWriteReduce 或 SendRecvReadReduce（取决于 isDmaRead）
 */
SliceInfoList GenNhrReduceScatterSteps(const NhrAlgParam& param,
                                       const std::vector<SliceOffset>& dataSlices,
                                       bool isDmaRead);

/**
 * NHR AllGather 原语
 * 每步包含：srcRank（rx来源）、dstRank（tx目标）、txSlices、rxSlices
 * 传输操作类型：SendRecvWrite 或 SendRecvRead（取决于 isDmaRead）
 */
SliceInfoList GenNhrAllGatherSteps(const NhrAlgParam& param,
                                   const std::vector<SliceOffset>& dataSlices,
                                   bool isDmaRead);

/**
 * NHR Scatter 原语
 * root rank 将数据分片发送给所有 rank
 * 通过 NHR 的递归半数通信模式实现
 */
SliceInfoList GenNhrScatterSteps(const NhrAlgParam& param, u32 root,
                                 const std::vector<SliceOffset>& dataSlices,
                                 bool isDmaRead);

}  // namespace ops_hccl

#endif  // NHR_PRIMITIVES_H
