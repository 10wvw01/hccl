/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_ALG_NHR_OPT_WRAPPER
#define DPU_ALG_NHR_OPT_WRAPPER

#include "alg_param.h"
#include "template_utils.h"

namespace ops_hccl {

// 用于 step 同步的专用 notify 索引（不与 ACK(0)/DATA(1)/FIN_ACK(2) 冲突）
constexpr u32 NOTIFY_IDX_STEP_SYNC = 0;
constexpr u32 STEP_SYNC_TIMEOUT = 18000;

/**
 * @brief 统一的 BatchSR 接口，覆盖 NHR 所有收发场景：
 *        - 只发（tx 非空, rx 空）
 *        - 只收（tx 空, rx 非空）
 *        - 同对端收发（tx/rx 都非空, toRank == fromRank）
 *        - 不同对端收发（tx/rx 都非空, toRank != fromRank）
 *
 *        内部自动完成 step 同步 + 确定性排序，避免串行场景下的空转和死锁。
 *
 * @param stepInfo     当前 step 的收发信息（txSliceIdxs / rxSliceIdxs / toRank / fromRank）
 * @param channels     所有 rank 的 channel 映射
 * @param tempAlgParam 算法参数
 * @param repeat       当前 repeat 索引
 * @param myRank       当前 rank 的 user rank
 * @param templateRankSize 模板 rank 总数
 */
HcclResult BatchSRUnified(
    const AicpuNHRStepInfo &stepInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const TemplateDataParams &tempAlgParam,
    u32 repeat,
    u32 myRank,
    u32 templateRankSize);

}  // namespace ops_hccl

#endif  // DPU_ALG_NHR_OPT_WRAPPER
