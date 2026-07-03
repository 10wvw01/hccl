/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_KERNEL_ARG_BASE_H
#define OPS_HCCL_CCU_KERNEL_ARG_BASE_H

#include <cstdint>
#include <vector>

#include "alg_param.h"

namespace ops_hccl {

/**
 * CCU_MS (mem2mem) 模式的通用 KernelArg 基类。
 * 继承自 src/ops/op_common/inc/alg_param.h 的 CcuKernelArgBase（含 channels[CCU_MAX_RANK_SIZE] + channelCount）。
 * 具体算子（如 AllGatherMeshMsKernelCtx）可再派生本类添加算子特定字段。
 * 参考实现：src/ops/all_reduce/template/ccu/kernel/ccu_kernel_all_reduce_mesh1d_mem2mem.h 的
 * CcuKernelArgAllReduceMeshMem2Mem1D。
 */
struct CcuKernelArgBaseMs : public CcuKernelArgBase {
    uint64_t rankSize{0};
    uint32_t rankId{INVALID_VALUE_RANKID};
    OpParam opParam{};
    std::vector<std::vector<uint32_t>> subCommRanks;
};

/**
 * CCU_SCHE (调度) 模式的通用 KernelArg 基类。
 * 当前结构与 CcuKernelArgBaseMs 相同，分离保留以支持后续调度模式专属扩展
 * （如 ifHandleSelfRank、axisId 等字段可下沉到本类）。
 * 参考实现：src/ops/all_gather/template/ccu/kernel/ccu_kernel_all_gather_mesh1d.h 的
 * CcuKernelArgAllGatherMesh1D。
 */
struct CcuKernelArgBaseSche : public CcuKernelArgBase {
    uint64_t rankSize{0};
    uint32_t rankId{INVALID_VALUE_RANKID};
    OpParam opParam{};
    std::vector<std::vector<uint32_t>> subCommRanks;
    bool ifHandleSelfRank{true};
};

}  // namespace ops_hccl

#endif  // OPS_HCCL_CCU_KERNEL_ARG_BASE_H
