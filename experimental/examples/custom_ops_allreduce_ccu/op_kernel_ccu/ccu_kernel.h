/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_MEM2MEM_H

#include <ios>
#include "common.h"
#include "log.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl_ar {

constexpr uint64_t CCU_MS_INTERLEAVE = 8;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint32_t CCU_LOCAL_COPY_MS_PER_LOOP = 8;
constexpr uint32_t CCU_MS_LOCAL_COPY_LOOP_COUNT = 8;

struct LoopGroupConfig {
    uint32_t msInterleave;
    uint32_t loopCount;
    uint64_t memSlice;
};

struct LoopGroupResource {
    ccu::Array<ccu::Event>     completedEvent{0};
    ccu::Array<ccu::CcuBuffer> ccuBuf{0};
    uint32_t  eventCount;
    uint32_t  bufCount;
};

struct GroupOpSizeVars {
    ccu::Variable addrOffset;
    ccu::Variable loopParam;
    ccu::Variable parallelParam;
    ccu::Variable residual;
};

struct CcuLoopEntity {
    std::unique_ptr<ccu::Func> body[2];
    std::unique_ptr<ccu::Loop> loops[2];
    ccu::Variable              loopParam[2];
};

// AllReduce kernel参数：rankSize、rankId、归约数据类型和归约操作类型
struct CcuKernelArgAllReduceMesh1D : public CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
    HcclDataType dataType;
    HcclReduceOp reduceOp;
};

// AllReduce kernel执行上下文
struct AllReduceMesh1DMem2MemContext {
    const CcuKernelArgAllReduceMesh1D* arg;

    ccu::Variable input;                       // 本卡输入地址
    ccu::Variable output;                      // 本卡输出地址
    ccu::Variable token;                       // 本卡内存token
    std::vector<ccu::Variable> remoteInput;    // 远端输入地址（通过前同步获取）
    std::vector<ccu::Variable> remoteToken;    // 远端内存token（通过前同步获取）
    ccu::Variable currentSliceOffset;          // 当前数据分片偏移
    ccu::Variable sliceSize;                   // 当前分片数据大小
    ccu::Event event;                          // 同步事件

    GroupOpSizeVars goSize;
    LoopGroupConfig   moConfig;
    LoopGroupResource moRes;
    bool resourceAllocated = false;

    std::map<std::string, CcuLoopEntity> loopMap;

    void CreateLoopEntity(std::string loopStr) {
        loopMap.emplace(loopStr, CcuLoopEntity());
    }

    bool IsLoopEntityRegistered(std::string loopStr) {
        return loopMap.count(loopStr) != 0;
    }
};

CcuResult CcuAllReduceMesh1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl_ar

#endif // HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_MEM2MEM_H
