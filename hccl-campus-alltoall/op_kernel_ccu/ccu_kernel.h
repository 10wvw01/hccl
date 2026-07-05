/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_MESH_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_ALL_TO_ALL_MESH_1D_MEM2MEM_H

#include <ios>
#include <memory>
#include <hccl/hccl_ccu_res.h>
#include <ccu/ccu_types.h>
#include <ccu/ccu_variable.hpp>
#include <ccu/ccu_event.hpp>
#include <ccu/ccu_primitives.hpp>
#include <ccu/ccu_launch.h>

#include "common.h"
#include "log.h"

namespace ccu = ::AscendC::ccu;

struct CcuKernelArgBase {
    ChannelHandle channels[CCU_MAX_RANK_SIZE];
    uint32_t      channelCount;
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel名称
    char kernelFuncName[64];
    // kernel函数
    void* kernelFunc;
    // KernelArg实例指针
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template<typename T>
    void setKernelArg(std::shared_ptr<T> arg) {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void*>(arg.get());
    }
};

struct AlgResourceCtxSerializable {
    CommBuffer cclMem;
    uint32_t notifyNumOnMainThread;
    std::vector<ThreadHandle> threads;
    // ccu
    std::vector<uint32_t> ccuKernelNum;
    std::vector<CcuKernelHandle> ccuKernels;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;

        binaryStream << cclMem;
        binaryStream << notifyNumOnMainThread;
        binaryStream << threads;
        binaryStream << ccuKernelNum;
        binaryStream << ccuKernels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);

        binaryStream >> cclMem;
        binaryStream >> notifyNumOnMainThread;
        binaryStream >> threads;
        binaryStream >> ccuKernelNum;
        binaryStream >> ccuKernels;
    }
};

// CCU返回码转换为HCCL返回码
inline HcclResult ConvertCcuToHccl(CcuResult ccuResult) {
    switch (ccuResult) {
        case CCU_SUCCESS: return HCCL_SUCCESS;
        case CCU_E_PARA: return HCCL_E_PARA;
        case CCU_E_PTR: return HCCL_E_PTR;
        case CCU_E_INTERNAL: return HCCL_E_INTERNAL;
        case CCU_E_NOT_SUPPORT: return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND: return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL: return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

namespace ops_ccu {

struct CcuKernelArgAllToAllMesh1DMem2Mem : public CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
};

struct AllToAllMesh1DMem2MemContext {
    const CcuKernelArgAllToAllMesh1DMem2Mem* arg;
    
    std::vector<ccu::Variable> input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable currentRankSliceInputOffset;
    ccu::Variable currentRankSliceOutputOffset;
    ccu::Variable sliceSize;
    ccu::Variable srcOffset;
    ccu::Variable srcStride;
    ccu::Variable dstOffset;
    ccu::Event event;

};

CcuResult CcuAlltoAllMesh1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_ccu

#endif // HCCL_CCU_KERNEL_ALL_TO_ALL_MESH_1D_MEM2MEM_H
