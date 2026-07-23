/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <hccl/hcomm_primitives.h>
#include "log.h"
#include "ccu_launch.h"
#include "ccu_res.h"
#include "common.h"
#include "ccu_kernel.h"
#include "utils.h"
using namespace ops_hccl_ar;

namespace ops_hccl_ar {
constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128; // 地址对齐
constexpr uint64_t UB_MAX_DATA_SIZE = 256*1024*1024; // UB协议一次传输最大size，单位Byte

// ---------------------------------------------------------------------------
// 下发一次CCU kernel slice
// AllReduce与AllGather的区别：
// - output大小等于input大小（AllGather中output是input的rankSize倍）
// - output偏移为0（AllGather中output偏移为dataSize * myRank）
// ---------------------------------------------------------------------------
static HcclResult LaunchCcuKernelSlice(const AlgResourceCtxSerializable &resCtx,
                                        uint64_t inputAddr, uint64_t outputAddr,
                                        uint64_t token, uint64_t sliceCount,
                                        uint64_t dataTypeSize)
{
    uint64_t sliceSize = sliceCount * dataTypeSize;
    uint64_t currentSliceOffset = 0; // AllReduce中input和output偏移均为0

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice     = CCU_MS_SIZE * CCU_LOCAL_COPY_MS_PER_LOOP;
    auto goSize         = CalGoSize(sliceSize, config);

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        currentSliceOffset,
        sliceSize,
        goSize[0],
        goSize[1],
        goSize[2],
        goSize[3],
    };

    CcuResult launchRet = HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0],
                                                taskArgs.data(), taskArgs.size());
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuAllReduceMesh1DMem2Mem::ExecOp] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_DEBUG("[CcuAllReduceMesh1DMem2Mem::ExecOp] start");

    uint32_t dataTypeSize = SIZE_TABLE[param.dataType];
    uint64_t dataSize = param.count * dataTypeSize;
    uint64_t count = param.count;

    if (count == 0) { // 数据量为0，直接返回
        HCCL_INFO("[CcuAllReduceMesh1DMem2Mem] DataCount == 0, ExecOp Run Ends.");
        return HcclResult::HCCL_SUCCESS;
    }

    if (param.rankSize == 1) { // 单卡，直接本地拷贝
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize)));
        HCCL_INFO("[CcuAllReduceMesh1DMem2Mem] RankSize == 1, ExecOp Run Ends.");
        return HCCL_SUCCESS;
    }

    // 大数据分片处理：每次最多传输UB_MAX_DATA_SIZE
    uint64_t maxDataSizePerLoop = UB_MAX_DATA_SIZE;
    uint64_t maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
    uint64_t loopCount = count / maxDataCountPerLoop + static_cast<uint64_t>(count % maxDataCountPerLoop != 0);
    uint64_t processedDataCount = 0;

    uint64_t token = 0;
    uint64_t baseInputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t baseOutputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    if (param.inputPtr != nullptr) {
        HcommCcuGetMemToken(baseInputAddr, static_cast<uint64_t>(dataSize), &token);
    } else if (param.outputPtr != nullptr) {
        HcommCcuGetMemToken(baseOutputAddr, static_cast<uint64_t>(dataSize), &token);
    }

    for (uint64_t loop = 0; loop < loopCount; loop++) {
        uint64_t sliceCount = std::min(maxDataCountPerLoop, count - loop * maxDataCountPerLoop);
        uint64_t inputAddr = baseInputAddr + processedDataCount * dataTypeSize;
        uint64_t outputAddr = baseOutputAddr + processedDataCount * dataTypeSize;

        CHK_RET(LaunchCcuKernelSlice(resCtx, inputAddr, outputAddr, token, sliceCount, dataTypeSize));

        processedDataCount += sliceCount;
    }

    HCCL_DEBUG("[CcuAllReduceMesh1DMem2Mem::ExecOp] end");
    return HCCL_SUCCESS;
}
} // namespace ops_hccl_ar
