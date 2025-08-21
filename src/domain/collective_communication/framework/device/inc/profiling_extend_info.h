/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_PROFILING_PROFILING_EXTEND_INFO_H_
#define ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_PROFILING_PROFILING_EXTEND_INFO_H_
#include "hccl/base.h"
#include "toolchain/prof_common.h"
class SqeInfo;
class AicpuComContext;
namespace dfx {
constexpr u32 AC_MAX_RANK_NUM = 32U;
struct ProfilingExtendInfo {
    uint16_t lastSqeIdxs[AC_MAX_RANK_NUM];
};

class ProfilingExtendInfoHelper {
    using Handle = void (*)(const SqeInfo &sqeInfo, MsprofAicpuHcclTaskInfo &msprofAicpuMC2HcclInfo);

public:
    static void Init(const AicpuComContext *ctx);
    static void SqeInfo2MsprofAicpuMC2HcclInfo(const SqeInfo &sqeInfo, MsprofAicpuHcclTaskInfo &msprofAicpuMC2HcclInfo);
    static void Ctx2MsprofAicpuMC2HcclInfo(const AicpuComContext *ctx, MsprofAicpuHcclTaskInfo &msprofAicpuMC2HcclInfo);
    static std::string MsprofAicpuMC2HcclInfoToString(const MsprofAicpuHcclTaskInfo &msprofAicpuMC2HcclInfo);
    static std::string AicpuKfcProfCommTurnToString(const AicpuKfcProfCommTurn &aicpuKfcProfCommTurn);
    static void InitProfItemId();
    static void InitHcclInfo(MsprofAicpuHcclTaskInfo &msprofAicpuMC2HcclInfo);
private:
    static void AssembleProfInfoByType(const SqeInfo &sqeInfo, MsprofAicpuHcclTaskInfo &msprofAicpuMC2HcclInfo);
};
}  // namespace dfx
#endif  // ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_PROFILING_PROFILING_EXTEND_INFO_H_
