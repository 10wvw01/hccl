/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_TRACE_EXECUTOR_TRACER_H_
#define ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_TRACE_EXECUTOR_TRACER_H_
#include "common/aicpu_hccl_def.h"
#include "utils/aicpu_hdc_utils.h"
#include "framework/aicpu_communicator.h"
#include "cann_error_reporter.h"

namespace dfx_tracer {
class ExecutorTracer {
public:
    explicit ExecutorTracer();
    static void BackGroundDfx(void *info);
    static void StopBackGroundDfx(void *info);
    static void SetCqeQueryInput(const uint32_t devId, const HcclComStreamInfo &streamInfo,
                                 CqeQueryInput &cqeQueryInput);
private:
    static void HandleKfcCommand(AicpuComContext *const ctx);
    static void HandleCqeStatus(AicpuComContext *const ctx);
    static void HandleCqeStatusByRank(AicpuComContext *const ctx, uint32_t rank);
    static void HandleBackGround(AicpuComContext *const ctx);
    static void StopLaunchCommandHandle(AicpuComContext *const ctx);
    static void KfcCommandHandle(AicpuComContext *const ctx);
    static void HandleCqeStatusInComm();
    static void HandleReportStatusInComm();
    static void HandleAICPUCommand(hccl::HcclCommAicpu *const commInfo);
    static void StopBackGround(AicpuComContext *const ctx,bool &isNotStop);
    static void StopKfcThread(AicpuComContext *const ctx,
                              std::vector<std::pair<std::string, hccl::HcclCommAicpu *>> aicpuCommInfo);
    static void HandleDestroyComm(AicpuComContext *const ctx);
    static void HandleSwitchNic(AicpuComContext *const ctx);
    static void PrintTaskException(const rtLogicCqReport_t &reportOfOne);
    static void HandleResumeChangeLink(AicpuComContext *const ctx);
};
// KfcCommand对应的处理函数，后续应该搞成注册的方式
class KfcCommandHandles {
public:
    static void ClearFunc(AicpuComContext *const ctx);
    static void StopFunc(AicpuComContext *const ctx);
};
class AICPUcommandHandles {
public:
    static void NsCommStop(hccl::HcclCommAicpu *const commInfo);
    static void NsCommClean(hccl::HcclCommAicpu *const commInfo);
};
}
#endif // ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_TRACE_EXECUTOR_TRACER_H_
