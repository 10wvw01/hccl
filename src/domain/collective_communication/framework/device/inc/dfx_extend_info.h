/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_DFX_EXTEND_INFO_H_
#define ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_DFX_EXTEND_INFO_H_
#include <cstdint>
#include <string>
#include <sstream>
#include <deque>
#include "aicpu_hccl_sqcq.h"

namespace dfx {
enum class KfcStatus : int64_t {
    kDefault = 0,
    kOneStart,
    kOneFinished,
    kTimeOut,
};

enum class PollStatus : int64_t {
    kDefault = 0,
    kStopAsException,
};
enum class CommandToKfc : int64_t {
    kDefault = 0,
    kClear,
    kStop,
    kRestart
};
enum class CommandToBackGroud : int64_t {
    kDefault = 0,
    kStop,
};
struct KfcRestartConfig {
    uint32_t tryRestartTimes{0U};
    uint32_t maxRestartTimes{1U}; // 最多重执行一次
};

struct TaskExceptionCqe {
    uint8_t sqeType{0};
    uint32_t errorCode{0};
};

struct DfxExtendInfo {
    KfcStatus kfcStatus = KfcStatus::kDefault;
    CqeStatus cqeStatus = CqeStatus::kDefault;
    PollStatus pollStatus = PollStatus::kDefault;
    TaskExceptionCqe cqeException;
    CommandToKfc commandToKfc = CommandToKfc::kDefault;
    KfcRestartConfig kfcRestartConfig;
    CommandToBackGroud commandToBackGroud = CommandToBackGroud::kDefault;
    DfxTimeOutConfig dfxTimeOutConfig;
};

class DfxExtendInfoHelper {
public:
    static void ResetTryRestartTimes(DfxExtendInfo &dfxExtendInfo);
    static void TryRestartOnceMore(DfxExtendInfo &dfxExtendInfo);
    static bool TryRestartTooManyTimes(const DfxExtendInfo &dfxExtendInfo);
};

}
#endif // ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_DFX_EXTEND_INFO_H_
