/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_PROF_H__
#define __AICPU_PROF_H__

#include "aicpu_hccl_def.h"

enum class KfcTimeLine : int64_t {
    HCC_EXEC_START_TIME,
    SEND_TASK_START_TIME,
    SEND_SQE_FINISH_TIME,
};
class HcclCommProf {
public:
    static void SetCurrentProf(uint64_t launchTime);
    static void SetKfcTimeLine(KfcTimeLine kfcTimeLine);
    static void SetDebugMode(uint8_t debugMode);
    static bool IsDebugModeEquals(const uint8_t mode);
    static bool NeedRecordTimeTaken();
    static AicpuComProf *GetCurrentAicpuProf();
    static void OutputProfLog();
    static uint32_t GetCurrentLoopCnt();
    static void AddRcdCnt(uint32_t idx, int32_t sqeCnt);
    static void AddProfLoopCnt(uint32_t addCnt);
private:
    static AicpuComProf *GetaicpuProfInst();
    static uint32_t GetProfTotalCnt();
    static void AddProfTotalCnt(uint32_t addCnt);
    static void SetProfLoopCnt(uint32_t setCnt);
    static uint8_t debugMode_;
};
#endif  // __AICPU_PROF_H__