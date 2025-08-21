/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MC2_AICPU_UTILS_H__
#define __MC2_AICPU_UTILS_H__

#include <string>
#include "aicpu_schedule/aicpu_sharder/aicpu_context.h"
#include "common/aicpu_hccl_def.h"
#include "hdc_pub.h"

class MC2AicpuUtils {
public:
    static int32_t GetCpuId();
    static int32_t GetCurClusterId();
    static void PrintHcclCombinOpParam(const HccCommResParamTask &commParam);
    static void PrintHcclCommParamDesc(const HcclCommParamDesc &desc);
    static void PrintKFCTask(const KFCTask &task);
    static void PrintTilingData(const HcclKFCTilingData &tilingData);
    static void PrintTilingDataError(const HcclKFCTilingData &tilingData);
    static void PrintTilingData(const Mc2InitTilingInner &tilingData);
    static void PrintTilingDataError(const Mc2InitTilingInner &tilingData);
    static void PrintMC2AicpuContext(const AicpuComContext &ctx);
    static void PrintMC2AicpuContextError(const AicpuComContext &ctx);
    static void PrintMC2HcclOpResParam(const HcclOpResParam *resParam);
    static void PrintApiBuffer(const void * const buffer, uint64_t totalSize, const std::string &desc);
    static void PrintBuffer(const void * const buffer, uint32_t totalSize, const std::string &desc);
    static void PrintBuffer(AicpuComContext *ctx, const AivAicpuOpParam &msgAddr);
    static HcclResult WaitTaskFinish(AicpuComContext *ctx, bool isWaitTask = true);
    static HcclResult TraceProfSubmit();
    static int GetSendCnt(AicpuComContext *ctx);
    static int GetRecvCnt(AicpuComContext *ctx);
    static void PrintAicDebugInfo(AicpuComContext *ctx);
    static bool IsDebugModeEquals(const AicpuComContext &ctx, const uint8_t Mode);
    static bool NeedRecordTimeTaken(const AicpuComContext &ctx);
    static void PrintAllHcclMsgAreaError(HcclMsgArea *hcclMsgArea, u32 rankSize);
    static void PrintAllHcclMsgAreaEvent(HcclMsgArea *hcclMsgArea, u32 rankSize);
    static void PrintAllHcclMsgAreaError(HcclMsgAreaForMultiQue *hcclMsgArea, u32 rankSize);
    static void PrintAllHcclMsgAreaEvent(HcclMsgAreaForMultiQue *hcclMsgArea, u32 rankSize);
    static uint32_t GenXor(HcclMsg *msg);
    static HcclResult Getkey(const AicpuComContext &ctx, u32 remoteRankId, const void *userAddr,
        u64 length, u32 &outKey, int32_t keyType);
    static HcclResult PostSend(const AicpuComContext &ctx, u32 remoteRankId, struct std::vector<hccl::Transport::Buffer> &remoteBuf,
        struct std::vector<hccl::Transport::Buffer> &localBuf, bool isWrite);
    static u32 GetBlockNum();
    static u32 GetBlockIdx();
};
#endif // __MC2_AICPU_UTILS_HPP__