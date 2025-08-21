/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_SHARE_DATA_MANAGER
#define AICPU_SHARE_DATA_MANAGER

#include "base.h"
#include "aicpu_operator_pub.h"
#include "coll_alg_param.h"

namespace hccl {

// 管理aicpu和custom进程的公共数据
class AicpuShareDataManager {
public:
    AicpuShareDataManager() {};
    ~AicpuShareDataManager() {};
    HcclResult Init(u64 addr, u64 size);

    // taskException
    u32 GetOpRingBufferIdx();
    HcclResult RecordOpInfo(const std::string &newTag, OpParam &opParam, u32 userRank);
    const AicpuOpInfo* GetAicpuOpInfo(u32 opRingBufferIdx);

private:
    AicpuCustomParam *aicpuCustomParam_ = nullptr;
};
}
#endif