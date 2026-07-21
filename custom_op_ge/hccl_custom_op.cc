/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

#include "hccl_custom_op.h"

#include "hccl/hcom.h"
#include "log.h"

namespace hccl {
ge::graphStatus HcclCustomOpBase::Execute(gert::EagerOpExecutionContext *ctx)
{
    HcclOpState st;
    st.ctx = ctx;
    HCCL_GE_CHK_RET(ExtractParams(st));
    HCCL_GE_CHK_RET(GetCommunicator(st));
    HCCL_GE_CHK_RET(CalcResources(st));
    HCCL_GE_CHK_RET(LaunchHcclOp(st));
    HCCL_GE_CHK_RET(HandleOutput(st));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcclCustomOpBase::GetCommunicator(HcclOpState &st)
{
    if (st.group == nullptr) {
        HCCL_ERROR("HcclCustomOpBase::GetCommunicator: group is null.");
        return ge::GRAPH_FAILED;
    }
    HcclComm comm = nullptr;
    HcclResult ret = HcomGetCommHandleByGroup(st.group, &comm);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("HcclCustomOpBase::GetCommunicator: HcomGetCommHandleByGroup failed for group '%s', ret=%d.",
                   st.group, static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    st.comm = comm;
    HCCL_INFO("HcclCustomOpBase::GetCommunicator: group='%s' comm=%p.", st.group, st.comm);
    return ge::GRAPH_SUCCESS;
}
}  // namespace hccl
