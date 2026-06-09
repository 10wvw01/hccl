/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_ALG_DATA_TRANS_WRAPPER
#define DPU_ALG_DATA_TRANS_WRAPPER

#include "hcomm_primitives.h"
#include "hccl_types.h"
#include "alg_param.h"
#include "template_utils.h"

namespace ops_hccl {

struct DpuTaskexceptionInfo {
    HcclResult retCode;
    ChannelHandle handle;
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_RESERVED;
    EndpointLocType locationType = EndpointLocType::ENDPOINT_LOC_TYPE_RESERVED;
};

HcclResult SendWrite(const DataInfo &sendInfo, void *taskexpShmem = nullptr);
HcclResult RecvWrite(const DataInfo &recvInfo, void *taskexpShmem = nullptr);
HcclResult SendRecvWrite(const SendRecvInfo &sendRecvInfo, void *taskexpShmem = nullptr);

/* 检查函数返回值, 并返回指定错误码，触发taskexception */
inline HcclResult ChkRetAndTaskexception(HcclResult hcclRet, void *taskexpShmem, ChannelInfo channelInfo)
{
    if (UNLIKELY(hcclRet != HCCL_SUCCESS)) {
        if (hcclRet == HCCL_E_AGAIN) {
            HCCL_WARNING("[%s]call trace: hcclRet -> %d", __func__, hcclRet);
        } else {
            HCCL_ERROR("[%s]call trace: hcclRet -> %d", __func__, hcclRet);
            if (taskexpShmem != nullptr) {
                DpuTaskexceptionInfo dpuTaskexceptionInfo{};
                dpuTaskexceptionInfo.retCode = hcclRet;
                dpuTaskexceptionInfo.handle = channelInfo.handle;
                dpuTaskexceptionInfo.protocol = channelInfo.protocol;
                dpuTaskexceptionInfo.locationType = channelInfo.locationType;
                uint8_t *dstDataPtr = reinterpret_cast<uint8_t *>(taskexpShmem);
                memcpy_s(dstDataPtr, sizeof(DpuTaskexceptionInfo), &dpuTaskexceptionInfo, sizeof(DpuTaskexceptionInfo));
                memcpy_s(dstDataPtr + sizeof(DpuTaskexceptionInfo), sizeof(HcclResult), &hcclRet, sizeof(HcclResult));
            }
        }
    }
    return hcclRet;           
};

}
#endif // DPU_ALG_DATA_TRANS_WRAPPER