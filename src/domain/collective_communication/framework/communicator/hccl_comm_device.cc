/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "hccl_comm_pub.h"
 
namespace hccl {
HcclResult hcclComm::RegistTaskAbortHandler() const
{
    return HCCL_SUCCESS;
}
 
HcclResult hcclComm::UnRegistTaskAbortHandler() const
{
    return HCCL_SUCCESS;
}

HcclResult hcclComm::GetOneSidedService(IHcclOneSidedService** service)
{
    return HCCL_SUCCESS;
}
HcclResult hcclComm::InitOneSidedServiceNetDevCtx(u32 remoteRankId)
{
    return HCCL_SUCCESS;
}
HcclResult hcclComm::OneSidedServiceStartListen(NicType nicType,HcclNetDevCtx netDevCtx)
{
    return HCCL_SUCCESS;
}
HcclResult hcclComm::GetOneSidedServiceDevIpAndPort(NicType nicType, HcclIpAddress& ipAddress, u32& port)
{
    return HCCL_SUCCESS;
}
HcclResult hcclComm::DeinitOneSidedService()
{
    return HCCL_SUCCESS;
}

}  // namespace hccl
 