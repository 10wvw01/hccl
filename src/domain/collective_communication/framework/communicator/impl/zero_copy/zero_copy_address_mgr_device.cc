/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "zero_copy_address_mgr.h"
#include "adapter_rts_common.h"
 
namespace hccl{
HcclResult ZeroCopyAddressMgr::InitRingBuffer()
{
    return HCCL_SUCCESS;
}
 
HcclResult ZeroCopyAddressMgr::PushOne(ZeroCopyRingBufferItem &item)
{
    HCCL_DEBUG("[ZeroCopyAddressMgr][PushOne] device");
    return HCCL_SUCCESS;
}
}