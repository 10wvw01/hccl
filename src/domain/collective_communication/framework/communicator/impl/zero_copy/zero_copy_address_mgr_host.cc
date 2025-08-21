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

namespace hccl
{
    HcclResult ZeroCopyAddressMgr::InitRingBuffer()
    {
        if (ringBuffer_.ptr() != nullptr)
        {
            return HCCL_SUCCESS;
        }

        ringBuffer_ = DeviceMem::alloc(ZERO_COPY_BUFFER_MAX_MAP_COUNT * sizeof(ZeroCopyRingBufferItem));
        CHK_PRT_RET(ringBuffer_.ptr() == nullptr,
                    HCCL_ERROR("[ZeroCopyAddressMgr][InitRingBuffer] alloc ring buffer failed"), HCCL_E_INTERNAL);
        CHK_RET(hrtMemSet(ringBuffer_.ptr(), ringBuffer_.size(), ringBuffer_.size()));

        ringBufferCtl_ = DeviceMem::alloc(sizeof(u32) + sizeof(u32));
        CHK_PRT_RET(ringBufferCtl_.ptr() == nullptr,
                    HCCL_ERROR("[ZeroCopyAddressMgr][InitRingBuffer] alloc ring buffer ctl failed"), HCCL_E_INTERNAL);
        CHK_RET(hrtMemSet(ringBufferCtl_.ptr(), ringBufferCtl_.size(), ringBufferCtl_.size()));

        devRingBufBase_ = reinterpret_cast<ZeroCopyRingBufferItem *>(ringBuffer_.ptr());
        devRingHead_ = reinterpret_cast<u32 *>(ringBufferCtl_.ptr());
        devRingTail_ = devRingHead_ + 1;

        HCCL_RUN_INFO("[ZeroCopyAddressMgr][InitRingBuffer] ringbuffer[%p] len[%lu] bufferCtl[%p] len[%lu] head[%p] tail[%p]",
                      ringBuffer_.ptr(), ringBuffer_.size(), ringBufferCtl_.ptr(), ringBufferCtl_.size(), devRingHead_, devRingTail_);

        HCCL_INFO("[ZeroCopyAddressMgr][InitRingBuffer] ringbuffer[%p] head[%p] tail[%p]", devRingBufBase_, devRingHead_, devRingTail_);
        return HCCL_SUCCESS;
    }

    HcclResult ZeroCopyAddressMgr::PushOne(ZeroCopyRingBufferItem &item)
    {
        if (!needPushOne)
        {
            HCCL_DEBUG("[ZeroCopyAddressMgr][PushOne] don't need push");
            return HCCL_SUCCESS;
        }

        // 检测RingBuffer是否已经初始化，没有的话就初始化一下
        CHK_RET(InitRingBuffer());

        u32 head = 0;
        CHK_RET(hrtMemSyncCopy(&head, sizeof(head), devRingHead_, sizeof(head), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));
        u32 tail = 0;
        CHK_RET(hrtMemSyncCopy(&tail, sizeof(tail), devRingTail_, sizeof(tail), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

        u32 updateTail = (tail + 1) % ZERO_COPY_BUFFER_MAX_MAP_COUNT;
        CHK_PRT_RET(updateTail == head,
                    HCCL_ERROR("[ZeroCopyAddressMgr][PushOne] ring buffer is full head[%u] tail[%u] capacity[%u]",
                               head, tail, ZERO_COPY_BUFFER_MAX_MAP_COUNT),
                    HCCL_E_INTERNAL);

        HCCL_INFO("[ZeroCopyAddressMgr][PushOne] type[%d] head[%u] tail[%u] updateTail[%u] tailAddr[%p]", item.type, head, tail, updateTail, devRingBufBase_ + tail);
        CHK_RET(hrtMemSyncCopy(devRingBufBase_ + tail, sizeof(ZeroCopyRingBufferItem), &item, sizeof(ZeroCopyRingBufferItem),
                               HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        CHK_RET(hrtMemSyncCopy(devRingTail_, sizeof(updateTail), &updateTail, sizeof(updateTail),
                               HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        return HCCL_SUCCESS;
    }

}