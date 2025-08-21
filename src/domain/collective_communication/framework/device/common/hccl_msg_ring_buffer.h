/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCL_MSG_RING_BUFFER_H
#define HCCL_MSG_RING_BUFFER_H

#include "aicpu_hccl_def.h"
class CommonHcclMsgRingBuffer {
public:
    static constexpr uint8_t DEFAULT_CAPACITY = 4;

    CommonHcclMsgRingBuffer() : CommonHcclMsgRingBuffer(DEFAULT_CAPACITY)
    {}

    CommonHcclMsgRingBuffer(uint8_t capacity) : capacity_(capacity)
    {
        if (capacity > 0) {
            buffer_ = new CommonHcclMsg[capacity_];
        }
    }

    ~CommonHcclMsgRingBuffer()
    {
        if (capacity_ > 0 && buffer_ != nullptr) {
            delete[] buffer_;
            buffer_ = nullptr;
            capacity_ = 0;
        }
    }

    bool Enqueue(const CommonHcclMsg *msg)
    {
        if (size_ >= capacity_ || capacity_ == 0) {
            HCCL_ERROR("CommonHcclMsgRingBuffer queue is full.");
            return false;
        }
        s32 sRet = memcpy_s(&buffer_[tail_], sizeof(CommonHcclMsg), msg, sizeof(CommonHcclMsg));
        if (sRet != EOK) {
            HCCL_ERROR("memcpy_s failed, errorno[%d]", sRet);
            return false;
        }
        tail_ = (tail_ + 1) % capacity_;
        size_++;
        HCCL_DEBUG("CommonHcclMsgRingBuffer: enqueue msg success, head_ %u, tail_ %u", head_, tail_);
        return true;
    }

    bool Peek(CommonHcclMsg *msg)
    {
        if (head_ == tail_) {
            return false;
        }
        HCCL_DEBUG("CommonHcclMsgRingBuffer: peek queue first msg success, head %u, tail %u", head_, tail_);
        s32 sRet = memcpy_s(msg, sizeof(CommonHcclMsg), &buffer_[head_], sizeof(CommonHcclMsg));
        if (sRet != EOK) {
            HCCL_ERROR("memcpy_s failed, errorno[%d]", sRet);
            return false;
        }
        return true;
    }

    bool Dequeue()
    {
        if (head_ == tail_ || capacity_ == 0) {
            HCCL_ERROR("CommonHcclMsgRingBuffer queue is empty.");
            return false;
        }
        size_--;
        head_ = (head_ + 1) % capacity_;
        HCCL_DEBUG("CommonHcclMsgRingBuffer: dequeue msg success, head_ %u, tail_ %u", head_, tail_);
        return true;
    }

    void Clear()
    {
        head_ = 0;
        tail_ = 0;
    }

private:
    uint8_t capacity_{0};
    uint32_t head_{0};
    uint32_t tail_{0};
    uint32_t size_{0};
    CommonHcclMsg *buffer_{nullptr};
};

#endif  // HCCL_MSG_RING_BUFFER_H