/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DFX_H
#define HCCL_DFX_H

#include <stdint.h>

#ifndef CHANNEL_HANDLE_DEFINED
#define CHANNEL_HANDLE_DEFINED
typedef uint64_t ChannelHandle;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Dump key context information of HCOMM channels for exception diagnosis.
 *
 * The dump content is provided by the HCOMM runtime. When supported by the
 * runtime, it includes registered local/remote buffer addresses, SQ/CQ context,
 * and SQ/CQ VA content. The interface is intended to be called from an ACL
 * exception callback after extracting channel handles from the exception context.
 *
 * @param channelNum Number of channel handles to dump.
 * @param channels Channel handle array.
 */
void HcommChannelInfoDump(uint32_t channelNum, ChannelHandle *channels);

#ifdef __cplusplus
}
#endif

#endif // HCCL_DFX_H
