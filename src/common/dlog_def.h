/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DLOG_DEF_H
#define DLOG_DEF_H

#include <stdint.h>
#include <base/log_types.h>

// log level id
#define DLOG_DEBUG 0x0      // debug level id
#define DLOG_INFO  0x1      // info level id
#define DLOG_WARN  0x2      // warning level id
#define DLOG_ERROR 0x3      // error level id
#define DLOG_NULL  0x4      // don't print log

#define RUN_LOG_MASK        (0x01000000U)    // print log to directory run

#ifdef __cplusplus
extern "C" {
#endif
void DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...) __attribute__((weak));
#ifdef __cplusplus
}
#endif

#endif // DLOG_DEF_H