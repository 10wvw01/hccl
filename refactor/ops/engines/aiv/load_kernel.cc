/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// AIV 引擎复用 AICPU 的 kernel 二进制加载实现（g_binKernelHandle、LoadAICPUKernel）。
// 两者加载的二进制路径相同，且 g_binKernelHandle 为全局单例，无需重复定义。
// 真正的实现位于 engines/aicpu/load_kernel.cc，本文件为空以避免符号重复定义。
