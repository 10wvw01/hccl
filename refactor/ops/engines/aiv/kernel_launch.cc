/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// AIV 引擎 kernel 下发复用 AICPU 的实现路径：
//   - HcclLaunchAicpuKernel / RestoreVarData* 系列函数：由 engines/aicpu/kernel_launch.cc 提供
//   - RegisterKernel（AIV 算子二进制注册）：由 src/ops/op_common/template/aiv/hccl_aiv_utils.cc 提供
// 本文件原为 extern "C" HcclLaunchAicpuKernel 旧版实现，与 aicpu/kernel_launch.cc 重复，
// 重构后改为空编译单元以保留目录结构与未来扩展位，不引入重复符号。
