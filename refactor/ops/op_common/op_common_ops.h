/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_OP_COMMON_OPS_H
#define OPS_HCCL_OP_COMMON_OPS_H

// refactor 侧算子公共聚合头文件。
// 对标 src/ops/op_common/op_common_ops.h，聚合算子入口（如 all_gather_op.cc）所需的全部声明，
// 避免重构目录直接引用 src 目录下的头文件。

#include "adapter_acl.h"
#include "adapter_error_manager_pub.h"
#include "alg_env_config.h"
#include "hccl_common.h"
#include "hccl.h"
#include "hccl/base.h"
#include "hccl_inner.h"
#include "op_common.h"
#include "param_check.h"
#include "sal.h"
#include "hcomm_dlsym.h"
#include "hcom.h"
#include "securec.h"

#endif // OPS_HCCL_OP_COMMON_OPS_H
