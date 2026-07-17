/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_REDUCE_SCATTER_OP
#define OPS_HCCL_SRC_OPS_REDUCE_SCATTER_OP

#include <string>
#include "hccl.h"
#include "op_common/inc/alg_param.h"
#include "alg_type.h"
#include "hccl_algorithm.h"
#include "executor_v2_base.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                             HcclReduceOp op, HcclComm comm, aclrtStream stream);
HcclResult HcclReduceScatterGraphMode(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                                      HcclReduceOp op, const char* group, aclrtStream stream, const char *tag,
                                      void **streams, size_t streamCount, void *scratchMemAddr, uint64_t scratchMemSize);
#ifdef __cplusplus
}
#endif

namespace ops_hccl {
HcclResult ReduceScatterOutPlace(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                                 HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag);
HcclResult ReduceScatterOutPlaceGraphMode(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                                          HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag,
                                          const ResPackGraphMode &resPack);
HcclResult ReduceScatterOutPlaceCommon(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                                       HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag,
                                       OpMode opMode, const ResPackGraphMode &resPack);

HcclResult CheckReduceScatterInputPara(const HcclComm comm, const void* sendBuf, const void* recvBuf,
                                       const aclrtStream stream);

HcclResult ReduceScatterInitAndCheck(HcclComm comm, void *sendBuf, void *recvBuf, uint64_t recvCount,
                                     HcclDataType dataType, HcclReduceOp op, aclrtStream stream, std::string &opTag);
HcclResult ReduceScatterEntryLog(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                                 HcclReduceOp op, aclrtStream stream, const std::string &tag,
                                 const std::string &opName);

}
#endif
