/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_H_
#define HCCL_H_

#include <hccl/hccl_types.h>
#include <hccl/hccl_comm.h>
#include <acl/acl.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief OmniRun custom collective communication operator.
 *
 * @param sendBuf A pointer identifying the input data address of the operator.
 * @param recvBuf A pointer identifying the output data address of the operator.
 * @param sendType Datatype of send buffer elements, must be one of the following types: int8, int16, int32, int64,
 * uint8, uint16, uint32, uint64, float16, float32, float64, bfp16.
 * @param recvType Datatype of receive buffer elements, must be one of the following types: int8, int16, int32, int64,
 * uint8, uint16, uint32, uint64, float16, float32, float64, bfp16.
 * @param xmlPath A pointer identifying the XML configuration path for algorithm selection.
 * @param opParam A pointer identifying the operator parameter data for custom communication configuration.
 * The opParam blob layout (4104 bytes, little-endian):
 *   byte 0: op_name(4b low) + reduce_op(4b high)
 *   bytes 1-3: reserved
 *   bytes 4-7: data_count (uint32)
 *   bytes 8-1031: send_counts (int64[128])
 *   bytes 1032-2055: recv_counts (int64[128])
 *   bytes 2056-3079: sdispls (int64[128])
 *   bytes 3080-4103: rdispls (int64[128])
 * @param opParamSize Size of the operator parameter data in bytes (must be >= 4104).
 * @param comm A pointer identifying the communication resource based on.
 * @param stream A pointer identifying the stream information.
 * @return HcclResult
 */
extern HcclResult HcclOmniRun(const void *sendBuf, const void *recvBuf,
                         HcclDataType sendType, HcclDataType recvType,
                         const char *xmlPath, const void *opParam, uint64_t opParamSize,
                         HcclComm comm, aclrtStream stream);


#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_OPS_H