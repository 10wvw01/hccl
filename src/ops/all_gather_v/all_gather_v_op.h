/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_ALL_GATHER_V_OP
#define OPS_HCCL_SRC_OPS_ALL_GATHER_V_OP

#include <string>
#include <memory>
#include "hccl.h"

#include "alg_param.h"
#include "executor_v2_base.h"
#include "alg_type.h"
#include "execute_selector.h"
#include "alg_param.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts,
    const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream);

HcclResult HcclAllGatherVGraphMode(void *sendBuf, void *recvBuf, uint64_t sendCount,const void *recvCounts,
 	const void *recvDispls,  HcclDataType dataType, const char* group, aclrtStream stream, const char *tag, 
 	void **streams, size_t streamCount, void *scratchMemAddr, uint64_t scratchMemSiz);

#ifdef __cplusplus
}
#endif

namespace ops_hccl {
HcclResult AllGatherVOutPlace(void *sendBuf, void *recvBuf, uint64_t sendCount, const void *recvCounts,
    const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream, const std::string &tag);
HcclResult AllGatherVEntryLog(void *sendBuf, void *recvBuf, uint64_t sendCount, const void *recvCounts, const void *recvDispls,
    HcclDataType dataType, aclrtStream stream, const std::string &tag, const u32 totalRanks, const std::string &opName, bool forceLog = false);

HcclResult AllGatherVOutPlaceGraphMode(void *sendBuf, void *recvBuf, uint64_t sendCount, const void *recvCounts,
 	const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream, const std::string &tag, const ResPackGraphMode &resPack);

HcclResult CheckAllGatherVInputPara(const HcclComm comm, const void *recvCounts, const void *recvDispls,
    const aclrtStream stream, void *sendBuf, uint64_t sendCount);

HcclResult CheckAllGatherVRecvAndGetRank(const HcclComm comm, const void *recvBuf, const void *recvCounts,
    u32 &rankSize, u32 &userRank, bool &allRecvCountsZero);

// 计算 AllGatherV 输出缓冲需要覆盖的字节跨度
// recvCounts 表示各 rank 接收元素个数数组
// recvDispls 表示各 rank 接收数据在输出缓冲中的元素偏移数组
// rankSize 表示通信域内 rank 数量
// perDataSize 表示单个数据元素的字节数
// outputSize 表示计算得到的输出字节跨度
// 返回值表示计算是否成功，溢出时返回参数错误
HcclResult CalcAllGatherVOutputSize(const void *recvCounts, const void *recvDispls, u32 rankSize, u32 perDataSize,
    u64 &outputSize);

HcclResult AllGatherVExecOp(HcclComm comm, OpParam &param);

HcclResult CheckCountAGV(const u64 count);

HcclResult CheckDataTypeAGV(const HcclDataType dataType);

std::string GetSupportDataTypeAGV();

HcclResult CalcBaseTopoInfoAllGatherV(HcclComm comm, OpParam &param, TopoInfoWithNetLayerDetails **topoInfo);

}  // namespace ops_hccl
#endif
