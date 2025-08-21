/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALG_CMD_TYPE_H
#define ALG_CMD_TYPE_H

#include <map>

enum class HcclCMDType {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_GATHER,
    HCCL_CMD_SCATTER,
    HCCL_CMD_BATCH_SEND_RECV,
    HCCL_CMD_BATCH_PUT,
    HCCL_CMD_BATCH_GET,
    HCCL_CMD_ALLGATHER_V,
    HCCL_CMD_REDUCE_SCATTER_V,
    HCCL_CMD_BATCH_WRITE,
    HCCL_CMD_ALL,
    HCCL_CMD_MAX
};

const std::map<HcclCMDType, std::string> HCOM_CMD_TYPE_STR_MAP{
    {HcclCMDType::HCCL_CMD_INVALID, "invalid"},
    {HcclCMDType::HCCL_CMD_BROADCAST, "broadcast"},
    {HcclCMDType::HCCL_CMD_ALLREDUCE, "allreduce"},
    {HcclCMDType::HCCL_CMD_REDUCE, "reduce"},
    {HcclCMDType::HCCL_CMD_SEND, "send"},
    {HcclCMDType::HCCL_CMD_RECEIVE, "receive"},
    {HcclCMDType::HCCL_CMD_ALLGATHER, "allgather"},
    {HcclCMDType::HCCL_CMD_ALLGATHER_V, "allgather_v"},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "reduce_scatter"},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, "reduce_scatter_v"},
    {HcclCMDType::HCCL_CMD_ALLTOALLV, "alltoallv"},
    {HcclCMDType::HCCL_CMD_ALLTOALLVC, "alltoallvc"},
    {HcclCMDType::HCCL_CMD_ALLTOALL, "alltoall"},
    {HcclCMDType::HCCL_CMD_GATHER, "gather"},
    {HcclCMDType::HCCL_CMD_SCATTER, "scatter"},
    {HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, "batch_send_recv"},
    {HcclCMDType::HCCL_CMD_BATCH_WRITE, "batch_write"},
    {HcclCMDType::HCCL_CMD_ALL, "all"},
    {HcclCMDType::HCCL_CMD_MAX, "max"}
};

inline std::string GetCMDTypeEnumStr(HcclCMDType cmdType)
{
    auto iter = HCOM_CMD_TYPE_STR_MAP.find(cmdType);
    if (iter == HCOM_CMD_TYPE_STR_MAP.end()) {
        return "Invalid HcclCMDType";
    } else {
        return iter->second;
    }
}
#endif /* ALG_CMD_TYPE_H */
