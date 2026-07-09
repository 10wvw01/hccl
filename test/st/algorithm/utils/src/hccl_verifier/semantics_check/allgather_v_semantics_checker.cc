/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_v_semantics_checker.h"

#include <map>
#include "log.h"

namespace HcclSim {
HcclResult TaskCheckAllGatherVSemantics(std::map<RankId, RankMemorySemantics> &allRankMemSemantics, VDataDesTag &vDataDes)
{
    u32 rankSize = allRankMemSemantics.size();

    u64 totalDataSize = 0;
    // AllGatherV 输入不等长
    for (u32 i = 0; i < rankSize; i++) {
        u64 curCounts = vDataDes.counts[i];
        u64 curLength = curCounts * SIZE_TABLE[vDataDes.dataType];
        totalDataSize += curLength;
    }

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        // 对应的rank不存在需要报错
        if (allRankMemSemantics.count(rankId) == 0) {
            HCCL_ERROR("Missing rank %d mem semantics", rankId);
            return HcclResult::HCCL_E_PARA;
        }

        u64    checkedDataSize = 0;
        RankId curRankId      = 0;
        u64    curDataSize    = 0;
        for (auto &ele : allRankMemSemantics[rankId][BufferType::OUTPUT]) {
            while (curRankId < rankSize && vDataDes.counts[curRankId] == 0) {
                curRankId++;
            }
            if (curRankId >= rankSize) {
                HCCL_ERROR("[rankId:%u]Unexpected extra buffer semantic: cur buffer semantic is %s",
                    rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
            u64 inputSize = vDataDes.counts[curRankId] * SIZE_TABLE[vDataDes.dataType];
            u64 expectedStartAddr = vDataDes.displs[curRankId] * SIZE_TABLE[vDataDes.dataType] + curDataSize;

            if (ele.startAddr != expectedStartAddr) {
                HCCL_ERROR("[rankId:%u]Missing buffer semantic: "
                "expected startAddr is %llu, while cur buffer semantic startAddr is %llu, cur buffer semantic is %s",
                    rankId, expectedStartAddr, ele.startAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_ERROR("[rankId:%u]Cur buffer semantic should not be reduce, which mean srcBufs size should be 1, "
                    "while cur buffer semantic is %s", rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->rankId != curRankId) {
                HCCL_ERROR("[rankId:%u]Cur buffer semantic should come from rank %u, while it come from rank %u, "
                    "cur buffer semantic is %s", rankId, curRankId, ele.srcBufs.begin()->rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->bufType != BufferType::INPUT) {
                HCCL_ERROR("[rankId:%u]Cur buffer semantic srcBufs bufType is not INPUT, cur buffer semantic is %s",
                    rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.begin()->srcAddr != curDataSize) {
                HCCL_ERROR("[rankId:%u]Cur buffer semantic srcBufs srcAddr should be %llu, while it is %llu, cur buffer semantic is %s",
                    rankId, curDataSize, ele.srcBufs.begin()->srcAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;

            if (curDataSize == inputSize) {
                curDataSize = 0;
                curRankId++;
            } else if (curDataSize > inputSize) {
                HCCL_ERROR("[rankId:%u]Accumulated semantic size from rank %u is %llu, greater than expected %llu",
                    rankId, curRankId, curDataSize, inputSize);
                return HcclResult::HCCL_E_PARA;
            }

            checkedDataSize += ele.size;
        }
        if (checkedDataSize != totalDataSize) {
            HCCL_ERROR("[rankId:%u]Missing buffer semantics in tail: already checked total size is %llu, "
                "while totalDataSize is %llu, rankSize is %u",
                rankId, checkedDataSize, totalDataSize, rankSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace checker
