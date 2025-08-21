/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALL_REDUCE_RING_ZEROCOPY_EXECUTOR_H
#define COLL_ALL_REDUCE_RING_ZEROCOPY_EXECUTOR_H

#include "coll_all_reduce_executor.h"

namespace hccl {
class CollAllReduceRingZerocopyExecutor : public CollAllReduceExecutor {

public:
    CollAllReduceRingZerocopyExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAllReduceRingZerocopyExecutor() = default;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(TransportMemType inputType,
        TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType);

    /* *************** 算法编排 *************** */
    HcclResult DoubleRingReduceScatter(const std::string &tag,
        DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream,
        s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
        const std::vector<std::vector<Slice>> multRingsUserMemSlice);
    HcclResult DoubleRingAllGather(
        const std::string &tag, DeviceMem inputMem, DeviceMem outputMem,
        const u64 count, const HcclDataType dataType, const std::vector<std::vector<Slice> > multRingsSliceZero,
        Stream stream, s32 profStage, const u64 baseOffset, HcomCollOpInfo *opInfo,
        const std::vector<std::vector<Slice>> multRingsUserMemSlice);
    HcclResult KernelRunInterServerAllReduceSingleSuperpod(const OpParam &param, const ExecMem &execMem, const u64 level1DataSize);
    HcclResult KernelRunInterServerAllReduceMultiSuperpod(const OpParam &param, const ExecMem &execMem, const u64 level1DataSize);
    HcclResult KernelRunIntraServerPre(const OpParam &param, ExecMem &execMem) override;
    HcclResult KernelRunInterServer(const OpParam &param, ExecMem &execMem) override;
    HcclResult KernelRunIntraServerPost(const OpParam &param, ExecMem &execMem) override;

    u32 level0Rank_ = INVALID_VALUE_RANKID;
    u32 level1Rank_ = INVALID_VALUE_RANKID;
    u32 level2Rank_ = INVALID_VALUE_RANKID;
    u32 level0RankSize_ = INVALID_VALUE_RANKSIZE;
    u32 level1RankSize_ = INVALID_VALUE_RANKSIZE;
    u32 level2RankSize_ = INVALID_VALUE_RANKSIZE;
    std::vector<std::vector<Slice>> level0MultiRingDataSlices_;
};

} // namespace hccl

#endif