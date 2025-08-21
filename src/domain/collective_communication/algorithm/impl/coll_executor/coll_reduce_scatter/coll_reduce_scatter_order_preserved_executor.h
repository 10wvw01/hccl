/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_REDUCESCATTER_ORDER_PRESERVED_EXECUTOR_H
#define COLL_REDUCESCATTER_ORDER_PRESERVED_EXECUTOR_H
#include "coll_reduce_scatter_executor.h"

namespace hccl {
class CollReduceScatterOrderPreservedExecutor : public CollReduceScatterExecutor {
public:
    explicit CollReduceScatterOrderPreservedExecutor(const HcclDispatcher dispatcher,
        std::unique_ptr<TopoMatcher> &topoMatcher);

private:
    void ParseParam(const OpParam& param) override;
    /* *************** 资源计算 *************** */
    HcclResult CalcScratchMemSize(u64& scratchMemSize) override;
    u32 CalReduceStreamNum(const u32& localRankSize);
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel1CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType);

    /* *************** 算法编排 *************** */
    bool IsSmallData(const u64 totalSize, const u64 curSize) override;
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
    HcclResult RunReduceScatterLevel0(const OpParam &param, ExecMem &execMem, SubCommInfo &level0CommInfo);
    HcclResult RunReduceScatterLevel0HD(const OpParam &param, ExecMem &execMem, SubCommInfo &level0CommInfo);
    HcclResult RunReduceScatterLevel1(const OpParam &param, ExecMem &execMem, SubCommInfo &level0CommInfo);

    u32 all2allOffset_ = 0;
    bool isUseHDAlg_ = false;
};

} // namespace hccl

#endif