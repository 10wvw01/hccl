/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_GATHER_MESH1DNHR_CONCURRENT_MEM2MEM_H
#define HCCL_CCU_TEMP_ALL_GATHER_MESH1DNHR_CONCURRENT_MEM2MEM_H

#include "utils.h"
#include "ccu_alg_template_base.h"
#include "ccu_temp_all_gather_mesh_1D_mem2mem.h"
#include "ccu_temp_all_gather_nhr_1D_mem2mem.h"

namespace ops_hccl {

// Concurrent template: mesh 路径 + NHR(CLOS) 路径并发执行，数据按带宽比切分。
// subCommRanks_[0] 给 mesh 子 template, subCommRanks_[1] 给 NHR 子 template。
// 线程分配: threads[0] -> mesh 主流, threads[1] -> NHR 主流, threads[2] -> NHR 从流。
// kernel 分配: ccuKernels[0] -> mesh, ccuKernels[1] -> NHR。
class CcuTempAllGatherMesh1DNHRConcurrentMem2Mem : public CcuAlgTemplateBase {
public:
    CcuTempAllGatherMesh1DNHRConcurrentMem2Mem() = default;
    explicit CcuTempAllGatherMesh1DNHRConcurrentMem2Mem(const OpParam &param, const u32 rankId,
                                                        const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempAllGatherMesh1DNHRConcurrentMem2Mem() override = default;

    std::string Describe() const override
    {
        return StringFormat("Template of AllGather ccu mesh1d+nhr1d concurrent mem2mem with tempRankSize [%u].",
                            templateRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest &resourceRequest) const override;
    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
                         TemplateResource &templateResource) override;
    u64 GetThreadNum() const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

private:
    // 按带宽比切分当前 chunk 的 count
    void CalcDataSplit(u64 totalCount, u64 dataTypeSize, u64 &meshCount, u64 &closCount) const;
    // 构造 mesh 子 template 参数
    void GenMeshParams(const TemplateDataParams &src, u64 meshCount, u64 meshSize,
                       TemplateDataParams &dst) const;
    // 构造 NHR 子 template 参数
    void GenNhrParams(const TemplateDataParams &src, u64 closCount, u64 closSize, u64 meshSize,
                      TemplateDataParams &dst) const;

    uint32_t mySubCommRank_ = 0;
};

} // namespace ops_hccl

#endif // HCCL_CCU_TEMP_ALL_GATHER_MESH1DNHR_CONCURRENT_MEM2MEM_H
