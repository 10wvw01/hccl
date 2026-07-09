/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_OMNIPIPE_MESH_1D_H
#define INS_TEMP_ALL_GATHER_OMNIPIPE_MESH_1D_H

#include "ins_temp_all_gather_mesh_1D.h"
namespace ops_hccl {

class InsTempAllGatherOmniPipeMesh1D : public InsTempAllGatherMesh1D {
public:
    explicit InsTempAllGatherOmniPipeMesh1D(const OpParam& param, const u32 rankId,  // 传通信域的rankId，userRank
                                            const std::vector<std::vector<u32>>& subCommRanks);
    // Host侧调用
    ~InsTempAllGatherOmniPipeMesh1D() override;

    std::string Describe() const override
    {
        std::string info = "Template of all gather mesh (omniPipe) with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }
    HcclResult KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
                         TemplateResource& templateResource) override;

    // 计算 OmniPipe last-step read 访问用户输出缓冲时使用的字节偏移
    // baseOff 表示当前分片在缓冲区内的基础偏移
    // stepStride 表示当前 rank 在本 step 内的起始偏移
    // processedDataCount 表示前序 loop 已处理的数据个数
    // dataTypeSize 表示单个数据元素的字节数
    // 返回值表示叠加 step 偏移和已处理数据量后的字节偏移
    static u64 CalcOmniLastStepReadOffset(u64 baseOff, u64 stepStride, u64 processedDataCount, u32 dataTypeSize);

private:
    HcclResult RunAllGatherMesh(const std::vector<ThreadHandle>& threads,
                                const std::map<u32, std::vector<ChannelInfo>>& channels) override;
    bool omniLastStepRead_ = false;
};

}  // namespace ops_hccl

#endif  // INS_TEMP_ALL_GATHER_OMNIPIPE_MESH_1D_H
