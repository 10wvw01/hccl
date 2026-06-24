/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_MESH_1D_OCS_H
#define INS_TEMP_REDUCE_SCATTER_MESH_1D_OCS_H

#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"
#include "ins_temp_reduce_scatter_mesh_1D.h"

namespace ops_hccl {

// 4层OCS拓扑最高层(net_layer_3, SuperNode级)的 ReduceScatter Mesh1D 模板。
// 与父类 InsTempReduceScatterMesh1D 的区别: CalcRes 时仅在指定的 net_layer(默认 layer3)
// 上建立 channel，避免遍历所有层误取低层(layer0/1/2)链路。
class InsTempReduceScatterMesh1DOcs : public InsTempReduceScatterMesh1D {
public:
    InsTempReduceScatterMesh1DOcs() = default;
    explicit InsTempReduceScatterMesh1DOcs(const OpParam& param, const u32 rankId,
                                           const std::vector<std::vector<u32>> &subCommRanks);

    ~InsTempReduceScatterMesh1DOcs() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter Mesh 1D OCS(layer3) with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;

private:
    // OCS 层物理 net_layer 编号，4层拓扑下为 3
    u32 ocsNetLayer_{3};
};

} // namespace ops_hccl

#endif // INS_TEMP_REDUCE_SCATTER_MESH_1D_OCS_H
