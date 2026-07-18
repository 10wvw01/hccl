/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCH_CONCURRENT
#define TOPO_MATCH_CONCURRENT

#include <sstream>
#include "topo_match_base.h"

namespace ops_hccl {
// Concurrent 拓扑匹配器：layer0 和 layer1 都是 server 内同一组卡（不同视角），
// layer2 是跨 server 同 idx 卡的 NHR 子域。仅支持 3 层拓扑。
class TopoMatchConcurrent : public TopoMatchBase {
public:
    explicit TopoMatchConcurrent();
    ~TopoMatchConcurrent() override;
    std::string Describe() const override
    {
        return "Topo Match for Concurrent Algorithm: layer 0/1 same-server mesh, layer 2 cross-server NHR.";
    }
    HcclResult MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                         AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;
};
}  // namespace ops_hccl
#endif  // !TOPO_MATCH_CONCURRENT
