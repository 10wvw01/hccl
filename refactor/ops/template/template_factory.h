#ifndef OPS_HCCL_TEMPLATE_FACTORY_H
#define OPS_HCCL_TEMPLATE_FACTORY_H

#include <memory>
#include "base_template.h"
#include "aicpu/allgather_mesh.h"
#include "aicpu/allgather_nhr.h"
#include "aicpu/reducescatter_mesh.h"
#include "aicpu/reducescatter_nhr.h"

namespace ops_hccl {

inline std::unique_ptr<BaseTemplate> GetTemplate(const TemplateDesc &templateDesc,
    const std::vector<u32> &ranks, u32 myRank)
{
    if (templateDesc.hcclCmdType == HcclCMDType::HCCL_CMD_ALLGATHER) {
        if (templateDesc.algType == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH) {
            return std::make_unique<AllGatherMeshTemplate>(myRank, ranks, templateDesc);
        }
        if (templateDesc.algType == HcclAlgoType::HCCL_ALGO_TYPE_NHR) {
            return std::make_unique<AllGatherNhrTemplate>(myRank, ranks, templateDesc);
        }
    }
    if (templateDesc.hcclCmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        if (templateDesc.algType == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH) {
            return std::make_unique<ReduceScatterMeshTemplate>(myRank, ranks, templateDesc);
        }
        if (templateDesc.algType == HcclAlgoType::HCCL_ALGO_TYPE_NHR) {
            return std::make_unique<ReduceScatterNhrTemplate>(myRank, ranks, templateDesc);
        }
    }
    return std::make_unique<BaseTemplate>(myRank, ranks, templateDesc);
}

} // namespace ops_hccl

#endif // OPS_HCCL_TEMPLATE_FACTORY_H
