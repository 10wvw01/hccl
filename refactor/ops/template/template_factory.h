#ifndef OPS_HCCL_TEMPLATE_FACTORY_H
#define OPS_HCCL_TEMPLATE_FACTORY_H

#include <memory>
#include "base_template.h"
#include "aicpu/allgather_mesh.h"

namespace ops_hccl {

// 根据算法类型创建对应的模板子类。
// 当前仅实现 AllGather Mesh 1D，后续扩展时按 templateDesc 分发到不同子类。
inline std::unique_ptr<BaseTemplate> GetTemplate(const TemplateDesc &templateDesc,
    const std::vector<u32> &ranks, u32 myRank)
{
    if (templateDesc.hcclCmdType == HcclCMDType::HCCL_CMD_ALLGATHER &&
        (templateDesc.algType == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH ||
         templateDesc.algType == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT)) {
        return std::make_unique<AllGatherMeshTemplate>(myRank, ranks, templateDesc);
    }
    // 兜底：返回基类（KernelRun 默认实现为空，不产生数据任务）
    return std::make_unique<BaseTemplate>(myRank, ranks, templateDesc);
}

} // namespace ops_hccl

#endif // OPS_HCCL_TEMPLATE_FACTORY_H
