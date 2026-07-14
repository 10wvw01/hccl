#ifndef OPS_HCCL_TEMPLATE_FACTORY_H
#define OPS_HCCL_TEMPLATE_FACTORY_H

#include <memory>
#include "base_template.h"

namespace ops_hccl {

// 使用智能指针返回，避免抽象类按值返回
inline std::unique_ptr<BaseTemplate> GetTemplate(const TemplateDesc &templateDesc,
    const std::vector<u32> &ranks, u32 myRank)
{
    return std::make_unique<BaseTemplate>(myRank, ranks, templateDesc);
}

} // namespace ops_hccl

#endif // OPS_HCCL_TEMPLATE_FACTORY_H
