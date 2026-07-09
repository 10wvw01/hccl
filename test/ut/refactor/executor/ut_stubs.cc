/**
 * ops_executor UT 桩函数
 */
#include <vector>
#include <hccl/hccl_types.h>

// src/common/log.cc — 全局作用域
bool IsErrorToWarn() { return false; }
int HcclCheckLogLevel(int, int) { return 0; }

// refactor/ops/utils/utils.h — namespace ops_hccl
namespace ops_hccl {

HcclResult PreSyncInterThreads(const unsigned long &,
                               const std::vector<unsigned long> &,
                               const std::vector<unsigned int> &)
{
    return HCCL_SUCCESS;
}

HcclResult PostSyncInterThreads(const unsigned long &,
                                const std::vector<unsigned long> &,
                                const std::vector<unsigned int> &)
{
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
