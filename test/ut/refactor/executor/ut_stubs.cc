/**
 * ops_executor UT 桩函数
 */
#include <vector>
#include <iostream>
#include <cstring>
#include <hccl/hccl_types.h>
#include <hccl/hccl_rank_graph.h>

extern "C" errno_t memset_s(void *dest, size_t destMax, int c, size_t count)
{
    if (dest == nullptr || count > destMax) {
        return 1;
    }
    (void)memset(dest, c, count);
    return 0;
}

extern "C" HcclResult HcclRankGraphGetLinks(HcclComm, uint32_t, uint32_t, uint32_t,
                                            CommLink **links, uint32_t *linkNum)
{
    static CommLink link;
    (void)CommLinkInit(&link, 1);
    *links = &link;
    *linkNum = 1;
    return HCCL_SUCCESS;
}

// src/common/log.cc — 全局作用域
bool IsErrorToWarn() { return false; }
int HcclCheckLogLevel(int, int) { return 0; }

// refactor/ops/utils/utils.h — namespace ops_hccl
namespace ops_hccl {

HcclResult PreSyncInterThreads(const unsigned long &,
                               const std::vector<unsigned long> &,
                               const std::vector<unsigned int> &)
{
    std::cout << "PreSyncInterThreads" << std::endl;    
    return HCCL_SUCCESS;
}

HcclResult PostSyncInterThreads(const unsigned long &,
                                const std::vector<unsigned long> &,
                                const std::vector<unsigned int> &)
{
    std::cout << "PostSyncInterThreads" << std::endl;
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
