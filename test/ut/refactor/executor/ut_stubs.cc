/**
 * ops_executor UT 桩函数
 */
#include <vector>
#include <iostream>
#include <cstring>
#include <hccl/hccl_types.h>
#include <hccl/hccl_rank_graph.h>
#include "hccl_algorithm.h"

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
bool HcclCheckLogLevel(int, int) { return false; }

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

// ============================================================
// TopoMatchBase 桩 (替代 topo_match_base.cc)
// ============================================================
TopoMatchBase::TopoMatchBase() = default;
TopoMatchBase::~TopoMatchBase() = default;
HcclResult TopoMatchBase::MatchTopo(const HcclComm, TopoInfoWithNetLayerDetails *,
                                    AlgHierarchyInfoForAllLevel &) { return HCCL_SUCCESS; }

// ============================================================
// g_allGatherTemplateDescMap 桩 (替代 all_gather_template_desc.cc)
// ============================================================
TemplateDesc g_allGatherTemplateDescMap[] = {
    // ALLGATHER_TEMPLATE_NHR_SINGLE_JETTY
    {HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_NHR,
     HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY},
    // ALLGATHER_TEMPLATE_FULLMESH_SINGLE_JETTY
    {HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH,
     HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY},
    // ALLGATHER_TEMPLATE_FULLMESH_MULTIPLE_JETTY
    {HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH,
     HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::MULTIPLE_JETTY},
    // ALLGATHER_TEMPLATE_NHR_MULTIPLE_JETTY
    {HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_NHR,
     HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::MULTIPLE_JETTY},
};

} // namespace ops_hccl
