/**
 * ops_executor UT 桩函数
 */
#include <vector>
#include <iostream>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <hccl/hccl_types.h>
#include <hccl/hccl_rank_graph.h>
#include "hccl_algorithm.h"
#include "base_template.h"

extern "C" errno_t memset_s(void *dest, size_t destMax, int c, size_t count)
{
    if (dest == nullptr || count > destMax) {
        return 1;
    }
    (void)memset(dest, c, count);
    return 0;
}

extern "C" HcclResult HcclRankGraphGetLinks(HcclComm, uint32_t, uint32_t, uint32_t, CommLink **links, uint32_t *linkNum)
{
    static CommLink link;
    (void)CommLinkInit(&link, 1);
    *links = &link;
    *linkNum = 1;
    return HCCL_SUCCESS;
}

// src/common/log.cc — 全局作用域
bool IsErrorToWarn()
{
    return false;
}
// DLOG_DEBUG=0, DLOG_INFO=1, DLOG_WARN=2, DLOG_ERROR=3
// 返回 true 打开对应级别日志；这里 INFO 及以上都打开
bool HcclCheckLogLevel(int logType, int)
{
    return logType >= 3;
}

// dlog_pub.h 声明的 weak 符号，UT 没链接真实实现，这里提供桩：vfprintf 输出到 stderr
extern "C" void DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...)
{
    (void)moduleId;
    fprintf(stderr, "[HCCL][lvl=%d] ", level);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

// refactor/ops/utils/utils.h — namespace ops_hccl
namespace ops_hccl {

HcclResult PreSyncInterThreads(const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads,
    const std::vector<u32> &notifyIdxMainToSub)
{
    HCCL_INFO("PreSyncInterThreads mainThread: %d", mainThread);
    for (size_t i = 0; i < subThreads.size(); ++i) {
        HCCL_INFO("subThread[%d] = %d", i, subThreads[i]);
    }
    return HCCL_SUCCESS;
}

HcclResult PostSyncInterThreads(const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads,
    const std::vector<u32> &notifyIdxSubToMain)
{
    HCCL_INFO("PostSyncInterThreads mainThread: %d", mainThread);
    for (size_t i = 0; i < subThreads.size(); ++i) {
        HCCL_INFO("subThread[%d] = %d", i, subThreads[i]);
    }
    return HCCL_SUCCESS;
}

// ============================================================
// BaseTemplate 桩 (替代 base_template.cc，提供 vtable)
// ============================================================
BaseTemplate::~BaseTemplate() {}

// ============================================================
// TopoMatchBase 桩 (替代 topo_match_base.cc)
// ============================================================
TopoMatchBase::TopoMatchBase() = default;
TopoMatchBase::~TopoMatchBase() = default;
HcclResult TopoMatchBase::MatchTopo(const HcclComm, TopoInfoWithNetLayerDetails *, AlgHierarchyInfoForAllLevel &)
{
    return HCCL_SUCCESS;
}

// ============================================================
// g_allGatherTemplateDescMap 桩 (替代 all_gather_template_desc.cc)
// ============================================================
TemplateDesc g_allGatherTemplateDescMap[] = {
    // ALLGATHER_TEMPLATE_NHR_SINGLE_JETTY
    {HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_NHR, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::SINGLE_JETTY},
    // ALLGATHER_TEMPLATE_FULLMESH_SINGLE_JETTY
    {HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::SINGLE_JETTY},
    // ALLGATHER_TEMPLATE_FULLMESH_MULTIPLE_JETTY
    {HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH, HcclAlgShotMode::ONE_SHOT,
        HcclAlgJettyMode::MULTIPLE_JETTY},
    // ALLGATHER_TEMPLATE_NHR_MULTIPLE_JETTY
    {HCCL_CMD_ALLGATHER, HcclAlgoType::HCCL_ALGO_TYPE_NHR, HcclAlgShotMode::ONE_SHOT, HcclAlgJettyMode::MULTIPLE_JETTY},
};

} // namespace ops_hccl
