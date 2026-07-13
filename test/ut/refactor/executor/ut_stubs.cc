/**
 * ops_executor UT 桩函数
 */
#include <map>
#include <memory>
#include <vector>
#include <iostream>
#include <cstring>
#include <hccl/hccl_types.h>
<<<<<<< Updated upstream
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
=======
#include "hccl_algorithm.h"
#include "alg_param.h"
#include "base_template.h"
>>>>>>> Stashed changes

// src/common/log.cc
bool IsErrorToWarn() { return false; }
bool HcclCheckLogLevel(int, int) { return false; }

namespace ops_hccl {

// refactor/ops/utils/utils.h
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
{ return HCCL_SUCCESS; }

// ============================================================
// Mock 模板: 不同算法类型返回不同的资源值
// ============================================================

// Mesh 模板 (FULLMESH): slaveThreadNum=3, notifyOnMain=2, notifyPerThread={3,3,3}
class MockMeshTemplate : public BaseTemplate {
public:
    using BaseTemplate::BaseTemplate;
    HcclResult CalcRes(HcclComm comm, AlgResourceRequest &res) override
    {
        res.slaveThreadNum = 3;
        res.notifyNumOnMainThread = 2;
        res.notifyNumPerThread = {3, 3, 3};
        res.channels = {{}};
        std::cout << "[MockMesh] slaveThread=" << res.slaveThreadNum
                  << " notifyOnMain=" << res.notifyNumOnMainThread
                  << " notifyPerThreadSz=" << res.notifyNumPerThread.size() << std::endl;
        return HCCL_SUCCESS;
    }
};

// NHR 模板: slaveThreadNum=2, notifyOnMain=1, notifyPerThread={2,2}
class MockNhrTemplate : public BaseTemplate {
public:
    using BaseTemplate::BaseTemplate;
    HcclResult CalcRes(HcclComm comm, AlgResourceRequest &res) override
    {
        res.slaveThreadNum = 2;
        res.notifyNumOnMainThread = 1;
        res.notifyNumPerThread = {2, 2};
        res.channels = {{}};
        std::cout << "[MockNHR]  slaveThread=" << res.slaveThreadNum
                  << " notifyOnMain=" << res.notifyNumOnMainThread
                  << " notifyPerThreadSz=" << res.notifyNumPerThread.size() << std::endl;
        return HCCL_SUCCESS;
    }
};

// GetTemplate: 根据 algType 返回不同 mock
std::unique_ptr<BaseTemplate> GetTemplate(
    HcclAlgEngineType, const TemplateDesc &tmpl, const std::vector<u32> &ranks, u32 myRank)
{
    const char *type = (tmpl.algType == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH) ? "FULLMESH" : "NHR";
    std::cout << "[GetTemplate] algType=" << type
              << " ranksSz=" << ranks.size() << " myRank=" << myRank << std::endl;
    if (tmpl.algType == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH) {
        return std::make_unique<MockMeshTemplate>(myRank, ranks, HcclAlgEngineType::AICPU, tmpl);
    }
    return std::make_unique<MockNhrTemplate>(myRank, ranks, HcclAlgEngineType::AICPU, tmpl);
}

} // namespace ops_hccl
