#include "sole_executor.h"

namespace ops_hccl {

SoleExecutor::SoleExecutor(HcclAlgorithm &algo)
    : BaseExecutor(algo) {}

SoleExecutor::SoleExecutor(HcclAlgorithm &algo, BaseOpParam &param)
    : BaseExecutor(algo, param) {}

SoleExecutor::~SoleExecutor() {}

HcclResult SoleExecutor::CalcRes(const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resReq)
{
    auto allTemplates = GenAllTemplates(algHierarchyInfo);
    // TODO：待确认template需要什么参数
    HcclResult res = allTemplates.at(0).at(0).CalcRes(resReq);
    return res;
}

u64 SoleExecutor::GetMaxProCntPerLoop()
{
    // 计算当前cclBuffer每轮最大能处理的数据量（这个数据量不直接切分input/output，跟算子有关，建议直接基于dataCount_计算）
    u64 multiple = algTemplate.CalcScratchMultiple();
    u64 maxProCntPerLoop = RoundUp(dataCount_, multiple);
    return maxProCntPerLoop;
}

HcclResult SoleExecutor::OrchestrateLoop(const u64 maxProCntPerLoop, TemplateDataParams dataParams)
{
    // 按照每轮最大处理数据量，计算每轮处理的数据信息
    TemplateDataParams dataParams;
    GenTemplateDataParams(dataParams);
    
    // 调用算法模板的展开
    algTemplate.KernelRun(dataParams, algRes);
}

} // namespace ops_hccl