#include "base_executor.h"

BaseExecutor::BaseExecutor(HcclAlgorithm &algo)
    : algo_(algo) {}

BaseExecutor::BaseExecutor(HcclAlgorithm &algo, BaseExecutorParam &param)
    : algo_(algo), myRank_(param.myRank), rankSize_(param.rankSize),
      dataType_(param.dataType), dataTypeSize_(param.dataTypeSize), dataCount_(param.dataCount), dataSize_(param.dataSize)
      reduceOp_(param.reduceOp), root_(param.root) {}

BaseExecutor::~BaseExecutor() {}

HcclResult BaseExecutor::CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // TODO：topoMatch暂不修改参数
    algo_.topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo);
    return HCCL_SUCCESS;
}

HcclResult BaseExecutor::Init(AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    
}

HcclResult CalcRes(const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resReq)
{
    // TODO：子类自行实现，或者只实现MergeResReq
    std::vector<AlgResourceRequest> resReqList;
    for (auto i = 0; i < algo_.templates.size(); ++i) {
        // 实例化template
        std::vector<u32> &rankList = algHierarchyInfo.at(i);
        TemplateDesc &templateDesc = algo_.templates.at(i);
        auto singleTemplate = GenTemplate(templateDesc, subRankSize);
        // 计算每个template资源
        AlgResourceRequest resReqTmp;
        singleTemplate.CalcRes(algo_.opType, resReqTmp);
        resReqList.push_back(resReqTmp);
    }

    // 合并每个实例的template资源，不同的Executer合并方式不同
    MergeResReq(resReqList, resReq);
}

std::vector<std::vector<std::shared_ptr<BaseTemplate>>> GenAllTemplates(const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 如果用多维数组，基类提供一个生成所有template的函数
    std::vector<std::vector<std::shared_ptr<BaseTemplate>>> allTemplates;
    for (auto i = 0; i < algo_.templateDescs.size(); ++i) {
        
    }
}

std::shared_ptr<BaseTemplate> GenTemplate(TemplateDesc templateDesc, std::vector<u32> &rankList)
{
    // 根据templateDesc实例化Template
    // auto singleTemplate = map[algo_.engineType][opType][algoType][ShotType][JettyType];
    // return singleTemplate;
}

HcclResult Orchestrate(const BaseExecutorParam &baseExecutorParam, ConfigParam &configParam,
    const BufferParam &bufferParam, const CommInfoList &CommInfoList)
{
    // TODO：子类自行实现
    // 切分资源给每个Template
    SplitRes();
    // 切分数据循环
    SplitDataLoop();
    // 循环展开
    OrchestrateLoop();
    // TODO：储存队列和任务信息，用于FastLauch
    SaveCtx();
}

// private

HcclResult MergeResReq(std::vector<AlgResourceRequest> &resReqList, AlgResourceRequest &resReq)
{
    // 不同Executor整合资源的形式不同，有的取最大，有的加起来
    switch (algo_.executorType)
    {
    case ExecutorType::SOLE:
        // TODO
        break;
    case ExecutorType::PARALLEL:
        // TODO
        break;
    default:
        break;
    }
}

HcclResult SplitRes()
{
    // 不同Executor切分资源给每个template，Merge的逆向操作
    // 取代PrepareResForTemplate函数
    switch (algo_.executorType)
    {
    case ExecutorType::SOLE:
        // TODO
        break;
    case ExecutorType::PARALLEL:
        // TODO
        break;
    default:
        break;
    }
}
