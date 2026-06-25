#include "base_executor.h"

BaseExecutor::BaseExecutor(HcclAlgorithm &algo)
    : algo_(algo) {}

BaseExecutor::BaseExecutor(HcclAlgorithm &algo, BaseOpParam &param)
    : algo_(algo), myRank_(param.myRank), rankSize_(param.rankSize),
      dataType_(param.dataType), dataCount_(param.dataCount), reduceOp_(param.reduceOp), root_(param.root)
{
    dataTypeSize_ = DATATYPE_SIZE_TABLE[baseOpParam.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
}

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
    auto allTemplates = GenAllTemplates(algHierarchyInfo);
    for (auto i = 0; i < allTemplates.size(); ++i) {
        for (auto j = 0; j < allTemplates.at(i).size(); ++j) {
            allTemplates.at(i).at(j).Init();
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CalcRes(const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resReq)
{
    // TODO：子类自行实现，以下只是示例，或者只实现MergeResReq
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
    // TODO：子类自行实现，以下只是示例
    // 初始化资源
    InitRes();
    // 切分资源给每个Template
    SplitRes();
    // 切分数据循环
    SplitDataLoop();
    // 循环展开
    OrchestrateLoop();
    // TODO：储存队列和任务信息，用于FastLauch
    SaveCtx();
}

// 公共工具类函数

std::vector<std::vector<std::shared_ptr<BaseTemplate>>> GenAllTemplates(const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 如果用多维数组，基类提供一个生成所有template的函数
    std::vector<std::vector<std::shared_ptr<BaseTemplate>>> allTemplates;
    for (auto i = 0; i < algo_.templateDescs.size(); ++i) {
        std::vector<std::shared_ptr<BaseTemplate>> temp;
        for (auto j = 0; j < algo_.templateDescs.at(i).size(); ++j) {
            temp.push_back(GenTemplate(algo_.templateDescs.at(i).at(j), algHierarchyInfo.at(i).at(j)));
        }
        allTemplates.push_back(temp);
    }
}

std::shared_ptr<BaseTemplate> GenTemplate(TemplateDesc templateDesc, std::vector<u32> &rankList)
{
    // 根据templateDesc实例化Template
    // auto singleTemplate = map[algo_.engineType][opType][algoType][ShotType][JettyType];
    // return singleTemplate;
}

HcclResult InitRes(const AlgResourceCtxSerializable &resCtx)
{
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;
    mainThread_ = threads_.at(0);
}
