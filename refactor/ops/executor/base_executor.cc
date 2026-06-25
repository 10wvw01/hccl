#include "base_executor.h"

BaseExecutor::BaseExecutor(HcclAlgorithm &algo)
    : algo_(algo) {}

BaseExecutor::BaseExecutor(HcclAlgorithm &algo, BaseExecutorParam &param)
    : algo_(algo), myRank_(param.myRank), rankSize_(param.rankSize),
      dataType_(param.dataType), dataTypeSize_(param.dataTypeSize), dataCount_(param.dataCount), dataSize_(param.dataSize)
      reduceOp_(param.reduceOp), root_(param.root) {}

BaseExecutor::~BaseExecutor() {}

// HcclResult BaseExecutor::CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
//     AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgTopoMatch topoMatch)
// {
//     CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
//     return HCCL_SUCCESS;
// }

// TODO：其实叫SplitSubComm更贴切，subComm直接就是每个子通信域的集合

using CommInfo = std::vector<u32>;
using CommInfoList = std::vector<CommInfo>;

HcclResult BaseExecutor::SplitSubComm(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
    AlgTopoMatch topoMatch, CommInfoList &CommInfoList)
{
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, CommInfoList));
    return HCCL_SUCCESS;
}

HcclResult BaseExecutor::Plan(const CommInfoList &commInfoList, std::vector<RankPair> &rankPairList)
{
    // 遍历每个子通信域，计算建链关系
    for (auto i = 0; i < commInfoList.size(); ++i) {
        CalcChannelReq(commInfoList.at(i), algo_.templates.at(i).algType, rankPairList);
    }
    // TODO：这里考虑是直接push_back到rankPairList最后，还是每个计算完merge一把（可能要去重）
}

HcclResult CalcRes(const CommInfoList &CommInfoList, AlgResourceRequest &resReq)
{
    std::vector<AlgResourceRequest> resReqList;
    for (auto i = 0; i < algo_.templates.size(); ++i) {
        // 实例化template
        u32 subRankSize = CommInfoList.at(i).size();
        auto singleTemplate = GenTemplate(subRankSize);
        // 计算每个template资源
        AlgResourceRequest resReqTmp;
        singleTemplate.CalcRes(algo_.opType, resReqTmp);
        resReqList.push_back(resReqTmp);
    }

    // TODO:合并每个实例的template资源，不同的Executer合并方式不同
    MergeResReq(resReqList, resReq);
}

HcclResult Orchestrate(const BaseExecutorParam &baseExecutorParam, ConfigParam &configParam,
    const BufferParam &bufferParam, const CommInfoList &CommInfoList)
{
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

HcclResult CalcChannelReq(const CommInfo &commInfo, const AlgoType algoType, std::vector<RankPair> &rankPairList)
{
    // commInfo里就是一个rank列表
    // 根据commInfo和algoType足以计算建链关系
}

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
