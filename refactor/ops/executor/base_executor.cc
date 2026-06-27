#include "base_executor.h"

BaseExecutor::BaseExecutor(HcclAlgorithm &algo, OpParam &param)
    : algo_(algo), myRank_(param.myRank), rankSize_(param.rankSize),
      dataType_(param.dataType), dataCount_(param.dataCount), reduceOp_(param.reduceOp), root_(param.root)
{
    dataTypeSize_ = DATATYPE_SIZE_TABLE[baseOpParam.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    // 提取一个BaseOpParam
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

HcclResult BaseExecutor::CalcRes(const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resReq)
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

HcclResult BaseExecutor::Orchestrate(const BaseExecutorParam &baseExecutorParam,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceCtxSerializable &resCtx)
{
    // 初始化资源信息
    InitRes(resCtx);
    // 切分资源阶段（Sole不需要，跳过）
    // 切分数据阶段（子类实现GetMaxProCntPerLoop函数）
    // maxProcessCount表示每次循环能处理的数据量，该数据量定义与入参dataCount保持一致（不同op有区别）
    GetMaxProCntPerLoop(dataCount_, maxProCntPerLoop);
    // 循环下发阶段（按照每轮最大处理数据量，循环展开）
    u64 loopTimes = RoundUp(dataCount_, maxProCntPerLoop);
    u64 processCount = maxProCntPerLoop;
    u64 tailCount = maxProCntPerLoop;
    if (dataCount_ % maxProCntPerLoop != 0) {
        tailCount = dataCount_ % maxProCntPerLoop;
    }
    for (u64 loopIdx = 0; loopIdx < loopTimes; ++loopIdx) {
        TemplateDataParams dataParams;
        GenTemplateDataParams(loopIdx, maxProCntPerLoop, dataParams);
        // 子类实现
        OrchestrateLoop(maxProCntPerLoop, dataParams);
    }
    // TODO：储存队列和任务信息，用于FastLauch
    SaveCtx();
}

// 公共工具类函数

std::vector<std::vector<std::shared_ptr<BaseTemplate>>> BaseExecutor::GenAllTemplates(
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
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

std::shared_ptr<BaseTemplate> BaseExecutor::GenTemplate(TemplateDesc templateDesc, std::vector<u32> &rankList)
{
    // 根据templateDesc实例化Template
    // auto singleTemplate = map[algo_.engineType][opType][algoType][ShotType][JettyType];
    // return singleTemplate;
}

HcclResult BaseExecutor::InitRes(const AlgResourceCtxSerializable &resCtx)
{
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;
    mainThread_ = threads_.at(0);
    // TODO：考虑不同Executor
    // 需要restore原因，resCtx中储存用双层嵌套vector<vector<ChannelInfo>>，remoteRank信息在ChannelInfo中，查询不方便
    channelTable_ = RestoreChannelMap();

    // TODO：加rankSize数组初始化
}

std::vector<std::map<u32, std::vector<ChannelInfo>> BaseExecutor::RestoreChannelMap(
    const AlgResourceCtxSerializable &resCtx)
{
    // 桥接用函数，理论上直接resCtx直接用该结构表即可
    // 使用原函数，略做改造，直接返回结构表（是否有性能问题？）
}

HcclResult SoleExecutor::GenTemplateDataParams(u64 processCount, TemplateDataParams &dataParams)
{
    // 1.处理数据片大小
    u64 sliceCount = processCount;
    u64 sliceSize = sliceCount * dataTypeSize_;

    // 2.计算输入Buffer偏移和参数
    void* inBufferPtr;
    BufferType inBufferType;
    u64 inBufferOffset;
    // u64 inBufferStride;

    // 3.计算输出Buffer偏移和参数
    void* outBufferPtr;
    BufferType outBufferType;
    u64 outBufferOffset;
    // u64 outBufferStride;

    // 4.计算cclBuffer偏移和参数
    void* cclBufferPtr;
    BufferType cclBufferType;
    u64 cclBufferOffset;

    // 5.计算其他参数
    // u64 repeatNum;

    // TODO：
    // root/dataType放Template构造里传入
    // enableRemoteMemAccess，区分单算子还是图模式，放Template构造里传入
    // 带V算子的参数传入
}
