#include "base_executor.h"

BaseExecutor::BaseExecutor(HcclAlgorithm &algo, OpParam &param)
    : algo_(algo), myRank_(param.myRank), rankSize_(param.rankSize), root_(param.root)
{
    dataInfo_.inputPtr = param.inputPtr;
    dataInfo_.inputSize = param.inputSize;
    dataInfo_.outputPtr = param.outputPtr;
    dataInfo_.outputSize = param.outputSize;
    dataInfo_.reduceOp = param.reduceOp;
    // TODO：把param中union的结构体复制到dataInfo中

    dataTypeSize_ = DATATYPE_SIZE_TABLE[baseOpParam.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
}

BaseExecutor::~BaseExecutor() {}

HcclResult BaseExecutor::CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo)
{
    // 储存通信域指针
    hcclComm_ = comm;
    // TODO：topoMatch暂不修改参数
    algo_.topoMatch.MatchTopo(hcclComm_, topoInfo, algHierarchyInfo_);
    return HCCL_SUCCESS;
}

HcclResult BaseExecutor::CalcRes(AlgResourceRequest &resReq)
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

    // 合并每个实例的template资源，不同的Executer合并方式不同，Merge包含thread/Notify/Channel
    MergeResReq(resReqList, resReq);
}

HcclResult BaseExecutor::Orchestrate(const BaseExecutorParam &baseExecutorParam,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceCtxSerializable &resCtx)
{
    // 初始化资源信息
    InitRes(resCtx);
    // 切分数据阶段（子类实现GetMaxProCntPerLoop函数）
    // maxProcessCount表示每次循环能处理的数据量，该数据量定义与入参dataCount保持一致（不同op有区别）
    GetMaxProcCntPerLoop(dataCount_, maxProcCntPerLoop);
    // 循环下发阶段（按照每轮最大处理数据量，循环展开）
    u64 loopTimes = RoundUp(dataCount_, maxProcCntPerLoop);
    u64 processCount = maxProCntPerLoop;
    u64 offsetCount = 0;
    for (u64 loopIdx = 0; loopIdx < loopTimes; ++loopIdx) {
        if (dataCount_ % maxProCntPerLoop != 0) {
            processCount = dataCount_ % maxProcCntPerLoop;
        }
        // 子类实现
        OrchestrateLoop(processCount, offsetCount);
        // 偏移增加
        offsetCount += processCount;
    }
    // TODO：储存队列和任务信息，用于FastLauch
    SaveCtx();
}

// 公共工具类函数

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

HcclResult BaseExecutor::SplitRes()
{
    // 需要切分的资源
    // algHierarchyInfo：根据TemplateExecDesc.subCommIndex切分
    // Thread: slaveThreadNum：需要算法提供GetRes
    // Notify: notifyNumOnMainThread, notifyNumPerThread：需要算法提供GetRes
    // Channel: 当前直接按照level切分，后续根据TemplateExecDesc.subCommIndex切分

    // 从map表里获取资源，相当于GetRes
    // map需要提供：节点数量和ID（ranks），thread数量和ID，Notify数量和ID，Channel数量和ID
}
