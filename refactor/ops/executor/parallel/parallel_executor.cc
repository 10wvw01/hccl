#include "parallel_2ops_executor.h"

namespace ops_hccl {

HcclResult ParallelExecutor::PreSyncBySubCommMask(const AlgoExecDesc &execDesc)
{
    auto it = execDescSubCommMask.find(execDesc);
    if (it == execDescSubCommMask.end()) {
        // temlate类型的节点可能不存在execDescSubCommMask
        HCCL_ERROR("[ParallelExecutor] AlgoExecDesc not found in execDescSubCommMask");
        return HCCL_SUCCESS;
    }
    std::vector<ThreadHandle> syncInterThreads;
    std::vector<u32> syncNotifyOnAlgoExec;
    for (int i = 0; i < sizeof(it->second) * CHAR_BIT; i++) {
        if (it->second & (1u << i)) {
            syncInterThreads.emplace_back(subThreads_.at(i).at(0));
            // 每个通信子域维度主线程notify - 1才是需要同步的notify数量
            syncNotifyOnAlgoExec.emplace_back(notifyNumOnSubMainThread_.at(i) - 1);
        }
    }
    return PreSyncInterThreads(mainThread_, syncInterThreads, syncNotifyOnAlgoExec);
}

HcclResult ParallelExecutor::PostSyncBySubCommMask(const AlgoExecDesc &execDesc)
{
    auto it = execDescSubCommMask.find(execDesc);
    if (it == execDescSubCommMask.end()) {
        // temlate类型的节点可能不存在execDescSubCommMask
        HCCL_ERROR("[ParallelExecutor] AlgoExecDesc not found in execDescSubCommMask");
        return HCCL_SUCCESS;
    }
    std::vector<ThreadHandle> syncInterThreads;
    std::vector<u32> syncNotifyOnMain;
    for (int i = 0; i < sizeof(it->second) * CHAR_BIT; i++) {
        if (it->second & (1u << i)) {
            syncInterThreads.emplace_back(subThreads_.at(i).at(0));
            syncNotifyOnMain.emplace_back(i);
        }
    }
    return PostSyncInterThreads(mainThread_, syncInterThreads, syncNotifyOnMain);
}

// 线程布局如下所示，maxIntra/maxInter表示每个阶段所需的最大线程数量，notifyNumPerThread需要多预留1个用于和主进程同步
// thread[0]                                = main thread
// thread[1]                                = intra main → notifyNumPerThread[0]
// threads[2..maxIntra+1]                   = intra slaves → notifyNumPerThread[1..maxIntra+1]
// thread[maxIntra+2]                       = inter main → notifyNumPerThread[maxIntra+2]
// threads[maxIntra+3..maxIntra+maxInter+2] = inter slaves → notifyNumPerThread[maxIntra+3..]
// resourceRequest.notifyNumPerThread布局如下所示，notifyNumPerThread需要多预留1个用于和main thread同步
// notifyNumPerThread[0]                    = intra NotifyNumOnMainThread + 1
// notifyNumPerThread[1..maxIntra]          = intra notifyNumPerThread[...]必须预留每个阶段的最大值
// notifyNumPerThread[maxIntra]             = inter NotifyNumOnMainThread + 1
// notifyNumPerThread[maxIntra+1..maxIntra+maxIntra]= inter notifyNumPerThread[...]

HcclResult ParallelExecutor::CalcResRecursion(AlgoExecDesc &nodeAloExecDesc, u32 &subCommMask)
{
    size_t childrenSize = nodeAloExecDesc.children.size();
    u32 subCommMask = 0;
    for (size_t i = 0; i < childrenSize; ++i) {
        u32 childrenSubCommMask = 0;
        VariantType &v = nodeAloExecDesc.children[i];
        // 处理 TemplateExecDesc
        if (TemplateExecDesc *templateExeDes = std::get_if<TemplateExecDesc>(&v)) {
            int subCommIndex = templateExeDes->subCommIndex;
            childrenSubCommMask |= (1U << subCommIndex);
            std::vector<RankInfo> templateRanks = algHierarchyInfo_.infos[subCommIndex];
            BaseTemplate baseTemplate
                = GetTemplate(algo_.engineType, templateExeDes->templateDesc, templateRanks, myRank_);
            AlgResourceRequest tempRequest;
            CHK_RET(baseTemplate.CalcRes(hcclComm_, tempRequest));
            maxSlaveThreadNum_.at(subCommIndex) = max(maxSlaveThreadNum_.at(subCommIndex), tempRequest.slaveThreadNum);
            maxNotifyNumOnMainThread_.at(subCommIndex)
                = max(maxNotifyNumOnMainThread_.at(subCommIndex), tempRequest.notifyNumOnMainThread);
            auto it = std::max_element(tempRequest.notifyNumPerThread.begin(), tempRequest.notifyNumPerThread.end());
            maxNotifyNumPerThread_.at(subCommIndex) = max(maxNotifyNumPerThread_.at(subCommIndex), *it);
        }
        // 处理 AlgoExecDesc（递归）
        else if (auto *algoDescPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&v)) {
            // 注意：*algoDescPtr 是 std::shared_ptr<AlgoExecDesc>
            // 使用 **algoDescPtr 或 algoDescPtr->get() 解引用 shared_ptr
            CHK_RET(CalcResRecursion(**algoDescPtr, childrenSubCommMask));
        } else {
            return HCCL_ERR_INVALID_TYPE; // 或者其他错误码
        }
        subCommMask |= childrenSubCommMask;
    }
    // 需要将本节点的subCommMask插入到map表中
    UpdateSubCommMask(nodeAloExecDesc, subCommMask);
    return HCCL_SUCCESS;
}

inline void ParallelExecutor::UpdateSubCommMask(AlgoExecDesc &nodeAloExecDesc, const u32 subCommMask)
{
    auto it = execDescSubCommMask_.find(nodeAloExecDesc);
    if (it != execDescSubCommMask_.end()) {
        it->second = subCommMask;
    } else {
        execDescSubCommMask_.emplace(nodeAloExecDesc, subCommMask);
    }
}

HcclResult ParallelExecutor::CalcRes(AlgResourceRequest &resourceRequest)
{
    auto topoLevelNum = algHierarchyInfo_.infos.size();
    maxSlaveThreadNum_.assign(topoLevelNum, 0);
    maxNotifyNumOnMainThread_.assign(topoLevelNum, 0);
    maxNotifyNumPerThread_.assign(topoLevelNum, 0);
    u32 rootSubCommMask = 0;
    CHK_RET(CalcResRecursion(algo_.algoExecDesc, rootSubCommMask));

    auto subThreadBegin = threads_.begin;
    auto subThreadEnd = threads_.begin;
    resourceRequest.notifyNumOnMainThread = topoLevelNum;
    resourceRequest.slaveThreadNum = 0;
    for (size_t subCommIndex = 0; subCommIndex < topoLevelNum; subCommIndex++) {
        // 每个通信子域还需要一条主流，所以求和还需要+1
        resourceRequest.slaveThreadNum += maxSlaveThreadNum_.at(subCommIndex) + 1;
        resourceRequest.notifyNumPerThread.emplace_back(maxNotifyNumOnMainThread_.at(subCommIndex) + 1);
        notifyNumOnSubMainThread_.emplace_back(maxNotifyNumOnMainThread_.at(subCommIndex) + 1);
        // 再插入maxSlaveThreadNum个maxNotifyNumPerThreadnotifyNumPerThread
        resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
            maxSlaveThreadNum_.at(subCommIndex), maxNotifyNumPerThread_.at(subCommIndex));
        subThreadBegin = (subCommIndex == 0 ? subThreadBegin : subThreadEnd) + 1;
        subThreadEnd = subThreadBegin + 1 + maxSlaveThreadNum_.at(subCommIndex);
        subThreads_.at(subCommIndex).assign(subThreadBegin, subThreadEnd);
    }
    return HCCL_SUCCESS;
}

HcclResult ParallelExecutor::GenTemplateRes(
    const AlgResourceCtxSerializable &resCtx, const u32 subCommIndex, TemplateResource &templateResource)
{
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo));
    templateResource.channels = remoteRankToChannelInfo.at(subCommIndex);
    templateResource.threads = subThreads_.at(subCommIndex);
    templateResource.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    // 其他参数待确认是否还需要保留
    return HCCL_SUCCESS;
}

inline void ParallelExecutor::GenTemplateDataParams(const AlgResourceCtxSerializable &resCtx,
    AlgoExecDataDesc &algoExecDataDesc, TemplateDataParams &templateDataParams)
{
    templateDataParams.inputBufferPtr = dataInfo_.inputPtr;
    templateDataParams.outputBufferPtr = dataInfo_.outputPtr;
    templateDataParams.cclBufferPtr = resCtx.cclMem;
    templateDataParams.buffInfo.inBuffType = algoExecDataDesc.inputBufferType;
    templateDataParams.buffInfo.outBuffType = algoExecDataDesc.outputBufferType;
    templateDataParams.buffInfo.hcclBuffType = algoExecDataDesc.cclBufferType;
    templateDataParams.dataType = dataInfo_.dataDesUnion.dataType;
    templateDataParams.sliceCount = algoExecDataDesc.dataCount;
    templateDataParams.tailCount = tailCount;
    templateDataParams.dataOffset = algoExecDataDesc.dataOffset;
    templateDataParams.cclBufferOffset = cclBufferOffset;
    templateDataParams.reduceOp = dataInfo_.reduceOp;
    templateDataParams.root = root_;
    templateDataParams.enableRemoteMemAccess = opMode_ == OpMode::OFFLOAD;
    return;
}

inline void ParallelExecutor::UpdateDataSplit(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc,
    u32 childrenId, std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc)
{
    size_t childrenSize = algoExecDesc.children.size();
    u32 dataSplitRatioSum = algoExecDesc.dataSplitRatio.iter().sum();
    childrenAlgoExecDataDesc.at(childrenId) = algoExecDataDesc;
    // 先赋值父节点的信息，然后在根据并行/串行策略分开处理
    if (algoExecDesc.execPolicy == ExecPolicy::PARALLEL) {
        // 根据algoExecDesc中的数据切分比例切分
        u64 dataCount = algoExecDataDesc.dataCount * algoExecDataDesc.dataSplitRatio.at(childrenId) / dataSplitRatioSum;
        childrenAlgoExecDataDesc.at(childrenId).dataOffset
            = algoExecDataDesc.dataOffset + dataCount * childrenId * dataTypeSize_;
        if (childrenId == childrenSize - 1) {
            dataCount = algoExecDataDesc.dataCount;
            for(size_t i = 0; i < childrenSize - 1; i++){
                dataCount = dataCount - childrenAlgoExecDataDesc.at(i).dataCount;
            }
        }
        childrenAlgoExecDataDesc.at(childrenId).dataCount = dataCount;

    } else {
        if (childrenId > 0) {
            // 如果是串行需要将当前节点的输入设置为上个子节点的输出
            childrenAlgoExecDataDesc.at(childrenId).ranksForInputData
                = childrenAlgoExecDataDesc.at(childrenId - 1).ranksForOutputData;
            childrenAlgoExecDataDesc.at(childrenId).inputBufferType
                = childrenAlgoExecDataDesc.at(childrenId - 1).outputBufferType;
        }
        childrenAlgoExecDataDesc.at(childrenId).outputBufferType
            = (childrenId < childrenSize - 1) ? childrenAlgoExecDataDesc.at(childrenId).cclBufferType
                                              : childrenAlgoExecDataDesc.at(childrenId).outputBufferType;
    }
    return;
}

HcclResult ParallelExecutor::RunTemplateDesc(
    const AlgResourceCtxSerializable &resCtx, TemplateExecDesc *templateExeDes, AlgoExecDataDesc &algoExecDataDesc)
{
    std::vector<RankInfo> templateRanks = algHierarchyInfo_.infos[templateExeDes->subCommIndex];
    BaseTemplate baseTemplate = GetTemplate(algo_.engineType, templateExeDes->templateDesc, templateRanks, myRank_);
    // 根据阶段生成template的资源参数
    TemplateResource templateResource;
    CHK_RET(GenTemplateRes(resCtx, templateExeDes->subCommIndex, templateResource));
    // 根据阶段生成template的数据参数
    TemplateDataParams templateDataParams;
    GenTemplateDataParams(resCtx, algoExecDataDesc, templateDataParams);
    std::vector<u32> ranksForOutputData;
    CHK_RET(baseTemplate.KernelRun(templateDataParams, templateResource, ranksForOutputData));
    algoExecDataDesc.ranksForOutputData = ranksForOutputData;
}

HcclResult ParallelExecutor::OrchestrateLoop(
    const AlgResourceCtxSerializable &resCtx, AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc)
{
    size_t childrenSize = algoExecDesc.children.size();
    std::vector<AlgoExecDataDesc> childrenAlgoExecDataDesc;
    childrenAlgoExecDataDesc.resize(childrenSize);
    for (size_t i = 0; i < childrenSize; ++i) {
        // 如果是串行需要开始前同步
        if (algoExecDesc.execPolicy == ExecPolicy::SEQUENCE && childrenSize > 1) {
            CHK_RET(PreSyncBySubCommMask(algoExecDesc));
        }
        UpdateDataSplit(algoExecDesc, algoExecDataDesc, i, childrenAlgoExecDataDesc);
        VariantType &v = algoExecDesc.children[i];
        // 处理 TemplateExecDesc
        if (TemplateExecDesc *templateExeDes = std::get_if<TemplateExecDesc>(&v)) {
            CHK_RET(RunTemplateDesc(resCtx, templateExeDes, childrenAlgoExecDataDesc.at(i)));
        }
        // 处理 AlgoExecDesc（递归）
        else if (auto *algoDescPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&v)) {
            CHK_RET(OrchestrateLoop(resCtx, **algoDescPtr, childrenAlgoExecDataDesc.at(i)));
        } else {
            return HCCL_ERR_INVALID_TYPE; // 或者其他错误码
        }
        // 如果是串行需要回到主流做尾同步
        if (algoExecDesc.execPolicy == ExecPolicy::SEQUENCE && childrenSize > 1) {
            CHK_RET(PostSyncBySubCommMask(algoExecDesc));
        }
    }
    // 整个执行器的数据输出直接用最后一个子节点的执行器的数据输出
    algoExecDataDesc.ranksForOutputData = childrenAlgoExecDataDesc.at(childrenSize - 1).ranksForOutputData;
    return HCCL_SUCCESS;
}

} // namespace ops_hccl