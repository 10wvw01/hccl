#include "parallel_2ops_executor.h"

namespace ops_hccl {

HcclResult ParallelExecutor::PreSyncByResTable(const AlgoExecDesc &execDesc)
{
    auto it = resTable_.find(execDesc);
    if (it == resTable_.end()) {
        HCCL_ERROR("[ParallelExecutor] AlgoExecDesc not found in resTable_");
        return HCCL_E_INTERNAL;
    }
    std::vector<ThreadHandle> syncInterThreads;
    std::vector<u32> syncNotifyOnAlgoExec;
    for (int i = 0; i < sizeof(it->second.subCommMask) * CHAR_BIT; i++) {
        if (it->second.subCommMask & (1u << i)) {
            syncInterThreads.emplace_back(subThreads_.at(i).at(0));
            // 每个通信子域维度主线程notify - 1才是需要同步的notify数量
            syncNotifyOnAlgoExec.emplace_back(notifyNumOnSubMainThread_.at(i) - 1);
        }
    }
    return PreSyncInterThreads(mainThread_, syncInterThreads, syncNotifyOnAlgoExec);
}

HcclResult ParallelExecutor::PostSyncByResTable(const AlgoExecDesc &execDesc)
{
    auto it = resTable_.find(execDesc);
    if (it == resTable_.end()) {
        HCCL_ERROR("[ParallelExecutor] AlgoExecDesc not found in resTable_");
        return HCCL_E_INTERNAL;
    }
    std::vector<ThreadHandle> syncInterThreads;
    std::vector<u32> syncNotifyOnMain;
    for (int i = 0; i < sizeof(it->second.subCommMask) * CHAR_BIT; i++) {
        if (it->second.subCommMask & (1u << i)) {
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
    // todo需要计算resTable_
    size_t childrenSize = nodeAloExecDesc.children.size();
    u32 childrenSubCommMask = 0;
    for (size_t i = 0; i < childrenSize; ++i) {
        VariantType &v = nodeAloExecDesc.children[i];
        // 处理 TemplateExecDesc
        if (TemplateExecDesc *templateExeDes = std::get_if<TemplateExecDesc>(&v)) {
            subCommMask |= (1 << templateExeDes->subCommIndex);
            std::vector<RankInfo> templateRanks = algHierarchyInfo_.infos[templateExeDes->subCommIndex];
            BaseTemplate baseTemplate
                = GetTemplate(algo_.engineType, templateExeDes->templateDesc, templateRanks, myRank_);
            AlgResourceRequest tempRequest;
            CHK_RET(baseTemplate.CalcRes(hcclComm_, tempRequest));
            maxSlaveThreadNum_.at(templateTopoIndex)
                = max(maxSlaveThreadNum_.at(templateTopoIndex), tempRequest.slaveThreadNum);
            maxNotifyNumOnMainThread_.at(templateTopoIndex)
                = max(maxNotifyNumOnMainThread_.at(templateTopoIndex), tempRequest.notifyNumOnMainThread);
            auto it = std::max_element(tempRequest.notifyNumPerThread.begin(), tempRequest.notifyNumPerThread.end());
            maxNotifyNumPerThread_.at(templateExeDes->subCommIndex)
                = max(maxNotifyNumPerThread_.at(templateExeDes->subCommIndex), *it);
        }
        // 处理 AlgoExecDesc（递归）
        else if (auto *algoDescPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&v)) {
            // 注意：*algoDescPtr 是 std::shared_ptr<AlgoExecDesc>
            // 使用 **algoDescPtr 或 algoDescPtr->get() 解引用 shared_ptr
            CHK_RET(CalcResRecursion(**algoDescPtr, childrenSubCommMask));
        } else {
            return HCCL_ERR_INVALID_TYPE; // 或者其他错误码
        }
    }
    subCommMask |= childrenSubCommMask;
    // 需要将本节点的subCommMask插入到map表中
    UpdateResTable(nodeAloExecDesc, subCommMask);
    return HCCL_SUCCESS;
}

inline void ParallelExecutor::UpdateResTable(AlgoExecDesc &nodeAloExecDesc, const u32 subCommMask)
{
    auto it = resTable_.find(nodeAloExecDesc);
    if (it != resTable_.end()) {
        it->second.subCommMask = subCommMask;
    } else {
        resTable_.emplace(nodeAloExecDesc, AlgoExecRes{subCommMask});
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
    for (auto templateTopoIndex = 0; templateTopoIndex < topoLevelNum; templateTopoIndex++) {
        // 每个通信子域还需要一条主流，所以求和还需要+1
        resourceRequest.slaveThreadNum += maxSlaveThreadNum_.at(templateTopoIndex) + 1;
        resourceRequest.notifyNumPerThread.emplace_back(maxNotifyNumOnMainThread_.at(templateTopoIndex) + 1);
        notifyNumOnSubMainThread_.emplace_back(maxNotifyNumOnMainThread_.at(templateTopoIndex) + 1);
        // 再插入maxSlaveThreadNum个maxNotifyNumPerThreadnotifyNumPerThread
        resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
            maxSlaveThreadNum_.at(templateTopoIndex), maxNotifyNumPerThread_.at(templateTopoIndex));
        subThreadBegin = (templateTopoIndex == 0 ? subThreadBegin : subThreadEnd) + 1;
        subThreadEnd = subThreadBegin + 1 + maxSlaveThreadNum_(templateTopoIndex);
        subThreads_.at(templateTopoIndex).assign(subThreadBegin, subThreadEnd);
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
HcclResult ParallelExecutor::GenTemplateDataParams(
    const AlgResourceCtxSerializable &resCtx, TemplateDataParams &templateDataParams, u64 sliceOffset, u64 sliceCount)
{
    void *inputPtr = nullptr;
    u64 inputSize = 0;
    void *outputPtr = nullptr;
    u64 outputSize = 0;
    DataDesUnion dataDesUnion;
    HcclReduceOp reduceOp_ = HCCL_REDUCE_RESERVED;

    templateDataParams.buffInfo.inputPtr = dataInfo_.inputPtr;
    templateDataParams.buffInfo.outputPtr = dataInfo_.outputPtr;
    templateDataParams.buffInfo.hcclBuff = resCtx.cclMem;
    templateDataParams.buffInfo.inBuffType = BufferType::INPUT;
    templateDataParams.buffInfo.outBuffType = BufferType::OUTPUT;
    templateDataParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    templateDataParams.buffInfo.inputSize = dataInfo_.inputSize;
    templateDataParams.buffInfo.outputSize = dataInfo_.outputSize;
    templateDataParams.dataType = dataInfo_.dataDesUnion.dataType;

    templateDataParams.buffInfo.inBuffBaseOff = sliceOffset;
    templateDataParams.buffInfo.hcclBuffBaseOff = scratchOffset;
    templateDataParams.count = sliceCount;
    templateDataParams.tailCout = sliceCount % ;
    // todo ,待计算应该只有部分源语需要计算填充，例如scatter/allgather等

    templateDataParams.enableRemoteMemAccess = opMode_ == OpMode::OFFLOAD;
    return;
}

void ParallelExecutor::GetParallelDataSplit(
    u64 offset, u64 count, u32 childrenSize, std::vector<u64> &childrenOffset, std::vector<u64> &childrenCount) const
{
    // 先均分数据，后续再看看是否需要优化
    childrenOffset.clear();
    childrenCount.clear();
    if (childrenSize == 0 || count == 0) {
        return;
    }
    childrenOffset.reserve(childrenSize);
    childrenCount.reserve(childrenSize);
    u64 childrenCountFloor = count / childrenSize;
    u32 lastIndex = childrenSize - 1;
    for (u32 i = 0; i < lastIndex; ++i) {
        childrenCount.push_back(childrenCountFloor);
        childrenOffset.push_back(offset + i * childrenCountFloor * dataTypeSize_);
    }
    childrenCount.push_back(count - childrenCountFloor * lastIndex);
    childrenOffset.push_back(offset + lastIndex * childrenCountFloor * dataTypeSize_);
    return;
}

HcclResult ParallelExecutor::OrchestrateLoop(const AlgResourceCtxSerializable &resCtx, AlgoExecDesc nodeAloExecDesc,
    u64 offset, u64 count, u64 inputStride, u64 &outputStride)
{
    size_t childrenSize = nodeAloExecDesc.children.size();
    std::vector<u64> childrenOffset(childrenSize, offset);
    std::vector<u64> childrenCount(childrenSize, count);
    u64 childrenInputStride = inputStride;
    u64 childrenOutputStride;
    for (size_t i = 0; i < childrenSize; ++i) {
        // 如果是串行需要开始前同步
        if (nodeAloExecDesc.execPolicy == ExecPolicy::SEQUENCE) {
            CHK_RET(PreSyncByResTable(nodeAloExecDesc));
        } else {
            // 如果是并行需要切分数据
            GetParallelDataSplit(offset, count, childrenSize, childrenOffset, childrenCount);
        }

        VariantType &v = nodeAloExecDesc.children[i];
        // 处理 TemplateExecDesc
        if (TemplateExecDesc *templateExeDes = std::get_if<TemplateExecDesc>(&v)) {
            std::vector<RankInfo> templateRanks = algHierarchyInfo_.infos[templateExeDes->subCommIndex];
            BaseTemplate baseTemplate
                = GetTemplate(algo_.engineType, templateExeDes->templateDesc, templateRanks, myRank_);
            // 根据阶段生成template的资源参数
            TemplateResource templateResource;
            CHK_RET(GenTemplateRes(resCtx, templateExeDes->subCommIndex, templateResource));
            // 根据inputStride和Topo信息计算outputStride;
            CHK_RET(CalOutputStride(childrenInputStride, childrenOutputStride));
            // 根据阶段生成template的数据参数
            TemplateDataParams templateDataParams;
            CHK_RET(GenTemplateDataParams(resCtx, templateDataParams, childrenOffset.at(i), childrenCount.at(i)),
                childrenInputStride, childrenOutputStride);
            CHK_RET(template.KernelRun(templateDataParams, templateResource));
        }
        // 处理 AlgoExecDesc（递归）
        else if (auto *algoDescPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&v)) {
            CHK_RET(OrchestrateLoop(resCtx, **algoDescPtr, childrenOffset.at(i), childrenCount.at(i),
                childrenInputStride, &childrenOutputStride));
        } else {
            return HCCL_ERR_INVALID_TYPE; // 或者其他错误码
        }
        // 如果是串行需要回到主流做尾同步
        if (nodeAloExecDesc.execPolicy == ExecPolicy::SEQUENCE) {
            CHK_RET(PostSyncByResTable(nodeAloExecDesc));
            childrenInputStride = childrenOutputStride;
        }
    }
    outputStride = childrenOutputStride;
    return HCCL_SUCCESS;
}

HcclResult ParallelExecutor::CalOutputStride(u64 inputStride, u64 &outputStride)
{
}

} // namespace ops_hccl