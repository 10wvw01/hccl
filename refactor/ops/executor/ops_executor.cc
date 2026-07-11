#include "ops_executor.h"

namespace ops_hccl {
OpsExecutor::OpsExecutor(HcclAlgorithm &algo, OpParam &param)
    : algo_(algo),
      myRank_(param.userRank),
      rankSize_(0),
      root_(param.root)
{
    opMode_ = param.opMode;
    dataInfo_.inputPtr = param.inputPtr;
    dataInfo_.inputSize = param.inputSize;
    dataInfo_.outputPtr = param.outputPtr;
    dataInfo_.outputSize = param.outputSize;
    dataInfo_.reduceOp = param.reduceType;
    dataInfo_.dataType = param.DataDes.dataType;

    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
}

OpsExecutor::~OpsExecutor()
{
}

HcclResult OpsExecutor::CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo)
{
    // 储存通信域指针
    hcclComm_ = comm;
    // TODO：topoMatch暂不修改参数
    algo_.topoMatch->MatchTopo(hcclComm_, topoInfo, algHierarchyInfo_);
    // 算rankSize
    u32 topoLevelNum = algHierarchyInfo_.infos.size();
    rankSize_ = 1;
    for (size_t i = 0; i < topoLevelNum; i++) {
        rankSize_ *= algHierarchyInfo_.infos.at(i).at(0).size();
    }
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::Orchestrate(AlgResourceCtxSerializable &resCtx)
{
    // 初始化资源信息
    InitRes(resCtx);
    // 切分数据阶段（子类实现GetMaxProCntPerLoop函数）
    // maxProcessCount表示每次循环能处理的数据量，该数据量定义与入参dataCount保持一致（不同op有区别）
    u64 dataCount = dataInfo_.inputSize / dataTypeSize_;
    if (dataCount == 0) {
        HCCL_ERROR("[OpsExecutor] dataCount is zero");
        return HCCL_SUCCESS;
    }
    u64 maxProcCntPerLoop = GetMaxProcCntPerLoop(dataCount);
    // 循环下发阶段（按照每轮最大处理数据量，循环展开）
    u64 loopTimes = (dataCount + maxProcCntPerLoop - 1) / maxProcCntPerLoop;
    u64 offsetCount = 0;
    for (u64 loopIdx = 0; loopIdx < loopTimes; ++loopIdx) {
        u64 processCount = maxProcCntPerLoop;
        if (loopIdx == loopTimes - 1) {
            u64 remainder = dataCount % maxProcCntPerLoop;
            processCount = (remainder == 0) ? maxProcCntPerLoop : remainder;
        }
        u64 tailCount = (loopIdx == loopTimes - 1) ? (processCount % rankSize_) : 0;
        // 子类实现
        AlgoExecDataDesc algoExecDataDesc;
        InitAlgoExecDataDesc(algoExecDataDesc, offsetCount * dataTypeSize_, processCount, tailCount);
        OrchestrateLoop(algo_.algoExecDesc, algoExecDataDesc);
        // 偏移增加
        offsetCount += processCount;
    }
    // TODO：储存队列和任务信息，用于FastLauch
    // SaveCtx();
    return HCCL_SUCCESS;
}

u64 OpsExecutor::GetMaxProcCntPerLoop(u64 dataCount)
{
    if (scratchMultiple_ == 0 || dataTypeSize_ == 0) {
        return std::max(dataCount, 1ULL);
    }
    // CCL buffer scratch容量约束：每element需要scratchMultiple_倍dataTypeSize_的scratch空间
    u64 maxByCcl = cclBufferInfo_.size / (static_cast<u64>(scratchMultiple_) * dataTypeSize_);
    // UB传输约束：硬件单次传输的element数上限
    u64 maxByUb = UB_MAX_DATA_SIZE / dataTypeSize_;
    // 取最小值（总量、CCL scratch、UB传输三者约束）
    u64 resCount = std::min({dataCount, maxByCcl, maxByUb});
    // 如果不是allgather需要对齐,其他情况一定能够保证resCount > ranksize
    if (algo_.hcclCmdType != HCCL_CMD_ALLGATHER) {
        resCount = (resCount / rankSize_) * rankSize_;
    }
    // 保护：保证至少返回 1，避免 Orchestrate 中 (dataCount_ + maxProcCntPerLoop - 1) / maxProcCntPerLoop 除零
    return std::max(resCount, 1ULL);
}

// 公共工具类函数

HcclResult OpsExecutor::InitRes(const AlgResourceCtxSerializable &resCtx)
{
    cclBufferInfo_.ptr = resCtx.cclMem.addr;
    cclBufferInfo_.size = resCtx.cclMem.size;
    cclBufferInfo_.bufferType = BufferType::HCCL_BUFFER;
    threads_ = resCtx.threads;
    mainThread_ = threads_.at(0);
    // TODO：考虑不同Executor
    // 需要restore原因，resCtx中储存用双层嵌套vector<vector<ChannelInfo>>，remoteRank信息在ChannelInfo中，查询不方便
    channelTable_ = RestoreChannelMap(resCtx);

    return HCCL_SUCCESS;
}

std::vector<std::map<u32, std::vector<ChannelInfo>>> OpsExecutor::RestoreChannelMap(
    const AlgResourceCtxSerializable &resCtx)
{
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo = resCtx.algHierarchyInfo;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> rankIdToChannelInfo(algHierarchyInfo.infos.size());
    for (u32 level = 0; level < algHierarchyInfo.infos.size(); level++) {
        for (auto &channel : resCtx.channels[level]) {
            u32 remoteRank = channel.remoteRank;
            rankIdToChannelInfo[level][remoteRank].push_back(channel);
        }
    }
    return rankIdToChannelInfo;
}

HcclResult OpsExecutor::PreSyncBySubCommMask(const AlgoExecDesc &execDesc)
{
    auto it = execDescSubCommMaskMap_.find(&execDesc);
    if (it == execDescSubCommMaskMap_.end()) {
        // temlate类型的节点可能不存在execDescSubCommMaskMap_
        HCCL_ERROR("[OpsExecutor] AlgoExecDesc not found in execDescSubCommMaskMap_");
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

HcclResult OpsExecutor::PostSyncBySubCommMask(const AlgoExecDesc &execDesc)
{
    auto it = execDescSubCommMaskMap_.find(&execDesc);
    if (it == execDescSubCommMaskMap_.end()) {
        // temlate类型的节点可能不存在execDescSubCommMaskMap_
        HCCL_ERROR("[OpsExecutor] AlgoExecDesc not found in execDescSubCommMaskMap_");
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

HcclResult OpsExecutor::CalcResRecursion(AlgoExecDesc &algoExecDesc, u32 &subCommMask)
{
    size_t childrenSize = algoExecDesc.children.size();
    u32 localSubCommMask = 0;
    for (size_t i = 0; i < childrenSize; ++i) {
        u32 childrenSubCommMask = 0;
        VariantType &v = algoExecDesc.children[i];
        // 处理 TemplateExecDesc
        if (TemplateExecDesc *templateExeDes = std::get_if<TemplateExecDesc>(&v)) {
            childrenSubCommMask |= (1U << templateExeDes->subCommIndex);
            CHK_RET(CalcTemplateRes(*templateExeDes));
        }
        // 处理 AlgoExecDesc（递归）
        else if (auto *algoDescPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&v)) {
            CHK_RET(CalcResRecursion(**algoDescPtr, childrenSubCommMask));
        } else {
            return HCCL_E_INTERNAL;
        }
        localSubCommMask |= childrenSubCommMask;
    }
    // 需要将本节点的subCommMask插入到map表中
    UpdateSubCommMaskMap(algoExecDesc, localSubCommMask);
    subCommMask = localSubCommMask;
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::CalcTemplateRes(const TemplateExecDesc &templateExeDes)
{
    int subCommIndex = templateExeDes.subCommIndex;
    std::vector<u32> templateRanks = algHierarchyInfo_.infos[subCommIndex].at(0);
    std::unique_ptr<BaseTemplate> baseTemplate
        = GetTemplate(algo_.engineType, templateExeDes.templateDesc, templateRanks, myRank_);
    AlgResourceRequest tempRequest;
    CHK_RET(baseTemplate->CalcRes(hcclComm_, tempRequest));
    maxSlaveThreadNum_.at(subCommIndex) = std::max(maxSlaveThreadNum_.at(subCommIndex), tempRequest.slaveThreadNum);
    maxNotifyNumOnMainThread_.at(subCommIndex)
        = std::max(maxNotifyNumOnMainThread_.at(subCommIndex), tempRequest.notifyNumOnMainThread);
    auto it = std::max_element(tempRequest.notifyNumPerThread.begin(), tempRequest.notifyNumPerThread.end());
    maxNotifyNumPerThread_.at(subCommIndex) = std::max(maxNotifyNumPerThread_.at(subCommIndex), *it);
    // todo 需要确认一下这个地方细节
    requestChannels_.at(subCommIndex) = tempRequest.channels.at(0);
    return HCCL_SUCCESS;
}

inline void OpsExecutor::UpdateSubCommMaskMap(AlgoExecDesc &algoExecDesc, const u32 subCommMask)
{
    auto it = execDescSubCommMaskMap_.find(&algoExecDesc);
    if (it != execDescSubCommMaskMap_.end()) {
        it->second = subCommMask;
    } else {
        execDescSubCommMaskMap_.emplace(&algoExecDesc, subCommMask);
    }
}

HcclResult OpsExecutor::CalcRes(AlgResourceRequest &resourceRequest)
{
    auto topoLevelNum = algHierarchyInfo_.infos.size();
    maxSlaveThreadNum_.assign(topoLevelNum, 0);
    maxNotifyNumOnMainThread_.assign(topoLevelNum, 0);
    maxNotifyNumPerThread_.assign(topoLevelNum, 0);
    subThreads_.assign(topoLevelNum, {});
    requestChannels_.assign(topoLevelNum, {});
    u32 rootSubCommMask = 0;
    CHK_RET(CalcResRecursion(algo_.algoExecDesc, rootSubCommMask));

    auto subThreadBegin = threads_.begin();
    auto subThreadEnd = threads_.begin();
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
        resourceRequest.channels.emplace_back(requestChannels_.at(subCommIndex));
    }
    // 全尺寸布局内存，保证所有的template的CCL buffer内存布局一致
    scratchMultiple_ = algo_.hcclCmdType == HCCL_CMD_ALLGATHER ? rankSize_ : 1;
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::GenTemplateRes(const u32 subCommIndex, TemplateResource &templateResource)
{
    templateResource.channels = channelTable_.at(subCommIndex);
    templateResource.threads = subThreads_.at(subCommIndex);
    // 其他参数待确认是否还需要保留
    return HCCL_SUCCESS;
}

inline void OpsExecutor::InitAlgoExecDataDesc(
    AlgoExecDataDesc &algoExecDataDesc, u64 dataOffset, u64 dataCount, u64 tailCount)
{
    algoExecDataDesc.dataOffset = dataOffset;
    algoExecDataDesc.tailCount = tailCount;

    algoExecDataDesc.inputBufferType = BufferType::INPUT;
    algoExecDataDesc.outputBufferType
        = (algo_.hcclCmdType == HCCL_CMD_BROADCAST) ? BufferType::INPUT : BufferType::OUTPUT;
    algoExecDataDesc.cclBufferType = BufferType::HCCL_BUFFER;
    if (algo_.hcclCmdType == HCCL_CMD_ALLGATHER) {
        algoExecDataDesc.sliceCount = dataCount;
        algoExecDataDesc.ranksForInputData.emplace_back(myRank_);
    } else {
        algoExecDataDesc.sliceCount = dataCount / rankSize_;
        for (int i = 0; i < rankSize_; i++) {
            algoExecDataDesc.ranksForInputData.emplace_back(i);
        }
    }
    // 初始化之后这个值递归过程中不再变化，后续传递给template使用
    algoExecDataDesc.stride = algoExecDataDesc.sliceCount;
}

inline void OpsExecutor::GenTemplateDataParams(
    AlgoExecDataDesc &algoExecDataDesc, TemplateDataParams &templateDataParams)
{
    templateDataParams.inputBufferPtr = dataInfo_.inputPtr;
    templateDataParams.outputBufferPtr = dataInfo_.outputPtr;
    templateDataParams.cclBufferPtr = cclBufferInfo_.ptr;
    templateDataParams.inputBufferType = algoExecDataDesc.inputBufferType;
    templateDataParams.outputBufferType = algoExecDataDesc.outputBufferType;
    templateDataParams.cclBufferType = algoExecDataDesc.cclBufferType;
    templateDataParams.dataType = dataInfo_.dataType;
    templateDataParams.sliceCount = algoExecDataDesc.sliceCount;
    templateDataParams.sliceOffset = algoExecDataDesc.sliceOffset;
    templateDataParams.tailCount = algoExecDataDesc.tailCount;
    templateDataParams.dataOffset = algoExecDataDesc.dataOffset;
    templateDataParams.reduceOp = dataInfo_.reduceOp;
    templateDataParams.root = root_;
    templateDataParams.enableRemoteMemAccess = opMode_ == OpMode::OFFLOAD;
    templateDataParams.ranksForInputData = algoExecDataDesc.ranksForInputData;
    templateDataParams.stride = algoExecDataDesc.stride;
    return;
}

inline void OpsExecutor::UpdateDataSplitParallel(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc,
    u32 childrenId, std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc)
{
    size_t childrenSize = algoExecDesc.children.size();
    u32 dataSplitRatioSum = std::accumulate(algoExecDesc.dataSplitRatio.begin(), algoExecDesc.dataSplitRatio.end(), 0);
    float dataSplitRatio = static_cast<float>(algoExecDesc.dataSplitRatio.at(childrenId)) / dataSplitRatioSum;
    // 根据algoExecDesc中的数据切分比例切分
    if (childrenId > 0) {
        childrenAlgoExecDataDesc.at(childrenId).sliceOffset
            = childrenAlgoExecDataDesc.at(childrenId - 1).sliceOffset
              + childrenAlgoExecDataDesc.at(childrenId - 1).sliceCount * dataTypeSize_;
    }
    u64 sliceCount = algoExecDataDesc.sliceCount * dataSplitRatio;
    if (childrenId == childrenSize - 1) {
        sliceCount = algoExecDataDesc.sliceCount;
        for (size_t i = 0; i < childrenSize - 1; i++) {
            sliceCount -= childrenAlgoExecDataDesc.at(i).sliceCount;
        }
    }
    childrenAlgoExecDataDesc.at(childrenId).sliceCount = sliceCount;
    childrenAlgoExecDataDesc.at(childrenId).tailCount
        = (childrenId == childrenSize - 1) ? algoExecDataDesc.tailCount : 0;
    return;
}

inline void OpsExecutor::UpdateDataSplitSequence(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc,
    u32 childrenId, std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc)
{
    size_t childrenSize = algoExecDesc.children.size();
    if (childrenId > 0) {
        // 如果是串行需要将当前节点的输入设置为上个子节点的输出
        childrenAlgoExecDataDesc.at(childrenId).ranksForInputData
            = childrenAlgoExecDataDesc.at(childrenId - 1).ranksForOutputData;
        childrenAlgoExecDataDesc.at(childrenId).inputBufferType
            = childrenAlgoExecDataDesc.at(childrenId - 1).outputBufferType;
    }
    childrenAlgoExecDataDesc.at(childrenId).outputBufferType
        = (childrenId == childrenSize - 1) ? algoExecDataDesc.outputBufferType : algoExecDataDesc.cclBufferType;
    return;
}

inline void OpsExecutor::MergeChildrenOutput(const AlgoExecDesc &algoExecDesc,
    const std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc, AlgoExecDataDesc &algoExecDataDesc)
{
    if (algoExecDesc.execPolicy == HcclAlgExecPolicy::PARALLEL) {
        u64 sliceCount = 0;
        for (const auto &child : childrenAlgoExecDataDesc) {
            sliceCount += child.sliceCount;
        }
        algoExecDataDesc.sliceCount = sliceCount;
    }
    algoExecDataDesc.ranksForOutputData = childrenAlgoExecDataDesc.back().ranksForOutputData;
    return;
}

HcclResult OpsExecutor::RunTemplateDesc(TemplateExecDesc *templateExeDes, AlgoExecDataDesc &algoExecDataDesc)
{
    std::vector<u32> templateRanks = algHierarchyInfo_.infos[templateExeDes->subCommIndex].at(0);
    std::unique_ptr<BaseTemplate> baseTemplate
        = GetTemplate(algo_.engineType, templateExeDes->templateDesc, templateRanks, myRank_);
    // 根据阶段生成template的资源参数
    TemplateResource templateResource;
    CHK_RET(GenTemplateRes(templateExeDes->subCommIndex, templateResource));
    // 根据阶段生成template的数据参数
    TemplateDataParams templateDataParams;
    GenTemplateDataParams(algoExecDataDesc, templateDataParams);
    CHK_RET(baseTemplate->KernelRun(templateDataParams, templateResource, algoExecDataDesc.ranksForOutputData));
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::OrchestrateLoop(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc)
{
    size_t childrenSize = algoExecDesc.children.size();
    std::vector<AlgoExecDataDesc> childrenAlgoExecDataDesc(childrenSize, algoExecDataDesc);
    // 如果是并行需要开始前同步
    if (algoExecDesc.execPolicy == HcclAlgExecPolicy::PARALLEL && childrenSize > 1) {
        CHK_RET(PreSyncBySubCommMask(algoExecDesc));
    }
    for (size_t i = 0; i < childrenSize; ++i) {
        if (algoExecDesc.execPolicy == HcclAlgExecPolicy::PARALLEL && childrenSize > 1) {
            UpdateDataSplitParallel(algoExecDesc, algoExecDataDesc, i, childrenAlgoExecDataDesc);
        } else {
            UpdateDataSplitSequence(algoExecDesc, algoExecDataDesc, i, childrenAlgoExecDataDesc);
        }
        VariantType &v = algoExecDesc.children[i];
        // 处理 TemplateExecDesc
        if (TemplateExecDesc *templateExeDes = std::get_if<TemplateExecDesc>(&v)) {
            CHK_RET(RunTemplateDesc(templateExeDes, childrenAlgoExecDataDesc.at(i)));
        }
        // 处理 AlgoExecDesc（递归）
        else if (auto *algoDescPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&v)) {
            CHK_RET(OrchestrateLoop(**algoDescPtr, childrenAlgoExecDataDesc.at(i)));
        } else {
            return HCCL_E_INTERNAL;
        }
    }
    // 整个执行器的数据输出直接用最后一个子节点的执行器的数据输出
    MergeChildrenOutput(algoExecDesc, childrenAlgoExecDataDesc, algoExecDataDesc);
    // 如果是并行需要回到主流做尾同步
    if (algoExecDesc.execPolicy == HcclAlgExecPolicy::PARALLEL && childrenSize > 1) {
        CHK_RET(PostSyncBySubCommMask(algoExecDesc));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
