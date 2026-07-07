#include "ops_executor.h"

constexpr u64 UB_MAX_DATA_SIZE = 256 * 1024 * 1024;  // 256MB, UB单次最大传输量

namespace ops_hccl {
OpsExecutor::OpsExecutor(HcclAlgorithm &algo, OpParam &param)
    : algo_(algo),
      myRank_(param.myRank),
      rankSize_(param.rankSize),
      root_(param.root)
{
    dataInfo_.inputPtr = param.inputPtr;
    dataInfo_.inputSize = param.inputSize;
    dataInfo_.outputPtr = param.outputPtr;
    dataInfo_.outputSize = param.outputSize;
    dataInfo_.reduceOp = param.reduceOp;
    // TODO: DataDesUnion赋值

    dataTypeSize_ = DATATYPE_SIZE_TABLE[baseOpParam.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    opMode_ = param.opMode;

    bufferInfo_.inputBuffer.ptr = inputPtr;
    bufferInfo_.inputBuffer.size = inputSize;
    bufferInfo_.inputBuffer.bufferType = BufferType::INPUT;
    bufferInfo_.outputBuffer.ptr = outputPtr;
    bufferInfo_.outputBuffer.size = outputSize;
    bufferInfo_.outputBuffer.bufferType = BufferType::OUTPUT;
}

OpsExecutor::~OpsExecutor()
{
}

HcclResult OpsExecutor::CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo)
{
    // 储存通信域指针
    hcclComm_ = comm;
    // TODO：topoMatch暂不修改参数
    algo_.topoMatch.MatchTopo(hcclComm_, topoInfo, algHierarchyInfo_);
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::Orchestrate(const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceCtxSerializable &resCtx)
{
    // 初始化资源信息
    InitRes(resCtx);
    // 切分数据阶段（子类实现GetMaxProCntPerLoop函数）
    // maxProcessCount表示每次循环能处理的数据量，该数据量定义与入参dataCount保持一致（不同op有区别）
    u64 maxProcCntPerLoop = GetMaxProcCntPerLoop(dataCount_);
    // 循环下发阶段（按照每轮最大处理数据量，循环展开）
    u64 loopTimes = RoundUp(dataCount_, maxProcCntPerLoop);
    u64 processCount = maxProcCntPerLoop;
    u64 offsetCount = 0;
    for (u64 loopIdx = 0; loopIdx < loopTimes; ++loopIdx) {
        if (loopIdx == loopTimes - 1) {
            processCount = dataCount_ % maxProcCntPerLoop;
        }
        // 子类实现
        AlgoExecDataDesc algoExecDataDesc;
        InitAlgoExecDataDesc(algoExecDataDesc, offsetCount * dataTypeSize_, processCount);
        OrchestrateLoop(resCtx, algo_.algoExecDesc, algoExecDataDesc);
        // 偏移增加
        offsetCount += processCount;
    }
    // TODO：储存队列和任务信息，用于FastLauch
    SaveCtx();
}

u64 OpsExecutor::GetMaxProcCntPerLoop(u64 dataCount)
{
    if (scratchMultiple_ == 0 || dataTypeSize_ == 0) {
        return dataCount;
    }
    // CCL buffer scratch容量约束：每element需要scratchMultiple_倍dataTypeSize_的scratch空间
    u64 maxByCcl = bufferInfo_.cclBuffer.size / (static_cast<u64>(scratchMultiple_) * dataTypeSize_);
    // UB传输约束：硬件单次传输的element数上限
    u64 maxByUb = UB_MAX_DATA_SIZE / dataTypeSize_;
    // 取最小值（总量、CCL scratch、UB传输三者约束）
    u64 resCount = std::min({dataCount, maxByCcl, maxByUb});
    // 对齐?
    return resCount;
}

// 公共工具类函数

HcclResult OpsExecutor::InitRes(const AlgResourceCtxSerializable &resCtx)
{
    bufferInfo_.cclBuffer = Buffer;
    bufferInfo_.cclBuffer.ptr = resCtx.cclMem.addr;
    bufferInfo_.cclBuffer.size = resCtx.cclMem.size;
    bufferInfo_.cclBuffer.buffetType = BufferType::HCCL_BUFFER;

    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;
    mainThread_ = threads_.at(0);
    // TODO：考虑不同Executor
    // 需要restore原因，resCtx中储存用双层嵌套vector<vector<ChannelInfo>>，remoteRank信息在ChannelInfo中，查询不方便
    channelTable_ = RestoreChannelMap();

    // TODO：加rankSize数组初始化
}

std::vector
    < std::map<u32, std::vector<ChannelInfo>> OpsExecutor::RestoreChannelMap(const AlgResourceCtxSerializable &resCtx)
{
    // 桥接用函数，理论上直接resCtx直接用该结构表即可
    // 使用原函数，略做改造，直接返回结构表（是否有性能问题？）
}

HcclResult OpsExecutor::SplitRes()
{
    // 需要切分的资源
    // algHierarchyInfo：根据TemplateExecDesc.subCommIndex切分
    // Thread: slaveThreadNum：需要算法提供GetRes
    // Notify: notifyNumOnMainThread, notifyNumPerThread：需要算法提供GetRes
    // Channel: 当前直接按照level切分，后续根据TemplateExecDesc.subCommIndex切分

    // 从map表里获取资源，相当于GetRes
    // map需要提供：节点数量和ID（ranks），thread数量和ID，Notify数量和ID，Channel数量和ID
}

HcclResult OpsExecutor::PreSyncBySubCommMask(const AlgoExecDesc &execDesc)
{
    auto it = execDescSubCommMask.find(execDesc);
    if (it == execDescSubCommMask.end()) {
        // temlate类型的节点可能不存在execDescSubCommMask
        HCCL_ERROR("[OpsExecutor] AlgoExecDesc not found in execDescSubCommMask");
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
    auto it = execDescSubCommMask.find(execDesc);
    if (it == execDescSubCommMask.end()) {
        // temlate类型的节点可能不存在execDescSubCommMask
        HCCL_ERROR("[OpsExecutor] AlgoExecDesc not found in execDescSubCommMask");
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

HcclResult OpsExecutor::CalcResRecursion(
    AlgoExecDesc &algoExecDesc, u32 rankSizeForInputData, u32 &rankSizeForOutputData, u32 &subCommMask)
{
    size_t childrenSize = algoExecDesc.children.size();
    u32 subCommMask = 0;
    u32 childrenRankSizeForInputData = rankSizeForInputData;
    u32 childrenRankSizeForOutputData = 0;
    for (size_t i = 0; i < childrenSize; ++i) {
        u32 childrenSubCommMask = 0;
        VariantType &v = algoExecDesc.children[i];
        if (algoExecDesc.execPolicy == ExecPolicy::SEQUENCE && i > 0) {
            childrenRankSizeForInputData = childrenRankSizeForOutputData;
        }
        // 处理 TemplateExecDesc
        if (TemplateExecDesc *templateExeDes = std::get_if<TemplateExecDesc>(&v)) {
            childrenSubCommMask |= (1U << templateExeDes->subCommIndex);
            float dataSplitSum = algo_.algoExecDesc.execPolicy == ExecPolicy::SEQUENCE
                                     ? algo_.algoExecDesc.dataSplitRatio.at(0)
                                     : std::accumulate(algo_.algoExecDesc.dataSplitRatio.begin(),
                                           algo_.algoExecDesc.dataSplitRatio.end(), 0);
            float dataSplitRatio = static_cast<float>(algoExecDesc.dataSplitRatio.at(i)) / dataSplitSum;
            CHK_RET(CalcTemplateRes(*templateExeDes, algoExecDesc, childrenRankSizeForInputData, dataSplitRatio,
                childrenRankSizeForOutputData));
        }
        // 处理 AlgoExecDesc（递归）
        else if (auto *algoDescPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&v)) {
            // 注意：*algoDescPtr 是 std::shared_ptr<AlgoExecDesc>
            // 使用 **algoDescPtr 或 algoDescPtr->get() 解引用 shared_ptr
            CHK_RET(CalcResRecursion(
                **algoDescPtr, childrenRankSizeForInputData, childrenRankSizeForOutputData, childrenSubCommMask));
        } else {
            return HCCL_ERR_INVALID_TYPE; // 或者其他错误码
        }
        subCommMask |= childrenSubCommMask;
    }
    // 需要将本节点的subCommMask插入到map表中
    UpdateSubCommMask(algoExecDesc, subCommMask);
    rankSizeForOutputData = childrenRankSizeForOutputData;
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::CalcTemplateRes(const TemplateExecDesc &templateExeDes, const AlgoExecDesc &algoExecDesc,
    u32 childrenRankSizeForInputData, float dataSplitRatio, u32 &childrenRankSizeForOutputData)
{
    int subCommIndex = templateExeDes.subCommIndex;
    std::vector<RankInfo> templateRanks = algHierarchyInfo_.infos[subCommIndex];
    BaseTemplate baseTemplate = GetTemplate(algo_.engineType, templateExeDes.templateDesc, templateRanks, myRank_);
    AlgResourceRequest tempRequest;
    CHK_RET(baseTemplate.CalcRes(hcclComm_, tempRequest));
    maxSlaveThreadNum_.at(subCommIndex) = max(maxSlaveThreadNum_.at(subCommIndex), tempRequest.slaveThreadNum);
    maxNotifyNumOnMainThread_.at(subCommIndex)
        = max(maxNotifyNumOnMainThread_.at(subCommIndex), tempRequest.notifyNumOnMainThread);
    auto it = std::max_element(tempRequest.notifyNumPerThread.begin(), tempRequest.notifyNumPerThread.end());
    maxNotifyNumPerThread_.at(subCommIndex) = max(maxNotifyNumPerThread_.at(subCommIndex), *it);
    float ScratchMultiple
        = static_cast<float> baseTemplate.CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    ScratchMultiple = ScratchMultiple * dataSplitRatio * childrenRankSizeForInputData;
    maxSubScratchMutiple_.at(subCommIndex) = max(maxSubScratchMutiple_.at(subCommIndex), ScratchMultiple);
    // 如果原语是allgather累乘，否则累除
    if (templateExeDes.templateDesc.hcclCmdType == ALLGATHER) {
        childrenRankSizeForOutputData = childrenRankSizeForInputData * templateRanks.size();
    } else {
        childrenRankSizeForOutputData = childrenRankSizeForInputData / templateRanks.size();
    }
    requestChannels_.at(subCommIndex) = tempRequest.channels.at(0);
    return HCCL_SUCCESS;
}

inline void OpsExecutor::UpdateSubCommMask(AlgoExecDesc &algoExecDesc, const u32 subCommMask)
{
    auto it = execDescSubCommMask_.find(algoExecDesc);
    if (it != execDescSubCommMask_.end()) {
        it->second = subCommMask;
    } else {
        execDescSubCommMask_.emplace(algoExecDesc, subCommMask);
    }
}

HcclResult OpsExecutor::CalcRes(AlgResourceRequest &resourceRequest)
{
    auto topoLevelNum = algHierarchyInfo_.infos.size();
    maxSlaveThreadNum_.assign(topoLevelNum, 0);
    maxNotifyNumOnMainThread_.assign(topoLevelNum, 0);
    maxNotifyNumPerThread_.assign(topoLevelNum, 0);
    u32 rootSubCommMask = 0;
    u32 rankSizeForInputData = algo_.hcclCmdType == ALLGATHER ? 1 : rankSize_;
    u32 rankSizeForOutputData = 0;
    CHK_RET(CalcResRecursion(algo_.algoExecDesc, rankSizeForInputData, rankSizeForOutputData, rootSubCommMask));

    auto subThreadBegin = threads_.begin;
    auto subThreadEnd = threads_.begin;
    resourceRequest.notifyNumOnMainThread = topoLevelNum;
    resourceRequest.slaveThreadNum = 0;
    float scratchMultiple = 0;
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
        scratchMultiple += maxSubScratchMutiple_.at(subCommIndex);
        resourceRequest.channels.emplace_back(requestChannels.at(subCommIndex));
    }
    scratchMultiple_ = std::ceil(scratchMultiple);
    return HCCL_SUCCESS;
}

HcclResult OpsExecutor::GenTemplateRes(
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

inline void OpsExecutor::InitAlgoExecDataDesc(AlgoExecDataDesc &algoExecDataDesc, u64 dataOffset, u64 dataCount)
{
    algoExecDataDesc.dataOffset = dataOffset;
    algoExecDataDesc.dataCount = dataCount;
    algoExecDataDesc.scratchOffset = dataOffset * scratchMultiple_;
    algoExecDataDesc.ranksForInputData.emplace_back(myRank_);
    algoExecDataDesc.inputBufferType = BufferType::INPUT;
    algoExecDataDesc.outputBufferType = algo_.hcclCmdType == BROADCAST ? BufferType::INPUT : BufferType::OUTPUT;
    algoExecDataDesc.cclBufferType = BufferType::HCCL_BUFFER;
}

inline void OpsExecutor::GenTemplateDataParams(const AlgResourceCtxSerializable &resCtx,
    AlgoExecDataDesc &algoExecDataDesc, TemplateDataParams &templateDataParams)
{
    templateDataParams.inputBufferPtr = dataInfo_.inputPtr;
    templateDataParams.outputBufferPtr = dataInfo_.outputPtr;
    templateDataParams.cclBufferPtr = resCtx.cclMem;
    templateDataParams.buffInfo.inBuffType = algoExecDataDesc.inputBufferType;
    templateDataParams.buffInfo.outBuffType = algoExecDataDesc.outputBufferType;
    templateDataParams.buffInfo.hcclBuffType = algoExecDataDesc.cclBufferType;
    templateDataParams.dataType = dataInfo_.dataType;
    templateDataParams.sliceCount = algoExecDataDesc.dataCount;
    // todo 待支持tailcount
    templateDataParams.tailCount = algoExecDataDesc.dataCount;
    templateDataParams.dataOffset = algoExecDataDesc.dataOffset;
    templateDataParams.cclBufferOffset = algoExecDataDesc.scratchOffset;
    templateDataParams.reduceOp = dataInfo_.reduceOp;
    templateDataParams.root = root_;
    templateDataParams.enableRemoteMemAccess = opMode_ == OpMode::OFFLOAD;
    return;
}

inline void OpsExecutor::UpdateDataSplitParallel(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc,
    u32 childrenId, std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc)
{
    size_t childrenSize = algoExecDesc.children.size();
    u32 dataSplitRatioSum = std::accumulate(algoExecDesc.dataSplitRatio.begin(), algoExecDesc.dataSplitRatio.end(), 0);
    childrenAlgoExecDataDesc.at(childrenId) = algoExecDataDesc;
    // 根据algoExecDesc中的数据切分比例切分
    childrenAlgoExecDataDesc.at(childrenId).dataOffset
        = childrenId == 0
              ? algoExecDataDesc.dataOffset
              : (childrenAlgoExecDataDesc.at(childrenId - 1).dataOffset
                    + childrenAlgoExecDataDesc.at(childrenId - 1).dataCount
                          * childrenAlgoExecDataDesc.at(childrenId - 1).ranksForInputData.size() * dataTypeSize_);
    u64 dataCount = algoExecDataDesc.dataCount * algoExecDataDesc.dataSplitRatio.at(childrenId) / dataSplitRatioSum;
    if (childrenId == childrenSize - 1) {
        dataCount = algoExecDataDesc.dataCount;
        for (size_t i = 0; i < childrenSize - 1; i++) {
            dataCount = dataCount - childrenAlgoExecDataDesc.at(i).dataCount;
        }
    }
    childrenAlgoExecDataDesc.at(childrenId).dataCount = dataCount;
    childrenAlgoExecDataDesc.at(childrenId).scratchOffset
        = childrenId == 0 ? algoExecDataDesc.scratchOffset
                          : (childrenAlgoExecDataDesc.at(childrenId - 1).scratchOffset
                                + childrenAlgoExecDataDesc.at(childrenId - 1).scratchSize);
    return;
}

inline void OpsExecutor::UpdateDataSplitSequence(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc,
    u32 childrenId, std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc)
{
    size_t childrenSize = algoExecDesc.children.size();
    childrenAlgoExecDataDesc.at(childrenId) = algoExecDataDesc;
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
    return;
}

inline void OpsExecutor::MergeChildrenOutput(const AlgoExecDesc &algoExecDesc,
    const std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc, AlgoExecDataDesc &algoExecDataDesc)
{
    u64 scratchSize = 0;
    if (algoExecDesc.execPolicy == ExecPolicy::SEQUENCE) {
        for (const auto &child : childrenAlgoExecDataDesc) {
            scratchSize = max(scratchSize, child.scratchSize);
        }
    } else {
        for (const auto &child : childrenAlgoExecDataDesc) {
            scratchSize += child.scratchSize;
        }
    }
    algoExecDataDesc.scratchSize = scratchSize;
    algoExecDataDesc.ranksForOutputData = childrenAlgoExecDataDesc.back().ranksForOutputData;
}

HcclResult OpsExecutor::RunTemplateDesc(
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
    float scratchMutiple = baseTemplate.CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    algoExecDataDesc.scratchSize = std::ceil(
        algoExecDataDesc.dataCount * algoExecDataDesc.ranksForInputData.size() * dataTypeSize_ * scratchMutiple);
}

HcclResult OpsExecutor::OrchestrateLoop(
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
        if (algoExecDesc.execPolicy == ExecPolicy::PARALLEL) {
            UpdateDataSplitParallel(algoExecDesc, algoExecDataDesc, i, childrenAlgoExecDataDesc);
        } else {
            UpdateDataSplitSequence(algoExecDesc, algoExecDataDesc, i, childrenAlgoExecDataDesc);
        }
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
    MergeChildrenOutput(algoExecDesc, childrenAlgoExecDataDesc, algoExecDataDesc);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
