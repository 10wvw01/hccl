#include "parallel_2ops_executor.h"

namespace ops_hccl {

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

HcclResult ParallelExecutor::CalcResforNode(AlgoExecDesc nodeAloExecDesc, AlgResourceRequest &resourceRequest)
{
    size_t childrenSize = nodeAloExecDesc.children.size();
    std::vector<AlgResourceRequest> tempRequest(childrenSize);
    for (size_t i = 0; i < childrenSize; ++i) {
        VariantType &v = nodeAloExecDesc.children[i];
        // 处理 TemplateExecDesc
        if (TemplateExecDesc *templateExeDes = std::get_if<TemplateExecDesc>(&v)) {
            std::vector<RankInfo> templateRanks = algHierarchyInfo_.infos[templateExeDes->subCommIndex];
            BaseTemplate baseTemplate
                = GetTemplate(algo_.engineType, templateExeDes->templateDesc, templateRanks, myRank_);
            CHK_RET(baseTemplate.CalcRes(hcclComm_, tempRequest.at(i)));
        }
        // 处理 AlgoExecDesc（递归）
        else if (auto *algoDescPtr = std::get_if<std::shared_ptr<AlgoExecDesc>>(&v)) {
            // 注意：*algoDescPtr 是 std::shared_ptr<AlgoExecDesc>
            // 使用 **algoDescPtr 或 algoDescPtr->get() 解引用 shared_ptr
            CHK_RET(CalcResforNode(**algoDescPtr, tempRequest.at(i)));
        } else {
            // 不应该到达这里，说明 variant 包含了未预期的类型
            return HCCL_ERR_INVALID_TYPE; // 或者其他错误码
        }
    }
    CHK_RET(MergeResRequest(tempRequest, nodeAloExecDesc.execPolicy, resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult ParallelExecutor::MergeResRequest(
    std::vector<AlgResourceRequest> &tempRequest, ExecPolicy execPolicy, AlgResourceRequest &resourceRequest)
{
    u32 notifyNumOnMainThread = 0;
    u32 slaveThreadNum = 0;
    std::vector<u32> notifyNumPerThread;
    size_t vectorSize = tempRequest.size();
    // 并行notifyNumOnMainThread等于vectorSize，slaveThreadNum求和
    // 串行notifyNumOnMainThread和slaveThreadNum都取最大值
    if (nodeAloExecDesc.execPolicy == ExecPolicy::PARALLEL) {
        notifyNumOnMainThread = vectorSize;
        slaveThreadNum += tempRequest.slaveThreadNum;
    } else if (nodeAloExecDesc.execPolicy == ExecPolicy::SEQUENCE) {
        for (size_t i = 0; i < vectorSize; ++i) {
            slaveThreadNum = max(slaveThreadNum, tempRequest.at(i).slaveThreadNum);
            notifyNumOnMainThread = max(notifyNumOnMainThread, tempRequest.at(i).notifyNumOnMainThread);
        }
        notifyNumOnMainThread = max(notifyNumOnMainThread, tempRequest.notifyNumOnMainThread);
    }
    resourceRequest.notifyNumOnMainThread = notifyNumOnMainThread;
    resourceRequest.slaveThreadNum = slaveThreadNum;
}

HcclResult ParallelExecutor::CalcRes(AlgResourceRequest &resourceRequest)
{
    CHK_RET(CalcResforNode(comm, topoInfo, algo_.algoExecDesc, resourceRequest));
}

HcclResult ParallelExecutor::CalcRes(HcclComm comm, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    uint32_t parallelNum = algo_.templateDescs.at(0).size(); // 每一步计算的数据部分数
    // 第一个元素表示maxIntra，第二个元素表示maxInter，后续待扩展
    std::vector<u32> maxSlaveThreadNum(parallelNum);
    std::vector<u32> maxNotifyNumOnMainThread(parallelNum);
    std::vector<u32> maxNotifyNumPerThread(parallelNum);
    CHK_RET(CalcSubTopoMaxRes(
        comm, topoInfo, algHierarchyInfo, maxSlaveThreadNum, maxNotifyNumOnMainThread, maxNotifyNumPerThread));
    resourceRequest.notifyNumOnMainThread = parallelNum;
    resourceRequest.slaveThreadNum = 0;
    mainThread_ = threads_.at(0);
    auto subThreadBegin = threads_.begin;
    auto subThreadEnd = threads_.begin;
    for (auto templateTopoIndex = 0; templateTopoIndex < parallelNum; templateTopoIndex++) {
        // 所有通信维度(intra/inter)maxSlaveThreadNum求和
        resourceRequest.slaveThreadNum += maxSlaveThreadNum.at(templateTopoIndex);
        // 先插入NotifyNumOnMainThread
        resourceRequest.notifyNumPerThread.emplace_back(maxNotifyNumOnMainThread.at(templateTopoIndex) + 1);
        // 再插入maxSlaveThreadNum个maxNotifyNumPerThreadnotifyNumPerThread
        resourceRequest.notifyNumPerThread.insert(resourceRequest.notifyNumPerThread.end(),
            maxSlaveThreadNum.at(templateTopoIndex), maxNotifyNumPerThread.at(templateTopoIndex));
        subThreadBegin = (templateTopoIndex == 0 ? subThreadBegin : subThreadEnd) + 1;
        subThreadEnd = subThreadBegin + 1 + maxSlaveThreadNum(templateTopoIndex);
        subThreads_.at(templateTopoIndex).assign(subThreadBegin, subThreadEnd);
    }

    HCCL_DEBUG("[ParallelExecutor][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u], "
               "channels[%u]",
        myRank_, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum,
        resourceRequest.channels.size());
    for (auto i = 0; i < resourceRequest.notifyNumPerThread.size(); i++) {
        HCCL_DEBUG("[ParallelExecutor][CalcRes] myRank[%u], notifyNumPerThread[%u]=[%u]", myRank_, i,
            resourceRequest.notifyNumPerThread[i]);
    }

    return HCCL_SUCCESS;
}

HcclResult ParallelExecutor::CalcSubTopoMaxRes(HcclComm comm, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, std::vector<u32> maxSlaveThreadNum,
    std::vector<u32> maxNotifyNumOnMainThread, std::vector<u32> maxNotifyNumPerThread);
{
    uint32_t stageNum = algo_.templateDescs.size();          // 并行计算的步骤
    uint32_t parallelNum = algo_.templateDescs.at(0).size(); // 每一步计算的数据部分数

    for (auto stage = 0; stage < stageNum; stage++) {
        for (auto dataPart = 0; dataPart < parallelNum; dataPart++) {
            // 根据TemplateDescrb获取实例化生成算法的template
            auto templateTopoIndex = algo_.templateTopoIndex.at(stage).at(dataPart);
            vector<RankInfo> templateRanks = algHierarchyInfo.infos[templateTopoIndex];
            BaseTemplate template = Func(algo_.templates.at(stage).at(dataPart), templateRanks);

            AlgResourceRequest tempRequest;
            CHK_RET(template.CalcRes(comm, param, topoInfo, tempRequest));
            maxSlaveThreadNum.at(templateTopoIndex)
                = max(maxSlaveThreadNum.at(templateTopoIndex), tempRequest.slaveThreadNum);
            maxNotifyNumOnMainThread.at(templateTopoIndex)
                = max(maxNotifyNumOnMainThread.at(templateTopoIndex), tempRequest.notifyNumOnMainThread);
            // 为了保险起见每个通信维度(intra/inter)的notifyNum取最大值
            auto it = std::max_element(tempRequest.notifyNumPerThread.begin(), tempRequest.notifyNumPerThread.end());
            maxNotifyNumPerThread.at(templateTopoIndex) = max(maxNotifyNumPerThread.at(templateTopoIndex), *it);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ParallelExecutor::GenerateTemplateRes(u32 stage, u32 dataPart, TemplateResource templateResource)
{
}
HcclResult ParallelExecutor::GenTemplateDataParams(u32 stage, u32 dataPart, TemplateDataParams &templateDataParams)
{
    // Allgather算子的输出目前是用的output，统一调整为scratch内存，对应的地址/类型/size统一调整

    // HcclBuffBaseOff待分析是否可以归一

    // 按照数据类型是分拆/还是聚合的/还是原位拷贝三种不一样计算下面的参数
    if (OP == allgather) {
        // outBuffBaseOff/inputSliceStride/outputSliceStride/repeatNum/InputRepeatStride/OutputRepeatStride
    } else if {
    } else {
        // outBuffBaseOff/inputSliceStride/outputSliceStride/repeatNum/InputRepeatStride/OutputRepeatStride
    }
}
HcclResult ParallelExecutor::OrchestrateLoop(const AlgResourceCtxSerializable &resCtx)
{
    // 参考现有的allGather算子实现
    uint32_t stageNum = algo_.templateDescs.size();          // 并行计算的步骤
    uint32_t dataPartNum = algo_.templateDescs.at(0).size(); // 每一步计算的数据部分数
    std::vector<ThreadHandle> templateMainThreads_;
    // templateMainThreads_第一个元素是intra的主线程位, 第二个元素是inter的主线程
    templateMainThreads_.emplace_back(subThreads_.at(0).at(0));
    templateMainThreads_.emplace_back(subThreads_.at(1).at(0));

    for (auto stage = 0; stage < stageNum; stage++) {
        // 第一步开始前同步
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_.at(stage)));
        for (auto dataPart = 0; stage < dataPartNum; dataPart++) {
            // 根据TemplateDescrb获取实例化生成算法的template
            BaseTemplate template = Func(algo_.templates.at(stage).at(dataPart));

            // 根据阶段生成template的资源参数
            TemplateResource templateResource;
            CHK_RET(GenTemplateRes(stage, dataPart, templateResource));

            // 根据阶段生成template的数据参数
            TemplateDataParams templateDataParams;
            CHK_RET(GenTemplateDataParams(stage, dataPart, templateDataParams));

            CHK_RET(template.KernelRun(templateDataParams, templateResource, algo_.engineType));
        }
        // 第一步做完后回到主流做尾同步
        CHK_RET(PostSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnMain_));
    }
}

HcclResult ParallelExecutor::PrepareResForTemplate()
{
    uint32_t stageNum = algo_.templateDescs.size();          // 并行计算的步骤
    uint32_t dataPartNum = algo_.templateDescs.at(0).size(); // 每一步计算的数据部分数
    uint32_t intraThreadsNum = 0;
    uint32_t interThreadsNum = 0;
    uint32_t intraNotifyOnMainThread = 0;
    uint32_t interNotifyOnMainThread = 0;
    for (auto stage = 0; stage < stageNum; stage++) {
        for (auto dataPart = 0; stage < dataPartNum; dataPart++) {
            // 根据TemplateDescrb获取实例化生成算法的template
            BaseTemplate template = Func(algo_.templates.at(stage).at(dataPart));
            AlgResourceRequest TempRequest;
            template.GetRes(TempRequest);
            // 判断这个Template到底是在intra方向还是inter方向
            if () {
                intraThreadsNum = max(intraThreadsNum, template.slaveThreadNum + 1);
                intraNotifyOnMainThread = template.notifyNumOnMainThread;
            } else {
                interThreadsNum = max(interThreadsNum, template.slaveThreadNum + 1);
                interNotifyOnMainThread = template.notifyNumOnMainThread;
            }
        }
        syncNotifyOnTemplates_.at(stage) = {intraNotifyOnMainThread, interNotifyOnMainThread};
    }

    syncNotifyOnMain_.clear();
    for (auto dataPart = 0; stage < dataPartNum; dataPart++) {
        syncNotifyOnMain_.emplace_back(dataPart);
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl