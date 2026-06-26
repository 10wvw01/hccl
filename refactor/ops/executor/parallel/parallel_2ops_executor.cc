#include "parallel_2ops_executor.h"

namespace ops_hccl {
HcclResult Parallel2OpsExecutor::CalcRes(HcclComm comm, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    // 根据基类中的alg算法信息计算所需资源

    // 根据基类中的alg算法信息构建template
    // 根据template计算CalcRes
    // 最后根据2ops资源相加返回
}

HcclResult Parallel2OpsExecutor::Orchestrate(const AlgResourceCtxSerializable &resCtx)
{
    // 下面的成员变量在基类的成员函数中初始化，避免每个子类操作
    // maxTmpMemSize_ = resCtx.cclMem.size;
    // myRank_ = resCtx.topoInfo.userRank;
    // dataCount_ = param.DataDes.count;
    // dataType_ = param.DataDes.dataType;
    // dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    // dataSize_ = dataCount_ * dataTypeSize_;

    // 将计算资源分配个每个算法

    // 算法展开
}

HcclResult Parallel2OpsExecutor::GenerateTemplateRes(TemplateResource templateResource){templateResource}

HcclResult Parallel2OpsExecutor::GenerateTemplateRes(TemplateDataParams &templateResource)
{
}

HcclResult Parallel2OpsExecutor::OrchestrateLoop(const AlgResourceCtxSerializable &resCtx)
{
    // 参考现有的allGather算子实现
    uint32_t stageNum = algo_.templateDescs.size();          // 并行计算的步骤
    uint32_t dataPartNum = algo_.templateDescs.at(0).size(); // 每一步计算的数据部分数

    for (auto stage = 0; stage < stageNum; stage++) {
        // 第一步开始前同步
        CHK_RET(PreSyncInterThreads(mainThread_, templateMainThreads_, syncNotifyOnTemplates_));
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

HcclResult Parallel2OpsExecutor::HcclResult PrepareResForTemplate()
{
    uint32_t stageNum = algo_.templateDescs.size();          // 并行计算的步骤    
    uint32_t dataPartNum = algo_.templateDescs.at(0).size(); // 每一步计算的数据部分数
    uint32_t intraThreadsNum = 0;
    uint32_t interThreadsNum = 0;
    for (auto stage = 0; stage < stageNum; stage++) {    
        for (auto dataPart = 0; stage < dataPartNum; dataPart++) {
            // 根据TemplateDescrb获取实例化生成算法的template
            BaseTemplate template = Func(algo_.templates.at(stage).at(dataPart));
            AlgResourceRequest TempRequest;
            template.GetRes(TempRequest);
            // 判断这个Template到底是在intra方向还是inter方向
            if (){
                intraThreadsNum = max(intraThreadsNum, template.slaveThreadNum + 1);
            }else{
                interThreadsNum = max(interThreadsNum, template.slaveThreadNum + 1);
            }
        }
    }


    auto intraNotifyOnMainThread = intraTempRequest.notifyNumOnMainThread;
    auto interNotifyOnMainThread = interTempRequest.notifyNumOnMainThread;

    intraThreads_.assign(threads_.begin() + 1, threads_.begin() + intraThreadsNum + 1);
    interThreads_.assign(threads_.begin() + intraThreadsNum + 1, threads_.end());
    // 用于两个算法同步
    mainThread_ = threads_.at(0);
    templateMainThreads_.emplace_back(intraThreads_.at(0));
    templateMainThreads_.emplace_back(interThreads_.at(0));
    syncNotifyOnTemplates_ = {intraNotifyOnMainThread, interNotifyOnMainThread};

    syncNotifyOnMain_.clear();
    for (auto dataPart = 0; stage < dataPartNum; dataPart++) {
        syncNotifyOnMain_.emplace_back(dataPart);
    }    
    return HCCL_SUCCESS;
}

void GenTemplateAlgParamsIntra0(const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAixs0, const u64 scratchOffset, TemplateDataParams &tempAlgParamsIntra0)
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

void GenTemplateAlgParamsInter0(const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAixs0, const u64 scratchOffset, TemplateDataParams &tempAlgParamsIntra0)
{
    // Allgather算子的输出目前是用的output，统一调整为scratch内存，对应的地址/类型/size统一调整

    // HcclBuffBaseOff待分析是否可以归一

    // 按照数据类型是分拆/还是聚合的/还是原位拷贝三种不一样计算下面的参数
    if (OP == allgather) {
        // outBuffBaseOff/inputSliceStride/outputSliceStride/repeatNum/InputRepeatStride/OutputRepeatStride
    } else {
        // outBuffBaseOff/inputSliceStride/outputSliceStride/repeatNum/InputRepeatStride/OutputRepeatStride
    }
}

void GenTemplateAlgParamsInter1(const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAixs0, const u64 scratchOffset, TemplateDataParams &tempAlgParamsIntra0)
{
    // Allgather算子的输出目前是用的output，统一调整为scratch内存，对应的地址/类型/size统一调整

    // HcclBuffBaseOff待分析是否可以归一

    // 按照数据类型是分拆/还是聚合的/还是原位拷贝三种不一样计算下面的参数
    if (OP == allgather) {
        // outBuffBaseOff/inputSliceStride/outputSliceStride/repeatNum/InputRepeatStride/OutputRepeatStride
    } else {
        // outBuffBaseOff/inputSliceStride/outputSliceStride/repeatNum/InputRepeatStride/OutputRepeatStride
    }
}

void GenTemplateAlgParamsIntra1(const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
    const u64 dataCountPerLoopAixs0, const u64 scratchOffset, TemplateDataParams &tempAlgParamsIntra0)
{
    // Allgather算子的输出目前是用的output，统一调整为scratch内存，对应的地址/类型/size统一调整

    // HcclBuffBaseOff待分析是否可以归一

    // 按照数据类型是分拆/还是聚合的/还是原位拷贝三种不一样计算下面的参数
    if (OP == allgather) {
        // outBuffBaseOff/inputSliceStride/outputSliceStride/repeatNum/InputRepeatStride/OutputRepeatStride
    } else {
        // outBuffBaseOff/inputSliceStride/outputSliceStride/repeatNum/InputRepeatStride/OutputRepeatStride
    }
}

} // namespace ops_hccl