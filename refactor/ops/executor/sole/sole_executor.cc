#include "sole_executor.h"

namespace ops_hccl {

SoleExecutor::SoleExecutor(HcclAlgorithm &algo)
    : BaseExecutor(algo) {}

SoleExecutor::SoleExecutor(HcclAlgorithm &algo, BaseOpParam &param)
    : BaseExecutor(algo, param) {}

SoleExecutor::~SoleExecutor() {}

HcclResult SoleExecutor::CalcRes(const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resReq)
{
    auto allTemplates = GenAllTemplates(algHierarchyInfo);
    // TODO：待确认template需要什么参数
    HcclResult res = allTemplates.at(0).at(0).CalcRes(resReq);
    return res;
}

HcclResult SoleExecutor::Orchestrate(const BaseExecutorParam &baseExecutorParam,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceCtxSerializable &resCtx)
{
    // 初始化资源信息
    InitRes(resCtx);
    // 切分资源阶段（Sole不需要，跳过）
    // 切分数据阶段
    SplitData();
    // 初始化Template
    GenAllTemplate();
    // 循环下发阶段（按照每轮最大处理数据量，循环展开）
    u64 loopTimes = RoundUp(dataCount_, maxProcessCount);
    for (u64 loop = 0; loop < loopTimes; ++loop) {
        OrchestrateLoop(maxProcessCount);
    }
    // TODO：储存队列和任务信息，用于FastLauch
    SaveCtx();
}

HcclResult SoleExecutor::SplitData()
{
    // 计算当前cclBuffer每轮最大能处理的数据量（这个数据量不直接切分input/output，跟算子有关，建议直接基于dataCount_计算）
    u64 multiple = algTemplate.CalcScratchMultiple();
    u64 maxProcessCount = CalcProcessCount(dataCount_, multiple);
}

HcclResult SoleExecutor::OrchestrateLoop(const u64 maxProcessCount)
{
    // 按照每轮最大处理数据量，计算每轮处理的数据信息
    TemplateDataParams dataParams;
    GenTemplateDataParams(dataParams);
    
    // 调用算法模板的展开
    algTemplate.KernelRun(dataParams, algRes);
}

HcclResult SoleExecutor::GenTemplateDataParams(TemplateDataParams dataParams)
{
    // 1.处理数据片大小
    u64 sliceCount = maxProcessCount;
    u64 sliceSize = sliceCount * dataTypeSize_;
    // TODO：尾块处理
    u64 tailCount;
    u64 tailSize;

    // 2.计算输入Buffer偏移和参数
    void* inBufferPtr;
    BufferType inBufferType;
    u64 inBufferOffset;
    u64 inBufferStride;

    // 3.计算输出Buffer偏移和参数
    void* outBufferPtr;
    BufferType outBufferType;
    u64 outBufferOffset;
    u64 outBufferStride;

    // 4.计算cclBuffer偏移和参数
    void* cclBufferPtr;
    BufferType cclBufferType;
    u64 cclBufferOffset;

    // 5.计算其他参数
    u64 repeatNum;

    // TODO：
    // root/dataType放Template构造里传入
    // enableRemoteMemAccess，区分单算子还是图模式，放Template构造里传入
    // 带V算子的参数传入
}

} // namespace ops_hccl