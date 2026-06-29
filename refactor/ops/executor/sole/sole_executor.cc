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

u64 SoleExecutor::GetMaxProcCntPerLoop()
{
    // 计算当前cclBuffer每轮最大能处理的数据量（这个数据量不直接切分input/output，跟算子有关，建议直接基于dataCount_计算）
    u64 multiple = algTemplate.CalcScratchMultiple();
    u64 maxProCntPerLoop = RoundUp(dataCount_, multiple);
    return maxProCntPerLoop;
}

HcclResult SoleExecutor::GenTemplateDataParams(u64 processCount, TemplateDataParams &dataParams)
{
    dataCount = 
    dataSize = 
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

HcclResult SoleExecutor::OrchestrateLoop(const u64 maxProCntPerLoop, TemplateDataParams dataParams)
{
    // 按照每轮最大处理数据量，计算每轮处理的数据信息
    TemplateDataParams dataParams;
    GenTemplateDataParams(dataParams);
    
    // 调用算法模板的展开
    algTemplate.KernelRun(dataParams, algRes);
}

} // namespace ops_hccl