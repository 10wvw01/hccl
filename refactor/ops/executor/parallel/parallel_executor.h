#include <cstdint>

namespace ops_hccl {
class ParallelExecutor : public BaseExecutor {
public:
    explicit ParallelExecutor(HcclAlgorithm &alg);
    ~ParallelExecutor() override = default;
    HcclResult CalcRes(AlgResourceRequest &resourceRequest);

private:
    HcclResult CalcResRecursion(AlgoExecDesc &nodeAloExecDesc, u32 &subCommMask);
    HcclResult PrepareResForTemplate();
    HcclResult OrchestrateLoop(const AlgResourceCtxSerializable &resCtx, AlgoExecDesc &nodeAloExecDesc,
        u64 offset, u64 count, u64 inputStride, u64 &outputStride);
    HcclResult GenTemplateRes(
        const AlgResourceCtxSerializable &resCtx, const u32 subCommIndex, TemplateResource &templateResource);
    HcclResult GenTemplateDataParams(const AlgResourceCtxSerializable &resCtx, TemplateDataParams &templateDataParams,
        u64 sliceOffset, u64 sliceCount, u64 InputStride, u64 OutputStride);
    inline void UpdateSubCommMask(AlgoExecDesc &nodeAloExecDesc, const u32 subCommMask);
    HcclResult PreSyncBySubCommMask(const AlgoExecDesc &execDesc);
    HcclResult PostSyncBySubCommMask(const AlgoExecDesc &execDesc);
    void GetParallelDataSplit(AlgoExecDesc &nodeAloExecDesc, u64 offset, u64 count, u32 childrenSize,
        std::vector<u64> &childrenOffset, std::vector<u64> &childrenCount) const;
    // 处理单个 TemplateExecDesc 子节点：实例化template、生成资源/数据参数、计算stride、KernelRun
    HcclResult RunTemplateDesc(const AlgResourceCtxSerializable &resCtx, TemplateExecDesc *templateExeDes,
        u64 sliceOffset, u64 sliceCount, u64 inputStride, u64 &outputStride);
    void CalcOutputStride(u32 subCommIndex, u64 &outputStride);
}
} // namespace ops_hccl