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
    HcclResult OrchestrateLoop(const AlgResourceCtxSerializable &resCtx);
    HcclResult GenTemplateRes(
        const AlgResourceCtxSerializable &resCtx, const u32 subCommIndex, TemplateResource &templateResource);
    HcclResult GenTemplateDataParams(const AlgResourceCtxSerializable &resCtx, TemplateDataParams &templateDataParams);
    inline void UpdateResTable(AlgoExecDesc &nodeAloExecDesc, const u32 subCommMask);
    HcclResult PreSyncByResTable(const AlgoExecDesc &execDesc);
    HcclResult PostSyncByResTable(const AlgoExecDesc &execDesc);
}
} // namespace ops_hccl