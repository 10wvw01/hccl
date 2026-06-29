#include <cstdint>

namespace ops_hccl {
class ParallelExecutor : public BaseExecutor {
public:
    explicit ParallelExecutor(HcclAlgorithm &alg);
    ~ParallelExecutor() override = default;
    HcclResult CalcRes(AlgResourceRequest &resourceRequest);

private:
    HcclResult CalcResRecursion(AlgoExecDesc nodeAloExecDesc);
    HcclResult MergeResRequest(
        std::vector<AlgResourceRequest> &tempRequest, ExecPolicy execPolicy, AlgResourceRequest &resourceRequest);
    HcclResult PrepareResForTemplate();
    HcclResult OrchestrateLoop(const AlgResourceCtxSerializable &resCtx);
    HcclResult GenTemplateRes(u32 stage, u32 dataPart, TemplateResource templateResource);
    HcclResult GenTemplateDataParams(u32 stage, u32 dataPart, TemplateDataParams &templateDataParams);

}
} // namespace ops_hccl