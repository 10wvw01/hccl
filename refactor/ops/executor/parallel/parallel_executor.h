#include <cstdint>

namespace ops_hccl {
class ParallelExecutor : public BaseExecutor {
public:
    explicit ParallelExecutor(HcclAlgorithm &alg);
    ~ParallelExecutor() override = default;
    HcclResult CalcRes(HcclComm comm, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest) = 0;
    HcclResult Orchestrate(const AlgResourceCtxSerializable &resCtx) = 0;

private:
    HcclResult PrepareResForTemplate();
    HcclResult OrchestrateLoop(const AlgResourceCtxSerializable &resCtx);
    HcclResult GenTemplateRes(u32 stage, u32 dataPart, TemplateResource templateResource);
    HcclResult GenTemplateDataParams(u32 stage, u32 dataPart, TemplateDataParams &templateDataParams);

}
} // namespace ops_hccl