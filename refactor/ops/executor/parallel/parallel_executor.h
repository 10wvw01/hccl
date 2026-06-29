#include <cstdint>

namespace ops_hccl {
class ParallelExecutor : public BaseExecutor {
public:
    explicit ParallelExecutor(HcclAlgorithm &alg);
    ~ParallelExecutor() override = default;
    HcclResult CalcRes(AlgResourceRequest &resourceRequest);

private:
    HcclResult CalcResforNode(AlgoExecDesc nodeAloExecDesc, AlgResourceRequest &resourceRequest);
    HcclResult MergeResRequest(
        std::vector<AlgResourceRequest> &tempRequest, ExecPolicy execPolicy, AlgResourceRequest &resourceRequest);
    HcclResult PrepareResForTemplate();
    HcclResult OrchestrateLoop(const AlgResourceCtxSerializable &resCtx);
    HcclResult GenTemplateRes(u32 stage, u32 dataPart, TemplateResource templateResource);
    HcclResult GenTemplateDataParams(u32 stage, u32 dataPart, TemplateDataParams &templateDataParams);

    HcclResult CalcSubTopoMaxRes(HcclComm comm, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, std::vector<u32> maxSlaveThreadNum,
        std::vector<u32> maxNotifyNumOnMainThread, std::vector<u32> maxNotifyNumPerThread) = 0;
}
} // namespace ops_hccl