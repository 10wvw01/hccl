#include <cstdint>

namespace ops_hccl {
class Parallel2OpsExecutor : public BaseExecutor {
public:
    explicit Parallel2OpsExecutor(HcclAlgorithm &alg);
    ~Parallel2OpsExecutor() override = default;
    HcclResult CalcRes(HcclComm comm, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest) = 0;
    HcclResult Orchestrate(const AlgResourceCtxSerializable &resCtx) = 0;

private:
    HcclResult PrepareResForTemplate();
    HcclResult OrchestrateLoop(const AlgResourceCtxSerializable &resCtx);
    HcclResult GenTemplateRes(u32 stage, u32 dataPart, TemplateResource templateResource);
    HcclResult GenTemplateDataParams(u32 stage, u32 dataPart, TemplateDataParams &templateDataParams);

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};
    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};
    std::vector<std::vector<u32>> intraHierarchyInfo_;
    std::vector<std::vector<u32>> interHierarchyInfo_;
    std::vector<ThreadHandle> intraThreads_;
    std::vector<ThreadHandle> interThreads_;
    std::map<u32, std::vector<ChannelInfo>> intraLinkMap_;
    std::map<u32, std::vector<ChannelInfo>> interLinkMap_;
    std::vector<ThreadHandle> threads_;
}
} // namespace ops_hccl