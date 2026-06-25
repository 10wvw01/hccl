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
    HcclResult OrchestrateLoop(
        const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter);
    void GenTemplateAlgParamsIntra0(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                                    const u64 dataOffset, const u64 dataCountPerLoopAixs0, const u64 scratchOffset,
                                    TemplateDataParams &tempAlgParamsIntra0) const;
    void GenTemplateAlgParamsIntra1(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                                    const u64 dataOffset, const u64 dataCountPerLoopAixs1, const u64 scratchOffset,
                                    TemplateDataParams &tempAlgParamsIntra1) const;
    void GenTemplateAlgParamsInter0(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                                    const u64 dataOffset, const u64 dataCountPerLoopAixs0, const u64 scratchOffset,
                                    TemplateDataParams &tempAlgParamsInter0) const;
    void GenTemplateAlgParamsInter1(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                                    const u64 dataOffset, const u64 dataCountPerLoopAixs1, const u64 scratchOffset,
                                    TemplateDataParams &tempAlgParamsInter1) const;        
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