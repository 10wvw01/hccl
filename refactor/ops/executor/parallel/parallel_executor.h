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
    HcclResult OrchestrateLoop(const AlgResourceCtxSerializable &resCtx, AlgoExecDesc &algoExecDesc,
        AlgoExecDataDesc &algoExecDataDesc) override;
    HcclResult GenTemplateRes(
        const AlgResourceCtxSerializable &resCtx, const u32 subCommIndex, TemplateResource &templateResource);
    inline void GenTemplateDataParams(const AlgResourceCtxSerializable &resCtx, AlgoExecDataDesc &algoExecDataDesc,
        TemplateDataParams &templateDataParams);
    inline void UpdateSubCommMask(AlgoExecDesc &nodeAloExecDesc, const u32 subCommMask);
    HcclResult PreSyncBySubCommMask(const AlgoExecDesc &execDesc);
    HcclResult PostSyncBySubCommMask(const AlgoExecDesc &execDesc);
    inline void UpdateDataSplit(AlgoExecDesc &algoExecDesc, AlgoExecDataDesc &algoExecDataDesc,
    u32 childrenId, std::vector<AlgoExecDataDesc> &childrenAlgoExecDataDesc);
    // 处理单个 TemplateExecDesc 子节点：实例化template、生成资源/数据参数、计算stride、KernelRun
    HcclResult RunTemplateDesc(
        const AlgResourceCtxSerializable &resCtx, TemplateExecDesc *templateExeDes, AlgoExecDataDesc &algoExecDataDesc);

    struct AlgoExecDataDesc {
        u64 dataOffset{0};
        u64 dataCount{0};
        std::vector<u32> ranksForInputData;
        std::vector<u32> ranksForOutputData;
        BufferType inputBufferType{BufferType::INPUT};
        BufferType outputBufferType{BufferType::OUTPUT};
        BufferType cclBufferType{BufferType::HCCL_BUFFER};
    };
}
} // namespace ops_hccl