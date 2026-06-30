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
    HcclResult GenTemplateRes(u32 stage, u32 dataPart, TemplateResource templateResource);
    HcclResult GenTemplateDataParams(u32 stage, u32 dataPart, TemplateDataParams &templateDataParams);
    inline void UpdateResTable(AlgoExecDesc &nodeAloExecDesc, const u32 subCommMask);

    // 从resTable_中按AlgoExecDesc查找AlgoExecRes，并触发PreSyncInterThreads
    // 仅在串行策略分支内使用，调用方需保证已开启对应的AlgoExecRes记录
    HcclResult PresyncByResTable(const AlgoExecDesc &execDesc);

    // 从resTable_中按AlgoExecDesc查找AlgoExecRes，并触发PostSyncInterThreads
    // notify索引取自类成员syncNotifyOnMain_，因为收方向槽位由并行编排统一分配
    HcclResult PostSyncByResTable(const AlgoExecDesc &execDesc);
    std::vector<u32> notifyNumOnSubMainThread_;    
    std::map<AlgoExecDesc, AlgoExecRes> resTable_;
    struct AlgoExecRes {
        // 数组下表表示templateTopoIndex
        u32 subCommMask;
    };
}
} // namespace ops_hccl