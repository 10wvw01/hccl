#include "hccl_algorithm.h"

namespace ops_hccl {
class SoleExecutor : public BaseExecutor {
public:
    HcclResult CalcRes(const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resReq) override;

private:
    u64 SoleExecutor::GetMaxProcCntPerLoop() override;

    HcclResult OrchestrateLoop(const BaseExecutorParam &baseExecutorParam,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceCtxSerializable &resCtx) override;
}
} // namespace ops_hccl