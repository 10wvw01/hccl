#ifndef OPS_HCCL_UTILS_UTILS_H
#define OPS_HCCL_UTILS_UTILS_H

#include <vector>
#include <algorithm>
#include "alg_param.h"
#include "primitives/mesh_primitives.h"

namespace ops_hccl {

/** 工具：计算 rankId 在 ranks 中的 algRank（即索引位置）。 */
inline HcclResult GetAlgRank(u32 rankId, const std::vector<u32> &ranks, u32 &algRank)
{
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    CHK_PRT_RET(it == ranks.end(),
                HCCL_ERROR("[GetAlgRank] rank[%u] not in ranks.", rankId),
                HCCL_E_PARA);
    algRank = static_cast<u32>(std::distance(ranks.begin(), it));
    return HCCL_SUCCESS;
}

/** 本地拷贝：srcSlice -> dstSlice。 */
HcclResult LocalCopy(const ThreadHandle &thread, const DataSlice &srcSlice, const DataSlice &dstSlice);

/** 前同步：主线程通知从线程可以开始通信。 */
HcclResult PreSyncInterThreads(const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads,
                               const std::vector<u32> &notifyIdxMainToSub);

/** 后同步：从线程通知主线程通信完成。 */
HcclResult PostSyncInterThreads(const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads,
                                const std::vector<u32> &notifyIdxSubToMain);

}  // namespace ops_hccl

#endif // OPS_HCCL_UTILS_UTILS_H
