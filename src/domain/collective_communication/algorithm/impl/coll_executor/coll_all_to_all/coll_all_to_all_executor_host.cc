#include "coll_all_to_all_executor.h"
#include "device_capacity.h"

namespace hccl {
HcclResult CollAlltoAllExecutor::SetParallelTaskLoader(ParallelTaskLoader* parallelTaskLoader)
{
    parallelTaskLoader_ = parallelTaskLoader;
    HCCL_DEBUG("[%s] process success", __func__);
    return HCCL_SUCCESS;
}

}