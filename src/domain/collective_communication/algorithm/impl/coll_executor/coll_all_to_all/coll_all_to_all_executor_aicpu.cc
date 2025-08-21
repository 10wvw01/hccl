#include "coll_all_to_all_executor.h"
#include "device_capacity.h"

namespace hccl {
HcclResult CollAlltoAllExecutor::SetParallelTaskLoader(ParallelTaskLoader* parallelTaskLoader)
{
    return HCCL_SUCCESS;
}

}