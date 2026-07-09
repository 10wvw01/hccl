#ifndef ALG_DATA_TRANS_WRAPPER
#define ALG_DATA_TRANS_WRAPPER

namespace ops_hccl {
HcclResult PreSyncInterThreads(const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads,
                               const std::vector<u32> &notifyIdxMainToSub);

HcclResult PostSyncInterThreads(const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads,
                                const std::vector<u32> &notifyIdxSubToMain);
}

#endif // !ALG_DATA_TRANS_WRAPPER