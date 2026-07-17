#include "ts_wrapper.h"
#include <assert.h>

extern "C" {
    void TS_Send(const void* buf, uint64_t len, uint32_t dst, void* stream);
    void TS_Recv(void* buf, uint64_t len, uint32_t src, void* stream);
    void TS_Sync(void* stream);
}

void ts_send(const void* buf, uint64_t size_bytes, int dst_rank, aclrtStream stream) {
    // 实际调用: TS_Send(buf, size_bytes, (uint32_t)dst_rank, stream);
}

void ts_recv(void* buf, uint64_t size_bytes, int src_rank, aclrtStream stream) {
    // 实际调用: TS_Recv(buf, size_bytes, (uint32_t)src_rank, stream);
}

void ts_synchronize_stream(aclrtStream stream) {
    // 实际调用: TS_Sync(stream);
}