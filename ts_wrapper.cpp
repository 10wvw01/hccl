#include "ts_wrapper.h"
#include <assert.h>

// 模拟TS接口（实际CANN中对应TS_Send/TS_Recv）
// 这里使用伪代码示意，真实开发需包含 ts_engine.h
extern "C" {
void TS_Send(const void *buf, uint64_t len, uint32_t dst, void *stream);
void TS_Recv(void *buf, uint64_t len, uint32_t src, void *stream);
void TS_Sync(void *stream);
}

void ts_send(const void *buf, uint64_t size_bytes, int dst_rank, aclrtStream stream)
{
    // 实际调用: TS_Send(buf, size_bytes, (uint32_t)dst_rank, stream);
    // 仿真直接返回
}

void ts_recv(void *buf, uint64_t size_bytes, int src_rank, aclrtStream stream)
{
    // 实际调用: TS_Recv(buf, size_bytes, (uint32_t)src_rank, stream);
}

void ts_synchronize_stream(aclrtStream stream)
{
    // 实际调用: TS_Sync(stream);
    // 或 aclrtSynchronizeStream(stream);
}