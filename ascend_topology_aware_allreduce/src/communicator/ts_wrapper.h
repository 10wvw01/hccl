#ifndef TS_WRAPPER_H
#define TS_WRAPPER_H

#include <stdint.h>
#include "../../include/all_reduce.h"

void ts_send(const void* buf, uint64_t size_bytes, int dst_rank, aclrtStream stream);
void ts_recv(void* buf, uint64_t size_bytes, int src_rank, aclrtStream stream);
void ts_synchronize_stream(aclrtStream stream);

#endif // TS_WRAPPER_H