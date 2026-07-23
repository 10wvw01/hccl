#ifndef DLOG_DEF_H
#define DLOG_DEF_H

#include <stdint.h>

// log level id
#define DLOG_DEBUG 0x0      // debug level id
#define DLOG_INFO  0x1      // info level id
#define DLOG_WARN  0x2      // warning level id
#define DLOG_ERROR 0x3      // error level id
#define DLOG_NULL  0x4      // don't print log

#define RUN_LOG_MASK        (0x01000000U)    // print log to directory run

enum {
    HCCL = 3,
};

#ifdef __cplusplus
extern "C" {
#endif
void DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...) __attribute__((weak));
int32_t dlog_getlevel(int32_t moduleId, int32_t *enableEvent) __attribute__((weak));
#ifdef __cplusplus
}
#endif

#endif // DLOG_DEF_H