#ifndef UT_VERSION_MACRO_H
#define UT_VERSION_MACRO_H

#ifndef CANN_VERSION
#define CANN_VERSION(major, minor, patch) ((major) * 10000000 + (minor) * 100000 + (patch) * 1000)
#endif

#endif
