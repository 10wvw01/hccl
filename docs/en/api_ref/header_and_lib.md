# Overview

HCCL (Huawei Collective Communication Library) provides high-performance collective and point-to-point communication capabilities for single-server and multi-server environments based on Ascend AI Processors. It is a core component of CANN. HCCL and the HCOMM communication foundation library are decoupled through dynamic loading with `dlsym`, and can be built and versioned independently. For information about the public header files and libraries of HCOMM, see the [HCOMM documentation (Chinese)](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/header_and_lib.md).

This section describes the public header files and libraries provided by HCCL.

## Header Files and Libraries

### API Categories

**Table 1** API categories

| API category | Description |
| --- | --- |
| Collective communication operators | Eleven collective communication APIs, including AllReduce, Broadcast, AllGather(V), ReduceScatter(V), Scatter, Reduce, and AlltoAll(V)(C). |
| Point-to-point communication operators | Three point-to-point communication APIs: Send, Recv, and BatchSendRecv. |
| MC2 custom operators | Nine MC2 (Kernel Fusion Custom) framework APIs, including HcclKfc\* parameter objects and HcclCreateOpResCtx. |

### Header Files and Libraries Required by the APIs

Install the firmware, drivers, and CANN software package before compiling or running applications that use HCCL APIs.

Include the header files required by the HCCL APIs used in your application. The purpose of each public header file is described in the following table.

HCCL header files are installed in `${INSTALL_DIR}/include/hccl/`, and libraries are installed in `${INSTALL_DIR}/lib64/`. Replace `${INSTALL_DIR}` with the CANN installation path. For example, the default path for an installation performed by the `root` user is `/usr/local/Ascend/cann`.

> **NOTICE:**
> When compiling an application that uses HCCL APIs, link only the libraries required by the included header files. Linking unnecessary shared libraries may cause version compatibility issues or unexpected behavior after an upgrade.

**Table 2** Header files

| Header file | Purpose | Library |
| --- | --- | --- |
| `hccl/hccl.h` | Defines 14 collective and point-to-point communication operator APIs, including AllReduce, Broadcast, AllGather(V), ReduceScatter(V), Scatter, Reduce, AlltoAll(V)(C), Send, Recv, and BatchSendRecv. | `libhccl.so` |
| `hccl/hccl_mc2.h` | Defines nine MC2 custom operator framework APIs, including HcclKfc\* parameter object allocation and configuration APIs and HcclCreateOpResCtx for creating a communication resource context. | `libhccl.so` |

HCCL is released as the `cann-hccl_<version>_linux-<arch>.run` package. The package contains `libhccl.so`, public header files, and `aicpu_hccl.tar.gz` (the HCCL AICPU operator package). A static build also produces `libhccl_static.a`.

For source build and installation instructions, see the [Build Guide](../build/build.md). For function prototypes, parameters, supported data types, and constraints of communication operators, see the [Chinese Communication Operator API Reference](../../zh/api_ref/comm_op_interface/README.md).
