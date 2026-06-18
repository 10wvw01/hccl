# Custom Communication Operators – AllGather Communication

## Sample

This sample demonstrates how to use the HCCL AIV communication programming API to develop a custom AllGather communication operator. It covers the following functions:

1.  Implements the AllGather collective communication operator using the AI Vector (AIV) communication engine.
2.  Includes the operator logic on the host and kernel implementation on the device.
3.  Provides a complete compilation, build, test, and verification process.

## Directory Structure

```text
├── CMakeLists.txt                      # Root-level build configuration file
├── op_host/
│   ├── CMakeLists.txt
│   ├── all_gather.cc                   # HcclAllGatherCustom operator implementation on the host
│   ├── launch_kernel.cc                # Kernel launch logic implementation
│   └── launch_kernel.h                 # Kernel launch API definition
├── op_kernel/
│   ├── CMakeLists.txt
│   └── launch_kernel_asc.asc           # Operator implementation on the kernel (Ascend C)
├── inc/
│   ├── hccl_custom_allgather.h         # External API header file for the custom operator
│   ├── common.h                        # Common type definitions and macros
│   ├── aiv_all_gather_mesh_1d.h        # AIV AllGather core algorithm implementation
│   ├── aiv_communication_base_v2.h     # AIV communication base class
│   ├── log.h                           # Log tool
│   ├── extra_args.h                    # Extra parameter definitions
│   └── sync_interface.h                # Synchronization API definition
└── testcase/
    ├── CMakeLists.txt                  # CMake configuration file for test cases
    ├── Makefile                        # Makefile for test cases (for build and running)
    └── main.cc                         # Main program for test cases
```

## I. Environment Setup

### 1. Prerequisites

This sample supports the following products in a single-node N-card configuration (N ≥ 2):

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>

### 2. Install the CANN Toolkit

Install the CANN Toolkit of the latest version. For details, see [Ascend Documentation - CANN Software Installation](https://hiascend.com/document/redirect/CannCommercialInstSoftware).

### 3. Configure Environment Variables

Use the default installation path of the `root user as an example:

```bash
source /usr/local/Ascend/cann/set_env.sh
```

The MPI environment is required for running test cases. Ensure that the MPI has been installed and configured.

## II. Build and Running

This sample provides a CMake-based build process and a Makefile-based test running script.

### 1. Build the Custom Operator Library

Run the following command in the sample root directory:

```bash
# 1. Create a build directory
mkdir build

# 2. Go to the build directory
cd build

# 3. Run the CMake configuration
cmake ..

# 4. Build the project (to generate libhccl_custom_allgather.so)
make
```

### 2. Run Test Cases

After the build is complete, go to the `testcase` directory and run the test:

```bash
# 5. Go to the test case directory
cd ../testcase

# 6. Build and run the test cases
# This command automatically builds the test program, sets LD_LIBRARY_PATH, and runs the program using mpirun.
make run
```

### 3. Expected Result

After the command is executed, the terminal displays logs similar to the following (using 2-card configuration as an example):

```text
[INFO] MPI Initialized. World Size: 2
[INFO] Device 0 selected (Total devices: 8)
[INFO] Device 1 selected (Total devices: 8)
[INFO] HCCL Comm Initialized
[INFO] Buffers allocated and initialized
[INFO] Starting HcclAllGatherCustom...
[INFO] HcclAllGatherCustom completed and synchronized
[INFO] Test Passed!
```
