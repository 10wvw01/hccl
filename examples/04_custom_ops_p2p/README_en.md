# Custom Communication Operators - Point-to-Point Communication

## Sample

This sample demonstrates how to use the HCCL communication programming API to develop Send/Recv point-to-point communication operators. It covers the following functions:

1. Implements point-to-point communication operators using the AICPU communication engine.
2. Supports independent build and deployment of custom operator packages.

## Directory Structure

```text
├── CMakeLists.txt                      # Build configuration file
├── op_host/
│   ├── send.cc                         # Source file for implementing the HcclSendCustom operator
│   ├── recv.cc                         # Source file for implementing the HcclRecvCustom operator
│   ├── load_kernel.cc                  # Logic for loading AICPU kernels on the host
│   ├── launch_kernel.cc                # Logic for delivering AICPU kernels from the host
│   └── utils.cc                        # Tool module
├── op_kernel_aicpu/
│   ├── libp2p_aicpu_kernel.json        # AICPU kernel operator description file
│   ├── aicpu_kernel.cc                 #  AICPU kernel implementation logic
│   └── exec_op.cc                      # AICPU operator orchestration logic
├── inc/
│   ├── hccl_custom_p2p.h               # Header file for custom send/recv operator APIs
│   ├── common.h                        # Common header file
│   └── log.h                           # Log macro definitions
├── scripts/
│   └── hccl_custom_p2p_check_cfg.xml   # Signature configuration file
└── testcase/
    ├── main.cc                         # Sample implementation source file
     └── Makefile                        # Build configuration file
```

> The custom operator build project depends on the [cmake](../../cmake) configuration and build script [build.sh](../../build.sh) in the HCCL code repository.
> 
> - `cmake` contains the CMake configurations and MakeSelf packaging configurations.
> - `build.sh` is the project build entry.

## I. Environment Setup

### 1. Prerequisites

This sample supports the following products in a single-node N-card configuration (N ≥ 2):

- <term>Atlas A3 training products </term> / <term>Atlas A3 inference products</term>
- <term>Atlas A2 training products</term>

- gcc & g++ : 7.3.0 - 13.3.x
- cmake >= 3.16.0

### 2. Install the CANN Toolkit

Install the CANN Toolkit of the latest version. For details, see [Ascend Documentation - CANN Software Installation](https://hiascend.com/document/redirect/CannCommercialInstSoftware).

### 3. Configure Environment Variables

Set CANN environment variables.

```bash
# Installation in the default path (using the root user as an example)
source /usr/local/Ascend/cann/set_env.sh
# Installation in the default path (using a non-root user as an example)
source $HOME/Ascend/cann/set_env.sh
# Installation in a specified path
source ${install_path}/cann/set_env.sh
```

## II. Build the Custom Operator Package

The HCCL code repository provides a custom operator build and package project, which depends on the following files in the repository:

```text
├── build.sh                        # Entry point for the build project in the HCCL code repository root directory
├── CMakeLists.txt                  # Root-level build configuration file for the HCCL code repository
├── cmake/
│   ├── config.cmake                # CMake variable definitions
│   ├── func.cmake                  # CMake function definitions
│   ├── package.cmake               # Signature and packaging function definitions
│   └── makeself_custom.cmake       # MakeSelf packaging logic
└── scripts/
    ├── custom/install.sh           # Installation script for the custom operator package
    └── sign/add_header_sign.py     # Signature script for the AICPU operator package
```

First, you need to download the HCCL code repository, run `build.sh` from the repository root directory, specifying the custom operator project path using the `custom_ops_path` parameter.

```bash
# Download the HCCL code repository
git clone https://gitcode.com/cann/hccl.git

# build the custom operator package
bash build.sh --vendor=cust --ops=p2p --custom_ops_path=./examples/04_custom_ops_p2p
```

> Where:
> 
> - `--vendor` specifies the custom operator ID.
> - `--ops` specifies the custom operator name.
> - `--custom_ops_path` specifies the path to the custom operator project.

## III. Install the Custom Operator Package

The custom operator installation package is stored in the `./build_out` directory. You can install it using the `--install` parameter.

```bash
./build_out/cann-hccl_custom_p2p_linux-<arch>.run --install --install-path=<ascend_cann_path>
```

> Where:
> 
> - `<arch>` is the system architecture of the current build environment.
> - `<ascend_cann_path>` is an optional parameter that specifies the installation directory of the CANN package. If not specified, the path defaults to the value set by the `ASCEND_CUSTOM_OPP_PATH` or `ASCEND_OPP_PATH` environment variable.

The installation information of the custom operator package is as follows:

- Header file: `${ASCEND_HOME_PATH}/opp/vendors/cust/include/hccl_custom_p2p.h`
- Dynamic library: `${ASCEND_HOME_PATH}/opp/vendors/cust/lib64/libhccl_custom_p2p.so`
- AICPU operator description file: `${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/config/libp2p_aicpu_kernel.json`
- AICPU operator package: `${ASCEND_HOME_PATH}/opp/vendors/cust/aicpu/kernel/aicpu_hccl_custom_p2p.tar.gz`
- Installation script: `${ASCEND_HOME_PATH}/opp/vendors/cust/scripts/install.sh`

## IV. Execute the Custom operator

### 1. Disable the AICPU Operator Signature Verification Function

```bash
# Query the status of the AICPU operator user-defined signature verification capability
# False: The user-defined signature verification capability is disabled.
# True: The user-defined signature verification capability is enabled.
for i in {0..7}; do npu-smi info -t custom-op-secverify-enable -i $i; done

# Enable the AICPU operator user-defined signature verification capability
for i in {0..7}; do npu-smi set -t custom-op-secverify-enable -i $i -d 1; done

# Query the AICPU operator signature verification mode
# 0: Verification disabled
# 1: Huawei certificate (default)
# 2: User-defined certificate
# 3: Huawei certificate + user-defined certificate
# 4: open-source community certificate
# 5: Huawei certificate + open-source community certificate
# 6: User-defined certificate + open-source community certificate
# 7: Huawei certificate + user-defined certificate + open-source community certificate
for i in {0..7}; do npu-smi info -t custom-op-secverify-mode -i $i; done

# Disable the AICPU operator signature verification mode
for i in {0..7}; do npu-smi set -t custom-op-secverify-mode -i $i -d 0; done
```

### 2. Modify the AI CPU Trustlist

By default, AICPU loads only the packages in the trustlist. Custom AICPU operator packages developed by users must be added to this trustlist.

```bash
# Edit the configuration file (use the default installation path of the root user as an example)
vim /usr/local/Ascend/cann/conf/ascend_package_load.ini
```

Append the following content to the `ascend_package_load.ini` file:

```ini
name:aicpu_hccl_custom_p2p.tar.gz
install_path:2
optional:true
package_path:opp/vendors/cust/aicpu/kernel
load_as_per_soc:false
```

The fields are described as follows:

- `name`: name of the `.tar` package
- `install_path`: installation path on the device
- `optional`: defaults to `true`.
- `optional`: relative path of the `.tar` package in the CANN Toolkit package on the host
- `load_as_per_soc`: whether to load each chip type

### 3. Build the Sample

Run the following command in the `examples/04_custom_ops_p2p/testcase` directory:

```bash
# Build the sample
make
```

### 4. Run the Sample

```bash
# Run the sample using make
make test

# Or execute the binary directly
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH}
./send_recv
```

### 5. Output

Even-numbered nodes initialize their `sendBuf` with their own `DeviceId`, then send the data to the next odd-numbered node. Therefore, each odd-numbered node receives the `DeviceId` of the preceding even-numbered node.

```text
Found 8 NPU device(s) available
rankId: 1, output: [ 0 0 0 0 0 0 0 0 ]
rankId: 3, output: [ 2 2 2 2 2 2 2 2 ]
rankId: 5, output: [ 4 4 4 4 4 4 4 4 ]
rankId: 7, output: [ 6 6 6 6 6 6 6 6 ]
```
