# Collective Communication - Reduce

## Sample

This sample demonstrates how to use the `HcclReduce()` API to perform the Reduce operation. It covers the following functions:

- Call `aclrtGetDeviceCount()` to detect devices and query the number of available devices.
- Call `HcclGetRootInfo()` and use `rank 0` as the root rank to generate the rootinfo identifier.

  > The rootinfo identifier contains the device IP address and device ID. This information must be broadcast to all ranks in the cluster to initialize the communicator.

- In each thread, call `HcclCommInitRootInfo()` to initialize the communicator based on the rootinfo identifier.
- Call the `HcclReduce()` API to sum the input data of all ranks, send the result to the root node, and display the result.

## Directory Structure

```text
├── main.cc   # Sample source file
├── Makefile  # Build configuration file
└── reduce    # Compiled executable file
```

## Environment Setup

### Prerequisites

This sample supports the following products in a single-node N-card configuration (N ≥ 2):

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>
- <term>Atlas A3 training products </term> / <term>Atlas A3 inference products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas training products</term>

### Configure Environment Variables

```bash
# Set CANN environment variables (use the root user's default installation path as an example)
source /usr/local/Ascend/cann/set_env.sh
```

## Build and Run the Sample

Run the following commands in the sample code directory:

```bash
make
make test
```

> Note: You can set the `HCCL_OP_EXPANSION_MODE` environment variable to configure the expansion mode of the communication operator. Supported ranges vary by product model. For details, see the usage of this environment variable in [Environment Variable List](https://hiascend.com/document/redirect/CannCommunityEnvRef).
>
> ```bash
> # Sets the communication operator expansion mode to AICPU communication engine.
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## Output

Each rank is initialized with data ranging from 0 to 7. After the Reduce operation, the root node outputs the sum of the corresponding data across all ranks (sum across 8 ranks).

```
Found 8 NPU device(s) available
rankId: 0, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 1, output: [ 0 0 0 0 0 0 0 0 ]
rankId: 2, output: [ 0 0 0 0 0 0 0 0 ]
rankId: 3, output: [ 0 0 0 0 0 0 0 0 ]
rankId: 4, output: [ 0 0 0 0 0 0 0 0 ]
rankId: 5, output: [ 0 0 0 0 0 0 0 0 ]
rankId: 6, output: [ 0 0 0 0 0 0 0 0 ]
rankId: 7, output: [ 0 0 0 0 0 0 0 0 ]
```
