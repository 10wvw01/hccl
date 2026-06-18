# Collective communication - ReduceScatter

## Sample

This sample demonstrates how to use the `HcclReduceScatter()` API to perform the ReduceScatter operation. It covers the following functions:

- Call `aclrtGetDeviceCount()` to detect devices and query the number of available devices.
- Call `HcclGetRootInfo()` and use `rank 0` as the root rank to generate the rootinfo identifier.

  > The rootinfo identifier contains the device IP address and device ID. This information must be broadcast to all ranks in the cluster to initialize the communicator.

- In each thread, call `HcclCommInitRootInfo()` to initialize the communicator based on the rootinfo identifier.
- Call `HcclReduceScatter()` to sum the input data of all ranks, evenly distribute the result to each rank according to `rank_id`, and display the result.

## Directory Structure

```
├── main.cc           # Sample source file
├── Makefile          # Build configuration file
└── reduce_scatter    # Compiled executable file
```

## Environment Setup

### Prerequisites

This sample supports the following products in a single-node N-card configuration (N ≥ 2):

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>
- <term>Atlas A3 training products </term> / <term>Atlas A3 inference products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas training products</term> and <term>Atlas inference products</term>

### Configure Environment Variables

```bash
# Set CANN environment variables. (Use the root user's default installation path as an example.)
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

Each rank is initialized with data ranging from 0 to 7. After the `ReduceScatter` operation, each rank outputs the sum of the corresponding data across all ranks (sum across 8 ranks).

```text
Found 8 NPU device(s) available
rankId: 0, output: [ 0 ]
rankId: 1, output: [ 8 ]
rankId: 2, output: [ 16 ]
rankId: 3, output: [ 24 ]
rankId: 4, output: [ 32 ]
rankId: 5, output: [ 40 ]
rankId: 6, output: [ 48 ]
rankId: 7, output: [ 56 ]
```
