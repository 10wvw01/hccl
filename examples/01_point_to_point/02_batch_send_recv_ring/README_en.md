# Point-to-Point Communication – HcclBatchSendRecv (Ring Implementation)

## Sample

This sample demonstrates how to use the `HcclBatchSendRecv()` API to implement point‑to‑point communicationin a ring topology. It covers the following functions:

- Call `aclrtGetDeviceCount()` to detect devices and query the number of available devices.
- Call `HcclGetRootInfo()` and use `rank 0` as the root rank to generate the rootinfo identifier.

  > The rootinfo identifier contains the device IP address and device ID. This information must be broadcast to all ranks in the cluster to initialize the communicator.

- In each thread, call `HcclCommInitRootInfo()` to initialize the communicator based on the rootinfo identifier.
- Call the `HcclBatchSendRecv()` API to send data to the next node while receiving data from the previous node, and display the result.

## Directory Structure

```text
├── main.cc                 # Sample source file
├── Makefile                # Build configuration file
└── batch_send_recv_ring    # Compiled executable file
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

> Note: You can set the `HCCL_OP_EXPANSION_MODE` environment variable to configure the task orchestration and expansion location of the communication algorithm.  Supported ranges vary by product model. For details, see the usage of the environment variable in [Environment Variable List](https://hiascend.com/document/redirect/CannCommunityEnvRef).
>
> ```bash
> # Set the communication algorithm expansion location to the AI CPU on the device; the device automatically selects the appropriate scheduler based on the hardware model
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## Output

Each node initializes its `sendBuf` with its own Device ID, sends the data to the next node, and receives data from the previous node. Therefore, each node receives the Device ID of the previous node.

```text
Found 8 NPU device(s) available
rankId: 0, output: [ 7 7 7 7 7 7 7 7 ]
rankId: 1, output: [ 0 0 0 0 0 0 0 0 ]
rankId: 2, output: [ 1 1 1 1 1 1 1 1 ]
rankId: 3, output: [ 2 2 2 2 2 2 2 2 ]
rankId: 4, output: [ 3 3 3 3 3 3 3 3 ]
rankId: 5, output: [ 4 4 4 4 4 4 4 4 ]
rankId: 6, output: [ 5 5 5 5 5 5 5 5 ]
rankId: 7, output: [ 6 6 6 6 6 6 6 6 ]
```
