# Point-to-Point Communication - HcclBatchSendRecv (Ring)

## Sample Description

This sample demonstrates how to implement ring-based point-to-point communication using the `HcclBatchSendRecv()` interface. It covers the following features:

- Device detection: Query the number of available devices using the `aclrtGetDeviceCount()` interface.
- Use rank0 as the root node and generate rootinfo identification information using the `HcclGetRootInfo()` interface.

  > The rootinfo identification information mainly includes the Device IP, Device ID, and so on. This information must be broadcast to all ranks in the cluster to initialize the communication domain.

- In each thread, initialize the communication domain using the `HcclCommInitRootInfo()` interface based on the rootinfo identification information.
- Call the `HcclBatchSendRecv()` interface to send data to the next node and receive data from the previous node. Print the results.

## Directory Structure

```text
|-- main.cc                 # Sample source file
|-- Makefile                # Compilation and build configuration file
|-- batch_send_recv_ring    # Compiled executable file
```

## Environment Preparation

### Environment Requirements

This sample supports the following products in a single-server N-card configuration (N >= 2):

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>
- <term>Atlas A3 Training Series Products</term> / <term>Atlas A3 Inference Series Products</term>
- <term>Atlas A2 Training Series Products</term>
- <term>Atlas Training Series Products</term>

### Setting Environment Variables

```bash
# Set CANN environment variables. The following uses the root user default installation path as an example.
source /usr/local/Ascend/cann/set_env.sh
```

## Compiling and Running the Sample

Run the following commands in the sample code directory:

```bash
make
make test
```

> Note: You can set the `HCCL_OP_EXPANSION_MODE` environment variable to configure the task orchestration expansion location of communication algorithms. For the supported ranges for different product models, see the usage instructions for this environment variable in the [Environment Variable List](https://hiascend.com/document/redirect/CannCommunityEnvRef).
>
> ```bash
> # Set the orchestration expansion location of communication algorithms to the AI CPU on the Device side. The Device side automatically selects the corresponding scheduler based on the hardware model.
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## Sample Output

The `sendBuf` content on each node is initialized to the Device ID. Data is sent to the next node and received from the previous node. Therefore, each node receives the Device ID of the previous node.

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
