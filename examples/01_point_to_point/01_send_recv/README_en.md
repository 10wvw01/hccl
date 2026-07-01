# Point-to-Point Communication - HcclSend and HcclRecv (Basic Send and Receive)

## Sample Description

This sample demonstrates how to implement point-to-point communication using the `HcclSend()` and `HcclRecv()` interfaces. It covers the following features:

- Device detection: Query the number of available devices using the `aclrtGetDeviceCount()` interface.
- Use rank0 as the root node and generate rootinfo identification information using the `HcclGetRootInfo()` interface.

  > The rootinfo identification information mainly includes the Device IP, Device ID, and so on. This information must be broadcast to all ranks in the cluster to initialize the communication domain.

- In each thread, initialize the communication domain using the `HcclCommInitRootInfo()` interface based on the rootinfo identification information.
- Call the `HcclSend()` and `HcclRecv()` interfaces. Cards 0, 2, 4, and 6 send data, and cards 1, 3, 5, and 7 receive data. Print the received results.

## Directory Structure

```text
|-- main.cc      # Sample source file
|-- Makefile     # Compilation and build configuration file
|-- send_recv    # Compiled executable file
```

## Environment Preparation

### Environment Requirements

This sample supports the following products in a single-server N-card configuration (N >= 2, even number):

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

> Note: You can set the `HCCL_OP_EXPANSION_MODE` environment variable to configure the expansion mode of communication operators. For the supported ranges for different product models, see the usage instructions for this environment variable in the [Environment Variable List](https://hiascend.com/document/redirect/CannCommunityEnvRef).
>
> ```bash
> # Set the communication operator expansion mode to the AI CPU communication engine
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## Sample Output

The `sendBuf` content on even-numbered nodes is initialized to the Device ID. The data is then sent to the next odd-numbered node. Therefore, each odd-numbered node receives the Device ID of the previous node.

```text
Found 8 NPU device(s) available
rankId: 1, output: [ 0 0 0 0 0 0 0 0 ]
rankId: 3, output: [ 2 2 2 2 2 2 2 2 ]
rankId: 5, output: [ 4 4 4 4 4 4 4 4 ]
rankId: 7, output: [ 6 6 6 6 6 6 6 6 ]
```
