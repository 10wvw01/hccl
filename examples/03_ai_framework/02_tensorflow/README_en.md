# Executing the AllReduce Operation Using TensorFlow

## Sample

This sample demonstrates how to use the `TensorFlow` API to perform an AllReduce​ operation. It covers the following functions:

- Use the `ranktable.json` configuration file to initialize the communicator.

## Environment Setup

### Prerequisites

This sample supports the following products in a single-node eight-card configuration:

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>
- <term>Atlas A3 training products </term> / <term>Atlas A3 inference products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas training products</term> / <term>Atlas inference products</term>

Note: This sample code is developed based on TensorFlow 1.x​ and is not compatible with TensorFlow 2.x. TensorFlow 1.15.0 is recommended.
### Configure Environment Variables

```bash
# Set CANN environment variables (use the root user's default installation path as an example)
source /usr/local/Ascend/cann/set_env.sh

# Sets the path to the rank_table.json configuration file
export RANK_TABLE_FILE=ranktable.json
```

## Run the Sample

```bash
bash run_tensorflow.sh
```

> Note: You can set the  `HCCL_OP_EXPANSION_MODE` environment variable to configure the expansion mode of the communication operator. Supported ranges vary by product model. For details, see the usage of this environment variable in [Environment Variable List](https://hiascend.com/document/redirect/CannCommunityEnvRef).
>
> ```bash
> # Sets the communication operator expansion mode to AICPU communication engine.
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## Output

Each rank is initialized with data ranging from 0 to 7. After the `AllReduce` operation, each rank outputs the sum of the corresponding data across all ranks (sum across 8 ranks).

```
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:0 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:1 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:2 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:3 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:4 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:5 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:6 tensorflow hccl test success
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:7 tensorflow hccl test success
```
