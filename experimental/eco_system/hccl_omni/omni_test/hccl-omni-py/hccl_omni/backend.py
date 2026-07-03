'''
Backend interface for HCCL-OMNI framework.

Handles tensor extraction, op_param serialization, and dispatch to hccl_omni_run.
User-provided input/output tensors are used directly as send/recv buffers.
'''

import torch
import torch.distributed as dist

from .op_param import serialize_op_param, build_op_param, OpName, _normalize_enum
from .hccl.ops.comm_context import create_comm_context
from .hccl.ops.hccl_omni_run import hccl_omni_run


# send_type / recv_type = -1 means "infer from tensor dtype" in the C++ extension.
HCCL_DTYPE_AUTO = -1


def _extract_tensors(op_param: dict, op_name: int):
    """Extract send_buf and recv_buf from high-level op_param.

    For Allgather the output is a list of tensors which are concatenated
    into a single flat recv_buf.  After the backend call the flat result
    is copied back into each output tensor (see _scatter_allgather_result).
    Other operators use the output tensor directly — the backend writes
    results in-place.
    """
    input_tensor = op_param['input']
    output = op_param['output']

    if op_name == OpName.Allgather:
        recv_buf = torch.cat([t.reshape(-1) for t in output])
    else:
        recv_buf = output

    return input_tensor, recv_buf


def _scatter_allgather_result(flat_buf, output_list, recv_counts):
    """Copy flat Allgather result back into each output list tensor."""
    offset = 0
    for i, out_tensor in enumerate(output_list):
        count = recv_counts[i]
        chunk = flat_buf[offset:offset + count].reshape(out_tensor.shape)
        out_tensor.copy_(chunk)
        offset += count


def invoke_backend(op_param: dict, file_path: str):
    '''Invoke the HCCL backend with serialized operator parameters.

    Parameters:
        op_param: High-level dictionary.  Required keys:
            - op_name: OpName enum or int
            - input: torch.Tensor
            - output: torch.Tensor, or list[torch.Tensor] for Allgather
          Optional keys:
            - reduce_op: ReduceOp enum or int (default SUM)
            - input_split_sizes: list[int] (ReduceScatter / Alltoall)
            - output_split_sizes: list[int] (Alltoall)
        file_path: Path to the rank-specific binary instruction file.

    The function converts op_param to the low-level binary form via
    build_op_param(), uses the user-provided tensors as send/recv
    buffers, and calls hccl_omni_run.
    '''
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Convert to low-level dict (validates all required fields)
    op_param_low = build_op_param(op_param, world_size)
    op_name = op_param_low['op_name']
    send_buf, recv_buf = _extract_tensors(op_param, op_name)

    # Serialize op_param to binary block (stays on CPU)
    serialized = serialize_op_param(op_param_low)
    op_param_tensor = torch.tensor(list(serialized), dtype=torch.uint8)

    # HCCL comm handle
    group = dist.group.WORLD
    if group is None:
        group = dist.new_group(list(range(world_size)))

    mgr = create_comm_context(group)
    mgr.create_context()
    comm_handle = mgr.comm_handle

    result = hccl_omni_run(
        send_buf, recv_buf,
        HCCL_DTYPE_AUTO, HCCL_DTYPE_AUTO,
        file_path, op_param_tensor,
        comm_handle,
        synchronize=True,
        world_size=world_size,
    )

    # For Allgather, torch.cat created a new buffer — copy results back
    # into the user's output list tensors.
    if op_name == OpName.Allgather:
        _scatter_allgather_result(
            recv_buf,
            op_param['output'],
            op_param_low['recv_counts'],
        )

    return result
