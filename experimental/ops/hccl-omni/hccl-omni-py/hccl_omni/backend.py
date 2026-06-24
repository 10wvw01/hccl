'''
Backend interface for HCCL-OMNI framework.

Handles tensor allocation, op_param serialization, and dispatch to hccl_omni_run.
Counts/displs are encoded in the op_param blob and read by the C++ implementation.
'''

import torch
import torch.distributed as dist

from .op_param import serialize_op_param
from .hccl.ops.comm_context import create_comm_context
from .hccl.ops.hccl_omni_run import hccl_omni_run


# send_type / recv_type = -1 means "infer from tensor dtype" in the C++ extension.
# The extension maps at::ScalarType -> HcclDataType internally when send_type >= 0,
# so passing raw HCCL enum values is WRONG. Use HCCL_DTYPE_AUTO to let the extension
# auto-detect from the tensor's scalar_type.
HCCL_DTYPE_AUTO = -1


def _allocate_buffers(
    rank: int,
    world_size: int,
    send_counts: list[int],
    recv_counts: list[int],
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Allocate send/recv data buffers for AlltoAllV.

    Counts/displs are encoded in the op_param blob — no separate tensors needed.

    Args:
        rank: Current rank.
        world_size: Total ranks.
        send_counts: Per-destination element counts.
        recv_counts: Per-source element counts.
        dtype: Data buffer element type.

    Returns:
        (send_buf, recv_buf) — both on NPU device.
    """
    npu_device = f"npu:{rank}"

    send_total = sum(send_counts)
    recv_total = sum(recv_counts)

    send_buf = torch.full((send_total,), float(rank + 1), dtype=dtype, device=npu_device)
    recv_buf = torch.zeros(recv_total, dtype=dtype, device=npu_device)

    return send_buf, recv_buf


def invoke_backend(op_param: dict, file_path: str):
    '''
    Invoke the HCCL backend with serialized operator parameters and instruction file.

    Parameters:
        op_param: Dictionary containing operator parameters. Must include:
            - op_name: Operator type (OpName enum or int)
          May include:
            - send_counts: List of per-destination element counts
            - recv_counts: List of per-source element counts
            - sdispls: List of send displacements
            - rdispls: List of receive displacements
            - data_count: Total data count
            - reduce_op: Reduction operation type
        file_path: Path to the rank-specific binary instruction file.

    The function serializes op_param into a binary blob (which includes
    counts/displs), allocates data buffers, then calls hccl_omni_run.
    Counts/displs are read from the blob by the C++ implementation.
    '''
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Extract counts from op_param (zero-padded to 128 in serialization,
    # but we only need the first world_size entries for buffer allocation)
    raw_send_counts = op_param.get('send_counts', [0] * world_size)
    raw_recv_counts = op_param.get('recv_counts', [0] * world_size)

    # Trim to world_size entries (serialization pads to 128, but actual
    # count is world_size)
    send_counts = list(raw_send_counts[:world_size])
    recv_counts = list(raw_recv_counts[:world_size])

    # Ensure all counts are non-negative integers
    send_counts = [max(0, int(c)) for c in send_counts]
    recv_counts = [max(0, int(c)) for c in recv_counts]

    # Determine dtype from op_param or default to float32
    dtype = op_param.get('dtype', torch.float32)
    if isinstance(dtype, str):
        dtype_map = {
            'fp32': torch.float32, 'float32': torch.float32,
            'fp16': torch.float16, 'float16': torch.float16,
            'int32': torch.int32, 'int64': torch.int64,
            'int8': torch.int8, 'bfp16': torch.bfloat16,
        }
        dtype = dtype_map.get(dtype, torch.float32)

    # Allocate data buffers (no separate counts/displs tensors needed)
    send_buf, recv_buf = _allocate_buffers(rank, world_size, send_counts, recv_counts, dtype)

    # Serialize op_param to binary block (includes counts/displs, stays on CPU)
    serialized = serialize_op_param(op_param)
    op_param_tensor = torch.tensor(list(serialized), dtype=torch.uint8)

    # Get HCCL comm handle
    group = dist.group.WORLD
    if group is None:
        group = dist.new_group(list(range(world_size)))

    mgr = create_comm_context(group)
    mgr.create_context()
    comm_handle = mgr.comm_handle

    # send_type/recv_type=HCCL_DTYPE_AUTO: C++ extension infers from tensor dtype
    # Parameter order matches C++ HcclOmniRun API:
    #   sendBuf, recvBuf, sendType, recvType, xmlPath, opParam, comm, stream
    result = hccl_omni_run(
        send_buf, recv_buf,
        HCCL_DTYPE_AUTO, HCCL_DTYPE_AUTO,
        file_path, op_param_tensor,
        comm_handle,
        synchronize=True,
        world_size=world_size,
        validate=False,
    )

    return result
