"""
Operator parameter binary serialization for HCCL-OMNI framework.

Binary Layout (little-endian, bit-addressed, 32832 bits = 4104 bytes total):

| Field        | Offset (bits) | Size (bits) | Type       | Notes                          |
|--------------|---------------|-------------|------------|--------------------------------|
| op_name      | 0             | 4           | uint4      | Enum: Allgather=0, ReduceScatter=1, Allreduce=2, Alltoall=3, Alltoallv=4 |
| reduce_op    | 4             | 4           | uint4      | Enum: SUM=0, MAX=1, MIN=2, PROD=3 |
| reserved     | 8             | 24          | —          | Zero-filled                    |
| data_count   | 32            | 32          | uint32     | 1~4 GB; sendCount/recvCount/total |
| send_counts  | 64            | 8192        | int64[128] | 128 x 64 bits, little-endian   |
| recv_counts  | 8256          | 8192        | int64[128] | 128 x 64 bits, little-endian   |
| sdispls      | 16448         | 8192        | int64[128] | 128 x 64 bits, little-endian   |
| rdispls      | 24640         | 8192        | int64[128] | 128 x 64 bits, little-endian   |
| **total**    | —             | 32832 bits (4104 bytes) | — | —                    |
"""

from enum import IntEnum

import torch


class OpName(IntEnum):
    """HCCL communication operator types."""
    Allgather = 0
    ReduceScatter = 1
    Allreduce = 2
    Alltoall = 3
    Alltoallv = 4


class ReduceOp(IntEnum):
    """Reduction operations for collective communication."""
    SUM = 0
    MAX = 1
    MIN = 2
    PROD = 3


# Total size of the serialized op_param block in bytes.
OP_PARAM_SIZE = 4104

# Maximum number of elements per array field.
MAX_ARRAY_LENGTH = 128

# Bit offsets per binary layout (all units in bits).
_OFFSET_OP_NAME = 0
_OFFSET_REDUCE_OP = 4
_OFFSET_DATA_COUNT = 32
_OFFSET_SEND_COUNTS = 64
_OFFSET_RECV_COUNTS = 64 + 128 * 64   # = 8256
_OFFSET_SDISPLS = 8256 + 128 * 64     # = 16448
_OFFSET_RDISPLS = 16448 + 128 * 64    # = 24640
_TOTAL_BITS = 64 + 4 * 128 * 64       # = 32832


def _set_bits(value, width, offset, target):
    """Set width bits of value at bit offset in target int."""
    mask = ((1 << width) - 1) << offset
    cleared = target & ~mask
    return cleared | ((value << offset) & mask)


def _normalize_enum(value, enum_cls):
    """Convert enum member to its integer value; pass integers through."""
    if value is None:
        return 0
    if isinstance(value, enum_cls):
        return int(value)
    return int(value)


def _pad_array(data, max_len=MAX_ARRAY_LENGTH):
    """Validate array length and pad with zeros to max_len.

    Raises ValueError if data length exceeds max_len.
    """
    if data is None:
        return [0] * max_len
    arr = list(data)
    if len(arr) > max_len:
        raise ValueError(
            f"Array length {len(arr)} exceeds maximum of {max_len}"
        )
    return arr + [0] * (max_len - len(arr))


def _write_int64_array(target, offset, values):
    """Write an array of int64 values into target at the given bit offset.

    Each value is masked to 64 bits for unsigned bit-field operations;
    the resulting byte representation preserves two's complement semantics.
    """
    result = target
    for i, val in enumerate(values):
        masked = val & 0xFFFFFFFFFFFFFFFF
        result = _set_bits(masked, 64, offset + i * 64, result)
    return result


def serialize_op_param(op_param):
    """Serialize an operator parameter dictionary to a fixed-layout binary block.

    The returned bytes object is always exactly OP_PARAM_SIZE (4104) bytes.

    Args:
        op_param: Dictionary containing operator parameters. Required key:
            - op_name: Operator type (int or OpName enum).
          Optional keys (missing scalars zero-filled, missing arrays all-zero):
            - reduce_op: Reduction type (int or ReduceOp enum).
            - data_count: Data count (int).
            - send_counts: List of up to 128 integers.
            - recv_counts: List of up to 128 integers.
            - sdispls: List of up to 128 integers.
            - rdispls: List of up to 128 integers.

    Returns:
        bytes: Exactly 4104 bytes representing the serialized parameters.

    Raises:
        ValueError: If op_name is missing or an array exceeds 128 elements.
        TypeError: If op_param is not a dict.
    """
    if not isinstance(op_param, dict):
        raise TypeError(f"op_param must be a dict, got {type(op_param).__name__}")

    if 'op_name' not in op_param:
        raise ValueError("op_name is required in op_param")

    op_name_val = _normalize_enum(op_param['op_name'], OpName)
    reduce_op_val = _normalize_enum(op_param.get('reduce_op', 0), ReduceOp)
    data_count_val = int(op_param.get('data_count', 0))

    send_counts = _pad_array(op_param.get('send_counts'))
    recv_counts = _pad_array(op_param.get('recv_counts'))
    sdispls = _pad_array(op_param.get('sdispls'))
    rdispls = _pad_array(op_param.get('rdispls'))

    # Build serialized block as a big int for bit-field packing.
    result = 0

    # op_name: 4 bits at bit offset 0
    result = _set_bits(op_name_val & 0xF, 4, _OFFSET_OP_NAME, result)
    # reduce_op: 4 bits at bit offset 4
    result = _set_bits(reduce_op_val & 0xF, 4, _OFFSET_REDUCE_OP, result)
    # reserved: 24 bits at bit offset 8 (stays zero)
    # data_count: 32 bits at bit offset 32
    result = _set_bits(data_count_val & 0xFFFFFFFF, 32, _OFFSET_DATA_COUNT, result)

    # Arrays: each 128 × 64 bits, padded to 128 elements.
    result = _write_int64_array(result, _OFFSET_SEND_COUNTS, send_counts)
    result = _write_int64_array(result, _OFFSET_RECV_COUNTS, recv_counts)
    result = _write_int64_array(result, _OFFSET_SDISPLS, sdispls)
    result = _write_int64_array(result, _OFFSET_RDISPLS, rdispls)

    return result.to_bytes(OP_PARAM_SIZE, 'little')


def _get_elem_count(t) -> int:
    """Extract element count: calls numel() if available, otherwise treat as int."""
    if hasattr(t, 'numel'):
        return t.numel()
    return int(t)


def _cumsum_zero_prefix(arr: list[int]) -> list[int]:
    """Compute cumulative-sum offsets with a leading zero.

    Example: [3, 5, 2] -> [0, 3, 8]
    """
    offsets = []
    acc = 0
    for v in arr:
        offsets.append(acc)
        acc += v
    return offsets


def build_op_param(op_param: dict, world_size: int) -> dict:
    """Convert a high-level op_param dict to the low-level form for serialize_op_param.

    High-level fields:
        - op_name: OpName or int (required)
        - input: torch.Tensor (required)
        - output: torch.Tensor or list[torch.Tensor] for Allgather (required)
        - reduce_op: ReduceOp or int (optional, default SUM)
        - input_split_sizes: list[int] | None (ReduceScatter / Alltoall)
        - output_split_sizes: list[int] | None (Alltoall)

    Returns a dict with keys:
        op_name, reduce_op, data_count, send_counts, recv_counts, sdispls, rdispls
    """
    if not isinstance(op_param, dict):
        raise TypeError(f"op_param must be a dict, got {type(op_param).__name__}")

    if 'op_name' not in op_param:
        raise ValueError("op_name is required in op_param")
    if 'input' not in op_param:
        raise ValueError("input is required in op_param")
    if 'output' not in op_param:
        raise ValueError("output is required in op_param")

    op_name = _normalize_enum(op_param['op_name'], OpName)
    reduce_op = _normalize_enum(op_param.get('reduce_op', 0), ReduceOp)

    input_tensor = op_param['input']
    output = op_param['output']
    input_size = _get_elem_count(input_tensor)

    input_split_sizes = op_param.get('input_split_sizes')
    output_split_sizes = op_param.get('output_split_sizes')

    # ------------------------------------------------------------------
    # Allgather
    # ------------------------------------------------------------------
    if op_name == OpName.Allgather:
        if not isinstance(output, list):
            raise ValueError(
                f"Allgather output must be a list of tensors, got {type(output).__name__}"
            )
        if len(output) != world_size:
            raise ValueError(
                f"Allgather output list length ({len(output)}) must equal world_size ({world_size})"
            )
        output_sizes = [t.numel() if hasattr(t, 'numel') else int(t) for t in output]
        send_counts = [input_size] * world_size
        recv_counts = output_sizes
        sdispls = [0] * world_size
        rdispls = _cumsum_zero_prefix(recv_counts)

    # ------------------------------------------------------------------
    # ReduceScatter
    # ------------------------------------------------------------------
    elif op_name == OpName.ReduceScatter:
        if input_split_sizes is None:
            if input_size % world_size != 0:
                raise ValueError(
                    f"ReduceScatter input size ({input_size}) must be divisible "
                    f"by world_size ({world_size}) when input_split_sizes is not provided"
                )
            input_split_sizes = [input_size // world_size] * world_size
        else:
            input_split_sizes = [int(s) for s in input_split_sizes]
            if len(input_split_sizes) != world_size:
                raise ValueError(
                    f"input_split_sizes length ({len(input_split_sizes)}) "
                    f"must equal world_size ({world_size})"
                )
        output_size = _get_elem_count(output)

        send_counts = input_split_sizes
        recv_counts = [output_size]
        sdispls = _cumsum_zero_prefix(send_counts)
        rdispls = [0]

    # ------------------------------------------------------------------
    # Allreduce
    # ------------------------------------------------------------------
    elif op_name == OpName.Allreduce:
        output_size = _get_elem_count(output)
        if input_size != output_size:
            raise ValueError(
                f"Allreduce input size ({input_size}) must equal output size ({output_size})"
            )
        send_counts = [input_size] * world_size
        recv_counts = [output_size]
        sdispls = [0] * world_size
        rdispls = [0]

    # ------------------------------------------------------------------
    # Alltoall / Alltoallv
    # ------------------------------------------------------------------
    elif op_name == OpName.Alltoall:
        if input_split_sizes is None:
            if input_size % world_size != 0:
                raise ValueError(
                    f"Alltoall input size ({input_size}) must be divisible "
                    f"by world_size ({world_size}) when input_split_sizes is not provided"
                )
            input_split_sizes = [input_size // world_size] * world_size
        else:
            input_split_sizes = [int(s) for s in input_split_sizes]
            if len(input_split_sizes) != world_size:
                raise ValueError(
                    f"input_split_sizes length ({len(input_split_sizes)}) "
                    f"must equal world_size ({world_size})"
                )

        if output_split_sizes is None:
            output_split_sizes = list(input_split_sizes)
        else:
            output_split_sizes = [int(s) for s in output_split_sizes]
            if len(output_split_sizes) != world_size:
                raise ValueError(
                    f"output_split_sizes length ({len(output_split_sizes)}) "
                    f"must equal world_size ({world_size})"
                )

        output_size = _get_elem_count(output)

        if sum(input_split_sizes) != input_size:
            raise ValueError(
                f"Alltoall sum(input_split_sizes) ({sum(input_split_sizes)}) "
                f"must equal input size ({input_size})"
            )
        if sum(output_split_sizes) != output_size:
            raise ValueError(
                f"Alltoall sum(output_split_sizes) ({sum(output_split_sizes)}) "
                f"must equal output size ({output_size})"
            )

        send_counts = input_split_sizes
        recv_counts = output_split_sizes
        sdispls = _cumsum_zero_prefix(send_counts)
        rdispls = _cumsum_zero_prefix(recv_counts)

    # ------------------------------------------------------------------
    # Alltoallv (full explicit counts)
    # ------------------------------------------------------------------
    elif op_name == OpName.Alltoallv:
        # Alltoallv requires explicit counts/displs; use them directly
        send_counts_raw = op_param.get('send_counts', [])
        recv_counts_raw = op_param.get('recv_counts', [])
        sdispls_raw = op_param.get('sdispls', [])
        rdispls_raw = op_param.get('rdispls', [])

        send_counts = [int(c) for c in send_counts_raw]
        recv_counts = [int(c) for c in recv_counts_raw]
        sdispls = [int(d) for d in sdispls_raw]
        rdispls = [int(d) for d in rdispls_raw]

        if sum(send_counts) != input_size:
            raise ValueError(
                f"Alltoallv sum(send_counts) ({sum(send_counts)}) "
                f"must equal input size ({input_size})"
            )
        output_size = _get_elem_count(output)
        if sum(recv_counts) != output_size:
            raise ValueError(
                f"Alltoallv sum(recv_counts) ({sum(recv_counts)}) "
                f"must equal output size ({output_size})"
            )

    else:
        raise ValueError(f"Unknown op_name: {op_name}")

    data_count = sum(send_counts)

    return {
        'op_name': op_name,
        'reduce_op': reduce_op,
        'data_count': data_count,
        'send_counts': send_counts,
        'recv_counts': recv_counts,
        'sdispls': sdispls,
        'rdispls': rdispls,
    }
