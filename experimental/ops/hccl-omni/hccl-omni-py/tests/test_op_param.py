"""
Unit tests for hccl_omni.op_param binary serialization.

Validates the bit-addressed fixed-layout binary encoding: field bit offsets,
endianness, bit-field packing, error handling, and default zero-filling.

Binary layout (bit offsets):
  op_name      @ 0,   4 bits
  reduce_op    @ 4,   4 bits
  reserved     @ 8,   24 bits
  data_count   @ 32,  32 bits
  send_counts  @ 64,  8192 bits (128 × 64)
  recv_counts  @ 8256, 8192 bits
  sdispls      @ 16448, 8192 bits
  rdispls      @ 24640, 8192 bits
  total: 32832 bits = 4104 bytes
"""

import struct
import pytest

from hccl_omni.op_param import (
    serialize_op_param,
    OpName,
    ReduceOp,
    OP_PARAM_SIZE,
    MAX_ARRAY_LENGTH,
)


# Byte offsets derived from the bit-addressed layout (int64 arrays).
BYTE_OFS_DATA_COUNT = 4
BYTE_OFS_SEND_COUNTS = 8
BYTE_OFS_RECV_COUNTS = 8 + 128 * 8       # = 1032
BYTE_OFS_SDISPLS = 1032 + 128 * 8        # = 2056
BYTE_OFS_RDISPLS = 2056 + 128 * 8        # = 3080
ARRAY_BYTE_SIZE = 1024  # 128 × 8 bytes (int64)


def test_serialize_minimal():
    """Minimal dict produces exactly 4104 bytes with correct op_name in bits 0-3."""
    result = serialize_op_param({'op_name': 0})
    assert len(result) == OP_PARAM_SIZE
    # op_name occupies lower 4 bits of byte 0
    assert result[0] & 0x0F == 0


def test_serialize_all_fields():
    """Full dict with all fields; verify each field at correct offset."""
    op_param = {
        'op_name': OpName.Alltoallv,   # 4
        'reduce_op': ReduceOp.PROD,    # 3
        'data_count': 42,
        'send_counts': [1, 2, 3],
        'recv_counts': [4, 5],
        'sdispls': [0, 1, 2, 3],
        'rdispls': [10, 20],
    }
    result = serialize_op_param(op_param)
    assert len(result) == OP_PARAM_SIZE

    # op_name in bits 0-3, reduce_op in bits 4-7 of byte 0
    assert result[0] & 0x0F == 4        # Alltoallv
    assert (result[0] >> 4) & 0x0F == 3  # PROD

    # reserved: bits 8-31 (bytes 1-3) should be zeros
    assert result[1:4] == b'\x00' * 3

    # data_count at bytes 4-7 (32 bits, little-endian)
    assert struct.unpack_from('<I', result, BYTE_OFS_DATA_COUNT)[0] == 42

    # send_counts at bytes 8-1031 (int64 LE)
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS)[0] == 1
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + 8)[0] == 2
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + 16)[0] == 3

    # recv_counts at bytes 1032-2055
    assert struct.unpack_from('<q', result, BYTE_OFS_RECV_COUNTS)[0] == 4
    assert struct.unpack_from('<q', result, BYTE_OFS_RECV_COUNTS + 8)[0] == 5

    # sdispls at bytes 2056-3079
    assert struct.unpack_from('<q', result, BYTE_OFS_SDISPLS)[0] == 0
    assert struct.unpack_from('<q', result, BYTE_OFS_SDISPLS + 8)[0] == 1

    # rdispls at bytes 3080-4103
    assert struct.unpack_from('<q', result, BYTE_OFS_RDISPLS)[0] == 10
    assert struct.unpack_from('<q', result, BYTE_OFS_RDISPLS + 8)[0] == 20


def test_serialize_endianness():
    """Verify little-endian encoding for multi-byte fields."""
    op_param = {
        'op_name': 0xA,   # only low 4 bits matter
        'reduce_op': 0xB,
        'data_count': 0x12345678,
    }
    result = serialize_op_param(op_param)

    # op_name + reduce_op packed in byte 0
    assert result[0] == 0xBA  # reduce_op in high nibble, op_name in low nibble

    # data_count at bytes 4-7: little-endian => LSB first
    assert result[4] == 0x78
    assert result[5] == 0x56
    assert result[6] == 0x34
    assert result[7] == 0x12


def test_serialize_alignment():
    """Verify fields are packed contiguously with no unexpected padding."""
    op_param = {
        'op_name': 1,
        'reduce_op': 2,
        'data_count': 0xDEADBEEF,
        'send_counts': [100],
        'recv_counts': [200],
        'sdispls': [300],
        'rdispls': [400],
    }
    result = serialize_op_param(op_param)

    # Byte 0: op_name=1 (bits 0-3), reduce_op=2 (bits 4-7)
    assert result[0] == (2 << 4) | 1

    # data_count at bytes 4-7
    assert struct.unpack_from('<I', result, 4)[0] == 0xDEADBEEF

    # send_counts[0] at bytes 8-15 (int64 LE)
    assert struct.unpack_from('<q', result, 8)[0] == 100

    # recv_counts[0] at bytes 1032-1039
    assert struct.unpack_from('<q', result, BYTE_OFS_RECV_COUNTS)[0] == 200

    # sdispls[0] at bytes 2056-2063
    assert struct.unpack_from('<q', result, BYTE_OFS_SDISPLS)[0] == 300

    # rdispls[0] at bytes 3080-3087
    assert struct.unpack_from('<q', result, BYTE_OFS_RDISPLS)[0] == 400

    # Verify no gap between header (8 bytes) and send_counts (starts at byte 8)
    # Header = 1 byte (op_name+reduce_op) + 3 reserved + 4 data_count = 8 bytes
    assert BYTE_OFS_SEND_COUNTS == 8


def test_missing_op_name_raises():
    """Missing op_name raises ValueError with descriptive message."""
    with pytest.raises(ValueError, match='op_name'):
        serialize_op_param({})


def test_array_too_long_raises():
    """Array with more than 128 elements raises ValueError."""
    too_long = [1] * (MAX_ARRAY_LENGTH + 1)
    with pytest.raises(ValueError, match='exceeds maximum'):
        serialize_op_param({'op_name': 0, 'send_counts': too_long})

    # Test each array field
    for field in ['recv_counts', 'sdispls', 'rdispls']:
        with pytest.raises(ValueError, match='exceeds maximum'):
            serialize_op_param({'op_name': 0, field: too_long})


def test_missing_arrays_zero_filled():
    """Missing array fields produce 128 zeros each (int64 = 8 bytes per element)."""
    result = serialize_op_param({'op_name': 0})

    # send_counts at bytes 8-1031
    assert result[BYTE_OFS_SEND_COUNTS:BYTE_OFS_RECV_COUNTS] == b'\x00' * ARRAY_BYTE_SIZE

    # recv_counts at bytes 1032-2055
    assert result[BYTE_OFS_RECV_COUNTS:BYTE_OFS_SDISPLS] == b'\x00' * ARRAY_BYTE_SIZE

    # sdispls at bytes 2056-3079
    assert result[BYTE_OFS_SDISPLS:BYTE_OFS_RDISPLS] == b'\x00' * ARRAY_BYTE_SIZE

    # rdispls at bytes 3080-4103
    assert result[BYTE_OFS_RDISPLS:] == b'\x00' * ARRAY_BYTE_SIZE


def test_missing_scalars_zero_filled():
    """Missing scalar fields reduce_op and data_count are zero."""
    result = serialize_op_param({'op_name': 1})

    # reduce_op in bits 4-7 of byte 0
    assert (result[0] >> 4) & 0x0F == 0

    # data_count at bytes 4-7
    assert struct.unpack_from('<I', result, BYTE_OFS_DATA_COUNT)[0] == 0


def test_enum_input():
    """Accept OpName and ReduceOp enum values."""
    result = serialize_op_param({
        'op_name': OpName.Allreduce,   # 2
        'reduce_op': ReduceOp.MAX,     # 1
    })
    # op_name in bits 0-3
    assert result[0] & 0x0F == 2
    # reduce_op in bits 4-7
    assert (result[0] >> 4) & 0x0F == 1
    # data_count defaults to 0
    assert struct.unpack_from('<I', result, BYTE_OFS_DATA_COUNT)[0] == 0


def test_total_size_4104():
    """Verify total length is exactly 4104 bytes with no padding region."""
    result = serialize_op_param({
        'op_name': OpName.Allgather,
        'reduce_op': ReduceOp.SUM,
        'send_counts': [1] * 128,
        'recv_counts': [2] * 128,
        'sdispls': [3] * 128,
        'rdispls': [4] * 128,
    })
    # With int64[128] arrays, total = 8 + 4*1024 = 4104 bytes, no padding
    assert len(result) == 4104

    # Last array (rdispls) ends at byte 4104 — no trailing padding
    assert BYTE_OFS_RDISPLS + ARRAY_BYTE_SIZE == 4104


def test_data_count_32_bits():
    """Verify data_count occupies exactly 32 bits (4 bytes) at bit offset 32."""
    result = serialize_op_param({
        'op_name': 0,
        'data_count': 0x12345678,
    })
    data_count_region = result[BYTE_OFS_DATA_COUNT:BYTE_OFS_DATA_COUNT + 4]
    assert len(data_count_region) == 4
    val = struct.unpack_from('<I', result, BYTE_OFS_DATA_COUNT)[0]
    assert val == 0x12345678

    # Verify max uint32 value
    result2 = serialize_op_param({
        'op_name': 0,
        'data_count': 0xFFFFFFFF,
    })
    assert struct.unpack_from('<I', result2, BYTE_OFS_DATA_COUNT)[0] == 0xFFFFFFFF

    # Verify zero default
    result3 = serialize_op_param({'op_name': 0})
    assert struct.unpack_from('<I', result3, BYTE_OFS_DATA_COUNT)[0] == 0


def test_bitfield_packing():
    """Verify op_name (4b) + reduce_op (4b) + reserved (24b) pack into first 4 bytes."""
    result = serialize_op_param({
        'op_name': 3,       # Alltoall
        'reduce_op': 2,     # MIN
    })
    # First 4 bytes as uint32 LE:
    # op_name=3 at bits 0-3, reduce_op=2 at bits 4-7, reserved=0 at bits 8-31
    word0 = struct.unpack_from('<I', result, 0)[0]
    expected = (2 << 4) | 3  # reduce_op in high nibble, op_name in low nibble
    assert word0 == expected, f'Expected 0x{expected:08X}, got 0x{word0:08X}'

    # Verify bit-exact values
    assert result[0] & 0x0F == 3        # op_name bits 0-3
    assert (result[0] >> 4) & 0x0F == 2  # reduce_op bits 4-7
    assert result[1] == 0                # reserved byte 1
    assert result[2] == 0                # reserved byte 2
    assert result[3] == 0                # reserved byte 3


def test_type_error_on_non_dict():
    """Passing non-dict raises TypeError."""
    with pytest.raises(TypeError):
        serialize_op_param(None)
    with pytest.raises(TypeError):
        serialize_op_param('not a dict')


def test_negative_array_values():
    """Verify negative int64 values are correctly preserved in two's complement."""
    result = serialize_op_param({
        'op_name': 0,
        'send_counts': [-1, -9223372036854775808, 9223372036854775807],
    })
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS)[0] == -1
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + 8)[0] == -9223372036854775808
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + 16)[0] == 9223372036854775807


def test_exactly_128_array_elements():
    """Verify arrays with exactly 128 elements are accepted and fully written."""
    data = list(range(128))
    result = serialize_op_param({
        'op_name': 0,
        'send_counts': data,
    })
    assert len(result) == OP_PARAM_SIZE
    for i in range(128):
        val = struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + i * 8)[0]
        assert val == i, f'send_counts[{i}] = {val}, expected {i}'


def test_single_element_arrays():
    """Verify arrays with a single element are padded to 128 zeros after."""
    result = serialize_op_param({
        'op_name': 0,
        'send_counts': [42],
        'recv_counts': [99],
    })
    # First element
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS)[0] == 42
    assert struct.unpack_from('<q', result, BYTE_OFS_RECV_COUNTS)[0] == 99
    # Remaining 127 elements should be zero
    for i in range(1, 128):
        assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + i * 8)[0] == 0
        assert struct.unpack_from('<q', result, BYTE_OFS_RECV_COUNTS + i * 8)[0] == 0


def test_large_int64_values():
    """Verify int64 arrays can hold values beyond int32 range."""
    result = serialize_op_param({
        'op_name': 0,
        'send_counts': [2**31 + 100, 2**32 + 50],  # > INT32_MAX
    })
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS)[0] == 2**31 + 100
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + 8)[0] == 2**32 + 50
