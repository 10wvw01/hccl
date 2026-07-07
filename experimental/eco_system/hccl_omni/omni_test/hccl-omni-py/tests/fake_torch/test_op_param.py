"""
Unit tests for hccl_omni.op_param — binary serialization and high-level build_op_param.

Covers:
- serialize_op_param: bit-addressed binary encoding, endianness, padding, error handling
- build_op_param: high-level → low-level conversion with per-operator count/displ computation
"""

import struct
import pytest
from unittest.mock import Mock

from hccl_omni.op_param import (
    serialize_op_param,
    build_op_param,
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


# ============================================================================
# Helpers for build_op_param tests
# ============================================================================

def _mtensor(numel):
    """Create a minimal mock object that exposes numel()."""
    m = Mock()
    m.numel = Mock(return_value=int(numel))
    m.reshape = Mock(return_value=m)
    return m


# ============================================================================
# serialize_op_param tests
# ============================================================================

def test_serialize_minimal():
    result = serialize_op_param({'op_name': 0})
    assert len(result) == OP_PARAM_SIZE
    assert result[0] & 0x0F == 0


def test_serialize_all_fields():
    op_param = {
        'op_name': OpName.Alltoallv,
        'reduce_op': ReduceOp.PROD,
        'data_count': 42,
        'send_counts': [1, 2, 3],
        'recv_counts': [4, 5],
        'sdispls': [0, 1, 2, 3],
        'rdispls': [10, 20],
    }
    result = serialize_op_param(op_param)
    assert len(result) == OP_PARAM_SIZE
    assert result[0] & 0x0F == 4
    assert (result[0] >> 4) & 0x0F == 3
    assert result[1:4] == b'\x00' * 3
    assert struct.unpack_from('<I', result, BYTE_OFS_DATA_COUNT)[0] == 42
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS)[0] == 1
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + 8)[0] == 2
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + 16)[0] == 3
    assert struct.unpack_from('<q', result, BYTE_OFS_RECV_COUNTS)[0] == 4
    assert struct.unpack_from('<q', result, BYTE_OFS_RECV_COUNTS + 8)[0] == 5
    assert struct.unpack_from('<q', result, BYTE_OFS_SDISPLS)[0] == 0
    assert struct.unpack_from('<q', result, BYTE_OFS_SDISPLS + 8)[0] == 1
    assert struct.unpack_from('<q', result, BYTE_OFS_RDISPLS)[0] == 10
    assert struct.unpack_from('<q', result, BYTE_OFS_RDISPLS + 8)[0] == 20


def test_serialize_endianness():
    op_param = {'op_name': 0xA, 'reduce_op': 0xB, 'data_count': 0x12345678}
    result = serialize_op_param(op_param)
    assert result[0] == 0xBA
    assert result[4] == 0x78
    assert result[5] == 0x56
    assert result[6] == 0x34
    assert result[7] == 0x12


def test_serialize_alignment():
    op_param = {
        'op_name': 1, 'reduce_op': 2, 'data_count': 0xDEADBEEF,
        'send_counts': [100], 'recv_counts': [200],
        'sdispls': [300], 'rdispls': [400],
    }
    result = serialize_op_param(op_param)
    assert result[0] == (2 << 4) | 1
    assert struct.unpack_from('<I', result, 4)[0] == 0xDEADBEEF
    assert struct.unpack_from('<q', result, 8)[0] == 100
    assert struct.unpack_from('<q', result, BYTE_OFS_RECV_COUNTS)[0] == 200
    assert struct.unpack_from('<q', result, BYTE_OFS_SDISPLS)[0] == 300
    assert struct.unpack_from('<q', result, BYTE_OFS_RDISPLS)[0] == 400
    assert BYTE_OFS_SEND_COUNTS == 8


def test_missing_op_name_raises():
    with pytest.raises(ValueError, match='op_name'):
        serialize_op_param({})


def test_array_too_long_raises():
    too_long = [1] * (MAX_ARRAY_LENGTH + 1)
    with pytest.raises(ValueError, match='exceeds maximum'):
        serialize_op_param({'op_name': 0, 'send_counts': too_long})
    for field in ['recv_counts', 'sdispls', 'rdispls']:
        with pytest.raises(ValueError, match='exceeds maximum'):
            serialize_op_param({'op_name': 0, field: too_long})


def test_missing_arrays_zero_filled():
    result = serialize_op_param({'op_name': 0})
    assert result[BYTE_OFS_SEND_COUNTS:BYTE_OFS_RECV_COUNTS] == b'\x00' * ARRAY_BYTE_SIZE
    assert result[BYTE_OFS_RECV_COUNTS:BYTE_OFS_SDISPLS] == b'\x00' * ARRAY_BYTE_SIZE
    assert result[BYTE_OFS_SDISPLS:BYTE_OFS_RDISPLS] == b'\x00' * ARRAY_BYTE_SIZE
    assert result[BYTE_OFS_RDISPLS:] == b'\x00' * ARRAY_BYTE_SIZE


def test_missing_scalars_zero_filled():
    result = serialize_op_param({'op_name': 1})
    assert (result[0] >> 4) & 0x0F == 0
    assert struct.unpack_from('<I', result, BYTE_OFS_DATA_COUNT)[0] == 0


def test_enum_input():
    result = serialize_op_param({'op_name': OpName.Allreduce, 'reduce_op': ReduceOp.MAX})
    assert result[0] & 0x0F == 2
    assert (result[0] >> 4) & 0x0F == 1
    assert struct.unpack_from('<I', result, BYTE_OFS_DATA_COUNT)[0] == 0


def test_total_size_4104():
    result = serialize_op_param({
        'op_name': OpName.Allgather, 'reduce_op': ReduceOp.SUM,
        'send_counts': [1] * 128, 'recv_counts': [2] * 128,
        'sdispls': [3] * 128, 'rdispls': [4] * 128,
    })
    assert len(result) == 4104
    assert BYTE_OFS_RDISPLS + ARRAY_BYTE_SIZE == 4104


def test_data_count_32_bits():
    result = serialize_op_param({'op_name': 0, 'data_count': 0x12345678})
    assert struct.unpack_from('<I', result, BYTE_OFS_DATA_COUNT)[0] == 0x12345678

    result2 = serialize_op_param({'op_name': 0, 'data_count': 0xFFFFFFFF})
    assert struct.unpack_from('<I', result2, BYTE_OFS_DATA_COUNT)[0] == 0xFFFFFFFF

    result3 = serialize_op_param({'op_name': 0})
    assert struct.unpack_from('<I', result3, BYTE_OFS_DATA_COUNT)[0] == 0


def test_bitfield_packing():
    result = serialize_op_param({'op_name': 3, 'reduce_op': 2})
    word0 = struct.unpack_from('<I', result, 0)[0]
    expected = (2 << 4) | 3
    assert word0 == expected
    assert result[0] & 0x0F == 3
    assert (result[0] >> 4) & 0x0F == 2
    assert result[1] == 0
    assert result[2] == 0
    assert result[3] == 0


def test_type_error_on_non_dict():
    with pytest.raises(TypeError):
        serialize_op_param(None)
    with pytest.raises(TypeError):
        serialize_op_param('not a dict')


def test_negative_array_values():
    result = serialize_op_param({
        'op_name': 0,
        'send_counts': [-1, -9223372036854775808, 9223372036854775807],
    })
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS)[0] == -1
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + 8)[0] == -9223372036854775808
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + 16)[0] == 9223372036854775807


def test_exactly_128_array_elements():
    data = list(range(128))
    result = serialize_op_param({'op_name': 0, 'send_counts': data})
    assert len(result) == OP_PARAM_SIZE
    for i in range(128):
        val = struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + i * 8)[0]
        assert val == i


def test_single_element_arrays():
    result = serialize_op_param({
        'op_name': 0, 'send_counts': [42], 'recv_counts': [99],
    })
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS)[0] == 42
    assert struct.unpack_from('<q', result, BYTE_OFS_RECV_COUNTS)[0] == 99
    for i in range(1, 128):
        assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + i * 8)[0] == 0
        assert struct.unpack_from('<q', result, BYTE_OFS_RECV_COUNTS + i * 8)[0] == 0


def test_large_int64_values():
    result = serialize_op_param({
        'op_name': 0,
        'send_counts': [2**31 + 100, 2**32 + 50],
    })
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS)[0] == 2**31 + 100
    assert struct.unpack_from('<q', result, BYTE_OFS_SEND_COUNTS + 8)[0] == 2**32 + 50


# ============================================================================
# build_op_param tests
# ============================================================================

class TestBuildOpParam:

    # ---- Allgather ----
    def test_allgather_basic(self):
        w = 4
        low = build_op_param({
            'op_name': OpName.Allgather,
            'input': _mtensor(10),
            'output': [_mtensor(10) for _ in range(w)],
        }, world_size=w)

        assert low['op_name'] == OpName.Allgather
        assert low['send_counts'] == [10] * w
        assert low['recv_counts'] == [10] * w
        assert low['sdispls'] == [0] * w
        assert low['rdispls'] == [0, 10, 20, 30]
        assert low['data_count'] == 40

    def test_allgather_int_sizes(self):
        w = 2
        low = build_op_param({
            'op_name': 0,
            'input': 5,
            'output': [3, 3],
        }, world_size=w)
        assert low['send_counts'] == [5, 5]
        assert low['recv_counts'] == [5, 5]
        assert low['sdispls'] == [0, 0]
        assert low['rdispls'] == [0, 5]

    def test_allgather_output_not_list_raises(self):
        with pytest.raises(ValueError, match='list'):
            build_op_param({
                'op_name': OpName.Allgather, 'input': 10, 'output': 10,
            }, world_size=2)

    def test_allgather_output_wrong_length_raises(self):
        with pytest.raises(ValueError, match='world_size'):
            build_op_param({
                'op_name': OpName.Allgather, 'input': 10, 'output': [10, 10, 10],
            }, world_size=2)

    # ---- ReduceScatter ----
    def test_reduce_scatter_even_split(self):
        w = 4
        low = build_op_param({
            'op_name': OpName.ReduceScatter,
            'input': _mtensor(64),
            'output': _mtensor(16),
        }, world_size=w)

        assert low['send_counts'] == [16, 16, 16, 16]
        assert low['recv_counts'] == [16, 16, 16, 16]
        assert low['sdispls'] == [0, 16, 32, 48]
        assert low['rdispls'] == [0]
        assert low['data_count'] == 64

    def test_reduce_scatter_explicit_splits(self):
        w = 3
        low = build_op_param({
            'op_name': 1, 'input': 60, 'output': 20,
            'input_split_sizes': [10, 20, 30],
        }, world_size=w)
        assert low['send_counts'] == [10, 20, 30]
        assert low['sdispls'] == [0, 10, 30]
        assert low['recv_counts'] == [10, 20, 30]
        assert low['data_count'] == 0

    def test_reduce_scatter_uneven_raises(self):
        with pytest.raises(ValueError, match='divisible'):
            build_op_param({
                'op_name': OpName.ReduceScatter, 'input': 10, 'output': 5,
            }, world_size=3)

    # ---- Allreduce ----
    def test_allreduce_basic(self):
        w = 4
        low = build_op_param({
            'op_name': OpName.Allreduce,
            'input': _mtensor(100),
            'output': _mtensor(100),
        }, world_size=w)

        assert low['send_counts'] == [100] * w
        assert low['recv_counts'] == [100] * w
        assert low['sdispls'] == [0] * w
        assert low['rdispls'] == [0] * w
        assert low['data_count'] == 100

    def test_allreduce_size_mismatch_raises(self):
        with pytest.raises(ValueError, match='equal'):
            build_op_param({
                'op_name': OpName.Allreduce, 'input': 10, 'output': 20,
            }, world_size=2)

    # ---- Alltoall ----
    def test_alltoall_even_split(self):
        w = 4
        low = build_op_param({
            'op_name': OpName.Alltoall,
            'input': _mtensor(64),
            'output': _mtensor(64),
        }, world_size=w)

        assert low['send_counts'] == [16, 16, 16, 16]
        assert low['recv_counts'] == [16, 16, 16, 16]
        assert low['sdispls'] == [0, 16, 32, 48]
        assert low['rdispls'] == [0, 16, 32, 48]

    def test_alltoall_explicit_splits(self):
        w = 2
        low = build_op_param({
            'op_name': 3, 'input': 100, 'output': 80,
            'input_split_sizes': [60, 40],
            'output_split_sizes': [30, 50],
        }, world_size=w)

        assert low['send_counts'] == [60, 40]
        assert low['recv_counts'] == [30, 50]
        assert low['sdispls'] == [0, 60]
        assert low['rdispls'] == [0, 30]

    def test_alltoall_uneven_raises(self):
        with pytest.raises(ValueError, match='divisible'):
            build_op_param({
                'op_name': OpName.Alltoall, 'input': 10, 'output': 10,
            }, world_size=3)

    def test_alltoall_send_sum_mismatch_raises(self):
        with pytest.raises(ValueError, match='sum.*input'):
            build_op_param({
                'op_name': OpName.Alltoall, 'input': 100, 'output': 100,
                'input_split_sizes': [10, 20],
            }, world_size=2)

    def test_alltoall_recv_sum_mismatch_raises(self):
        with pytest.raises(ValueError, match='sum.*output'):
            build_op_param({
                'op_name': 3, 'input': 100, 'output': 100,
                'input_split_sizes': [60, 40],
                'output_split_sizes': [10, 20],
            }, world_size=2)

    # ---- Alltoallv ----
    def test_alltoallv_basic(self):
        w = 2
        low = build_op_param({
            'op_name': OpName.Alltoallv,
            'input': 100,
            'output': 100,
            'send_counts': [60, 40],
            'recv_counts': [30, 70],
            'sdispls': [0, 60],
            'rdispls': [0, 30],
        }, world_size=w)

        assert low['send_counts'] == [60, 40]
        assert low['recv_counts'] == [30, 70]
        assert low['sdispls'] == [0, 60]
        assert low['rdispls'] == [0, 30]
        assert low['data_count'] == 0

    # ---- General ----
    def test_missing_op_name_raises(self):
        with pytest.raises(ValueError, match='op_name'):
            build_op_param({}, world_size=2)

    def test_missing_input_raises(self):
        with pytest.raises(ValueError, match='input'):
            build_op_param({'op_name': 0, 'output': 10}, world_size=2)

    def test_missing_output_raises(self):
        with pytest.raises(ValueError, match='output'):
            build_op_param({'op_name': 0, 'input': 10}, world_size=2)

    def test_type_error_on_non_dict(self):
        with pytest.raises(TypeError):
            build_op_param(None, world_size=2)

    def test_reduce_op_default(self):
        low = build_op_param({
            'op_name': OpName.Allreduce, 'input': 10, 'output': 10,
        }, world_size=2)
        assert low['reduce_op'] == 0

    def test_reduce_op_explicit(self):
        low = build_op_param({
            'op_name': OpName.Allreduce, 'input': 10, 'output': 10,
            'reduce_op': ReduceOp.MAX,
        }, world_size=2)
        assert low['reduce_op'] == 1

    def test_roundtrip_allreduce(self):
        """build_op_param output can be fed directly to serialize_op_param."""
        low = build_op_param({
            'op_name': OpName.Allreduce, 'input': 42, 'output': 42,
        }, world_size=4)
        blob = serialize_op_param(low)
        assert len(blob) == 4104
        assert blob[0] & 0x0F == OpName.Allreduce
        assert struct.unpack_from('<q', blob, 8)[0] == 42
