"""
Tests for hccl_omni.backend components.

Uses mock_torch environment.  Tests _extract_tensors and the full
build_op_param → serialize flow that invoke_backend uses internally.
"""

import struct
from unittest.mock import Mock, patch

from hccl_omni.op_param import build_op_param, serialize_op_param, OP_PARAM_SIZE, OpName
from hccl_omni.backend import _extract_tensors


class TestExtractTensors:
    """Tests for _extract_tensors helper."""

    def test_non_allgather_uses_output_directly(self):
        inp = Mock()
        out = Mock()
        send_buf, recv_buf = _extract_tensors(
            {'input': inp, 'output': out}, OpName.Allreduce,
        )
        assert send_buf is inp
        assert recv_buf is out

    def test_allgather_concatenates_output_list(self):
        t1 = Mock()
        t2 = Mock()
        t1.reshape = Mock(return_value=t1)
        t2.reshape = Mock(return_value=t2)

        concat_result = Mock()
        with patch('hccl_omni.backend.torch.cat', return_value=concat_result) as mock_cat:
            send_buf, recv_buf = _extract_tensors(
                {'input': t1, 'output': [t1, t2]},
                OpName.Allgather,
            )
            assert send_buf is t1
            assert recv_buf is concat_result
            mock_cat.assert_called_once()


class TestBackendBlob:
    """Verify the blob that invoke_backend would pass to hccl_omni_run."""

    def _roundtrip(self, op_param, world_size):
        """Simulate what invoke_backend does: build → serialize."""
        low = build_op_param(op_param, world_size)
        return serialize_op_param(low)

    def test_allreduce_blob(self):
        blob = self._roundtrip({
            'op_name': OpName.Allreduce, 'input': 100, 'output': 100,
        }, world_size=4)
        assert len(blob) == OP_PARAM_SIZE
        assert blob[0] & 0x0F == OpName.Allreduce
        assert struct.unpack_from('<q', blob, 8)[0] == 100

    def test_reduce_scatter_blob(self):
        blob = self._roundtrip({
            'op_name': OpName.ReduceScatter, 'input': 64, 'output': 16,
        }, world_size=4)
        assert len(blob) == OP_PARAM_SIZE
        assert struct.unpack_from('<q', blob, 8)[0] == 16  # 64 / 4
        assert struct.unpack_from('<q', blob, 2056)[0] == 0  # sdispls[0]

    def test_alltoall_blob(self):
        blob = self._roundtrip({
            'op_name': OpName.Alltoall, 'input': 100, 'output': 100,
            'input_split_sizes': [60, 40], 'output_split_sizes': [30, 70],
        }, world_size=2)
        assert len(blob) == OP_PARAM_SIZE
        assert struct.unpack_from('<q', blob, 8)[0] == 60   # send_counts[0]
        assert struct.unpack_from('<q', blob, 1032)[0] == 30 # recv_counts[0]

    def test_allgather_blob(self):
        blob = self._roundtrip({
            'op_name': OpName.Allgather, 'input': 10,
            'output': [10, 10, 10, 10],
        }, world_size=4)
        assert len(blob) == OP_PARAM_SIZE
        assert struct.unpack_from('<q', blob, 8)[0] == 10   # send_counts[0]
        # rdispls should be [0, 10, 20, 30]
        assert struct.unpack_from('<q', blob, 3080)[0] == 0
        assert struct.unpack_from('<q', blob, 3088)[0] == 10
