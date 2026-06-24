"""
Integration tests for hccl_omni.backend.invoke_backend.

Verifies that invoke_backend correctly serializes op_param and calls
hccl_omni_run with the new API (no separate counts/displs tensors).

Uses mock_torch environment (configured by tests/conftest.py).
"""

import struct
from unittest.mock import Mock

from hccl_omni.op_param import serialize_op_param, OP_PARAM_SIZE


class TestInvokeBackendOpParam:
    """Tests verifying op_param blob is correctly passed through invoke_backend."""

    def _patch_and_invoke(self, op_param, file_path):
        """Patch torch ops on the backend module's torch ref,
           invoke invoke_backend, and return captured op_param tensor data."""
        import hccl_omni.backend as backend_mod

        captured_op_param = []

        def intercept_tensor(data, dtype=None):
            """Capture the op_param tensor (uint8 CPU tensor)."""
            captured_op_param.append(list(data) if data else [])
            mock_obj = Mock()
            mock_obj.size = Mock(return_value=len(list(data)) if data else 0)
            mock_obj.device = Mock()
            mock_obj.dtype = Mock()
            mock_obj.npu = Mock(return_value=mock_obj)
            return mock_obj

        orig_tensor = backend_mod.torch.tensor
        orig_empty = backend_mod.torch.empty
        backend_mod.torch.tensor = intercept_tensor
        backend_mod.torch.empty = Mock()

        try:
            backend_mod.invoke_backend(op_param, file_path)
        finally:
            backend_mod.torch.tensor = orig_tensor
            backend_mod.torch.empty = orig_empty

        return captured_op_param

    def test_op_param_blob_passed_through(self):
        """op_param blob is serialized and passed as uint8 CPU tensor."""
        op_param = {'op_name': 0}
        captured = self._patch_and_invoke(op_param, '/tmp/test.bin')

        assert len(captured) >= 1, 'torch.tensor was not called for op_param'
        blob_data = captured[0]
        expected = list(serialize_op_param(op_param))
        assert blob_data == expected

    def test_op_param_blob_size_is_4104(self):
        """Serialized op_param blob is exactly 4104 bytes."""
        op_param = {'op_name': 0}
        captured = self._patch_and_invoke(op_param, '/tmp/test.bin')

        assert len(captured) >= 1
        blob_data = captured[0]
        assert len(blob_data) == OP_PARAM_SIZE

    def test_op_param_contains_counts(self):
        """op_param blob includes send_counts/recv_counts in int64[128] arrays."""
        op_param = {
            'op_name': 4,
            'send_counts': [10, 20],
            'recv_counts': [30, 40],
        }
        captured = self._patch_and_invoke(op_param, '/tmp/test.bin')

        assert len(captured) >= 1
        blob = bytes(captured[0])
        assert len(blob) == OP_PARAM_SIZE

        # Verify send_counts[0] at byte offset 8 (int64 LE)
        assert struct.unpack_from('<q', blob, 8)[0] == 10
        assert struct.unpack_from('<q', blob, 16)[0] == 20

        # Verify recv_counts[0] at byte offset 1032 (int64 LE)
        assert struct.unpack_from('<q', blob, 1032)[0] == 30
        assert struct.unpack_from('<q', blob, 1040)[0] == 40

    def test_no_separate_counts_tensors(self):
        """invoke_backend should NOT create separate counts/displs tensors."""
        import hccl_omni.backend as backend_mod

        tensor_calls = []

        def intercept_tensor(data, dtype=None):
            tensor_calls.append({
                'data_len': len(list(data)) if data else 0,
                'dtype': dtype,
            })
            mock_obj = Mock()
            mock_obj.size = Mock(return_value=len(list(data)) if data else 0)
            mock_obj.device = Mock()
            mock_obj.dtype = Mock()
            mock_obj.npu = Mock(return_value=mock_obj)
            return mock_obj

        orig_tensor = backend_mod.torch.tensor
        orig_empty = backend_mod.torch.empty
        backend_mod.torch.tensor = intercept_tensor
        backend_mod.torch.empty = Mock()

        try:
            backend_mod.invoke_backend({'op_name': 0}, '/tmp/test.bin')
        finally:
            backend_mod.torch.tensor = orig_tensor
            backend_mod.torch.empty = orig_empty

        # Should only have ONE torch.tensor call (for op_param blob)
        # No separate calls for send_counts_t, sdispls_t, recv_counts_t, rdispls_t
        op_param_calls = [c for c in tensor_calls if c['dtype'] is not None]
        assert len(op_param_calls) == 1, \
            f"Expected 1 torch.tensor call (op_param only), got {len(op_param_calls)}"
