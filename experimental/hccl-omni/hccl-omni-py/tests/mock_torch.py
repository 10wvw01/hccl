'''
Mock torch module for testing HCCL-OMNI framework.
Provides patching utilities to mock torch and torch.distributed interfaces.
'''

from unittest.mock import Mock
import sys


def setup_mock_torch():
    '''
    Setup mock torch environment for testing.
    Should be called before importing hccl_omni modules.
    '''
    # Create mock torch.distributed module
    mock_dist = Mock()

    # Mock distributed functions
    mock_dist.is_initialized = Mock(return_value=True)
    mock_dist.init_process_group = Mock(return_value=None)
    mock_dist.get_rank = Mock(return_value=0)
    mock_dist.get_world_size = Mock(return_value=1)
    mock_dist.broadcast = Mock(side_effect=lambda tensor, src: tensor)
    mock_dist.barrier = Mock(return_value=None)
    mock_dist.all_to_all = Mock(return_value=None)

    # Create mock torch module
    mock_torch = Mock()
    mock_torch.distributed = mock_dist

    # Mock tensor creation functions
    def mock_tensor(data=None, dtype=None):
        mock_tensor_obj = Mock()
        mock_tensor_obj.tolist = Mock(return_value=list(data) if data else [])
        mock_tensor_obj.item = Mock(return_value=len(data) if data else 0)
        mock_tensor_obj.npu = Mock(return_value=mock_tensor_obj)  # npu() returns self
        return mock_tensor_obj

    def mock_zeros(size, dtype=None):
        mock_tensor_obj = Mock()
        mock_tensor_obj.tolist = Mock(return_value=[0] * size)
        mock_tensor_obj.item = Mock(return_value=size)
        mock_tensor_obj.npu = Mock(return_value=mock_tensor_obj)
        return mock_tensor_obj

    def mock_empty_like(tensor):
        mock_tensor_obj = Mock()
        mock_tensor_obj.npu = Mock(return_value=mock_tensor_obj)
        return mock_tensor_obj

    # Mock Tensor class
    mock_tensor_class = Mock()
    mock_tensor_class.side_effect = mock_tensor

    # Setup mock functions
    mock_torch.tensor = Mock(side_effect=mock_tensor)
    mock_torch.zeros = Mock(side_effect=mock_zeros)
    mock_torch.empty_like = Mock(side_effect=mock_empty_like)
    mock_torch.Tensor = mock_tensor_class
    mock_torch.long = Mock()
    mock_torch.uint8 = Mock()

    # Pre-install mock modules in sys.modules
    # This ensures torch imports will use our mock
    sys.modules['torch'] = mock_torch
    sys.modules['torch.distributed'] = mock_dist

    # Store references for cleanup if needed
    return [mock_torch, mock_dist]


def teardown_mock_torch(patches):
    '''
    Teardown mock torch environment.

    Args:
        patches: List of mock objects returned by setup_mock_torch()
    '''
    # Remove mock modules from sys.modules
    for module_name in ['torch', 'torch.distributed']:
        if module_name in sys.modules:
            del sys.modules[module_name]
