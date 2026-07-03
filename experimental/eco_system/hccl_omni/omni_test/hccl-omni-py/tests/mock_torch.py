'''
Mock torch module for testing HCCL-OMNI framework.
Provides patching utilities to mock torch and torch.distributed interfaces.

MockTensor is a real class (not unittest.mock.Mock) so that isinstance checks,
``torch.Tensor | None`` union-type syntax, and numel() calls all work.
'''

import os
import tempfile
from unittest.mock import Mock
import sys


class MockTensor:
    """A real class standing in for torch.Tensor in tests.

    Being a real class allows:
    - ``isinstance(x, torch.Tensor)``
    - ``torch.Tensor | None`` type-union syntax
    - ``.numel()`` / ``.reshape()`` / ``.npu()`` calls
    """

    def __init__(self, data=None, dtype=None):
        if data is None:
            data = []
        self._data = list(data)
        self._numel = len(self._data)

    def numel(self) -> int:
        return self._numel

    def tolist(self):
        return list(self._data)

    def item(self):
        return self._numel

    def npu(self):
        return self

    def reshape(self, *args):
        return self

    def size(self):
        return self._numel

    def __repr__(self):
        return f'MockTensor(numel={self._numel})'


def setup_mock_torch():
    '''
    Setup mock torch environment for testing.
    Should be called before importing hccl_omni modules.
    '''
    # Create temp CANN path so builder.py include_paths() doesn't choke
    cann_tmp = tempfile.mkdtemp(prefix='mock_cann_')
    os.makedirs(os.path.join(cann_tmp, 'include'), exist_ok=True)
    os.makedirs(os.path.join(cann_tmp, 'lib'), exist_ok=True)
    os.environ['ASCEND_HOME_PATH'] = cann_tmp

    # Create mock torch_npu include dir (builder.py reads PYTORCH_NPU_INSTALL_PATH)
    npu_tmp = tempfile.mkdtemp(prefix='mock_npu_')
    os.makedirs(os.path.join(npu_tmp, 'include'), exist_ok=True)
    os.makedirs(os.path.join(npu_tmp, 'lib'), exist_ok=True)

    # Create mock torch.distributed module
    mock_dist = Mock()
    mock_dist.is_initialized = Mock(return_value=True)
    mock_dist.is_available = Mock(return_value=True)
    mock_dist.init_process_group = Mock(return_value=None)
    mock_dist.get_rank = Mock(return_value=0)
    mock_dist.get_world_size = Mock(return_value=1)
    mock_dist.broadcast = Mock(side_effect=lambda tensor, src: tensor)
    mock_dist.barrier = Mock(return_value=None)
    mock_dist.all_to_all = Mock(return_value=None)
    mock_dist.group = Mock()
    mock_dist.group.WORLD = Mock()
    mock_dist.new_group = Mock(return_value=Mock())

    # Create mock torch module
    mock_torch = Mock()
    mock_torch.distributed = mock_dist

    # Mock _C (used by builder.py: torch._C._GLIBCXX_USE_CXX11_ABI)
    mock_C = Mock()
    mock_C._GLIBCXX_USE_CXX11_ABI = False
    mock_torch._C = mock_C

    # Use real classes for Tensor / tensor factory
    mock_torch.Tensor = MockTensor
    mock_torch.tensor = MockTensor

    # Other common torch functions
    mock_torch.zeros = Mock(side_effect=lambda size, dtype=None: MockTensor([0] * size))
    mock_torch.empty = Mock(side_effect=lambda *a, **kw: MockTensor())
    mock_torch.empty_like = Mock(side_effect=lambda t: MockTensor())
    mock_torch.full = Mock(side_effect=lambda size, fill_value, dtype=None: MockTensor([fill_value] * size))
    mock_torch.cat = Mock(side_effect=lambda tensors: MockTensor(sum(t.numel() for t in tensors)))
    mock_torch.long = 4
    mock_torch.uint8 = 1
    mock_torch.float32 = 0
    mock_torch.device = Mock(return_value='npu')
    mock_torch.ops = Mock()
    mock_torch.ops.npu = Mock()
    mock_torch.ops.npu.hccl_omni_run = Mock(return_value=MockTensor())

    # Mock torch.library (used by builder.py)
    mock_library = Mock()
    mock_library.Library = Mock(return_value=Mock())
    mock_library.impl = Mock(return_value=lambda f: f)

    # Mock torch.utils.cpp_extension (used by builder.py)
    mock_cpp_ext = Mock()
    mock_cpp_ext.load = Mock(return_value=Mock())

    mock_utils = Mock()
    mock_utils.cpp_extension = mock_cpp_ext
    mock_torch.utils = mock_utils

    # Mock torch_npu (used by comm_context.py, builder.py)
    mock_torch_npu = Mock()
    mock_torch_npu.__file__ = os.path.join(npu_tmp, '__init__.py')

    # Pre-install mock modules in sys.modules
    sys.modules['torch'] = mock_torch
    sys.modules['torch.distributed'] = mock_dist
    sys.modules['torch.library'] = mock_library
    sys.modules['torch.utils.cpp_extension'] = mock_cpp_ext
    sys.modules['torch_npu'] = mock_torch_npu

    return [mock_torch, mock_dist, mock_torch_npu]


def teardown_mock_torch(patches):
    '''Teardown mock torch environment.'''
    for module_name in [
        'torch', 'torch.distributed', 'torch.library',
        'torch.utils.cpp_extension', 'torch_npu',
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]
