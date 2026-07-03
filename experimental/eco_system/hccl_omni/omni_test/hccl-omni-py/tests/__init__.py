'''
Test package for HCCL-OMNI framework.

This package provides test utilities for running tests.
When running with pytest, the conftest.py file automatically sets up:
1. Project root directory in sys.path
2. Mock torch environment for fake_torch tests
'''

from .test_utils import (
    create_test_configuration,
    cleanup_test_environment,
    setup_mock_torch_only,
    setup_unified_test_environment,
    get_mock_torch_patches,
    teardown_mock_torch,
)

__all__ = [
    'create_test_configuration',
    'cleanup_test_environment',
    'setup_mock_torch_only',
    'setup_unified_test_environment',
    'get_mock_torch_patches',
    'teardown_mock_torch',
]