'''
Pytest configuration and fixtures for HCCL-OMNI tests.

This file is automatically loaded by pytest and sets up:
1. Project root directory in sys.path
2. Mock torch environment for fake_torch tests
'''

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add tests directory to sys.path for direct imports
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Setup mock torch environment (only if torch is not already imported)
if 'torch' not in sys.modules:
    from mock_torch import setup_mock_torch
    setup_mock_torch()