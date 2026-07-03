#!/usr/bin/env python3
'''
Common test utilities for HCCL-OMNI framework tests.
Provides unified setup for mock environment and test infrastructure.

This module contains both shared utilities (usable by fake and real torch tests)
and mock-specific utilities (for fake torch tests only).
'''

import os
import sys
import tempfile
import json
import shutil

# Mock torch patches storage
_mock_patches = None


# =============================================================================
# Shared Utilities (usable by both fake_torch and real_torch tests)
# =============================================================================

def create_test_configuration():
    '''
    Create test configuration files without automatically setting environment variables.

    Returns:
        tuple: (temp_dir_path, config_file_path, topo_file_path)

    This function is safe to use in both fake_torch and real_torch tests.
    '''
    # Create temporary directory for test files
    temp_dir = '.hccl_cache'
    os.makedirs(temp_dir, exist_ok=True)

    # Create topology file
    topo_file = os.path.join(temp_dir, 'topology.json')
    with open(topo_file, 'w') as f:
        f.write('{"nodes": 1}')

    # Create config file pointing to topology file
    config_file = os.path.join(temp_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump({'topo_file_path': topo_file}, f)

    return temp_dir, config_file, topo_file


def cleanup_test_environment(temp_dir, config_file=None):
    '''
    Clean up test environment after tests complete.

    Args:
        temp_dir: Temporary directory path returned by create_test_configuration
        config_file: Optional config file path for additional cleanup

    This function is safe to use in both fake_torch and real_torch tests.
    '''
    # Remove environment variable
    if 'HCCL_TOPO_FILE_PATH' in os.environ:
        del os.environ['HCCL_TOPO_FILE_PATH']

    # Clean up temporary files
    if config_file and os.path.exists(config_file):
        try:
            os.unlink(config_file)
        except:
            pass

    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Mock-Specific Utilities (for fake_torch tests only)
# =============================================================================

def setup_mock_torch_only():
    '''
    Setup mock torch only, without creating test configuration.
    This should be called at module level before any hccl_omni imports.

    Returns:
        bool: True if mock was set up, False if already set up

    Note: This function is for fake_torch tests only.
    '''
    global _mock_patches

    # Setup mock torch if not already set up
    if _mock_patches is None:
        from mock_torch import setup_mock_torch
        _mock_patches = setup_mock_torch()
        return True
    return False


def setup_unified_test_environment():
    '''
    Setup unified test environment with mock torch and temporary configuration.

    Returns:
        tuple: (temp_dir_path, config_file_path)

    This function should be called at the beginning of each fake_torch test module.

    Note: This function is for fake_torch tests only.
    '''
    # Setup mock torch
    setup_mock_torch_only()

    # Create test configuration
    temp_dir, config_file, _ = create_test_configuration()

    # Set environment variable for cache path resolution
    os.environ['HCCL_TOPO_FILE_PATH'] = config_file

    return temp_dir, config_file


def get_mock_torch_patches():
    '''
    Get the mock torch patches for cleanup if needed.

    Returns:
        List of mock patches or None if not set up

    Note: This function is for fake_torch tests only.
    '''
    return _mock_patches


def teardown_mock_torch():
    '''
    Teardown mock torch environment.
    Usually not needed as patches persist for the test session.

    Note: This function is for fake_torch tests only.
    '''
    global _mock_patches
    if _mock_patches is not None:
        from mock_torch import teardown_mock_torch as teardown
        teardown(_mock_patches)
        _mock_patches = None
