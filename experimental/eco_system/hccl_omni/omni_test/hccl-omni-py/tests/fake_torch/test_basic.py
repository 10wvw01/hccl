#!/usr/bin/env python3
'''
Simple test for HCCL-OMNI framework core logic.
Tests caching, interpreter modes, and basic functionality.

Mock torch environment is automatically set up by conftest.py.
'''

import os
import sys
import tempfile
import shutil
import json

import hccl_omni
from hccl_omni import HcclOmniConfig
from hccl_omni.cache import CacheManager
from test_utils import setup_unified_test_environment, cleanup_test_environment


# Set up test configuration
temp_dir, config_file = setup_unified_test_environment()


def test_basic_import():
    '''Test basic import and module availability.'''
    print('Testing basic imports...')

    # Check main module
    assert hasattr(hccl_omni, 'jit'), 'jit not found in hccl_omni'
    assert hasattr(hccl_omni, 'HcclOmniConfig'), 'HcclOmniConfig not found'

    # Check that torch is mocked and distributed functions work
    import torch.distributed as dist
    assert dist.is_initialized() == True, 'dist should be initialized'
    assert dist.get_rank() == 0, 'get_rank should return 0 with mock'
    assert dist.get_world_size() == 1, 'get_world_size should return 1 with mock'

    print('[OK] Basic imports passed')


def test_hccl_omni_config():
    '''Test HcclOmniConfig class.'''
    print('Testing HcclOmniConfig...')

    # Test with all parameters
    config = HcclOmniConfig(
        op_key='test_op',
        ins_file_path='/tmp/test',
        ins_gen_mode='DSL'
    )

    assert config.op_key == 'test_op'
    assert config.ins_file_path == '/tmp/test'
    assert config.ins_gen_mode == 'DSL'

    # Test to_dict method
    config_dict = config.to_dict()
    assert config_dict['op_key'] == 'test_op'
    assert config_dict['ins_file_path'] == '/tmp/test'
    assert config_dict['ins_gen_mode'] == 'DSL'

    # Test with partial parameters
    config2 = HcclOmniConfig(op_key='partial')
    assert config2.op_key == 'partial'
    assert config2.ins_file_path is None
    assert config2.ins_gen_mode is None

    print('[OK] HcclOmniConfig passed')


def test_cache_manager():
    '''Test CacheManager in single-process mode.'''
    print('Testing CacheManager...')

    # Create temporary directory for cache
    temp_dir = tempfile.mkdtemp(prefix='hccl_test_cache_')

    try:
        # Create cache manager with temp directory
        cache = CacheManager(cache_path=temp_dir)

        # Test data with combined cache key
        cluster_key = 'cluster123'
        op_key = 'op456'
        cache_key = f"{cluster_key}-{op_key}"
        test_data = b"<?xml version='1.0'?><test>data</test>"

        # Test put and get
        cache.put(cache_key, test_data)

        # Check file was created
        expected_file = os.path.join(temp_dir, f"{cache_key}.bin")
        assert os.path.exists(expected_file), f"Cache file not created: {expected_file}"

        # Test get
        retrieved_data = cache.get(cache_key)
        assert retrieved_data == test_data, f"Retrieved data doesn't match: {retrieved_data}"

        # Test get non-existent
        non_existent = cache.get('nonexistent-key')
        assert non_existent is None, f"Should return None for non-existent key"

        # Test clear specific
        other_cache_key = f"{cluster_key}-other_op"
        cache.put(other_cache_key, b"other_data")
        cache.clear(cache_key=cache_key)
        assert cache.get(cache_key) is None, 'Should be cleared'
        assert cache.get(other_cache_key) is not None, 'Other op should still exist'

        # Test clear all
        cache.clear()
        assert cache.get(other_cache_key) is None, 'All should be cleared'

        print('[OK] CacheManager passed')

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_jit_decorator():
    '''Test jit decorator basic functionality.'''
    print('Testing jit decorator...')

    # Test without parameters
    @hccl_omni.jit
    def simple_op(cluster_config, op_param):
        return 'test_result'

    assert hasattr(simple_op, 'func'), 'JITFunction should have func attribute'
    assert simple_op.func.__name__ == 'simple_op', f"Wrong function name: {simple_op.func.__name__}"

    # Test with parameters using HcclOmniConfig
    config = HcclOmniConfig(op_key='test_key', ins_gen_mode='DSL')

    @hccl_omni.jit(version=2, hccl_omni_cfg=config, debug=True)
    def configured_op(cluster_config, op_param):
        return 'configured_result'

    assert configured_op.version == 2, f"Version should be 2, got {configured_op.version}"
    assert configured_op.debug == True, f"Debug should be True, got {configured_op.debug}"
    assert configured_op.hccl_omni_cfg is not None, 'hccl_omni_cfg should not be None'
    assert configured_op.hccl_omni_cfg.op_key == 'test_key'
    assert configured_op.hccl_omni_cfg.ins_gen_mode == 'DSL'

    # Test with None configuration (should create default instance)
    @hccl_omni.jit(hccl_omni_cfg=None)
    def no_config_op(cluster_config, op_param):
        return 'no_config_result'

    assert no_config_op.hccl_omni_cfg is not None, 'hccl_omni_cfg should not be None (default instance created)'
    assert isinstance(no_config_op.hccl_omni_cfg, HcclOmniConfig), 'hccl_omni_cfg should be HcclOmniConfig instance'
    assert no_config_op.hccl_omni_cfg.op_key is None, 'Default instance should have None op_key'
    assert no_config_op.hccl_omni_cfg.ins_file_path is None, 'Default instance should have None ins_file_path'
    assert no_config_op.hccl_omni_cfg.ins_gen_mode is None, 'Default instance should have None ins_gen_mode'

    print('[OK] jit decorator passed')


def test_operator_invocation():
    '''Test operator invocation syntax.'''
    print('Testing operator invocation...')

    # Define a simple operator
    @hccl_omni.jit
    def test_operator(cluster_config, op_param):
        # This function body is not executed in DSL mode
        return None

    # Create test configurations
    kernel_config = {'kernel': 'test'}
    cluster_config = {'nodes': 1}
    op_param = {'op_name': 2, 'input': 1024, 'output': 1024}

    # Test operator[kernel_config](cluster_config, op_param) syntax
    operator = test_operator[kernel_config]

    # The result will be None because invoke_backend returns None in placeholder
    result = operator(cluster_config, op_param)

    # Just verify it doesn't crash
    print('[OK] Operator invocation syntax passed')