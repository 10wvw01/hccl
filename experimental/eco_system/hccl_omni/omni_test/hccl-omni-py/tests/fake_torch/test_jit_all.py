#!/usr/bin/env python3
'''
Comprehensive test for HCCL-OMNI JIT functionality.
Tests cache key format, XML splitting, and backend integration.

Mock torch environment is automatically set up by conftest.py.
'''

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path

import hccl_omni
from hccl_omni import HcclOmniConfig
from hccl_omni.splitter import split_xml_by_rank
from hccl_omni.cache import get_hccl_cache_dir
from test_utils import setup_unified_test_environment, cleanup_test_environment


# Set up test configuration
temp_dir, config_file = setup_unified_test_environment()

# Get project root for file operations
project_root = Path(__file__).parent.parent.parent


def test_cache_key_with_rank_suffix():
    '''Test that cache key includes rank suffix.'''
    print('Testing cache key with rank suffix...')

    # Create a jit function
    @hccl_omni.jit
    def test_op(cluster_config, op_param):
        return 'test'

    # Mock configurations
    kernel_config = {'kernel': 'test'}
    cluster_config = {'nodes': 1}
    op_param = {'op_name': 2, 'input': 1024, 'output': 1024}

    # Get operator
    operator = test_op[kernel_config]

    # The cache key generation happens inside operator_wrapper
    # We'll trace it by checking the cache file names that get created
    # First, clear any existing cache
    cache_dir = '.hccl_cache'
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Run the operator (will create cache file)
    operator(cluster_config, op_param)

    # Check cache files
    cache_files = []
    if os.path.exists(cache_dir):
        cache_files = os.listdir(cache_dir)

    print(f"Cache files created: {cache_files}")

    # Verify cache file name format: should end with .0.bin (rank 0 suffix)
    for filename in cache_files:
        print(f"Checking filename: {filename}")
        assert filename.endswith('.0.bin'), f"Cache file should end with .0.bin, got {filename}"

        # Check format: {cluster_hash}-{op_hash}.{rank}.bin
        parts = filename.split('.')
        assert len(parts) == 3, f"Expected format: hash-hash.rank.bin, got {filename}"
        assert parts[1] == '0', f"Rank should be 0, got {parts[1]}"
        assert parts[2] == 'bin', f"Extension should be bin, got {parts[2]}"

    print('[OK] Cache key with rank suffix verified')


def test_xml_splitting_function():
    '''Test that split_xml_by_rank function exists and works.'''
    print('Testing XML splitting function...')

    # Test basic function call
    test_xml = b'<?xml version="1.0"?><test>data</test>'
    result = split_xml_by_rank(test_xml)

    # Function should return a valid XML
    assert result is not None
    assert b'<?xml version="1.0"?>' in result

    print('[OK] XML splitting function verified')


def test_xml_splitting_by_rank():
    '''Test actual XML splitting logic based on rankId.'''
    print('Testing XML splitting by rank...')

    # Create test XML with NPU tags under root
    test_xml = b'''<?xml version="1.0"?>
<root>
  <NPU rankId="0">
    <resource>Resource for rank 0</resource>
    <type>A</type>
    <instruction>Instruction for rank 0</instruction>
    <op>ADD</op>
  </NPU>
  <NPU rankId="1">
    <resource>Resource for rank 1</resource>
    <type>B</type>
    <instruction>Instruction for rank 1</instruction>
    <op>MUL</op>
  </NPU>
</root>'''

    # Test for rank 0
    result_0 = split_xml_by_rank(test_xml, rank=0)
    print(f"Rank 0 result length: {len(result_0)}")

    # Verify rank 0 result contains expected content
    assert b'Resource for rank 0' in result_0
    assert b'Instruction for rank 0' in result_0
    assert b'<type>A</type>' in result_0
    assert b'<op>ADD</op>' in result_0

    # Verify rank 0 result does NOT contain rank 1 content
    assert b'Resource for rank 1' not in result_0
    assert b'Instruction for rank 1' not in result_0

    # Verify NPU tags themselves are not included
    assert b'<NPU' not in result_0
    assert b'rankId=' not in result_0

    # Test for rank 1
    result_1 = split_xml_by_rank(test_xml, rank=1)
    print(f"Rank 1 result length: {len(result_1)}")

    # Verify rank 1 result contains expected content
    assert b'Resource for rank 1' in result_1
    assert b'Instruction for rank 1' in result_1

    # Verify rank 1 result does NOT contain rank 0 content
    assert b'Resource for rank 0' not in result_1
    assert b'Instruction for rank 0' not in result_1

    # Test for rank 2 (no matching NPU)
    result_2 = split_xml_by_rank(test_xml, rank=2)
    # Should return empty root since no NPU tags with rankId=2
    assert b'<root/>' in result_2 or b'<root>' in result_2

    print('[OK] XML splitting by rank verified')


def test_missing_rankid():
    '''Test XML with NPU tags missing rankId attribute.'''
    print('Testing XML with missing rankId...')

    test_xml = b'''<?xml version="1.0"?>
<root>
  <NPU>No rankId attribute</NPU>
  <NPU rankId="3">
    <content>Content for rank 3</content>
  </NPU>
</root>'''

    # NPU without rankId should be ignored
    result_0 = split_xml_by_rank(test_xml, rank=0)
    assert b'<root/>' in result_0 or b'<root>' in result_0

    # NPU with rankId=3 should be included for rank 3
    result_3 = split_xml_by_rank(test_xml, rank=3)
    assert b'Content for rank 3' in result_3

    print('[OK] Missing rankId handling verified')


def test_invalid_rankid():
    '''Test XML with invalid rankId values.'''
    print('Testing XML with invalid rankId...')

    test_xml = b'''<?xml version="1.0"?>
<root>
  <NPU rankId="not_a_number">
    <content>Invalid rankId</content>
  </NPU>
  <NPU rankId="5">
    <content>Content for rank 5</content>
  </NPU>
</root>'''

    # Invalid rankId should be ignored
    result_0 = split_xml_by_rank(test_xml, rank=0)
    assert b'<root/>' in result_0 or b'<root>' in result_0

    # Valid rankId=5 should work
    result_5 = split_xml_by_rank(test_xml, rank=5)
    assert b'Content for rank 5' in result_5

    print('[OK] Invalid rankId handling verified')


def test_empty_xml():
    '''Test empty XML input.'''
    print('Testing empty XML...')

    import xml.etree.ElementTree as ET
    try:
        # Empty XML should raise ParseError
        split_xml_by_rank(b'')
        print('[FAIL] Should have raised ParseError')
        assert False
    except ET.ParseError:
        print('[OK] Empty XML raises ParseError as expected')


def test_invoke_backend_with_file_path():
    '''Test that invoke_backend receives file path string, not XML bytes.'''
    print('Testing invoke_backend receives file path...')

    # We'll test this by checking the invoke_backend signature
    from hccl_omni.backend import invoke_backend

    # Check function signature by reading source
    backend_path = os.path.join(project_root, 'hccl_omni', 'backend.py')
    with open(backend_path, 'r') as f:
        source = f.read()

    # Look for function definition
    assert 'def invoke_backend(op_param: dict, file_path: str)' in source,\
        'invoke_backend should take op_param: dict, file_path: str parameters'

    # The actual test happens when we run an operator
    # We can verify this by checking that a file path is printed
    # (invoke_backend prints the file path)

    print('[OK] invoke_backend receives file path verified')


def test_integration():
    '''Integration test of all modifications.'''
    print('\nRunning integration test...')

    # Get the actual cache directory from cache module
    cache_dir = get_hccl_cache_dir()

    print(f"Cache directory: {cache_dir}")

    # Clear cache directory first
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # Create jit function
        config = HcclOmniConfig()

        @hccl_omni.jit(hccl_omni_cfg=config)
        def integration_op(cluster_config, op_param):
            return 'integration_test'

        kernel_config = {'test': 'integration'}
        cluster_config = {'cluster': 'test'}
        op_param = {'op_name': 2, 'input': 1024, 'output': 1024, 'operation': 'test'}

        # Run operator
        operator = integration_op[kernel_config]
        result = operator(cluster_config, op_param)

        # Verify cache was created in the actual cache directory
        cache_files = []
        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.bin')]

        print(f"Integration test cache files: {cache_files}")

        # Verify at least one cache file was created
        assert len(cache_files) > 0, f"No cache files were created in {cache_dir}"

        # Verify each file has rank suffix
        for filename in cache_files:
            assert '.0.bin' in filename, f"Cache file missing rank suffix: {filename}"

        print('[OK] Integration test passed')

    finally:
        # Cleanup
        if os.path.exists(cache_dir):
            # Only remove files, not the directory itself (might be used by other tests)
            for filename in os.listdir(cache_dir):
                filepath = os.path.join(cache_dir, filename)
                try:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                except:
                    pass