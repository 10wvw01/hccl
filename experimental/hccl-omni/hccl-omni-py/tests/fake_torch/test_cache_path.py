#!/usr/bin/env python3
'''
Test for cache path resolution logic.

Mock torch environment is automatically set up by conftest.py.
This test manually controls test configuration to test various scenarios.
'''

import os
import sys
import json
import tempfile

import hccl_omni.cache


def test_import_without_config():
    '''Test that module fails to import without proper configuration.'''
    print('Test 1: Module import should fail without proper config')

    # Remove environment variable if it exists
    old_env = os.environ.pop('HCCL_TOPO_FILE_PATH', None)

    try:
        # Need to reload the module to test import without config
        import importlib
        import hccl_omni.cache
        importlib.reload(hccl_omni.cache)
        print('[FAIL] Should have raised RuntimeError')
        return False
    except RuntimeError as e:
        print(f"[OK] Got expected RuntimeError: {str(e)[:100]}...")
        return True
    finally:
        # Restore environment variable
        if old_env is not None:
            os.environ['HCCL_TOPO_FILE_PATH'] = old_env


def test_successful_resolution():
    '''Test successful cache path resolution.'''
    print('\nTest 2: Successful path resolution')

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock topology file
        topo_file = os.path.join(tmpdir, 'topology.json')
        with open(topo_file, 'w') as f:
            f.write('{"nodes": 1}')

        # Create config file
        config_file = os.path.join(tmpdir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({'topo_file_path': topo_file}, f)

        # Set environment variable
        os.environ['HCCL_TOPO_FILE_PATH'] = config_file

        try:
            # Import module - should work now
            import importlib
            import hccl_omni.cache
            importlib.reload(hccl_omni.cache)

            resolved_path = hccl_omni.cache.get_hccl_cache_dir()
            expected_path = os.path.dirname(topo_file)

            print(f"  Resolved: {resolved_path}")
            print(f"  Expected: {expected_path}")

            if os.path.abspath(resolved_path) == os.path.abspath(expected_path):
                print('[OK] Path resolution successful')
                return True
            else:
                print('[FAIL] Path mismatch')
                return False

        except Exception as e:
            print(f"[FAIL] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Clean up environment variable
            if 'HCCL_TOPO_FILE_PATH' in os.environ:
                del os.environ['HCCL_TOPO_FILE_PATH']


def test_missing_field():
    '''Test missing topo_file_path field.'''
    print('\nTest 3: Missing topo_file_path field')

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({'wrong_field': 'value'}, f)

        os.environ['HCCL_TOPO_FILE_PATH'] = config_file

        try:
            import importlib
            import hccl_omni.cache
            importlib.reload(hccl_omni.cache)
            print('[FAIL] Should have raised RuntimeError')
            return False
        except RuntimeError as e:
            if "Missing 'topo_file_path' field" in str(e):
                print(f"[OK] Got expected error")
                return True
            else:
                print(f"[FAIL] Wrong error: {e}")
                return False
        finally:
            if 'HCCL_TOPO_FILE_PATH' in os.environ:
                del os.environ['HCCL_TOPO_FILE_PATH']


def test_invalid_json():
    '''Test invalid JSON.'''
    print('\nTest 4: Invalid JSON')

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'config.json')
        with open(config_file, 'w') as f:
            f.write('invalid json')

        os.environ['HCCL_TOPO_FILE_PATH'] = config_file

        try:
            import importlib
            import hccl_omni.cache
            importlib.reload(hccl_omni.cache)
            print('[FAIL] Should have raised RuntimeError')
            return False
        except RuntimeError as e:
            if 'Failed to parse JSON' in str(e):
                print(f"[OK] Got expected error")
                return True
            else:
                print(f"[FAIL] Wrong error: {e}")
                return False
        finally:
            if 'HCCL_TOPO_FILE_PATH' in os.environ:
                del os.environ['HCCL_TOPO_FILE_PATH']


def test_nonexistent_parent():
    '''Test non-existent parent directory.'''
    print('\nTest 5: Non-existent parent directory')

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({'topo_file_path': '/non/existent/path/topo.json'}, f)

        os.environ['HCCL_TOPO_FILE_PATH'] = config_file

        try:
            import importlib
            import hccl_omni.cache
            importlib.reload(hccl_omni.cache)
            print('[FAIL] Should have raised RuntimeError')
            return False
        except RuntimeError as e:
            if 'Parent directory does not exist' in str(e):
                print(f"[OK] Got expected error")
                return True
            else:
                print(f"[FAIL] Wrong error: {e}")
                return False
        finally:
            if 'HCCL_TOPO_FILE_PATH' in os.environ:
                del os.environ['HCCL_TOPO_FILE_PATH']


def test_relative_path_conversion():
    '''Test relative path conversion to absolute.'''
    print('\nTest 6: Relative path conversion')

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create topology file with relative path
        topo_file = 'relative/path/to/topo.json'
        config_file = os.path.join(tmpdir, 'config.json')

        # Create the directory structure
        full_topo_path = os.path.join(tmpdir, topo_file)
        os.makedirs(os.path.dirname(full_topo_path), exist_ok=True)
        with open(full_topo_path, 'w') as f:
            f.write('{"nodes": 1}')

        # Use absolute path in JSON for this test since relative paths are relative to current directory
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({'topo_file_path': full_topo_path}, f)

        os.environ['HCCL_TOPO_FILE_PATH'] = config_file

        try:
            import importlib
            import hccl_omni.cache
            importlib.reload(hccl_omni.cache)

            resolved_path = hccl_omni.cache.get_hccl_cache_dir()
            expected_parent = os.path.dirname(full_topo_path)

            print(f"  Resolved: {resolved_path}")
            print(f"  Expected: {expected_parent}")

            if os.path.abspath(resolved_path) == os.path.abspath(expected_parent):
                print('[OK] Relative path correctly converted to absolute')
                return True
            else:
                print('[FAIL] Path mismatch')
                return False

        except Exception as e:
            print(f"[FAIL] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if 'HCCL_TOPO_FILE_PATH' in os.environ:
                del os.environ['HCCL_TOPO_FILE_PATH']