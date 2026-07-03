#!/usr/bin/env python3
'''
Test for cache path resolution logic.

Mock torch environment is automatically set up by conftest.py.
'''

import os
import json
import tempfile
import importlib

import hccl_omni.cache


def _reload():
    """Reload cache module and reset its lazy singleton."""
    hccl_omni.cache._HCCL_OMNI_BIN_FILE_CACHE_DIR = None
    importlib.reload(hccl_omni.cache)


def test_fallback_without_config():
    '''Without HCCL_TOPO_FILE_PATH, falls back to .hccl_cache.'''
    old_env = os.environ.pop('HCCL_TOPO_FILE_PATH', None)

    try:
        _reload()
        resolved = hccl_omni.cache.get_hccl_cache_dir()
        assert resolved.endswith('.hccl_cache')
    finally:
        if old_env is not None:
            os.environ['HCCL_TOPO_FILE_PATH'] = old_env


def test_resolve_from_topology():
    '''With valid topology config, resolves to parent/omni_cache.'''
    with tempfile.TemporaryDirectory() as tmpdir:
        topo_file = os.path.join(tmpdir, 'topology.json')
        with open(topo_file, 'w') as f:
            f.write('{"nodes": 1}')

        config_file = os.path.join(tmpdir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({'topo_file_path': topo_file}, f)

        os.environ['HCCL_TOPO_FILE_PATH'] = config_file

        try:
            _reload()
            resolved = hccl_omni.cache.get_hccl_cache_dir()
            expected = os.path.join(tmpdir, 'omni_cache')
            assert os.path.abspath(resolved) == os.path.abspath(expected)
        finally:
            if 'HCCL_TOPO_FILE_PATH' in os.environ:
                del os.environ['HCCL_TOPO_FILE_PATH']


def test_missing_field_falls_back():
    '''Missing topo_file_path field → fallback to .hccl_cache.'''
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({'wrong_field': 'value'}, f)

        os.environ['HCCL_TOPO_FILE_PATH'] = config_file

        try:
            _reload()
            resolved = hccl_omni.cache.get_hccl_cache_dir()
            assert resolved.endswith('.hccl_cache')
        finally:
            if 'HCCL_TOPO_FILE_PATH' in os.environ:
                del os.environ['HCCL_TOPO_FILE_PATH']


def test_invalid_json_falls_back():
    '''Invalid JSON → fallback to .hccl_cache.'''
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'config.json')
        with open(config_file, 'w') as f:
            f.write('invalid json')

        os.environ['HCCL_TOPO_FILE_PATH'] = config_file

        try:
            _reload()
            resolved = hccl_omni.cache.get_hccl_cache_dir()
            assert resolved.endswith('.hccl_cache')
        finally:
            if 'HCCL_TOPO_FILE_PATH' in os.environ:
                del os.environ['HCCL_TOPO_FILE_PATH']


def test_nonexistent_parent_falls_back():
    '''Non-existent parent directory → fallback to .hccl_cache.'''
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({'topo_file_path': '/non/existent/path/topo.json'}, f)

        os.environ['HCCL_TOPO_FILE_PATH'] = config_file

        try:
            _reload()
            resolved = hccl_omni.cache.get_hccl_cache_dir()
            assert resolved.endswith('.hccl_cache')
        finally:
            if 'HCCL_TOPO_FILE_PATH' in os.environ:
                del os.environ['HCCL_TOPO_FILE_PATH']
