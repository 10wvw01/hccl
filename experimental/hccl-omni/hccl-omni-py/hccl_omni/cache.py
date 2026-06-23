'''
Cache management for HCCL-OMNI communication operators.
'''

import json
import os


_HCCL_OMNI_BIN_FILE_CACHE_DIR = None

_DEFAULT_CACHE_DIR = '.hccl_cache'


def _resolve_cache_path_from_topology() -> str | None:
    '''Try to resolve cache directory from topology JSON. Returns None on failure.'''
    json_path = os.environ.get('HCCL_TOPO_FILE_PATH', '/etc/hccl_rootinfo.json')

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    topo_file_path = data.get('topo_file_path')
    if topo_file_path is None:
        return None

    abs_topo_file_path = os.path.abspath(topo_file_path)
    parent_dir = os.path.dirname(abs_topo_file_path)

    if os.path.exists(parent_dir):
        return parent_dir

    return None


def _get_default_cache_dir() -> str:
    '''Get a usable cache directory, creating it if necessary.'''
    # 1. Try HCCL_OMNI_CACHE_DIR env var
    env_dir = os.environ.get('HCCL_OMNI_CACHE_DIR')
    if env_dir:
        os.makedirs(env_dir, exist_ok=True)
        return env_dir

    # 2. Try topology-based resolution
    topo_dir = _resolve_cache_path_from_topology()
    if topo_dir:
        cache_dir = os.path.join(topo_dir, 'omni_cache')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    # 3. Fall back to local .hccl_cache directory
    os.makedirs(_DEFAULT_CACHE_DIR, exist_ok=True)
    return os.path.abspath(_DEFAULT_CACHE_DIR)


def get_hccl_cache_dir() -> str:
    """Lazy getter for cache directory."""
    global _HCCL_OMNI_BIN_FILE_CACHE_DIR
    if _HCCL_OMNI_BIN_FILE_CACHE_DIR is None:
        _HCCL_OMNI_BIN_FILE_CACHE_DIR = _get_default_cache_dir()
    return _HCCL_OMNI_BIN_FILE_CACHE_DIR


class CacheManager:
    '''Manages cache for generated binary instruction files.'''

    def __init__(self, cache_path: str | None = None):
        self.cache_path = cache_path or get_hccl_cache_dir()
        os.makedirs(self.cache_path, exist_ok=True)

    def get(self, cache_key: str) -> bytes | None:
        filepath = self.get_cache_filepath(cache_key)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'rb') as f:
                return f.read()
        except (IOError, OSError):
            return None

    def put(self, cache_key: str, data: bytes) -> None:
        filepath = self.get_cache_filepath(cache_key)
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            with open(filepath, 'wb') as f:
                f.write(data)
        except (IOError, OSError) as e:
            print(f"Warning: Failed to cache binary data: {e}")

    def clear(self, cache_key: str | None = None) -> None:
        cache_dir = self.cache_path
        if not os.path.exists(cache_dir):
            return

        if cache_key is None:
            for filename in os.listdir(cache_dir):
                filepath = os.path.join(cache_dir, filename)
                try:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                except (IOError, OSError):
                    pass
        else:
            filepath = self.get_cache_filepath(cache_key)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except (IOError, OSError):
                    pass

    def get_cache_filepath(self, cache_key: str) -> str:
        filename = f"{cache_key}.bin"
        return os.path.join(self.cache_path, filename)
