'''
JIT decorator for HCCL-OMNI communication operators.
'''

from typing import TypeVar, Callable, overload
import functools
import hashlib
import torch
import torch.distributed as dist

from .cache import CacheManager
from .interpreter import alg_synthesizer_mode, dsl_mode, vanilla_mode
from .backend import invoke_backend
from .splitter import split_xml_by_rank
from .converter import xml_to_bin_converter
from . import HcclOmniConfig


def _broadcast_data(data: bytes, src: int = 0) -> bytes:
    '''Broadcast data from source rank to all ranks.'''
    current_rank = dist.get_rank()

    if current_rank == src:
        data_tensor = torch.tensor(list(data), dtype=torch.uint8).npu()
        data_size = torch.tensor([len(data)], dtype=torch.long).npu()
        dist.broadcast(data_size, src=src)
        dist.broadcast(data_tensor, src=src)
        return data
    else:
        data_size = torch.tensor([0], dtype=torch.long).npu()
        dist.broadcast(data_size, src=src)
        data_tensor = torch.zeros(data_size.item(), dtype=torch.uint8).npu()
        dist.broadcast(data_tensor, src=src)
        return bytes(data_tensor.tolist())

T = TypeVar('T')
F = TypeVar('F', bound=Callable)


class JITFunction:
    def __init__(
        self,
        func: Callable,
        version: int = 1,
        launch_metadata: Callable | None = None,
        hccl_omni_cfg: HcclOmniConfig | None = None,
        debug: bool = False,
    ):
        self.func = func
        self.version = version
        self.launch_metadata = launch_metadata
        self.hccl_omni_cfg = hccl_omni_cfg if hccl_omni_cfg is not None else HcclOmniConfig()
        self.debug = debug

        # Initialize cache manager
        self.cache = CacheManager()

        # Ensure distributed environment is ready
        if not dist.is_initialized():
            dist.init_process_group(backend='hccl', init_method='env://')

        # Preserve function metadata
        functools.update_wrapper(self, func)

    def __getitem__(self, kernel_config: dict[str, object]) -> Callable:
        def operator_wrapper(
            cluster_config: dict[str, object],
            op_param: dict[str, object],
            **kwargs
        ) -> object:
            op_key = self._get_op_key(op_param)
            cluster_key = self._get_cluster_key(cluster_config)

            # Get current rank
            current_rank = dist.get_rank()

            # Determine cache key based on user configuration with rank suffix
            if self.hccl_omni_cfg.op_key is not None:
                cache_key = f"{self.hccl_omni_cfg.op_key}.{current_rank}"
            else:
                cache_key = f"{cluster_key}-{op_key}.{current_rank}"

            # Try to get from cache
            cached_xml = self.cache.get(cache_key)
            if cached_xml is not None:
                # Cache hit: get file path and pass to backend
                file_path = self.cache.get_cache_filepath(cache_key)
            else:
                # Cache miss: generate XML
                mode = self.hccl_omni_cfg.ins_gen_mode or 'AlgSynthesizer'

                if mode == 'AlgSynthesizer':
                    xml_bytes = alg_synthesizer_mode(
                        hccl_omni_cfg=self.hccl_omni_cfg,
                        kernel_config=kernel_config,
                        cluster_config=cluster_config,
                        op_param=op_param
                    )
                    # Broadcast complete XML
                    complete_xml = _broadcast_data(xml_bytes, src=0)
                    # Split XML for current rank
                    rank_specific_xml = split_xml_by_rank(complete_xml)
                elif mode == 'DSL':
                    # DSL mode generates rank-specific XML directly
                    rank_specific_xml = dsl_mode(
                        hccl_omni_cfg=self.hccl_omni_cfg,
                        kernel_config=kernel_config,
                        cluster_config=cluster_config,
                        op_param=op_param,
                        func=self.func
                    )
                elif mode == 'Vanilla':
                    xml_bytes = vanilla_mode(
                        ins_file_path=self.hccl_omni_cfg.ins_file_path
                    )
                    # Broadcast for Vanilla mode
                    complete_xml = _broadcast_data(xml_bytes, src=0)
                    # Split XML for current rank
                    rank_specific_xml = split_xml_by_rank(complete_xml)
                else:
                    raise ValueError(f"Unknown instruction generation mode: {mode}")

                # Convert XML to binary format
                binary_data = xml_to_bin_converter(rank_specific_xml)

                # Cache the binary data (not XML)
                self.cache.put(cache_key, binary_data)

                # Get file path
                file_path = self.cache.get_cache_filepath(cache_key)

            # Invoke backend with file path
            result = invoke_backend(op_param, file_path)
            return result

        return operator_wrapper

    def _get_op_key(self, op_param: dict[str, object]) -> str:
        if self.hccl_omni_cfg.op_key is not None:
            return self.hccl_omni_cfg.op_key

        # Otherwise, compute hash from op_param
        return self._dict_hash(op_param)

    def _get_cluster_key(self, cluster_config: dict[str, object]) -> str:
        return self._dict_hash(cluster_config)

    def _dict_hash(self, d: dict[str, object]) -> str:
        def flatten_dict(obj: object) -> tuple:
            if isinstance(obj, dict):
                # Recursively flatten nested dictionaries
                items = []
                for key in sorted(obj.keys()):
                    items.append((key, flatten_dict(obj[key])))
                return tuple(items)
            elif isinstance(obj, (list, tuple)):
                # Flatten sequences
                return tuple(flatten_dict(item) for item in obj)
            else:
                # Base case: immutable value
                return obj

        flattened = flatten_dict(d)
        # Compute hash
        hash_obj = hashlib.sha256(str(flattened).encode())
        return hash_obj.hexdigest()


@overload
def jit(
    func: F,
) -> JITFunction: ...


@overload
def jit(
    *,
    version: int = 1,
    launch_metadata: Callable | None = None,
    hccl_omni_cfg: HcclOmniConfig | None = None,
    debug: bool = False,
) -> Callable[[F], JITFunction]: ...


def jit(
    func: F | None = None,
    *,
    version: int = 1,
    launch_metadata: Callable | None = None,
    hccl_omni_cfg: HcclOmniConfig | None = None,
    debug: bool = False,
) -> JITFunction | Callable[[F], JITFunction]:
    '''
    JIT decorator for HCCL-OMNI communication operators.

    Can be used with or without parameters:

    @hccl_omni.jit
    def my_operator(cluster_config, op_param):
        pass

    @hccl_omni.jit(version=2, debug=True, hccl_omni_cfg=config)
    def my_operator(cluster_config, op_param):
        pass
    '''
    def decorator(f: F) -> JITFunction:
        return JITFunction(
            func=f,
            version=version,
            launch_metadata=launch_metadata,
            hccl_omni_cfg=hccl_omni_cfg,
            debug=debug,
        )

    # Support both @jit and @jit(...) syntax
    if func is None:
        return decorator
    else:
        return decorator(func)
