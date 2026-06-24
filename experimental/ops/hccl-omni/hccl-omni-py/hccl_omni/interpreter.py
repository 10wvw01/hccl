from typing import Any, Callable
import torch.distributed as dist
from . import HcclOmniConfig


def alg_synthesizer_mode(
    hccl_omni_cfg: HcclOmniConfig | None = None,
    kernel_config: dict[str, Any] | None = None,
    cluster_config: dict[str, Any] | None = None,
    op_param: dict[str, Any] | None = None,
) -> bytes:
    # Generate XML only on root rank
    if dist.get_rank() == 0:
        # TODO: Implement actual solver call
        xml_content = f"""<?xml version="1.0"?>
<instruction>
    <mode>AlgSynthesizer</mode>
    <cluster_config>{str(cluster_config)}</cluster_config>
    <op_param>{str(op_param)}</op_param>
    <kernel_config>{str(kernel_config)}</kernel_config>
    <generated_by>AlgSynthesizer solver</generated_by>
</instruction>"""
        xml_bytes = xml_content.encode('utf-8')
    else:
        xml_bytes = b""

    return xml_bytes


def dsl_mode(
    hccl_omni_cfg: HcclOmniConfig | None = None,
    kernel_config: dict[str, Any] | None = None,
    cluster_config: dict[str, Any] | None = None,
    op_param: dict[str, Any] | None = None,
    func: Callable | None = None,
) -> bytes:
    # TODO: Implement DSL compiler
    xml_content = f"""<?xml version="1.0"?>
<instruction>
    <mode>DSL</mode>
    <cluster_config>{str(cluster_config)}</cluster_config>
    <op_param>{str(op_param)}</op_param>
    <kernel_config>{str(kernel_config)}</kernel_config>
    <dsl_function>{func.__name__ if func else 'unknown'}</dsl_function>
    <generated_by>DSL compiler</generated_by>
</instruction>"""
    return xml_content.encode('utf-8')


def vanilla_mode(ins_file_path: str | None) -> bytes:
    if ins_file_path is None:
        raise ValueError('ins_file_path is not set in Vanilla mode!')

    try:
        with open(ins_file_path, 'rb') as f:
            xml_bytes = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File not found: {ins_file_path}"
        ) from None
    except IOError as e:
        raise IOError(
            f"Unable to read file: {ins_file_path}: {e}"
        ) from None

    return xml_bytes
