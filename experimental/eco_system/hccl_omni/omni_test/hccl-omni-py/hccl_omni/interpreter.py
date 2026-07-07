from __future__ import annotations

from typing import Any, Callable
import torch.distributed as dist
from . algsynthesizer.generate_check_solver_input import *


def alg_synthesizer_mode(
    hccl_omni_cfg: HcclOmniConfig | None = None,
    kernel_config: dict[str, Any] | None = None,
    cluster_config: dict[str, Any] | None = None,
    op_param: dict[str, Any] | None = None,
) -> bytes:
    # Generate XML only on root rank
    path = "/etc/"
    name = hccl_omni_cfg.op_key+"_" + \
        str(op_param['op_name'])+"_" + \
        str(op_param['data_count'])+"_AlgSynthesizer_"
    param = solverParam(name, chunksize=op_param['dataCount'],
                        collective=op_param['opName'], reduce_op=op_param["reduceOp"])
    rootinfo_path = path+"hccl_rootinfo.json"
    ranktable_path = path+"ranktable.json"
    xml_dir_path = "/temp/xml/"
    xml_path = xml_dir_path+name+".xml"

    def load_json(filepath: str) -> Dict[str, Any]:
        """load json file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    try:
        load_json(ranktable_path)
        rootinfo = load_json(rootinfo_path)
        topo_path = rootinfo.get("topo_file_path")
    except:
        raise ValueError(
            f"path:{ranktable_path} lacks ranktable.json or rootinfo load failed")
    config = clusterConfig(name, rootinfo_path=rootinfo_path, ranktable_path=ranktable_path, topo_path=topo_path, xml_path=xml_path,
                           json2xml_path=xml_dir_path, in_bandwidth=45, switch_bandwidth=180, auto_generate=True)
    generate_xml(param, config)
    if dist.get_rank() == 0:
        import os
        try:
            if not os.path.exists(xml_path):
                raise FileNotFoundError(f"XML file not found: {xml_path}")
            # Open file in read mode with UTF-8 encoding
            with open(xml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Encode the string content to UTF-8 bytes
            xml_bytes = content.encode('utf-8')
            print(
                f"✅ Successfully read and encoded XML file: {xml_path} (Size: {len(xml_bytes)} bytes)")
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Encoding error: File {xml_path} is not valid UTF-8.") from e
        except Exception as e:
            # Re-raise other exceptions with a clear message
            raise RuntimeError(
                f"Failed to read XML file {xml_path}: {e}") from e
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
