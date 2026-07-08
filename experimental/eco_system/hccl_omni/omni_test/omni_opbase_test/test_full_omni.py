#!/usr/bin/env python3
"""
Real-torch integration test — HCCL-OMNI operators vs torch.distributed native.

Verifies Allgather, ReduceScatter, Allreduce, and Alltoall results against
torch native collective APIs.  Requires Ascend NPU + HCCL backend.

Run:
    # Single process (world_size=1)
    pytest testr_full_omni.py -v

    # Single test, distributed (4 NPUs)
    torchrun --nproc_per_node=4 testr_full_omni.py -k test_allreduce

    # All tests, distributed
    torchrun --nproc_per_node=4 testr_full_omni.py
"""

import json
import os

import hccl_omni
import pytest
import torch
import torch.distributed as dist


def create_test_configuration():
    """
    Create test configuration files without automatically setting environment variables.

    Returns:
        tuple: (temp_dir_path, config_file_path, topo_file_path)

    This function is safe to use in both fake_torch and real_torch tests.
    """
    temp_dir = '.hccl_cache'
    os.makedirs(temp_dir, exist_ok=True)

    topo_file = os.path.join(temp_dir, 'topology.json')
    with open(topo_file, 'w', encoding='utf-8') as f:
        f.write('{"nodes": 1}')

    config_file = os.path.join(temp_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({'topo_file_path': topo_file}, f)

    return temp_dir, config_file, topo_file


def _xml_path(name):
    path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'xml_example', name)
    if os.path.isfile(path):
        return path
    return name


# ---------------------------------------------------------------------------
# Per-module distributed setup / teardown
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module', autouse=True)
def _distributed():
    """Initialise torch.distributed once for the whole test module."""
    if not dist.is_initialized():
        dist.init_process_group(backend='hccl', init_method='env://')
    rank = dist.get_rank()
    torch.npu.set_device(rank)

    _, config_file, _ = create_test_configuration()
    os.environ['HCCL_TOPO_FILE_PATH'] = config_file

    yield

    dist.destroy_process_group()


def _rank():
    return dist.get_rank()


def _world_size():
    return dist.get_world_size()


def _device():
    return f'npu:{_rank()}'


# ===================================================================
# Allgather
# ===================================================================

def test_allgather():
    N = 16
    rank = _rank()
    w = _world_size()
    device = _device()

    cfg = hccl_omni.HcclOmniConfig(
        ins_file_path=_xml_path('4pfullmesh.xml'),
        ins_gen_mode='Vanilla',
    )

    @hccl_omni.jit(hccl_omni_cfg=cfg)
    def allgather_op(cluster_config, op_param):
        pass

    inp = (torch.arange(N, dtype=torch.float32,
           device=device) + rank * N).contiguous()
    out_list = [torch.empty(N, dtype=torch.float32, device=device)
                for _ in range(w)]

    kernel_config = {}
    cluster_config = {}
    op_param = {
        'op_name': hccl_omni.OpName.Allgather,
        'input': inp,
        'output': out_list,
    }
    allgather_op[kernel_config](cluster_config, op_param)

    # torch native
    native_list = [torch.empty(
        N, dtype=torch.float32, device=device) for _ in range(w)]
    dist.all_gather(native_list, inp)

    for i in range(w):
        assert torch.allclose(out_list[i], native_list[i]), \
            f'Allgather rank={rank} out[{i}] mismatch'


# ===================================================================
# ReduceScatter
# ===================================================================

def test_reduce_scatter():
    w = _world_size()
    rank = _rank()
    device = _device()
    chunk = 32
    N = chunk * w

    cfg = hccl_omni.HcclOmniConfig(
        ins_file_path=_xml_path('4pfullmesh.xml'),
        ins_gen_mode='Vanilla',
    )

    @hccl_omni.jit(hccl_omni_cfg=cfg)
    def reduce_scatter_op(cluster_config, op_param):
        pass

    inp = (torch.arange(N, dtype=torch.float32,
           device=device) + rank * N).contiguous()
    out = torch.empty(chunk, dtype=torch.float32, device=device)

    kernel_config = {}
    cluster_config = {}
    op_param = {
        'op_name': hccl_omni.OpName.ReduceScatter,
        'input': inp,
        'output': out,
        'reduce_op': hccl_omni.ReduceOp.SUM,
    }
    reduce_scatter_op[kernel_config](cluster_config, op_param)

    # torch native — split inp into w chunks, reduce-scatter
    input_list = list(inp.split(chunk))
    native_out = torch.empty(chunk, dtype=torch.float32, device=device)
    dist.reduce_scatter(native_out, input_list, op=dist.ReduceOp.SUM)

    assert torch.allclose(out, native_out), \
        f'ReduceScatter rank={rank} mismatch'


# ===================================================================
# Allreduce
# ===================================================================

def test_allreduce():
    N = 64
    rank = _rank()
    device = _device()

    cfg = hccl_omni.HcclOmniConfig(
        ins_file_path=_xml_path('4pfullmesh.xml'),
        ins_gen_mode='Vanilla',
    )

    @hccl_omni.jit(hccl_omni_cfg=cfg)
    def allreduce_op(cluster_config, op_param):
        pass

    inp = (torch.arange(N, dtype=torch.float32,
           device=device) + rank * N).contiguous()
    out = torch.empty(N, dtype=torch.float32, device=device)

    kernel_config = {}
    cluster_config = {}
    op_param = {
        'op_name': hccl_omni.OpName.Allreduce,
        'input': inp,
        'output': out,
        'reduce_op': hccl_omni.ReduceOp.SUM,
    }
    allreduce_op[kernel_config](cluster_config, op_param)

    # torch native
    native_out = inp.clone()
    dist.all_reduce(native_out, op=dist.ReduceOp.SUM)

    assert torch.allclose(out, native_out), \
        f'Allreduce rank={rank} mismatch'


# ===================================================================
# Alltoall (unequal split)
# ===================================================================

def test_alltoall():
    w = _world_size()
    rank = _rank()
    device = _device()

    # Deterministic non-uniform split: all ranks share the same send pattern.
    # send_sizes[i] = how many elements rank i sends to each other rank.
    # [2, 4] for w=2; [2, 4, 6, 8] for w=4
    send_sizes = [(i + 1) * 2 for i in range(w)]
    # recv_sizes[j] = how many elements this rank receives from rank j
    recv_sizes = [send_sizes[rank]] * w

    cfg = hccl_omni.HcclOmniConfig(
        ins_file_path=_xml_path('2pfullmesh.xml'),
        ins_gen_mode='Vanilla',
    )

    @hccl_omni.jit(hccl_omni_cfg=cfg)
    def alltoall_op(cluster_config, op_param):
        pass

    send_total = sum(send_sizes)
    recv_total = sum(recv_sizes)

    inp = (torch.arange(send_total, dtype=torch.float32,
           device=device) + rank * 100).contiguous()
    out = torch.empty(recv_total, dtype=torch.float32, device=device)

    kernel_config = {}
    cluster_config = {}
    op_param = {
        'op_name': hccl_omni.OpName.Alltoall,
        'input': inp,
        'output': out,
        'input_split_sizes': send_sizes,
        'output_split_sizes': recv_sizes,
    }
    alltoall_op[kernel_config](cluster_config, op_param)

    # torch native
    native_out = torch.empty(recv_total, dtype=torch.float32, device=device)
    dist.all_to_all_single(native_out, inp, recv_sizes, send_sizes)

    assert torch.allclose(out, native_out), \
        f'Alltoall rank={rank} mismatch'


# ---------------------------------------------------------------------------
# Entry point for torchrun — each rank runs pytest inside its own process.
# torchrun --nproc_per_node=4 testr_full_omni.py -k test_allreduce
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    sys.exit(pytest.main(sys.argv[1:]))
