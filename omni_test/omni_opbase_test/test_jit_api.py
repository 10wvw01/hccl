#!/usr/bin/env python3
"""
Test hccl_omni high-level JIT API with real PyTorch.

Uses HcclOmniConfig + @hccl_omni.jit to handle XML conversion and
running together, similar to the user-provided pattern.
"""

import os
import sys

import torch
import torch.distributed as dist
import torch_npu

import hccl_omni
from hccl_omni import HcclOmniConfig, OpName


def main():
    # Force AICPU_TS expansion mode
    os.environ['HCCL_OP_EXPANSION_MODE'] = 'AI_CPU'
    os.environ.setdefault('HCCL_BUFFSIZE', '200')

    # Init torch.distributed with HCCL backend
    dist.init_process_group(backend='hccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.npu.set_device(rank)

    print(f'[Rank {rank}] Initialized: rank={rank}, world_size={world_size}')

    try:
        # Configure OMNI with Vanilla mode (read existing XML file)
        xml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'xml_example', '2pfullmesh.xml'
        )
        if not os.path.isfile(xml_path):
            xml_path = '/home/yhb/omni_opbase_test/xml_example/2pfullmesh.xml'

        hccl_omni_config = HcclOmniConfig(
            ins_file_path=xml_path,
            ins_gen_mode='Vanilla'
        )

        # Define the operator with JIT decorator
        @hccl_omni.jit(version=2, hccl_omni_cfg=hccl_omni_config, debug=True)
        def alltoallv_op(cluster_config, op_param):
            pass

        # Per-rank counts for AlltoAllV (equal split, 1024 elements per rank)
        data_count = 1024 * world_size
        per_rank_count = data_count // world_size

        op_param = {
            'op_name': OpName.Alltoallv,
            'reduce_op': 0,
            'data_count': data_count,
            'send_counts': [per_rank_count] * world_size,
            'recv_counts': [per_rank_count] * world_size,
            'sdispls': [i * per_rank_count for i in range(world_size)],
            'rdispls': [i * per_rank_count for i in range(world_size)],
        }

        kernel_config = {}
        cluster_config = {}

        # Invoke the operator
        result = alltoallv_op[kernel_config](cluster_config, op_param)

        print(f'[Rank {rank}] JIT API call completed successfully')

    except Exception as e:
        print(f'[Rank {rank}] EXCEPTION: {e}')
        import traceback
        traceback.print_exc()
    finally:
        dist.destroy_process_group()

    if rank == 0:
        print('=== test_jit_api.py Done ===')


if __name__ == '__main__':
    main()
