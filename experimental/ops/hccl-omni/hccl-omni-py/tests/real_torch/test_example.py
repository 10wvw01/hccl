#!/usr/bin/env python3

import os
import sys

# Import test utilities from tests package
# The tests/__init__.py automatically sets up the project path
from tests import create_test_configuration

import torch
import torch.distributed as dist

import hccl_omni


def main():
    '''Run all tests.'''
    # Setup test configuration
    _, config_file, _ = create_test_configuration()
    os.environ['HCCL_TOPO_FILE_PATH'] = config_file

    # Setup distributed environment if using torchrun
    dist.init_process_group(backend='hccl')
    rank = dist.get_rank() if dist.is_initialized() else 0

    torch.npu.set_device(rank)

    is_succ = False
    try:
        print('=' * 60)
        print(f'Running HCCL-OMNI real torch tests (rank={rank})')
        print('=' * 60)

        hccl_omni_config = hccl_omni.HcclOmniConfig(ins_file_path='xml_example/4pfullmesh.xml', ins_gen_mode='Vanilla')
        kernel_config = {}
        cluster_config = {}
        op_param = { 'op_name': hccl_omni.OpName.Alltoall }

        @hccl_omni.jit(version=2, hccl_omni_cfg=hccl_omni_config, debug=True)
        def custom_op(cluster_config, op_param):
            pass

        custom_op[kernel_config](cluster_config, op_param)

        is_succ = True
    finally:
        dist.destroy_process_group()

    if is_succ:
        print('[OK] Test passed!')
        return 0
    else:
        print(f'[FAIL] Test failed!')
        return 1

if __name__ == '__main__':
    sys.exit(main())
