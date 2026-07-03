#!/usr/bin/env python3
'''
Standalone XML to BIN conversion script for HCCL-OMNI framework.

This script converts XML configuration files to binary format for specified ranks.
It requires only the hccl_omni package installed via pip - no real PyTorch needed.
torch is mocked to bypass the dependency.

Usage:
    python run_xml_to_bin_standalone.py [--input-file PATH] [--output-dir PATH] [--num-ranks N]

Example:
    python run_xml_to_bin_standalone.py --input-file xml_example/4pfullmesh.xml --num-ranks 4
    python run_xml_to_bin_standalone.py --output-dir ./my_bins --num-ranks 2
'''

import os
import sys
import argparse
import importlib
from unittest.mock import Mock


def setup_mock_torch():
    '''
    Setup mock torch environment before importing hccl_omni.
    This bypasses the torch dependency in jit.py.
    '''
    # Create mock torch.distributed module
    mock_dist = Mock()
    mock_dist.is_initialized = Mock(return_value=False)
    mock_dist.init_process_group = Mock(return_value=None)
    mock_dist.get_rank = Mock(return_value=0)
    mock_dist.get_world_size = Mock(return_value=1)

    # Create mock torch module
    mock_torch = Mock()
    mock_torch.distributed = mock_dist
    mock_torch.tensor = Mock()
    mock_torch.zeros = Mock()
    mock_torch.empty = Mock()
    mock_torch.Tensor = Mock()
    mock_torch.long = Mock()
    mock_torch.uint8 = Mock()

    # Pre-install mock modules in sys.modules BEFORE any hccl_omni import
    sys.modules['torch'] = mock_torch
    sys.modules['torch.distributed'] = mock_dist


# Setup mock torch BEFORE any other imports
setup_mock_torch()


def convert_xml_to_bin(input_file: str, output_dir: str, num_ranks: int):
    '''
    Convert XML file to binary format for specified number of ranks.

    Args:
        input_file: Path to input XML file
        output_dir: Directory to save output bin files
        num_ranks: Number of ranks to process

    Returns:
        bool: True if conversion was successful, False otherwise
    '''
    print('Starting XML to BIN conversion...')

    # Resolve input file path
    if not os.path.isabs(input_file):
        input_file = os.path.abspath(input_file)

    if not os.path.exists(input_file):
        print(f'[FAIL] Input file not found: {input_file}')
        return False

    print(f'Input XML file: {input_file}')

    # Read the XML file
    with open(input_file, 'rb') as f:
        xml_content = f.read()

    print(f'XML file size: {len(xml_content)} bytes')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')

    try:
        # Import hccl_omni submodules
        converter = importlib.import_module('hccl_omni.converter')
        splitter = importlib.import_module('hccl_omni.splitter')

        xml_to_bin_converter = converter.xml_to_bin_converter
        split_xml_by_rank = splitter.split_xml_by_rank

        results = []

        for rank in range(num_ranks):
            print(f'\n[rank={rank}] Processing...')

            # Step 1: Split XML by rank
            print(f'  [rank={rank}] Step 1: Splitting XML...')
            rank_specific_xml = split_xml_by_rank(xml_content, rank=rank)
            print(f'  [rank={rank}] Rank-specific XML size: {len(rank_specific_xml)} bytes')

            # Step 2: Convert XML to binary
            print(f'  [rank={rank}] Step 2: Converting XML to binary...')
            binary_data = xml_to_bin_converter(rank_specific_xml)
            print(f'  [rank={rank}] Binary data size: {len(binary_data)} bytes')

            # Step 3: Save bin file
            bin_filename = f'rank_{rank}.bin'
            bin_filepath = os.path.join(output_dir, bin_filename)

            with open(bin_filepath, 'wb') as f:
                f.write(binary_data)

            print(f'  [rank={rank}] Step 3: Bin file saved to: {bin_filepath}')
            print(f'  [rank={rank}] Bin file size: {os.path.getsize(bin_filepath)} bytes')

            # Verify the file was created
            if os.path.exists(bin_filepath):
                results.append(True)
            else:
                results.append(False)

        # Summary
        print('\n' + '=' * 60)
        print('Conversion Summary:')
        print('=' * 60)
        print(f'Input file: {input_file}')
        print(f'Output directory: {output_dir}')
        print(f'Number of ranks: {num_ranks}')
        print(f'Bin files generated:')
        for rank in range(num_ranks):
            bin_filepath = os.path.join(output_dir, f'rank_{rank}.bin')
            size = os.path.getsize(bin_filepath) if os.path.exists(bin_filepath) else 0
            status = '[OK]' if results[rank] else '[FAIL]'
            print(f'  {status} rank_{rank}.bin: {size} bytes')

        print(f'\nNote: Output directory preserved at: {output_dir}')

        if all(results):
            print('\n[OK] XML to BIN conversion completed successfully for all ranks!')
            return True
        else:
            print('\n[FAIL] Some ranks failed to generate bin files')
            return False

    except Exception as e:
        print(f'\n[FAIL] Error during XML to BIN conversion: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    '''Main entry point for the XML to BIN conversion script.'''
    # Get script directory for default output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, 'bin_output')

    parser = argparse.ArgumentParser(
        description='Standalone XML to BIN conversion script for HCCL-OMNI framework.'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        default='xml_example/4pfullmesh.xml',
        help='Input XML file path (default: xml_example/4pfullmesh.xml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=default_output_dir,
        help=f'Output directory for bin files (default: {default_output_dir})'
    )
    parser.add_argument(
        '--num-ranks',
        type=int,
        default=4,
        help='Number of ranks to process (default: 4)'
    )
    args = parser.parse_args()

    print('=' * 60)
    print('HCCL-OMNI XML to BIN Conversion (Standalone)')
    print('=' * 60)

    success = convert_xml_to_bin(
        input_file=args.input_file,
        output_dir=args.output_dir,
        num_ranks=args.num_ranks
    )

    print('=' * 60)
    if success:
        print('[OK] Conversion completed successfully!')
        return 0
    else:
        print('[FAIL] Conversion failed!')
        return 1


if __name__ == '__main__':
    sys.exit(main())