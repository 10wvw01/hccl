#!/usr/bin/env python3
'''
Test XML to BIN conversion flow for HCCL-OMNI framework.

This test reads XML files from xml_example directory, splits them by rank,
converts to binary format, and saves the bin files.

Run:
    python tests/fake_torch/test_xml_to_bin.py [--output-dir /path/to/output]

Mock torch environment is automatically set up by conftest.py.
'''

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path

from hccl_omni.converter import xml_to_bin_converter
from hccl_omni.splitter import split_xml_by_rank
from test_utils import setup_unified_test_environment, cleanup_test_environment


# Set up test configuration
temp_dir, config_file = setup_unified_test_environment()

# Get project root for file operations
project_root = Path(__file__).parent.parent.parent


def test_xml_to_bin_flow(input_file: str, output_dir: str, cleanup_output: bool):
    '''Test the complete XML to BIN conversion flow for all ranks.'''
    print('Starting XML to BIN flow test...')

    # Get the XML file path
    xml_file = os.path.join(project_root, input_file)

    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"XML example file not found: {xml_file}")

    print(f'Reading XML file: {xml_file}')

    # Read the XML file
    with open(xml_file, 'rb') as f:
        xml_content = f.read()

    print(f'XML file size: {len(xml_content)} bytes')

    # Create output directory if not exists
    if not cleanup_output:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Process all 4 ranks
        num_ranks = 4
        results = []

        for rank in range(num_ranks):
            print(f'\n[rank={rank}] Processing...')

            # Step 1: Split XML by rank using split_xml_by_rank
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
        print('Summary:')
        print('=' * 60)
        print(f'Output directory: {output_dir}')
        print(f'Bin files generated:')
        for rank in range(num_ranks):
            bin_filepath = os.path.join(output_dir, f'rank_{rank}.bin')
            size = os.path.getsize(bin_filepath) if os.path.exists(bin_filepath) else 0
            status = '[OK]' if results[rank] else '[FAIL]'
            print(f'  {status} rank_{rank}.bin: {size} bytes')

        if not cleanup_output:
            print(f'\nNote: Output directory preserved at: {output_dir}')

        if all(results):
            print('[OK] XML to BIN flow completed successfully for all ranks!')
            return True
        else:
            print('[FAIL] Some ranks failed to generate bin files')
            return False

    except Exception as e:
        print(f'[FAIL] Error during XML to BIN flow: {e}')
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup temp test directory (but keep output_dir if user specified it)
        cleanup_test_environment(temp_dir, config_file)
        if cleanup_output:
            shutil.rmtree(output_dir, ignore_errors=True)


def main():
    '''Run the XML to BIN flow test.'''
    parser = argparse.ArgumentParser(
        description='Test XML to BIN conversion flow for HCCL-OMNI framework.'
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
        default=None,
        help='Output directory for bin files (default: temp dir, will be cleaned up)'
    )
    args = parser.parse_args()

    print('=' * 60)
    print('HCCL-OMNI XML to BIN Flow Test')
    print('=' * 60)

    # Determine output directory and cleanup flag
    if args.output_dir:
        output_dir = args.output_dir
        cleanup_output = False
    else:
        output_dir = tempfile.mkdtemp(prefix='hccl_xml_to_bin_output_')
        cleanup_output = True

    success = test_xml_to_bin_flow(args.input_file, output_dir, cleanup_output)

    print('=' * 60)
    if success:
        print('[OK] Test passed!')
        return 0
    else:
        print('[FAIL] Test failed!')
        return 1


if __name__ == '__main__':
    sys.exit(main())
