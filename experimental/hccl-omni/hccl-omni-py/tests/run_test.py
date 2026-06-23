#!/usr/bin/env python3
'''
Test runner for HCCL-OMNI framework.
Supports running fake torch tests (with mock) and real torch tests (with torchrun).

Usage:
    # Run fake torch tests (uses mock_torch)
    python tests/run_all.py --mode fake

    # Run real torch tests (requires torchrun)
    python tests/run_all.py --mode real

    # Run real torch tests with specific number of processes
    torchrun --nproc_per_node=4 tests/run_all.py --mode real
'''

import os
import sys
import argparse
import subprocess
from pathlib import Path


def get_tests_dir():
    '''Get the tests directory path.'''
    return Path(__file__).parent


def run_fake_torch_tests():
    '''Run fake torch tests (using mock_torch).'''
    print('HCCL-OMNI Fake Torch Test Suite')
    print('=' * 60)

    tests_dir = get_tests_dir()
    fake_torch_dir = tests_dir / 'fake_torch'

    if not fake_torch_dir.exists():
        print(f'Error: fake_torch directory not found at {fake_torch_dir}')
        return 1

    # Collect all test files
    test_files = []
    for test_file in fake_torch_dir.glob('test_*.py'):
        test_files.append(test_file)

    if not test_files:
        print('No test files found in fake_torch directory')
        return 1

    print(f'Found {len(test_files)} test files:')
    for test_file in test_files:
        print(f'  - {test_file.name}')

    # Get project root
    project_root = tests_dir.parent

    # Run tests
    passed = 0
    failed = 0
    results = []

    for test_file in test_files:
        print(f'\n{"="*60}')
        print(f'Running test: {test_file.name}')
        print(f'{"="*60}')

        try:
            # Set up Python path
            env = os.environ.copy()
            env['PYTHONPATH'] = str(project_root)

            # Run the test
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            # Output results
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f'Error output:\n{result.stderr}')

            # Check return code
            if result.returncode == 0:
                print(f'[OK] Test passed: {test_file.name}')
                passed += 1
                results.append((test_file.name, '[OK] Passed'))
            else:
                print(f'[FAIL] Test failed: {test_file.name} (return code: {result.returncode})')
                failed += 1
                results.append((test_file.name, '[FAIL] Failed'))

        except Exception as e:
            failed += 1
            results.append((test_file.name, f'[ERROR] {e}'))
            print(f'[ERROR] Failed to run {test_file.name}: {e}')

    # Print summary
    print(f'\n{"="*60}')
    print('Test Results Summary:')
    print(f'{"="*60}')

    for test_name, status in results:
        print(f'{test_name:40} {status}')

    print(f'\nTotal: {passed} passed, {failed} failed')

    if failed == 0:
        print('[OK] All tests passed!')
        return 0
    else:
        print(f'[FAIL] {failed} test(s) failed')
        return 1


def run_real_torch_tests():
    print('HCCL-OMNI Real Torch Test Suite')
    print('=' * 60)

    tests_dir = get_tests_dir()
    real_torch_dir = tests_dir / 'real_torch'

    if not real_torch_dir.exists():
        print(f'Error: real_torch directory not found at {real_torch_dir}')
        return 1

    # Collect all test files
    test_files = []
    for test_file in real_torch_dir.glob('test_*.py'):
        test_files.append(test_file)

    if not test_files:
        print('No test files found in real_torch directory')
        return 1

    print(f'Found {len(test_files)} test files:')
    for test_file in test_files:
        print(f'  - {test_file.name}')

    # Check if running under torchrun
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_distributed:
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        print(f'Running in distributed mode (rank={rank}, world_size={world_size})')
    else:
        print('Running in single-process mode')

    # Get project root
    project_root = tests_dir.parent

    # Run tests
    passed = 0
    failed = 0
    results = []

    for test_file in test_files:
        print(f'\n{"="*60}')
        print(f'Running test: {test_file.name}')
        print(f'{"="*60}')

        try:
            # Set up Python path
            env = os.environ.copy()
            env['PYTHONPATH'] = str(project_root)

            # Run the test
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            # Output results
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f'Error output:\n{result.stderr}')

            # Check return code
            if result.returncode == 0:
                print(f'[OK] Test passed: {test_file.name}')
                passed += 1
                results.append((test_file.name, '[OK] Passed'))
            else:
                print(f'[FAIL] Test failed: {test_file.name} (return code: {result.returncode})')
                failed += 1
                results.append((test_file.name, '[FAIL] Failed'))

        except Exception as e:
            failed += 1
            results.append((test_file.name, f'[ERROR] {e}'))
            print(f'[ERROR] Failed to run {test_file.name}: {e}')

    # Print summary
    if is_distributed:
        print(f'\n{"="*60}')
        print(f'Test Results Summary (rank={rank}):')
        print(f'{"="*60}')
    else:
        print(f'\n{"="*60}')
        print('Test Results Summary:')
        print(f'{"="*60}')

    for test_name, status in results:
        print(f'{test_name:40} {status}')

    print(f'\nTotal: {passed} passed, {failed} failed')

    if failed == 0:
        print('[OK] All tests passed!')
        return 0
    else:
        print(f'[FAIL] {failed} test(s) failed')
        return 1


def main():
    '''Main function.'''
    parser = argparse.ArgumentParser(
        description='HCCL-OMNI Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Run fake torch tests (uses mock_torch)
    python tests/run_all.py --mode fake

    # Run real torch tests (single process)
    python tests/run_all.py --mode real

    # Run real torch tests (distributed)
    torchrun --nproc_per_node=4 tests/run_all.py --mode real
        '''
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['fake', 'real'],
        help='Test mode: "fake" for mock_torch tests, "real" for real torch tests'
    )

    args = parser.parse_args()

    try:
        if args.mode == 'fake':
            return run_fake_torch_tests()
        else:
            return run_real_torch_tests()
    except Exception as e:
        print(f'Error running tests: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
