#!/usr/bin/env python3
"""
Test runner script for SAC v446

SAC v446用のテスト実行スクリプト
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*50}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, cwd=os.getcwd())

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"✗ Error running {description}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run SAC v446 tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--all', action='store_true', help='Run all tests (default)')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fail-fast', action='store_true', help='Stop on first failure')
    parser.add_argument('--no-cov', action='store_true', help='Skip coverage reporting')

    args = parser.parse_args()

    # Default to running all tests
    if not any([args.unit, args.integration, args.performance]):
        args.all = True

    # Build pytest command
    pytest_cmd = [sys.executable, '-m', 'pytest']

    # Add test paths based on selection
    if args.unit or args.all:
        pytest_cmd.extend(['tests/unit/'])
    if args.integration or args.all:
        pytest_cmd.extend(['tests/integration/'])
    if args.performance or args.all:
        pytest_cmd.extend(['tests/performance/'])

    # Add options
    if args.verbose:
        pytest_cmd.append('-v')
    if args.fail_fast:
        pytest_cmd.append('--tb=short')
    if not args.no_cov and (args.coverage or args.all):
        pytest_cmd.extend(['--cov=ztb', '--cov-report=html', '--cov-report=term'])

    # Run tests
    success = run_command(pytest_cmd, "SAC v446 Test Suite")

    if success:
        print(f"\n{'='*50}")
        print("✓ All tests passed!")
        if not args.no_cov and (args.coverage or args.all):
            print("Coverage report generated in htmlcov/")
        print(f"{'='*50}")
        return 0
    else:
        print(f"\n{'='*50}")
        print("✗ Some tests failed!")
        print(f"{'='*50}")
        return 1


if __name__ == '__main__':
    sys.exit(main())