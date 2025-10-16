#!/usr/bin/env python3
"""
Integration test for SAC model in live_trader system.
Tests the live_trader with sac_v420_hold_relaxed model for 0.025 hours.
"""

import subprocess
import sys
import os

def test_sac_live_trader():
    """Test SAC model integration with live_trader."""

    # Get the absolute path to the project root
    # tests/integration/test_sac_live_trader.py -> tests/integration -> tests -> project_root
    test_file_dir = os.path.dirname(os.path.abspath(__file__))  # tests/integration
    tests_dir = os.path.dirname(test_file_dir)  # tests
    project_root = os.path.dirname(tests_dir)  # project_root

    # Path to main.py
    main_py_path = os.path.join(project_root, 'ztb', 'trading', 'live_trader', 'main.py')

    # Command to run live_trader with SAC model
    cmd = [
        sys.executable,
        main_py_path,
        '--model-path', 'models/sac_v420_hold_relaxed.zip',
        '--algorithm', 'sac',
        '--venue', 'coincheck',
        '--duration', '0.025',
        '--dry-run'
    ]

    print("Running SAC live_trader test...")
    print(f"Command: {' '.join(cmd)}")

    try:
        # Run the command
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )

        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        print(f"Return code: {result.returncode}")

        # Check if the command succeeded
        if result.returncode == 0:
            print("✓ SAC live_trader test completed successfully")
            return True
        else:
            print("✗ SAC live_trader test failed")
            return False

    except subprocess.TimeoutExpired:
        print("✗ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_sac_live_trader()
    sys.exit(0 if success else 1)
