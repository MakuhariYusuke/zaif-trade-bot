#!/usr/bin/env python3
"""Test: Does importing the script trigger training?"""

import sys
from pathlib import Path

# Project root setup
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("TEST: Attempting to import run_ab_reward_experiments")
print("=" * 70)

# This should NOT trigger training
from scripts.v459 import run_ab_reward_experiments

print("=" * 70)
print("TEST: Import completed without running main()")
print("=" * 70)
