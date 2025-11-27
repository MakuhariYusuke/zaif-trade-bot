import os
from pathlib import Path


def test_quick_mtf_runner_exists():
    path = Path("tools/training/run_quick_mtf_ab.py")
    assert path.exists()
    assert path.stat().st_size > 0
