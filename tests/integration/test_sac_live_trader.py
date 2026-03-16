#!/usr/bin/env python3
"""
Integration smoke test for SAC model in the live_trader system.
"""

from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


def test_sac_live_trader() -> None:
    project_root = Path(__file__).resolve().parents[2]
    main_py_path = project_root / "ztb" / "trading" / "live_trader" / "main.py"
    model_path = project_root / "models" / "sac_v420_hold_relaxed.zip"

    if not model_path.exists():
        pytest.skip("live_trader smoke asset is not present in this repository snapshot")

    cmd = [
        sys.executable,
        str(main_py_path),
        "--model-path",
        str(model_path.relative_to(project_root)),
        "--algorithm",
        "sac",
        "--venue",
        "coincheck",
        "--duration",
        "0.005",
        "--dry-run",
    ]

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        f"live_trader smoke failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
