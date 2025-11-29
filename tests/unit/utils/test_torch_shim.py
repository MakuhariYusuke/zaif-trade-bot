import os
import subprocess
import sys


def _run_force_stub_subprocess():
    env = os.environ.copy()
    env["ZTB_FORCE_TORCH_STUB"] = "1"
    # Use -c to run inline code, keep small
    code = (
        "import importlib, ztb.utils.torch_utils as t; t.ensure_cpu_mode(); import torch; "
        "print(getattr(torch, '__version__', 'None'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    return result


def test_force_torch_stub_prints_version_0():
    # Running in a subprocess avoids pollution of local interpreter
    res = _run_force_stub_subprocess()
    assert res.returncode == 0, f"Subprocess failed: {res.stderr}"
    output = res.stdout.strip()
    # When forced stub is active, torch.__version__ should be 0.0.0 (our stub)
    assert output == "0.0.0"
