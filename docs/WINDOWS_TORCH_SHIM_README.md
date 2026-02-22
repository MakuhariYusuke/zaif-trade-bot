# Windows PyTorch & Test Collection Troubleshooting

When running tests on Windows, importing packages that depend on `torch` may cause a fatal DLL initialization error (e.g. WinError 1114) if the local environment has a GPU-only build of PyTorch or incompatible drivers.

This project provides a lightweight "torch shim" (a small fake module) to reduce import-time crashes during test collection and allow unit tests that don't require real PyTorch to run safely.

## Workarounds & Recommendations

- Option A: Install CPU-only PyTorch wheel on CI/workstations where GPU drivers may not be available:

  - Windows (PowerShell/cmd):
    pip install torch --index-url https://download.pytorch.org/whl/cpu

- Option B: Force the shim for test runs (recommended when your tests don't require real PyTorch):

  - Set environment variable to force the stub:

    - Windows (cmd.exe):
      set ZTB_FORCE_TORCH_STUB=1
      pytest

    - Alternatively, in PowerShell:
      $env:ZTB_FORCE_TORCH_STUB="1"; pytest

  - This will override `torch` with the project's lightweight stub and avoid native DLL initialization.

- Option C: Disable the shim (use real torch even if available):

  - If you have a functioning CPU/GPU environment and need actual PyTorch features, set:

    - Windows (cmd.exe):
      set ZTB_DISABLE_TORCH_STUB=1
      pytest

  - This disables the project shim and uses the installed torch package.

## Notes

- The shim intentionally provides a minimal subset of `torch`'s API surface to satisfy imports from third-party libraries such as stable-baselines3 or sb3_contrib during test collection. It is not a replacement for real PyTorch.
- Tests which actually need CUDA or real numeric operations must run with a real CPU/GPU build of `torch` installed and should not use the shim.
- If you see crashes when importing `opacus` or other libraries that import `torch` at top-level, prefer Option A or Option B for the test environment.

## Debugging Tips

- To verify the shim is active:
  - Run `python -c "import os, importlib; os.environ['ZTB_FORCE_TORCH_STUB'] = '1'; import importlib; import ztb.utils.torch_utils as t; t.ensure_cpu_mode(); import torch; print(getattr(torch, '__version__', None))"` and expect `0.0.0` printed.

- If the shim doesn't appear to be active, ensure tests are executed with the environment variables exported before Python process starts.

## CI

- CI jobs that run on Windows should either use CPU-only torch wheels or set `ZTB_FORCE_TORCH_STUB=1` as a workaround if PyTorch DLLs are causing failures.
