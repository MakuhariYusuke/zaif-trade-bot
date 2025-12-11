import os
import sys
import traceback


def main() -> int:
    print("PYTHONEXECUTABLE:", sys.executable)
    print("ENVIRONMENT SUMMARY:")
print("  CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("  TORCH_USE_CUDA=", os.environ.get("TORCH_USE_CUDA"))
print(
    "  PATH=", os.environ.get("PATH")[:200] + "..." if os.environ.get("PATH") else None
)
print("  PYTHONPATH=", os.environ.get("PYTHONPATH"))
print("  PYTHONHOME=", os.environ.get("PYTHONHOME"))
print("  VIRTUAL_ENV=", os.environ.get("VIRTUAL_ENV"))

    successful = True
try:
    import importlib

    tmod = importlib.import_module("torch")
    print("torch available:", getattr(tmod, "__version__", "n/a"))
    print("torch cuda available:", getattr(tmod.cuda, "is_available", lambda: False)())
except Exception:
    print("torch import failed:")
    traceback.print_exc()
    successful = False

try:
    import importlib

    importlib.import_module("ztb")
    print("ZTB package imported")
except Exception:
    print("ztb import failed:")
    traceback.print_exc()
    successful = False

try:
    import importlib

    importlib.import_module("ztb.training.unified_trainer.trainer")
    print("trainer imported successfully")
except Exception:
    print("trainer import failed:")
    traceback.print_exc()
    successful = False

    return 0 if successful else 1


if __name__ == '__main__':
    from ztb.utils.cli import run_main

    run_main(main)
