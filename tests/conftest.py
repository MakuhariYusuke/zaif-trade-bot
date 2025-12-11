import sys
from pathlib import Path

# Workaround for [WinError 1114] DLL initialization failed
# Torch must be imported before pandas/scipy/numpy in some environments
# Catch any Exception during import to avoid crashing test collection in
# environments where torch or its compiled dependencies are incompatible.
try:
    # Guard against numpy/torch ABI mismatches (e.g., numpy 2.x vs torch built against 1.x)
    try:
        import numpy as _np

        np_major = (
            int(_np.__version__.split(".")[0]) if hasattr(_np, "__version__") else 0
        )
    except Exception:
        np_major = 0

    if np_major >= 2:
        # Avoid importing torch in environments where numpy 2.x is present, to prevent
        # segfaults from binary incompatibility. Tests requiring torch should explicitly
        # import or skip based on availability.
        torch = None
    else:
        import torch
except Exception:
    # Keep going; tests that require torch will explicitly mark/skip as needed
    torch = None  # type: ignore

try:
    # Prefer using project's path_utils to locate root
    from ztb.utils.path_utils import get_project_root

    project_root = get_project_root()
except Exception:
    # Fallback: assume repository root is two levels up from tests/
    project_root = Path(__file__).resolve().parent.parent

# Ensure project root is on sys.path for test collection
proj_root_str = str(project_root)
if proj_root_str not in sys.path:
    sys.path.insert(0, proj_root_str)
