import sys
from pathlib import Path

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
