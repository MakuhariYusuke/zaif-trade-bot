"""Process-wide bootstrap to keep torch DLLs available on Windows."""

from __future__ import annotations

import os


def _bootstrap_torch() -> None:
    # Allow disabling in CI or specialized tooling by toggling an env flag.
    if os.environ.get("ZTB_DISABLE_TORCH_SITE_HOOK") == "1":
        return

    if os.name != "nt":  # Non-Windows environments do not need the guard.
        return

    try:
        from ztb.utils.torch_utils import ensure_torch_dll_search_path
    except Exception:
        return

    try:
        ensure_torch_dll_search_path()
        import torch  # type: ignore  # noqa: F401
    except Exception:
        # Do not crash interpreter startup; diagnostics still available via
        # ztb.utils.torch_utils._TORCH_IMPORT_ERROR if needed later.
        return


_bootstrap_torch()
