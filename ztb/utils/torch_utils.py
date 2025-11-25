"""
Small helper utilities to standardize torch import and device handling.

Use this in training modules to decide whether CUDA is available and to reserve fallback
for CPU-only execution. This module tries to import torch, but will not error out if
torch isn't installed; instead it returns an indicator.
"""


def is_torch_available() -> bool:
    try:
        pass  # type: ignore
    except Exception:
        return False
    return True


def get_preferred_device() -> str:
    """Return 'cuda' if available else 'cpu'."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def ensure_cpu_mode() -> None:
    """Utility to set torch to CPU mode if possible. No-op if torch not installed.

    This is a convenience for administrative scripts which should avoid enabling
    GPU features when torch could be installed only with GPU support requiring drivers.
    """
    try:
        pass  # type: ignore

        # Nothing to do: simply ensure tensors go to CPU by default at runtime.
        # Users should still set device explicitly in their training code if needed.
    except Exception:
        pass
