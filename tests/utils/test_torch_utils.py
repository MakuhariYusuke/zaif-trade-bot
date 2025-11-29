import sys
from importlib import import_module

from ztb.utils.torch_utils import ensure_cpu_mode, is_torch_available


def test_ensure_cpu_mode_injects_stub(monkeypatch):
    # Ensure torch is not importable
    if "torch" in sys.modules:
        monkeypatch.delitem(sys.modules, "torch")

    ensure_cpu_mode()

    # After ensure_cpu_mode, torch should be in sys.modules and have basic attributes
    torch = import_module("torch")
    assert hasattr(torch, "__version__")
    assert hasattr(torch, "nn")
    assert hasattr(torch, "optim")
    assert hasattr(torch.optim, "SGD")
    assert hasattr(torch.optim, "Optimizer")
    assert not is_torch_available() or hasattr(torch, "__version__")
