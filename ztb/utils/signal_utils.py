"""
Signal handling utilities to mitigate spurious SIGINT on Windows.
"""

from __future__ import annotations

import os
import signal
import threading
from contextlib import contextmanager
from typing import Iterable

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

def configure_signal_handling(policy: str, log: object | None = None) -> None:
    """Configure process-wide signal handling based on policy.

    Policies:
      - default: leave handlers unchanged
      - ignore: ignore SIGINT/SIGTERM/SIGBREAK
      - log: log signals but do not raise KeyboardInterrupt
    """
    policy = (policy or "default").strip().lower()
    if policy in {"default", "none", "off"}:
        return

    if threading.current_thread() is not threading.main_thread():
        _get_logger(log).debug("Signal policy ignored (not in main thread)")
        return

    targets = _signal_targets()
    if policy == "ignore":
        _set_console_ctrl_handler(True, log)
        for sig in targets:
            signal.signal(sig, signal.SIG_IGN)
        _get_logger(log).warning("SIGINT policy=ignore enabled")
        return

    if policy == "log":
        for sig in targets:
            signal.signal(sig, _log_signal)
        _get_logger(log).warning("SIGINT policy=log enabled")
        return

    _get_logger(log).warning("Unknown signal policy: %s", policy)

@contextmanager
def suppress_signals(
    policy: str = "ignore",
    *,
    enabled: bool = True,
    log: object | None = None,
) -> Iterable[None]:
    """Temporarily suppress signals within a critical section."""
    if not enabled or threading.current_thread() is not threading.main_thread():
        yield
        return

    targets = _signal_targets()
    previous = {sig: signal.getsignal(sig) for sig in targets}
    console_disabled = False

    if policy == "ignore":
        console_disabled = _set_console_ctrl_handler(True, log)
        handler = signal.SIG_IGN
    elif policy == "log":
        handler = _log_signal
    else:
        handler = None

    if handler is not None:
        for sig in targets:
            signal.signal(sig, handler)

    try:
        yield
    finally:
        for sig, prev in previous.items():
            signal.signal(sig, prev)
        if console_disabled:
            _set_console_ctrl_handler(False, log)

def _signal_targets() -> list[int]:
    targets = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        targets.append(signal.SIGBREAK)
    return targets

def _log_signal(signum: int, frame) -> None:  # type: ignore[override]
    name = None
    try:
        name = signal.Signals(signum).name
    except Exception:
        name = str(signum)
    logger.error("Signal received: %s", name)

def _set_console_ctrl_handler(ignore: bool, log: object | None) -> bool:
    if os.name != "nt":
        return False

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        result = kernel32.SetConsoleCtrlHandler(None, int(ignore))
        if result == 0:
            _get_logger(log).debug("SetConsoleCtrlHandler failed")
            return False
        return True
    except Exception as exc:
        _get_logger(log).debug("SetConsoleCtrlHandler unavailable: %s", exc)
        return False

def _get_logger(log: object | None):
    if log is None:
        return logger
    return log
