#!/usr/bin/env python3
"""Thread lifecycle helpers for distributed callback components."""

from __future__ import annotations

import threading
from typing import Callable

ThreadTarget = Callable[[], None]

class BackgroundThreadController:
    """Provide reusable start/join helpers for background threads."""

    def _start_background_thread(
        self,
        *,
        attr_name: str,
        target: ThreadTarget,
        name: str,
        daemon: bool = True,
    ) -> threading.Thread:
        existing = self._get_background_thread(attr_name)
        if existing and existing.is_alive():
            return existing

        thread = threading.Thread(target=target, name=name, daemon=daemon)
        setattr(self, attr_name, thread)
        thread.start()
        return thread

    def _join_background_thread(self, *, attr_name: str, timeout: float = 5.0) -> None:
        thread = self._get_background_thread(attr_name)
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

    def _get_background_thread(self, attr_name: str) -> threading.Thread | None:
        candidate = getattr(self, attr_name, None)
        if isinstance(candidate, threading.Thread):
            return candidate
        return None
