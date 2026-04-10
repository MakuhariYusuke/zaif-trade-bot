"""Shared sidecar scheduler helper tests."""

from __future__ import annotations

from collections.abc import Callable
import logging
import signal
import threading
from typing import cast
from unittest.mock import patch

from ztb.training.sidecar.scheduler_common import install_shutdown_signal_handlers


def test_install_shutdown_signal_handlers_sets_event() -> None:
    shutdown_event = threading.Event()
    logger = logging.getLogger("tests.sidecar_scheduler_common")
    registered_handlers: dict[signal.Signals, object] = {}

    def _capture_signal(sig: signal.Signals, handler: object) -> None:
        registered_handlers[sig] = handler

    with (
        patch(
            "ztb.training.sidecar.scheduler_common.signal.signal",
            side_effect=_capture_signal,
        ),
        patch.object(logger, "warning") as warning_mock,
    ):
        install_shutdown_signal_handlers(
            shutdown_event=shutdown_event,
            logger_obj=logger,
            label="[test]",
        )

        handler = registered_handlers[signal.SIGTERM]
        assert callable(handler)
        cast(Callable[[int, object], None], handler)(signal.SIGTERM, None)

    assert signal.SIGINT in registered_handlers
    assert shutdown_event.is_set()
    warning_mock.assert_called_once_with(
        "%s Received %s — scheduling graceful shutdown",
        "[test]",
        "SIGTERM",
    )
