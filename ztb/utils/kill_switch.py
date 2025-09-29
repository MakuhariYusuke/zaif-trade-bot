"""
Global Kill Switch for Emergency Shutdown

Provides emergency shutdown capabilities for trading operations.
Can be triggered via file, signal, or API call for immediate safety.
"""

import atexit
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class GlobalKillSwitch:
    """Global kill switch for emergency shutdown of trading operations."""

    def __init__(self, kill_file: Optional[Path] = None):
        """
        Initialize kill switch.

        Args:
            kill_file: Path to kill switch file. Defaults to /tmp/ztb.kill
        """
        self.kill_file = kill_file or Path("/tmp/ztb.kill")
        self._killed = False
        self._shutdown_callbacks: list[Callable[[], None]] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Register cleanup
        atexit.register(self._cleanup)

        # Start file monitor
        self._start_file_monitor()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.warning(f"Received signal {signum}, initiating emergency shutdown")
        self.kill()

    def _start_file_monitor(self):
        """Start monitoring kill file in background thread."""
        self._monitor_thread = threading.Thread(
            target=self._monitor_kill_file, daemon=True
        )
        self._monitor_thread.start()

    def _monitor_kill_file(self):
        """Monitor kill file for creation/deletion."""
        while not self._stop_monitor.is_set():
            if self.kill_file.exists():
                logger.warning(
                    f"Kill file {self.kill_file} detected, initiating shutdown"
                )
                self.kill()
                break
            time.sleep(1.0)  # Check every second

    def kill(self):
        """Trigger emergency shutdown."""
        if self._killed:
            return

        self._killed = True
        logger.critical("EMERGENCY SHUTDOWN TRIGGERED")

        # Execute shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")

        # Stop monitoring
        self._stop_monitor.set()

    def is_killed(self) -> bool:
        """Check if kill switch has been triggered."""
        return self._killed

    def register_shutdown_callback(self, callback: Callable[[], None]):
        """Register callback to execute on shutdown."""
        self._shutdown_callbacks.append(callback)

    def _cleanup(self):
        """Cleanup resources."""
        self._stop_monitor.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)


# Global instance
_kill_switch: Optional[GlobalKillSwitch] = None


def get_kill_switch() -> GlobalKillSwitch:
    """Get global kill switch instance."""
    global _kill_switch
    if _kill_switch is None:
        _kill_switch = GlobalKillSwitch()
    return _kill_switch


def kill_switch_active() -> bool:
    """Check if global kill switch is active."""
    return get_kill_switch().is_killed()
