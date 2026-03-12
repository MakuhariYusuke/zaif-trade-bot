# Memory management utilities for trading environment
# 取引環境のメモリ管理ユーティリティ

import gc
from pathlib import Path

import pandas as pd
import psutil

from ztb.trading.environment.constants import BYTES_PER_MB
from ztb.utils.logging_utils import get_logger
from ztb.utils.path_utils import ensure_dir

logger = get_logger(__name__)

class MemoryManager:
    """Handles memory management and logging for the trading environment."""

    def __init__(
        self,
        memory_log_path: str | None = None,
        memory_logging_enabled: bool = False,
        memory_log_interval_steps: int = 2000,
        gc_step_interval: int = 0,
    ):
        self._process = psutil.Process()
        self._memory_log_path = Path(memory_log_path) if memory_log_path else None
        self._memory_logging_enabled = memory_logging_enabled
        self._memory_log_interval_steps = memory_log_interval_steps
        self._gc_step_interval = gc_step_interval
        self._last_memory_log_step = 0

        # Initialize memory log file if needed
        if self._memory_log_path and not self._memory_log_path.exists():
            ensure_dir(self._memory_log_path.parent)
            self._memory_log_path.write_text(
                "timestamp,context,df_mb,rss_mb\n", encoding="utf-8"
            )

    def log_memory_usage(
        self, context: str, *, df_override: pd.DataFrame | None = None
    ) -> None:
        """Log memory usage for debugging."""
        """
        Log memory usage for debugging.

        Args:
            context (str): Description or identifier for the logging context.
            df_override (pd.DataFrame | None, optional): DataFrame whose memory usage will be logged instead of the default. Defaults to None.
        """

        rss_mb = self._process.memory_info().rss / BYTES_PER_MB
        target_df = df_override if df_override is not None else pd.DataFrame()
        # NOTE: Using deep=False to avoid Windows C extension SIGINT issues
        # deep=True calls lib.memory_usage_of_objects() which triggers SIGINT
        df_mem_mb = (
            target_df.memory_usage(deep=False).sum() / BYTES_PER_MB
            if isinstance(target_df, pd.DataFrame)
            else 0.0
        )

        log_payload = {
            "event": "memory_usage",
            "context": context,
            "df_mb": round(df_mem_mb, 4),
            "rss_mb": round(rss_mb, 4),
        }

        logger.debug("memory_usage", extra=log_payload)

        if self._memory_log_path is not None:
            ensure_dir(self._memory_log_path.parent)
            with self._memory_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{pd.Timestamp.now().isoformat()},{context},{df_mem_mb:.4f},{rss_mb:.4f}\n"
                )

    def should_log_memory(self, current_step: int) -> bool:
        """Check if memory should be logged at current step."""
        if not self._memory_logging_enabled or not self._memory_log_interval_steps:
            return False

        if current_step % self._memory_log_interval_steps == 0 and (
            current_step != self._last_memory_log_step
        ):
            self._last_memory_log_step = current_step
            return True
        return False

    def should_collect_garbage_at_step(self, current_step: int) -> bool:
        """Check if garbage collection should run at current step."""
        if not self._gc_step_interval:
            return False

        return current_step % self._gc_step_interval == 0

    def collect_garbage(self, generation: int = 2) -> None:
        """Perform garbage collection."""
        gc.collect(generation=generation)

    def collect_garbage_aggressive(self) -> None:
        """Perform aggressive garbage collection (all generations)."""
        collected_count = 0
        for i in range(3):
            collected = gc.collect(generation=i)
            collected_count += collected

    @property
    def is_gc_enabled(self) -> bool:
        """Return True if garbage collection is enabled (step interval is set)."""
        return self._gc_step_interval != 0

    @property
    def should_collect_garbage(self) -> bool:
        """Check if garbage collection should be performed based on step interval."""
        return self._gc_step_interval != 0

    @property
    def memory_logging_enabled(self) -> bool:
        """Get memory logging enabled status."""
        return self._memory_logging_enabled

    @property
    def gc_step_interval(self) -> int:
        """Get garbage collection step interval."""
        return self._gc_step_interval
