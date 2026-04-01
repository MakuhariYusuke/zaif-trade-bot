"""Shared helpers for sidecar retrain schedulers.

SAC / PPO の scheduler で重複しやすい最小責務だけをまとめる。
ここでは result/history/mtime-trigger に限定し、学習ロジック自体は共有しない。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import gc
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Generic, TypeVar

from ztb.utils.memory_utils import clear_cuda_cache


@dataclass(slots=True)
class BaseRetrainResult:
    """Scheduler 1 cycle の共通結果."""

    status: str
    timestamp: str = ""
    model_version: str = ""
    training_time_sec: float = 0.0
    total_timesteps: int = 0
    warm_start: bool = False
    error_message: str = ""
    debug_details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "training_time_sec": round(self.training_time_sec, 1),
            "total_timesteps": self.total_timesteps,
            "warm_start": self.warm_start,
            "error_message": self.error_message,
        }
        if self.debug_details:
            payload["debug_details"] = self.debug_details
        return payload


ConfigT = TypeVar("ConfigT")
ResultT = TypeVar("ResultT")
_MISSING_RESULT = object()


class DataFileRetrainTrigger(Generic[ConfigT]):
    """データファイルの mtime と retrain interval に基づく共通 trigger.

    680#: mtime unchanged でも max_staleness_mult × interval 経過後は
    強制 retrain する時間ベースフォールバックを追加。
    649# の ensure_data_fresh が max_data_stale_hours 内ではデータ更新
    しないため、mtime が変わらず永久に retrain 不発になるバグを修正。
    """

    #: mtime unchanged でも強制 retrain するまでの interval 倍数
    MAX_STALENESS_MULT: float = 3.0

    def __init__(
        self,
        *,
        cfg: ConfigT,
        data_path_getter: Callable[[ConfigT], str],
    ) -> None:
        self.cfg = cfg
        self._data_path_getter = data_path_getter
        self._last_retrain_time = 0.0
        self._last_data_mtime = 0.0
        self._consecutive_failures = 0

    def should_retrain(self) -> tuple[bool, str]:
        now = time.time()
        elapsed = now - self._last_retrain_time
        effective_interval = self._get_effective_interval()
        if elapsed < effective_interval:
            remaining = effective_interval - elapsed
            return False, f"interval_wait ({remaining:.0f}s remaining)"

        data_path = Path(self._data_path_getter(self.cfg))
        if not data_path.exists():
            return False, f"data_not_found: {data_path}"

        try:
            current_mtime = data_path.stat().st_mtime
        except OSError as exc:
            return False, f"stat_failed: {exc}"

        if self._last_data_mtime > 0 and current_mtime <= self._last_data_mtime:
            # 680#: 時間ベースフォールバック — mtime 不変でも
            # MAX_STALENESS_MULT × interval 経過で強制 retrain
            force_threshold = effective_interval * self.MAX_STALENESS_MULT
            if elapsed >= force_threshold:
                return True, (
                    f"time_forced ({elapsed:.0f}s >= "
                    f"{force_threshold:.0f}s threshold)"
                )
            return False, "data_unchanged"

        return True, "data_updated"

    def record_result(self, status: str) -> None:
        self._last_retrain_time = time.time()
        if status == "deployed":
            self._consecutive_failures = 0
        elif status in ("oos_failed", "error"):
            self._consecutive_failures += 1

        data_path = Path(self._data_path_getter(self.cfg))
        if data_path.exists():
            try:
                self._last_data_mtime = data_path.stat().st_mtime
            except OSError:
                pass

    def _get_effective_interval(self) -> float:
        base = float(getattr(self.cfg, "retrain_interval_sec"))
        max_interval = float(getattr(self.cfg, "retrain_interval_max_sec"))
        if self._consecutive_failures > 0:
            backoff_mult = 2.0 ** min(self._consecutive_failures, 4)
            return min(base * backoff_mult, max_interval)
        return base

    @property
    def effective_interval(self) -> float:
        return self._get_effective_interval()


def append_history_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    """JSONL history を 1 行追記する."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def atomic_replace_with_tmp(
    *,
    target_path: Path,
    prefix: str,
    suffix: str,
    writer: Callable[[str], None],
) -> None:
    """Write a file via tmp + replace to avoid partial deploy artifacts."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd_tmp, tmp_path = tempfile.mkstemp(
        dir=str(target_path.parent),
        prefix=prefix,
        suffix=suffix,
    )
    os.close(fd_tmp)
    try:
        writer(tmp_path)
        os.replace(tmp_path, str(target_path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def run_with_timeout(
    *,
    timeout_sec: float,
    target: Callable[[], ResultT],
    timeout_message: str,
) -> ResultT:
    """Run a callable in a daemon thread and fail fast on timeout."""
    result: object = _MISSING_RESULT
    captured_error: BaseException | None = None

    def _target() -> None:
        nonlocal result, captured_error
        try:
            result = target()
        except BaseException as exc:  # pragma: no cover - exercised via wrappers
            captured_error = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=timeout_sec)

    if worker.is_alive():
        raise TimeoutError(timeout_message)
    if captured_error is not None:
        raise captured_error
    if result is _MISSING_RESULT:
        raise RuntimeError("Timed worker returned no result")
    return result  # type: ignore[return-value]


def best_effort_training_cleanup() -> None:
    """Release transient CPU/GPU memory after a retrain cycle."""
    clear_cuda_cache()
    gc.collect()
