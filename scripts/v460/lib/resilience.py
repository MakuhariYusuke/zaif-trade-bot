"""
fill_test 耐障害性モジュール — 113# CircuitBreaker / HealthMonitor / StatePersistence 統合.

112# §3.1 Tier-1/Tier-2 を fill_test に低侵襲で導入するためのファサード.
既存の ztb/ モジュールを再利用し、fill_test 固有のロジックのみをここに定義.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict

from ztb.utils.dataclass_utils import filter_known_dataclass_fields, shallow_asdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# §1  CircuitBreaker  — API 通信ラッパー
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from ztb.utils.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerOpenException,
        CircuitState,
    )

# re-export
__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenException",
    "CircuitState",
    "FillTestHealthMonitor",
    "FillTestStatePersistence",
    "create_api_circuit_breaker",
]

_CIRCUIT_BREAKER_EXPORTS = frozenset(
    {
        "CircuitBreaker",
        "CircuitBreakerConfig",
        "CircuitBreakerOpenException",
        "CircuitState",
    }
)


def __getattr__(name: str) -> object:
    if name not in _CIRCUIT_BREAKER_EXPORTS:
        raise AttributeError(f"module {__name__} has no attribute {name!r}")
    from ztb.utils import circuit_breaker as cb

    value = getattr(cb, name)
    globals()[name] = value
    return value


def create_api_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 120.0,
    success_threshold: int = 2,
    timeout: float = 30.0,
) -> CircuitBreaker:
    """fill_test 用 API サーキットブレーカーを生成.

    デフォルト: 5 連続失敗 → 120s OPEN → 2 連続成功で CLOSE.
    """
    from ztb.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

    cfg = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=success_threshold,
        timeout=timeout,
    )
    return CircuitBreaker("coincheck_api", cfg)


# ---------------------------------------------------------------------------
# §2  HealthMonitor  — RSS / ディスク / GC 定期チェック
# ---------------------------------------------------------------------------

@dataclass
class HealthThresholds:
    """ヘルスチェック閾値."""

    rss_warn_mb: float = 1500.0      # RSS 警告 (MB)
    rss_critical_mb: float = 2500.0   # RSS 危険 (MB) — OOM 回避
    disk_free_warn_gb: float = 2.0    # ディスク空き警告 (GB)
    gc_interval_cycles: int = 100     # GC 実行間隔 (サイクル数)
    check_interval_sec: float = 60.0   # ヘルスチェック間隔 (秒)


HealthLevel = Literal["ok", "warning", "critical", "unknown"]


class HealthStatus(TypedDict, total=False):
    uptime_sec: float
    cycle_count: int
    rss_mb: float
    cpu_percent: float
    threads: int
    disk_free_gb: float
    level: HealthLevel
    gc_counts: list[int]
    gc_thresholds: list[int]
    pressure_gc_collected: int
    last_pressure_gc_age_sec: float


class _MemoryInfoLike(Protocol):
    rss: int


class _ProcessLike(Protocol):
    def memory_info(self) -> _MemoryInfoLike: ...
    def cpu_percent(self) -> float: ...
    def num_threads(self) -> int: ...


class _DiskUsageLike(Protocol):
    free: int


class _PsutilLike(Protocol):
    def Process(self) -> _ProcessLike: ...
    def disk_usage(self, path: str) -> _DiskUsageLike: ...


class FillTestHealthMonitor:
    """fill_test 向け軽量ヘルスモニター.

    psutil が利用できない環境では gracefully degrade.
    """

    def __init__(self, thresholds: HealthThresholds | None = None) -> None:
        self._thresholds = thresholds or HealthThresholds()
        self._gc_counter = 0
        self._last_check_time = 0.0
        self._start_time = time.time()
        self._last_pressure_gc_time = 0.0
        self._last_pressure_gc_collected = 0
        self._pressure_gc_cooldown_sec = 300.0
        self._psutil_available = False
        self._psutil: _PsutilLike | None = None
        self._process: _ProcessLike | None = None
        try:
            import psutil

            self._psutil = psutil
            self._process = self._psutil.Process()
            self._psutil_available = True
        except ImportError:
            logger.info("[health] psutil not available — RSS monitoring disabled")

    def maybe_check(self, cycle_count: int) -> HealthStatus | None:
        """定期チェック. 閾値超過時は警告ログ + ステータス辞書を返す."""
        now = time.time()
        if now - self._last_check_time < self._thresholds.check_interval_sec:
            return None
        self._last_check_time = now

        status: HealthStatus = {
            "uptime_sec": now - self._start_time,
            "cycle_count": cycle_count,
            "gc_counts": list(gc.get_count()),
            "gc_thresholds": list(gc.get_threshold()),
        }

        if self._psutil_available and self._process is not None and self._psutil is not None:
            rss_mb = self._process.memory_info().rss / (1024 * 1024)
            status["rss_mb"] = round(rss_mb, 1)
            status["cpu_percent"] = self._process.cpu_percent()
            status["threads"] = self._process.num_threads()

            try:
                disk = self._psutil.disk_usage(".")
                status["disk_free_gb"] = round(disk.free / (1024**3), 2)
            except Exception as e:
                # 255# bare except → debug log (disk_usage 例外可観測化)
                logger.debug("disk_usage check failed: %s", e, exc_info=True)

            if rss_mb >= self._thresholds.rss_critical_mb:
                logger.error(
                    f"[health] CRITICAL: RSS={rss_mb:.0f}MB >= "
                    f"{self._thresholds.rss_critical_mb:.0f}MB — OOM risk"
                )
                status["level"] = "critical"
            elif rss_mb >= self._thresholds.rss_warn_mb:
                logger.warning(
                    f"[health] WARNING: RSS {rss_mb:.0f}MB exceeds warn threshold "
                    f"{self._thresholds.rss_warn_mb:.0f}MB"
                )
                status["level"] = "warning"
            else:
                status["level"] = "ok"

            disk_free = status.get("disk_free_gb")
            if isinstance(disk_free, (int, float)) and disk_free < self._thresholds.disk_free_warn_gb:
                logger.warning(
                    f"[health] WARNING: disk_free={disk_free:.2f}GB < "
                    f"{self._thresholds.disk_free_warn_gb:.1f}GB"
                )
                # 209# H1: severity escalation — critical を warning で上書きしない
                if status.get("level") != "critical":
                    status["level"] = "warning"

            if status.get("level") in {"warning", "critical"}:
                collected = self._maybe_pressure_gc(now)
                status["pressure_gc_collected"] = collected
                if self._last_pressure_gc_time > 0:
                    status["last_pressure_gc_age_sec"] = max(
                        now - self._last_pressure_gc_time,
                        0.0,
                    )
        else:
            status["level"] = "unknown"

        return status

    def maybe_gc(self) -> None:
        """定期 GC. gc_interval_cycles ごとに gc.collect() を実行."""
        self._gc_counter += 1
        if self._gc_counter >= self._thresholds.gc_interval_cycles:
            collected = gc.collect()
            self._gc_counter = 0
            if collected > 0:
                logger.debug(f"[health] GC collected {collected} objects")

    def snapshot_memory_diagnostics(
        self,
        *,
        now_ts: float | None = None,
    ) -> dict[str, int | float | list[int]]:
        """Exit diagnostics 向けの軽量メモリ状態スナップショット."""
        snapshot: dict[str, int | float | list[int]] = {
            "gc_counts": list(gc.get_count()),
            "gc_thresholds": list(gc.get_threshold()),
            "gc_cycle_counter": self._gc_counter,
            "last_pressure_gc_collected": self._last_pressure_gc_collected,
            "pressure_gc_cooldown_sec": self._pressure_gc_cooldown_sec,
        }
        if self._last_pressure_gc_time > 0:
            reference_now = time.time() if now_ts is None else now_ts
            snapshot["last_pressure_gc_age_sec"] = max(
                reference_now - self._last_pressure_gc_time,
                0.0,
            )
        return snapshot

    def _maybe_pressure_gc(self, now: float) -> int:
        """RSS 圧迫時に追加 GC を走らせ、連打は cooldown で抑える."""
        if now - self._last_pressure_gc_time < self._pressure_gc_cooldown_sec:
            return 0
        collected = gc.collect()
        self._last_pressure_gc_time = now
        self._last_pressure_gc_collected = collected
        logger.warning(
            "[health] pressure GC collected %d objects after memory warning",
            collected,
        )
        return collected


# ---------------------------------------------------------------------------
# §3  StatePersistence  — fill_test 状態の JSON 永続化
# ---------------------------------------------------------------------------

@dataclass
class FillTestState:
    """fill_test の永続化対象状態."""

    # 基本
    run_id: str = ""
    cycle_count: int = 0
    total_count: int = 0
    filled_count: int = 0
    cumulative_pnl_jpy: float = 0.0
    # ロット
    current_lot: float = 0.001
    soft_loss_cap_triggered: bool = False
    # 適応パラメータ
    base_offset_ratio: float = 0.0010
    base_offset_ratio_buy: float | None = None
    base_offset_ratio_sell: float | None = None
    # 121# A4: regime state persistence — 再起動時 warm-up 不要化
    regime_confirmed: str = "unknown"  # FillTestRegime.value
    regime_stability: int = 0
    regime_prices: list[list[float]] | None = None  # [[ts, price], ...]
    regime_raw_history: list[str] | None = None  # [regime_value, ...]
    # 168# §4.1 #3: 日次ドローダウンガード状態
    daily_drawdown_state: dict[str, object] | None = None
    # 207# §1: toxic veto 永続化 (再起動時に veto 状態を復元)
    toxic_veto: dict[str, int] | None = None
    # 210# L-2: one-sided 連続実行カウンタ永続化 (再起動時に復元)
    one_sided_consecutive_count: int = 0
    # 224# B1: soft drawdown interval 乗数永続化 (再起動時に復元)
    soft_drawdown_interval_multiplier: float = 1.0
    # 216# E: Guard 発火カウンタ永続化 (累積。再起動時に復元)
    guard_fire_counts: dict[str, int] | None = None
    # 244# Guard reason category totals (市場都合/システム都合/回復動作)
    guard_category_totals: dict[str, int] | None = None
    # 209# H4: DynamicKillManager 状態永続化 (rolling PnL window + cooldown)
    sell_kill_state: dict[str, object] | None = None
    buy_kill_state: dict[str, object] | None = None
    # 225# MCB/SAD 状態永続化 (再起動時に price_buffer/halt_until を復元)
    mcb_state: dict[str, object] | None = None
    sad_state: dict[str, object] | None = None
    # 236# 234# エスカレーション・縮退カウンタ永続化
    degraded_liquidation_duty_counter: int = 0
    # 269# Inventory Escape Mode duty cycle カウンタ
    inventory_escape_duty_counter: int = 0
    one_sided_cooldown_remaining: int = 0
    one_sided_freeze_remaining: int = 0
    # 254# 250# P1-4 永続化漏れ修正: freeze/cooldown の対象 side
    one_sided_frozen_side: str | None = None
    consecutive_no_feasible: dict[str, int] | None = None
    # 237# phantom position guard メトリクス永続化
    phantom_guard_metrics: dict[str, int | float] | None = None
    # タイムスタンプ
    saved_at: float = 0.0
    saved_at_iso: str = ""


class FillTestStatePersistence:
    """fill_test 状態の JSON 永続化.

    ztb/trading/production/state_persistence.py を利用し、
    fill_test 固有のフィールドを FillTestState として管理.
    """

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / "fill_test_state.json"

    def save(self, state: FillTestState) -> None:
        """状態を JSON に保存."""
        from ztb.io import write_state_payload

        state.saved_at = time.time()
        state.saved_at_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            write_state_payload(self._state_file, shallow_asdict(state))
            logger.debug(f"[state] Saved: cycle={state.cycle_count}, pnl={state.cumulative_pnl_jpy:.1f}")
        except Exception as e:
            logger.warning(f"[state] Failed to save: {e}")

    def load(self) -> FillTestState | None:
        """状態を JSON から復元. ファイルなし/パースエラーは None."""
        if not self._state_file.exists():
            return None
        try:
            from ztb.io import read_state_payload

            data = read_state_payload(self._state_file)
            filtered = filter_known_dataclass_fields(FillTestState, data)
            return FillTestState(**filtered)
        except Exception as e:
            logger.warning(f"[state] Failed to load: {e}")
            return None

    @property
    def state_file(self) -> Path:
        """状態ファイルパス."""
        return self._state_file
