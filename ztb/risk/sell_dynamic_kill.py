"""136# P1-03: SellDynamicKillManager — sell 動的 kill の抽出・強化.

run_fill_test.py 内の _is_sell_killed / _track_sell_pnl を
単体テスト可能なクラスとして抽出し、以下を追加:
  - テレメトリ記録 (rolling stats を JSON 返却)
  - レジーム別閾値サポート (regime_thresholds)
  - 統計情報 (kill 回数、累計 cooldown サイクル数)

Usage:
    from ztb.risk.sell_dynamic_kill import SellDynamicKillManager, SellKillConfig

    mgr = SellDynamicKillManager(SellKillConfig(
        window=50, threshold_bps=-0.5, resume_window=20,
    ))
    mgr.track(pnl_bps=0.3)
    killed, info = mgr.check_kill()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SellKillConfig:
    """sell dynamic kill 設定."""

    enabled: bool = True
    window: int = 50
    threshold_bps: float = -0.5
    resume_window: int = 20
    #: レジーム別閾値 (レジーム名 → threshold_bps)。
    #: キーが一致すれば基本閾値の代わりに使用。
    regime_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class SellKillTelemetry:
    """テレメトリ情報."""

    killed: bool
    cooldown_remaining: int
    rolling_mean: float | None
    rolling_count: int
    threshold_used: float
    regime: str | None
    total_kills: int
    total_cooldown_cycles: int


class SellDynamicKillManager:
    """sell 動的 kill マネージャ.

    133# P0-10 のロジックを独立クラスに抽出し、
    テレメトリ + レジーム別閾値をサポート。
    """

    __slots__ = (
        "_config",
        "_pnl_history",
        "_cooldown",
        "_total_kills",
        "_total_cooldown_cycles",
    )

    def __init__(self, config: SellKillConfig | None = None) -> None:
        self._config = config or SellKillConfig()
        self._pnl_history: list[float] = []
        self._cooldown: int = 0
        self._total_kills: int = 0
        self._total_cooldown_cycles: int = 0

    @property
    def config(self) -> SellKillConfig:
        return self._config

    def track(self, pnl_bps: float) -> None:
        """sell fill の PnL (bps) を追跡."""
        self._pnl_history.append(pnl_bps)
        # メモリ制限: 最大 window*3
        max_keep = self._config.window * 3
        if len(self._pnl_history) > max_keep:
            self._pnl_history = self._pnl_history[-max_keep:]

    def check_kill(self, regime: str | None = None) -> tuple[bool, SellKillTelemetry]:
        """sell を kill すべきか判定.

        Args:
            regime: 現在のマーケットレジーム名。
                regime_thresholds にキーがあれば、その閾値を使用。

        Returns:
            (killed, telemetry)
        """
        if not self._config.enabled:
            return False, self._make_telemetry(
                killed=False, rolling_mean=None, threshold=self._config.threshold_bps, regime=regime
            )

        # cooldown 中
        if self._cooldown > 0:
            self._cooldown -= 1
            self._total_cooldown_cycles += 1
            return True, self._make_telemetry(
                killed=True, rolling_mean=None, threshold=self._config.threshold_bps, regime=regime
            )

        window = self._config.window
        if len(self._pnl_history) < window:
            return False, self._make_telemetry(
                killed=False, rolling_mean=None, threshold=self._config.threshold_bps, regime=regime
            )

        recent = self._pnl_history[-window:]
        rolling_mean = sum(recent) / len(recent)

        # レジーム別閾値
        threshold = self._config.threshold_bps
        if regime and regime in self._config.regime_thresholds:
            threshold = self._config.regime_thresholds[regime]

        if rolling_mean < threshold:
            self._cooldown = self._config.resume_window
            self._total_kills += 1
            logger.warning(
                f"[136# P1-03] sell dynamic kill activated: "
                f"rolling{window} mean={rolling_mean:.3f}bps < {threshold}bps, "
                f"regime={regime or 'default'}, "
                f"cooldown={self._config.resume_window}, "
                f"total_kills={self._total_kills}"
            )
            return True, self._make_telemetry(
                killed=True, rolling_mean=rolling_mean, threshold=threshold, regime=regime
            )

        return False, self._make_telemetry(
            killed=False, rolling_mean=rolling_mean, threshold=threshold, regime=regime
        )

    def _make_telemetry(
        self,
        killed: bool,
        rolling_mean: float | None,
        threshold: float,
        regime: str | None,
    ) -> SellKillTelemetry:
        return SellKillTelemetry(
            killed=killed,
            cooldown_remaining=self._cooldown,
            rolling_mean=rolling_mean,
            rolling_count=len(self._pnl_history),
            threshold_used=threshold,
            regime=regime,
            total_kills=self._total_kills,
            total_cooldown_cycles=self._total_cooldown_cycles,
        )

    def reset(self) -> None:
        """状態リセット."""
        self._pnl_history.clear()
        self._cooldown = 0
        self._total_kills = 0
        self._total_cooldown_cycles = 0
