"""136# P1-03: DynamicKillManager — side 別動的 kill の抽出・強化.

run_fill_test.py 内の rolling PnL kill ロジックを
単体テスト可能なクラスとして抽出し、以下を追加:
  - テレメトリ記録 (rolling stats を JSON 返却)
  - レジーム別閾値サポート (regime_thresholds)
  - 統計情報 (kill 回数、累計 cooldown サイクル数)
  - 157# §19: side パラメータで buy/sell 共用 (DRY)

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
class DynamicKillConfig:
    """side 共用 dynamic kill 設定 (157# §19 DRY 化)."""

    enabled: bool = True
    window: int = 50
    threshold_bps: float = -0.5
    resume_window: int = 20
    #: レジーム別閾値 (レジーム名 → threshold_bps)。
    #: キーが一致すれば基本閾値の代わりに使用。
    regime_thresholds: dict[str, float] = field(default_factory=dict)
    #: 218# anti-stagnation: kill 発動中に track() が呼ばれずに
    #: この回数 check_kill() が True を返し続けたら 1 サイクルだけ許可して
    #: 新鮮な PnL データを取得するプローブサイクルを発動する。
    #: 0 = 無効 (従来互換)。
    max_stale_kill_cycles: int = 30

    def __post_init__(self) -> None:
        """173# バリデーション: window/resume_window >= 1."""
        if self.window < 1:
            raise ValueError(f"DynamicKillConfig.window must be >= 1, got {self.window}")
        if self.resume_window < 0:
            raise ValueError(
                f"DynamicKillConfig.resume_window must be >= 0, got {self.resume_window}"
            )

# 後方互換エイリアス
SellKillConfig = DynamicKillConfig

@dataclass
class DynamicKillTelemetry:
    """テレメトリ情報."""

    killed: bool
    cooldown_remaining: int
    rolling_mean: float | None
    rolling_count: int
    threshold_used: float
    regime: str | None
    total_kills: int
    total_cooldown_cycles: int
    side: str = ""

# 後方互換エイリアス
SellKillTelemetry = DynamicKillTelemetry

class DynamicKillManager:
    """side 共用動的 kill マネージャ (157# §19).

    133# P0-10 のロジックを独立クラスに抽出し、
    テレメトリ + レジーム別閾値をサポート。
    side パラメータで buy/sell 両方に対応 (DRY)。
    """

    __slots__ = (
        "_config",
        "_pnl_history",
        "_cooldown",
        "_total_kills",
        "_total_cooldown_cycles",
        "_side",
        "_stale_counter",
        "_total_probe_cycles",
    )

    def __init__(self, config: DynamicKillConfig | None = None, *, side: str = "sell") -> None:
        self._config = config or DynamicKillConfig()
        self._pnl_history: list[float] = []
        self._cooldown: int = 0
        self._total_kills: int = 0
        self._total_cooldown_cycles: int = 0
        self._side = side
        self._stale_counter: int = 0  # 218# anti-stagnation
        self._total_probe_cycles: int = 0

    @property
    def config(self) -> DynamicKillConfig:
        return self._config

    def track(self, pnl_bps: float) -> None:
        """fill の PnL (bps) を追跡."""
        self._pnl_history.append(pnl_bps)
        self._stale_counter = 0  # 218# 新データ投入 → stale リセット
        # メモリ制限: 最大 window*3
        max_keep = self._config.window * 3
        if len(self._pnl_history) > max_keep:
            self._pnl_history = self._pnl_history[-max_keep:]

    def check_kill(self, regime: str | None = None) -> tuple[bool, DynamicKillTelemetry]:
        """kill すべきか判定.

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

        # 218# anti-stagnation: stale probe check
        max_stale = self._config.max_stale_kill_cycles
        if max_stale > 0 and self._stale_counter >= max_stale:
            self._stale_counter = 0
            self._total_probe_cycles += 1
            logger.warning(
                f"[218#] {self._side} dynamic kill probe: "
                f"stale for {max_stale} cycles without new data — "
                f"allowing 1 probe cycle (total_probes={self._total_probe_cycles})"
            )
            self._cooldown = 0  # cooldown もリセット
            return False, self._make_telemetry(
                killed=False, rolling_mean=None, threshold=self._config.threshold_bps, regime=regime
            )

        # cooldown 中
        if self._cooldown > 0:
            self._cooldown -= 1
            self._total_cooldown_cycles += 1
            self._stale_counter += 1  # 218#
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
            self._stale_counter += 1  # 218#
            logger.warning(
                f"[157# §19] {self._side} dynamic kill activated: "
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
    ) -> DynamicKillTelemetry:
        return DynamicKillTelemetry(
            killed=killed,
            cooldown_remaining=self._cooldown,
            rolling_mean=rolling_mean,
            rolling_count=len(self._pnl_history),
            threshold_used=threshold,
            regime=regime,
            total_kills=self._total_kills,
            total_cooldown_cycles=self._total_cooldown_cycles,
            side=self._side,
        )

    def reset(self) -> None:
        """状態リセット."""
        self._pnl_history.clear()
        self._cooldown = 0
        self._total_kills = 0
        self._total_cooldown_cycles = 0
        self._stale_counter = 0
        self._total_probe_cycles = 0

    # ------------------------------------------------------------------
    # 209# H4: 状態永続化 — export / import
    # ------------------------------------------------------------------
    def export_state(self) -> dict[str, object]:
        """永続化用に内部状態を dict にエクスポート.

        Returns:
            pnl_history, cooldown, total_kills, total_cooldown_cycles, side を含む dict。
        """
        return {
            "pnl_history": list(self._pnl_history),
            "cooldown": self._cooldown,
            "total_kills": self._total_kills,
            "total_cooldown_cycles": self._total_cooldown_cycles,
            "side": self._side,
            "stale_counter": self._stale_counter,
            "total_probe_cycles": self._total_probe_cycles,
        }

    def import_state(self, state: dict[str, object]) -> None:
        """export_state() で保存した dict から内部状態を復元.

        Args:
            state: export_state() の戻り値。キーが欠落している場合は
                   デフォルト値 (空リスト / 0) にフォールバック。
        """
        raw_history = state.get("pnl_history", [])
        if isinstance(raw_history, list):
            self._pnl_history = [float(v) for v in raw_history]
        else:
            self._pnl_history = []
        self._cooldown = int(state.get("cooldown", 0))
        self._total_kills = int(state.get("total_kills", 0))
        self._total_cooldown_cycles = int(state.get("total_cooldown_cycles", 0))
        self._stale_counter = int(state.get("stale_counter", 0))
        self._total_probe_cycles = int(state.get("total_probe_cycles", 0))
        # side は import しない (コンストラクタで固定)
        # メモリ制限: window*3 に収める
        max_keep = self._config.window * 3
        if len(self._pnl_history) > max_keep:
            self._pnl_history = self._pnl_history[-max_keep:]

    @property
    def side(self) -> str:
        """管理対象 side."""
        return self._side

# 後方互換エイリアス (136# の import を破壊しない)
SellDynamicKillManager = DynamicKillManager

class BuyDynamicKillManager(DynamicKillManager):
    """157# §19: buy 側動的 kill マネージャ.

    DynamicKillManager(side="buy") のコンビニエンスサブクラス。
    """

    def __init__(self, config: DynamicKillConfig | None = None) -> None:
        super().__init__(config, side="buy")
