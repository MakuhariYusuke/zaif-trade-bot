"""179# RegimePolicyConfig + CycleStrategy — regime 別制御量の分離.

178# 設計方針:
- FillTestConfig の regime 系パラメータを RegimePolicyConfig に集約
- CycleStrategy Protocol で orchestrator から制御量分岐を外部に押し出す
- orchestrator は self._cycle_strategy.effective_interval(regime) を呼ぶだけ
- 各 Strategy 実装は < 200 行に収める

MAX LINES: 250 (超えたら strategy 実装を別ファイルに分割)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ======================================================================
# RegimePolicyConfig — regime 別制御量の統合設定
# ======================================================================

@dataclass
class RegimePolicyConfig:
    """Regime 別制御量の設定 (C/D/offset/skip_gate).

    FillTestConfig.regime_policy として合成。
    hot-reload 時はこの dataclass 単位で差分更新可能。
    """

    # --- C: Dynamic Cycle Interval ---
    dynamic_cycle_enabled: bool = False
    cycle_intervals: dict[str, float] = field(default_factory=lambda: {
        "ranging": 120.0,
        "trending": 60.0,
        "trending_up": 60.0,
        "trending_down": 60.0,
        "high_vol": 120.0,
    })

    # --- D: Regime-linked Post-Fill Wait ---
    dynamic_wait_enabled: bool = False
    post_fill_wait: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "ranging": {"buy": 30.0, "sell": 90.0},
        "trending_up": {"buy": 15.0, "sell": 45.0},
        "trending_down": {"buy": 45.0, "sell": 15.0},
        "trending": {"buy": 20.0, "sell": 45.0},
        "high_vol": {"buy": 30.0, "sell": 90.0},
    })

    # --- Chase: stale reprice 拡張 ---
    chase_enabled: bool = False
    chase_drift_bps: float = 3.0          # この drift を超えたら即 reprice
    chase_max_reprice: int = 5            # 1 サイクル内 chase 上限
    chase_regimes: list[str] = field(default_factory=lambda: [
        "trending_up", "trending_down", "trending",
    ])

    # --- 停止条件 (C/D 安全弁) ---
    # API エラー率が閾値を超えたら自動で ranging モードにフォールバック
    api_error_rate_threshold: float = 0.03  # 3%
    api_error_window_sec: float = 7200.0    # 2h
    # fill_rate が閾値を下回ったらフォールバック
    fill_rate_floor: float = 0.35
    fill_rate_window_sec: float = 21600.0   # 6h
    # avg pnl30 が閾値を下回ったらフォールバック
    pnl_floor_bps: float = -0.8
    pnl_window_sec: float = 21600.0         # 6h

    @classmethod
    def from_yaml(cls, yaml_cfg: dict) -> RegimePolicyConfig:
        """YAML の regime_policy セクションからパース."""
        kwargs: dict = {}
        rp = yaml_cfg.get("regime_policy", {})
        if not rp:
            return cls()

        # Dynamic Cycle Interval
        dc = rp.get("dynamic_cycle", {})
        if dc.get("enabled") is not None:
            kwargs["dynamic_cycle_enabled"] = dc["enabled"]
        if "intervals" in dc and isinstance(dc["intervals"], dict):
            kwargs["cycle_intervals"] = {
                str(k): float(v) for k, v in dc["intervals"].items()
            }

        # Dynamic Post-Fill Wait
        dw = rp.get("dynamic_wait", {})
        if dw.get("enabled") is not None:
            kwargs["dynamic_wait_enabled"] = dw["enabled"]
        if "waits" in dw and isinstance(dw["waits"], dict):
            waits: dict[str, dict[str, float]] = {}
            for regime, sides in dw["waits"].items():
                if isinstance(sides, dict):
                    waits[str(regime)] = {
                        str(s): float(v) for s, v in sides.items()
                    }
            if waits:
                kwargs["post_fill_wait"] = waits

        # Chase
        ch = rp.get("chase", {})
        if ch.get("enabled") is not None:
            kwargs["chase_enabled"] = ch["enabled"]
        if "drift_bps" in ch:
            kwargs["chase_drift_bps"] = float(ch["drift_bps"])
        if "max_reprice" in ch:
            kwargs["chase_max_reprice"] = int(ch["max_reprice"])
        if "regimes" in ch and isinstance(ch["regimes"], list):
            kwargs["chase_regimes"] = [str(r) for r in ch["regimes"]]

        # Stop conditions
        sc = rp.get("stop_conditions", {})
        for yaml_key, config_key in {
            "api_error_rate_threshold": "api_error_rate_threshold",
            "api_error_window_sec": "api_error_window_sec",
            "fill_rate_floor": "fill_rate_floor",
            "fill_rate_window_sec": "fill_rate_window_sec",
            "pnl_floor_bps": "pnl_floor_bps",
            "pnl_window_sec": "pnl_window_sec",
        }.items():
            if yaml_key in sc:
                kwargs[config_key] = float(sc[yaml_key])

        return cls(**kwargs)


# ======================================================================
# CycleStrategy Protocol
# ======================================================================

@runtime_checkable
class CycleStrategy(Protocol):
    """サイクル制御量を regime に応じて返す Protocol.

    178# 設計: orchestrator は strategy.effective_interval(regime) 等を呼ぶだけ。
    制御量の全分岐は strategy 実装に押し出される。
    """

    def effective_interval(self, regime: str | None) -> float:
        """現在の regime に応じたサイクル間隔 (秒) を返す."""
        ...

    def effective_post_fill_wait(self, side: str, regime: str | None) -> float:
        """regime × side 別の post-fill wait (秒) を返す."""
        ...

    def is_chase_enabled(self, regime: str | None) -> bool:
        """Chase ロジックが有効かどうかを返す."""
        ...

    def chase_drift_bps(self) -> float:
        """Chase 発動の drift 閾値 (bps)."""
        ...

    def chase_max_reprice(self) -> int:
        """1 サイクル内の Chase 最大 reprice 回数."""
        ...


# ======================================================================
# DefaultCycleStrategy — RegimePolicyConfig ベースの実装
# ======================================================================

class DefaultCycleStrategy:
    """RegimePolicyConfig と FillTestConfig を参照する標準 CycleStrategy.

    - dynamic_cycle_enabled=False → config.cycle_interval_sec 固定
    - dynamic_wait_enabled=False → config.post_fill_wait_sec 固定
    - chase_enabled=False → Chase 無効
    """

    def __init__(
        self,
        base_interval: float,
        base_wait_buy: float,
        base_wait_sell: float,
        policy: RegimePolicyConfig,
    ) -> None:
        self._base_interval = base_interval
        self._base_wait_buy = base_wait_buy
        self._base_wait_sell = base_wait_sell
        self._policy = policy
        # 停止条件によるフォールバック状態
        self._fallback_active: bool = False
        self._fallback_until: float = 0.0

    @property
    def policy(self) -> RegimePolicyConfig:
        return self._policy

    def activate_fallback(self, duration_sec: float = 3600.0) -> None:
        """停止条件トリガー: 一定時間 ranging モードにフォールバック."""
        self._fallback_active = True
        self._fallback_until = time.time() + duration_sec
        logger.warning(
            f"[179# CycleStrategy] Fallback activated for {duration_sec:.0f}s "
            f"— all cycle intervals revert to base"
        )

    def _check_fallback(self) -> bool:
        """フォールバック期間中かどうかチェック."""
        if self._fallback_active:
            if time.time() >= self._fallback_until:
                self._fallback_active = False
                logger.info("[179# CycleStrategy] Fallback expired — resuming dynamic mode")
                return False
            return True
        return False

    def effective_interval(self, regime: str | None) -> float:
        """C: regime 別サイクル間隔."""
        if not self._policy.dynamic_cycle_enabled or self._check_fallback():
            return self._base_interval
        if regime is None:
            return self._base_interval
        return self._policy.cycle_intervals.get(regime, self._base_interval)

    def effective_post_fill_wait(self, side: str, regime: str | None) -> float:
        """D: regime × side 別 post-fill wait."""
        if not self._policy.dynamic_wait_enabled or self._check_fallback():
            return self._base_wait_sell if side == "sell" else self._base_wait_buy
        if regime is None:
            return self._base_wait_sell if side == "sell" else self._base_wait_buy
        regime_waits = self._policy.post_fill_wait.get(regime)
        if regime_waits is None:
            return self._base_wait_sell if side == "sell" else self._base_wait_buy
        return regime_waits.get(
            side,
            self._base_wait_sell if side == "sell" else self._base_wait_buy,
        )

    def is_chase_enabled(self, regime: str | None) -> bool:
        """Chase: trending 系 regime 限定で有効."""
        if not self._policy.chase_enabled or self._check_fallback():
            return False
        if regime is None:
            return False
        return regime in self._policy.chase_regimes

    def chase_drift_bps(self) -> float:
        return self._policy.chase_drift_bps

    def chase_max_reprice(self) -> int:
        return self._policy.chase_max_reprice
