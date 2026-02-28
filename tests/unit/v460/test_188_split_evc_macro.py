"""188# テスト: ファイル分割 + Phase C ev_weighted + Phase D Macro Regime.

- regime_policy.py → cycle_strategy.py 分割の後方互換性
- fill_cycle_executor.py _build_fill_record 抽出
- Phase C: SkipGate ev_weighted デュアルモデル統合判定
- Phase D: MacroRegimeDetector 基盤
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ======================================================================
# 1. regime_policy → cycle_strategy 分割の後方互換テスト
# ======================================================================


class TestSplitBackwardCompat:
    """188# 分割後も既存 import パスが動作することを検証."""

    def test_import_from_regime_policy(self) -> None:
        """regime_policy から DefaultCycleStrategy を import できること."""
        from scripts.v460.lib.regime_policy import (
            CycleStrategy,
            DefaultCycleStrategy,
            RegimePolicyConfig,
        )
        assert RegimePolicyConfig is not None
        assert CycleStrategy is not None
        assert DefaultCycleStrategy is not None

    def test_import_from_cycle_strategy(self) -> None:
        """cycle_strategy から直接 DefaultCycleStrategy を import できること."""
        from scripts.v460.lib.cycle_strategy import DefaultCycleStrategy
        assert DefaultCycleStrategy is not None

    def test_same_class_identity(self) -> None:
        """両 import パスで同一クラスを取得すること."""
        from scripts.v460.lib.cycle_strategy import DefaultCycleStrategy as Direct
        from scripts.v460.lib.regime_policy import DefaultCycleStrategy as Reexport
        assert Direct is Reexport

    def test_default_cycle_strategy_protocol_compliance(self) -> None:
        """DefaultCycleStrategy が CycleStrategy Protocol を満たすこと."""
        from scripts.v460.lib.regime_policy import CycleStrategy, RegimePolicyConfig
        from scripts.v460.lib.cycle_strategy import DefaultCycleStrategy

        policy = RegimePolicyConfig()
        strategy = DefaultCycleStrategy(120.0, 30.0, 90.0, policy)
        assert isinstance(strategy, CycleStrategy)

    def test_all_methods_available(self) -> None:
        """分割後も全 Protocol メソッドが使えること."""
        from scripts.v460.lib.cycle_strategy import DefaultCycleStrategy
        from scripts.v460.lib.regime_policy import RegimePolicyConfig

        policy = RegimePolicyConfig(dynamic_cycle_enabled=True, chase_enabled=True)
        s = DefaultCycleStrategy(120.0, 30.0, 90.0, policy)

        assert s.effective_interval("ranging") == 120.0
        assert s.effective_post_fill_wait("sell", "ranging") == 90.0
        assert isinstance(s.is_chase_enabled("trending_up", "buy"), bool)
        assert s.chase_drift_bps() == 3.0
        assert s.chase_max_reprice() == 5

    def test_gated_regime_hysteresis(self) -> None:
        """分割後もヒステリシスが正しく動作すること."""
        from scripts.v460.lib.cycle_strategy import DefaultCycleStrategy
        from scripts.v460.lib.regime_policy import RegimePolicyConfig

        policy = RegimePolicyConfig(
            trend_min_confidence=0.45,
            trend_exit_confidence=0.30,
            trend_min_dwell=2,
        )
        s = DefaultCycleStrategy(120.0, 30.0, 90.0, policy)

        # Enter: conf >= 0.45
        assert s.gated_regime("trending_up", 0.50) == "trending_up"
        # Dwell 1 → min_dwell=2 なので exit しない
        assert s.gated_regime("trending_up", 0.25) == "trending_up"
        # Dwell 2 → exit 可能
        assert s.gated_regime("trending_up", 0.20) == "ranging"


# ======================================================================
# 2. Phase C: SkipGate ev_weighted テスト
# ======================================================================


@dataclass
class _MockSkipGateConfig:
    mode: str = "pnl"
    as_threshold: float = 0.5
    threshold_bps: float = 0.0
    max_skip_rate: float = 0.7
    buy_enabled: bool = True
    sell_enabled: bool = True
    as_threshold_buy: float | None = None
    as_threshold_sell: float | None = None
    use_ob_features: bool = False
    adaptive_threshold: bool = False
    target_skip_rate_buy: float = 0.1
    target_skip_rate_sell: float = 0.2
    adaptive_window: int = 50
    adaptive_min_samples: int = 20
    adaptive_step: float = 0.05
    adaptive_floor: float = 0.35
    adaptive_ceiling: float = 0.80
    regime_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class _MockSkipDecision:
    should_skip: bool = False
    predicted_pnl_bps: float = 0.0
    threshold_bps: float = 0.0
    features_used: int = 10
    reason: str = "pass"
    model_used: str = "primary"
    as_probability: float | None = None
    threshold_used: float | None = 0.0

    # _SkipDecisionLike Protocol 互換


class _MockGate:
    """SkipGate モック — evaluate 返却値をコントロール."""

    def __init__(self, pred_pnl: float = 0.5) -> None:
        self.config = _MockSkipGateConfig()
        self.metadata: dict[str, object] = {"target": "pnl30"}
        self.feature_cols = ["f1", "f2"]
        self._pred_pnl = pred_pnl

    def evaluate(
        self,
        features: dict[str, object],
        *,
        side: str | None = None,
        regime: str | None = None,
        threshold_offset: float = 0.0,
    ) -> _MockSkipDecision:
        return _MockSkipDecision(
            should_skip=self._pred_pnl < (self.config.threshold_bps + threshold_offset),
            predicted_pnl_bps=self._pred_pnl,
            threshold_bps=self.config.threshold_bps,
            threshold_used=self.config.threshold_bps + threshold_offset,
        )


class TestEvWeightedDecision:
    """188# C-1: _try_ev_weighted_decision テスト."""

    def _make_evaluator(
        self,
        *,
        ev_enabled: bool = True,
        gate_alt_buy: object = None,
        gate_alt_sell: object = None,
        ev_w30: float = 0.4,
        ev_w120: float = 0.6,
    ) -> object:
        """最小限の SkipGateEvaluator モックを構築."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        config = MagicMock()
        config.skip_gate_enabled = False  # __init__ でモデルロードしない
        config.skip_gate_ev_weighted_enabled = ev_enabled
        config.skip_gate_ev_w30 = ev_w30
        config.skip_gate_ev_w120 = ev_w120
        # 190# 追加フィールド (MagicMock のまま放置すると > 演算エラー)
        config.skip_gate_ev_max_consecutive_skip = 0
        config.skip_gate_ev_one_sided_threshold_shift = 0.0

        evaluator = SkipGateEvaluator(config, Path("."))
        evaluator._gate_alt_buy = gate_alt_buy
        evaluator._gate_alt_sell = gate_alt_sell
        return evaluator

    def test_ev_weighted_disabled_returns_none(self) -> None:
        """ev_weighted_enabled=False → None 返却."""
        evaluator = self._make_evaluator(ev_enabled=False)
        primary = _MockSkipDecision(predicted_pnl_bps=1.0)
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is None

    def test_no_alt_model_returns_none(self) -> None:
        """alt モデルなし → None."""
        evaluator = self._make_evaluator(ev_enabled=True)
        primary = _MockSkipDecision(predicted_pnl_bps=1.0)
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is None

    def test_buy_ev_weighted_combination(self) -> None:
        """buy: ev = w30 * primary + w120 * alt."""
        alt_buy = _MockGate(pred_pnl=2.0)  # pnl120
        evaluator = self._make_evaluator(
            ev_enabled=True,
            gate_alt_buy=alt_buy,
            ev_w30=0.4,
            ev_w120=0.6,
        )
        primary = _MockSkipDecision(
            predicted_pnl_bps=1.0,  # pnl30
            threshold_used=0.0,
            threshold_bps=0.0,
            features_used=10,
        )
        result = evaluator._try_ev_weighted_decision(
            "buy", {"f1": 1.0}, "ranging", 0.0, primary,
        )
        assert result is not None
        # ev = 0.4 * 1.0 + 0.6 * 2.0 = 1.6
        assert abs(result.predicted_pnl_bps - 1.6) < 0.01
        assert result.should_skip is False
        assert "ev_weighted" in result.reason

    def test_sell_ev_weighted_combination(self) -> None:
        """sell: ev = w30 * alt + w120 * primary."""
        alt_sell = _MockGate(pred_pnl=0.5)  # pnl30
        evaluator = self._make_evaluator(
            ev_enabled=True,
            gate_alt_sell=alt_sell,
            ev_w30=0.4,
            ev_w120=0.6,
        )
        primary = _MockSkipDecision(
            predicted_pnl_bps=1.0,  # pnl120
            threshold_used=0.0,
            threshold_bps=0.0,
            features_used=10,
        )
        result = evaluator._try_ev_weighted_decision(
            "sell", {"f1": 1.0}, "ranging", 0.0, primary,
        )
        assert result is not None
        # ev = 0.4 * 0.5 + 0.6 * 1.0 = 0.8
        assert abs(result.predicted_pnl_bps - 0.8) < 0.01

    def test_ev_weighted_skip_decision(self) -> None:
        """ev_weighted score < threshold → should_skip=True."""
        alt_buy = _MockGate(pred_pnl=-3.0)  # pnl120 = -3.0
        evaluator = self._make_evaluator(
            ev_enabled=True,
            gate_alt_buy=alt_buy,
        )
        primary = _MockSkipDecision(
            predicted_pnl_bps=1.0,  # pnl30 = 1.0
            threshold_used=0.0,
            threshold_bps=0.0,
            features_used=10,
        )
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is not None
        # ev = 0.4 * 1.0 + 0.6 * (-3.0) = -1.4 < 0.0
        assert result.should_skip is True
        assert "skip" in result.reason

    def test_as_mode_alt_returns_none(self) -> None:
        """alt model が AS mode → ev_weighted 不適用."""
        alt_buy = _MockGate(pred_pnl=1.0)
        alt_buy.config.mode = "as"
        evaluator = self._make_evaluator(
            ev_enabled=True,
            gate_alt_buy=alt_buy,
        )
        primary = _MockSkipDecision(predicted_pnl_bps=1.0)
        result = evaluator._try_ev_weighted_decision(
            "buy", {}, "ranging", 0.0, primary,
        )
        assert result is None

    def test_config_yaml_parsing(self) -> None:
        """FillTestConfig YAML パースで ev_weighted 設定が反映されること."""
        from scripts.v460.lib.fill_config import FillTestConfig

        yaml_cfg = {
            "skip_gate": {
                "ev_weighted_enabled": True,
                "ev_w30": 0.3,
                "ev_w120": 0.7,
                "model_path_buy_long": "models/test_buy_long.pkl",
                "model_path_sell_short": "models/test_sell_short.pkl",
            }
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.skip_gate_ev_weighted_enabled is True
        assert config.skip_gate_ev_w30 == 0.3
        assert config.skip_gate_ev_w120 == 0.7
        assert config.skip_gate_model_path_buy_long == "models/test_buy_long.pkl"
        assert config.skip_gate_model_path_sell_short == "models/test_sell_short.pkl"


# ======================================================================
# 3. Phase D: MacroRegimeDetector テスト
# ======================================================================


class TestMacroRegimeDetector:
    """188# D: MacroRegimeDetector 基盤テスト."""

    def test_insufficient_data(self) -> None:
        """データ不足時は INSUFFICIENT."""
        from scripts.v460.lib.macro_regime import MacroRegimeDetector, MacroTrend

        det = MacroRegimeDetector()
        result = det.update(time.time(), 10_000_000)
        assert result.trend == MacroTrend.INSUFFICIENT

    def test_neutral_flat_prices(self) -> None:
        """横ばい価格 → NEUTRAL."""
        from scripts.v460.lib.macro_regime import (
            MacroRegimeConfig,
            MacroRegimeDetector,
            MacroTrend,
        )

        cfg = MacroRegimeConfig(bucket_sec=1.0, slope_window_5m=5)
        det = MacroRegimeDetector(cfg)
        t0 = time.time()
        for i in range(30):
            det.update(t0 + i * 2.0, 10_000_000)

        result = det.update(t0 + 60, 10_000_000)
        assert result.trend == MacroTrend.NEUTRAL
        assert abs(result.slope_5m_bps_per_min) < 1.0

    def test_uptrend_detection(self) -> None:
        """上昇トレンド → WEAK_UP or STRONG_UP."""
        from scripts.v460.lib.macro_regime import (
            MacroRegimeConfig,
            MacroRegimeDetector,
            MacroTrend,
        )

        cfg = MacroRegimeConfig(
            bucket_sec=1.0,
            slope_window_5m=5,
            slope_threshold_bps_per_min=0.5,
        )
        det = MacroRegimeDetector(cfg)
        t0 = time.time()
        base = 10_000_000
        for i in range(20):
            # 毎秒 100 JPY 上昇 → ~1 bps/sec → ~60 bps/min
            price = base + i * 100
            det.update(t0 + i * 2.0, price)

        result = det.update(t0 + 40, base + 2000)
        assert result.trend in (MacroTrend.WEAK_UP, MacroTrend.STRONG_UP)
        assert result.slope_5m_bps_per_min > 0

    def test_downtrend_detection(self) -> None:
        """下降トレンド → WEAK_DOWN or STRONG_DOWN."""
        from scripts.v460.lib.macro_regime import (
            MacroRegimeConfig,
            MacroRegimeDetector,
            MacroTrend,
        )

        cfg = MacroRegimeConfig(
            bucket_sec=1.0,
            slope_window_5m=5,
            slope_threshold_bps_per_min=0.5,
        )
        det = MacroRegimeDetector(cfg)
        t0 = time.time()
        base = 10_000_000
        for i in range(20):
            price = base - i * 100
            det.update(t0 + i * 2.0, price)

        result = det.update(t0 + 40, base - 2000)
        assert result.trend in (MacroTrend.WEAK_DOWN, MacroTrend.STRONG_DOWN)
        assert result.slope_5m_bps_per_min < 0

    def test_to_dict(self) -> None:
        """MacroRegimeResult.to_dict() が正しく辞書化されること."""
        from scripts.v460.lib.macro_regime import MacroRegimeResult, MacroTrend

        result = MacroRegimeResult(
            trend=MacroTrend.STRONG_UP,
            slope_5m_bps_per_min=2.5,
            slope_15m_bps_per_min=1.8,
            confidence=0.75,
            buckets_available=15,
        )
        d = result.to_dict()
        assert d["trend"] == "macro_strong_up"
        assert d["slope_5m"] == 2.5
        assert d["confidence"] == 0.75

    def test_compose_aligned(self) -> None:
        """micro/macro 一致 → aligned=True."""
        from scripts.v460.lib.macro_regime import (
            MacroRegimeResult,
            MacroTrend,
            compose_regimes,
        )

        macro = MacroRegimeResult(
            trend=MacroTrend.STRONG_UP,
            confidence=0.8,
            buckets_available=20,
        )
        regime, aligned = compose_regimes("trending_up", 0.6, macro)
        assert regime == "trending_up"
        assert aligned is True

    def test_compose_conflict(self) -> None:
        """micro=trending_up, macro=strong_down → aligned=False."""
        from scripts.v460.lib.macro_regime import (
            MacroRegimeResult,
            MacroTrend,
            compose_regimes,
        )

        macro = MacroRegimeResult(
            trend=MacroTrend.STRONG_DOWN,
            confidence=0.8,
            buckets_available=20,
        )
        regime, aligned = compose_regimes("trending_up", 0.6, macro)
        assert aligned is False

    def test_compose_insufficient_macro(self) -> None:
        """macro INSUFFICIENT → always aligned."""
        from scripts.v460.lib.macro_regime import (
            MacroRegimeResult,
            MacroTrend,
            compose_regimes,
        )

        macro = MacroRegimeResult(
            trend=MacroTrend.INSUFFICIENT,
            buckets_available=3,
        )
        regime, aligned = compose_regimes("trending_down", 0.7, macro)
        assert aligned is True

    def test_bucket_overflow(self) -> None:
        """バケット数が max_buckets を超えないこと."""
        from scripts.v460.lib.macro_regime import (
            MacroRegimeConfig,
            MacroRegimeDetector,
        )

        cfg = MacroRegimeConfig(bucket_sec=1.0, max_buckets=10, slope_window_5m=3)
        det = MacroRegimeDetector(cfg)
        t0 = time.time()
        for i in range(50):
            det.update(t0 + i * 2.0, 10_000_000 + i)
        assert det.buckets_available <= 10

    def test_invalid_price_ignored(self) -> None:
        """NaN / 0 / 負の価格は INSUFFICIENT."""
        import math
        from scripts.v460.lib.macro_regime import MacroRegimeDetector, MacroTrend

        det = MacroRegimeDetector()
        r = det.update(time.time(), float("nan"))
        assert r.trend == MacroTrend.INSUFFICIENT
        r = det.update(time.time(), 0.0)
        assert r.trend == MacroTrend.INSUFFICIENT
        r = det.update(time.time(), -100.0)
        assert r.trend == MacroTrend.INSUFFICIENT


# ======================================================================
# 4. hot-reload 設定テスト
# ======================================================================


class TestHotReloadConfig:
    """188# ev_weighted 関連の hot-reload 可能設定テスト."""

    def test_ev_weighted_in_hot_reload_keys(self) -> None:
        """ev_weighted 関連キーが _HOT_RELOADABLE_FIELDS に含まれること."""
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS

        assert "skip_gate_ev_weighted_enabled" in _HOT_RELOADABLE_FIELDS
        assert "skip_gate_ev_w30" in _HOT_RELOADABLE_FIELDS
        assert "skip_gate_ev_w120" in _HOT_RELOADABLE_FIELDS
