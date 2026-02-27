"""Tests for 182# EV_weighted YAML外部化 + Trend Mode 厳格化 + 在庫偏り regime 別緩和."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.regime_policy import DefaultCycleStrategy, RegimePolicyConfig


# ======================================================================
# EV_weighted weights YAML 外部化
# ======================================================================


class TestEvWeightedYamlExternalization:
    """182# EV_weighted w30/w120 が RegimePolicyConfig に外部化."""

    def test_default_weights(self):
        rp = RegimePolicyConfig()
        assert rp.ev_weighted_w30 == 0.4
        assert rp.ev_weighted_w120 == 0.6

    def test_from_yaml_custom_weights(self):
        yaml_cfg = {
            "regime_policy": {
                "ev_weighted_w30": 0.3,
                "ev_weighted_w120": 0.7,
            },
        }
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.ev_weighted_w30 == pytest.approx(0.3)
        assert rp.ev_weighted_w120 == pytest.approx(0.7)

    def test_from_yaml_invalid_weight_fallback(self):
        yaml_cfg = {
            "regime_policy": {
                "ev_weighted_w30": "bad",
                "ev_weighted_w120": 0.8,
            },
        }
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.ev_weighted_w30 == 0.4  # default (parse failure)
        assert rp.ev_weighted_w120 == pytest.approx(0.8)


# ======================================================================
# Trend Mode 厳格化: confidence gating
# ======================================================================


class TestTrendMinConfidence:
    """182# trend_min_confidence で C/D/Chase を ranging 降格."""

    def test_default_trend_min_confidence(self):
        rp = RegimePolicyConfig()
        assert rp.trend_min_confidence == 0.55

    def test_from_yaml_custom(self):
        yaml_cfg = {"regime_policy": {"trend_min_confidence": 0.7}}
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.trend_min_confidence == pytest.approx(0.7)

    def test_from_yaml_invalid(self):
        yaml_cfg = {"regime_policy": {"trend_min_confidence": "invalid"}}
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.trend_min_confidence == 0.55  # default


class TestGatedRegime:
    """182# DefaultCycleStrategy.gated_regime() テスト."""

    @pytest.fixture
    def strategy(self) -> DefaultCycleStrategy:
        return DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(
                dynamic_cycle_enabled=True,
                dynamic_wait_enabled=True,
                chase_enabled=True,
                trend_min_confidence=0.55,
            ),
        )

    def test_high_confidence_keeps_trending(self, strategy: DefaultCycleStrategy):
        """confidence >= threshold → trending 維持."""
        strategy.update_confidence(0.7)
        assert strategy.gated_regime("trending_up") == "trending_up"
        assert strategy.gated_regime("trending_down") == "trending_down"
        assert strategy.gated_regime("trending") == "trending"

    def test_low_confidence_downgrades_to_ranging(self, strategy: DefaultCycleStrategy):
        """confidence < threshold → ranging に降格."""
        strategy.update_confidence(0.4)
        assert strategy.gated_regime("trending_up") == "ranging"
        assert strategy.gated_regime("trending_down") == "ranging"
        assert strategy.gated_regime("trending") == "ranging"

    def test_non_trending_unaffected(self, strategy: DefaultCycleStrategy):
        """non-trending regimes は confidence に関係なくそのまま."""
        strategy.update_confidence(0.1)  # very low
        assert strategy.gated_regime("ranging") == "ranging"
        assert strategy.gated_regime("high_vol") == "high_vol"

    def test_none_regime(self, strategy: DefaultCycleStrategy):
        assert strategy.gated_regime(None) is None

    def test_explicit_confidence_overrides_cached(self, strategy: DefaultCycleStrategy):
        """明示的 confidence が cached を上書き."""
        strategy.update_confidence(0.1)  # cached: low
        assert strategy.gated_regime("trending_up", confidence=0.8) == "trending_up"
        assert strategy.gated_regime("trending_up", confidence=0.3) == "ranging"

    def test_boundary_confidence(self, strategy: DefaultCycleStrategy):
        """ちょうど 0.55 → 降格されない (< strict)."""
        strategy.update_confidence(0.55)
        assert strategy.gated_regime("trending_up") == "trending_up"


class TestConfidenceGatingIntegration:
    """182# effective_interval / effective_post_fill_wait / is_chase_enabled が gating を内包."""

    @pytest.fixture
    def strategy(self) -> DefaultCycleStrategy:
        return DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(
                dynamic_cycle_enabled=True,
                dynamic_wait_enabled=True,
                chase_enabled=True,
                trend_min_confidence=0.55,
                cycle_intervals={"trending_up": 60.0, "ranging": 120.0},
                post_fill_wait={
                    "trending_up": {"buy": 15.0, "sell": 45.0},
                    "ranging": {"buy": 30.0, "sell": 90.0},
                },
                chase_regimes=["trending_up", "trending_down"],
            ),
        )

    def test_effective_interval_high_confidence(self, strategy: DefaultCycleStrategy):
        strategy.update_confidence(0.7)
        assert strategy.effective_interval("trending_up") == 60.0

    def test_effective_interval_low_confidence_degrades(self, strategy: DefaultCycleStrategy):
        strategy.update_confidence(0.4)
        assert strategy.effective_interval("trending_up") == 120.0  # → ranging

    def test_post_fill_wait_high_confidence(self, strategy: DefaultCycleStrategy):
        strategy.update_confidence(0.7)
        assert strategy.effective_post_fill_wait("buy", "trending_up") == 15.0

    def test_post_fill_wait_low_confidence_degrades(self, strategy: DefaultCycleStrategy):
        strategy.update_confidence(0.4)
        assert strategy.effective_post_fill_wait("buy", "trending_up") == 30.0  # → ranging

    def test_chase_high_confidence(self, strategy: DefaultCycleStrategy):
        strategy.update_confidence(0.7)
        assert strategy.is_chase_enabled("trending_up") is True

    def test_chase_low_confidence_disabled(self, strategy: DefaultCycleStrategy):
        strategy.update_confidence(0.4)
        assert strategy.is_chase_enabled("trending_up") is False  # → ranging


# ======================================================================
# 在庫偏り regime 別緩和
# ======================================================================


class TestDeadlockLimitTrending:
    """182# deadlock_limit_trending が YAML パース可能."""

    def test_default_value(self):
        rp = RegimePolicyConfig()
        assert rp.deadlock_limit_trending == 5

    def test_from_yaml_custom(self):
        yaml_cfg = {"regime_policy": {"deadlock_limit_trending": 8}}
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.deadlock_limit_trending == 8

    def test_from_yaml_invalid(self):
        yaml_cfg = {"regime_policy": {"deadlock_limit_trending": "bad"}}
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.deadlock_limit_trending == 5  # default


# ======================================================================
# update_confidence
# ======================================================================


class TestUpdateConfidence:
    """update_confidence メソッド."""

    def test_initial_confidence_zero(self):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(),
        )
        assert strategy._current_confidence == 0.0

    def test_update_stores_value(self):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(),
        )
        strategy.update_confidence(0.85)
        assert strategy._current_confidence == 0.85


# ======================================================================
# RegimeDetector.current_confidence property
# ======================================================================


class TestRegimeDetectorCurrentConfidence:
    """182# current_confidence プロパティ."""

    def test_no_result_returns_zero(self):
        from scripts.v460.lib.regime_detector import FillTestRegimeDetector
        rd = FillTestRegimeDetector()
        assert rd.current_confidence == 0.0

    def test_after_update_returns_confidence(self):
        """十分なデータを投入して confidence > 0 を確認."""
        from scripts.v460.lib.regime_detector import FillTestRegimeDetector
        rd = FillTestRegimeDetector()
        # window (default 20) 分のデータを投入
        base_price = 15000000.0
        for i in range(25):
            rd.update(1700000000.0 + i * 60.0, base_price + i * 100)
        assert rd.current_confidence > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
