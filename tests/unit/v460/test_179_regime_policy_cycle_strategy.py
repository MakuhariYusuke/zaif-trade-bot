"""Tests for 179# RegimePolicyConfig + CycleStrategy + Chase.

178# 設計:
- RegimePolicyConfig: regime 別制御量の統合設定
- CycleStrategy Protocol: 制御量分岐の外部化
- DefaultCycleStrategy: base 値 + regime オーバーライド
- _effective_sleep: 14 箇所の sleep 一元化
- Chase: stale reprice の aggressive 拡張
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.v460.lib.regime_policy import (
    CycleStrategy,
    DefaultCycleStrategy,
    RegimePolicyConfig,
)


# ======================================================================
# RegimePolicyConfig
# ======================================================================


class TestRegimePolicyConfig:
    """RegimePolicyConfig dataclass + YAML parser."""

    def test_defaults(self):
        rp = RegimePolicyConfig()
        assert rp.dynamic_cycle_enabled is False
        assert rp.dynamic_wait_enabled is False
        assert rp.chase_enabled is False
        assert rp.cycle_intervals["ranging"] == 120.0
        assert rp.cycle_intervals["trending"] == 60.0
        assert rp.post_fill_wait["ranging"]["buy"] == 30.0
        assert rp.post_fill_wait["trending_down"]["sell"] == 15.0
        assert rp.api_error_rate_threshold == 0.03
        assert rp.fill_rate_floor == 0.35

    def test_from_yaml_empty(self):
        rp = RegimePolicyConfig.from_yaml({})
        assert rp.dynamic_cycle_enabled is False

    def test_from_yaml_dynamic_cycle(self):
        yaml_cfg = {
            "regime_policy": {
                "dynamic_cycle": {
                    "enabled": True,
                    "intervals": {
                        "ranging": 150.0,
                        "trending": 45.0,
                    },
                },
            },
        }
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.dynamic_cycle_enabled is True
        assert rp.cycle_intervals["ranging"] == 150.0
        assert rp.cycle_intervals["trending"] == 45.0

    def test_from_yaml_dynamic_wait(self):
        yaml_cfg = {
            "regime_policy": {
                "dynamic_wait": {
                    "enabled": True,
                    "waits": {
                        "trending_up": {"buy": 10.0, "sell": 50.0},
                    },
                },
            },
        }
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.dynamic_wait_enabled is True
        assert rp.post_fill_wait["trending_up"]["buy"] == 10.0

    def test_from_yaml_chase(self):
        yaml_cfg = {
            "regime_policy": {
                "chase": {
                    "enabled": True,
                    "drift_bps": 2.5,
                    "max_reprice": 8,
                    "regimes": ["trending_up"],
                },
            },
        }
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.chase_enabled is True
        assert rp.chase_drift_bps == 2.5
        assert rp.chase_max_reprice == 8
        assert rp.chase_regimes == ["trending_up"]

    def test_from_yaml_stop_conditions(self):
        yaml_cfg = {
            "regime_policy": {
                "stop_conditions": {
                    "api_error_rate_threshold": 0.05,
                    "fill_rate_floor": 0.25,
                    "pnl_floor_bps": -1.5,
                },
            },
        }
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.api_error_rate_threshold == 0.05
        assert rp.fill_rate_floor == 0.25
        assert rp.pnl_floor_bps == -1.5

    def test_from_yaml_full(self):
        """全セクション同時指定."""
        yaml_cfg = {
            "regime_policy": {
                "dynamic_cycle": {"enabled": True, "intervals": {"trending": 30.0}},
                "dynamic_wait": {"enabled": True, "waits": {"trending": {"buy": 5.0, "sell": 10.0}}},
                "chase": {"enabled": True, "drift_bps": 1.0},
                "stop_conditions": {"api_error_rate_threshold": 0.01},
            },
        }
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.dynamic_cycle_enabled is True
        assert rp.dynamic_wait_enabled is True
        assert rp.chase_enabled is True
        assert rp.api_error_rate_threshold == 0.01


# ======================================================================
# CycleStrategy Protocol conformance
# ======================================================================


class TestCycleStrategyProtocol:
    """DefaultCycleStrategy が CycleStrategy Protocol を満たすことを検証."""

    def test_isinstance_check(self):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(),
        )
        assert isinstance(strategy, CycleStrategy)

    def test_protocol_methods_exist(self):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(),
        )
        assert hasattr(strategy, "effective_interval")
        assert hasattr(strategy, "effective_post_fill_wait")
        assert hasattr(strategy, "is_chase_enabled")
        assert hasattr(strategy, "chase_drift_bps")
        assert hasattr(strategy, "chase_max_reprice")


# ======================================================================
# DefaultCycleStrategy — effective_interval (C)
# ======================================================================


class TestEffectiveInterval:
    """C: Dynamic Cycle Interval テスト."""

    @pytest.fixture
    def base_strategy(self) -> DefaultCycleStrategy:
        return DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(),
        )

    def test_disabled_returns_base(self, base_strategy: DefaultCycleStrategy):
        """dynamic_cycle_enabled=False → 常に base."""
        assert base_strategy.effective_interval(None) == 120.0
        assert base_strategy.effective_interval("trending") == 120.0
        assert base_strategy.effective_interval("ranging") == 120.0

    def test_enabled_regime_lookup(self):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(
                dynamic_cycle_enabled=True,
                cycle_intervals={"trending": 60.0, "ranging": 120.0},
            ),
        )
        assert strategy.effective_interval("trending") == 60.0
        assert strategy.effective_interval("ranging") == 120.0
        assert strategy.effective_interval("unknown") == 120.0  # fallback to base
        assert strategy.effective_interval(None) == 120.0

    @pytest.mark.parametrize(
        "regime,expected",
        [
            ("trending_up", 60.0),
            ("trending_down", 60.0),
            ("trending", 60.0),
            ("ranging", 120.0),
            ("high_vol", 120.0),
        ],
    )
    def test_default_intervals(self, regime: str, expected: float):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(dynamic_cycle_enabled=True),
        )
        assert strategy.effective_interval(regime) == expected


# ======================================================================
# DefaultCycleStrategy — effective_post_fill_wait (D)
# ======================================================================


class TestEffectivePostFillWait:
    """D: Regime-linked Post-Fill Wait テスト."""

    def test_disabled_returns_base(self):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(),
        )
        assert strategy.effective_post_fill_wait("buy", "trending") == 30.0
        assert strategy.effective_post_fill_wait("sell", "trending") == 90.0

    def test_enabled_regime_side_lookup(self):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(
                dynamic_wait_enabled=True,
                post_fill_wait={
                    "trending_up": {"buy": 15.0, "sell": 45.0},
                    "ranging": {"buy": 30.0, "sell": 90.0},
                },
            ),
        )
        assert strategy.effective_post_fill_wait("buy", "trending_up") == 15.0
        assert strategy.effective_post_fill_wait("sell", "trending_up") == 45.0
        assert strategy.effective_post_fill_wait("buy", "ranging") == 30.0
        assert strategy.effective_post_fill_wait("sell", "ranging") == 90.0
        # Unknown regime → base
        assert strategy.effective_post_fill_wait("buy", "unknown") == 30.0
        assert strategy.effective_post_fill_wait("sell", "unknown") == 90.0

    @pytest.mark.parametrize(
        "side,regime,expected",
        [
            ("buy", "trending_up", 15.0),
            ("sell", "trending_up", 45.0),
            ("buy", "trending_down", 45.0),
            ("sell", "trending_down", 15.0),
            ("buy", "ranging", 30.0),
            ("sell", "ranging", 90.0),
        ],
    )
    def test_default_waits(self, side: str, regime: str, expected: float):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(dynamic_wait_enabled=True),
        )
        assert strategy.effective_post_fill_wait(side, regime) == expected


# ======================================================================
# DefaultCycleStrategy — Chase
# ======================================================================


class TestChase:
    """Chase (stale reprice 拡張) テスト."""

    def test_disabled_by_default(self):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(),
        )
        assert strategy.is_chase_enabled("trending") is False
        assert strategy.is_chase_enabled("trending_up") is False

    def test_enabled_in_trending_regimes(self):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(chase_enabled=True),
        )
        assert strategy.is_chase_enabled("trending_up") is True
        assert strategy.is_chase_enabled("trending_down") is True
        assert strategy.is_chase_enabled("trending") is True
        assert strategy.is_chase_enabled("ranging") is False
        assert strategy.is_chase_enabled("high_vol") is False
        assert strategy.is_chase_enabled(None) is False

    def test_custom_chase_regimes(self):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(
                chase_enabled=True,
                chase_regimes=["high_vol"],
            ),
        )
        assert strategy.is_chase_enabled("high_vol") is True
        assert strategy.is_chase_enabled("trending_up") is False

    def test_chase_params(self):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(
                chase_enabled=True,
                chase_drift_bps=2.0,
                chase_max_reprice=10,
            ),
        )
        assert strategy.chase_drift_bps() == 2.0
        assert strategy.chase_max_reprice() == 10


# ======================================================================
# Fallback mechanism
# ======================================================================


class TestFallback:
    """停止条件トリガー: フォールバックモード."""

    def test_fallback_reverts_to_base(self):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(
                dynamic_cycle_enabled=True,
                chase_enabled=True,
            ),
        )
        # Before fallback
        assert strategy.effective_interval("trending") == 60.0
        assert strategy.is_chase_enabled("trending") is True

        # Activate fallback
        strategy.activate_fallback(duration_sec=10.0)

        # During fallback → base values
        assert strategy.effective_interval("trending") == 120.0
        assert strategy.is_chase_enabled("trending") is False

    def test_fallback_expires(self):
        strategy = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=RegimePolicyConfig(dynamic_cycle_enabled=True),
        )
        # Activate with 0 duration → expires immediately
        strategy._fallback_active = True
        strategy._fallback_until = time.time() - 1.0  # already expired

        # Should resume dynamic mode
        assert strategy.effective_interval("trending") == 60.0


# ======================================================================
# OrderMonitor Chase integration
# ======================================================================


class TestOrderMonitorChaseIntegration:
    """Chase パラメータが OrderMonitor に正しく渡されることを検証."""

    def test_chase_params_override_stale_default(self):
        """chase_drift_bps_override / chase_max_reprice_override が
        stale_drift / stale_max_reprice を上書きすることを確認."""
        from scripts.v460.lib.fill_config import FillTestConfig

        config = FillTestConfig(
            stale_order_enabled=True,
            stale_drift_bps=5.0,
            stale_max_reprice=3,
        )

        # Chase override values
        chase_drift = 2.0
        chase_max = 8

        # Verify that default config has higher thresholds
        assert config.stale_drift_bps > chase_drift
        assert config.stale_max_reprice < chase_max

        # The actual integration is tested via the stale detection block
        # in OrderMonitor.monitor() — we verify the contract here
        from scripts.v460.lib.order_monitor import OrderMonitor
        monitor = OrderMonitor(config)

        # Verify monitor() accepts chase kwargs
        import inspect
        sig = inspect.signature(monitor.monitor)
        assert "chase_drift_bps_override" in sig.parameters
        assert "chase_max_reprice_override" in sig.parameters


# ======================================================================
# PnlMeasurer wait_sec_override integration
# ======================================================================


class TestPnlMeasurerOverride:
    """179# D: PnlMeasurer に wait_sec_override が渡せることを検証."""

    def test_measure_accepts_wait_sec_override(self):
        import inspect
        from scripts.v460.lib.pnl_measurer import PnlMeasurer
        sig = inspect.signature(PnlMeasurer.measure)
        assert "wait_sec_override" in sig.parameters


# ======================================================================
# _effective_sleep method exists
# ======================================================================


class TestEffectiveSleep:
    """179# S1: _effective_sleep がオーケストレーターに存在することを検証."""

    def test_method_defined(self):
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        assert hasattr(FillLoopOrchestratorMixin, "_effective_sleep")

    def test_is_async(self):
        import asyncio
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        method = getattr(FillLoopOrchestratorMixin, "_effective_sleep")
        assert asyncio.iscoroutinefunction(method)


# ======================================================================
# Hot-reload Protocol conformance
# ======================================================================


class TestHotReloadProtocol:
    """179# _rebuild_cycle_strategy が Protocol に含まれることを検証."""

    def test_protocol_has_rebuild(self):
        from scripts.v460.lib.config_hot_reload import _HotReloadableRunner
        import inspect
        members = inspect.getmembers(_HotReloadableRunner)
        member_names = [name for name, _ in members]
        assert "_rebuild_cycle_strategy" in member_names


# ======================================================================
# Cross-cutting: regime × side × control_quantity matrix
# ======================================================================


class TestRegimeSideMatrix:
    """regime × side の全組み合わせで CycleStrategy が正しく動作するか検証."""

    REGIMES = [None, "ranging", "trending", "trending_up", "trending_down", "high_vol", "unknown"]
    SIDES = ["buy", "sell"]

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
            ),
        )

    @pytest.mark.parametrize("regime", REGIMES)
    def test_effective_interval_no_crash(self, strategy: DefaultCycleStrategy, regime):
        """全 regime で例外なく float を返す."""
        result = strategy.effective_interval(regime)
        assert isinstance(result, float)
        assert result > 0

    @pytest.mark.parametrize("regime", REGIMES)
    @pytest.mark.parametrize("side", SIDES)
    def test_effective_post_fill_wait_no_crash(self, strategy: DefaultCycleStrategy, regime, side):
        """全 regime × side で例外なく float を返す."""
        result = strategy.effective_post_fill_wait(side, regime)
        assert isinstance(result, float)
        assert result > 0

    @pytest.mark.parametrize("regime", REGIMES)
    def test_is_chase_enabled_no_crash(self, strategy: DefaultCycleStrategy, regime):
        """全 regime で例外なく bool を返す."""
        result = strategy.is_chase_enabled(regime)
        assert isinstance(result, bool)

    def test_trending_up_buy_faster_than_sell(self, strategy: DefaultCycleStrategy):
        """trending_up: buy wait < sell wait (順方向 buy は早く、sell は慎重)."""
        buy_wait = strategy.effective_post_fill_wait("buy", "trending_up")
        sell_wait = strategy.effective_post_fill_wait("sell", "trending_up")
        assert buy_wait < sell_wait

    def test_trending_down_sell_faster_than_buy(self, strategy: DefaultCycleStrategy):
        """trending_down: sell wait < buy wait (順方向 sell は早く、buy は慎重)."""
        buy_wait = strategy.effective_post_fill_wait("buy", "trending_down")
        sell_wait = strategy.effective_post_fill_wait("sell", "trending_down")
        assert sell_wait < buy_wait


# ======================================================================
# 180# from_yaml 不正入力耐性テスト
# ======================================================================


class TestFromYamlMalformedInput:
    """180# self-review: from_yaml が不正 YAML 入力で例外を出さないことを検証."""

    def test_regime_policy_not_dict(self):
        """regime_policy が文字列でもクラッシュしない."""
        rp = RegimePolicyConfig.from_yaml({"regime_policy": "invalid"})
        assert rp.dynamic_cycle_enabled is False

    def test_intervals_with_non_numeric(self):
        """intervals に変換不能な値があってもデフォルトにフォールバック."""
        yaml_cfg = {
            "regime_policy": {
                "dynamic_cycle": {
                    "enabled": True,
                    "intervals": {"ranging": "not_a_number"},
                },
            },
        }
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.dynamic_cycle_enabled is True
        # intervals パースが失敗 → デフォルト値が使われる
        assert rp.cycle_intervals["ranging"] == 120.0

    def test_waits_with_malformed_sides(self):
        """waits の side 値が数値でなくてもクラッシュしない."""
        yaml_cfg = {
            "regime_policy": {
                "dynamic_wait": {
                    "enabled": True,
                    "waits": {
                        "trending_up": {"buy": "bad", "sell": 50.0},
                    },
                },
            },
        }
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.dynamic_wait_enabled is True
        # trending_up パース失敗 → デフォルト値
        assert rp.post_fill_wait["ranging"]["buy"] == 30.0

    def test_chase_drift_bps_non_numeric(self):
        """chase.drift_bps が文字列でもクラッシュしない."""
        yaml_cfg = {
            "regime_policy": {
                "chase": {
                    "enabled": True,
                    "drift_bps": "invalid",
                },
            },
        }
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.chase_enabled is True
        assert rp.chase_drift_bps == 3.0  # default

    def test_stop_conditions_non_numeric(self):
        """stop_conditions の値が文字列でもクラッシュしない."""
        yaml_cfg = {
            "regime_policy": {
                "stop_conditions": {
                    "api_error_rate_threshold": "not_valid",
                    "fill_rate_floor": 0.5,
                },
            },
        }
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.api_error_rate_threshold == 0.03  # default (failed parse)
        assert rp.fill_rate_floor == 0.5  # this one parsed OK

    def test_none_regime_policy(self):
        """regime_policy が None でもクラッシュしない."""
        rp = RegimePolicyConfig.from_yaml({"regime_policy": None})
        assert rp.dynamic_cycle_enabled is False

    def test_empty_dynamic_cycle(self):
        """dynamic_cycle が空 dict でもクラッシュしない."""
        yaml_cfg = {"regime_policy": {"dynamic_cycle": {}}}
        rp = RegimePolicyConfig.from_yaml(yaml_cfg)
        assert rp.dynamic_cycle_enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
