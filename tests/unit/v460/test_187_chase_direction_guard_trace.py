"""187# Phase B テスト: Chase 方向制御 + guard_trace 記録 + clamp YAML外部化.

B-1: Chase 方向制限 (trending_up→buy only, trending_down→sell only)
B-2: FillRecord に gated_regime / effective_cycle_interval 追加
追加改善: clamp 値の YAML 外部化 + hot-reload 対応
"""

from __future__ import annotations

import pytest
from dataclasses import fields

from scripts.v460.lib.regime_policy import (
    DefaultCycleStrategy,
    RegimePolicyConfig,
)
from ztb.metrics.fill_quality import FillRecord


# ======================================================================
# B-1: Chase 方向制御
# ======================================================================

class TestChaseDirectionControl:
    """187# B-1: trending_up/down で Chase 方向を制限."""

    @pytest.fixture
    def strategy(self) -> DefaultCycleStrategy:
        policy = RegimePolicyConfig(
            chase_enabled=True,
            chase_drift_bps=3.0,
            chase_max_reprice=5,
            chase_regimes=["trending_up", "trending_down", "trending"],
            dynamic_cycle_enabled=True,
            trend_min_confidence=0.45,
            trend_exit_confidence=0.30,
            trend_min_dwell=1,  # テスト簡略化のため 1
        )
        s = DefaultCycleStrategy(
            base_interval=120.0,
            base_wait_buy=30.0,
            base_wait_sell=90.0,
            policy=policy,
        )
        s.update_confidence(0.8)  # trending 確実発火
        return s

    # --- trending_up: buy OK, sell NG ---
    def test_trending_up_buy_chase_enabled(self, strategy: DefaultCycleStrategy) -> None:
        assert strategy.is_chase_enabled("trending_up", "buy") is True

    def test_trending_up_sell_chase_blocked(self, strategy: DefaultCycleStrategy) -> None:
        assert strategy.is_chase_enabled("trending_up", "sell") is False

    # --- trending_down: sell OK, buy NG ---
    def test_trending_down_sell_chase_enabled(self, strategy: DefaultCycleStrategy) -> None:
        assert strategy.is_chase_enabled("trending_down", "sell") is True

    def test_trending_down_buy_chase_blocked(self, strategy: DefaultCycleStrategy) -> None:
        assert strategy.is_chase_enabled("trending_down", "buy") is False

    # --- trending (方向不明): 両方OK (後方互換) ---
    def test_trending_both_sides_allowed(self, strategy: DefaultCycleStrategy) -> None:
        assert strategy.is_chase_enabled("trending", "buy") is True
        assert strategy.is_chase_enabled("trending", "sell") is True

    # --- ranging: Chase 無効 (gated_regime で降格) ---
    def test_ranging_chase_disabled(self, strategy: DefaultCycleStrategy) -> None:
        assert strategy.is_chase_enabled("ranging", "buy") is False
        assert strategy.is_chase_enabled("ranging", "sell") is False

    # --- side=None (後方互換): 方向フィルタ非適用 ---
    def test_side_none_backward_compat(self, strategy: DefaultCycleStrategy) -> None:
        assert strategy.is_chase_enabled("trending_up", None) is True
        assert strategy.is_chase_enabled("trending_down", None) is True

    # --- chase_enabled=False: 全て無効 ---
    def test_chase_disabled_globally(self) -> None:
        policy = RegimePolicyConfig(chase_enabled=False)
        s = DefaultCycleStrategy(120.0, 30.0, 90.0, policy)
        assert s.is_chase_enabled("trending_up", "buy") is False

    # --- confidence 不足でトレンドに入れない場合 ---
    def test_low_confidence_no_chase(self, strategy: DefaultCycleStrategy) -> None:
        strategy.update_confidence(0.1)
        # ヒステリシス状態リセット
        strategy._in_trend_mode = False
        strategy._trend_dwell = 0
        assert strategy.is_chase_enabled("trending_up", "buy") is False


# ======================================================================
# B-2: guard_trace — FillRecord 新フィールド
# ======================================================================

class TestGuardTraceFillRecord:
    """187# B-2: FillRecord に gated_regime + effective_cycle_interval フィールド確認."""

    def test_fillrecord_has_gated_regime(self) -> None:
        field_names = {f.name for f in fields(FillRecord)}
        assert "gated_regime" in field_names

    def test_fillrecord_has_effective_cycle_interval(self) -> None:
        field_names = {f.name for f in fields(FillRecord)}
        assert "effective_cycle_interval" in field_names

    def test_gated_regime_default_none(self) -> None:
        rec = FillRecord(cycle_id="test", timestamp=0.0, side="buy", order_price=100.0, order_quantity=0.001)
        assert rec.gated_regime is None

    def test_effective_cycle_interval_default_none(self) -> None:
        rec = FillRecord(cycle_id="test", timestamp=0.0, side="buy", order_price=100.0, order_quantity=0.001)
        assert rec.effective_cycle_interval is None

    def test_fillrecord_with_guard_trace(self) -> None:
        rec = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.001,
            gated_regime="trending_up",
            effective_cycle_interval=60.0,
        )
        assert rec.gated_regime == "trending_up"
        assert rec.effective_cycle_interval == 60.0

    def test_fillrecord_to_dict_includes_guard_trace(self) -> None:
        rec = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.001,
            gated_regime="ranging",
            effective_cycle_interval=120.0,
        )
        d = rec.to_dict()
        assert d["gated_regime"] == "ranging"
        assert d["effective_cycle_interval"] == 120.0

    def test_fillrecord_from_dict_with_guard_trace(self) -> None:
        d = {
            "cycle_id": "test",
            "timestamp": 0.0,
            "side": "buy",
            "order_price": 100.0,
            "order_quantity": 0.001,
            "gated_regime": "trending_down",
            "effective_cycle_interval": 60.0,
        }
        rec = FillRecord.from_dict(d)
        assert rec.gated_regime == "trending_down"
        assert rec.effective_cycle_interval == 60.0

    def test_fillrecord_from_dict_backward_compat(self) -> None:
        """既存 JSONL (guard_trace フィールドなし) からの読み込み."""
        d = {
            "cycle_id": "test",
            "timestamp": 0.0,
            "side": "buy",
            "order_price": 100.0,
            "order_quantity": 0.001,
        }
        rec = FillRecord.from_dict(d)
        assert rec.gated_regime is None
        assert rec.effective_cycle_interval is None


# ======================================================================
# 追加改善: clamp YAML 外部化
# ======================================================================

class TestClampYAMLExternalization:
    """187# 追加改善: clamp 値が FillTestConfig に外部化されていることを確認."""

    def test_fill_config_has_offset_floor(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.skip_gate_offset_floor == pytest.approx(-0.3)

    def test_fill_config_has_offset_ceil(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.skip_gate_offset_ceil == pytest.approx(0.5)

    def test_hot_reloadable_fields_include_clamp(self) -> None:
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
        assert "skip_gate_offset_floor" in _HOT_RELOADABLE_FIELDS
        assert "skip_gate_offset_ceil" in _HOT_RELOADABLE_FIELDS

    def test_yaml_parsing_offset_floor_ceil(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        yaml_cfg = {
            "skip_gate": {
                "offset_floor": -0.5,
                "offset_ceil": 0.8,
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.skip_gate_offset_floor == pytest.approx(-0.5)
        assert cfg.skip_gate_offset_ceil == pytest.approx(0.8)


# ======================================================================
# CycleStrategy Protocol 互換性
# ======================================================================

class TestCycleStrategyProtocol:
    """CycleStrategy Protocol と DefaultCycleStrategy の互換性確認."""

    def test_is_chase_enabled_accepts_side(self) -> None:
        """is_chase_enabled が side パラメータを受け付ける."""
        from scripts.v460.lib.regime_policy import CycleStrategy
        policy = RegimePolicyConfig(chase_enabled=True)
        s = DefaultCycleStrategy(120.0, 30.0, 90.0, policy)
        # Protocol 互換: side=None OK, side="buy" OK
        assert isinstance(s.is_chase_enabled("ranging"), bool)
        assert isinstance(s.is_chase_enabled("ranging", "buy"), bool)
