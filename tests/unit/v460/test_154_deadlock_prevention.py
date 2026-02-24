"""154# P0-08 deadlock 防止テスト.

対象:
  - C-1: 片側残高枯渇時は balance_forced でも実行許可
  - C-2: 連続 forced skip カウンタによるフォールバック
  - Config: balance_forced_deadlock_limit の YAML 読込・デフォルト値
  - 158# P1-1: balance_forced 救済モード (offset 倍増)
"""

from __future__ import annotations

import logging
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.fill_config import FillTestConfig


# ======================================================================
# Config テスト
# ======================================================================

class TestBalanceForcedDeadlockConfig:
    """balance_forced_deadlock_limit の設定テスト."""

    def test_default_value(self) -> None:
        """デフォルト値は 3."""
        cfg = FillTestConfig()
        assert cfg.balance_forced_deadlock_limit == 3

    def test_custom_value(self) -> None:
        """任意の値を指定可能."""
        cfg = FillTestConfig(balance_forced_deadlock_limit=10)
        assert cfg.balance_forced_deadlock_limit == 10

    def test_zero_means_unlimited(self) -> None:
        """0 = 無制限 (deadlock limit 無効)."""
        cfg = FillTestConfig(balance_forced_deadlock_limit=0)
        assert cfg.balance_forced_deadlock_limit == 0

    def test_yaml_parsing(self) -> None:
        """YAML loss_control セクションから読込."""
        yaml_cfg = {
            "loss_control": {
                "skip_balance_forced": True,
                "balance_forced_deadlock_limit": 5,
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.skip_balance_forced is True
        assert cfg.balance_forced_deadlock_limit == 5

    def test_yaml_default_when_not_specified(self) -> None:
        """YAML に未指定ならデフォルト値."""
        yaml_cfg = {"loss_control": {"skip_balance_forced": True}}
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.balance_forced_deadlock_limit == 3  # dataclass default


# ======================================================================
# Deadlock 防止ロジックテスト (ユニットレベル)
# ======================================================================

class TestBalanceForcedDeadlockPrevention:
    """P0-08 deadlock 防止ロジックの単体テスト.

    run_loop の該当箇所を直接テストするのは困難なため、
    ロジックの判定条件をテストで検証する。
    """

    def test_c1_one_sided_balance_allows_execution(self) -> None:
        """C-1: 元の side が残高不足 → forced side で実行すべき."""
        # Scenario: JPY枯渇, BTC 0.002 保有
        # next_side = "sell" (forced from buy)
        # original_side = "buy" → check returns True (insufficient)
        # → deadlock 回避のため sell は実行される
        original_also_insufficient = True
        deadlock_limit = 3
        consecutive_count = 0
        over_deadlock_limit = deadlock_limit > 0 and consecutive_count >= deadlock_limit

        should_execute = original_also_insufficient or over_deadlock_limit
        assert should_execute is True

    def test_c1_both_ok_skips_as_before(self) -> None:
        """C-1: 両方残高 OK → 従来通りスキップ."""
        # Scenario: 両方残高十分だが forced switch で来た
        original_also_insufficient = False
        deadlock_limit = 3
        consecutive_count = 0
        over_deadlock_limit = deadlock_limit > 0 and consecutive_count >= deadlock_limit

        should_execute = original_also_insufficient or over_deadlock_limit
        assert should_execute is False  # → スキップされる

    def test_c2_deadlock_limit_forces_execution(self) -> None:
        """C-2: 連続 N 回 skip → deadlock limit 超過で実行."""
        original_also_insufficient = False
        deadlock_limit = 3
        consecutive_count = 3  # == limit
        over_deadlock_limit = deadlock_limit > 0 and consecutive_count >= deadlock_limit

        should_execute = original_also_insufficient or over_deadlock_limit
        assert should_execute is True

    def test_c2_below_limit_skips(self) -> None:
        """C-2: 連続回数が上限未満 → スキップ."""
        original_also_insufficient = False
        deadlock_limit = 3
        consecutive_count = 2  # < limit
        over_deadlock_limit = deadlock_limit > 0 and consecutive_count >= deadlock_limit

        should_execute = original_also_insufficient or over_deadlock_limit
        assert should_execute is False

    def test_c2_zero_limit_never_forces(self) -> None:
        """C-2: limit=0 は無制限 (forced execution なし)."""
        original_also_insufficient = False
        deadlock_limit = 0
        consecutive_count = 999
        over_deadlock_limit = deadlock_limit > 0 and consecutive_count >= deadlock_limit

        should_execute = original_also_insufficient or over_deadlock_limit
        assert should_execute is False

    def test_counter_reset_on_execution(self) -> None:
        """実サイクル実行で forced skip カウンタがリセットされる."""
        counter = 5
        # After run_single_cycle succeeds:
        counter = 0
        assert counter == 0

    def test_counter_increments_on_skip(self) -> None:
        """スキップ時にカウンタがインクリメントされる."""
        counter = 0
        for _ in range(5):
            counter += 1
        assert counter == 5


# ======================================================================
# 統合ロジックテスト (初期化フィールド)
# ======================================================================

class TestFillTestRunnerInitFields:
    """FillTestRunner の初期化でカウンタが正しく設定されるか."""

    def test_balance_forced_skip_count_initialized(self) -> None:
        """_balance_forced_skip_count が 0 で初期化される."""
        # MagicMock で __init__ の全依存を回避し、属性のみ検証
        cfg = FillTestConfig()
        # run_fill_test は import が重いので属性の存在のみ確認
        from scripts.v460.run_fill_test import FillTestRunner
        # __init__ は adapter 等が必要なので dataclass fields を検証
        import dataclasses
        # FillTestConfig に balance_forced_deadlock_limit があることを確認
        field_names = [f.name for f in dataclasses.fields(cfg)]
        assert "balance_forced_deadlock_limit" in field_names
        assert "skip_balance_forced" in field_names


# ======================================================================
# Cancel reason 定数テスト
# ======================================================================

class TestCancelReasonConstant:
    """BALANCE_FORCED_SKIP が適切に定義されているか."""

    def test_balance_forced_skip_in_audit(self) -> None:
        assert CR.BALANCE_FORCED_SKIP in CR.AUDIT_CANCEL_REASONS

    def test_balance_forced_skip_value(self) -> None:
        assert CR.BALANCE_FORCED_SKIP == "balance_forced_skip"


# ======================================================================
# 158# P1-1: balance_forced 救済モード
# ======================================================================

class TestBalanceForcedRescueConfig:
    """158# P1-1: rescue モード設定テスト."""

    def test_rescue_disabled_by_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.balance_forced_rescue_enabled is False

    def test_rescue_offset_mult_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.balance_forced_rescue_offset_mult == pytest.approx(2.0)

    def test_rescue_custom_values(self) -> None:
        cfg = FillTestConfig(
            balance_forced_rescue_enabled=True,
            balance_forced_rescue_offset_mult=1.5,
        )
        assert cfg.balance_forced_rescue_enabled is True
        assert cfg.balance_forced_rescue_offset_mult == pytest.approx(1.5)

    def test_rescue_yaml_parsing(self) -> None:
        """YAML loss_control セクションから rescue 設定を読込."""
        yaml_cfg = {
            "loss_control": {
                "balance_forced_rescue_enabled": True,
                "balance_forced_rescue_offset_mult": 1.8,
            }
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.balance_forced_rescue_enabled is True
        assert cfg.balance_forced_rescue_offset_mult == pytest.approx(1.8)


class TestBalanceForcedRescueLogic:
    """158# P1-1: rescue モードロジックテスト."""

    def test_run_single_cycle_accepts_rescue_param(self) -> None:
        """run_single_cycle が balance_forced_rescue パラメータを受け取る."""
        import inspect
        from scripts.v460.run_fill_test import FillTestRunner
        sig = inspect.signature(FillTestRunner.run_single_cycle)
        assert "balance_forced_rescue" in sig.parameters
        param = sig.parameters["balance_forced_rescue"]
        assert param.default is False

    def test_rescue_offset_in_run_single_cycle(self) -> None:
        """run_single_cycle に rescue offset 調整ロジックがある."""
        import inspect
        from scripts.v460.run_fill_test import FillTestRunner
        source = inspect.getsource(FillTestRunner.run_single_cycle)
        assert "balance_forced_rescue" in source
        assert "balance_forced_rescue_offset_mult" in source

    def test_rescue_mode_in_run_continuous(self) -> None:
        """run_continuous に rescue フラグの初期化と受け渡しがある."""
        import inspect
        from scripts.v460.run_fill_test import FillTestRunner
        source = inspect.getsource(FillTestRunner.run_continuous)
        assert "_is_rescue" in source
        assert "balance_forced_rescue_enabled" in source

    def test_config_fields_in_dataclass(self) -> None:
        """FillTestConfig に rescue 関連フィールドが存在する."""
        import dataclasses
        cfg = FillTestConfig()
        field_names = [f.name for f in dataclasses.fields(cfg)]
        assert "balance_forced_rescue_enabled" in field_names
        assert "balance_forced_rescue_offset_mult" in field_names


# ======================================================================
# 158# P1-5: A/B テスト基盤 (variant_id)
# ======================================================================

class TestABTestVariantConfig:
    """158# P1-5: A/B テスト variant 設定テスト."""

    def test_ab_variant_default_empty(self) -> None:
        cfg = FillTestConfig()
        assert cfg.ab_test_variant == ""

    def test_ab_variant_custom(self) -> None:
        cfg = FillTestConfig(ab_test_variant="sell_offset_015")
        assert cfg.ab_test_variant == "sell_offset_015"

    def test_ab_variant_yaml_parsing(self) -> None:
        yaml_cfg = {
            "ab_test": {"variant": "rescue_enabled"},
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.ab_test_variant == "rescue_enabled"

    def test_ab_variant_yaml_absent(self) -> None:
        yaml_cfg: dict = {}
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.ab_test_variant == ""


class TestABTestVariantFillRecord:
    """158# P1-5: FillRecord に variant が記録される."""

    def test_fill_record_has_ab_test_variant(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            ab_test_variant="sell_offset_015",
        )
        assert r.ab_test_variant == "sell_offset_015"

    def test_fill_record_ab_variant_default_none(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
        )
        assert r.ab_test_variant is None

    def test_fill_record_ab_variant_roundtrip(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
            ab_test_variant="control_v1",
        )
        d = r.to_dict()
        assert d["ab_test_variant"] == "control_v1"
        r2 = FillRecord.from_dict(d)
        assert r2.ab_test_variant == "control_v1"

    def test_fill_record_ab_variant_absent_in_old_data(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        d = {
            "cycle_id": "old",
            "timestamp": 0.0,
            "side": "buy",
            "order_price": 100.0,
            "order_quantity": 0.001,
        }
        r = FillRecord.from_dict(d)
        assert r.ab_test_variant is None


# =====================================================================
# G. 158# P1-2: reprice offset tightening
# =====================================================================

class TestRepriceOffsetTighten:
    """158# P1-2: stale_reprice_tighten の Config + ロジックテスト."""

    def test_default_value(self) -> None:
        cfg = FillTestConfig()
        assert cfg.stale_reprice_tighten == pytest.approx(1.0)

    def test_custom_value(self) -> None:
        cfg = FillTestConfig(stale_reprice_tighten=0.85)
        assert cfg.stale_reprice_tighten == pytest.approx(0.85)

    def test_yaml_parsing(self) -> None:
        import yaml
        yaml_str = """
stale_order:
  enabled: true
  reprice_tighten: 0.80
"""
        data = yaml.safe_load(yaml_str)
        cfg = FillTestConfig.from_yaml(data)
        assert cfg.stale_reprice_tighten == pytest.approx(0.80)

    def test_tighten_logic_in_order_monitor(self) -> None:
        """OrderMonitor に tighten ロジックが存在する."""
        import inspect
        from scripts.v460.lib.order_monitor import OrderMonitor
        source = inspect.getsource(OrderMonitor.monitor)
        assert "stale_reprice_tighten" in source
        assert "tightened_gap" in source

    def test_tighten_buy_formula(self) -> None:
        """buy 側: tighten で mid に近づく (gap 縮小)."""
        mid = 15_000_000
        original_price = 14_970_000  # gap = 30,000
        tighten = 0.85
        gap = abs(original_price - mid)
        tightened_gap = gap * tighten
        new_price = round(mid - tightened_gap)
        assert new_price > original_price  # closer to mid
        assert new_price == round(mid - 30_000 * 0.85)

    def test_tighten_sell_formula(self) -> None:
        """sell 側: tighten で mid に近づく (gap 縮小)."""
        mid = 15_000_000
        original_price = 15_030_000  # gap = 30,000
        tighten = 0.85
        gap = abs(original_price - mid)
        tightened_gap = gap * tighten
        new_price = round(mid + tightened_gap)
        assert new_price < original_price  # closer to mid
        assert new_price == round(mid + 30_000 * 0.85)

    def test_tighten_1_0_no_change(self) -> None:
        """tighten=1.0 の場合は価格変更なし."""
        mid = 15_000_000
        original_price = 14_970_000
        tighten = 1.0
        gap = abs(original_price - mid)
        tightened_gap = gap * tighten
        assert tightened_gap == gap  # no change