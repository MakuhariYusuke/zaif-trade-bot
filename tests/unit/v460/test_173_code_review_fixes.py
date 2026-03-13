"""173# コードレビュー修正テスト.

CRITICAL/HIGH/MED の修正を検証するテストスイート。
"""

from __future__ import annotations

import inspect
import typing
from unittest.mock import MagicMock

import pytest

from scripts.v460.analysis.hindsight_filter import (
    _DIRECT_CATEGORY_BY_REASON,
    _REGIME_GUARD_REASONS,
)
from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.cancel_reasons import CancelReason
from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator
from ztb.risk.sell_dynamic_kill import DynamicKillConfig


# ======================================================================
# 1. DynamicKillConfig validation (CRITICAL)
# ======================================================================

class TestDynamicKillConfigValidation:
    """DynamicKillConfig.__post_init__ のバリデーション."""

    def test_window_zero_raises(self) -> None:
        """window=0 で ValueError."""
        with pytest.raises(ValueError, match="window must be >= 1"):
            DynamicKillConfig(window=0)

    def test_window_negative_raises(self) -> None:
        """window=-1 で ValueError."""
        with pytest.raises(ValueError, match="window must be >= 1"):
            DynamicKillConfig(window=-1)

    def test_resume_window_negative_raises(self) -> None:
        """resume_window=-1 で ValueError."""
        with pytest.raises(ValueError, match="resume_window must be >= 0"):
            DynamicKillConfig(resume_window=-1)

    def test_valid_config(self) -> None:
        """正常値で例外なし."""
        cfg = DynamicKillConfig(window=10, resume_window=0)
        assert cfg.window == 10
        assert cfg.resume_window == 0


# ======================================================================
# 2. sell_guard_inv_bypass_threshold validation (MED)
# ======================================================================

class TestSellGuardInvBypassValidation:
    """fill_config の sell_guard_inv_bypass_threshold バリデーション."""

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="sell_guard_inv_bypass_threshold"):
            FillTestConfig(sell_guard_inv_bypass_threshold=-0.1)

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sell_guard_inv_bypass_threshold"):
            FillTestConfig(sell_guard_inv_bypass_threshold=1.1)

    def test_valid_boundary(self) -> None:
        cfg = FillTestConfig(sell_guard_inv_bypass_threshold=0.0)
        assert cfg.sell_guard_inv_bypass_threshold == 0.0
        cfg2 = FillTestConfig(sell_guard_inv_bypass_threshold=1.0)
        assert cfg2.sell_guard_inv_bypass_threshold == 1.0


# ======================================================================
# 3. CancelReason Literal type exists (MED)
# ======================================================================

class TestCancelReasonLiteralType:
    """cancel_reasons.CancelReason 型が存在し有効."""

    def test_cancel_reason_type_exists(self) -> None:
        # CancelReason は Literal 型
        assert CancelReason is not None

    def test_all_constants_in_literal(self) -> None:
        """全定数が CancelReason Literal に含まれる."""
        # Literal のメンバーを取得
        args = typing.get_args(CR.CancelReason)
        # 定数がすべて含まれていることを確認
        for name in dir(CR):
            val = getattr(CR, name)
            if isinstance(val, str) and name.isupper() and not name.startswith("_"):
                if name in ("CancelReason",):
                    continue
                assert val in args, f"Constant {name}='{val}' not in CancelReason Literal"


# ======================================================================
# 4. DailyDrawdownGuard 機会損失カウント (MED)
# ======================================================================

class TestDailyDrawdownHaltBlockedCycles:
    """halt_blocked_cycles カウンタのテスト."""

    def test_halt_blocked_cycles_increments(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-10.0)
        guard.update_pnl(-15.0)  # halt trigger
        assert guard.is_halted()
        assert guard.state.halt_blocked_cycles == 1
        assert guard.is_halted()
        assert guard.state.halt_blocked_cycles == 2

    def test_halt_blocked_in_metrics(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-10.0)
        guard.update_pnl(-15.0)
        guard.is_halted()
        m = guard.get_metrics()
        assert "halt_blocked_cycles" in m
        assert m["halt_blocked_cycles"] == 1

    def test_halt_blocked_in_export(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-10.0)
        guard.update_pnl(-15.0)
        guard.is_halted()
        exported = guard.export_state()
        assert "halt_blocked_cycles" in exported
        assert exported["halt_blocked_cycles"] == 1


# ======================================================================
# 5. DrawdownAction TypedDict (MED)
# ======================================================================

class TestDrawdownActionTypedDict:
    """update_pnl の戻り値が DrawdownAction TypedDict."""

    def test_update_pnl_returns_typed_dict(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0)
        result = guard.update_pnl(-10.0)
        assert isinstance(result, dict)
        assert "halted" in result
        assert "soft_triggered" in result
        assert "daily_pnl_bps" in result
        assert isinstance(result["halted"], bool)
        assert isinstance(result["daily_pnl_bps"], float)


# ======================================================================
# 6. OrderbookProvider 型適用 (HIGH)
# ======================================================================

class TestOrderbookProviderType:
    """maker_price.py の adapter パラメータ型が OrderbookProvider."""

    def test_compute_imbalance_type_annotation(self) -> None:
        sig = inspect.signature(MakerPriceCalculator.compute_imbalance)
        assert "OrderbookProvider" in str(sig.parameters["adapter"].annotation)

    def test_get_mid_price_type_annotation(self) -> None:
        sig = inspect.signature(MakerPriceCalculator.get_mid_price)
        assert "OrderbookProvider" in str(sig.parameters["adapter"].annotation)

    def test_compute_type_annotation(self) -> None:
        sig = inspect.signature(MakerPriceCalculator.compute)
        assert "OrderbookProvider" in str(sig.parameters["adapter"].annotation)


# ======================================================================
# 7. hot-reload fields 追加 (HIGH)
# ======================================================================

class TestHotReloadFieldsAdded:
    """config_hot_reload の _HOT_RELOADABLE_FIELDS に必要フィールドが存在."""

    def test_sell_offset_floor_reloadable(self) -> None:
        assert "sell_offset_floor" in _HOT_RELOADABLE_FIELDS

    def test_sell_offset_floor_inv_discount_reloadable(self) -> None:
        assert "sell_offset_floor_inv_discount" in _HOT_RELOADABLE_FIELDS

    def test_sell_max_spread_jpy_reloadable(self) -> None:
        assert "sell_max_spread_jpy" in _HOT_RELOADABLE_FIELDS

    def test_unknown_buy_offset_boost_reloadable(self) -> None:
        assert "unknown_buy_offset_boost" in _HOT_RELOADABLE_FIELDS

    def test_fallback_stale_sec_reloadable(self) -> None:
        assert "fallback_stale_sec" in _HOT_RELOADABLE_FIELDS

    def test_post_fill_wait_sec_sell_reloadable(self) -> None:
        assert "post_fill_wait_sec_sell" in _HOT_RELOADABLE_FIELDS


# ======================================================================
# 8. hindsight_filter CR 定数参照 + 欠落 reason 追加 (HIGH)
# ======================================================================

class TestHindsightFilterReasons:
    """hindsight_filter の cancel_reason 分類が CR 定数を使用."""

    def test_regime_guard_has_ranging_low_vol(self) -> None:
        assert "ranging_low_vol_skip" in _REGIME_GUARD_REASONS

    def test_regime_guard_has_velocity_skip(self) -> None:
        assert "skip_gate_rule_velocity_sell" in _REGIME_GUARD_REASONS
        assert "skip_gate_rule_velocity_buy" in _REGIME_GUARD_REASONS

    def test_direct_category_has_daily_drawdown(self) -> None:
        assert "daily_drawdown_halt" in _DIRECT_CATEGORY_BY_REASON

    def test_regime_guard_has_unknown_sell_skip(self) -> None:
        assert "unknown_regime_sell_skip" in _REGIME_GUARD_REASONS


# ======================================================================
# 9. sell_offset_floor 動的化 (P1)
# ======================================================================

class TestDynamicSellOffsetFloor:
    """sell_offset_floor の動的フロア — InvSkew 活性時にフロア割引."""

    @staticmethod
    def _make_calc(cfg: object) -> object:
        """MakerPriceCalculator を必須引数付きで生成."""
        ffd = MagicMock()
        return MakerPriceCalculator(
            cfg,  # type: ignore[arg-type]
            fast_fill_defense=ffd,
            regime_detector=None,
            base_offset_ratio=0.001,
        )

    def test_effective_floor_no_invskew(self) -> None:
        """InvSkew 非活性時は通常フロア."""
        cfg = FillTestConfig(
            sell_offset_floor=0.20,
            sell_guard_inv_bypass_threshold=0.3,
            sell_offset_floor_inv_discount=0.5,
        )
        calc = self._make_calc(cfg)
        # imbalance = 0.0 (初期値) → threshold 未達 → 通常フロア
        assert calc._effective_sell_offset_floor() == 0.20

    def test_effective_floor_with_invskew(self) -> None:
        """InvSkew 活性時はフロアが割引."""
        cfg = FillTestConfig(
            sell_offset_floor=0.20,
            sell_guard_inv_bypass_threshold=0.3,
            sell_offset_floor_inv_discount=0.5,
        )
        calc = self._make_calc(cfg)
        # InvSkew のために imbalance を手動設定
        calc._inv_net_imbalance = 0.5  # >= 0.3
        assert calc._effective_sell_offset_floor() == pytest.approx(0.10)

    def test_effective_floor_disabled(self) -> None:
        """sell_offset_floor=0 のとき常に 0."""
        cfg = FillTestConfig(sell_offset_floor=0.0)
        calc = self._make_calc(cfg)
        assert calc._effective_sell_offset_floor() == 0.0

    def test_config_default_inv_discount(self) -> None:
        """sell_offset_floor_inv_discount のデフォルトは 0.5."""
        cfg = FillTestConfig()
        assert cfg.sell_offset_floor_inv_discount == 0.5
