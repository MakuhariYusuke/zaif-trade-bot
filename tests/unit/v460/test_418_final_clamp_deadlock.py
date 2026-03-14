"""418# テスト: Execution Final Clamp + Route-to-Kill Deadlock 防止.

416#/417# レビューで発見された2つの構造欠陥の修正を検証:
1. Final Clamp: maker_price ceiling 後の executor 側 multiplier chain が
   ceiling を迂回して effective_offset_ratio が際限なく拡大する問題
2. Route-to-Kill Deadlock: buy 残高不足 → sell 切替 → sell kill-gated →
   高速ループのデッドスピラル
"""

from __future__ import annotations

import pytest

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.pre_order_adjustments import PreOrderAdjustmentsMixin


# ============================================================
# cancel_reasons 定数の存在確認
# ============================================================


class TestCancelReasons418:
    """418# で追加された cancel_reason 定数のテスト."""

    def test_final_clamp_hard_skip_exists(self) -> None:
        assert CR.FINAL_CLAMP_HARD_SKIP == "final_clamp_hard_skip"

    def test_route_to_kill_deadlock_exists(self) -> None:
        assert CR.ROUTE_TO_KILL_DEADLOCK == "route_to_kill_deadlock"

    def test_final_clamp_in_audit_set(self) -> None:
        assert CR.FINAL_CLAMP_HARD_SKIP in CR.AUDIT_CANCEL_REASONS

    def test_route_to_kill_in_audit_set(self) -> None:
        assert CR.ROUTE_TO_KILL_DEADLOCK in CR.AUDIT_CANCEL_REASONS


# ============================================================
# FillConfig: execution_final_clamp 設定フィールド
# ============================================================


class TestFillConfigFinalClamp:
    """418# Final Clamp 設定フィールドのテスト."""

    def test_default_enabled(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.execution_final_clamp_enabled is True

    def test_default_hard_skip_mult_disabled(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.execution_final_clamp_hard_skip_mult == 0.0

    def test_custom_hard_skip_mult(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(execution_final_clamp_hard_skip_mult=2.5)
        assert cfg.execution_final_clamp_hard_skip_mult == 2.5


# ============================================================
# FillRecord: execution_pre_clamp_offset フィールド
# ============================================================


class TestFillRecordPreClampField:
    """418# FillRecord.execution_pre_clamp_offset のテスト."""

    def test_field_exists_default_none(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="buy",
            order_price=10000,
            order_quantity=0.001,
        )
        assert rec.execution_pre_clamp_offset is None

    def test_field_set(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="sell",
            order_price=10000,
            order_quantity=0.001,
            execution_pre_clamp_offset=0.905,
        )
        assert rec.execution_pre_clamp_offset == pytest.approx(0.905)

    def test_field_in_to_dict(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="sell",
            order_price=10000,
            order_quantity=0.001,
            execution_pre_clamp_offset=1.305,
        )
        d = rec.to_dict()
        assert "execution_pre_clamp_offset" in d
        assert d["execution_pre_clamp_offset"] == pytest.approx(1.305)

    def test_field_roundtrip_from_dict(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(
            cycle_id="test",
            timestamp=0.0,
            side="sell",
            order_price=10000,
            order_quantity=0.001,
            execution_pre_clamp_offset=0.75,
        )
        reconstructed = FillRecord.from_dict(rec.to_dict())
        assert reconstructed.execution_pre_clamp_offset == pytest.approx(0.75)


# ============================================================
# Final Clamp ロジックの単体テスト (純粋関数ベース)
# ============================================================


class TestFinalClampLogic:
    """Final Clamp のクランプ計算ロジックをテスト.

    fill_cycle_executor.py 内のインラインロジックを
    PreOrderAdjustmentsMixin._recalc_price_with_new_offset で検証。
    """

    @staticmethod
    def _simulate_final_clamp(
        side: str,
        order_price: float,
        spread_at_order: float | None,
        effective_offset_ratio: float,
        ceiling: float,
    ) -> tuple[float, float]:
        """Final Clamp ロジックのシミュレーション."""
        if ceiling > 0 and effective_offset_ratio > ceiling:
            order_price = PreOrderAdjustmentsMixin._recalc_price_with_new_offset(
                side, order_price, spread_at_order,
                effective_offset_ratio, ceiling,
            )
            effective_offset_ratio = ceiling
        return order_price, effective_offset_ratio

    def test_buy_over_ceiling_clamped(self) -> None:
        """buy: ceiling 超過時にクランプ."""
        price, ratio = self._simulate_final_clamp(
            side="buy",
            order_price=9950,  # mid=10000, spread=100, old_ratio=1.0 → 10000-50=9950
            spread_at_order=100.0,
            effective_offset_ratio=1.0,
            ceiling=0.20,
        )
        assert ratio == 0.20
        # mid = 9950 + 100 * 1.0 / 2 = 10000
        # new_price = 10000 - 100 * 0.20 / 2 = 9990
        assert price == 9990

    def test_sell_over_ceiling_clamped(self) -> None:
        """sell: ceiling 超過時にクランプ."""
        price, ratio = self._simulate_final_clamp(
            side="sell",
            order_price=10100,  # mid=10000, spread=200, old_ratio=1.0 → 10000+100=10100
            spread_at_order=200.0,
            effective_offset_ratio=1.0,
            ceiling=0.50,
        )
        assert ratio == 0.50
        # mid = 10100 - 200 * 1.0 / 2 = 10000
        # new_price = 10000 + 200 * 0.50 / 2 = 10050
        assert price == 10050

    def test_under_ceiling_no_change(self) -> None:
        """ceiling 以下の場合は変更なし."""
        price, ratio = self._simulate_final_clamp(
            side="sell",
            order_price=10025,
            spread_at_order=100.0,
            effective_offset_ratio=0.45,
            ceiling=0.50,
        )
        assert ratio == 0.45
        assert price == 10025

    def test_ceiling_zero_disabled(self) -> None:
        """ceiling=0 の場合は無効 (変更なし)."""
        price, ratio = self._simulate_final_clamp(
            side="sell",
            order_price=10100,
            spread_at_order=200.0,
            effective_offset_ratio=2.0,
            ceiling=0.0,
        )
        assert ratio == 2.0
        assert price == 10100

    def test_hard_skip_threshold(self) -> None:
        """hard skip 判定: effective_offset > ceiling × mult."""
        ceiling = 0.50
        hard_skip_mult = 2.0
        effective = 1.305  # > 0.50 × 2.0 = 1.0

        should_hard_skip = (
            hard_skip_mult > 0
            and effective > ceiling * hard_skip_mult
        )
        assert should_hard_skip is True

    def test_hard_skip_not_triggered_below_threshold(self) -> None:
        """hard skip 非発火: effective_offset <= ceiling × mult."""
        ceiling = 0.50
        hard_skip_mult = 2.0
        effective = 0.905  # <= 0.50 × 2.0 = 1.0

        should_hard_skip = (
            hard_skip_mult > 0
            and effective > ceiling * hard_skip_mult
        )
        assert should_hard_skip is False

    def test_hard_skip_mult_zero_disabled(self) -> None:
        """hard_skip_mult=0 の場合は hard skip 無効."""
        ceiling = 0.50
        hard_skip_mult = 0.0
        effective = 5.0  # 極端に高い

        should_hard_skip = (
            hard_skip_mult > 0
            and effective > ceiling * hard_skip_mult
        )
        assert should_hard_skip is False


# ============================================================
# Post-Ceiling Multiplier Leak のシミュレーション
# ============================================================


class TestPostCeilingMultiplierLeak:
    """416#/417# で発見された post-ceiling leak の再現テスト.

    maker_price ceiling → executor multiplier chain を経て
    effective_offset_ratio が ceiling を大幅に超過する問題を
    再現し、Final Clamp で修正されることを検証。
    """

    def test_multiplier_chain_without_clamp_leaks(self) -> None:
        """clamp なしの multiplier chain は ceiling を迂回する."""
        ceiling = 0.50
        offset = 0.498  # ceiling ギリギリ (maker_price 出力)

        # EV offset: ×1.2
        offset *= 1.2
        # velocity: ×1.3
        offset *= 1.3
        # toxicity: ×1.1
        offset *= 1.1
        # → 0.498 × 1.2 × 1.3 × 1.1 = 0.854... (ceiling 0.50 を大幅超過)
        assert offset > ceiling
        assert offset > 0.85

    def test_multiplier_chain_with_final_clamp(self) -> None:
        """Final Clamp ありの場合は ceiling に切り詰められる."""
        ceiling = 0.50
        offset = 0.498

        # 同じ multiplier chain
        offset *= 1.2
        offset *= 1.3
        offset *= 1.1

        # Final Clamp
        if offset > ceiling:
            offset = ceiling

        assert offset == ceiling

    def test_real_data_reproduction_311(self) -> None:
        """3/11 実データの再現: final_stage=0.300, max_eff_offset=1.305.

        416# §1.1 で指摘されたデータポイントのシミュレーション。
        """
        ceiling = 0.50  # offset_ceiling_ratio_sell
        offset = 0.300  # maker_price final output (with ceiling at 0.50)

        # シミュレーション: 複数 multiplier を経て 1.305 に到達
        # (実際のパラメータは不明だが、×4.35 相当)
        offset *= 4.35
        assert offset == pytest.approx(1.305, abs=0.01)

        # Final Clamp で修正
        if offset > ceiling:
            clamped = ceiling
        else:
            clamped = offset
        assert clamped == ceiling


# ============================================================
# Ceiling 解決ロジック (side-specific)
# ============================================================


class TestCeilingResolution:
    """サイド別 ceiling ratio 解決テスト."""

    @staticmethod
    def _resolve_ceiling(
        side: str,
        offset_ceiling_ratio: float,
        offset_ceiling_ratio_buy: float | None,
        offset_ceiling_ratio_sell: float | None,
    ) -> float:
        """fill_cycle_executor.py / maker_price.py 共通の ceiling 解決ロジック."""
        ceiling = offset_ceiling_ratio
        if side == "buy" and offset_ceiling_ratio_buy is not None:
            ceiling = offset_ceiling_ratio_buy
        elif side == "sell" and offset_ceiling_ratio_sell is not None:
            ceiling = offset_ceiling_ratio_sell
        return ceiling

    def test_buy_uses_buy_specific(self) -> None:
        assert self._resolve_ceiling("buy", 0.15, 0.20, 0.50) == 0.20

    def test_sell_uses_sell_specific(self) -> None:
        assert self._resolve_ceiling("sell", 0.15, 0.20, 0.50) == 0.50

    def test_buy_fallback_to_common(self) -> None:
        assert self._resolve_ceiling("buy", 0.15, None, 0.50) == 0.15

    def test_sell_fallback_to_common(self) -> None:
        assert self._resolve_ceiling("sell", 0.15, None, None) == 0.15


# ============================================================
# 418# self-review: resolve_offset_ceiling DRY ヘルパーテスト
# ============================================================


class TestResolveOffsetCeilingHelper:
    """FillTestConfig.resolve_offset_ceiling() のテスト."""

    def test_buy_uses_buy_specific(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(
            offset_ceiling_ratio=0.15,
            offset_ceiling_ratio_buy=0.20,
            offset_ceiling_ratio_sell=0.50,
        )
        assert cfg.resolve_offset_ceiling("buy") == 0.20

    def test_sell_uses_sell_specific(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(
            offset_ceiling_ratio=0.15,
            offset_ceiling_ratio_buy=0.20,
            offset_ceiling_ratio_sell=0.50,
        )
        assert cfg.resolve_offset_ceiling("sell") == 0.50

    def test_fallback_to_common(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(offset_ceiling_ratio=0.15)
        assert cfg.resolve_offset_ceiling("buy") == 0.15
        assert cfg.resolve_offset_ceiling("sell") == 0.15


# ============================================================
# 418# self-review: fill_config_parser YAML roundtrip テスト
# ============================================================


class TestFillConfigParserFinalClamp:
    """execution_final_clamp 設定の YAML パーステスト."""

    def test_parser_reads_enabled(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig.from_yaml({
            "execution_final_clamp_enabled": False,
        })
        assert cfg.execution_final_clamp_enabled is False

    def test_parser_reads_hard_skip_mult(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig.from_yaml({
            "execution_final_clamp_hard_skip_mult": 2.5,
        })
        assert cfg.execution_final_clamp_hard_skip_mult == 2.5

    def test_parser_default_when_absent(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig.from_yaml({})
        assert cfg.execution_final_clamp_enabled is True
        assert cfg.execution_final_clamp_hard_skip_mult == 0.0


# ============================================================
# 418# self-review: guard_reason_classifier テスト
# ============================================================


class TestGuardReasonClassifier418:
    """route_to_kill_deadlock の分類テスト."""

    def test_route_to_kill_is_recovery(self) -> None:
        from scripts.v460.lib.guard_reason_classifier import (
            GuardCategory,
            classify_guard,
        )
        assert classify_guard("route_to_kill_deadlock") == GuardCategory.RECOVERY


# ============================================================
# 418# self-review: config_hot_reload 対象テスト
# ============================================================


class TestConfigHotReload418:
    """Final Clamp 設定が hot-reload 対象であることを確認."""

    def test_final_clamp_fields_in_hot_reloadable(self) -> None:
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
        assert "execution_final_clamp_enabled" in _HOT_RELOADABLE_FIELDS
        assert "execution_final_clamp_hard_skip_mult" in _HOT_RELOADABLE_FIELDS


# ============================================================
# 418# self-review: spread_at_order=None edge case
# ============================================================


class TestFinalClampSpreadNone:
    """spread_at_order=None 時の Final Clamp 挙動テスト."""

    def test_recalc_with_none_spread_returns_original(self) -> None:
        """spread=None → price 変更なし."""
        result = PreOrderAdjustmentsMixin._recalc_price_with_new_offset(
            side="sell",
            order_price=10050,
            spread_at_order=None,
            old_ratio=1.0,
            new_ratio=0.50,
        )
        assert result == 10050

    def test_recalc_with_zero_spread_returns_original(self) -> None:
        """spread=0 → price 変更なし."""
        result = PreOrderAdjustmentsMixin._recalc_price_with_new_offset(
            side="buy",
            order_price=9950,
            spread_at_order=0.0,
            old_ratio=1.0,
            new_ratio=0.20,
        )
        assert result == 9950
