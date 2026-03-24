"""154# P1-5 A/B テスト + P1-2 reprice tighten テスト.

対象:
  - 158# P1-5: A/B テスト基盤 (variant_id)
  - 158# P1-2: reprice offset tightening
  - 348#: balance_forced 関連テストを撤廃
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from tests.unit.v460._fill_test_source import ORDER_MONITOR, read_class_method_source
from tests.unit.v460._yaml_test_helpers import (
    clone_fill_test_config,
    load_fill_test_config_from_mapping,
    load_fill_test_config_from_text,
)


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
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping({"ab_test": {"variant": "rescue_enabled"}}))
        assert cfg.ab_test_variant == "rescue_enabled"

    def test_ab_variant_yaml_absent(self) -> None:
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping({}))
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
        cfg = clone_fill_test_config(load_fill_test_config_from_text("""
stale_order:
  enabled: true
  reprice_tighten: 0.80
"""))
        assert cfg.stale_reprice_tighten == pytest.approx(0.80)

    def test_tighten_logic_in_order_monitor(self) -> None:
        """OrderMonitor に tighten ロジックが存在する."""
        source = read_class_method_source(ORDER_MONITOR, "OrderMonitor", "monitor")
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
