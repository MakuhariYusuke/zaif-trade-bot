from __future__ import annotations

from scripts.v460.lib import fast_fill_defense as shim_ffd
from scripts.v460.lib import lot_sizer as shim_lot
from ztb.trading.risk import fast_fill_defense as canonical_ffd
from ztb.trading.sizing import lot_sizer as canonical_lot


class TestLotSizerCanonicalMigration:
    def test_shim_and_canonical_lot_config_defaults_match(self) -> None:
        assert shim_lot.LotSizingConfig() == canonical_lot.LotSizingConfig()

    def test_shim_and_canonical_lot_compute_match(self) -> None:
        shim_result = shim_lot.compute_lot_size(
            fill_rate=0.8,
            as_ratio=0.1,
            recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0,
            sample_count=100,
            config=shim_lot.LotSizingConfig(current_lot=0.001, lot_step=0.001),
        )
        canonical_result = canonical_lot.compute_lot_size(
            fill_rate=0.8,
            as_ratio=0.1,
            recent_pnl_bps=0.5,
            cumulative_pnl_jpy=100.0,
            sample_count=100,
            config=canonical_lot.LotSizingConfig(current_lot=0.001, lot_step=0.001),
        )
        assert shim_result == canonical_result


class TestFastFillDefenseCanonicalMigration:
    def test_shim_and_canonical_defaults_match(self) -> None:
        assert shim_ffd.FastFillDefenseConfig() == canonical_ffd.FastFillDefenseConfig()

    def test_shim_and_canonical_behavior_match(self) -> None:
        shim_defense = shim_ffd.FastFillDefense(
            shim_ffd.FastFillDefenseConfig(enabled=True, threshold_sec=5.0, offset_boost=2.0),
            base_offset_ratio=0.05,
        )
        canonical_defense = canonical_ffd.FastFillDefense(
            canonical_ffd.FastFillDefenseConfig(enabled=True, threshold_sec=5.0, offset_boost=2.0),
            base_offset_ratio=0.05,
        )

        shim_defense.evaluate_fill("buy", queue_wait_sec=2.0, fill_price=101_000, mid_at_fill=100_000)
        canonical_defense.evaluate_fill(
            "buy", queue_wait_sec=2.0, fill_price=101_000, mid_at_fill=100_000
        )

        assert shim_defense.get_boost_multiplier("buy") == canonical_defense.get_boost_multiplier("buy")
        assert shim_defense.is_boost_active("buy") == canonical_defense.is_boost_active("buy")
