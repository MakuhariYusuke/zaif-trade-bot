"""259# テスト — AS σ² vol_ratio 統合 + adaptation_engine hasattr 排除.

MT-4: Avellaneda-Stoikov σ² 推定の RegimeDetector volatility_ratio 統合
F-3: adaptation_engine hasattr(regime_detector, "current_regime") 排除
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ztb.trading.risk.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator as MakerPrice
from ztb.trading.signal.regime.regime_detector import (
    FillTestRegime,
    FillTestRegimeDetector,
    RegimeDetectorLike,
)
from tests.unit.v460._fill_test_source import (
    ADAPTATION_ENGINE,
    MAKER_MICROSTRUCTURE,
    read_class_method_source,
)

_ESTIMATE_SIGMA_SOURCE = read_class_method_source(
    MAKER_MICROSTRUCTURE,
    "MicrostructureMixin",
    "_estimate_sigma",
)
_TRY_AUTO_ADAPT_SOURCE = read_class_method_source(
    ADAPTATION_ENGINE,
    "AdaptationEngine",
    "try_auto_adapt",
)
_TRY_AUTO_LOT_SIZE_SOURCE = read_class_method_source(
    ADAPTATION_ENGINE,
    "AdaptationEngine",
    "try_auto_lot_size",
)


# ======================================================================
# helpers
# ======================================================================


def _make_config(**overrides: object) -> FillTestConfig:
    defaults: dict[str, object] = dict(
        spread_offset_ratio=0.001,
        min_offset_jpy=1.0,
        max_offset_ratio=0.30,
        min_offset_ratio=0.01,
        inventory_skewing_enabled=True,
        inventory_skewing_window=10,
        inventory_skewing_max_factor=0.5,
        inventory_skewing_neutral_band=0.1,
        inv_decay_tau_sec=0.0,
        loss_boost_decay_tau_sec=300.0,
        as_reservation_enabled=True,
        as_reservation_gamma=100.0,
        as_reservation_tau_sec=1000.0,
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)


def _make_mock_detector(
    regime: FillTestRegime = FillTestRegime.RANGING,
    vol_ratio: float = 1.0,
    confidence: float = 0.8,
) -> RegimeDetectorLike:
    """RegimeDetectorLike 準拠の mock を生成."""
    return SimpleNamespace(
        current_regime=regime,
        last_volatility_ratio=vol_ratio,
        current_confidence=confidence,
    )


def _make_mp(
    config: FillTestConfig | None = None,
    regime_detector: RegimeDetectorLike | None = None,
) -> MakerPrice:
    cfg = config or _make_config()
    ffd_cfg = FastFillDefenseConfig(enabled=False)
    ffd = FastFillDefense(ffd_cfg, base_offset_ratio=cfg.spread_offset_ratio)
    return MakerPrice(
        config=cfg,
        fast_fill_defense=ffd,
        regime_detector=regime_detector,
        base_offset_ratio=cfg.spread_offset_ratio,
    )


# ======================================================================
# MT-4: AS σ² vol_ratio 統合
# ======================================================================


class TestASVolRatioIntegration:
    """258# MT-4: AS σ² × vol_ratio hybrid estimator."""

    def test_vol_ratio_amplifies_shift_in_high_vol(self) -> None:
        """vol_ratio > 1 → σ 増幅 → shift 増大."""
        det_normal = _make_mock_detector(vol_ratio=1.0)
        det_high = _make_mock_detector(vol_ratio=2.0)

        cfg = _make_config()
        mp_normal = _make_mp(cfg, regime_detector=det_normal)
        mp_high = _make_mp(cfg, regime_detector=det_high)

        for _ in range(10):
            mp_normal.update_inventory("buy")
            mp_high.update_inventory("buy")

        initial = 0.05
        r_normal = mp_normal._apply_as_reservation_shift(
            "buy", 10000.0, 10_000_000.0, initial,
        )
        r_high = mp_high._apply_as_reservation_shift(
            "buy", 10000.0, 10_000_000.0, initial,
        )
        assert r_high > r_normal, (
            f"High vol ratio should amplify shift: normal={r_normal}, high={r_high}"
        )

    def test_vol_ratio_suppresses_shift_in_low_vol(self) -> None:
        """vol_ratio < 1 → σ 抑制 → shift 減少."""
        det_normal = _make_mock_detector(vol_ratio=1.0)
        det_low = _make_mock_detector(vol_ratio=0.5)

        cfg = _make_config()
        mp_normal = _make_mp(cfg, regime_detector=det_normal)
        mp_low = _make_mp(cfg, regime_detector=det_low)

        for _ in range(10):
            mp_normal.update_inventory("buy")
            mp_low.update_inventory("buy")

        initial = 0.05
        r_normal = mp_normal._apply_as_reservation_shift(
            "buy", 10000.0, 10_000_000.0, initial,
        )
        r_low = mp_low._apply_as_reservation_shift(
            "buy", 10000.0, 10_000_000.0, initial,
        )
        assert r_low < r_normal, (
            f"Low vol ratio should suppress shift: normal={r_normal}, low={r_low}"
        )

    def test_vol_ratio_1_equiv_to_no_detector(self) -> None:
        """vol_ratio=1.0 → regime_detector=None と同等 (Roll proxy のみ)."""
        det_one = _make_mock_detector(vol_ratio=1.0)

        cfg = _make_config()
        mp_with = _make_mp(cfg, regime_detector=det_one)
        mp_without = _make_mp(cfg, regime_detector=None)

        for _ in range(10):
            mp_with.update_inventory("buy")
            mp_without.update_inventory("buy")

        initial = 0.05
        r_with = mp_with._apply_as_reservation_shift(
            "buy", 10000.0, 10_000_000.0, initial,
        )
        r_without = mp_without._apply_as_reservation_shift(
            "buy", 10000.0, 10_000_000.0, initial,
        )
        assert r_with == pytest.approx(r_without, abs=1e-10), (
            f"vol_ratio=1.0 should be equivalent to no detector: "
            f"with={r_with}, without={r_without}"
        )

    def test_vol_ratio_floor_prevents_zero_sigma(self) -> None:
        """vol_ratio=0 → floor=0.1 で σ が完全 0 にならない."""
        det_zero = _make_mock_detector(vol_ratio=0.0)

        cfg = _make_config()
        mp = _make_mp(cfg, regime_detector=det_zero)

        for _ in range(10):
            mp.update_inventory("buy")

        initial = 0.05
        result = mp._apply_as_reservation_shift(
            "buy", 10000.0, 10_000_000.0, initial,
        )
        # vol_ratio=0 → max(0, 0.1) = 0.1 → σ is 10% of Roll estimate
        # shift should be small but non-zero
        assert result > initial, "Floor should prevent zero sigma"

    def test_vol_ratio_quadratic_sigma_effect(self) -> None:
        """σ² ∝ vol_ratio² → shift は vol_ratio の 2乗に比例."""
        cfg = _make_config()
        initial = 0.05

        results = []
        for vr in [0.5, 1.0, 2.0]:
            det = _make_mock_detector(vol_ratio=vr)
            mp = _make_mp(cfg, regime_detector=det)
            for _ in range(10):
                mp.update_inventory("buy")
            r = mp._apply_as_reservation_shift(
                "buy", 10000.0, 10_000_000.0, initial,
            )
            results.append(r)

        # shift ratio between vol_ratio=2 and vol_ratio=1 should be ~4× the shift
        # (since σ² ∝ vol_ratio²)
        shift_1 = results[1] - initial  # vol_ratio=1.0
        shift_2 = results[2] - initial  # vol_ratio=2.0
        if shift_1 > 1e-12:
            ratio = shift_2 / shift_1
            assert ratio == pytest.approx(4.0, rel=0.01), (
                f"Shift should scale with vol_ratio²: "
                f"ratio={ratio:.3f}, expected=4.0"
            )

    def test_source_references_vol_ratio(self) -> None:
        """ソースに vol_ratio 統合コードが存在."""
        src = _ESTIMATE_SIGMA_SOURCE
        assert "vol_ratio" in src
        assert "_regime_detector" in src
        assert "last_volatility_ratio" in src


# ======================================================================
# F-3: adaptation_engine hasattr 排除
# ======================================================================


class TestAdaptationEngineHasattrRemoval:
    """258# F-3: adaptation_engine から hasattr(regime_detector) 排除."""

    def test_try_auto_adapt_no_hasattr(self) -> None:
        """try_auto_adapt に hasattr が残っていない."""
        src = _TRY_AUTO_ADAPT_SOURCE
        assert "hasattr" not in src, (
            "try_auto_adapt should not use hasattr — "
            "RegimeDetectorLike Protocol makes it unnecessary"
        )

    def test_try_auto_lot_size_no_hasattr(self) -> None:
        """try_auto_lot_size に hasattr が残っていない."""
        src = _TRY_AUTO_LOT_SIZE_SOURCE
        assert "hasattr" not in src, (
            "try_auto_lot_size should not use hasattr — "
            "RegimeDetectorLike Protocol makes it unnecessary"
        )

    def test_regime_tag_extraction_with_none(self) -> None:
        """regime_detector=None → regime_tag='n/a'."""
        # ソースコードで regime_detector is not None 分岐があることを確認
        src_adapt = _TRY_AUTO_ADAPT_SOURCE
        src_lot = _TRY_AUTO_LOT_SIZE_SOURCE
        assert "n/a" in src_adapt
        assert "n/a" in src_lot

    def test_regime_tag_uses_direct_access(self) -> None:
        """regime_detector.current_regime.value を直接アクセスしている."""
        src = _TRY_AUTO_ADAPT_SOURCE
        assert "regime_detector.current_regime.value" in src
        assert "getattr" not in src
