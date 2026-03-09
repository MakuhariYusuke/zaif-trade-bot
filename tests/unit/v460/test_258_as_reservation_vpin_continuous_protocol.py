"""258# AS Reservation Price / VPIN Continuous / RegimeDetectorLike Protocol テスト.

MT-1: Avellaneda-Stoikov reservation price 在庫×ボラ連動 offset stage
MT-3: VPIN binary → continuous quadratic modulator
F-2: RegimeDetectorLike Protocol による型安全化
"""
from __future__ import annotations

import inspect
import math
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator as MakerPrice
from scripts.v460.lib.regime_detector import (
    FillTestRegime,
    FillTestRegimeDetector,
    RegimeDetectorLike,
)
from tests.unit.v460._fill_test_source import ORDER_MONITOR, read_class_method_source


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
        loss_boost_decay_tau_sec=300.0,
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)


def _make_mp(config: FillTestConfig | None = None, **kwargs: object) -> MakerPrice:
    cfg = config or _make_config()
    ffd_cfg = FastFillDefenseConfig(enabled=False)
    ffd = FastFillDefense(ffd_cfg, base_offset_ratio=cfg.spread_offset_ratio)
    return MakerPrice(
        config=cfg,
        fast_fill_defense=ffd,
        regime_detector=kwargs.get("regime_detector"),
        base_offset_ratio=cfg.spread_offset_ratio,
    )


# ======================================================================
# F-2: RegimeDetectorLike Protocol
# ======================================================================


class TestRegimeDetectorProtocol:
    """257# F-2: RegimeDetectorLike Protocol テスト."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol が runtime_checkable で isinstance 判定可能."""
        detector = FillTestRegimeDetector()
        assert isinstance(detector, RegimeDetectorLike)

    def test_protocol_rejects_plain_object(self) -> None:
        """Protocol 非準拠の object は isinstance False."""
        assert not isinstance(object(), RegimeDetectorLike)

    def test_protocol_accepts_mock_with_correct_attrs(self) -> None:
        """current_regime と last_volatility_ratio を持つ mock は Protocol 準拠."""
        mock_det = MagicMock(spec=["current_regime", "last_volatility_ratio"])
        mock_det.current_regime = FillTestRegime.RANGING
        mock_det.last_volatility_ratio = 1.0
        assert isinstance(mock_det, RegimeDetectorLike)

    def test_maker_price_accepts_protocol_detector(self) -> None:
        """MakerPriceCalculator が RegimeDetectorLike を受け取れる."""
        detector = FillTestRegimeDetector()
        mp = _make_mp(regime_detector=detector)
        assert mp._regime_detector is detector

    def test_order_monitor_resolve_regime_name_with_protocol(self) -> None:
        """order_monitor._resolve_regime_name が Protocol 型で動作."""
        from scripts.v460.lib.order_monitor import OrderMonitor

        cfg = _make_config()
        om = OrderMonitor(cfg)

        # None → None
        assert om._resolve_regime_name(None) is None

        # RegimeDetectorLike → value string
        detector = FillTestRegimeDetector()
        name = om._resolve_regime_name(detector)
        assert name == "unknown"  # default regime

    def test_resolve_regime_no_getattr_in_source(self) -> None:
        """257# _resolve_regime_name に getattr/hasattr が残っていない."""
        src = read_class_method_source(
            ORDER_MONITOR,
            "OrderMonitor",
            "_resolve_regime_name",
        )
        assert "getattr" not in src
        assert "hasattr" not in src

    def test_adaptation_engine_accepts_protocol(self) -> None:
        """adaptation_engine の regime_detector パラメータが Protocol 型."""
        from scripts.v460.lib.adaptation_engine import AdaptationEngine

        src = inspect.getsource(AdaptationEngine.try_auto_adapt)
        assert "RegimeDetectorLike" in src

    def test_maker_price_constructor_uses_protocol_type(self) -> None:
        """maker_price.__init__ が RegimeDetectorLike 型を使用."""
        src = inspect.getsource(MakerPrice.__init__)
        assert "RegimeDetectorLike" in src


# ======================================================================
# MT-1: AS Reservation Price Stage
# ======================================================================


class TestASReservationShift:
    """257# MT-1: Avellaneda-Stoikov reservation price offset shift テスト."""

    def test_disabled_by_default(self) -> None:
        """as_reservation_enabled=False なら offset 変化なし."""
        mp = _make_mp()
        result = mp._apply_as_reservation_shift("buy", 1000.0, 10_000_000.0, 0.05)
        assert result == 0.05

    def test_no_shift_when_inventory_neutral(self) -> None:
        """在庫偏重がニュートラルバンド内なら shift なし."""
        cfg = _make_config(
            as_reservation_enabled=True,
            as_reservation_gamma=0.1,
            as_reservation_tau_sec=120.0,
        )
        mp = _make_mp(cfg)
        # inv_net_imbalance = 0 (default)
        result = mp._apply_as_reservation_shift("buy", 1000.0, 10_000_000.0, 0.05)
        assert result == 0.05

    def test_buy_offset_increases_when_long(self) -> None:
        """buy 偏重 (q>0) → buy offset 増加 (less aggressive buying)."""
        cfg = _make_config(
            as_reservation_enabled=True,
            as_reservation_gamma=100.0,  # high gamma for visible effect
            as_reservation_tau_sec=1000.0,
        )
        mp = _make_mp(cfg)
        # Simulate long inventory
        for _ in range(10):
            mp.update_inventory("buy")
        # inv_net_imbalance = 1.0 (all buys)
        initial = 0.05
        result = mp._apply_as_reservation_shift(
            "buy", 10000.0, 10_000_000.0, initial,
        )
        assert result > initial, f"Expected offset > {initial}, got {result}"

    def test_sell_offset_decreases_when_long(self) -> None:
        """buy 偏重 (q>0) → sell offset 減少 (more aggressive selling)."""
        cfg = _make_config(
            as_reservation_enabled=True,
            as_reservation_gamma=100.0,
            as_reservation_tau_sec=1000.0,
        )
        mp = _make_mp(cfg)
        for _ in range(10):
            mp.update_inventory("buy")
        initial = 0.05
        result = mp._apply_as_reservation_shift(
            "sell", 10000.0, 10_000_000.0, initial,
        )
        assert result < initial, f"Expected offset < {initial}, got {result}"

    def test_shift_clamped_by_min_max_offset(self) -> None:
        """shift 結果が min/max_offset_ratio でクランプされる."""
        cfg = _make_config(
            as_reservation_enabled=True,
            as_reservation_gamma=100000.0,  # extreme gamma
            as_reservation_tau_sec=100000.0,
            min_offset_ratio=0.01,
            max_offset_ratio=0.30,
        )
        mp = _make_mp(cfg)
        for _ in range(10):
            mp.update_inventory("buy")
        result = mp._apply_as_reservation_shift(
            "buy", 10000.0, 10_000_000.0, 0.15,
        )
        assert result <= 0.30

    def test_gamma_zero_no_shift(self) -> None:
        """γ=0 (リスク中立) ならシフトなし."""
        cfg = _make_config(
            as_reservation_enabled=True,
            as_reservation_gamma=0.0,
            as_reservation_tau_sec=120.0,
        )
        mp = _make_mp(cfg)
        for _ in range(10):
            mp.update_inventory("buy")
        result = mp._apply_as_reservation_shift("buy", 1000.0, 10_000_000.0, 0.05)
        assert result == 0.05

    def test_shift_proportional_to_spread(self) -> None:
        """spread が広いほど shift が大きい (σ² ∝ (spread/mid)²)."""
        cfg = _make_config(
            as_reservation_enabled=True,
            as_reservation_gamma=100.0,
            as_reservation_tau_sec=1000.0,
        )
        mp_narrow = _make_mp(cfg)
        mp_wide = _make_mp(cfg)
        for _ in range(10):
            mp_narrow.update_inventory("buy")
            mp_wide.update_inventory("buy")

        r_narrow = mp_narrow._apply_as_reservation_shift(
            "buy", 1000.0, 10_000_000.0, 0.05,
        )
        r_wide = mp_wide._apply_as_reservation_shift(
            "buy", 5000.0, 10_000_000.0, 0.05,
        )
        assert r_wide > r_narrow, "Wider spread should produce larger shift"

    def test_pipeline_includes_as_reservation(self) -> None:
        """compute() パイプラインに _apply_as_reservation_shift が含まれる."""
        src = inspect.getsource(MakerPrice.compute)
        assert "_apply_as_reservation_shift" in src


# ======================================================================
# MT-3: VPIN Continuous Modulator
# ======================================================================


class TestVPINContinuousModulator:
    """257# MT-3: VPIN continuous quadratic scaling テスト."""

    def test_binary_mode_unchanged_when_disabled(self) -> None:
        """vg_vpin_continuous_enabled=False で従来バイナリ動作."""
        cfg = _make_config(
            volatility_guard_enabled=True,
            vg_vpin_continuous_enabled=False,
            volatility_guard_vpin_threshold=0.70,
            volatility_guard_offset_boost_factor=2.0,
        )
        mp = _make_mp(cfg)
        mp._last_vpin = 0.50  # below threshold
        result = mp._apply_volatility_guard("buy", None, 0.05)
        assert result == 0.05, "Below threshold should not trigger in binary mode"

    def test_binary_mode_triggers_above_threshold(self) -> None:
        """binary mode: VPIN > threshold で full boost."""
        cfg = _make_config(
            volatility_guard_enabled=True,
            vg_vpin_continuous_enabled=False,
            volatility_guard_vpin_threshold=0.70,
            volatility_guard_offset_boost_factor=2.0,
        )
        mp = _make_mp(cfg)
        mp._last_vpin = 0.80
        result = mp._apply_volatility_guard("buy", None, 0.05)
        assert result > 0.05, "Above threshold should trigger boost"

    def test_continuous_mode_no_effect_below_min(self) -> None:
        """continuous mode: VPIN < min → boost なし."""
        cfg = _make_config(
            volatility_guard_enabled=True,
            vg_vpin_continuous_enabled=True,
            vg_vpin_continuous_min=0.40,
            volatility_guard_vpin_threshold=0.70,
            volatility_guard_offset_boost_factor=2.0,
        )
        mp = _make_mp(cfg)
        mp._last_vpin = 0.30  # below min
        result = mp._apply_volatility_guard("buy", None, 0.05)
        assert result == 0.05

    def test_continuous_mode_partial_boost_between_min_and_threshold(self) -> None:
        """continuous mode: VPIN = mid-range → partial quadratic boost."""
        cfg = _make_config(
            volatility_guard_enabled=True,
            vg_vpin_continuous_enabled=True,
            vg_vpin_continuous_min=0.40,
            volatility_guard_vpin_threshold=0.70,
            volatility_guard_offset_boost_factor=2.0,
        )
        mp = _make_mp(cfg)
        # VPIN = 0.55 → norm = (0.55 - 0.40) / (0.70 - 0.40) = 0.50
        # quadratic: 0.50^2 = 0.25 → boost = 1 + (2-1)*0.25 = 1.25
        mp._last_vpin = 0.55
        initial = 0.05
        result = mp._apply_volatility_guard("buy", None, initial)
        expected_boost = 1.25
        expected = initial * expected_boost
        assert abs(result - expected) < 0.001, (
            f"Expected ~{expected:.4f}, got {result:.4f}"
        )

    def test_continuous_mode_full_boost_at_threshold(self) -> None:
        """continuous mode: VPIN = threshold → full boost (norm=1.0)."""
        cfg = _make_config(
            volatility_guard_enabled=True,
            vg_vpin_continuous_enabled=True,
            vg_vpin_continuous_min=0.40,
            volatility_guard_vpin_threshold=0.70,
            volatility_guard_offset_boost_factor=2.0,
        )
        mp = _make_mp(cfg)
        mp._last_vpin = 0.70  # exactly at threshold
        initial = 0.05
        result = mp._apply_volatility_guard("buy", None, initial)
        expected = initial * 2.0  # full boost factor
        assert abs(result - expected) < 0.001

    def test_continuous_mode_capped_above_threshold(self) -> None:
        """continuous mode: VPIN > threshold → boost は full (1.0 clamp)."""
        cfg = _make_config(
            volatility_guard_enabled=True,
            vg_vpin_continuous_enabled=True,
            vg_vpin_continuous_min=0.40,
            volatility_guard_vpin_threshold=0.70,
            volatility_guard_offset_boost_factor=2.0,
        )
        mp = _make_mp(cfg)
        mp._last_vpin = 0.90  # above threshold, capped at norm=1.0
        initial = 0.05
        result = mp._apply_volatility_guard("buy", None, initial)
        expected = initial * 2.0
        assert abs(result - expected) < 0.001

    def test_continuous_quadratic_shape(self) -> None:
        """continuous mode: VPIN 増加に対して二次関数的に boost が増加."""
        cfg = _make_config(
            volatility_guard_enabled=True,
            vg_vpin_continuous_enabled=True,
            vg_vpin_continuous_min=0.40,
            volatility_guard_vpin_threshold=0.70,
            volatility_guard_offset_boost_factor=2.0,
        )
        initial = 0.05
        results = []
        for vpin in [0.45, 0.50, 0.55, 0.60, 0.65]:
            mp = _make_mp(cfg)
            mp._last_vpin = vpin
            r = mp._apply_volatility_guard("buy", None, initial)
            results.append(r)

        # Verify monotonically increasing
        for i in range(1, len(results)):
            assert results[i] > results[i - 1], (
                f"VPIN boost should increase monotonically: "
                f"{results[i - 1]:.4f} -> {results[i]:.4f}"
            )

        # Verify quadratic shape: increments should increase
        diffs = [results[i] - results[i - 1] for i in range(1, len(results))]
        for i in range(1, len(diffs)):
            assert diffs[i] > diffs[i - 1], (
                f"Quadratic ramp: increments should increase: "
                f"{diffs[i - 1]:.6f} -> {diffs[i]:.6f}"
            )

    def test_velocity_trigger_takes_priority(self) -> None:
        """velocity trigger と VPIN continuous の max が適用される."""
        cfg = _make_config(
            volatility_guard_enabled=True,
            vg_vpin_continuous_enabled=True,
            vg_vpin_continuous_min=0.40,
            volatility_guard_vpin_threshold=0.70,
            volatility_guard_velocity_threshold_bps=15.0,
            volatility_guard_offset_boost_factor=2.0,
        )
        mp = _make_mp(cfg)
        mp._last_vpin = 0.50  # partial VPIN boost < full
        initial = 0.05
        # velocity trigger → full boost = 2.0
        result = mp._apply_volatility_guard("buy", 20.0, initial)
        expected = initial * 2.0
        assert abs(result - expected) < 0.001

    def test_vg_source_no_binary_vpin_when_continuous(self) -> None:
        """VG ソースに continuous mode 分岐が存在."""
        src = inspect.getsource(MakerPrice._apply_volatility_guard)
        assert "vg_vpin_continuous_enabled" in src
        assert "quadratic" in src.lower() or "_norm * _norm" in src


# ======================================================================
# 353# VPIN 非対称 buy boost
# ======================================================================


class TestVPINBuyExtraMult:
    """353# VPIN asymmetric buy boost — buy 側の VPIN boost 追加増幅."""

    def test_default_mult_1_no_change(self) -> None:
        """vg_vpin_buy_extra_mult=1.0 (default): buy と sell で同じ boost."""
        cfg = _make_config(
            volatility_guard_enabled=True,
            vg_vpin_continuous_enabled=True,
            vg_vpin_continuous_min=0.40,
            volatility_guard_vpin_threshold=0.70,
            volatility_guard_offset_boost_factor=2.0,
            vg_vpin_buy_extra_mult=1.0,
        )
        mp_buy = _make_mp(cfg)
        mp_buy._last_vpin = 0.60
        mp_sell = _make_mp(cfg)
        mp_sell._last_vpin = 0.60
        buy_result = mp_buy._apply_volatility_guard("buy", None, 0.05)
        sell_result = mp_sell._apply_volatility_guard("sell", None, 0.05)
        assert abs(buy_result - sell_result) < 1e-8

    def test_buy_extra_mult_increases_buy_boost(self) -> None:
        """vg_vpin_buy_extra_mult=1.5: buy の boost が sell より大きい."""
        cfg = _make_config(
            volatility_guard_enabled=True,
            vg_vpin_continuous_enabled=True,
            vg_vpin_continuous_min=0.40,
            volatility_guard_vpin_threshold=0.70,
            volatility_guard_offset_boost_factor=2.0,
            vg_vpin_buy_extra_mult=1.5,
        )
        mp_buy = _make_mp(cfg)
        mp_buy._last_vpin = 0.60
        mp_sell = _make_mp(cfg)
        mp_sell._last_vpin = 0.60
        buy_result = mp_buy._apply_volatility_guard("buy", None, 0.05)
        sell_result = mp_sell._apply_volatility_guard("sell", None, 0.05)
        assert buy_result > sell_result, (
            f"buy boost should be larger: buy={buy_result:.4f} vs sell={sell_result:.4f}"
        )

    def test_buy_extra_mult_math_correctness(self) -> None:
        """vg_vpin_buy_extra_mult の数学的正しさを検証.

        VPIN=0.55, min=0.40, threshold=0.70 → norm=0.5, boost=1+0.5²=1.25
        buy_extra_mult=1.5 → buy_boost=1+(0.25)*1.5=1.375
        """
        cfg = _make_config(
            volatility_guard_enabled=True,
            vg_vpin_continuous_enabled=True,
            vg_vpin_continuous_min=0.40,
            volatility_guard_vpin_threshold=0.70,
            volatility_guard_offset_boost_factor=2.0,
            vg_vpin_buy_extra_mult=1.5,
        )
        mp = _make_mp(cfg)
        mp._last_vpin = 0.55
        initial = 0.05
        result = mp._apply_volatility_guard("buy", None, initial)
        # norm = (0.55 - 0.40) / (0.70 - 0.40) = 0.5
        # base vpin_boost = 1 + (2.0 - 1) * 0.25 = 1.25
        # buy extra: 1 + (1.25 - 1) * 1.5 = 1.375
        expected = initial * 1.375
        assert abs(result - expected) < 0.001, (
            f"Expected {expected:.4f}, got {result:.4f}"
        )

    def test_sell_unaffected_by_buy_extra_mult(self) -> None:
        """sell は vg_vpin_buy_extra_mult の影響を受けない."""
        cfg_base = _make_config(
            volatility_guard_enabled=True,
            vg_vpin_continuous_enabled=True,
            vg_vpin_continuous_min=0.40,
            volatility_guard_vpin_threshold=0.70,
            volatility_guard_offset_boost_factor=2.0,
            vg_vpin_buy_extra_mult=1.0,
        )
        cfg_extra = _make_config(
            volatility_guard_enabled=True,
            vg_vpin_continuous_enabled=True,
            vg_vpin_continuous_min=0.40,
            volatility_guard_vpin_threshold=0.70,
            volatility_guard_offset_boost_factor=2.0,
            vg_vpin_buy_extra_mult=2.0,
        )
        mp_base = _make_mp(cfg_base)
        mp_base._last_vpin = 0.60
        mp_extra = _make_mp(cfg_extra)
        mp_extra._last_vpin = 0.60
        sell_base = mp_base._apply_volatility_guard("sell", None, 0.05)
        sell_extra = mp_extra._apply_volatility_guard("sell", None, 0.05)
        assert abs(sell_base - sell_extra) < 1e-8, "sell should be unaffected"
