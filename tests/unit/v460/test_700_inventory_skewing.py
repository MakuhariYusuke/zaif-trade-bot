from __future__ import annotations

import time

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator
from ztb.metrics.fill_quality import build_skip_fill_record
from ztb.trading.signal.regime.regime_detector import FillTestRegime


class _StaticFFD:
    def maybe_expire_boost(self, _side: str) -> None:
        return None

    def _get_dynamic_boost(self, _: str) -> float | None:
        return None

    def get_boost_multiplier(self, _side: str) -> float:
        return 1.0


def _make_config(**overrides: object) -> FillTestConfig:
    defaults: dict[str, object] = dict(
        spread_offset_ratio=0.001,
        min_offset_jpy=1.0,
        max_offset_ratio=0.02,
        min_offset_ratio=0.0001,
        inventory_skewing_enabled=True,
        inventory_skewing_window=300,
        inventory_skewing_max_factor=0.4,
        inventory_skewing_max_factor_drift=0.6,
        inv_skew_max_factor_trending=0.15,
        inventory_skewing_neutral_band=0.05,
        drift_detection_threshold=0.15,
        drift_detection_sustain_sec=1800.0,
        spread_adaptive_enabled=False,
        imbalance_enabled=False,
        volatility_guard_enabled=False,
        fast_fill_defense_enabled=False,
        sell_offset_floor=0.0,
        sell_max_spread_jpy=0.0,
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)


def _make_maker_price(
    config: FillTestConfig,
    *,
    regime_value: str | None = None,
) -> MakerPriceCalculator:
    regime_detector = None
    if regime_value is not None:
        class _Detector:
            current_regime = FillTestRegime(regime_value)

        regime_detector = _Detector()
    return MakerPriceCalculator(
        config=config,
        fast_fill_defense=_StaticFFD(),
        regime_detector=regime_detector,
        base_offset_ratio=config.spread_offset_ratio,
    )


def _inject_imbalance(mp: MakerPriceCalculator, imbalance: float) -> None:
    n = mp._config.inventory_skewing_window
    mp._inv_fill_history.clear()
    n_buy = int(round(n * ((1 + imbalance) / 2)))
    n_buy = max(0, min(n, n_buy))
    n_sell = n - n_buy
    for _ in range(n_buy):
        mp._inv_fill_history.append("buy")
    for _ in range(n_sell):
        mp._inv_fill_history.append("sell")
    mp._inv_buy_count = n_buy
    mp._inv_net_imbalance = (2 * n_buy / n - 1) if n > 0 else 0.0
    mp._inv_last_update_time = time.time()


def test_window_expansion_default_300() -> None:
    assert FillTestConfig().inventory_skewing_window == 300


def test_window_backward_compat_explicit_100() -> None:
    assert FillTestConfig(inventory_skewing_window=100).inventory_skewing_window == 100


def test_drift_detection_requires_sustain() -> None:
    mp = _make_maker_price(_make_config(drift_detection_sustain_sec=60.0))
    assert mp._update_inventory_drift_state(0.2, 100.0) is False
    assert mp._update_inventory_drift_state(0.2, 159.0) is False
    assert mp._update_inventory_drift_state(0.2, 160.0) is True


def test_drift_reset_on_recovery() -> None:
    mp = _make_maker_price(_make_config(drift_detection_sustain_sec=10.0))
    assert mp._update_inventory_drift_state(0.2, 100.0) is False
    assert mp._update_inventory_drift_state(0.2, 111.0) is True
    assert mp._update_inventory_drift_state(0.05, 112.0) is False
    assert mp._drift_start_time is None


def test_drift_escalates_effective_max_factor() -> None:
    mp = _make_maker_price(_make_config(drift_detection_sustain_sec=0.0))
    _inject_imbalance(mp, 0.4)

    updated = mp._apply_inventory_skew("buy", time.time(), 0.05)

    assert updated > 0.05
    assert mp.last_inv_skew_drift_detected is True
    assert mp.last_inv_skew_effective_max_factor == pytest.approx(0.6)


def test_trending_uses_max_of_trending_and_drift() -> None:
    mp = _make_maker_price(
        _make_config(
            inventory_skewing_max_factor_drift=0.6,
            inv_skew_max_factor_trending=0.15,
            drift_detection_sustain_sec=0.0,
        ),
        regime_value="trending_down",
    )
    _inject_imbalance(mp, 0.4)

    mp._apply_inventory_skew("buy", time.time(), 0.05)

    assert mp.last_inv_skew_effective_max_factor == pytest.approx(0.6)


def test_fill_record_accepts_drift_fields() -> None:
    record = build_skip_fill_record(
        cycle_id="c1",
        timestamp=1.0,
        side="buy",
        order_price=100.0,
        order_quantity=0.1,
        cancel_reason="skip",
        run_id="run",
        git_sha="sha",
        inv_skew_drift_detected=True,
        inv_skew_effective_max_factor=0.6,
    )

    assert record.inv_skew_drift_detected is True
    assert record.inv_skew_effective_max_factor == pytest.approx(0.6)
