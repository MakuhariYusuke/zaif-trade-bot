"""506# tests: sell age cap, de-meaning (EMA basis), offset narrowing."""
from __future__ import annotations

import pytest

from scripts.v460.lib.cross_venue_lead_lag import (
    CrossVenueEMAState,
    VenueMidSnapshot,
    compute_cross_venue_lead_lag_hint,
    update_cross_venue_ema,
)


class TestUpdateCrossVenueEmaBasis:
    """506# ema_basis_bps tracking in update_cross_venue_ema."""

    def test_basis_alpha_zero_keeps_default(self) -> None:
        state = update_cross_venue_ema(
            None, ref_mid=100.0, spread_bps=-3.3, timestamp=1.0, alpha=0.3,
            basis_alpha=0.0,
        )
        assert state.ema_basis_bps == 0.0

    def test_basis_alpha_positive_initializes_from_first_spread(self) -> None:
        state = update_cross_venue_ema(
            None, ref_mid=100.0, spread_bps=-3.3, timestamp=1.0, alpha=0.3,
            basis_alpha=0.02,
        )
        assert state.ema_basis_bps == -3.3
        assert state.n_updates == 1

    def test_basis_ema_converges_toward_spread(self) -> None:
        state = update_cross_venue_ema(
            None, ref_mid=100.0, spread_bps=-3.0, timestamp=1.0, alpha=0.3,
            basis_alpha=0.5,
        )
        # Initial: ema_basis = -3.0
        state = update_cross_venue_ema(
            state, ref_mid=100.0, spread_bps=-5.0, timestamp=2.0, alpha=0.3,
            basis_alpha=0.5,
        )
        # ema_basis = 0.5 * (-5.0) + 0.5 * (-3.0) = -4.0
        assert state.ema_basis_bps == pytest.approx(-4.0)

    def test_basis_ema_stable_when_spread_constant(self) -> None:
        state = update_cross_venue_ema(
            None, ref_mid=100.0, spread_bps=-3.3, timestamp=1.0, alpha=0.3,
            basis_alpha=0.02,
        )
        for i in range(100):
            state = update_cross_venue_ema(
                state, ref_mid=100.0, spread_bps=-3.3,
                timestamp=2.0 + i, alpha=0.3, basis_alpha=0.02,
            )
        assert state.ema_basis_bps == pytest.approx(-3.3, abs=0.01)


class TestDeMeaningDirection:
    """506# basis_bps parameter changes direction determination."""

    def _local(self) -> VenueMidSnapshot:
        return VenueMidSnapshot("coincheck", 100.0, 100.0)

    def _ref(self) -> VenueMidSnapshot:
        # ref mid < local mid → spread < 0 → normally direction="down"
        return VenueMidSnapshot("bitflyer", 99.97, 100.5)

    def _prev_ref(self) -> VenueMidSnapshot:
        return VenueMidSnapshot("bitflyer", 99.95, 99.5)

    def test_without_basis_direction_is_down(self) -> None:
        """Without basis correction, spread=-3bps → direction=down → adverse=buy."""
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._local(),
            reference_snapshot=self._ref(),
            previous_reference_snapshot=self._prev_ref(),
            max_age_sec=3.0,
            spread_bps_threshold=1.0,
            velocity_bps_threshold=0.01,
            ema_spread_bps=-3.0,
            basis_bps=0.0,  # no correction
        )
        assert hint is not None
        assert hint.direction == "down"
        assert hint.adverse_side == "buy"

    def test_with_basis_correction_direction_flips(self) -> None:
        """With basis=-4.0, adjusted_spread=-3.0-(-4.0)=+1.0 → direction=up → adverse=sell."""
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._local(),
            reference_snapshot=self._ref(),
            previous_reference_snapshot=self._prev_ref(),
            max_age_sec=3.0,
            spread_bps_threshold=1.0,
            velocity_bps_threshold=0.01,
            ema_spread_bps=-3.0,
            basis_bps=-4.0,  # historical basis
        )
        assert hint is not None
        assert hint.direction == "up"
        assert hint.adverse_side == "sell"

    def test_basis_zero_is_backward_compatible(self) -> None:
        """basis_bps=0.0 produces same result as before 506#."""
        kwargs = dict(
            local_snapshot=VenueMidSnapshot("coincheck", 100.0, 100.0),
            reference_snapshot=VenueMidSnapshot("bitflyer", 100.6, 100.5),
            previous_reference_snapshot=VenueMidSnapshot("bitflyer", 100.2, 99.5),
            max_age_sec=3.0,
            spread_bps_threshold=1.0,
            velocity_bps_threshold=0.01,
            ema_spread_bps=6.0,
        )
        hint_no_basis = compute_cross_venue_lead_lag_hint(**kwargs)
        hint_zero_basis = compute_cross_venue_lead_lag_hint(**kwargs, basis_bps=0.0)
        assert hint_no_basis is not None
        assert hint_zero_basis is not None
        assert hint_no_basis.direction == hint_zero_basis.direction
        assert hint_no_basis.adverse_side == hint_zero_basis.adverse_side


class TestSellAgeCap:
    """506# sell_age_cap_sec config field test."""

    def test_fill_config_has_sell_age_cap_field(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.sell_age_cap_sec is None  # default disabled

    def test_sell_age_cap_in_hot_reloadable(self) -> None:
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
        assert "sell_age_cap_sec" in _HOT_RELOADABLE_FIELDS

    def test_basis_correction_fields_in_hot_reloadable(self) -> None:
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
        assert "cross_venue_basis_correction_enabled" in _HOT_RELOADABLE_FIELDS
        assert "cross_venue_basis_ema_alpha" in _HOT_RELOADABLE_FIELDS
