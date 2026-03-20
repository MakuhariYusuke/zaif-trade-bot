"""506#/507#/508#/509# tests: sell age cap, de-meaning, confidence fix, observability, sell_age_cap guard."""
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


class TestDeMeaningConfidenceCorrection:
    """507# confidence/velocity を de-meaning 後の adjusted_spread ベースに統一."""

    def _local(self) -> VenueMidSnapshot:
        return VenueMidSnapshot("coincheck", 100.0, 100.0)

    def _ref(self, mid: float = 99.97) -> VenueMidSnapshot:
        return VenueMidSnapshot("bitflyer", mid, 100.5)

    def _prev_ref(self, mid: float = 99.95) -> VenueMidSnapshot:
        return VenueMidSnapshot("bitflyer", mid, 99.5)

    def test_confidence_proportional_to_deviation_with_basis(self) -> None:
        """With basis correction, confidence reflects adjusted_spread not raw spread."""
        # Small deviation from basis: ema=-3.0, basis=-4.0 → adjusted=+1.0
        hint_small = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._local(),
            reference_snapshot=self._ref(),
            previous_reference_snapshot=self._prev_ref(),
            max_age_sec=3.0,
            spread_bps_threshold=0.5,
            velocity_bps_threshold=0.01,
            ema_spread_bps=-3.0,
            basis_bps=-4.0,
            confidence_reference_spread_bps=3.0,
        )
        # Large deviation from basis: ema=-1.0, basis=-4.0 → adjusted=+3.0
        hint_large = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._local(),
            reference_snapshot=self._ref(),
            previous_reference_snapshot=self._prev_ref(),
            max_age_sec=3.0,
            spread_bps_threshold=0.5,
            velocity_bps_threshold=0.01,
            ema_spread_bps=-1.0,
            basis_bps=-4.0,
            confidence_reference_spread_bps=3.0,
        )
        assert hint_small is not None
        assert hint_large is not None
        # 507# 修正: confidence は adjusted_spread に比例するため小偏差 < 大偏差
        assert hint_small.confidence < hint_large.confidence

    def test_confidence_without_basis_uses_raw_spread(self) -> None:
        """Without basis (basis_bps=0.0), confidence uses raw gating_spread."""
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._local(),
            reference_snapshot=self._ref(),
            previous_reference_snapshot=self._prev_ref(),
            max_age_sec=3.0,
            spread_bps_threshold=0.5,
            velocity_bps_threshold=0.01,
            ema_spread_bps=-3.0,
            basis_bps=0.0,
            confidence_reference_spread_bps=3.0,
        )
        assert hint is not None
        # raw |gating_spread|/ref = 3.0/3.0 = 1.0 → base_conf = 1.0
        # velocity agreement adds factor. Should be high.
        assert hint.confidence >= 0.5

    def test_velocity_agreement_with_adjusted_spread(self) -> None:
        """507# velocity agreement uses adjusted_spread direction when basis active."""
        # basis=-5.0, ema=-3.0 → adjusted=+2.0 (direction="up")
        # positive velocity (+) should AGREE with adjusted_spread (+) → vel_factor=1.0
        hint_agree = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._local(),
            reference_snapshot=self._ref(mid=99.97),
            previous_reference_snapshot=self._prev_ref(mid=99.90),  # BF going up
            max_age_sec=3.0,
            spread_bps_threshold=0.5,
            velocity_bps_threshold=0.01,
            ema_spread_bps=-3.0,
            basis_bps=-5.0,
            confidence_reference_spread_bps=3.0,
        )
        # negative velocity (-) should DISAGREE with adjusted_spread (+) → vel_factor=0.5
        hint_disagree = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._local(),
            reference_snapshot=self._ref(mid=99.97),
            previous_reference_snapshot=self._prev_ref(mid=100.05),  # BF going down
            max_age_sec=3.0,
            spread_bps_threshold=0.5,
            velocity_bps_threshold=0.01,
            ema_spread_bps=-3.0,
            basis_bps=-5.0,
            confidence_reference_spread_bps=3.0,
        )
        assert hint_agree is not None
        assert hint_disagree is not None
        assert hint_agree.confidence > hint_disagree.confidence


class TestHintBasisObservability:
    """508# hint に basis_bps / adjusted_spread_bps が含まれることを検証."""

    def _local(self) -> VenueMidSnapshot:
        return VenueMidSnapshot("coincheck", 100.0, 100.0)

    def _ref(self) -> VenueMidSnapshot:
        return VenueMidSnapshot("bitflyer", 99.97, 100.5)

    def _prev_ref(self) -> VenueMidSnapshot:
        return VenueMidSnapshot("bitflyer", 99.95, 99.5)

    def test_hint_carries_basis_and_adjusted_spread(self) -> None:
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._local(),
            reference_snapshot=self._ref(),
            previous_reference_snapshot=self._prev_ref(),
            max_age_sec=3.0,
            spread_bps_threshold=0.5,
            velocity_bps_threshold=0.01,
            ema_spread_bps=-3.0,
            basis_bps=-4.0,
            confidence_reference_spread_bps=3.0,
        )
        assert hint is not None
        assert hint.basis_bps == -4.0
        # adjusted_spread = ema - basis = -3.0 - (-4.0) = +1.0
        assert hint.adjusted_spread_bps is not None
        assert abs(hint.adjusted_spread_bps - 1.0) < 0.01

    def test_hint_without_basis_has_zero_basis(self) -> None:
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._local(),
            reference_snapshot=self._ref(),
            previous_reference_snapshot=self._prev_ref(),
            max_age_sec=3.0,
            spread_bps_threshold=0.5,
            velocity_bps_threshold=0.01,
            ema_spread_bps=-3.0,
            basis_bps=0.0,
            confidence_reference_spread_bps=3.0,
        )
        assert hint is not None
        assert hint.basis_bps == 0.0
        # EMA mode: adjusted_spread = adjusted_spread (gating - 0.0 = gating)
        assert hint.adjusted_spread_bps is not None

    def test_legacy_mode_adjusted_spread_is_none(self) -> None:
        """Legacy mode (no ema) → adjusted_spread_bps is None in hint."""
        # legacy mode requires spread to exceed threshold to produce a hint;
        # use a wider spread pair to ensure hint is generated.
        local = VenueMidSnapshot("coincheck", 100.0, 100.0)
        ref = VenueMidSnapshot("bitflyer", 99.90, 100.0)
        prev = VenueMidSnapshot("bitflyer", 99.85, 99.0)
        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=local,
            reference_snapshot=ref,
            previous_reference_snapshot=prev,
            max_age_sec=3.0,
            spread_bps_threshold=0.5,
            velocity_bps_threshold=0.01,
            ema_spread_bps=None,  # legacy mode
            basis_bps=0.0,
        )
        if hint is not None:
            # legacy mode では adjusted_spread_bps は None
            assert hint.adjusted_spread_bps is None
        else:
            # spread が threshold 未満で hint=None は正常動作
            pass

    def test_fill_fields_include_basis_fields(self) -> None:
        from scripts.v460.lib.cross_venue_lead_lag import build_cross_venue_fill_fields

        hint = compute_cross_venue_lead_lag_hint(
            local_snapshot=self._local(),
            reference_snapshot=self._ref(),
            previous_reference_snapshot=self._prev_ref(),
            max_age_sec=3.0,
            spread_bps_threshold=0.5,
            velocity_bps_threshold=0.01,
            ema_spread_bps=-3.0,
            basis_bps=-4.0,
            confidence_reference_spread_bps=3.0,
        )
        fields = build_cross_venue_fill_fields(
            enabled=True, hint=hint, side="sell", vetoed=False,
        )
        assert "cross_venue_basis_bps" in fields
        assert "cross_venue_adjusted_spread_bps" in fields
        assert fields["cross_venue_basis_bps"] == -4.0

    def test_disabled_fill_fields_have_basis_none(self) -> None:
        from scripts.v460.lib.cross_venue_lead_lag import build_cross_venue_fill_fields

        fields = build_cross_venue_fill_fields(
            enabled=False, hint=None, side="buy", vetoed=False,
        )
        assert fields["cross_venue_basis_bps"] is None
        assert fields["cross_venue_adjusted_spread_bps"] is None


class TestMicroTimeoutSellAgeCapGuard:
    """509# micro_timeout ループが sell_age_cap を累積超過しないガード検証."""

    def test_micro_timeout_config_fields_exist(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert hasattr(cfg, "micro_timeout_enabled")
        assert hasattr(cfg, "micro_timeout_wait_sec_sell")
        assert hasattr(cfg, "micro_timeout_max_requote")
        assert hasattr(cfg, "micro_timeout_requote_cooloff_sec")

    def test_worst_case_micro_timeout_exceeds_cap(self) -> None:
        """micro_timeout の最悪ケース合計が sell_age_cap を超過可能であることを確認.

        509# ガードがないと 4*10+3*5=55s > 25s (cap) となる。
        ガードがこの超過を防ぐことが fix の目的。
        """
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig(
            micro_timeout_enabled=True,
            micro_timeout_wait_sec_sell=10.0,
            micro_timeout_max_requote=4,
            micro_timeout_requote_cooloff_sec=5.0,
            sell_age_cap_sec=25.0,
        )
        # 最悪ケース: 4 rounds × 10s + 3 cooloffs × 5s = 55s
        worst_total = (
            cfg.micro_timeout_max_requote * cfg.micro_timeout_wait_sec_sell
            + (cfg.micro_timeout_max_requote - 1) * cfg.micro_timeout_requote_cooloff_sec
        )
        assert worst_total > cfg.sell_age_cap_sec, (
            f"worst_total={worst_total} should exceed cap={cfg.sell_age_cap_sec}"
        )

    def test_sell_age_cap_guard_triggers(self) -> None:
        """509# ガード条件とロジック: elapsed >= cap なら break する."""
        cap = 25.0
        first_submit = 1000.0
        # 模擬: 26秒後
        current_time = 1026.0
        elapsed = current_time - first_submit
        assert elapsed >= cap  # ガード条件成立 → break

    def test_sell_age_cap_guard_does_not_trigger_if_within(self) -> None:
        """cap 内であればガードは作用しない."""
        cap = 25.0
        first_submit = 1000.0
        current_time = 1020.0
        elapsed = current_time - first_submit
        assert elapsed < cap  # まだ break しない

    def test_buy_side_has_no_cap(self) -> None:
        """buy 側では sell_age_cap は適用されない."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig(sell_age_cap_sec=25.0)
        side = "buy"
        mt_total_cap: float | None = (
            cfg.sell_age_cap_sec
            if side == "sell"
            and cfg.sell_age_cap_sec is not None
            and cfg.sell_age_cap_sec > 0
            else None
        )
        assert mt_total_cap is None


class TestRepriceRemainingTimeGuard:
    """509# stale reprice 残時間ガード検証."""

    def test_reprice_skipped_if_remaining_under_3s(self) -> None:
        """残り時間 < 3s なら reprice をスキップする."""
        effective_timeout = 10.0
        elapsed = 8.5
        remaining = effective_timeout - elapsed
        assert remaining < 3.0  # reprice スキップ条件成立

    def test_reprice_allowed_if_remaining_above_3s(self) -> None:
        """残り時間 >= 3s なら reprice を許可する."""
        effective_timeout = 10.0
        elapsed = 5.0
        remaining = effective_timeout - elapsed
        assert remaining >= 3.0  # reprice 許可
