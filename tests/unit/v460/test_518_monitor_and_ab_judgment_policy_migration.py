from __future__ import annotations

from ztb.adaptation.ab_test.judgment_rules import (
    assess_avg_pnl30,
    assess_downside_p10,
    assess_fill_rate,
    combine_assessment_verdicts,
)
from ztb.trading.execution.order_monitor_policy import (
    compute_effective_timeout_policy,
    compute_stale_reprice_policy,
)


class TestOrderMonitorPolicyMigration:
    def test_effective_timeout_policy_applies_sell_cap_and_regime_offset(self) -> None:
        policy = compute_effective_timeout_policy(
            side="sell",
            order_timeout_sec=40.0,
            order_timeout_sec_sell=30.0,
            timeout_override_sec=None,
            timeout_reason=None,
            regime_name="trending_down",
            regime_timeout_multipliers={"trending_down": 1.5},
            regime_reprice_adjustments={"trending_down": 2},
            sell_age_cap_sec=25,
        )
        assert policy.base_timeout == 30.0
        assert policy.timeout_multiplier == 1.5
        assert policy.effective_timeout == 25.0
        assert policy.regime_reprice_offset == 2
        assert policy.sell_age_cap_applied is True

    def test_stale_reprice_policy_applies_side_override_chase_and_clamp(self) -> None:
        policy = compute_stale_reprice_policy(
            side="buy",
            stale_check_after_sec=10.0,
            stale_check_after_sec_buy=6.0,
            stale_check_after_sec_sell=12.0,
            stale_drift_bps=8.0,
            stale_drift_bps_buy=5.0,
            stale_drift_bps_sell=9.0,
            stale_max_reprice=1,
            stale_max_reprice_buy=2,
            stale_max_reprice_sell=3,
            chase_drift_bps_override=3.0,
            chase_max_reprice_override=0,
            regime_reprice_offset=-5,
        )
        assert policy.stale_check_sec == 6.0
        assert policy.stale_drift_bps == 3.0
        assert policy.stale_max_reprice == 0


class TestABJudgmentRulesMigration:
    def test_fill_rate_assessment_uses_absolute_and_relative_failures(self) -> None:
        assessment = assess_fill_rate(
            variant_fill_rate=0.25,
            control_fill_rate=0.50,
            fill_rate_min=0.30,
            fill_rate_degradation_tolerance=0.05,
        )
        assert assessment.name == "fill_rate"
        assert assessment.verdict == "fail"
        assert "below absolute min 30.0%" in assessment.detail
        assert "degraded 50.0% vs control" in assessment.detail

    def test_avg_pnl30_assessment_requires_improvement_when_enabled(self) -> None:
        assessment = assess_avg_pnl30(
            variant_avg_pnl30_bps=-0.2,
            control_avg_pnl30_bps=0.1,
            avg_pnl30_min_bps=-1.0,
            avg_pnl30_must_improve=True,
        )
        assert assessment.verdict == "fail"
        assert "no improvement vs control (+0.1000)" in assessment.detail

    def test_downside_and_combine_assessment(self) -> None:
        downside = assess_downside_p10(
            variant_downside_p10_bps=-6.5,
            control_downside_p10_bps=-3.0,
            downside_p10_min_bps=-5.0,
            downside_p10_degradation_max_bps=2.0,
        )
        fill_rate = assess_fill_rate(
            variant_fill_rate=0.5,
            control_fill_rate=0.5,
            fill_rate_min=0.3,
            fill_rate_degradation_tolerance=0.05,
        )
        assert downside.verdict == "fail"
        assert combine_assessment_verdicts([fill_rate, downside]) == "fail"
