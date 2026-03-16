"""Tests for 409# P0-C: reward_profit_corr_median gate condition in G3 judgment.

Verifies that evaluate_g3_checks correctly applies the E6 reward-profit
alignment check introduced per 392# P0-3.
"""

import pytest

from scripts.v460.lib.gate_judgment_core import evaluate_g3_checks


def _make_seed_metrics(
    *,
    pf: float = 1.2,
    sharpe: float = 2.0,
    max_dd: float = 0.05,
    avg_gross: float = 100.0,
    avg_fee: float = 0.0,
    corr: float = 0.5,
) -> dict:
    return {
        "pf": pf,
        "sharpe_annual": sharpe,
        "max_drawdown": max_dd,
        "avg_gross_per_trade": avg_gross,
        "avg_fee_per_trade": avg_fee,
        "reward_profit_corr": corr,
    }


class TestRewardProfitCorrGate:
    """E6: reward_profit_corr_median gate condition."""

    DEFAULT_THRESHOLDS = {
        "min_pf_median": 1.05,
        "min_pf_worst": 0.95,
        "max_drawdown": 0.15,
        "min_sharpe_annual": 0.8,
        "gross_gt_fee": True,
        "min_reward_profit_corr_median": 0.0,
    }

    def test_positive_corr_passes(self):
        """All seeds with positive correlation should pass E6."""
        seeds = [
            _make_seed_metrics(corr=0.5),
            _make_seed_metrics(corr=0.6),
            _make_seed_metrics(corr=0.3),
            _make_seed_metrics(corr=0.4),
        ]
        result = evaluate_g3_checks(seeds, self.DEFAULT_THRESHOLDS)
        assert result["gate_result"] == "PASS"
        assert result["checks"]["reward_profit_corr_median"]["pass"] is True
        assert result["checks"]["reward_profit_corr_median"]["value"] > 0

    def test_negative_median_corr_fails(self):
        """When median correlation is negative, E6 should fail."""
        seeds = [
            _make_seed_metrics(corr=-0.3),
            _make_seed_metrics(corr=-0.1),
            _make_seed_metrics(corr=0.05),
            _make_seed_metrics(corr=-0.5),
        ]
        result = evaluate_g3_checks(seeds, self.DEFAULT_THRESHOLDS)
        assert result["gate_result"] == "FAIL"
        assert result["checks"]["reward_profit_corr_median"]["pass"] is False

    def test_mixed_corr_median_positive_passes(self):
        """One negative seed but median positive should pass."""
        seeds = [
            _make_seed_metrics(corr=0.5),
            _make_seed_metrics(corr=0.6),
            _make_seed_metrics(corr=-0.2),  # outlier
            _make_seed_metrics(corr=0.4),
        ]
        result = evaluate_g3_checks(seeds, self.DEFAULT_THRESHOLDS)
        assert result["checks"]["reward_profit_corr_median"]["pass"] is True

    def test_exact_zero_corr_fails(self):
        """Corr = 0.0 exactly should fail (threshold is > 0, not >=)."""
        seeds = [
            _make_seed_metrics(corr=0.0),
            _make_seed_metrics(corr=0.0),
        ]
        result = evaluate_g3_checks(seeds, self.DEFAULT_THRESHOLDS)
        assert result["checks"]["reward_profit_corr_median"]["pass"] is False

    def test_missing_corr_treated_as_zero(self):
        """Seeds without reward_profit_corr field should default to 0."""
        seeds = [
            {"pf": 1.2, "sharpe_annual": 2.0, "max_drawdown": 0.05,
             "avg_gross_per_trade": 100, "avg_fee_per_trade": 0},
            {"pf": 1.3, "sharpe_annual": 2.5, "max_drawdown": 0.03,
             "avg_gross_per_trade": 100, "avg_fee_per_trade": 0},
        ]
        result = evaluate_g3_checks(seeds, self.DEFAULT_THRESHOLDS)
        # Missing corr → 0 → fails > 0 threshold
        assert result["checks"]["reward_profit_corr_median"]["pass"] is False

    def test_custom_threshold(self):
        """Custom min_reward_profit_corr_median threshold."""
        seeds = [
            _make_seed_metrics(corr=0.15),
            _make_seed_metrics(corr=0.10),
        ]
        strict = {**self.DEFAULT_THRESHOLDS, "min_reward_profit_corr_median": 0.2}
        result = evaluate_g3_checks(seeds, strict)
        assert result["checks"]["reward_profit_corr_median"]["pass"] is False

        lenient = {**self.DEFAULT_THRESHOLDS, "min_reward_profit_corr_median": 0.05}
        result2 = evaluate_g3_checks(seeds, lenient)
        assert result2["checks"]["reward_profit_corr_median"]["pass"] is True

    def test_reward_clean_data_passes(self):
        """Actual reward-clean experiment data should pass G3 with corr gate."""
        seeds = [
            _make_seed_metrics(pf=1.198, sharpe=5.76, max_dd=0.002, corr=0.54),
            _make_seed_metrics(pf=1.116, sharpe=6.00, max_dd=0.002, corr=0.56),
            _make_seed_metrics(pf=1.089, sharpe=3.02, max_dd=0.003, corr=-0.20),
            _make_seed_metrics(pf=1.174, sharpe=5.64, max_dd=0.002, corr=0.61),
        ]
        result = evaluate_g3_checks(seeds, self.DEFAULT_THRESHOLDS)
        assert result["gate_result"] == "PASS"
        # Median of [0.54, 0.56, -0.20, 0.61] = median of sorted [-0.20, 0.54, 0.56, 0.61] = (0.54+0.56)/2 = 0.55
        assert result["checks"]["reward_profit_corr_median"]["value"] == pytest.approx(0.55, abs=0.01)
