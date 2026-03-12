"""
Tests for calibration diagnostics.
"""

import numpy as np
import pytest

from ztb.utils.calibration import (
    compute_brier_score,
    compute_full_calibration_report,
    compute_reliability_curve,
)


def test_perfect_brier_score():
    """Test Brier score with perfect predictions."""
    # Perfect predictions: probability 1.0 for actual action, 0.0 for others
    predicted_probs = np.array(
        [
            [1.0, 0.0, 0.0],  # HOLD
            [0.0, 1.0, 0.0],  # BUY
            [0.0, 0.0, 1.0],  # SELL
        ]
    )
    actual_actions = np.array([0, 1, 2])

    result = compute_brier_score(predicted_probs, actual_actions)

    # Perfect predictions should have Brier score = 0
    assert result["overall"] == pytest.approx(0.0, abs=1e-6)
    assert result["per_action"]["HOLD"] == pytest.approx(0.0, abs=1e-6)
    assert result["per_action"]["BUY"] == pytest.approx(0.0, abs=1e-6)
    assert result["per_action"]["SELL"] == pytest.approx(0.0, abs=1e-6)


def test_worst_brier_score():
    """Test Brier score with worst predictions."""
    # Worst predictions: probability 0.0 for actual action, 1.0 for wrong action
    predicted_probs = np.array(
        [
            [0.0, 1.0, 0.0],  # Predict BUY, actual HOLD
            [0.0, 0.0, 1.0],  # Predict SELL, actual BUY
            [1.0, 0.0, 0.0],  # Predict HOLD, actual SELL
        ]
    )
    actual_actions = np.array([0, 1, 2])

    result = compute_brier_score(predicted_probs, actual_actions)

    # Worst predictions should have Brier score = 2.0
    # (squared error of 1 for correct action + 1 for wrong action = 2)
    assert result["overall"] == pytest.approx(2.0, abs=1e-6)


def test_uniform_brier_score():
    """Test Brier score with uniform predictions."""
    # Uniform predictions: equal probability for all actions
    predicted_probs = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 3, 1 / 3, 1 / 3],
        ]
    )
    actual_actions = np.array([0, 1, 2])

    result = compute_brier_score(predicted_probs, actual_actions)

    # Expected: (2/3)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3
    assert result["overall"] == pytest.approx(2 / 3, abs=1e-6)


def test_reliability_curve_perfect_calibration():
    """Test reliability curve with perfect calibration."""
    # Perfect calibration: predicted probabilities match observed frequencies
    n_samples = 1000
    np.random.seed(42)

    # Generate samples where predicted prob = 0.7 → 70% are action 0
    predicted_probs = np.zeros((n_samples, 3))
    predicted_probs[:, 0] = 0.7  # Predict 70% HOLD
    predicted_probs[:, 1] = 0.2  # Predict 20% BUY
    predicted_probs[:, 2] = 0.1  # Predict 10% SELL

    # Generate actual actions with matching frequencies
    actual_actions = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])

    result = compute_reliability_curve(
        predicted_probs, actual_actions, action_idx=0, n_bins=5
    )

    # ECE should be small for well-calibrated predictions
    assert result["expected_calibration_error"] < 0.1

    # All samples should be in same bin (predicted prob ~0.7)
    # Observed frequency should be close to 0.7
    non_zero_bins = [i for i, count in enumerate(result["bin_counts"]) if count > 0]
    assert len(non_zero_bins) >= 1  # At least one bin has samples

    # Check that observed frequency is close to predicted probability
    for i in non_zero_bins:
        if result["bin_observed_freq"][i] is not None:
            # Allow some variance due to sampling
            assert abs(result["bin_observed_freq"][i] - 0.7) < 0.15


def test_reliability_curve_poor_calibration():
    """Test reliability curve with poor calibration."""
    # Poor calibration: predicted high prob but actual freq is low
    n_samples = 100

    # Predict 90% probability for action 0
    predicted_probs = np.zeros((n_samples, 3))
    predicted_probs[:, 0] = 0.9
    predicted_probs[:, 1] = 0.05
    predicted_probs[:, 2] = 0.05

    # But only 30% are actually action 0
    actual_actions = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.35, 0.35])

    result = compute_reliability_curve(
        predicted_probs, actual_actions, action_idx=0, n_bins=10
    )

    # ECE should be large for poorly calibrated predictions
    # |0.9 - 0.3| = 0.6
    assert result["expected_calibration_error"] > 0.3


def test_full_calibration_report():
    """Test full calibration report."""
    n_samples = 200
    np.random.seed(42)

    # Generate realistic predictions
    predicted_probs = np.random.dirichlet([2, 1, 1], size=n_samples)
    actual_actions = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.25, 0.15])

    result = compute_full_calibration_report(predicted_probs, actual_actions, n_bins=5)

    # Check structure
    assert "brier_score" in result
    assert "reliability_curves" in result
    assert "n_samples" in result
    assert "n_actions" in result

    # Check Brier score
    assert "overall" in result["brier_score"]
    assert "per_action" in result["brier_score"]
    assert 0.0 <= result["brier_score"]["overall"] <= 2.0

    # Check reliability curves for each action
    assert "HOLD" in result["reliability_curves"]
    assert "BUY" in result["reliability_curves"]
    assert "SELL" in result["reliability_curves"]

    for action_name in ["HOLD", "BUY", "SELL"]:
        curve = result["reliability_curves"][action_name]
        assert "bin_edges" in curve
        assert "bin_counts" in curve
        assert "bin_predicted_prob" in curve
        assert "bin_observed_freq" in curve
        assert "expected_calibration_error" in curve

        # ECE should be between 0 and 1
        assert 0.0 <= curve["expected_calibration_error"] <= 1.0

        # Number of bins should match
        assert len(curve["bin_counts"]) == 5

    # Check metadata
    assert result["n_samples"] == n_samples
    assert result["n_actions"] == 3


def test_brier_score_with_no_samples_for_action():
    """Test Brier score when one action never occurs."""
    predicted_probs = np.array(
        [
            [0.8, 0.15, 0.05],
            [0.7, 0.25, 0.05],
            [0.9, 0.08, 0.02],
        ]
    )
    # Only action 0 occurs
    actual_actions = np.array([0, 0, 0])

    result = compute_brier_score(predicted_probs, actual_actions)

    # Should have overall score
    assert "overall" in result

    # HOLD should have a score, BUY and SELL should be None
    assert result["per_action"]["HOLD"] is not None
    assert result["per_action"]["BUY"] is None
    assert result["per_action"]["SELL"] is None


def test_reliability_curve_empty_bins():
    """Test reliability curve with some empty bins."""
    # All predictions in narrow range
    predicted_probs = np.zeros((50, 3))
    predicted_probs[:, 0] = 0.52  # All predictions around 0.52
    predicted_probs[:, 1] = 0.30
    predicted_probs[:, 2] = 0.18

    actual_actions = np.random.choice([0, 1, 2], size=50, p=[0.5, 0.3, 0.2])

    result = compute_reliability_curve(
        predicted_probs, actual_actions, action_idx=0, n_bins=10
    )

    # Most bins should be empty
    non_empty_count = sum(1 for count in result["bin_counts"] if count > 0)
    assert non_empty_count < 5  # Less than half of bins have samples

    # Empty bins should have None values
    for i, count in enumerate(result["bin_counts"]):
        if count == 0:
            assert result["bin_predicted_prob"][i] is None
            assert result["bin_observed_freq"][i] is None
