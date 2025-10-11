"""
Tests for action imbalance weights.
"""

import pytest

try:
    from ztb.training.weights import (
        ActionWeightCalculator,
        compute_action_weights,
        cosine_warmup_schedule,
    )
except ImportError:
    pytest.skip("ztb.training.weights module not available", allow_module_level=True)


def test_compute_weights_balanced():
    """Test weight computation with balanced actions."""
    counts = {"HOLD": 100, "BUY": 100, "SELL": 100}
    
    weights = compute_action_weights(counts, beta=3.0)
    
    # All weights should be 1.0 (balanced)
    assert weights["HOLD"] == pytest.approx(1.0, abs=1e-6)
    assert weights["BUY"] == pytest.approx(1.0, abs=1e-6)
    assert weights["SELL"] == pytest.approx(1.0, abs=1e-6)
    
    # Sum should be 3.0
    assert sum(weights.values()) == pytest.approx(3.0, abs=1e-6)


def test_compute_weights_imbalanced():
    """Test weight computation with imbalanced actions."""
    counts = {"HOLD": 100, "BUY": 20, "SELL": 10}
    
    weights = compute_action_weights(counts, beta=3.0)
    
    # HOLD should have lowest weight (most frequent)
    # SELL should have highest weight (least frequent)
    assert weights["HOLD"] < weights["BUY"]
    assert weights["BUY"] < weights["SELL"] or weights["BUY"] == pytest.approx(weights["SELL"], abs=1e-6)
    
    # Sum should be 3.0
    assert sum(weights.values()) == pytest.approx(3.0, abs=1e-6)
    
    # Check approximate values (from dry-run test)
    assert weights["HOLD"] == pytest.approx(0.534, abs=0.01)
    assert weights["BUY"] == pytest.approx(1.233, abs=0.01)
    assert weights["SELL"] == pytest.approx(1.233, abs=0.01)


def test_compute_weights_beta_clipping():
    """Test that beta clipping prevents extreme weights."""
    # Very imbalanced: HOLD dominates
    counts = {"HOLD": 1000, "BUY": 10, "SELL": 1}
    
    weights_beta3 = compute_action_weights(counts, beta=3.0)
    weights_beta5 = compute_action_weights(counts, beta=5.0)
    
    # Beta=3 should clip more aggressively
    assert max(weights_beta3.values()) <= 3.0 + 1e-6
    assert max(weights_beta5.values()) <= 5.0 + 1e-6
    
    # Higher beta allows higher max weight
    assert max(weights_beta5.values()) >= max(weights_beta3.values())


def test_compute_weights_zero_counts():
    """Test handling of zero total counts."""
    counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
    
    weights = compute_action_weights(counts)
    
    # Should return uniform weights
    assert weights["HOLD"] == 1.0
    assert weights["BUY"] == 1.0
    assert weights["SELL"] == 1.0


def test_action_weight_calculator_ema():
    """Test EMA smoothing in ActionWeightCalculator."""
    calc = ActionWeightCalculator(beta=3.0, ema_alpha=0.1)
    
    # First update: no smoothing
    counts1 = {"HOLD": 100, "BUY": 20, "SELL": 10}
    weights1 = calc.compute_weights(counts1, apply_ema=True)
    
    # Second update: should smooth with previous
    counts2 = {"HOLD": 50, "BUY": 50, "SELL": 50}
    weights2 = calc.compute_weights(counts2, apply_ema=True)
    
    # Weights should be different due to EMA
    assert weights1 != weights2
    
    # Third update: should continue smoothing
    weights3 = calc.compute_weights(counts2, apply_ema=True)
    
    # Should move closer to balanced (1.0, 1.0, 1.0)
    # but not reach it immediately due to EMA
    assert abs(weights3["HOLD"] - 1.0) < abs(weights2["HOLD"] - 1.0)


def test_safety_guards_entropy():
    """Test entropy safety guard."""
    calc = ActionWeightCalculator(entropy_min=0.05)
    
    # Low entropy should trigger guard
    should_revert, reason = calc.check_safety_guards(entropy=0.03, kl_violations_rate=0.0)
    
    assert should_revert is True
    assert "entropy" in reason.lower()
    assert not calc._weights_active


def test_safety_guards_kl_violations():
    """Test KL divergence safety guard."""
    calc = ActionWeightCalculator(target_kl_max=0.03, kl_consecutive_max=3)
    
    # First violation: should not revert
    should_revert1, _ = calc.check_safety_guards(entropy=0.5, kl_violations_rate=0.05)
    assert should_revert1 is False
    assert calc._kl_consecutive_violations == 1
    
    # Second violation: still not revert
    should_revert2, _ = calc.check_safety_guards(entropy=0.5, kl_violations_rate=0.05)
    assert should_revert2 is False
    assert calc._kl_consecutive_violations == 2
    
    # Third violation: should revert
    should_revert3, reason = calc.check_safety_guards(entropy=0.5, kl_violations_rate=0.05)
    assert should_revert3 is True
    assert "kl" in reason.lower()
    assert not calc._weights_active


def test_safety_guards_kl_reset():
    """Test KL violation counter resets on success."""
    calc = ActionWeightCalculator(target_kl_max=0.03, kl_consecutive_max=3)
    
    # Two violations
    calc.check_safety_guards(entropy=0.5, kl_violations_rate=0.05)
    calc.check_safety_guards(entropy=0.5, kl_violations_rate=0.05)
    assert calc._kl_consecutive_violations == 2
    
    # Success: should reset counter
    calc.check_safety_guards(entropy=0.5, kl_violations_rate=0.01)
    assert calc._kl_consecutive_violations == 0


def test_get_safe_weights_normal():
    """Test get_safe_weights with normal conditions."""
    calc = ActionWeightCalculator()
    
    counts = {"HOLD": 100, "BUY": 20, "SELL": 10}
    weights, guard_triggered, reason = calc.get_safe_weights(
        counts,
        entropy=0.5,
        kl_violations_rate=0.01,
        apply_ema=False,
    )
    
    # Should return normal weights
    assert guard_triggered is False
    assert reason == ""
    assert sum(weights.values()) == pytest.approx(3.0, abs=1e-6)


def test_get_safe_weights_guard_triggered():
    """Test get_safe_weights when guard is triggered."""
    calc = ActionWeightCalculator(entropy_min=0.05)
    
    counts = {"HOLD": 100, "BUY": 20, "SELL": 10}
    weights, guard_triggered, reason = calc.get_safe_weights(
        counts,
        entropy=0.03,  # Too low
        kl_violations_rate=0.01,
    )
    
    # Should return uniform weights
    assert guard_triggered is True
    assert "entropy" in reason.lower()
    assert weights["HOLD"] == 1.0
    assert weights["BUY"] == 1.0
    assert weights["SELL"] == 1.0


def test_reset_guards():
    """Test resetting safety guards."""
    calc = ActionWeightCalculator(entropy_min=0.05)
    
    # Trigger guard
    calc.check_safety_guards(entropy=0.03, kl_violations_rate=0.0)
    assert not calc._weights_active
    
    # Reset
    calc.reset_guards()
    assert calc._weights_active
    assert calc._kl_consecutive_violations == 0


def test_cosine_warmup_before_start():
    """Test warmup schedule before warmup start."""
    multiplier = cosine_warmup_schedule(current_step=1000, warmup_start=5000)
    
    assert multiplier == 0.0


def test_cosine_warmup_after_end():
    """Test warmup schedule after warmup end."""
    multiplier = cosine_warmup_schedule(current_step=20000, warmup_end=15000)
    
    assert multiplier == 1.0


def test_cosine_warmup_midpoint():
    """Test warmup schedule at midpoint."""
    # At midpoint (10000 between 5000 and 15000)
    multiplier = cosine_warmup_schedule(current_step=10000, warmup_start=5000, warmup_end=15000)
    
    # Should be 0.5 (cosine midpoint)
    assert multiplier == pytest.approx(0.5, abs=1e-6)


def test_cosine_warmup_monotonic():
    """Test warmup schedule is monotonically increasing."""
    steps = [5000, 7500, 10000, 12500, 15000]
    multipliers = [
        cosine_warmup_schedule(step, warmup_start=5000, warmup_end=15000)
        for step in steps
    ]
    
    # Should be strictly increasing
    for i in range(len(multipliers) - 1):
        assert multipliers[i] < multipliers[i + 1]
    
    # First should be 0.0, last should be 1.0
    assert multipliers[0] == 0.0
    assert multipliers[-1] == 1.0
