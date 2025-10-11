"""
Tests for advantage-aware tiebreaker and cost-aware decode gate.
"""

import numpy as np
import pytest

try:
    from ztb.inference.decode import decode_action, InferenceConfig
except ImportError:
    pytest.skip("ztb.inference.decode module not available (torch dependency)", allow_module_level=True)


def test_advantage_tiebreaker_activates():
    """Test advantage-aware tiebreaker activates when top2 has positive advantage."""
    # Setup: HOLD has higher probability but non-positive advantage
    # SELL has lower probability but positive advantage
    logits = np.array([2.0, 0.5, 1.8])  # HOLD > SELL > BUY
    legal_mask = np.array([1, 1, 1])
    advantages = np.array([-0.1, -0.2, 0.5])  # Only SELL has positive advantage
    
    config = InferenceConfig(
        temperature=1.0,
        enable_advantage_tiebreaker=True,
        deterministic=True,
    )
    
    action, info = decode_action(logits, legal_mask, config, advantages=advantages)
    
    # Should select SELL (action 2) via advantage tiebreaker
    assert action == 2
    assert info["tiebreaker_activated"] is True
    assert info["tiebreaker_reason"] == "advantage_sign"


def test_advantage_tiebreaker_not_activated_when_top1_positive():
    """Test advantage tiebreaker doesn't activate when top1 has positive advantage."""
    logits = np.array([2.0, 0.5, 1.8])  # HOLD > SELL > BUY
    legal_mask = np.array([1, 1, 1])
    advantages = np.array([0.5, -0.2, 0.3])  # HOLD has positive advantage
    
    config = InferenceConfig(
        temperature=1.0,
        enable_advantage_tiebreaker=True,
        deterministic=True,
    )
    
    action, info = decode_action(logits, legal_mask, config, advantages=advantages)
    
    # Should select HOLD (action 0) - top1 with positive advantage
    assert action == 0
    assert info["tiebreaker_activated"] is False
    assert info["tiebreaker_reason"] is None


def test_advantage_tiebreaker_disabled():
    """Test that advantage tiebreaker can be disabled."""
    logits = np.array([2.0, 0.5, 1.8])
    legal_mask = np.array([1, 1, 1])
    advantages = np.array([-0.1, -0.2, 0.5])
    
    config = InferenceConfig(
        temperature=1.0,
        enable_advantage_tiebreaker=False,  # Disabled
        deterministic=True,
    )
    
    action, info = decode_action(logits, legal_mask, config, advantages=advantages)
    
    # Should select HOLD (action 0) - standard argmax, no tiebreaker
    assert action == 0
    assert info["tiebreaker_activated"] is False


def test_prob_margin_tiebreaker_still_works():
    """Test that probability-margin tiebreaker still works when advantage not provided."""
    logits = np.array([2.0, 0.5, 1.95])  # HOLD slightly > SELL
    legal_mask = np.array([1, 1, 1])
    
    config = InferenceConfig(
        temperature=1.0,
        tiebreaker_tau=0.10,  # Margin threshold
        enable_tiebreaker=True,
        deterministic=True,
    )
    
    action, info = decode_action(logits, legal_mask, config)
    
    # Margin is small, top1 is HOLD → should trigger prob_margin tiebreaker
    assert action == 2  # SELL (top2)
    assert info["tiebreaker_activated"] is True
    assert info["tiebreaker_reason"] == "prob_margin"


def test_advantage_tiebreaker_priority_over_prob_margin():
    """Test that advantage tiebreaker has priority over probability-margin tiebreaker."""
    logits = np.array([2.0, 0.5, 1.95])  # HOLD slightly > SELL (small margin)
    legal_mask = np.array([1, 1, 1])
    advantages = np.array([-0.1, -0.2, 0.5])  # SELL has positive advantage
    
    config = InferenceConfig(
        temperature=1.0,
        tiebreaker_tau=0.10,  # Would trigger prob_margin
        enable_tiebreaker=True,
        enable_advantage_tiebreaker=True,
        deterministic=True,
    )
    
    action, info = decode_action(logits, legal_mask, config, advantages=advantages)
    
    # Should use advantage_sign tiebreaker (higher priority)
    assert action == 2
    assert info["tiebreaker_activated"] is True
    assert info["tiebreaker_reason"] == "advantage_sign"


def test_cost_gate_blocks_unprofitable_action():
    """Test cost gate blocks action when advantage delta is insufficient."""
    logits = np.array([1.0, 2.5, 1.5])  # BUY > SELL > HOLD
    legal_mask = np.array([1, 1, 1])
    advantages = np.array([0.0, 0.001, -0.1])  # BUY has tiny positive advantage
    current_position = 0  # Currently in HOLD
    
    config = InferenceConfig(
        temperature=1.0,
        enable_cost_gate=True,
        cost_gate_lambda=1.2,
        transaction_cost=0.001,
        slippage=0.0005,
        deterministic=True,
    )
    
    action, info = decode_action(
        logits, legal_mask, config, advantages=advantages, current_position=current_position
    )
    
    # Cost threshold = 1.2 * (0.001 + 0.0005) = 0.0018
    # Advantage delta = 0.001 - 0.0 = 0.001 < 0.0018
    # Should fall back to HOLD
    assert action == 0
    assert info["cost_gate_triggered"] is True
    assert info["estimated_cost"] == pytest.approx(0.0015, abs=1e-6)


def test_cost_gate_allows_profitable_action():
    """Test cost gate allows action when advantage delta exceeds threshold."""
    logits = np.array([1.0, 2.5, 1.5])  # BUY > SELL > HOLD
    legal_mask = np.array([1, 1, 1])
    advantages = np.array([0.0, 0.005, -0.1])  # BUY has sufficient advantage
    current_position = 0  # Currently in HOLD
    
    config = InferenceConfig(
        temperature=1.0,
        enable_cost_gate=True,
        cost_gate_lambda=1.2,
        transaction_cost=0.001,
        slippage=0.0005,
        deterministic=True,
    )
    
    action, info = decode_action(
        logits, legal_mask, config, advantages=advantages, current_position=current_position
    )
    
    # Cost threshold = 1.2 * 0.0015 = 0.0018
    # Advantage delta = 0.005 - 0.0 = 0.005 > 0.0018
    # Should allow BUY
    assert action == 1
    assert info["cost_gate_triggered"] is False


def test_cost_gate_disabled():
    """Test that cost gate can be disabled."""
    logits = np.array([1.0, 2.5, 1.5])
    legal_mask = np.array([1, 1, 1])
    advantages = np.array([0.0, 0.001, -0.1])  # Tiny advantage (would be blocked if enabled)
    current_position = 0
    
    config = InferenceConfig(
        temperature=1.0,
        enable_cost_gate=False,  # Disabled
        deterministic=True,
    )
    
    action, info = decode_action(
        logits, legal_mask, config, advantages=advantages, current_position=current_position
    )
    
    # Should select BUY (top1) without cost gate check
    assert action == 1
    assert info["cost_gate_triggered"] is False


def test_cost_gate_not_applied_to_hold():
    """Test cost gate doesn't apply when selected action is HOLD."""
    logits = np.array([2.5, 1.0, 1.5])  # HOLD > SELL > BUY
    legal_mask = np.array([1, 1, 1])
    advantages = np.array([0.001, -0.1, -0.05])  # HOLD has tiny advantage
    current_position = 1  # Currently in BUY
    
    config = InferenceConfig(
        temperature=1.0,
        enable_cost_gate=True,
        cost_gate_lambda=1.2,
        transaction_cost=0.001,
        slippage=0.0005,
        deterministic=True,
    )
    
    action, info = decode_action(
        logits, legal_mask, config, advantages=advantages, current_position=current_position
    )
    
    # Should select HOLD without cost gate triggering
    assert action == 0
    assert info["cost_gate_triggered"] is False


def test_combined_advantage_and_cost_gate():
    """Test advantage tiebreaker followed by cost gate check."""
    logits = np.array([2.0, 0.5, 1.8])  # HOLD > SELL > BUY
    legal_mask = np.array([1, 1, 1])
    advantages = np.array([-0.1, -0.2, 0.001])  # SELL has small positive advantage
    current_position = 0  # Currently HOLD
    
    config = InferenceConfig(
        temperature=1.0,
        enable_advantage_tiebreaker=True,
        enable_cost_gate=True,
        cost_gate_lambda=1.2,
        transaction_cost=0.001,
        slippage=0.0005,
        deterministic=True,
    )
    
    action, info = decode_action(
        logits, legal_mask, config, advantages=advantages, current_position=current_position
    )
    
    # Advantage tiebreaker selects SELL
    # Then cost gate checks: 0.001 - (-0.1) = 0.101 vs 0.0018 → allows
    # Wait, current position is HOLD (0), advantage is -0.1
    # Selected is SELL (2), advantage is 0.001
    # Delta = 0.001 - (-0.1) = 0.101 > 0.0018 → should allow
    assert action == 2
    assert info["tiebreaker_activated"] is True
    assert info["tiebreaker_reason"] == "advantage_sign"
    assert info["cost_gate_triggered"] is False


def test_cost_gate_overrides_tiebreaker():
    """Test cost gate can override advantage tiebreaker when cost is too high."""
    logits = np.array([2.0, 0.5, 1.8])
    legal_mask = np.array([1, 1, 1])
    advantages = np.array([0.0, -0.2, 0.001])  # SELL has tiny positive advantage
    current_position = 0
    
    config = InferenceConfig(
        temperature=1.0,
        enable_advantage_tiebreaker=True,
        enable_cost_gate=True,
        cost_gate_lambda=1.2,
        transaction_cost=0.001,
        slippage=0.0005,
        deterministic=True,
    )
    
    action, info = decode_action(
        logits, legal_mask, config, advantages=advantages, current_position=current_position
    )
    
    # Advantage tiebreaker would select SELL
    # But cost gate: delta = 0.001 - 0.0 = 0.001 < 0.0018 → blocks
    # Falls back to HOLD
    assert action == 0
    assert info["cost_gate_triggered"] is True
    # Tiebreaker cancelled by cost gate
    assert info["tiebreaker_activated"] is False


def test_batch_with_advantage_and_cost():
    """Test batch processing with advantage tiebreaker and cost gate."""
    logits = np.array([
        [2.0, 0.5, 1.8],  # Obs 1: HOLD > SELL
        [1.0, 2.5, 1.5],  # Obs 2: BUY > SELL
    ])
    legal_masks = np.array([
        [1, 1, 1],
        [1, 1, 1],
    ])
    advantages = np.array([
        [-0.1, -0.2, 0.5],  # Obs 1: SELL has advantage
        [0.0, 0.005, -0.1],  # Obs 2: BUY has advantage
    ])
    current_position = 0
    
    config = InferenceConfig(
        temperature=1.0,
        enable_advantage_tiebreaker=True,
        enable_cost_gate=True,
        cost_gate_lambda=1.2,
        transaction_cost=0.001,
        slippage=0.0005,
        deterministic=True,
    )
    
    actions, info = decode_action(
        logits, legal_masks, config, advantages=advantages, current_position=current_position
    )
    
    # Batch of 2 should return array
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (2,)
    
    # Obs 1: advantage tiebreaker → SELL, cost gate allows (0.5 > 0.0018)
    assert int(actions[0]) == 2
    assert info["tiebreaker_activated"][0] == True
    assert info["cost_gate_triggered"][0] == False
    
    # Obs 2: standard argmax → BUY, cost gate allows (0.005 > 0.0018)
    assert int(actions[1]) == 1
    assert info["tiebreaker_activated"][1] == False
    assert info["cost_gate_triggered"][1] == False


def test_backward_compatibility_no_advantages():
    """Test backward compatibility when advantages not provided."""
    logits = np.array([2.0, 0.5, 1.8])
    legal_mask = np.array([1, 1, 1])
    
    config = InferenceConfig(temperature=1.0, deterministic=True)
    
    action, info = decode_action(logits, legal_mask, config)
    
    # Should work normally without advantages
    assert action == 0  # HOLD (argmax)
    assert "tiebreaker_reason" in info
    assert info["tiebreaker_reason"] is None
    assert "cost_gate_triggered" in info
    assert info["cost_gate_triggered"] is False


def test_backward_compatibility_no_current_position():
    """Test backward compatibility when current_position not provided."""
    logits = np.array([2.0, 0.5, 1.8])
    legal_mask = np.array([1, 1, 1])
    advantages = np.array([-0.1, -0.2, 0.5])
    
    config = InferenceConfig(
        temperature=1.0,
        enable_advantage_tiebreaker=True,
        enable_cost_gate=True,
        deterministic=True,
    )
    
    action, info = decode_action(logits, legal_mask, config, advantages=advantages)
    
    # Advantage tiebreaker should work
    assert action == 2
    assert info["tiebreaker_activated"] is True
    # Cost gate should not trigger without current_position
    assert info["cost_gate_triggered"] is False
