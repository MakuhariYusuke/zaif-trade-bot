#!/usr/bin/env python3
"""Unit test to verify reward_components fix in calculate_reward_simple."""

import pytest
from unittest.mock import Mock, patch
from ztb.trading.environment.components.calculators.reward_calculator import RewardCalculator
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD


@patch('ztb.trading.environment.components.calculators.reward_calculator.BehavioralPenaltyCalculator')
@patch('ztb.trading.environment.components.calculators.reward_calculator.AsymmetricRewardScaler')
@patch('ztb.trading.environment.components.calculators.reward_calculator.DynamicRewardShaper')
@patch('ztb.trading.environment.components.calculators.reward_calculator.SignalIntegrator')
@patch('ztb.trading.environment.components.calculators.reward_calculator.OpportunityCostPenaltyCalculator')
@patch('ztb.trading.environment.components.calculators.reward_calculator.UnrealizedLossPenaltyCalculator')
def test_reward_components_populated_in_simple_reward(
    mock_unrealized_loss,
    mock_opportunity_cost,
    mock_signal_integrator,
    mock_dynamic_shaper,
    mock_asymmetric_scaler,
    mock_behavioral_penalty,
):
    """Test that _last_reward_components is populated when using simple reward."""
    # Setup mocks
    mock_behavioral_penalty.return_value.record_action = Mock()
    mock_behavioral_penalty.return_value._get_recent_counts = Mock(return_value=[0, 0, 0])
    
    mock_asymmetric_scaler.return_value.scale_reward = lambda r, p, pnl: r
    mock_dynamic_shaper.return_value.shape_reward = lambda r, p, s, pnl: r
    mock_signal_integrator.return_value.enabled = False
    
    # Create reward settings with use_simple_reward enabled
    reward_settings = {
        "use_simple_reward": True,
        "reward_scaling": 1.0,
        "reward_clip_value": 10.0,
        "hold_penalty_multiplier": 0.9,
        "trade_frequency_bonus": 0.01,
    }

    # Create mock config
    mock_config = Mock()
    mock_config.curriculum_stage = "simple"
    mock_config.max_position_size = 1.0
    mock_config.reward_settings = reward_settings
    mock_config.venue_settings = {}

    # Initialize RewardCalculator
    calculator = RewardCalculator(
        config=mock_config,
        reward_settings=reward_settings,
        initial_portfolio_value=100000.0,
    )    # Test BUY action with profit
    pnl = 100.0
    portfolio_value = 101000.0
    position = 0.5
    old_position = 0.3
    action = ACTION_BUY
    reward_history = []
    portfolio_value_history = [100000.0]
    current_price = 5000000.0
    step = 1
    transaction_cost = 10.0
    
    reward = calculator.calculate_reward_simple(
        pnl=pnl,
        portfolio_value=portfolio_value,
        position=position,
        old_position=old_position,
        action=action,
        reward_history=reward_history,
        portfolio_value_history=portfolio_value_history,
        current_price=current_price,
        step=step,
        transaction_cost=transaction_cost,
    )
    
    # Verify reward is calculated
    assert isinstance(reward, float)
    assert reward != 0.0
    
    # Verify _last_reward_components is populated
    components = calculator.get_last_reward_components()
    
    assert components is not None, "reward_components should not be None"
    assert len(components) > 0, "reward_components should not be empty"
    
    # Check expected keys
    assert "stage" in components
    assert components["stage"] == "simple_reward"
    
    assert "pnl" in components
    assert components["pnl"] == pnl
    
    assert "final_reward" in components
    assert components["final_reward"] == reward
    
    assert "trade_bonus_applied" in components
    assert components["trade_bonus_applied"] == True  # BUY action
    
    assert "hold_penalty_applied" in components
    assert components["hold_penalty_applied"] == False  # Not HOLD


@patch('ztb.trading.environment.components.calculators.reward_calculator.BehavioralPenaltyCalculator')
@patch('ztb.trading.environment.components.calculators.reward_calculator.AsymmetricRewardScaler')
@patch('ztb.trading.environment.components.calculators.reward_calculator.DynamicRewardShaper')
@patch('ztb.trading.environment.components.calculators.reward_calculator.SignalIntegrator')
@patch('ztb.trading.environment.components.calculators.reward_calculator.OpportunityCostPenaltyCalculator')
@patch('ztb.trading.environment.components.calculators.reward_calculator.UnrealizedLossPenaltyCalculator')
def test_reward_components_with_hold_action(
    mock_unrealized_loss,
    mock_opportunity_cost,
    mock_signal_integrator,
    mock_dynamic_shaper,
    mock_asymmetric_scaler,
    mock_behavioral_penalty,
):
    """Test that reward_components correctly identifies HOLD action."""
    # Setup mocks
    mock_behavioral_penalty.return_value.record_action = Mock()
    mock_behavioral_penalty.return_value._get_recent_counts = Mock(return_value=[0, 0, 0])
    
    mock_asymmetric_scaler.return_value.scale_reward = lambda r, p, pnl: r
    mock_dynamic_shaper.return_value.shape_reward = lambda r, p, s, pnl: r
    mock_signal_integrator.return_value.enabled = False
    
    reward_settings = {
        "use_simple_reward": True,
        "reward_scaling": 1.0,
        "reward_clip_value": 10.0,
        "hold_penalty_multiplier": 0.9,
    }

    mock_config = Mock()
    mock_config.curriculum_stage = "simple"
    mock_config.max_position_size = 1.0
    mock_config.reward_settings = reward_settings
    mock_config.venue_settings = {}

    calculator = RewardCalculator(
        config=mock_config,
        reward_settings=reward_settings,
        initial_portfolio_value=100000.0,
    )    # Test HOLD action
    reward = calculator.calculate_reward_simple(
        pnl=0.0,
        portfolio_value=100000.0,
        position=0.5,
        old_position=0.5,
        action=ACTION_HOLD,
        reward_history=[],
        portfolio_value_history=[100000.0],
        current_price=5000000.0,
        step=1,
        transaction_cost=0.0,
    )
    
    components = calculator.get_last_reward_components()
    
    assert components["hold_penalty_applied"] == True
    assert components["trade_bonus_applied"] == False


@patch('ztb.trading.environment.components.calculators.reward_calculator.BehavioralPenaltyCalculator')
@patch('ztb.trading.environment.components.calculators.reward_calculator.AsymmetricRewardScaler')
@patch('ztb.trading.environment.components.calculators.reward_calculator.DynamicRewardShaper')
@patch('ztb.trading.environment.components.calculators.reward_calculator.SignalIntegrator')
@patch('ztb.trading.environment.components.calculators.reward_calculator.OpportunityCostPenaltyCalculator')
@patch('ztb.trading.environment.components.calculators.reward_calculator.UnrealizedLossPenaltyCalculator')
def test_reward_components_with_exception(
    mock_unrealized_loss,
    mock_opportunity_cost,
    mock_signal_integrator,
    mock_dynamic_shaper,
    mock_asymmetric_scaler,
    mock_behavioral_penalty,
):
    """Test that reward_components is populated even when exception occurs."""
    # Setup mocks
    mock_behavioral_penalty.return_value.record_action = Mock()
    mock_behavioral_penalty.return_value._get_recent_counts = Mock(return_value=[0, 0, 0])
    
    # Mock dynamic_shaper to raise exception
    mock_asymmetric_scaler.return_value.scale_reward = lambda r, p, pnl: r
    mock_dynamic_shaper.return_value.shape_reward = Mock(side_effect=Exception("Test error"))
    mock_signal_integrator.return_value.enabled = False
    
    reward_settings = {
        "use_simple_reward": True,
    }

    mock_config = Mock()
    mock_config.curriculum_stage = "simple"
    mock_config.max_position_size = 1.0
    mock_config.reward_settings = reward_settings
    mock_config.venue_settings = {}

    calculator = RewardCalculator(
        config=mock_config,
        reward_settings=reward_settings,
        initial_portfolio_value=100000.0,
    )    # Call with valid inputs but expect exception handling
    reward = calculator.calculate_reward_simple(
        pnl=100.0,
        portfolio_value=100000.0,
        position=0.5,
        old_position=0.3,
        action=ACTION_BUY,
        reward_history=[],
        portfolio_value_history=[100000.0],
        current_price=5000000.0,
        step=1,
        transaction_cost=0.0,
    )
    
    # Should return 0.0 on exception
    assert reward == 0.0
    
    # But should still have reward_components with error info
    components = calculator.get_last_reward_components()
    
    assert components is not None
    assert "stage" in components
    assert components["stage"] == "simple_reward_error"
    assert "error" in components


@patch('ztb.trading.environment.components.calculators.reward_calculator.BehavioralPenaltyCalculator')
@patch('ztb.trading.environment.components.calculators.reward_calculator.AsymmetricRewardScaler')
@patch('ztb.trading.environment.components.calculators.reward_calculator.DynamicRewardShaper')
@patch('ztb.trading.environment.components.calculators.reward_calculator.SignalIntegrator')
@patch('ztb.trading.environment.components.calculators.reward_calculator.OpportunityCostPenaltyCalculator')
@patch('ztb.trading.environment.components.calculators.reward_calculator.UnrealizedLossPenaltyCalculator')
def test_reward_components_simple_reward_snapshot_is_detached(
    mock_unrealized_loss,
    mock_opportunity_cost,
    mock_signal_integrator,
    mock_dynamic_shaper,
    mock_asymmetric_scaler,
    mock_behavioral_penalty,
):
    """Returned simple_reward payload should be a detached snapshot."""
    mock_behavioral_penalty.return_value.record_action = Mock()
    mock_behavioral_penalty.return_value._get_recent_counts = Mock(return_value=[0, 0, 0])
    mock_asymmetric_scaler.return_value.scale_reward = lambda r, p, pnl: r
    mock_dynamic_shaper.return_value.shape_reward = lambda r, p, s, pnl: r
    mock_signal_integrator.return_value.enabled = False

    reward_settings = {
        "use_simple_reward": True,
        "reward_scaling": 1.0,
        "trade_frequency_bonus": 0.01,
    }
    mock_config = Mock()
    mock_config.curriculum_stage = "simple"
    mock_config.max_position_size = 1.0
    mock_config.reward_settings = reward_settings
    mock_config.venue_settings = {}

    calculator = RewardCalculator(
        config=mock_config,
        reward_settings=reward_settings,
        initial_portfolio_value=100000.0,
    )
    calculator.calculate_reward_simple(
        pnl=100.0,
        portfolio_value=101000.0,
        position=0.5,
        old_position=0.3,
        action=ACTION_BUY,
        reward_history=[],
        portfolio_value_history=[100000.0],
        current_price=5000000.0,
        step=1,
        transaction_cost=0.0,
    )

    snapshot = calculator.get_last_reward_components()
    snapshot["final_reward"] = -999.0
    snapshot["trade_bonus_applied"] = 0.0

    assert calculator._last_reward_components["final_reward"] != -999.0
    assert calculator._last_reward_components["trade_bonus_applied"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
