#!/usr/bin/env python3
"""Quick verification that reward_components are populated."""

import sys
from unittest.mock import Mock, patch
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.constants import ACTION_BUY


@patch('ztb.trading.environment.components.reward_calculator.BehavioralPenaltyCalculator')
@patch('ztb.trading.environment.components.reward_calculator.AsymmetricRewardScaler')
@patch('ztb.trading.environment.components.reward_calculator.DynamicRewardShaper')
@patch('ztb.trading.environment.components.reward_calculator.SignalIntegrator')
@patch('ztb.trading.environment.components.reward_calculator.OpportunityCostPenaltyCalculator')
@patch('ztb.trading.environment.components.reward_calculator.UnrealizedLossPenaltyCalculator')
def test_quick(
    mock_unrealized_loss,
    mock_opportunity_cost,
    mock_signal_integrator,
    mock_dynamic_shaper,
    mock_asymmetric_scaler,
    mock_behavioral_penalty,
):
    """Quick test of reward_components."""
    # Setup mocks
    mock_behavioral_penalty.return_value.record_action = Mock()
    mock_behavioral_penalty.return_value._get_recent_counts = Mock(return_value=[0, 0, 0])
    mock_asymmetric_scaler.return_value.scale_reward = lambda r, p, pnl: r
    mock_dynamic_shaper.return_value.shape_reward = lambda r, p, s, pnl: r
    mock_signal_integrator.return_value.enabled = False
    
    reward_settings = {"use_simple_reward": True}
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
    
    # Test BUY action
    reward = calculator.calculate_reward_simple(
        pnl=100.0,
        portfolio_value=101000.0,
        position=0.5,
        old_position=0.3,
        action=ACTION_BUY,
        reward_history=[],
        portfolio_value_history=[100000.0],
        current_price=5000000.0,
        step=1,
        transaction_cost=10.0,
    )
    
    components = calculator.get_last_reward_components()
    
    print("\n✅ REWARD COMPONENTS FIX VERIFICATION")
    print("=" * 50)
    print(f"Reward calculated: {reward:.4f}")
    print(f"\nComponents found: {len(components)} keys")
    print(f"Stage: {components.get('stage')}")
    print(f"PnL: {components.get('pnl')}")
    print(f"Final reward: {components.get('final_reward')}")
    print(f"Trade bonus applied: {components.get('trade_bonus_applied')}")
    print(f"Hold penalty applied: {components.get('hold_penalty_applied')}")
    
    # Verify critical fields
    assert components.get('stage') == 'simple_reward', "Stage should be 'simple_reward'"
    assert 'pnl' in components, "PnL should be present"
    assert 'final_reward' in components, "Final reward should be present"
    assert components.get('trade_bonus_applied') == True, "Trade bonus should be applied for BUY"
    assert components.get('hold_penalty_applied') == False, "Hold penalty should NOT be applied for BUY"
    
    print("\n✅ ALL CHECKS PASSED!")
    print("reward_components fix is working correctly.")
    return True


if __name__ == "__main__":
    try:
        success = test_quick()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
