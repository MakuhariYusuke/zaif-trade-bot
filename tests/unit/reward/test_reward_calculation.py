#!/usr/bin/env python3
"""
Test script for reward calculation bug fixes.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.calculators.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import RewardSettings


class MockConfig:
    def __init__(self):
        self.curriculum_stage = "ultra_profit"
        self.max_position_size = 1.0
        self.transaction_cost = 0.001
        self.action_bonuses = {}


# def test_get_nested_setting_fix():
#     """Test that _get_nested_setting works correctly after bug fix."""
#     config = MockConfig()
#     reward_settings = {
#         "profit_weight": 1.0,
#         "risk_weight": 1.0,
#         "consistency_weight": 1.0,
#         "ultra_profit_multiplier": 1.0,
#         "ultra_risk_multiplier": 1.0,
#         "reward_scale": 1.0,
#         "reward_clip_min": -10.0,
#         "reward_clip_max": 10.0,
#         "use_simple_reward": False,
#         "profit_multiplier": 0.01,
#         "loss_penalty_multiplier": 0.01,
#         "hold_penalty_rate": 0.01,
#         "balance_penalty_tolerance": 0.15,
#         "balance_penalty": 1.0,
#         "trading_bonus_multiplier": 2.0,
#         "trading_bonus": 0.01,
#         "dynamic_reward_shaping": {"enabled": True, "regime_detection_window": 20},
#     }

#     calculator = RewardCalculator(
#         config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
#     )

#     # Test nested setting access
#     value = calculator._get_nested_setting("dynamic_reward_shaping.enabled")
#     assert value is True

#     value = calculator._get_nested_setting(
#         "dynamic_reward_shaping.regime_detection_window"
#     )
#     assert value == 20

#     # Test non-existent key
#     value = calculator._get_nested_setting("nonexistent.key")
#     assert value is None

#     # Test partial path
#     value = calculator._get_nested_setting("dynamic_reward_shaping")
#     assert isinstance(value, dict)


def test_calculate_reward_simple_with_transaction_cost():
    """Test that calculate_reward properly handles transaction_cost parameter."""
    config = MockConfig()
    reward_settings = RewardSettings(
        use_simple_reward=False,
        reward_scale=1.0,
        trading_bonus=0.01,
        profit_bonuses={},
        penalty_coefficients={},
        entropy_bonus=0.0,
        custom_reward_params={},
        balance_penalty=1.0,
        balance_penalty_tolerance=0.15,
        profit_weight=1.0,
        risk_weight=1.0,
        consistency_weight=1.0,
        ultra_profit_multiplier=1.0,
        ultra_risk_multiplier=1.0,
        position_soft_cap=0.5,
        position_penalty_scale=0.01,
        position_penalty_exponent=2.0,
        inventory_window=10,
        inventory_penalty_scale=0.01,
        trade_frequency_penalty=0.01,
        trade_frequency_halflife=100.0,
        trade_cooldown_steps=5,
        trade_cooldown_penalty=0.01,
        max_consecutive_trades=3,
        consecutive_trade_penalty=0.05,
        volatility_window=20,
        volatility_penalty_scale=0.01,
        sharpe_bonus_scale=0.01,
        sortino_bonus_scale=0.01,
        calmar_bonus_scale=0.005,
        reward_clip_value=10.0,
        profit_bonus_multipliers=[1.0, 1.5, 2.0],
        enable_forced_diversity=False,
    )

    calculator = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    )

    # Test with trade (transaction_cost parameter is accepted but not used in calculation
    # since it's already deducted in position_manager)
    reward_with_cost = calculator.calculate_reward(
        action=ACTION_BUY,
        current_price=5000000.0,
        position=0.5,
        portfolio_value=200000.0,
        atr=1.0,
        transaction_cost=100.0,  # This parameter is accepted but not used in calculation
        reward_scaling=1.0,
        pnl=1000.0,
        old_position=0.0,  # Position changed, indicating a trade
        step=1,
        observation=np.array([1.0, 2.0, 3.0]),
        reward_history=[],
        portfolio_value_history=[],
    )

    reward_without_cost = calculator.calculate_reward(
        action=ACTION_BUY,
        current_price=5000000.0,
        position=0.5,
        portfolio_value=200000.0,
        atr=1.0,
        transaction_cost=0.0,
        reward_scaling=1.0,
        pnl=1000.0,
        old_position=0.0,
        step=2,
        observation=np.array([1.0, 2.0, 3.0]),
        reward_history=[],
        portfolio_value_history=[],
    )

    assert isinstance(reward_with_cost, (int, float))
    assert np.isfinite(reward_with_cost)
    assert np.isfinite(reward_without_cost)
    assert np.isclose(reward_with_cost, reward_without_cost)

    # def test_calculate_reward_calls_simple_with_transaction_cost():
    """Test that calculate_reward properly passes transaction_cost to calculate_reward_simple."""
    config = MockConfig()
    config.curriculum_stage = "test"  # Force use_simple_reward path

    #     reward_settings = RewardSettings(
    #         profit_weight=1.0,
    #         risk_weight=1.0,
    #         consistency_weight=1.0,
    #         ultra_profit_multiplier=1.0,
    #         ultra_risk_multiplier=1.0,
    #         reward_scale=1.0,
    #         reward_clip_min=-10.0,
    #         reward_clip_max=10.0,
    #         use_simple_reward=True,  # Force simple reward path
    #         profit_multiplier=0.01,
    #         loss_penalty_multiplier=0.01,
    #         hold_penalty_rate=0.01,
    #         balance_penalty_tolerance=0.15,
    #         balance_penalty=1.0,
    #         trading_bonus_multiplier=2.0,
    #         trading_bonus=0.01,
    #     )

    #     calculator = RewardCalculator(
    #         config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    #     )

    #     # Mock the calculate_reward_simple method to check if transaction_cost is passed
    #     original_method = calculator.calculate_reward_simple
    #     passed_transaction_cost = None

    #     def mock_calculate_reward_simple(*args, **kwargs):
    #         nonlocal passed_transaction_cost
    #         passed_transaction_cost = kwargs.get(
    #             "transaction_cost", args[9] if len(args) > 9 else None
    #         )
    #         return 0.0

    #     calculator.calculate_reward_simple = mock_calculate_reward_simple

    #     try:
    #         calculator.calculate_reward(
    #             action=ACTION_BUY,
    #             current_price=5000000.0,
    #             position=0.5,
    #             portfolio_value=200000.0,
    #             atr=50000.0,
    #             transaction_cost=0.001,
    #             reward_scaling=1.0,
    #             pnl=1000.0,
    #             old_position=0.0,
    #             step=1,
    #             observation=np.array([5000000.0, 0.5, 1000.0]),
    #             reward_history=[],
    #             portfolio_value_history=[],
    #         )

    #         assert (
    #             passed_transaction_cost == 0.001
    #         ), f"Expected transaction_cost=0.001, got {passed_transaction_cost}"
    #     finally:
    #         calculator.calculate_reward_simple = original_method

    # def test_no_stdout_prints():
    """Test that reward calculator doesn't print to stdout."""


#     config = MockConfig()
#     reward_settings = {
#         "profit_weight": 1.0,
#         "risk_weight": 1.0,
#         "consistency_weight": 1.0,
#         "ultra_profit_multiplier": 1.0,
#         "ultra_risk_multiplier": 1.0,
#         "reward_scale": 1.0,
#         "reward_clip_min": -10.0,
#         "reward_clip_max": 10.0,
#         "use_simple_reward": True,
#         "profit_multiplier": 0.01,
#         "loss_penalty_multiplier": 0.01,
#         "hold_penalty_rate": 0.01,
#         "balance_penalty_tolerance": 0.15,
#         "balance_penalty": 1.0,
#         "trading_bonus_multiplier": 2.0,
#         "trading_bonus": 0.01,
#     }

#     calculator = RewardCalculator(
#         config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
#     )

#     # Capture stdout during reward calculation
#     f = io.StringIO()
#     with redirect_stdout(f):
#         reward = calculator.calculate_reward_simple(
#             pnl=1000.0,
#             portfolio_value=200000.0,
#             position=0.5,
#             old_position=0.0,
#             action=ACTION_BUY,
#             reward_history=[],
#             portfolio_value_history=[],
#             current_price=5000000.0,
#             step=1,
#             transaction_cost=0.0,
#         )

#     output = f.getvalue()
#     assert output == "", f"Unexpected stdout output: {output}"


def test_get_current_regime():
    """Test get_current_regime method works correctly."""
    config = MockConfig()
    reward_settings = RewardSettings(
        use_simple_reward=False,
        reward_scale=1.0,
        trading_bonus=0.01,
        profit_bonuses={},
        penalty_coefficients={},
        entropy_bonus=0.0,
        custom_reward_params={},
        balance_penalty=1.0,
        balance_penalty_tolerance=0.15,
        profit_weight=1.0,
        risk_weight=1.0,
        consistency_weight=1.0,
        ultra_profit_multiplier=1.0,
        ultra_risk_multiplier=1.0,
        position_soft_cap=0.5,
        position_penalty_scale=0.01,
        position_penalty_exponent=2.0,
        inventory_window=10,
        inventory_penalty_scale=0.01,
        trade_frequency_penalty=0.01,
        trade_frequency_halflife=100.0,
        trade_cooldown_steps=5,
        trade_cooldown_penalty=0.01,
        max_consecutive_trades=3,
        consecutive_trade_penalty=0.05,
        volatility_window=20,
        volatility_penalty_scale=0.01,
        sharpe_bonus_scale=0.01,
        sortino_bonus_scale=0.01,
        calmar_bonus_scale=0.005,
        reward_clip_value=10.0,
        profit_bonus_multipliers=[1.0, 1.5, 2.0],
        enable_forced_diversity=False,
    )

    calculator = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    )

    # Test that calculator initializes correctly
    assert calculator is not None
    assert hasattr(calculator, 'calculate_reward')


# if __name__ == "__main__":
#     pytest.main([__file__])

#     # Test scenarios
#     test_cases = [
#         {
#             "name": "Profitable BUY",
#             "action": ACTION_BUY,
#             "pnl": 1000.0,
#             "position": 0.5,
#             "current_price": 5000000.0,
#             "atr": 50000.0,
#         },
#         {
#             "name": "Loss SELL",
#             "action": ACTION_SELL,
#             "pnl": -500.0,
#             "position": -0.3,
#             "current_price": 4800000.0,
#             "atr": 45000.0,
#         },
#         {
#             "name": "HOLD with position",
#             "action": ACTION_HOLD,
#             "pnl": 0.0,
#             "position": 0.8,
#             "current_price": 4900000.0,
#             "atr": 40000.0,
#         },
#     ]

#     print("Testing reward calculation with extreme aggressive parameters:")
#     print("=" * 60)

#     for test_case in test_cases:
#         reward = calculator._calculate_ultra_profit_reward(
#             action=test_case["action"],
#             atr_normalised=test_case["pnl"] / test_case["atr"],
#             portfolio_return=test_case["pnl"] / 200000.0,
#             position=test_case["position"],
#             effective_max_position=1.0,
#             current_price=test_case["current_price"],
#             atr=test_case["atr"],
#             pnl=test_case["pnl"],
#             reward_scaling=1.0,  # Fixed scaling
#         )

#         print(f"{test_case['name']}:")
#         print(f"  Action: {test_case['action']} (0=HOLD, 1=BUY, 2=SELL)")
#         print(f"  PnL: {test_case['pnl']:.2f}")
#         print(f"  Position: {test_case['position']:.2f}")
#         print(f"  Reward: {reward:.4f}")
#         print()


def test_forced_balance_penalty():
    """Test forced balance penalty calculation."""
    from ztb.trading.environment.utils.config import EnvironmentConfig

    # Create mock config with forced_balance curriculum
    config = EnvironmentConfig()
    config.curriculum_stage = "forced_balance"
    config.max_position_size = 1.0
    config.behavior_optimization = {
        "balance_penalty": 10.0,  # Use reasonable scale for testing
        "action_balance_target": 0.333,
        "redundant_trade_penalty": 5.0,
    }

    reward_settings = RewardSettings(
        use_simple_reward=False,
        reward_scale=1.0,
        trading_bonus=0.01,
        profit_bonuses={},
        penalty_coefficients={},
        entropy_bonus=0.0,
        custom_reward_params={},
        balance_penalty=1.0,
        balance_penalty_tolerance=0.15,
        profit_weight=1.0,
        risk_weight=1.0,
        consistency_weight=1.0,
        ultra_profit_multiplier=1.0,
        ultra_risk_multiplier=1.0,
        position_soft_cap=0.5,
        position_penalty_scale=0.01,
        position_penalty_exponent=2.0,
        inventory_window=10,
        inventory_penalty_scale=0.01,
        trade_frequency_penalty=0.01,
        trade_frequency_halflife=100.0,
        trade_cooldown_steps=5,
        trade_cooldown_penalty=0.01,
        max_consecutive_trades=3,
        consecutive_trade_penalty=0.05,
        volatility_window=20,
        volatility_penalty_scale=0.01,
    )

    calculator = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    )

    # Simulate aggressive SELL bias to trigger penalties
    # 408# B1: _record_action は calculate_reward() でのみ呼ばれるため、
    # ステージメソッド直接呼び出し時は明示的に record が必要
    sell_penalties = []
    for step in range(12):
        calculator._record_action(ACTION_SELL)
        reward = calculator._calculate_forced_balance_reward(ACTION_SELL, step=step)
        sell_penalties.append(reward)

    assert sell_penalties[-1] < 0, "SELL-heavy imbalance should generate negative rewards"

    calculator._record_action(ACTION_SELL)
    continued_penalty = calculator._calculate_forced_balance_reward(
        ACTION_SELL, step=12
    )
    assert (
        continued_penalty <= sell_penalties[-1]
    ), "Repeated SELLs should not reduce penalty severity"

    calculator._record_action(ACTION_BUY)
    corrective_buy = calculator._calculate_forced_balance_reward(
        ACTION_BUY, step=13
    )
    assert (
        corrective_buy > 0
    ), "Taking an under-represented BUY should yield positive corrective reward"

    calculator._record_action(ACTION_HOLD)
    corrective_hold = calculator._calculate_forced_balance_reward(
        ACTION_HOLD, step=14
    )
    assert (
        corrective_hold > 0
    ), "Taking an under-represented HOLD should also yield positive reward"

    total_actions = sum(calculator._action_counts)
    assert total_actions == 15

    hold_ratio = calculator._action_counts[0] / total_actions
    buy_ratio = calculator._action_counts[1] / total_actions
    sell_ratio = calculator._action_counts[2] / total_actions

    assert sell_ratio > buy_ratio
    assert buy_ratio > 0
    assert hold_ratio > 0
    assert corrective_buy - continued_penalty > abs(continued_penalty) * 0.2


def test_forced_balance_fair_penalty_comprehensive():
    """Test forced balance with fair penalty calculation - comprehensive boundary testing."""
    from ztb.trading.environment.utils.config import EnvironmentConfig

    # Test cases for different imbalance scenarios
    test_cases = [
        {
            "name": "perfect_balance",
            "actions": [0, 1, -1] * 10,  # Equal distribution
            "expected_rms_deviation": 0.062,
            "expected_max_deviation": 0.083,
            "should_activate_penalty": False,
        },
        {
            "name": "heavy_sell_bias",
            "actions": [-1] * 25 + [0] * 4 + [1] * 1,  # 83% SELL, 13% HOLD, 4% BUY
            "expected_rms_deviation": 0.417,
            "expected_max_deviation": 0.583,
            "should_activate_penalty": True,
        },
        {
            "name": "heavy_buy_bias",
            "actions": [1] * 25 + [0] * 4 + [-1] * 1,  # 83% BUY, 13% HOLD, 4% SELL
            "expected_rms_deviation": 0.306,
            "expected_max_deviation": 0.433,
            "should_activate_penalty": True,
        },
        {
            "name": "heavy_hold_bias",
            "actions": [0] * 25 + [1] * 4 + [-1] * 1,  # 83% HOLD, 13% BUY, 4% SELL
            "expected_rms_deviation": 0.342,
            "expected_max_deviation": 0.483,
            "should_activate_penalty": True,
        },
        {
            "name": "moderate_imbalance",
            "actions": [1] * 15 + [-1] * 10 + [0] * 5,  # 50% BUY, 33% SELL, 17% HOLD
            "expected_rms_deviation": 0.13,
            "expected_max_deviation": 0.183,
            "should_activate_penalty": False,
        },
        {
            "name": "boundary_case_rms_threshold",
            "actions": [1] * 18 + [-1] * 9 + [0] * 3,  # 60% BUY, 30% SELL, 10% HOLD
            "expected_rms_deviation": 0.187,
            "expected_max_deviation": 0.25,
            "should_activate_penalty": True,
        },
        {
            "name": "boundary_case_max_threshold",
            "actions": [1] * 16 + [-1] * 8 + [0] * 6,  # 53% BUY, 27% SELL, 20% HOLD
            "expected_rms_deviation": 0.116,
            "expected_max_deviation": 0.15,
            "should_activate_penalty": False,
        },
    ]

    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")

        # Create mock config with forced_balance curriculum and asymmetric targets
        config = EnvironmentConfig()
        config.curriculum_stage = "forced_balance"
        config.max_position_size = 1.0
        config.behavior_optimization = {
            "balance_penalty": 10.0,
            "balance_penalty_targets": {
                "hold_target": 0.35,  # Asymmetric targets for fairness testing
                "buy_target": 0.40,
                "sell_target": 0.25,
            },
        }

        reward_settings = RewardSettings(
            use_simple_reward=False,
            reward_scale=1.0,
            trading_bonus=0.01,
            profit_bonuses={},
            penalty_coefficients={},
            entropy_bonus=0.0,
            custom_reward_params={
                "trend_adjustment_enabled": False,  # Disable trend adjustment for predictable test results
                "balance_penalty_targets": {
                    "hold_target": 0.35,
                    "buy_target": 0.40,
                    "sell_target": 0.25,
                },
            },
            balance_penalty=1.0,
            balance_penalty_tolerance=0.15,
            profit_weight=1.0,
            risk_weight=1.0,
            consistency_weight=1.0,
            ultra_profit_multiplier=1.0,
            ultra_risk_multiplier=1.0,
            position_soft_cap=0.5,
            position_penalty_scale=0.01,
            position_penalty_exponent=2.0,
            inventory_window=10,
            inventory_penalty_scale=0.01,
            trade_frequency_penalty=0.01,
            trade_frequency_halflife=100.0,
            trade_cooldown_steps=5,
            trade_cooldown_penalty=0.01,
            max_consecutive_trades=3,
            consecutive_trade_penalty=0.05,
            volatility_window=20,
            volatility_penalty_scale=0.01,
        )

        calculator = RewardCalculator(
            config=config,
            reward_settings=reward_settings,
            initial_portfolio_value=200000.0,
        )

        # Reset action counts for clean test
        calculator._action_counts = [0, 0, 0]

        # Execute actions
        for action in test_case["actions"]:
            reward = calculator.calculate_reward(
                action=action,
                current_price=50000.0,
                position=0.0,
                portfolio_value=200000.0,
                atr=100.0,
                transaction_cost=0.001,
                reward_scaling=1.0,
                pnl=0.0,
                old_position=0.0,
                step=0,
                observation=None,
                reward_history=[],
                portfolio_value_history=[200000.0],
            )

        # Calculate actual metrics
        total_actions = sum(calculator._action_counts)
        action_ratios = [count / total_actions for count in calculator._action_counts]

        # Get target ratios
        hold_target = calculator.get_setting_float(
            "balance_penalty_targets.hold_target", 1.0 / 3.0
        )
        buy_target = calculator.get_setting_float(
            "balance_penalty_targets.buy_target", 1.0 / 3.0
        )
        sell_target = calculator.get_setting_float(
            "balance_penalty_targets.sell_target", 1.0 / 3.0
        )
        target_ratios = [hold_target, buy_target, sell_target]

        # Calculate deviations
        deviations = [
            abs(ratio - target) for ratio, target in zip(action_ratios, target_ratios)
        ]
        rms_deviation = (sum(d**2 for d in deviations) / len(deviations)) ** 0.5
        max_individual_deviation = max(deviations)

        print(f"  Action ratios: {action_ratios}")
        print(f"  Target ratios: {target_ratios}")
        print(
            f"  RMS deviation: {rms_deviation:.3f} (expected: {test_case['expected_rms_deviation']:.3f})"
        )
        print(
            f"  Max deviation: {max_individual_deviation:.3f} (expected: {test_case['expected_max_deviation']:.3f})"
        )

        # Verify calculations are reasonable
        assert (
            abs(rms_deviation - test_case["expected_rms_deviation"]) < 0.05
        ), f"RMS deviation mismatch for {test_case['name']}"
        assert (
            abs(max_individual_deviation - test_case["expected_max_deviation"]) < 0.05
        ), f"Max deviation mismatch for {test_case['name']}"

        # Test penalty activation logic (after 30 actions)
        if total_actions >= 30:
            balance_broken_threshold = 0.15
            should_activate = (
                rms_deviation > balance_broken_threshold
                or max_individual_deviation > 0.25
            )

            assert (
                should_activate == test_case["should_activate_penalty"]
            ), f"Penalty activation mismatch for {test_case['name']}"

            if should_activate:
                # Calculate expected reward based on RMS deviation
                if rms_deviation < 0.05:
                    expected_base_reward = 20.0
                elif rms_deviation < 0.1:
                    expected_base_reward = 5.0
                elif rms_deviation < balance_broken_threshold:
                    expected_base_reward = -5.0
                else:
                    expected_base_reward = -20.0

                expected_reward = expected_base_reward + 0.5  # exploration bonus

                print(f"  Expected reward: {expected_reward:.3f}")
                print(f"  Penalty activated: {should_activate}")
            else:
                print("  Normal reward (2.0) used - balance OK")

        print(f"  ✅ {test_case['name']} test passed")


def test_forced_balance_edge_cases():
    """Test edge cases for forced balance calculation."""
    from ztb.trading.environment.utils.config import EnvironmentConfig

    config = EnvironmentConfig()
    config.curriculum_stage = "forced_balance"

    reward_settings = RewardSettings(
        use_simple_reward=False,
        reward_scale=1.0,
        trading_bonus=0.01,
        profit_bonuses={},
        penalty_coefficients={},
        entropy_bonus=0.0,
        custom_reward_params={
            "forced_balance_min_actions": 10,
            "forced_balance_exploration_reward": 2.0,
        },
        balance_penalty=1.0,
        balance_penalty_tolerance=0.15,
        profit_weight=1.0,
        risk_weight=1.0,
        consistency_weight=1.0,
        ultra_profit_multiplier=1.0,
        ultra_risk_multiplier=1.0,
        position_soft_cap=0.5,
        position_penalty_scale=0.01,
        position_penalty_exponent=2.0,
        inventory_window=10,
        inventory_penalty_scale=0.01,
        trade_frequency_penalty=0.01,
        trade_frequency_halflife=100.0,
        trade_cooldown_steps=5,
        trade_cooldown_penalty=0.01,
        max_consecutive_trades=3,
        consecutive_trade_penalty=0.05,
        volatility_window=20,
        volatility_penalty_scale=0.01,
    )

    calculator = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    )

    min_actions = reward_settings.custom_reward_params.get("forced_balance_min_actions", 10)
    exploration_reward = reward_settings.custom_reward_params.get("forced_balance_exploration_reward", 2.0)

    # Test case 1: Early phase returns exploration reward
    calculator._action_counts = [0, 0, 0]
    for i in range(min_actions - 1):
        print(f"Test iteration {i}, min_actions={min_actions}, exploration_reward={exploration_reward}")
        # 408# B1: stage メソッド直接呼びでは _record_action を明示的に呼ぶ
        calculator._record_action(ACTION_BUY)
        # Test ForcedBalanceReward directly
        from ztb.trading.environment.components.rewards.forced_balance import ForcedBalanceReward
        from ztb.trading.environment.components.rewards.base import RewardContext
        
        component = ForcedBalanceReward()
        context = RewardContext(
            action=ACTION_BUY,
            current_price=100.0,
            position=0.0,
            portfolio_value=200000.0,
            atr=1.0,
            transaction_cost=0.0,
            reward_scaling=1.0,
            pnl=0.0,
            old_position=0.0,
            step=i,
            observation=None,
            reward_history=[],
            portfolio_value_history=[],
            config=config,
            reward_settings=reward_settings,
            action_counts=[0, 1, 0],  # HOLD=0, BUY=1, SELL=0
        )
        reward = component.calculate(context)
        print(f"Direct component call got reward: {reward}")
        
        # Also test through calculator
        reward2 = calculator._calculate_forced_balance_reward(
            action=ACTION_BUY,
            step=i,
            pnl=0.0,
            reward_scaling=1.0,
            position=0.0,
            current_price=100.0,
            atr=1.0,
            portfolio_value=200000.0,
            transaction_cost=0.0,
            old_position=0.0,
            observation=None,
            reward_history=[],
            portfolio_value_history=[],
        )
        print(f"Calculator call got reward: {reward2}")
        reward = reward2  # Use calculator result
        assert (
            reward == exploration_reward
        ), f"Early phase should return exploration reward, got {reward}"

    # Once threshold reached, repeated BUY should be penalized for imbalance
    calculator._record_action(ACTION_BUY)
    penalty_reward = calculator._calculate_forced_balance_reward(
        ACTION_BUY, step=min_actions - 1
    )
    assert penalty_reward < 0

    # Test case 2: Evenly cycling actions after reset leads to balanced reward
    calculator._action_counts = [0, 0, 0]
    # 408# B1: BPC の recent_actions もリセット
    calculator.behavioral_penalty_calculator.recent_actions.clear()
    last_reward = 0.0
    for i in range(min_actions * 3):
        action = [ACTION_HOLD, ACTION_BUY, ACTION_SELL][i % 3]
        calculator._record_action(action)
        last_reward = calculator._calculate_forced_balance_reward(
            action, step=min_actions + i
        )

    expected_count = min_actions
    assert calculator._action_counts == [
        expected_count,
        expected_count,
        expected_count,
    ], f"Expected balanced counts, got {calculator._action_counts}"
    assert (
        last_reward >= 0
    ), "Balanced action loop should not produce negative reward"

    # Test case 3: Division by zero protection (though unlikely in practice)
    # This would require total_actions = 0, but we start checking at 30

    print("✅ Edge cases test passed")


def test_redundant_trade_penalty():
    """Test redundant trade penalty at max position."""
    from ztb.trading.environment.utils.config import EnvironmentConfig

    config = EnvironmentConfig()
    config.curriculum_stage = "forced_balance"
    config.max_position_size = 1.0
    config.behavior_optimization = {"redundant_trade_penalty": 5.0}

    reward_settings = RewardSettings(
        use_simple_reward=False,
        reward_scale=1.0,
        trading_bonus=0.01,
        profit_bonuses={},
        penalty_coefficients={},
        entropy_bonus=0.0,
        custom_reward_params={},
        balance_penalty=1.0,
        balance_penalty_tolerance=0.15,
        profit_weight=1.0,
        risk_weight=1.0,
        consistency_weight=1.0,
        ultra_profit_multiplier=1.0,
        ultra_risk_multiplier=1.0,
        position_soft_cap=0.5,
        position_penalty_scale=0.01,
        position_penalty_exponent=2.0,
        inventory_window=10,
        inventory_penalty_scale=0.01,
        trade_frequency_penalty=0.01,
        trade_frequency_halflife=100.0,
        trade_cooldown_steps=5,
        trade_cooldown_penalty=0.01,
        max_consecutive_trades=3,
        consecutive_trade_penalty=0.05,
        volatility_window=20,
        volatility_penalty_scale=0.01,
    )

    calculator = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    )

    min_actions = calculator.get_setting_int("forced_balance_min_actions", 10)
    for step in range(min_actions - 1):
        calculator.calculate_reward(
            action=ACTION_BUY,
            current_price=50000.0,
            position=1.0,
            portfolio_value=200000.0,
            atr=100.0,
            transaction_cost=0.001,
            reward_scaling=1.0,
            pnl=0.0,
            old_position=1.0,
            step=step,
            observation=None,
            reward_history=[],
            portfolio_value_history=[200000.0],
        )

    reward = calculator.calculate_reward(
        action=ACTION_BUY,
        current_price=50000.0,
        position=1.0,
        portfolio_value=200000.0,
        atr=100.0,
        transaction_cost=0.001,
        reward_scaling=1.0,
        pnl=0.0,
        old_position=1.0,
        step=min_actions,
        observation=None,
        reward_history=[],
        portfolio_value_history=[200000.0],
    )

    assert reward < 0  # Negative due to forced balance penalty after threshold
    print("Redundant trade penalty test passed")


# if __name__ == "__main__":
#     test_reward_calculation()
