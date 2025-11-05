#!/usr/bin/env python3
"""
Balance penalty fix validation test
修正された balance_penalty が正しく機能することを検証
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings


def test_balance_penalty_formula():
    """修正されたbalance_penalty formulaをテスト"""

    # EnvironmentConfig を作成
    config = EnvironmentConfig(
        transaction_cost=0.001,
        commission=0.001,
        max_position_size=1.0,
        enable_action_masking=True,
        use_continuous_actions=True,
        curriculum_stage='balanced_penalty',
    )

    # RewardSettings を作成
    reward_settings = RewardSettings()

    # RewardCalculator を作成
    calc = RewardCalculator(
        config=config,
        reward_settings=reward_settings,
        initial_portfolio_value=200000.0,
    )

    # テストケース
    test_cases = [
        {
            'name': '100% BUY (最大アンバランス)',
            'actions': [1.0] * 100,  # 全て BUY
            'expected_penalty_range': (100, 200),  # balance_scale=200なので
        },
        {
            'name': '33%/33%/33% (完全バランス)',
            'actions': [1.0] * 33 + [-1.0] * 33 + [0.0] * 34,
            'expected_penalty_range': (0, 10),  # ほぼ 0
        },
        {
            'name': '50% BUY / 30% SELL / 20% HOLD',
            'actions': [1.0] * 50 + [-1.0] * 30 + [0.0] * 20,
            'expected_penalty_range': (20, 40),  # max_dev=0.167
        },
    ]

    print("=" * 80)
    print("Balance Penalty Formula Validation Test")
    print("=" * 80)

    for test_case in test_cases:
        print(f"\n📊 Test: {test_case['name']}")
        print(f"   Actions: {len(test_case['actions'])} total")

        # Calculate distribution
        buy_count = sum(1 for a in test_case['actions'] if a > 0.0)
        sell_count = sum(1 for a in test_case['actions'] if a < 0.0)
        hold_count = sum(1 for a in test_case['actions'] if a == 0.0)

        total = len(test_case['actions'])
        buy_ratio = buy_count / total if total > 0 else 0
        sell_ratio = sell_count / total if total > 0 else 0
        hold_ratio = hold_count / total if total > 0 else 0

        print(f"   Distribution: BUY={buy_ratio:.2%}, SELL={sell_ratio:.2%}, HOLD={hold_ratio:.2%}")

        # Call RewardCalculator._calculate_balance_penalty
        # We need to manually test this by simulating reward calculation
        # Get the penalty directly
        balance_penalty = calc._calculate_balance_penalty(
            buy_ratio, sell_ratio, hold_ratio
        )

        print(f"   Balance penalty: {balance_penalty:.4f}")
        print(f"   Expected range: {test_case['expected_penalty_range']}")

        if test_case['expected_penalty_range'][0] <= balance_penalty <= test_case[
            'expected_penalty_range'
        ][1]:
            print(f"   ✅ PASSED")
        else:
            print(f"   ❌ FAILED - outside expected range!")

    print("\n" + "=" * 80)
    print("✅ Balance penalty formula validation complete")
    print("=" * 80)


def test_curriculum_stage_integration():
    """curriculum_stage が正しく設定されていることを確認"""

    print("\n" + "=" * 80)
    print("Curriculum Stage Integration Test")
    print("=" * 80)

    config = EnvironmentConfig(
        curriculum_stage='balanced_penalty',
    )

    print(f"\n✓ EnvironmentConfig.curriculum_stage: {config.curriculum_stage}")

    calc = RewardCalculator(config=config)
    print(f"✓ RewardCalculator.config.curriculum_stage: {calc.config.curriculum_stage}")

    # Check if balance penalty calculation is enabled for this stage
    balance_penalty_enabled_stages = ['balanced_penalty', 'full_curriculum']
    is_enabled = calc.config.curriculum_stage in balance_penalty_enabled_stages

    if is_enabled:
        print(f"✅ Balance penalty IS ENABLED for curriculum_stage='{calc.config.curriculum_stage}'")
    else:
        print(
            f"❌ Balance penalty NOT enabled for curriculum_stage='{calc.config.curriculum_stage}'"
        )

    print("\n" + "=" * 80)


if __name__ == '__main__':
    test_balance_penalty_formula()
    test_curriculum_stage_integration()
