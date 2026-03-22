import pytest
import numpy as np
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.calculators.reward_kernel import RewardKernel, RewardParams
from ztb.trading.environment.components.calculators.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings

def test_reward_kernel_vs_calculator_simple():
    """RewardKernel が RewardCalculator.calculate_reward_simple と一致することを確認."""
    
    # 共通設定
    reward_scaling = 2.0
    hold_penalty_multiplier = 0.5
    trade_frequency_bonus = 0.1
    
    # Mock Config / Settings
    config = EnvironmentConfig()
    # 機能を無効化して純粋比較
    config.signal_guidance_enabled = False
    
    reward_settings = RewardSettings()
    reward_settings.reward_scaling = reward_scaling
    reward_settings.hold_penalty_multiplier = hold_penalty_multiplier
    reward_settings.trade_frequency_bonus = trade_frequency_bonus
    reward_settings.bankruptcy_penalty = -100.0
    reward_settings.reward_clip_value = 10.0
    
    # オプション機能の無効化 (設定経由)
    config.dynamic_reward_shaping = {"enabled": False}
    
    calculator = RewardCalculator(config, reward_settings, initial_portfolio_value=10000.0)
    # 強制的に無効化
    calculator.dynamic_reward_shaper.enabled = False
    calculator.asymmetric_reward_scaler.enabled = False # もしあれば
    
    # テストケース: BUY アクションでの利益
    pnl = 2.0 # Clipping にかからない程度に調整
    old_pos = 0.0
    curr_pos = 0.01
    action = ACTION_BUY
    
    # Calculator の結果
    calc_reward = calculator.calculate_reward_simple(
        pnl=pnl,
        old_position=old_pos,
        position=curr_pos,
        action=action,
        portfolio_value=10000.0
    )
    
    # Kernel の結果
    params = RewardParams(
        reward_scaling=reward_scaling,
        hold_penalty_multiplier=hold_penalty_multiplier,
        trade_frequency_bonus=trade_frequency_bonus,
        bankruptcy_penalty=-100.0,
        reward_clip_value=10.0
    )
    kernel_reward = RewardKernel.calculate_basic_reward(
        pnl=pnl,
        action=action,
        params=params,
        old_position=old_pos,
        current_position=curr_pos,
        portfolio_value=10000.0
    )
    
    assert calc_reward == pytest.approx(kernel_reward)
    assert calc_reward == pytest.approx(pnl * reward_scaling + trade_frequency_bonus)

def test_reward_kernel_bankruptcy():
    """破産時のペナルティが正しく適用されるか."""
    params = RewardParams(bankruptcy_penalty=-50.0, reward_clip_value=None)
    
    # 資産あり
    reward_alive = RewardKernel.calculate_basic_reward(
        pnl=0.0, action=ACTION_HOLD, params=params, portfolio_value=100.0
    )
    # 破産
    reward_dead = RewardKernel.calculate_basic_reward(
        pnl=0.0, action=ACTION_HOLD, params=params, portfolio_value=0.0
    )
    
    assert reward_dead == reward_alive - 50.0

def test_reward_kernel_advanced_helpers():
    """RewardUtils を使った高度な計算が正しく統合されているか."""
    params = RewardParams(
        activity_bonus_rate=0.1,
        balance_penalty_coeff=1.0,
        balance_penalty_targets=[0.5, 0.25, 0.25], # [HOLD, BUY, SELL]
        balance_penalty_tolerance=0.0,
        reward_clip_value=None # クリップ無効化
    )
    
    # アクティビティボーナス (10回中6回取引) -> 0.1 * (6/10) = 0.06
    recent_actions = [ACTION_BUY, ACTION_HOLD, ACTION_SELL, ACTION_HOLD, ACTION_BUY] * 2
    
    # バランスペナルティ (合計10回)
    # HOLD 4回 (0.4), BUY 4回 (0.4), SELL 2回 (0.2)
    # ターゲット: HOLD 0.5, BUY 0.25, SELL 0.25
    # 偏差: 
    #   HOLD: |0.4 - 0.5| = 0.1
    #   BUY:  |0.4 - 0.25| = 0.15
    #   SELL: |0.2 - 0.25| = 0.05
    # 合計偏差 = 0.1 + 0.15 + 0.05 = 0.3
    # ペナルティ = 1.0 * 0.3 = 0.3
    action_counts = [4, 4, 2]
    
    reward = RewardKernel.calculate_basic_reward(
        pnl=0.0,
        action=ACTION_HOLD,
        params=params,
        recent_actions=recent_actions,
        action_counts=action_counts
    )
    
    # 合計: 0.06 (ボーナス) - 0.3 (ペナルティ) = -0.24
    assert reward == pytest.approx(-0.24)
