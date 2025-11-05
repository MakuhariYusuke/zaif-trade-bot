#!/usr/bin/env python3
"""
Deep investigation of why SELL bias persists despite balance penalty fixes.
Tracks actual reward values and action distributions during training.
"""

import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np

# Setup path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.environment.components.reward_calculator import RewardCalculator, RewardSettings
from ztb.trading.constants import ACTION_BUY, ACTION_SELL, ACTION_HOLD

def deep_investigate_reward_calculation():
    """深く掘り下げて、実際の報酬計算を検査する"""
    
    print("\n" + "="*80)
    print("DEEP INVESTIGATION: Reward Calculation During Training")
    print("="*80)
    
    # Load config
    config_path = Path("config/sac_v444_3_balanced_penalty_scale_200.json")
    with open(config_path) as f:
        config_data = json.load(f)
    
    # Create environment config
    env_config = config_data['environment'].copy()
    env_config.update(env_config.get('behavior_optimization', {}))
    env_config.update(env_config.get('action_bonuses', {}))
    
    # Add curriculum_stage
    curriculum_config = config_data['training']['curriculum_learning']
    env_config['curriculum_stage'] = curriculum_config['curriculum_stage']
    
    print(f"\n1. CONFIGURATION LOADED")
    print(f"   curriculum_stage: {env_config['curriculum_stage']}")
    print(f"   balance_penalty: {env_config.get('balance_penalty', 'NOT SET')}")
    print(f"   action_balance_target: {env_config.get('action_balance_target', 'NOT SET')}")
    
    # Convert to EnvironmentConfig
    env_cfg = EnvironmentConfig.from_dict(env_config)
    
    print(f"\n2. ENVIRONMENT CONFIG CONVERSION")
    print(f"   curriculum_stage: {env_cfg.curriculum_stage}")
    print(f"   behavior_optimization: {env_cfg.behavior_optimization}")
    
    # Create reward calculator
    reward_settings = RewardSettings()
    reward_calc = RewardCalculator(env_cfg, reward_settings, 200000.0)
    
    print(f"\n3. REWARD CALCULATOR CREATED")
    print(f"   Using curriculum_stage: {reward_calc.config.curriculum_stage}")
    
    # Simulate SELL-locked behavior
    print(f"\n4. SIMULATING ALL-SELL ACTION SEQUENCE")
    all_sell_actions = [ACTION_SELL] * 20
    
    total_reward_all_sell = 0
    penalties_all_sell = []
    
    for i, action in enumerate(all_sell_actions):
        reward_calc._recent_actions.append(action)
        
        # Simulate reward calculation with all-sell actions
        # Base reward = PnL (random between -1000 and 1000)
        pnl = np.random.uniform(-1000, 1000)
        
        # Calculate reward
        reward = reward_calc.calculate_reward(
            action=action,
            pnl=pnl,
            position=0.0,
            current_price=5000000,
            atr=50000
        )
        
        total_reward_all_sell += reward
        
        if i >= 9:  # After at least 10 actions
            # Check what balance penalty was applied
            if len(reward_calc._recent_actions) >= 10:
                counter = Counter(reward_calc._recent_actions[-10:])
                buy_count = counter.get(ACTION_BUY, 0)
                sell_count = counter.get(ACTION_SELL, 0)
                hold_count = counter.get(ACTION_HOLD, 0)
                
                print(f"   Step {i+1}: Action={action}, Distribution: BUY={buy_count} SELL={sell_count} HOLD={hold_count}, Reward={reward:.2f}")
    
    print(f"\n   Total reward (ALL SELL): {total_reward_all_sell:.2f}")
    
    # Now try BUY-locked
    print(f"\n5. SIMULATING ALL-BUY ACTION SEQUENCE")
    reward_calc._recent_actions.clear()
    all_buy_actions = [ACTION_BUY] * 20
    
    total_reward_all_buy = 0
    
    for i, action in enumerate(all_buy_actions):
        reward_calc._recent_actions.append(action)
        
        pnl = np.random.uniform(-1000, 1000)
        
        reward = reward_calc.calculate_reward(
            action=action,
            pnl=pnl,
            position=0.0,
            current_price=5000000,
            atr=50000
        )
        
        total_reward_all_buy += reward
        
        if i >= 9:
            counter = Counter(reward_calc._recent_actions[-10:])
            buy_count = counter.get(ACTION_BUY, 0)
            sell_count = counter.get(ACTION_SELL, 0)
            hold_count = counter.get(ACTION_HOLD, 0)
            
            print(f"   Step {i+1}: Action={action}, Distribution: BUY={buy_count} SELL={sell_count} HOLD={hold_count}, Reward={reward:.2f}")
    
    print(f"\n   Total reward (ALL BUY): {total_reward_all_buy:.2f}")
    
    # Try balanced
    print(f"\n6. SIMULATING BALANCED ACTION SEQUENCE")
    reward_calc._recent_actions.clear()
    balanced_actions = [ACTION_BUY, ACTION_SELL, ACTION_HOLD] * 6 + [ACTION_BUY, ACTION_SELL]
    
    total_reward_balanced = 0
    
    for i, action in enumerate(balanced_actions):
        reward_calc._recent_actions.append(action)
        
        pnl = np.random.uniform(-1000, 1000)
        
        reward = reward_calc.calculate_reward(
            action=action,
            pnl=pnl,
            position=0.0,
            current_price=5000000,
            atr=50000
        )
        
        total_reward_balanced += reward
        
        if i >= 9:
            counter = Counter(reward_calc._recent_actions[-10:])
            buy_count = counter.get(ACTION_BUY, 0)
            sell_count = counter.get(ACTION_SELL, 0)
            hold_count = counter.get(ACTION_HOLD, 0)
            
            print(f"   Step {i+1}: Action={action}, Distribution: BUY={buy_count} SELL={sell_count} HOLD={hold_count}, Reward={reward:.2f}")
    
    print(f"\n   Total reward (BALANCED): {total_reward_balanced:.2f}")
    
    # Analysis
    print(f"\n7. REWARD ANALYSIS")
    print(f"   Average reward per action:")
    print(f"     ALL SELL:  {total_reward_all_sell/len(all_sell_actions):.2f}")
    print(f"     ALL BUY:   {total_reward_all_buy/len(all_buy_actions):.2f}")
    print(f"     BALANCED:  {total_reward_balanced/len(balanced_actions):.2f}")
    
    if total_reward_all_sell > total_reward_all_buy:
        print(f"\n   ⚠️  CRITICAL: ALL SELL reward ({total_reward_all_sell:.2f}) > ALL BUY ({total_reward_all_buy:.2f})")
        print(f"       This explains why model learns SELL bias!")
    elif total_reward_all_buy > total_reward_balanced:
        print(f"\n   ⚠️  CRITICAL: ALL BUY reward ({total_reward_all_buy:.2f}) > BALANCED ({total_reward_balanced:.2f})")
        print(f"       This explains why model avoids balance!")
    else:
        print(f"\n   ✓ Rewards properly penalize single-action bias")

if __name__ == "__main__":
    try:
        deep_investigate_reward_calculation()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
