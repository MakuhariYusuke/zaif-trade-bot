#!/usr/bin/env python3
"""
Deep investigation: Why balance_penalty is 0.00 in actual training
but 300.0 in verification test?
"""

import logging
import sys
from pathlib import Path
from collections import deque, Counter

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
import json

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_balance_penalty_in_real_env():
    """Test balance_penalty calculation in actual environment"""
    
    # Load configuration
    config_path = Path("config/sac_v444_3_balanced_penalty_scale_200.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    
    # Check raw config
    env_config = config_dict.get('environment', {})
    behavior_opt = env_config.get('behavior_optimization', {})
    logger.info(f"Raw config balance_penalty: {behavior_opt.get('balance_penalty', 'NOT FOUND')}")
    
    # Create environment
    config = EnvironmentConfig.from_dict(config_dict)
    logger.info(f"Environment config reward_settings: {config.reward_settings}")
    if config.reward_settings:
        logger.info(f"Environment config balance_penalty: {config.reward_settings.balance_penalty}")
    
    env = HeavyTradingEnv(config=config)
    
    # Reset environment
    obs, info = env.reset()
    logger.info(f"Environment reset. Initial observation shape: {obs.shape}")
    
    # Track internal state
    logger.info("\n=== INTERNAL STATE CHECK ===")
    reward_calc = env.reward_calculator
    logger.info(f"_recent_actions exists: {hasattr(reward_calc, '_recent_actions')}")
    logger.info(f"_recent_actions type: {type(reward_calc._recent_actions)}")
    logger.info(f"_recent_actions value: {reward_calc._recent_actions}")
    logger.info(f"_recent_actions length: {len(reward_calc._recent_actions)}")
    
    # Take 50 SELL actions to accumulate history
    logger.info("\n=== SIMULATING 50 SELL ACTIONS ===")
    for step in range(50):
        action = -1  # SELL
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0 and step > 0:
            logger.info(f"\nStep {step}:")
            logger.info(f"  _recent_actions length: {len(reward_calc._recent_actions)}")
            logger.info(f"  _recent_actions type: {type(reward_calc._recent_actions)}")
            if len(reward_calc._recent_actions) > 0:
                counter = Counter(reward_calc._recent_actions)
                logger.info(f"  Action distribution: {dict(counter)}")
                logger.info(f"  Reward (last step): {reward:.2f}")
    
    # Final state check
    logger.info(f"\n=== FINAL STATE AFTER 50 STEPS ===")
    logger.info(f"_recent_actions length: {len(reward_calc._recent_actions)}")
    logger.info(f"_recent_actions type: {type(reward_calc._recent_actions)}")
    logger.info(f"_recent_actions: {list(reward_calc._recent_actions)}")
    
    if len(reward_calc._recent_actions) > 0:
        counter = Counter(reward_calc._recent_actions)
        logger.info(f"Action distribution: {dict(counter)}")
        
        total = len(reward_calc._recent_actions)
        buy_ratio = counter.get(1, 0) / total if total > 0 else 0
        sell_ratio = counter.get(-1, 0) / total if total > 0 else 0
        hold_ratio = counter.get(0, 0) / total if total > 0 else 0
        
        logger.info(f"Ratios: BUY={buy_ratio:.2%}, SELL={sell_ratio:.2%}, HOLD={hold_ratio:.2%}")
        
        # Calculate expected penalty
        buy_target = 0.4
        sell_target = 0.25
        hold_target = 0.35
        
        dev_buy = abs(buy_ratio - buy_target)
        dev_sell = abs(sell_ratio - sell_target)
        dev_hold = abs(hold_ratio - hold_target)
        total_dev = dev_buy + dev_sell + dev_hold
        
        scale = config.reward_settings.behavior_optimization.get('balance_penalty', 200.0)
        expected_penalty = total_dev * scale
        
        logger.info(f"\nExpected balance_penalty:")
        logger.info(f"  Deviations: buy={dev_buy:.4f}, sell={dev_sell:.4f}, hold={dev_hold:.4f}")
        logger.info(f"  Total deviation: {total_dev:.4f}")
        logger.info(f"  Scale: {scale:.1f}")
        logger.info(f"  Expected penalty: {expected_penalty:.2f}")
    
    logger.info("\n=== INVESTIGATION COMPLETE ===")

if __name__ == "__main__":
    test_balance_penalty_in_real_env()
