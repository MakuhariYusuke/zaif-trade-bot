#!/usr/bin/env python3
"""
Debug script to check RewardCalculator state and balance_penalty calculation
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("\n" + "=" * 80)
print("DEBUG: RewardCalculator Balance Penalty Calculation")
print("=" * 80)

# Step 1: Create RewardCalculator
logger.info("\n[1] Creating EnvironmentConfig...")
env_config = EnvironmentConfig(
    curriculum_stage="balanced_penalty",
    exchange="coincheck"
)

logger.info(f"EnvironmentConfig.curriculum_stage = {env_config.curriculum_stage}")
logger.info(f"EnvironmentConfig.reward_settings = {env_config.reward_settings}")

# Step 2: Create RewardCalculator
logger.info("\n[2] Creating RewardCalculator...")
reward_settings = RewardSettings()
reward_settings.balance_penalty = 200.0
reward_settings.action_balance_target = 0.333

reward_calc = RewardCalculator(env_config, reward_settings, 200000.0)

logger.info(f"RewardCalculator created")
logger.info(f"RewardCalculator._recent_actions initial = {reward_calc._recent_actions}")

# Step 3: Simulate 25 steps of all SELL actions
logger.info("\n[3] Simulating 25 SELL actions...")
for i in range(25):
    # Simulate appending SELL action
    # deque automatically removes old items when maxlen is exceeded
    reward_calc._recent_actions.append(ACTION_SELL)
    
    if i % 5 == 4:
        logger.info(f"  After step {i+1}: _recent_actions len={len(reward_calc._recent_actions)}, "
                   f"content={list(reward_calc._recent_actions)}")

logger.info(f"\nFinal _recent_actions: {list(reward_calc._recent_actions)}")
logger.info(f"Final _recent_actions length: {len(reward_calc._recent_actions)}")

# Step 4: Check balance_penalty calculation
logger.info("\n[4] Checking balance_penalty calculation...")
import collections
from ztb.trading.constants import ACTION_SELL

total_actions = len(reward_calc._recent_actions)
counter = collections.Counter(reward_calc._recent_actions)
buy_count = counter[ACTION_BUY]
sell_count = counter[ACTION_SELL]
hold_count = counter[ACTION_HOLD]

buy_ratio = buy_count / total_actions if total_actions > 0 else 0
sell_ratio = sell_count / total_actions if total_actions > 0 else 0
hold_ratio = hold_count / total_actions if total_actions > 0 else 0

logger.info(f"total_actions = {total_actions}")
logger.info(f"action counts: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")
logger.info(f"action ratios: BUY={buy_ratio:.3f}, SELL={sell_ratio:.3f}, HOLD={hold_ratio:.3f}")

# Asymmetric targets
buy_target = 0.4
sell_target = 0.25
hold_target = 0.35

deviation_buy = abs(buy_ratio - buy_target)
deviation_sell = abs(sell_ratio - sell_target)
deviation_hold = abs(hold_ratio - hold_target)

total_deviation = deviation_buy + deviation_sell + deviation_hold
balance_penalty = total_deviation * 200.0

logger.info(f"\nAsymmetric targets: BUY={buy_target}, SELL={sell_target}, HOLD={hold_target}")
logger.info(f"Deviations: BUY={deviation_buy:.3f}, SELL={deviation_sell:.3f}, HOLD={deviation_hold:.3f}")
logger.info(f"Total deviation: {total_deviation:.3f}")
logger.info(f"Balance penalty (scale=200.0): {balance_penalty:.2f}")

# Expected for ALL_SELL
logger.info(f"\nExpected for ALL_SELL:")
logger.info(f"  |0 - 0.4| + |1 - 0.25| + |0 - 0.35| = 0.4 + 0.75 + 0.35 = 1.5")
logger.info(f"  Penalty = 1.5 * 200.0 = 300.0")
logger.info(f"  Actual calculated: {balance_penalty:.2f}")

if balance_penalty < 100:
    logger.warning("\n⚠️ WARNING: Balance penalty is suspiciously low!")
    logger.warning("This may indicate _recent_actions is not being updated correctly")
else:
    logger.info("\n✅ Balance penalty calculation appears correct")

print("\n" + "=" * 80)
