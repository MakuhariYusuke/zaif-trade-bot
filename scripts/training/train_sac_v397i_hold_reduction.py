"""
SAC v397i Training Script - HOLD Reduction & Curriculum Learning

設計:
- max_position_size: 0.05 (BTC価格500万円なら約25万円分)
- initial_balance: 200,000円 (0.04 BTC分)
- continuous_to_discrete_threshold: 0.05 (BUY/SELL判定緩和)
- reward_scale: 1000.0 (PnL報酬強調)
- trade_execution_bonus_rate: 0.2
- immediate_bonus_rate: 0.5 (BUY/SELL平等ボーナス)
- inactivity_penalty_rate: 0.01 (HOLDペナルティ強化)
- curriculum_stage: forced_balance (カリキュラム学習初期)
- target_entropy: -2.0 (探索増加)
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ['MPLBACKEND'] = 'Agg'

from scripts.optimization.train_with_config import main as train_main

def main():
    config_path = project_root / "configs" / "sac_v397i_hold_reduction.json"
    print("=" * 80)
    print("SAC v397i Training - HOLD Reduction & Curriculum Learning")
    print("=" * 80)
    print(f"Config: {config_path}")
    print("  max_position_size: 0.05")
    print("  initial_balance: 200,000円")
    print("  threshold: 0.05")
    print("  reward_scale: 1000.0")
    print("  trade_execution_bonus_rate: 0.2")
    print("  immediate_bonus_rate: 0.5")
    print("  inactivity_penalty_rate: 0.01")
    print("  curriculum_stage: forced_balance")
    print("  target_entropy: -2.0")
    print("=" * 80)

    # 通常の訓練スクリプトを使用（カリキュラム効果はRewardCalculatorで実現）
    sys.argv = ["train_with_config.py", "--config", str(config_path)]
    train_main()

if __name__ == "__main__":
    main()