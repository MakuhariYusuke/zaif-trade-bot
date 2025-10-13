"""
SAC v397g Training Script - Realistic Buy Size

設計:
- max_position_size: 0.01 (BTC価格500万円なら約5万円分)
- initial_balance: 200,000円 (0.04 BTC分)
- continuous_to_discrete_threshold: 0.10 (BUY/SELL判定敏感化)
- reward_scale: 1000.0 (PnL報酬強調)
- trade_execution_bonus_rate: 0.2
- inactivity_penalty_rate: 0.005
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.environ['MPLBACKEND'] = 'Agg'

from scripts.optimization.train_with_config import main as train_main

def main():
    config_path = project_root / "configs" / "sac_v397g_realistic_buy_size.json"
    print("=" * 80)
    print("SAC v397g Training - Realistic Buy Size")
    print("=" * 80)
    print(f"Config: {config_path}")
    print("  max_position_size: 0.01")
    print("  initial_balance: 200,000円")
    print("  threshold: 0.10")
    print("  reward_scale: 1000.0")
    print("  trade_execution_bonus_rate: 0.2")
    print("  inactivity_penalty_rate: 0.005")
    print("=" * 80)
    sys.argv = ["train_with_config.py", "--config", str(config_path)]
    train_main()

if __name__ == "__main__":
    main()
