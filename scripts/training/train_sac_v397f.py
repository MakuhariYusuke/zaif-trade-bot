"""
SAC v397f Training Script - Balanced Middle Ground

Changes from v397d/v397e:
1. Reward Scale: 500.0 (middle between v397d=1000.0 and v397e=100.0)
2. Trade Bonus: 0.2 (middle between v397d=0.05 and v397e=0.5)
3. Inactivity Penalty: 0.005 (middle between v397d=0.002 and v397e=0.02)
4. Threshold: 0.15 (same as v397e, for BUY sensitivity)

Expected Improvements:
- BUY/SELL balance: 15-25% each
- HOLD: 50-70%
- Trade count: 300-800
- Zero rewards: <70%
- Positive rewards: 3-8%
- Return: -10% to +5%
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ['MPLBACKEND'] = 'Agg'

from scripts.optimization.train_with_config import main as train_main


def main():
    config_path = project_root / "configs" / "sac_v397f_balanced_middle.json"
    
    print("=" * 80)
    print("SAC v397f Training - Balanced Middle Ground")
    print("=" * 80)
    print(f"Config: {config_path}")
    print()
    print("🔧 Balanced Parameters (middle values):")
    print("  continuous_to_discrete_threshold: 0.15 (same as v397e)")
    print("  reward_scale: 500.0 (v397d=1000.0, v397e=100.0)")
    print("  trade_execution_bonus_rate: 0.2 (v397d=0.05, v397e=0.5)")
    print("  inactivity_penalty_rate: 0.005 (v397d=0.002, v397e=0.02)")
    print()
    print("📊 Expected Outcomes:")
    print("  - BUY/SELL balance: 15-25% each")
    print("  - HOLD: 50-70%")
    print("  - Trade count: 300-800")
    print("  - Zero rewards: <70%")
    print("  - Positive rewards: 3-8%")
    print()
    print("🎯 Comparison with previous versions:")
    print("  v397d: HOLD 47.7%, SELL 42.8%, BUY 9.5%, Trades 2,615, Zero 94.6%")
    print("  v397e: HOLD 88.3%, SELL 11.7%, BUY 0.0%, Trades 583, Zero 36.5%")
    print("  v397f: Target balanced action distribution and reward signals")
    print("=" * 80)
    print()
    
    sys.argv = ["train_with_config.py", "--config", str(config_path)]
    train_main()


if __name__ == "__main__":
    main()
