"""
SAC v397b_balanced Training Script
改善した報酬設定でバランスの取れた取引を促進

主な変更点:
- reward_scale: 100.0 → 200.0 (PnL報酬を2倍に)
- inactivity_penalty_rate: 0.004 → 0.001 (ペナルティを75%削減)
- inactivity_penalty_window: 5 → 3 (累積も緩和)
- trade_execution_bonus_rate: 0.012 → 0.05 (ボーナスを4倍に)
- trade_execution_position_threshold: 0.05 → 0.01 (閾値を1/5に)
- opportunity_cost: 無効化 (シンプルな報酬構造に)
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ['MPLBACKEND'] = 'Agg'

from scripts.optimization.train_with_config import main

if __name__ == "__main__":
    config_path = project_root / "configs" / "sac_v397b_balanced.json"
    
    print("=" * 80)
    print("SAC v397b_balanced Training")
    print("=" * 80)
    print(f"Config: {config_path}")
    print("Target: バランスの取れた取引（HOLD < 60%, 取引 > 50回）")
    print("=" * 80)
    
    sys.argv = ["train_with_config.py", "--config", str(config_path)]
    main()
