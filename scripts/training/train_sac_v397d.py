"""
SAC v397d_balanced_fix Training Script
Next Steps実装版：環境設定反映 + ペナルティ正規化 + 報酬バランス最適化

重要な修正点:
1. EnvironmentConfig拡張
   - continuous_to_discrete_threshold=0.20 → 環境に正しく反映されるよう修正
   - EnvironmentConfig.from_dict()でcontinuous_to_discrete_thresholdを認識
   - HeavyTradingEnv.__init__()でaction_thresholdとして使用

2. Inactivityペナルティ正規化
   - inactivity_penalty_rate: 0.01 → 0.002 (1/5に削減)
   - inactivity_hold_threshold: 0.05 → 0.01 (実際の購入可能サイズ0.036 BTC考慮)
   - 資金200,000円、BTC価格5,000,000円の制約を反映

3. 報酬バランス再調整
   - PnL主体（86%維持）
   - Inactivityペナルティ: 14% → 3%目標（rate 1/5削減）
   - Trade Bonus: 0.2% → 適度に発火（threshold削減で検出向上）
   - Opportunity Cost: 無効化（資金制約で機能不全のため）

4. アクション閾値最適化
   - continuous_to_discrete_threshold: 0.25 → 0.20
   - HOLD範囲: [-0.2, 0.2]（v397c: [-0.25, 0.25]より狭い）
   - BUY/SELL判定をより敏感に

期待される効果:
- アクション分布: HOLD 50-60%, BUY 20-25%, SELL 20-25%
- 報酬バランス: PnL 90%, Penalty 3%, Bonus 5%, Cost 2%
- 取引頻度: 適度（100-500回/5000ステップ）
- 収益性: 市場トレンド（+2.5%）追従を期待
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
    config_path = project_root / "configs" / "sac_v397d_balanced_fix.json"
    
    print("=" * 80)
    print("SAC v397d_balanced_fix Training")
    print("=" * 80)
    print(f"Config: {config_path}")
    print("\n🔧 Next Steps実装:")
    print("  1. EnvironmentConfig拡張: continuous_to_discrete_threshold反映")
    print("  2. Inactivityペナルティ正規化: rate 0.002, threshold 0.01")
    print("  3. 報酬バランス再調整: PnL 90%, Penalty 3%, Bonus 5%")
    print("  4. アクション閾値最適化: threshold 0.20")
    print("\n期待される効果:")
    print("  - HOLD: 50-60% (v397c: 76%から改善)")
    print("  - 取引: 100-500回 (v397c: 745回から適度に)")
    print("  - 収益: 市場トレンド+2.5%追従")
    print("=" * 80)
    
    sys.argv = ["train_with_config.py", "--config", str(config_path)]
    main()
