"""
v394シリーズ4バージョンの訓練結果を比較
"""

import json
from pathlib import Path
import pandas as pd


def analyze_training_logs(version: str) -> dict:
    """訓練ログから主要メトリクスを抽出"""
    model_dir = Path(f"models/{version}")
    if not model_dir.exists():
        return {"error": "Model not found"}
    
    # TensorBoard logsを読み取る（簡易版）
    return {
        "version": version,
        "model_dir": str(model_dir),
        "exists": model_dir.exists()
    }


def compare_versions():
    """v394a, b, c, dの訓練結果を比較"""
    versions = [
        "ppo_v394a_hold_penalty",
        "ppo_v394b_trade_reward",
        "ppo_v394c_balanced",
        "ppo_v394d_aggressive"
    ]
    
    print("="*80)
    print("v394 Series Comparison")
    print("="*80)
    print()
    
    results = []
    for version in versions:
        config_path = f"configs/{version}.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        reward_settings = config['environment']['reward_settings']
        
        results.append({
            "Version": version.split('_')[-2:],
            "HOLD Penalty": reward_settings.get('hold_penalty_weight', 0.02),
            "Consec HOLD": reward_settings.get('consecutive_hold_penalty', 0.01),
            "Trade Bonus": reward_settings.get('successful_trade_bonus', 1.0),
            "Profit Mult": reward_settings.get('profit_reward_multiplier', 5.0),
            "Trade Freq": reward_settings.get('trading_frequency_bonus', 0.15),
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()
    
    # 期待される効果
    print("="*80)
    print("Expected Effects")
    print("="*80)
    print()
    print("v394a (HOLD Penalty):")
    print("  ✅ HOLD行動を強力に抑制")
    print("  ✅ 取引頻度増加")
    print("  ⚠️ 過度な取引リスク")
    print()
    print("v394b (Trade Reward):")
    print("  ✅ 利益のある取引を強化")
    print("  ✅ Quality向上")
    print("  ⚠️ HOLD比率は維持される可能性")
    print()
    print("v394c (Balanced):")
    print("  ✅ 両方を適度に強化")
    print("  ✅ 安定した学習")
    print("  ✅ バランスの取れた行動分布")
    print()
    print("v394d (Aggressive):")
    print("  ✅ 最大強化で積極的取引")
    print("  ⚠️ 学習不安定リスク")
    print("  ⚠️ 過学習リスク")
    print("="*80)


if __name__ == "__main__":
    compare_versions()
