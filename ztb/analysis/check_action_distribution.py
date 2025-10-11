#!/usr/bin/env python3
"""
アクション分布確認スクリプト

短時間でモデルのアクション分布を確認し、
HOLD偏重が改善されているかをチェックします。
"""
import argparse
import numpy as np
from pathlib import Path
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment.schema_env_factory import create_env_from_model_path
from ztb.utils.data_utils import load_csv_data_optimized


def check_action_distribution(model_path: str, data_path: str, steps: int = 1000):
    """
    モデルのアクション分布を短時間でチェック
    
    Args:
        model_path: モデルファイルパス
        data_path: データファイルパス
        steps: 実行ステップ数（デフォルト1000）
    """
    model_path = Path(model_path)
    print(f"\n{'='*80}")
    print(f"Action Distribution Check: {model_path.stem}")
    print(f"{'='*80}\n")
    
    # データ読み込み
    df = load_csv_data_optimized(data_path)
    print(f"Data: {len(df):,} rows")
    
    # 環境作成
    base_env = create_env_from_model_path(model_path, df)
    print(f"Environment: {base_env.observation_space.shape[0]} features")
    
    # VecEnv化
    env = DummyVecEnv([lambda: base_env])
    
    # モデル読み込み
    model = MaskablePPO.load(str(model_path), env=env)
    print(f"Model loaded\n")
    
    # アクション収集
    action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
    obs = env.reset()
    
    for step in range(steps):
        action_masks = np.array([base_env.action_mask()])
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        
        # カウント
        action_int = int(action[0])
        if action_int == 0:
            action_counts["HOLD"] += 1
        elif action_int == 1:
            action_counts["BUY"] += 1
        else:  # action_int == 2
            action_counts["SELL"] += 1
        
        # ステップ実行
        obs, _, done, _ = env.step(action)
        
        if done[0]:
            obs = env.reset()
        
        # 進捗表示
        if (step + 1) % 200 == 0:
            total = step + 1
            print(f"Progress: {total}/{steps} steps", end='\r')
    
    print(f"\n")
    
    # 結果表示
    total = sum(action_counts.values())
    print(f"{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Total Steps: {total:,}")
    print(f"\nAction Distribution:")
    
    for action_name, count in action_counts.items():
        pct = (count / total * 100) if total > 0 else 0
        bar_length = int(pct / 2)  # 50文字スケール
        bar = '█' * bar_length
        print(f"  {action_name:5s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    print(f"{'='*80}\n")
    
    # 判定
    hold_pct = (action_counts["HOLD"] / total * 100) if total > 0 else 0
    trade_pct = ((action_counts["BUY"] + action_counts["SELL"]) / total * 100) if total > 0 else 0
    
    print("Assessment:")
    if hold_pct > 95:
        print("  ❌ HOLD偏重 - ほぼ取引していません")
    elif hold_pct > 80:
        print("  ⚠️  HOLD多め - 取引頻度が低いです")
    elif hold_pct < 20:
        print("  ⚠️  取引過多 - オーバートレーディングの可能性")
    else:
        print("  ✅ バランス良好 - 適度な取引頻度です")
    
    print(f"\n  HOLD: {hold_pct:.1f}%")
    print(f"  Trade (BUY+SELL): {trade_pct:.1f}%")
    print(f"{'='*80}\n")
    
    return {
        'model_name': model_path.stem,
        'action_counts': action_counts,
        'action_percentages': {
            k: (v / total * 100) if total > 0 else 0
            for k, v in action_counts.items()
        },
        'total_steps': total,
        'hold_pct': hold_pct,
        'trade_pct': trade_pct
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--data", type=str, default="ml-dataset-enhanced.csv", help="Data path")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps")
    args = parser.parse_args()
    
    check_action_distribution(args.model, args.data, args.steps)
