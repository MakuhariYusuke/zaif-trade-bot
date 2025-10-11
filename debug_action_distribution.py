#!/usr/bin/env python3
"""
デバッグ版 - アクション分布チェック + 環境設定確認
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sb3_contrib import MaskablePPO

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.schema_env_factory import create_env_from_model_path
from ztb.training.core.feature_schema_manager import FeatureSchemaManager

def main() -> None:
    parser = argparse.ArgumentParser(description='デバッグ版アクション分布チェック')
    parser.add_argument('--model', type=str, required=True, help='モデルファイルパス (.zip)')
    parser.add_argument('--steps', type=int, default=100, help='実行ステップ数')
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ モデルが見つかりません: {model_path}")
        return

    # データ読み込み
    data_path = Path("ml-dataset-enhanced.csv")
    print(f"\n📊 データ読み込み: {data_path}")
    df = pd.read_csv(data_path)
    df = df.head(args.steps)
    
    # スキーマ読み込み
    model_name = model_path.stem
    manager = FeatureSchemaManager(model_name, model_path.parent)
    metadata = manager.load_schema()
    
    print(f"\n🔍 メタデータ確認:")
    print(f"  特徴量数: {metadata.num_features}")
    print(f"  スキーマハッシュ: {metadata.schema_hash}")
    
    # 訓練時の環境設定を確認
    training_env_config = metadata.training_config.get("environment", {})
    print(f"\n📋 訓練時の環境設定:")
    for key, value in training_env_config.items():
        print(f"  {key}: {value}")
    
    # 環境作成
    print(f"\n🏗️ 環境作成中...")
    env = create_env_from_model_path(str(model_path), df)
    
    # 環境設定を確認
    print(f"\n⚙️ 実際の環境設定:")
    print(f"  initial_portfolio_value: {env.config.initial_portfolio_value}")
    print(f"  max_position_size: {env.config.max_position_size}")
    print(f"  transaction_cost: {env.config.transaction_cost}")
    print(f"  curriculum_stage: {env.config.curriculum_stage}")
    
    # ActionValidatorの初期値確認
    print(f"\n🔍 ActionValidator設定:")
    print(f"  initial_portfolio_value: {env.action_validator.initial_portfolio_value}")
    
    # モデル読み込み
    print(f"\n🤖 モデル読み込み: {model_path.name}")
    model = MaskablePPO.load(str(model_path))
    
    # アクション分布チェック
    print(f"\n🎯 アクション分布チェック ({args.steps} steps):")
    action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL
    
    obs, _ = env.reset()
    for step in range(args.steps):
        action_masks = env.action_mask()
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=False)
        action_int = int(action)
        action_counts[action_int] += 1
        
        # 最初の10ステップで詳細ログ
        if step < 10:
            portfolio_value = env.action_validator.initial_portfolio_value + env.position_manager.total_pnl
            
            print(f"\n  Step {step}:")
            print(f"    資金: {portfolio_value:,.0f} 円")
            print(f"    max_position_size: {env.config.max_position_size}")
            print(f"    transaction_cost: {env.config.transaction_cost}")
            print(f"    マスク: HOLD={action_masks[0]}, BUY={action_masks[1]}, SELL={action_masks[2]}")
            print(f"    アクション: {['HOLD', 'BUY', 'SELL'][action_int]}")
        
        obs, _, terminated, truncated, _ = env.step(action_int)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    # 結果表示
    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)
    total = sum(action_counts.values())
    print(f"Total Steps: {total:,}")
    print(f"\nAction Distribution:")
    for action_id, count in action_counts.items():
        action_name = ['HOLD', 'BUY', 'SELL'][action_id]
        pct = (count / total * 100) if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {action_name:4s}: {count:5,} ({pct:5.1f}%) {bar}")
    
    # 判定
    hold_pct = (action_counts[0] / total * 100) if total > 0 else 0
    trade_pct = ((action_counts[1] + action_counts[2]) / total * 100) if total > 0 else 0
    
    print("\n" + "="*80)
    print("Assessment:")
    if hold_pct > 95:
        print("  ❌ HOLD偏重 - ほぼ取引していません")
    elif hold_pct > 80:
        print("  ⚠️  HOLDやや多め - 取引頻度が低いかもしれません")
    elif hold_pct < 20:
        print("  ⚠️  オーバートレーディング - 取引が多すぎるかもしれません")
    else:
        print("  ✅ バランス良好")
    
    print(f"\n  HOLD: {hold_pct:.1f}%")
    print(f"  Trade (BUY+SELL): {trade_pct:.1f}%")
    print("="*80)

if __name__ == "__main__":
    main()
