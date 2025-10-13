#!/usr/bin/env python3
"""
汎用訓練スクリプト - 最適化用
設定ファイルを引数で受け取り、SAC訓練を直接実行
"""
import os
# matplotlib import errorを回避するため、インポート前に環境変数を設定
os.environ['MPLBACKEND'] = 'Agg'

import sys
import json
import argparse
from pathlib import Path

import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# unified_trainer.pyを直接import
import ztb.training.unified_trainer as ut_module
UnifiedTrainer = ut_module.UnifiedTrainer


def main():
    parser = argparse.ArgumentParser(description='SAC訓練（最適化用）')
    parser.add_argument('--config', type=str, required=True, help='設定ファイルパス')
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        sys.exit(1)
    
    # 設定ファイル読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("🚀 SAC訓練開始")
    print(f"📄 設定: {config_path}")
    print(f"🎯 ステップ数: {config.get('total_timesteps', 'N/A')}")
    
    # SACパラメータ表示
    sac_params = config.get('sac_hyperparameters') or config.get('sac_params', {})
    print("\nSACパラメータ:")
    print(f"  learning_rate: {sac_params.get('learning_rate', 'N/A')}")
    print(f"  batch_size: {sac_params.get('batch_size', 'N/A')}")
    print(f"  gamma: {sac_params.get('gamma', 'N/A')}")
    print(f"  target_update_interval: {sac_params.get('target_update_interval', 'N/A')}")
    print()
    
    try:
        trainer = UnifiedTrainer(config)
        result = trainer.train()
        
        if result:
            print("\n✅ 訓練完了")
            print(f"Model: {result.get('model_path', 'N/A')}")
        else:
            print("\n❌ 訓練失敗")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
