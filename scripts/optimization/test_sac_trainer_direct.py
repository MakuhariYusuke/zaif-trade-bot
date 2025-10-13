#!/usr/bin/env python3
"""
SAC Trainer 直接テスト
メトリクス出力が正しく動作するか確認
"""
import os
os.environ['MPLBACKEND'] = 'Agg'

import sys
import json
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 必要最小限のインポート
print("Loading SAC Trainer...")

try:
    from ztb.training.trainers.sac_trainer import SACAlgorithmTrainer
    from ztb.training.core.config_manager import ConfigManager
    print("✅ SAC Trainer loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def main():
    # 設定ファイル読み込み
    config_path = project_root / 'configs' / 'sac_test_1ksteps.json'
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"📄 Config loaded: {config_path}")
    print(f"🎯 Timesteps: {config.get('total_timesteps', 'N/A')}")
    
    # ConfigManager作成
    config_manager = ConfigManager(config)
    
    # SACAlgorithmTrainer作成
    trainer = SACAlgorithmTrainer(config_manager)
    
    print("\n🚀 Starting training...")
    
    try:
        result = trainer.train(config)
        
        print("\n✅ Training completed successfully!")
        print(f"Model: {result.get('model_path', 'N/A')}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
