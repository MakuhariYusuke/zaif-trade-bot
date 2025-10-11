"""
SAC v395a Test - 1k timesteps quick iteration
連続行動空間での動作確認用
"""
import json
from ztb.training.unified_trainer import UnifiedTrainer

def main():
    print("🧪 SAC v395a Test - 1k timesteps quick iteration")
    print("=" * 60)
    
    config_path = "configs/sac_v395a_test_1k.json"
    
    # 設定ファイル読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("🚀 Starting 1k timesteps test run...")
    trainer = UnifiedTrainer(config)
    result = trainer.train()
    
    print("\n" + "=" * 60)
    if result:
        print("✅ Test run completed!")
        print(f"Model saved to: {result.get('model_path', 'N/A')}")
    else:
        print("❌ Test run failed")
    print("=" * 60)

if __name__ == "__main__":
    main()
