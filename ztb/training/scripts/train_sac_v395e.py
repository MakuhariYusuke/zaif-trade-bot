"""
SAC v395e - CRITICAL FIX: Positive target_entropy
連続行動空間では正のエントロピー目標が必要
"""
import json
from ztb.training.unified_trainer import UnifiedTrainer

def main():
    print("🔧 SAC v395e - Critical Entropy Fix")
    print("=" * 80)
    
    config_path = "configs/sac_v395e_positive_entropy.json"
    
    # 設定ファイル読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("🚨 Critical Insight:")
    print("-" * 80)
    insight = config['critical_insight']
    print(f"  Problem: {insight['problem']}")
    print(f"  Root Cause: {insight['root_cause']}")
    print()
    print("  Solution:")
    for sol in insight['solution']:
        print(f"    • {sol}")
    print()
    
    print("🎯 Expected Behavior:")
    print("-" * 80)
    for key, value in insight['expected_behavior'].items():
        print(f"  • {key:20s}: {value}")
    print()
    
    print("🚀 Starting 5k timesteps training...")
    print("=" * 80)
    trainer = UnifiedTrainer(config)
    result = trainer.train()
    
    print("\n" + "=" * 80)
    if result:
        print("✅ Training completed!")
        print(f"Model saved to: {result.get('model_path', 'N/A')}")
        print("\n📊 Expected Improvements:")
        print("  • ent_coef should be 0.5-1.5 (not 3-4)")
        print("  • ent_coef_loss should fluctuate around 0")
        print("  • Critic/Actor loss should remain stable")
    else:
        print("❌ Training failed")
    print("=" * 80)

if __name__ == "__main__":
    main()
