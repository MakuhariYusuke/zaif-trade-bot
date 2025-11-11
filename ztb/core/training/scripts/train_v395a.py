"""
v395a (SAC with Auto Entropy Tuning) を訓練
SAC初導入: ent_coef='auto'でエントロピー係数を動的に最適化
"""

import json

from ztb.training.unified_trainer import UnifiedTrainer


def main():
    config_path = "configs/sac_v395a_auto_entropy.json"

    print("=" * 80)
    print("🚀🚀🚀 v395a Training (SAC with AUTO ENTROPY) 🚀🚀🚀")
    print("=" * 80)
    print()
    print(f"Config: {config_path}")
    print()

    print("🎯 Key Innovation:")
    print("  - Algorithm: PPO → SAC (Soft Actor-Critic)")
    print("  - ent_coef: 0.2固定 → 'auto' (動的最適化) 🔥")
    print("  - Off-policy学習でサンプル効率向上")
    print("  - Maximum Entropy frameworkで探索・活用バランス自動調整")
    print()

    print("📊 PPO v394シリーズの課題:")
    print("  - ent_coef固定値では訓練後期に探索不足")
    print("  - v394d (ent_coef=0.01): HOLD 89%, entropy=0.61 🚨")
    print("  - v394f (ent_coef=0.2): 探索過多で学習不安定リスク ⚠️")
    print()

    print("🎯 SAC v395a Expected Advantages:")
    print("  - ✅ 訓練初期: 高探索でaction空間を広く学習")
    print("  - ✅ 訓練中期: 探索を徐々に減らし、効果的な戦略に収束")
    print("  - ✅ 訓練後期: 適度な探索を維持、過学習を防止")
    print("  - ✅ HOLD比率: 40-60%で安定する可能性")
    print()

    print("🔬 Technical Details:")
    print("  - buffer_size: 50000 (過去の経験を再利用)")
    print("  - learning_starts: 1000 (十分なデータ収集後に学習開始)")
    print("  - ent_coef='auto': alpha (entropy coefficient) を自動調整")
    print("  - target_entropy='auto': -dim(A) に自動設定（3 actions → -3.0）")
    print()

    print("🚀 Starting SAC training...")
    print()

    # 設定ファイル読み込み
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    trainer = UnifiedTrainer(config)
    result = trainer.train()

    print(f"\n{'='*80}")
    print("=== Training Complete ===")
    print(f"{'='*80}")
    if result:
        print("✅ Training successful!")
        print(f"Model saved to: {result.get('model_path', 'N/A')}")
        print(f"Logs saved to: {result.get('log_path', 'N/A')}")
    else:
        print("❌ Training failed or was interrupted")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
