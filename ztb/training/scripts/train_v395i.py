"""
SAC v395i Complete Fix - 5k timesteps
完全修正版: 観測値正規化 + 報酬改善
"""
import json
import time

from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.training_utils import display_training_complete


def main():
    print("🔧 SAC v395i Complete Fix - 5k timesteps")
    print("=" * 60)
    print("✓ Fix 1: Observation normalization (mean=0, std=1)")
    print("✓ Fix 2: Reward improvements (zero rewards 64.3% → 33.3%)")
    print("=" * 60)

    config_path = "configs/sac_v395i_complete_fix.json"

    # 設定ファイル読み込み
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("\n🚀 Starting training...")
    print("Expected improvements:")
    print("  - Critic Loss: < 1000 (was 3.28e+09)")
    print("  - Actor Loss: 0.1 to 100 (was 1e6)")
    print("  - Stable gradient updates")
    print("\n")

    start_time = time.time()
    trainer = UnifiedTrainer(config)
    result = trainer.train()
    training_time = time.time() - start_time

    print("\n" + "=" * 60)
    if result:
        display_training_complete(result, training_time)
        print("\nNext steps:")
        print("1. Check TensorBoard for Critic Loss improvement")
        print("2. If successful, extend to 10k-50k timesteps")
        print("3. Use compare_sac_sessions.py for statistical validation")
    else:
        print("❌ Training failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
