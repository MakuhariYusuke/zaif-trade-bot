"""Test basic PPO training without SELL mitigation."""

import json

from ztb.training.unified_trainer import UnifiedTrainer

# Load test config
with open("test_basic_ppo_config.json", "r") as f:
    config = json.load(f)

print("=" * 60)
print("Step 2: Basic PPO Training Test (No SELL Mitigation)")
print("=" * 60)
print(f'Data: {config["data_path"]}')
print(f'Timesteps: {config["total_timesteps"]}')
print(f'SELL mitigation: {config["enable_sell_mitigation"]}')
print("=" * 60)
print()

# Create trainer and run short test
trainer = UnifiedTrainer(config)
print("Starting basic PPO training...")

try:
    model = trainer.train()
    print()
    print("=" * 60)
    print("✅ SUCCESS: Basic PPO training completed!")
    print("=" * 60)
    print(f"Model type: {type(model)}")
    print("Model saved to: models_test/basic_ppo_test.zip")

except Exception as e:
    print()
    print("=" * 60)
    print(f"❌ FAILED: {e}")
    print("=" * 60)
    import traceback

    traceback.print_exc()
