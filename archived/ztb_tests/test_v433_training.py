import json
import sys

# Add ztb to path
sys.path.insert(0, "ztb")

from ztb.training.unified_trainer import UnifiedTrainer

# Load config directly from JSON
config_path = "ztb/configs/v433/sac_v433_production_migration.json"
with open(config_path, "r") as f:
    config = json.load(f)

print("Config loaded successfully")
print("Data path:", config.get("data_path", "NOT_FOUND"))
print(
    "Total timesteps:", config.get("training", {}).get("total_timesteps", "NOT_FOUND")
)

# Override total_timesteps for testing
config["training"]["total_timesteps"] = 1000

# Create trainer
trainer = UnifiedTrainer(config)
print("Trainer created successfully")

# Run training
success = trainer.run()
print(f"Training completed: {success}")
