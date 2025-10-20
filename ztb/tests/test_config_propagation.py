"""Test configuration propagation"""
import json

# Load config
config = json.load(open("configs/training/ppo_mem_opt_v370.json"))

# Simulate wrapped_config creation
ppo_config = {
    "learning_rate": config.get("learning_rate", 0.0003),
    "n_steps": config.get("n_steps", 2048),
    "total_timesteps": config.get("total_timesteps", 1000000),
    "data_rows_limit": config.get("data_rows_limit"),
    "max_features": config.get("max_features"),
}

wrapped_config = {
    "ppo": ppo_config,
    # Preserve top-level settings
    **{
        k: v for k, v in config.items() if k not in ppo_config and not k.startswith("_")
    },
}

print("=== Original config ===")
print(f"data_rows_limit: {config.get('data_rows_limit')}")
print(f"max_features: {config.get('max_features')}")

print("\n=== PPO config ===")
print(f"data_rows_limit: {ppo_config.get('data_rows_limit')}")
print(f"max_features: {ppo_config.get('max_features')}")

print("\n=== Wrapped config (top level) ===")
print(f"data_rows_limit: {wrapped_config.get('data_rows_limit')}")
print(f"max_features: {wrapped_config.get('max_features')}")

print("\n=== Wrapped config keys ===")
print(f"Top-level keys (first 20): {list(wrapped_config.keys())[:20]}")
print(f"Has 'ppo' key: {'ppo' in wrapped_config}")
print(f"Has 'data_rows_limit': {'data_rows_limit' in wrapped_config}")
