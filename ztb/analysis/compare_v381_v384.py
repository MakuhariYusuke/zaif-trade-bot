#!/usr/bin/env python3
"""Compare v381 (110 features) vs v384 (68 curated features)."""

import json
from pathlib import Path

# Load configurations
v381_config = json.loads(Path('configs/training/ppo_reward_v381_revised_profit_focused.json').read_text())
v384_config = json.loads(Path('configs/training/ppo_reward_v384_curated_60.json').read_text())

print("=" * 80)
print("v381 (BASELINE - All 110 Features) vs v384 (CURATED - 68 Features)")
print("=" * 80)

print("\n=== v381 Configuration ===")
print(f"Features: {v381_config.get('curated_features_list', 'ALL 110 features')}")
print(f"Feature filtering: {v381_config.get('enable_feature_filtering', False)}")
print(f"Total timesteps: {v381_config.get('total_timesteps', v381_config.get('ppo', {}).get('total_timesteps', 'N/A'))}")
print(f"Learning rate: {v381_config.get('ppo', {}).get('learning_rate', 'N/A')}")
print(f"VF coef: {v381_config.get('ppo', {}).get('vf_coef', 'N/A')}")
print(f"Target KL: {v381_config.get('ppo', {}).get('target_kl', 'N/A')}")

print("\n=== v384 Configuration ===")
print(f"Features: {v384_config.get('curated_features_list', 'N/A')}")
print(f"Feature filtering: {v384_config.get('enable_feature_filtering', False)}")
print(f"Feature filter mode: {v384_config.get('feature_filter_mode', 'N/A')}")
print(f"Total timesteps: {v384_config.get('total_timesteps', v384_config.get('ppo', {}).get('total_timesteps', 'N/A'))}")
print(f"Learning rate: {v384_config.get('ppo', {}).get('learning_rate', 'N/A')}")
print(f"VF coef: {v384_config.get('ppo', {}).get('vf_coef', 'N/A')}")
print(f"Target KL: {v384_config.get('ppo', {}).get('target_kl', 'N/A')}")

print("\n=== Key Differences ===")
print("1. Feature Count:")
print("   v381: 110 features (all features, including redundant ones)")
print("   v384: 68 features (curated, removed 42 redundant/correlated features)")
print("\n2. Removed Features (42):")
print("   - HeikinAshi OHLC (4): Color sequence is sufficient")
print("   - Time constants (5): Zero variance")
print("   - Ichimoku individual spans (5): Composites more meaningful")
print("   - High correlation pairs (20): Redundant information")
print("   - Training labels (2): pnl, win")
print("   - Other redundant indicators (6)")
print("\n3. Training Parameters:")
print("   Both use identical hyperparameters (lr=0.003, vf_coef=0.3, target_kl=0.01)")

print("\n=== Expected Benefits of v384 ===")
print("✓ Faster training (fewer features to process)")
print("✓ Better generalization (less overfitting to redundant features)")
print("✓ Clearer feature importance (no correlated noise)")
print("✓ More efficient model (smaller input space)")

print("\n=== Analysis Steps ===")
print("1. Open TensorBoard: http://localhost:6006")
print("2. Compare metrics:")
print("   - rollout/ep_rew_mean: Average episode reward")
print("   - train/approx_kl: KL divergence (should be ~0.07)")
print("   - train/loss: Total loss")
print("   - train/value_loss: Value function loss")
print("   - train/policy_gradient_loss: Policy loss")
print("3. Check action distribution (pan_action_counts)")
print("4. Evaluate final models on test data")

print("\n" + "=" * 80)
