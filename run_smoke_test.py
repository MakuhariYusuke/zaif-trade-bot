#!/usr/bin/env python3
"""
10k Smoke Test for 4 High-Impact Action Bias Mitigations.

Tests the integration of:
1. PAN (Per-Action Advantage Normalization)
2. Target Entropy Controller
3. Reverse-as-Close Flag (allow_reverse=False)
4. Stratified Mini-batch Sampler

Validation criteria:
- 合法SELL率 ≥12% (moving window 5k)
- grad_norm(SELL) > 0 maintained
- No crashes, normal convergence
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.sell_mitigation_ppo_trainer import SELLBiasMitigationPPOTrainer
from ztb.training.ppo_config import PPOConfig
from ztb.training.trainer_params import SELLMitigationParams


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_smoke_test(config_path: str = "smoke_test_10k_config.json"):
    """Run 10k smoke test with all 4 mitigations enabled."""
    print("\n" + "="*80)
    print("10k SMOKE TEST - 4 High-Impact Action Bias Mitigations")
    print("="*80 + "\n")
    
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config_dict = load_config(config_path)
    
    # Extract settings
    data_path = config_dict["data_path"]
    checkpoint_dir = config_dict["output"]["checkpoint_dir"]
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Create PPO config with environment settings merged
    ppo_config = PPOConfig(
        total_timesteps=config_dict["total_timesteps"],
        learning_rate=config_dict["learning_rate"],
        n_steps=config_dict["n_steps"],
        batch_size=config_dict["batch_size"],
        gamma=config_dict["gamma"],
        ent_coef=config_dict["ent_coef"],
        vf_coef=config_dict["vf_coef"],
        max_grad_norm=config_dict["max_grad_norm"],
        clip_range=config_dict["clip_range"],
        gae_lambda=config_dict["gae_lambda"],
        seed=config_dict["seed"],
        tensorboard_log=config_dict["output"]["tensorboard_log"],
    )
    
    # Merge environment settings into ppo_config
    env_config = config_dict["environment"]
    ppo_config["curriculum_stage"] = env_config["curriculum_stage"]
    ppo_config["transaction_cost"] = env_config["transaction_cost"]
    ppo_config["max_position_size"] = env_config["max_position_size"]
    ppo_config["risk_free_rate"] = env_config.get("risk_free_rate", 0.0)
    
    print("\nConfiguration:")
    print(f"  Total timesteps: {config_dict['total_timesteps']:,}")
    print(f"  Data: {data_path}")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Seed: {config_dict['seed']}")
    
    print("\nMitigation features:")
    mitigation = config_dict["mitigation"]
    for key, value in mitigation.items():
        status = "✓ Enabled" if value else "✗ Disabled"
        feature_name = key.replace("enable_", "").replace("_", " ").title()
        print(f"  {feature_name}: {status}")
    
    print("\nEnvironment settings:")
    env_config = config_dict["environment"]
    print(f"  Transaction cost: {env_config['transaction_cost']}")
    print(f"  Max position size: {env_config['max_position_size']}")
    print(f"  Allow reverse: {env_config['allow_reverse']}")
    print(f"  Curriculum stage: {env_config['curriculum_stage']}")
    
    print("\nValidation criteria:")
    validation = config_dict["validation"]
    print(f"  Target SELL rate (min): {validation['target_sell_rate_min']*100:.1f}%")
    print(f"  Target SELL rate (optimal): {validation['target_sell_rate_optimal']*100:.1f}%")
    print(f"  Gradient norm threshold: {validation['grad_norm_threshold']}")
    print(f"  Moving window: {validation['moving_window']:,} steps")
    
    print("\n" + "-"*80)
    print("Starting training...")
    print("-"*80 + "\n")
    
    try:
        # Create mitigation parameters
        mitigation_params = SELLMitigationParams(
            data_path=data_path,
            config=ppo_config,
            checkpoint_dir=checkpoint_dir,
            enable_lagrange=mitigation["enable_lagrange"],
            enable_probes=mitigation["enable_probes"],
            enable_weights=mitigation["enable_weights"],
            enable_pan=mitigation["enable_pan"],
            enable_target_entropy=mitigation["enable_target_entropy"],
            enable_stratified_sampling=mitigation["enable_stratified_sampling"],
            allow_reverse=env_config["allow_reverse"],
            probe_csv_path=config_dict["output"]["probe_csv"],
        )
        
        # Create trainer with unified params interface
        trainer = SELLBiasMitigationPPOTrainer(params=mitigation_params)
        
        print("✓ Trainer initialized successfully\n")
        
        # Run training
        print("Training in progress...")
        session_id = f"smoke_test_10k_seed{config_dict['seed']}"
        model = trainer.train(session_id=session_id)
        
        print("\n" + "="*80)
        print("Training completed!")
        print("="*80 + "\n")
        
        # Analyze results
        print("Analyzing results...")
        analyze_results(checkpoint_dir, config_dict["output"]["probe_csv"], validation)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_results(checkpoint_dir: str, probe_csv: str, validation: dict):
    """Analyze smoke test results."""
    print("\n" + "-"*80)
    print("RESULTS ANALYSIS")
    print("-"*80 + "\n")
    
    # Check if probe CSV exists
    probe_path = Path(probe_csv)
    if not probe_path.exists():
        print("⚠ Warning: Probe CSV not found. Cannot validate SELL metrics.")
        print(f"  Expected: {probe_csv}")
        return
    
    # Load probe data
    try:
        probe_df = pd.read_csv(probe_csv)
        print(f"✓ Loaded probe data: {len(probe_df)} records\n")
        
        # Check columns
        if len(probe_df) == 0:
            print("⚠ Warning: Probe data is empty")
            return
        
        print("Probe data columns:", list(probe_df.columns))
        
        # Analyze SELL rate
        if "sell_rate" in probe_df.columns:
            sell_rates = probe_df["sell_rate"].dropna()
            if len(sell_rates) > 0:
                mean_sell_rate = sell_rates.mean()
                final_sell_rate = sell_rates.iloc[-1] if len(sell_rates) > 0 else 0.0
                
                print(f"\nSELL Rate Analysis:")
                print(f"  Mean SELL rate: {mean_sell_rate*100:.2f}%")
                print(f"  Final SELL rate: {final_sell_rate*100:.2f}%")
                print(f"  Target (min): {validation['target_sell_rate_min']*100:.1f}%")
                print(f"  Target (optimal): {validation['target_sell_rate_optimal']*100:.1f}%")
                
                if final_sell_rate >= validation["target_sell_rate_min"]:
                    print(f"  ✓ PASSED: SELL rate meets minimum target")
                else:
                    print(f"  ✗ FAILED: SELL rate below minimum target")
        
        # Analyze gradient norms
        if "grad_norm" in probe_df.columns:
            grad_norms = probe_df["grad_norm"].dropna()
            if len(grad_norms) > 0:
                healthy_grads = (grad_norms > validation["grad_norm_threshold"]).sum()
                healthy_ratio = healthy_grads / len(grad_norms)
                
                print(f"\nGradient Health Analysis:")
                print(f"  Healthy gradients: {healthy_grads}/{len(grad_norms)} ({healthy_ratio*100:.1f}%)")
                print(f"  Threshold: {validation['grad_norm_threshold']}")
                
                if healthy_ratio > 0.8:
                    print(f"  ✓ PASSED: Gradient flow maintained")
                else:
                    print(f"  ⚠ WARNING: Gradient flow degraded")
        
        # Summary
        print("\n" + "="*80)
        print("SMOKE TEST SUMMARY")
        print("="*80)
        print("\nImplemented features:")
        print("  ✓ PAN (Per-Action Advantage Normalization)")
        print("  ✓ Target Entropy Controller")
        print("  ✓ Reverse-as-Close Flag (allow_reverse=False)")
        print("  ✓ Stratified Mini-batch Sampler")
        print("\nNext steps:")
        print("  1. Review TensorBoard logs for detailed metrics")
        print(f"  2. Check probe CSV for full gradient history: {probe_csv}")
        print("  3. If results are good, proceed to 50k validation")
        print("  4. If SELL rate still low, consider hyperparameter tuning")
        
    except Exception as e:
        print(f"⚠ Error analyzing results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run 10k smoke test")
    parser.add_argument(
        "--config",
        type=str,
        default="smoke_test_10k_config.json",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    success = run_smoke_test(args.config)
    sys.exit(0 if success else 1)
