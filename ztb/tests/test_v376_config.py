"""
Quick validation script to test ppo_improved_v376.json configuration
"""
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.utils.config import EnvironmentConfig

def main():
    config_path = project_root / "configs" / "training" / "ppo_improved_v376.json"
    
    print("=" * 80)
    print("Testing ppo_improved_v376.json Configuration")
    print("=" * 80)
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("\n1. Top-level Configuration:")
    print(f"   Session ID: {config.get('session_id')}")
    print(f"   Data rows limit: {config.get('data_rows_limit')} (None = use all)")
    print(f"   Max features: {config.get('max_features')} (None = use all with correlation filtering)")
    print(f"   Correlation reduction: {config.get('enable_correlation_reduction')}")
    
    print("\n2. PPO Settings:")
    ppo = config.get('ppo', {})
    print(f"   Learning rate: {ppo.get('learning_rate')}")
    print(f"   N steps: {ppo.get('n_steps')}")
    print(f"   Entropy coef: {ppo.get('ent_coef')}")
    print(f"   Target KL: {ppo.get('target_kl')}")
    
    print("\n3. Environment Settings:")
    env = config.get('environment', {})
    print(f"   Random start: {env.get('random_start')}")
    print(f"   Transaction cost: {env.get('transaction_cost')}")
    print(f"   Action masking: {env.get('enable_action_masking')}")
    
    reward_settings = env.get('reward_settings', {})
    print(f"\n4. Reward Settings:")
    print(f"   HOLD penalty: {reward_settings.get('hold_penalty_weight')}")
    print(f"   Consecutive HOLD penalty: {reward_settings.get('consecutive_hold_penalty')}")
    print(f"   Trading frequency bonus: {reward_settings.get('trading_frequency_bonus')}")
    print(f"   Profit multiplier: {reward_settings.get('profit_reward_multiplier')}")
    
    print("\n5. Training Settings:")
    training = config.get('training', {})
    print(f"   Total timesteps: {training.get('total_timesteps')}")
    print(f"   Eval frequency: {training.get('eval_freq')}")
    print(f"   Eval episodes: {training.get('n_eval_episodes')}")
    print(f"   Checkpoint interval: {training.get('checkpoint_interval')}")
    
    print("\n6. Custom PPO Settings:")
    custom = config.get('custom_ppo', {})
    print(f"   Enabled: {custom.get('enabled')}")
    print(f"   Use PAN: {custom.get('use_pan')}")
    print(f"   Use target entropy: {custom.get('use_target_entropy')}")
    print(f"   Target entropy value: {custom.get('target_entropy')}")
    
    # Test EnvironmentConfig creation
    print("\n7. Testing EnvironmentConfig Creation:")
    try:
        env_config = EnvironmentConfig.from_dict(env)
        print(f"   ✅ EnvironmentConfig created successfully")
        print(f"   Random start: {env_config.random_start}")
        print(f"   Transaction cost: {env_config.transaction_cost}")
        print(f"   Reward settings: {env_config.reward_settings}")
    except Exception as e:
        print(f"   ❌ Error creating EnvironmentConfig: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Configuration Validation Complete")
    print("=" * 80)
    
    # Verify key improvements
    print("\n8. Key Improvements Verification:")
    checks = [
        ("Data limit removed", config.get('data_rows_limit') is None),
        ("Features expanded", config.get('max_features') is None),
        ("Random start enabled", env.get('random_start') == True),
        ("Longer training", training.get('total_timesteps', 0) >= 20000),
        ("Frequent eval", training.get('eval_freq', 0) <= 1000),
        ("Entropy boosted", ppo.get('ent_coef', 0) >= 0.06),
        ("Target entropy set", custom.get('target_entropy') is not None),
    ]
    
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
    
    all_passed = all(passed for _, passed in checks)
    if all_passed:
        print("\n✅ All improvement checks passed!")
    else:
        print("\n⚠️  Some checks failed - review configuration")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
