"""
Quick training test for ppo_improved_v376
Test with just 2000 timesteps to verify everything works before full 30k run
"""
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    # Load and modify config for quick test
    config_path = project_root / "configs" / "training" / "ppo_improved_v376.json"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Modify for quick test
    test_session_id = 'ppo_v376_quick_test'
    config['session_id'] = test_session_id
    config['training']['total_timesteps'] = 2000  # Quick test
    config['training']['eval_freq'] = 1000
    config['training']['checkpoint_interval'] = 2000
    
    print("=" * 80)
    print("Quick Test: ppo_improved_v376 (2000 timesteps)")
    print("=" * 80)
    print(f"Data rows: {config.get('data_rows_limit', 'ALL')} (expecting ~1000 rows)")
    print(f"Max features: {config.get('max_features', 'ALL with correlation filter')} (expecting ~90-110)")
    print(f"Random start: {config['environment']['random_start']}")
    print(f"Entropy coef: {config['ppo']['ent_coef']}")
    print(f"Target entropy: {config['custom_ppo']['target_entropy']}")
    print("=" * 80 + "\n")
    
    # Import trainer
    from ztb.training.ppo_trainer import PPOTrainer
    
    # Prepare params dict for trainer
    params = {
        'data_path': str(config['data_path']),
        'checkpoint_dir': f'checkpoints/{test_session_id}',
        'checkpoint_interval': config['training']['checkpoint_interval'],
        'config': config,
    }
    
    # Create trainer
    print("Creating trainer...")
    trainer = PPOTrainer(params=params)
    
    # Train
    print("Starting training...")
    trainer.train(session_id=test_session_id)
    
    print("\n" + "=" * 80)
    print("Quick test completed successfully!")
    print("Ready to run full 30k timestep training")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
