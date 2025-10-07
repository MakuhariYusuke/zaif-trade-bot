#!/usr/bin/env python3
"""
Integration test for 4 high-impact modifications.

Tests that all 4 features are correctly integrated:
1. PAN (Per-Action Advantage Normalization)
2. Target Entropy Controller
3. Reverse-as-Close Flag (via environment config)
4. Stratified Mini-batch Sampler

This test does NOT run a full training loop, only verifies initialization
and basic functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.sell_mitigation_ppo_trainer import SELLBiasMitigationPPOTrainer
from ztb.training.ppo_trainer import PPOConfig


def test_integration():
    """Test that all 4 features initialize correctly."""
    print("\n=== 4 High-Impact Modifications Integration Test ===\n")
    
    # Create trainer with all features enabled
    print("Creating trainer with all 4 features enabled...")
    
    # Mock data path (not used for initialization test)
    data_path = "ml-dataset-enhanced.csv"
    checkpoint_dir = "./test_checkpoints"
    
    # Create configuration
    config = PPOConfig(
        total_timesteps=1000,  # Minimal for testing
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    
    try:
        trainer = SELLBiasMitigationPPOTrainer(
            data_path=data_path,
            config=config,
            checkpoint_dir=checkpoint_dir,
            # Original features
            enable_lagrange=True,
            enable_probes=True,
            enable_weights=True,
            # New: 4 high-impact modifications
            enable_pan=True,
            enable_target_entropy=True,
            enable_stratified_sampling=True,
            allow_reverse=False,  # Reverse-as-Close mode
        )
        
        print("✓ Trainer created successfully\n")
        
        # Check components initialization
        print("Checking component initialization:")
        
        # 1. Lagrange (existing)
        assert trainer.lagrange is not None, "Lagrange should be initialized"
        print("  ✓ Lagrange constraint initialized")
        
        # 2. Probes (existing)
        assert trainer.probe is not None, "Probe should be initialized"
        print("  ✓ Gradient probes initialized")
        
        # 3. Weights (existing)
        assert trainer.weight_calc is not None, "Weight calc should be initialized"
        print("  ✓ Action weighting initialized")
        
        # 4. PAN (NEW)
        assert trainer.pan_normalizer is not None, "PAN should be initialized"
        print("  ✓ PAN (Per-Action Advantage Normalization) initialized")
        
        # 5. Target Entropy (NEW)
        assert trainer.entropy_controller is not None, "Entropy controller should be initialized"
        print("  ✓ Target Entropy Controller initialized")
        
        # 6. Stratified Sampler (NEW)
        assert trainer.stratified_sampler is not None, "Stratified sampler should be initialized"
        print("  ✓ Stratified Mini-batch Sampler initialized")
        
        # 7. allow_reverse flag (NEW)
        assert trainer.allow_reverse == False, "allow_reverse should be False"
        print("  ✓ Reverse-as-Close flag set to False")
        
        print("\n=== All 4 modifications successfully integrated! ===")
        print("\nFeature Summary:")
        print("1. PAN: Prevents gradient crushing of minority actions")
        print("2. Target Entropy: Automatic exploration maintenance")
        print("3. Reverse-as-Close: Reduces SELL cost perception (allow_reverse=False)")
        print("4. Stratified Sampler: Boosts minority scenarios in batches")
        print("\n✅ Integration test PASSED!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
