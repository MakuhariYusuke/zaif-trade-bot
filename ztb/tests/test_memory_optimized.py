#!/usr/bin/env python3
"""
Memory-optimized training configuration for Bug #52 fix.

This script creates a minimal-memory training run to verify the system works.
"""

import sys
import logging
import gc
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def main():
    logger.info("=" * 80)
    logger.info("Memory-Optimized Training Test")
    logger.info("=" * 80)
    
    # Step 1: Load SMALL dataset
    logger.info("Step 1: Loading REDUCED dataset...")
    import pandas as pd
    df_full = pd.read_csv("ml-dataset-enhanced-balanced.csv")
    
    # Use only FIRST 200 rows instead of 1000
    df = df_full.head(200).copy()
    del df_full
    gc.collect()
    
    logger.info(f"  Dataset size: {len(df)} rows (reduced from 1000)")
    logger.info(f"  Columns: {len(df.columns)}")
    
    # Step 2: Create environment with MINIMAL features
    logger.info("\nStep 2: Creating environment with MINIMAL configuration...")
    
    from ztb.trading.environment.environment import HeavyTradingEnv
    from sb3_contrib.common.wrappers import ActionMasker
    
    # CRITICAL: Disable correlation reduction to avoid the warning
    # and manually select minimal features
    env_config = {
        "enable_correlation_reduction": False,  # Disable auto-reduction
        "feature_set": "basic",  # Request basic feature set if available
    }
    
    env = HeavyTradingEnv(df=df, config=env_config)
    
    logger.info(f"  Environment features: {len(env.features)}")
    logger.info(f"  Observation space: {env.observation_space}")
    
    def mask_fn(env):
        return env.get_legal_actions().astype(bool)
    
    env = ActionMasker(env, mask_fn)
    
    # Step 3: Create MINIMAL model
    logger.info("\nStep 3: Creating MINIMAL PPO model...")
    
    from sb3_contrib import MaskablePPO
    
    # Extremely small configuration
    policy_kwargs = {
        "net_arch": {"pi": [32, 32], "vf": [32, 32]}  # Tiny network
    }
    
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=64,  # VERY small buffer
        batch_size=16,  # VERY small batch
        n_epochs=2,  # Few epochs
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=42,
    )
    
    logger.info("  Model created successfully")
    
    # Step 4: Train for MINIMAL steps
    logger.info("\nStep 4: Training for 128 timesteps (2 iterations)...")
    logger.info(f"  Expected iterations: 128 / 64 = 2")
    
    import time
    start = time.time()
    
    model.learn(total_timesteps=128, progress_bar=True)
    
    elapsed = time.time() - start
    
    logger.info("=" * 80)
    logger.info("✅ TRAINING COMPLETED!")
    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info(f"  Model num_timesteps: {model.num_timesteps}")
    logger.info(f"  Expected: 128")
    logger.info(f"  Match: {'YES' if model.num_timesteps == 128 else 'NO'}")
    logger.info("=" * 80)
    
    if model.num_timesteps != 128:
        logger.error(f"❌ BUG CONFIRMED: Expected 128 timesteps, got {model.num_timesteps}")
        return False
    
    logger.info("\n✅ Bug #52 appears to be FIXED with memory-optimized configuration!")
    logger.info("\nNext steps:")
    logger.info("  1. Apply these memory optimizations to main training pipeline")
    logger.info("  2. Reduce n_steps from 2048 to 256-512")
    logger.info("  3. Implement proper feature selection to reduce observation space")
    logger.info("  4. Test with 5000 timesteps")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ ERROR: {e}", exc_info=True)
        sys.exit(1)
