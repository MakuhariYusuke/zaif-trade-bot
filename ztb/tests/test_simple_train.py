#!/usr/bin/env python3
"""
Simplified training test to identify Bug #52 root cause.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    logger.info("=" * 80)
    logger.info("Simplified Training Test - Bug #52 Investigation")
    logger.info("=" * 80)
    
    # Load data
    logger.info("Step 1: Loading data...")
    import pandas as pd
    df = pd.read_csv("ml-dataset-enhanced-balanced.csv")
    logger.info(f"  Data loaded: {len(df)} rows")
    
    # Create environment
    logger.info("Step 2: Creating environment...")
    from ztb.trading.environment.environment import HeavyTradingEnv
    from sb3_contrib.common.wrappers import ActionMasker
    
    env_config = {}  # Minimal config to isolate the issue
    
    env = HeavyTradingEnv(df=df, config=env_config)
    
    def mask_fn(env):
        return env.get_legal_actions().astype(bool)
    
    env = ActionMasker(env, mask_fn)
    logger.info("  Environment created")
    
    # Create model
    logger.info("Step 3: Creating PPO model...")
    from sb3_contrib import MaskablePPO
    
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        seed=42,
    )
    logger.info("  Model created")
    
    # Train for 5000 steps
    logger.info("Step 4: Training for 5000 timesteps...")
    logger.info("  Expected iterations: 5000 / 2048 ≈ 2.44 → 3 iterations")
    
    model.learn(total_timesteps=5000)
    
    logger.info("=" * 80)
    logger.info("✅ Training completed successfully!")
    logger.info(f"   Model num_timesteps: {model.num_timesteps}")
    logger.info("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
