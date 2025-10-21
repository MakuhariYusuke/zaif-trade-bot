#!/usr/bin/env python3
"""
Memory profiling script for Bug #52 investigation.

Monitors memory usage at each step to identify memory leaks or excessive allocation.
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
    return cast(float, mem)


def log_memory(step_name: str) -> float:
    """Log current memory usage."""
    mem = get_memory_usage()
    logger.info(f"  💾 Memory: {mem:.1f} MB - {step_name}")
    return mem


def main() -> bool:
    logger.info("=" * 80)
    logger.info("Bug #52 Memory Profiling")
    logger.info("=" * 80)

    start_mem = log_memory("Script start")

    try:
        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        import pandas as pd

        df = pd.read_csv("ml-dataset-enhanced-balanced.csv")
        logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        mem_after_data = log_memory("After data load")
        logger.info(f"  Memory increase: {mem_after_data - start_mem:.1f} MB")

        # Step 2: Create environment (THIS IS SLOW - 8 seconds!)
        logger.info("\nStep 2: Creating environment...")
        logger.info(
            "  ⚠️  This step takes ~8 seconds - environment initialization is slow"
        )

        from sb3_contrib.common.wrappers import ActionMasker

        from ztb.trading.environment.environment import HeavyTradingEnv

        env_config = {
            # Use minimal features to reduce memory
            "feature_config": {
                "use_technical_indicators": False,  # Disable if possible
                "use_orderbook_features": False,
            }
        }

        env = HeavyTradingEnv(df=df, config=env_config)
        mem_after_env = log_memory("After environment creation")
        logger.info(f"  Memory increase: {mem_after_env - mem_after_data:.1f} MB")

        def mask_fn(env: Any) -> Any:
            return env.get_legal_actions().astype(bool)

        env = ActionMasker(env, mask_fn)
        mem_after_wrapper = log_memory("After ActionMasker wrapper")

        # Step 3: Create model
        logger.info("\nStep 3: Creating MaskablePPO model...")
        logger.info("  Using MINIMAL configuration to reduce memory:")
        logger.info("    - n_steps=128 (instead of 2048)")
        logger.info("    - batch_size=32 (instead of 64)")
        logger.info("    - policy_kwargs with small network")

        from sb3_contrib import MaskablePPO

        # Use smaller network to reduce memory
        policy_kwargs = {
            "net_arch": [dict(pi=[32, 32], vf=[32, 32])]
        }  # Much smaller than default

        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=128,  # Reduced from 2048
            batch_size=32,  # Reduced from 64
            n_epochs=4,  # Reduced from 10
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=42,
        )

        mem_after_model = log_memory("After model creation")
        logger.info(f"  Memory increase: {mem_after_model - mem_after_wrapper:.1f} MB")

        # Step 4: Test rollout collection
        logger.info("\nStep 4: Testing rollout collection (128 steps)...")
        model.env.reset()

        # Manually collect rollout to monitor memory
        start_rollout_mem = get_memory_usage()

        model.collect_rollouts(
            env=model.env,
            callback=None,
            rollout_buffer=model.rollout_buffer,
            n_rollout_steps=model.n_steps,
        )

        mem_after_rollout = log_memory("After rollout collection")
        logger.info(
            f"  Memory increase: {mem_after_rollout - start_rollout_mem:.1f} MB"
        )

        # Step 5: Test VERY short training
        logger.info("\nStep 5: Testing MINIMAL training (256 steps = 2 rollouts)...")
        logger.info(f"  Current memory: {get_memory_usage():.1f} MB")
        logger.info(
            f"  Available memory: {psutil.virtual_memory().available / 1024 / 1024:.1f} MB"
        )

        # Force garbage collection before training
        gc.collect()
        mem_before_train = log_memory("Before training (after GC)")

        start_time = time.time()

        # Train for only 256 steps (2 iterations)
        model.learn(total_timesteps=256, progress_bar=False)

        elapsed = time.time() - start_time
        mem_after_train = log_memory("After training")

        logger.info("=" * 80)
        logger.info("✅ TRAINING COMPLETED")
        logger.info(f"  Time: {elapsed:.2f}s")
        logger.info(f"  Model timesteps: {model.num_timesteps}")
        logger.info(
            f"  Memory increase during training: {mem_after_train - mem_before_train:.1f} MB"
        )
        logger.info(f"  Total memory used: {mem_after_train:.1f} MB")
        logger.info("=" * 80)

        # Memory summary
        logger.info("\n📊 MEMORY SUMMARY:")
        logger.info(f"  Initial:              {start_mem:.1f} MB")
        logger.info(
            f"  After data load:      {mem_after_data:.1f} MB (+{mem_after_data - start_mem:.1f} MB)"
        )
        logger.info(
            f"  After environment:    {mem_after_env:.1f} MB (+{mem_after_env - mem_after_data:.1f} MB)"
        )
        logger.info(
            f"  After model:          {mem_after_model:.1f} MB (+{mem_after_model - mem_after_env:.1f} MB)"
        )
        logger.info(
            f"  After training:       {mem_after_train:.1f} MB (+{mem_after_train - mem_after_model:.1f} MB)"
        )
        logger.info(f"  Total increase:       {mem_after_train - start_mem:.1f} MB")

        return True

    except MemoryError as e:
        logger.error("=" * 80)
        logger.error("❌ MEMORY ERROR!")
        logger.error(f"  {e}")
        logger.error(f"  Current memory: {get_memory_usage():.1f} MB")
        logger.error(
            f"  System memory available: {psutil.virtual_memory().available / 1024 / 1024:.1f} MB"
        )
        logger.error("=" * 80)
        return False

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ ERROR: {e}")
        logger.error(f"  Current memory: {get_memory_usage():.1f} MB")
        logger.error("=" * 80)
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        logger.warning(f"  Final memory: {get_memory_usage():.1f} MB")
        sys.exit(1)
