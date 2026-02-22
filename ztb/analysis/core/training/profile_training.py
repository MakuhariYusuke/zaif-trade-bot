#!/usr/bin/env python3
"""
Detailed profiling script to identify Bug #52 root cause.

This script measures execution time at each step of the training process
to pinpoint where the hang/slowdown occurs.
"""

import logging
import signal
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any, Dict, Optional, Self

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str):
        self.name = name
        self.start_time: Optional[float] = None

    def __enter__(self) -> Self:
        self.start_time = time.time()
        logger.info(f"⏱️  {self.name} - START")
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = time.time() - self.start_time  # type: ignore
        logger.info(f"✅ {self.name} - COMPLETED in {elapsed:.2f}s")


class TimeoutException(Exception):
    """Raised when operation times out."""

    pass


def timeout_handler(signum: int, frame: Optional[FrameType]) -> None:
    """Signal handler for timeout."""
    raise TimeoutException("Operation timed out")


def profile_training(
    data_path: str = "ml-dataset-enhanced-balanced.csv",
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Profile training execution time.

    Args:
        data_path: Path to the training data
        config: Environment configuration

    Returns:
        True if profiling completed successfully, False otherwise
    """
    if config is None:
        config = {}

    logger.info("=" * 80)
    logger.info("Bug #52 Profiling - Detailed Execution Time Analysis")
    logger.info("=" * 80)

    # NOTE: Windows doesn't support SIGALRM, so no timeout mechanism
    logger.info(
        "⚠️  No timeout mechanism on Windows - use Ctrl+C to interrupt if needed"
    )

    try:
        # Step 1: Load data
        with Timer("Step 1: Load data"):
            from ztb.io.data_loader import DataLoader

            df = DataLoader.load_csv_strict(data_path)
            logger.info(f"  Loaded {len(df)} rows")

        # Step 2: Create environment
        with Timer("Step 2: Create environment"):
            from sb3_contrib.common.wrappers import ActionMasker

            from ztb.trading.environment.environment import HeavyTradingEnv

            env_config: Dict[str, Any] = config  # Use provided config
            env = HeavyTradingEnv(df=df, config=env_config)

            def mask_fn(env: Any) -> Any:
                return env.get_legal_actions().astype(bool)

            env = ActionMasker(env, mask_fn)
            logger.info("  Environment wrapped with ActionMasker")

        # Step 3: Test environment reset
        with Timer("Step 3: Environment reset"):
            obs, info = env.reset()
            logger.info(f"  Observation shape: {obs.shape}")

        # Step 4: Test environment step
        with Timer("Step 4: Single environment step"):
            action = 0  # HOLD
            obs, reward, done, truncated, info = env.step(action)
            logger.info(f"  Step completed: reward={reward}, done={done}")

        # Step 5: Test 10 steps
        with Timer("Step 5: 10 environment steps"):
            env.reset()
            for i in range(10):
                action = i % 3  # Cycle through actions
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    env.reset()

        # Step 6: Create model
        with Timer("Step 6: Create MaskablePPO model"):
            from sb3_contrib import MaskablePPO

            model = MaskablePPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=3e-4,
                n_steps=256,  # Reduced for faster testing
                batch_size=64,
                verbose=0,  # Reduce output
                seed=42,
            )
            logger.info("  Model created successfully")

        # Step 7: Test single rollout collection
        with Timer("Step 7: Collect single rollout (256 steps)"):
            model.env.reset()
            model.collect_rollouts(
                env=model.env,
                callback=None,
                rollout_buffer=model.rollout_buffer,
                n_rollout_steps=model.n_steps,
            )
            logger.info("  Rollout collected successfully")

        # Step 8: Test very short training (512 steps = 2 iterations)
        logger.info("=" * 80)
        logger.info("Step 8: SHORT TRAINING TEST (512 timesteps)")
        logger.info("  Expected iterations: 512 / 256 = 2")
        logger.info("=" * 80)

        start_time = time.time()
        model.learn(total_timesteps=512, progress_bar=False)
        elapsed = time.time() - start_time

        logger.info("=" * 80)
        logger.info(f"✅ SHORT TRAINING COMPLETED in {elapsed:.2f}s")
        logger.info(f"   Model num_timesteps: {model.num_timesteps}")
        logger.info("   Expected: 512")
        logger.info(f"   Match: {model.num_timesteps == 512}")
        logger.info("=" * 80)

        # Cancel alarm
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

        return True

    except TimeoutException:
        logger.error("=" * 80)
        logger.error("❌ TIMEOUT: Operation took longer than 60 seconds")
        logger.error("   This confirms the hang/performance issue")
        logger.error("=" * 80)
        return False

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ ERROR: {e}", exc_info=True)
        logger.error("=" * 80)
        return False


def main() -> None:
    """Main entry point for profiling."""
    try:
        success = profile_training()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        sys.exit(1)
