"""
Memory-efficient training runner with monitoring.

This script runs training with memory monitoring and automatic cleanup.
"""
import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

from ztb.trading.environment.constants import BYTES_PER_MB

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_memory_usage() -> float | None:
    """Get current memory usage in MB."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / BYTES_PER_MB
    except ImportError:
        return None

def monitor_memory(interval: int = 10) -> None:
    """Monitor memory usage periodically."""
    import threading

    def _monitor():
        peak_memory = 0
        while True:
            mem = get_memory_usage()
            if mem is not None:
                if mem > peak_memory:
                    peak_memory = mem
                logger.info(f"Memory: {mem:.1f} MB (Peak: {peak_memory:.1f} MB)")
            time.sleep(interval)

    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()

def run_training_with_memory_optimization(
    config_path: str, force: bool = False
) -> int | None:
    """Run training with memory optimization."""
    logger.info("=" * 80)
    logger.info("MEMORY-OPTIMIZED TRAINING RUNNER")
    logger.info("=" * 80)

    # Check initial memory
    initial_mem = get_memory_usage()
    if initial_mem:
        logger.info(f"Initial memory usage: {initial_mem:.1f} MB")

    # Start memory monitoring
    if get_memory_usage() is not None:
        logger.info("Starting memory monitor...")
        monitor_memory(interval=30)  # Monitor every 30 seconds

    # Force garbage collection before training
    logger.info("Pre-training memory cleanup...")
    gc.collect()

    # Import training modules only when needed
    logger.info(f"Loading configuration from {config_path}...")
    from run_training import main as run_training_main

    # Temporarily modify sys.argv to pass arguments
    original_argv = sys.argv
    try:
        sys.argv = ["run_training.py", "--config", config_path]
        if force:
            sys.argv.append("--force")

        logger.info("Starting training process...")
        result = run_training_main()

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)

        # Post-training cleanup
        logger.info("Post-training memory cleanup...")
        gc.collect()

        final_mem = get_memory_usage()
        if final_mem and initial_mem:
            mem_increase = final_mem - initial_mem
            logger.info(f"Final memory usage: {final_mem:.1f} MB")
            logger.info(f"Memory increase: {mem_increase:+.1f} MB")

        return result

    finally:
        sys.argv = original_argv
        # Final cleanup
        gc.collect()

def main() -> int | None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run training with memory optimization and monitoring"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration file"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force execution without confirmation"
    )

    args = parser.parse_args()

    # Verify config exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        return 1

    try:
        result = run_training_with_memory_optimization(
            config_path=args.config, force=args.force
        )
        return result
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user (Ctrl+C)")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
