#!/usr/bin/env python3
"""
Unified Training Runner for Zaif Trade Bot.

Integrates multiple training approaches into a single interface:

ALGORITHMS:
-----------
1. PPO Training (algorithm: 'ppo')
   - Uses PPOTrainer from ppo_trainer.py
   - Standard PPO algorithm with Stable Baselines3
   - Supports evaluation, checkpointing, tensorboard logging
   - Best for: Standard reinforcement learning training

2. Base ML Reinforcement (algorithm: 'base_ml')
   - Uses MLReinforcementExperiment from base_ml_reinforcement.py
   - Base class for ML reinforcement experiments
   - Simple step-based training loop (currently dummy implementation)
   - Best for: Custom reinforcement learning experiments, prototyping

3. Iterative Training (algorithm: 'iterative')
   - Uses logic from run_1m.py
   - Multi-iteration training with resume capability
   - Supports streaming data, validation, Discord notifications
   - Best for: Long-running training sessions, production training

4. Ensemble Training (algorithm: 'ensemble')
   - Uses EnsembleTradingSystem from ensemble.py
   - Combines multiple trained PPO models for improved predictions
   - Supports weighted voting and risk management
   - Best for: Leveraging multiple models for robust trading decisions

USAGE:
------
python -m ztb.training.unified_trainer --config config.json --algorithm ppo

TRADING MODES:
---------------
- scalping: High-frequency scalping with 15s timeframe, smaller positions, higher transaction costs
- normal: Standard trading with 1m timeframe, full feature set, normal position sizes

Examples:
python -m ztb.training.unified_trainer --config unified_training_config.json  # scalping mode
python -m ztb.training.unified_trainer --config unified_training_config_normal.json  # normal mode
"""

# Set environment variables before any imports to avoid PyTorch issues
import logging
import os

from ztb.utils.logging_utils import get_logger
from ztb.utils.errors import safe_operation

logger = get_logger(__name__)

os.environ["PYTORCH_DISABLE_TORCH_DYNAMO"] = "1"
# Disable CUDA to reduce memory usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_USE_CUDA_DSA"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
# Additional memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Conditional imports based on algorithm
ppo_available = True
try:
    # Delay torch import by importing PPOTrainer only when needed
    pass
except ImportError:
    ppo_available = False
    logger.warning("PPO trainer not available (torch import failed)")
    PPOTrainer = None

from ztb.training.entrypoints.base_ml_reinforcement import MLReinforcementExperiment
from ztb.utils import DiscordNotifier


class UnifiedTrainer:
    """
    Unified training interface for different algorithms.

    WORK ASSIGNMENT:
    ---------------
    - PPO Algorithm: @trading-team - Standard RL training, evaluation, logging
    - Base ML Algorithm: @ml-research-team - Custom experiments, prototyping
    - Iterative Algorithm: @production-team - Long-running training, monitoring
    """

    def __init__(
        self,
        config: Dict[str, Any],
        force: bool = False,
        dry_run: bool = False,
        enable_streaming: bool = False,
        stream_batch_size: int = 256,
        max_features: Optional[int] = None,
    ):
        self.config = config
        self.force = force
        self.dry_run = dry_run
        self.enable_streaming = enable_streaming
        self.stream_batch_size = stream_batch_size
        self.max_features = max_features
        self.algorithm = config.get("algorithm", "ppo")
        self.logger = get_logger(__name__)
        # Initialize Discord notifier (disabled in offline mode)
        if config.get("offline_mode", False):
            self.notifier = DiscordNotifier(webhook_url=None)  # Explicitly disable
        else:
            self.notifier = DiscordNotifier()

    def train(self) -> Any:
        """Execute training based on algorithm."""
        return safe_operation(
            logger=logger,
            operation=self._train_impl,
            context="training_execution",
            default_result=None,
        )

    def _train_impl(self) -> Any:
        """Implementation of training execution."""
        if self.algorithm == "ppo":
            return self._train_ppo()
        elif self.algorithm == "base_ml":
            return self._train_base_ml()
        elif self.algorithm == "iterative":
            return self._train_iterative()
        elif self.algorithm == "ensemble":
            return self._train_ensemble()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _train_ppo(self) -> Any:
        """Train using PPO algorithm."""
        # Set environment variables before importing torch
        import os

        os.environ["PYTORCH_DISABLE_TORCH_DYNAMO"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        try:
            from ztb.trading import PPOTrainer
        except ImportError as e:
            raise ImportError(
                f"PPO training is not available due to import failure: {e}. Try using 'base_ml' algorithm instead."
            )

        trainer = PPOTrainer(
            data_path=self.config.get("data_path"),
            config=self.config,
            checkpoint_dir=self.config.get("checkpoint_dir", "checkpoints"),
        )
        model = trainer.train(session_id=self.config.get("session_id", "ppo_session"))

        # Save final model to models directory
        if model is not None:
            import os
            from pathlib import Path

            model_dir = Path(self.config.get("model_dir", "models"))
            model_dir.mkdir(exist_ok=True)
            model_path = (
                model_dir / f"{self.config.get('session_id', 'ppo_session')}.zip"
            )
            model.save(str(model_path))
            self.logger.info(f"Final model saved to {model_path}")

        return model

    def _train_base_ml(self) -> Any:
        """Train using base ML reinforcement."""
        experiment = MLReinforcementExperiment(
            self.config, total_steps=self.config.get("total_steps", 1000)
        )
        return experiment.run()

    def _train_iterative(self) -> Any:
        """Train using iterative approach (from run_1m.py)."""
        # Apply trading mode presets
        trading_mode = self.config.get("trading_mode", "normal")
        if trading_mode == "scalping":
            # Scalping mode presets
            self.config.setdefault("feature_set", "scalping")
            self.config.setdefault("timeframe", "15s")
            self.config.setdefault("reward_scaling", 0.5)
            self.config.setdefault("transaction_cost", 0.002)
            self.config.setdefault("max_position_size", 0.3)
            self.config.setdefault(
                "total_timesteps", 1000000
            )  # Longer training for scalping
            # Update session IDs for scalping
            if "scalping" not in self.config.get("session_id", ""):
                self.config["session_id"] = (
                    f"scalping_{self.config.get('session_id', 'session')}"
                )
                self.config["correlation_id"] = (
                    f"scalping_{self.config.get('correlation_id', 'correlation')}"
                )
        else:
            # Normal trading mode presets
            self.config.setdefault("feature_set", "full")
            self.config.setdefault("timeframe", "1m")
            self.config.setdefault("reward_scaling", 1.0)
            self.config.setdefault("transaction_cost", 0.001)
            self.config.setdefault("max_position_size", 1.0)
            self.config.setdefault("total_timesteps", 100000)

        # Long-running operation confirmation
        total_timesteps = self.config.get("total_timesteps", 100000)
        if total_timesteps >= 100_000 and not self.force:
            from ztb.utils.long_running_confirm import confirm_long_running_operation

            if not confirm_long_running_operation(
                operation_name=f"PPO Training ({self.config.get('session_id', 'iterative_session')})",
                estimated_time=f"~{total_timesteps // 1000}k steps, several hours",
                risk_description="High CPU/memory usage, large log files, potential system slowdown",
            ):
                logger.info("Training cancelled by user")
                return None

        # Dry run mode
        print(f"DEBUG: config feature_set = {self.config.get('feature_set', 'full')}")
        if self.dry_run:
            logger.info(
                f"Dry run: would train with session_id {self.config.get('session_id', 'iterative_session')}"
            )
            logger.info(
                f"Data path: {self.config.get('data_path', 'ml-dataset-enhanced.csv')}"
            )
            logger.info(f"Total timesteps: {total_timesteps}")
            logger.info("Setup validation complete")
            return None

        # Import and use run_1m logic
        from ztb.training.run_1m import main as run_1m_main

        # Set up arguments for run_1m
        sys.argv = [
            "run_1m.py",
            "--data-path",
            self.config.get("data_path", "ml-dataset-enhanced.csv"),
            "--correlation-id",
            self.config.get("session_id", "iterative_session"),
            "--total-timesteps",
            str(total_timesteps),
            "--iterations",
            str(self.config.get("iterations", 10)),
            "--steps-per-iteration",
            str(self.config.get("steps_per_iteration", 100000)),
            "--feature-set",
            self.config.get("feature_set", "full"),
            "--timeframe",
            self.config.get("timeframe", "1m"),
            "--checkpoint-dir",
            self.config.get("checkpoint_dir", "checkpoints"),
            "--log-dir",
            self.config.get("log_dir", "logs"),
            "--model-dir",
            self.config.get("model_dir", "models"),
            "--reward-trade-frequency-penalty",
            str(self.config.get("reward_trade_frequency_penalty", 0.3)),
            "--reward-trade-frequency-halflife",
            str(self.config.get("reward_trade_frequency_halflife", 12.0)),
            "--reward-trade-cooldown-steps",
            str(self.config.get("reward_trade_cooldown_steps", 3)),
            "--reward-trade-cooldown-penalty",
            str(self.config.get("reward_trade_cooldown_penalty", 0.5)),
            "--reward-max-consecutive-trades",
            str(self.config.get("reward_max_consecutive_trades", 3)),
            "--reward-consecutive-trade-penalty",
            str(self.config.get("reward_consecutive_trade_penalty", 0.4)),
            "--transaction-cost",
            str(self.config.get("transaction_cost", 0.001)),
            "--max-position-size",
            str(self.config.get("max_position_size", 1.0)),
        ]

        # DEBUG: Print sys.argv
        print(f"DEBUG: sys.argv = {sys.argv}")
        print(f"DEBUG: feature-set value = {self.config.get('feature_set', 'full')}")  # type: ignore[unreachable]

        # Add optional arguments
        if self.dry_run:
            sys.argv.append("--dry-run")
        if self.force:
            sys.argv.append("--force")
        if self.enable_streaming:
            sys.argv.extend(
                [
                    "--enable-streaming",
                    "--stream-batch-size",
                    str(self.stream_batch_size),
                ]
            )
        if self.max_features is not None:
            sys.argv.extend(["--max-features", str(self.max_features)])
        if self.config.get("offline_mode", False):
            sys.argv.append("--offline-mode")

        # DEBUG: Print final config and sys.argv before calling run_1m_main
        print(f"DEBUG: Final config feature_set = {self.config.get('feature_set')}")
        print(f"DEBUG: Final sys.argv = {sys.argv}")

        return run_1m_main()

    def _train_ensemble(self) -> Any:
        """Train using ensemble approach (load and combine existing models)."""
        from ztb.trading.ensemble import EnsembleTradingSystem

        # Get model configurations from config
        model_configs = self.config.get("ensemble_models", [])
        if not model_configs:
            raise ValueError(
                "No ensemble_models specified in config for ensemble training"
            )

        # Create ensemble system
        ensemble_system = EnsembleTradingSystem(model_configs)

        self.logger.info(
            f"Ensemble system initialized with {len(ensemble_system.ensemble.models)} models"
        )

        # For ensemble, we don't train but validate the setup
        if self.dry_run:
            self.logger.info("Dry run: ensemble system setup validated")
            return ensemble_system

        # Save ensemble configuration for later use
        import json

        ensemble_config_path = (
            Path(self.config.get("model_dir", "models")) / "ensemble_config.json"
        )
        with open(ensemble_config_path, "w") as f:
            json.dump(
                {
                    "model_configs": model_configs,
                    "created_at": str(datetime.now()),
                    "session_id": self.config.get("session_id", "ensemble_session"),
                },
                f,
                indent=2,
            )

        self.logger.info(f"Ensemble configuration saved to {ensemble_config_path}")

        return ensemble_system


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified Training Runner")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "base_ml", "iterative", "ensemble"],
        help="Override algorithm from config file",
    )
    parser.add_argument(
        "--data-path", type=str, help="Override data path from config file"
    )
    parser.add_argument(
        "--total-timesteps", type=int, help="Override total timesteps from config file"
    )
    parser.add_argument(
        "--session-id", type=str, help="Override session ID from config file"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution without long-running operation confirmation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - validate setup without training",
    )
    parser.add_argument(
        "--enable-streaming",
        action="store_true",
        help="Enable streaming pipeline (default: disabled)",
    )
    parser.add_argument(
        "--stream-batch-size",
        type=int,
        default=256,
        help="Streaming batch size (default: 256)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features to use (default: all features)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = get_logger(__name__)

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")

        # Override config with command line arguments
        if args.algorithm:
            config["algorithm"] = args.algorithm
        if args.data_path:
            config["data_path"] = args.data_path
        if args.total_timesteps:
            config["total_timesteps"] = args.total_timesteps
        if args.session_id:
            config["session_id"] = args.session_id

        logger.info(f"Using algorithm: {config.get('algorithm', 'ppo')}")
        logger.info(f"Session ID: {config.get('session_id', 'default')}")

        # Create and run trainer
        trainer = UnifiedTrainer(
            config,
            args.force,
            args.dry_run,
            args.enable_streaming,
            args.stream_batch_size,
            args.max_features,
        )
        result = trainer.train()

        if result is None:
            logger.warning("Training returned None - may have been cancelled or failed")
        else:
            logger.info("Training completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
