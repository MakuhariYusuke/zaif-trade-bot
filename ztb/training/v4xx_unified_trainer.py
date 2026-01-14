#!/usr/bin/env python3
# ruff: noqa: E402
"""
Unified Training System for v4XX Series

A lightweight, focused training system that supports all v4XX versions
through unified configuration and minimal interface.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.config.loader import PriorityConfigLoader
from ztb.features.processors.optimization.features import OptimizerFeatureTracker
from ztb.training.core.unified_base import UnifiedBase
from ztb.training.unified_trainer.algorithms import create_algorithm_trainer
from ztb.utils.v4xx_config_converter import V4XXConfigConverter
from ztb.optimization.parallel import ParallelWindowEvaluator
from ztb.evaluation.walk_forward.checkpoint import CheckpointManager


class V4XXUnifiedTrainer(UnifiedBase):
    """Lightweight unified trainer for all v4XX series models."""

    def __init__(self, config_path: str, version: Optional[str] = None):
        """
        Initialize unified trainer.

        Args:
            config_path: Path to configuration file
            version: Override version detection (optional)
        """
        super().__init__(config_path, version)
        self.config = self._load_and_convert_config()
        self.trainer = None

        # Initialize optimizer tracker if configured (v442+ feature)
        optimizer_config = self.config.get("optimizer_features")
        if optimizer_config is not None:
            self.optimizer_tracker = OptimizerFeatureTracker(
                max_history=optimizer_config.get("max_history", 1000),
                enable_normalization=optimizer_config.get("enable_normalization", True),
                normalization_method=optimizer_config.get(
                    "normalization_method", "robust"
                ),
                outlier_threshold=optimizer_config.get("outlier_threshold", 1.5),
            )
            self.logger.info(
                "Optimizer features enabled for enhanced training monitoring"
            )
        else:
            self.optimizer_tracker = None
            self.logger.info("Optimizer features disabled (legacy configuration)")

    def _load_and_convert_config(self) -> Dict[str, Any]:
        """Load and convert configuration to unified format."""
        try:
            # Load raw config
            raw_config = self.load_config(str(self.config_path))

            # Validate config using pydantic loader
            config_loader = PriorityConfigLoader()
            validated_config = config_loader.validate_config(raw_config)

            # Detect version if not provided
            if self.version is None:
                self.version = V4XXConfigConverter.detect_config_version(
                    validated_config
                )
                self.logger.info(f"Detected configuration version: {self.version}")

            # Convert to unified format
            unified_config = V4XXConfigConverter.convert_to_unified(validated_config)

            self.logger.debug("unified_config keys: %s", list(unified_config.keys()))
            self.logger.debug(
                "algorithm in unified_config: %s", "algorithm" in unified_config
            )
            self.logger.debug(
                "model_name in unified_config: %s", "model_name" in unified_config
            )

            # Add metadata
            unified_config["_metadata"] = {
                "original_version": self.version,
                "converted_at": "unified_trainer",
                "config_path": str(self.config_path),
            }

            self.logger.info(f"Configuration loaded and converted for {self.version}")
            return unified_config

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def validate_config(self) -> bool:
        """Validate the unified configuration."""
        required_fields = ["algorithm", "model_name", "training"]
        return super().validate_config(self.config, required_fields)

    def initialize_trainer(self):
        """Initialize the algorithm trainer."""
        try:
            algorithm = self.config["algorithm"]
            self.logger.info(
                f"Initializing {algorithm.upper()} trainer for {self.config['model_name']}"
            )

            self.trainer = create_algorithm_trainer(
                algorithm,
                self.config,
                self.logger,
                optimizer_tracker=self.optimizer_tracker,
            )

            if (
                hasattr(self.trainer, "validate_config")
                and not self.trainer.validate_config()
            ):
                raise ValueError("Trainer configuration validation failed")

            self.logger.info("Trainer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize trainer: {e}")
            raise

    def train(self):
        """Execute training."""
        if self.trainer is None:
            self.initialize_trainer()

        try:
            self.logger.info(f"Starting training for {self.config['model_name']}")
            self.logger.info(
                f"Total timesteps: {self.config['training']['total_timesteps']:,}"
            )

            self.trainer.train()

            self.logger.info("Training completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def save_config(self, output_path: Optional[str] = None):
        """Save the unified configuration."""
        if output_path is None and self.config_path:
            output_path = str(
                self.config_path.parent / f"{self.config_path.stem}_unified.json"
            )
        elif output_path is None:
            output_path = "config_unified.json"

        super().save_config(self.config, output_path)

    def run(self):
        """Execute the main functionality (alias for train)."""
        self.train()

    def evaluate_parallel(
        self,
        df: Any,
        windows: list,
        timesteps: int,
        env_factory: Optional[Any] = None,
        algorithm_factory: Optional[Any] = None,
        num_workers: Optional[int] = None,
        run_id: Optional[str] = None,
        enable_checkpointing: bool = True,
    ) -> tuple:
        """
        Evaluate Walk-Forward windows in parallel using multiprocessing.

        Provides 87-92% speedup: 50 windows from ~25 hours → 2-4 hours.

        Args:
            df: Full dataset (DataFrame with OHLCV data)
            windows: List of (train_end, val_end, test_end) tuples
            timesteps: Training timesteps per window
            env_factory: Environment factory callable (from config if None)
            algorithm_factory: Algorithm factory callable (from config if None)
            num_workers: Number of parallel workers (CPU count if None)
            run_id: Run identifier for checkpointing
            enable_checkpointing: Whether to save/restore checkpoints

        Returns:
            Tuple[results_dict, errors_dict, summary_stats]
            - results_dict: Dict[window_id] → WindowPerformance
            - errors_dict: Dict[window_id] → error_message
            - summary_stats: Dict with aggregated metrics

        Example:
            results, errors, summary = trainer.evaluate_parallel(
                df=market_df,
                windows=[(1000, 1200, 1400), (1200, 1400, 1600), ...],
                timesteps=10000,
                run_id="backtest_v455"
            )
        """
        self.logger.info(f"Starting parallel evaluation of {len(windows)} windows")

        # Initialize checkpoint manager if requested
        checkpoint_mgr = None
        if enable_checkpointing and run_id:
            checkpoint_dir = self.config.get("evaluation", {}).get(
                "checkpoint_dir", "checkpoints/walk_forward"
            )
            checkpoint_mgr = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                compress="zstd"
            )
            self.logger.info(f"Checkpointing enabled: {checkpoint_dir}")

        # Initialize parallel evaluator
        evaluator = ParallelWindowEvaluator(
            num_workers=num_workers,
            checkpoint_mgr=checkpoint_mgr,
            enable_error_collection=True
        )

        # Get algorithm and environment factories from config if not provided
        if env_factory is None or algorithm_factory is None:
            env_factory, algorithm_factory = self._get_factories()

        # Evaluate windows in parallel
        results, errors = evaluator.evaluate_windows_parallel(
            df=df,
            windows=windows,
            timesteps=timesteps,
            env_factory=env_factory,
            algorithm_factory=algorithm_factory,
            policy=self.config.get("training", {}).get("policy", "MlpPolicy"),
            algorithm_params=self.config.get("algorithm_params", {}),
            run_id=run_id
        )

        # Generate summary statistics
        summary = evaluator.get_results_summary()
        
        self.logger.info(
            f"✓ Parallel evaluation completed: "
            f"completed={summary['total_windows']}, errors={summary['error_count']}"
        )

        return results, errors, summary

    def _get_factories(self) -> tuple:
        """
        Get environment and algorithm factories from configuration.

        Returns:
            Tuple[env_factory, algorithm_factory]
        """
        try:
            from ztb.evaluation.walk_forward.evaluator import (
                create_environment,
                create_sac_algorithm,
            )

            env_config = self.config.get("training", {}).get("environment", {})
            algorithm_config = self.config.get("algorithm_params", {})

            env_factory = lambda: create_environment(env_config)
            algorithm_factory = lambda env: create_sac_algorithm(
                env, algorithm_config
            )

            return env_factory, algorithm_factory
        except Exception as e:
            self.logger.error(f"Failed to create factories from config: {e}")
            raise

    def analyze_results(self):
        """Analyze training results."""
        if self.trainer is None:
            self.logger.warning("No trainer initialized, cannot analyze results")
            return

        try:
            if hasattr(self.trainer, "analyze_results"):
                self.logger.info("Analyzing training results...")
                self.trainer.analyze_results()
            else:
                self.logger.info("Trainer does not support result analysis")
        except Exception as e:
            self.logger.error(f"Failed to analyze results: {e}")
            raise


def main():
    """Main entry point for unified training system."""
    parser = argparse.ArgumentParser(
        description="Unified Training System for v4XX Series"
    )
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--version", help="Override version detection")
    parser.add_argument(
        "--save-config", action="store_true", help="Save converted configuration"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate configuration"
    )

    # Mutually exclusive override options: specify episodes OR total_timesteps via CLI.
    # If both are provided, argparse will enforce exclusivity and raise an error.
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--episodes",
        type=int,
        help=(
            "Number of episodes to run (overrides config). "
            "If provided, total_timesteps = episodes * max_episode_length"
        ),
    )
    group.add_argument(
        "--total-timesteps",
        type=int,
        help="Total timesteps to train for (overrides value in config)",
    )

    args = parser.parse_args()

    try:
        # Initialize trainer
        trainer = V4XXUnifiedTrainer(args.config, args.version)

        # CLI overrides: episodes or total_timesteps take precedence over config values.
        # If episodes is provided, compute total_timesteps from max_episode_length (config or default).
        if args.episodes is not None:
            try:
                from ztb.training.reward_function_optimizer.constants import (
                    DEFAULT_MAX_EPISODE_LENGTH,
                )

                cfg = trainer.config if isinstance(trainer.config, dict) else {}
                training_section = (
                    cfg.get("training", {}) if isinstance(cfg, dict) else {}
                )
                env_section = (
                    training_section.get("environment", {})
                    if isinstance(training_section, dict)
                    else {}
                )

                max_ep = None
                if isinstance(env_section, dict):
                    inner_cfg = env_section.get("config", env_section)
                    if isinstance(inner_cfg, dict):
                        max_ep = inner_cfg.get("max_episode_length") or inner_cfg.get(
                            "max_episode_steps"
                        )

                if max_ep is None:
                    max_ep = (
                        training_section.get("max_episode_length")
                        if isinstance(training_section, dict)
                        else None
                    )

                if max_ep is None:
                    max_ep = DEFAULT_MAX_EPISODE_LENGTH

                total_ts = int(args.episodes) * int(max_ep)

                if "training" not in trainer.config:
                    trainer.config["training"] = {}
                trainer.config["training"]["episodes"] = int(args.episodes)
                trainer.config["training"]["total_timesteps"] = total_ts
                trainer.logger.info(
                    f"Overriding config: episodes={args.episodes}, computed total_timesteps={total_ts} (max_episode_length={max_ep})"
                )
            except Exception as e:
                trainer.logger.error("Failed to apply --episodes override: %s", e)
                sys.exit(1)
        elif args.total_timesteps is not None:
            if "training" not in trainer.config:
                trainer.config["training"] = {}
            trainer.config["training"]["total_timesteps"] = int(args.total_timesteps)
            trainer.logger.info(
                f"Overriding config: total_timesteps={args.total_timesteps}"
            )

        # Validate configuration
        if not trainer.validate_config():
            sys.exit(1)

        if args.validate_only:
            trainer.logger.info("Configuration validation completed successfully")
            return

        # Save converted config if requested
        if args.save_config:
            trainer.save_config()

        # Execute training
        trainer.train()

    except Exception as e:
        trainer.logger.error("❌ Training system failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
