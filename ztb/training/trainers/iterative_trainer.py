#!/usr/bin/env python3
"""
Iterative Algorithm Trainer for Unified Training.

Handles iterative training logic using run_1m.py.
"""

import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from ztb.training.trainers.base_trainer import BaseTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class IterativeAlgorithmTrainer(BaseTrainer):
    """
    Handles iterative algorithm training.
    """

    def _apply_trading_mode_presets(
        self, unified_config: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Apply trading mode presets to config.

        Args:
            unified_config: Unified configuration

        Returns:
            Updated unified config
        """
        trading_mode = unified_config.get("trading_mode", "normal")

        if trading_mode == "scalping":
            # Scalping mode presets
            scalping_defaults = {
                "feature_set": "scalping",
                "timeframe": "15s",
                "reward_scaling": 0.5,
                "transaction_cost": 0.002,
                "max_position_size": 0.3,
                "total_timesteps": 1_000_000,
            }
            for key, value in scalping_defaults.items():
                self.config_manager.config.setdefault(key, value)
                unified_config.setdefault(key, value)
            # Update session IDs for scalping
            if "scalping" not in self.config_manager.config.get("session_id", ""):
                self.config_manager.config[
                    "session_id"
                ] = f"scalping_{self.config_manager.config.get('session_id', 'session')}"
                self.config_manager.config[
                    "correlation_id"
                ] = f"scalping_{self.config_manager.config.get('correlation_id', 'correlation')}"
                unified_config["session_id"] = self.config_manager.config["session_id"]
                unified_config["correlation_id"] = self.config_manager.config[
                    "correlation_id"
                ]
        else:
            # Normal trading mode presets
            normal_defaults = {
                "feature_set": "full",
                "timeframe": "1m",
                "reward_scaling": 1.0,
                "transaction_cost": 0.001,
                "max_position_size": 1.0,
                "total_timesteps": 100_000,
            }
            for key, value in normal_defaults.items():
                self.config_manager.config.setdefault(key, value)
                unified_config.setdefault(key, value)

        return unified_config

    def _build_run_1m_args(self, unified_config: dict[str, Any]) -> list[str]:
        """
        Build command line arguments for run_1m.py.

        Args:
            unified_config: Unified configuration

        Returns:
            list of command line arguments
        """
        args = [
            "run_1m.py",
            "--data-path",
            unified_config.get("data_path", "ml-dataset-enhanced.csv"),
            "--correlation-id",
            unified_config.get("session_id", "iterative_session"),
            "--total-timesteps",
            str(unified_config["training"]["total_timesteps"]),
            "--iterations",
            str(unified_config.get("iterations", 10)),
            "--steps-per-iteration",
            str(unified_config.get("steps_per_iteration", 100000)),
            "--feature-set",
            unified_config.get("feature_set", "full"),
            "--timeframe",
            unified_config.get("timeframe", "1m"),
            "--checkpoint-dir",
            unified_config.get("checkpoint_dir", "checkpoints"),
            "--checkpoint-interval",
            str(unified_config.get("checkpoint_interval", 10000)),
            "--log-dir",
            unified_config.get("log_dir", "logs"),
            "--model-dir",
            unified_config.get("model_dir", "models"),
            "--reward-trade-frequency-penalty",
            str(unified_config.get("reward_trade_frequency_penalty", 0.3)),
            "--reward-trade-frequency-halflife",
            str(unified_config.get("reward_trade_frequency_halflife", 12.0)),
            "--reward-trade-cooldown-steps",
            str(unified_config.get("reward_trade_cooldown_steps", 3)),
            "--reward-trade-cooldown-penalty",
            str(unified_config.get("reward_trade_cooldown_penalty", 0.5)),
            "--reward-max-consecutive-trades",
            str(unified_config.get("reward_max_consecutive_trades", 3)),
            "--reward-consecutive-trade-penalty",
            str(unified_config.get("reward_consecutive_trade_penalty", 0.4)),
            "--transaction-cost",
            str(unified_config.get("transaction_cost", 0.001)),
            "--max-position-size",
            str(unified_config.get("max_position_size", 1.0)),
        ]

        # Add optional arguments
        if unified_config.get("dry_run", False):
            args.append("--dry-run")
        if unified_config.get("force", False):
            args.append("--force")
        if unified_config.get("enable_streaming", False):
            args.extend(
                [
                    "--enable-streaming",
                    "--stream-batch-size",
                    str(unified_config.get("stream_batch_size", 256)),
                ]
            )

        max_features = unified_config.get("max_features") or (
            unified_config.get("memory_optimization", {}) or {}
        ).get("max_features")
        if max_features is not None:
            args.extend(["--max-features", str(max_features)])

        data_rows_limit = unified_config.get("data_rows_limit") or (
            unified_config.get("memory_optimization", {}) or {}
        ).get("data_rows_limit")
        if data_rows_limit is not None:
            args.extend(["--data-rows-limit", str(data_rows_limit)])

        if unified_config.get("offline_mode", False):
            args.append("--offline-mode")

        return args

    def train(self, unified_config: dict[str, Any]) -> Any:
        """
        Execute iterative training.

        Args:
            unified_config: Unified configuration

        Returns:
            Training result
        """
        # Apply trading mode presets
        unified_config = self._apply_trading_mode_presets(unified_config)

        # Long-running operation confirmation
        total_timesteps = unified_config["training"]["total_timesteps"]
        if total_timesteps >= 100_000 and not unified_config.get("force", False):
            from ztb.utils.long_running_confirm import confirm_long_running_operation

            if not confirm_long_running_operation(
                operation_name=f"PPO Training ({unified_config.get('session_id', 'iterative_session')})",
                estimated_time=f"~{total_timesteps // 1000}k steps, several hours",
                risk_description="High CPU/memory usage, large log files, potential system slowdown",
                message="This will train a PPO model for a long time. Continue?",
            ):
                logger.info("Training cancelled by user")
                return None

        # Dry run mode
        if unified_config.get("dry_run", False):
            logger.info(
                f"Dry run: would train with session_id {unified_config.get('session_id', 'iterative_session')}"
            )
            logger.info(
                f"Data path: {unified_config.get('data_path', 'ml-dataset-enhanced.csv')}"
            )
            logger.info(f"Total timesteps: {total_timesteps}")
            logger.info("Setup validation complete")
            return None

        # Import and use run_1m logic
        from ztb.training.scripts.run_1m import main as run_1m_main

        # set up arguments for run_1m
        sys.argv = self._build_run_1m_args(unified_config)

        # DEBUG: Print sys.argv
        logger.debug(f"sys.argv = {sys.argv}")
        logger.debug(f"feature-set value = {unified_config.get('feature_set', 'full')}")

        # DEBUG: Print final config and sys.argv before calling run_1m_main
        logger.debug(f"Final config feature_set = {unified_config.get('feature_set')}")
        logger.debug(f"Final sys.argv = {sys.argv}")

        def _json_default(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Enum):
                return value.value
            return str(value)

        serialized_config = json.dumps(unified_config, default=_json_default)
        os.environ["ZTB_UNIFIED_ITERATIVE_CONFIG"] = serialized_config

        try:
            return run_1m_main()
        finally:
            os.environ.pop("ZTB_UNIFIED_ITERATIVE_CONFIG", None)
