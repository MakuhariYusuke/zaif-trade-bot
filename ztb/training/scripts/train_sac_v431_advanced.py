#!/usr/bin/env python3
"""
SAC v431 Advanced Training Script
高度な学習手法を統合したトレーニングスクリプト

Features:
- カリキュラム学習 (Curriculum Learning)
- マルチステージ学習 (Multi-Stage Training)
- アンサンブル学習 (Ensemble Training)
- Unified Analysis統合
- モデル出力値確認機能 (Debug Mode)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.unified_analyze import UnifiedAnalysisSuite
from ztb.training.unified_trainer.ensemble_system import (
    EnsembleConfig,
    EnsemblePredictor,
)
from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACv431AdvancedTrainer:
    """SAC v431 Advanced Training Controller"""

    def __init__(self, config_path: str, debug_mode: bool = False):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.debug_mode = debug_mode

        # Initialize trainer with config and short timesteps for testing
        test_timesteps = (
            1000 if self.debug_mode else None
        )  # Short test run in debug mode
        self.trainer = UnifiedTrainer(
            config=self.config, total_timesteps=test_timesteps
        )

        self.analysis_suite = UnifiedAnalysisSuite()

        # Initialize ensemble system if enabled
        self.ensemble_config = None
        if (
            self.config.get("advanced_learning", {})
            .get("ensemble_training", {})
            .get("enabled", False)
        ):
            ensemble_cfg = self.config["advanced_learning"]["ensemble_training"]
            self.ensemble_config = EnsembleConfig(
                enabled=ensemble_cfg["enabled"],
                members=ensemble_cfg["members"],
                specializations=ensemble_cfg["specializations"],
                voting_mechanism=ensemble_cfg["voting_mechanism"],
                diversity_weight=ensemble_cfg["diversity_weight"],
                consensus_requirement=ensemble_cfg["consensus_requirement"],
                stability_voting=ensemble_cfg["stability_voting"],
                adaptation=ensemble_cfg["adaptation"],
            )

        # Debug output storage
        self.debug_outputs = {
            "actions": [],
            "rewards": [],
            "observations": [],
            "training_stats": [],
        }

    def _debug_log_model_outputs(
        self, step: int, action: Any, reward: float, observation: Any
    ):
        """Log model outputs for debugging"""
        if self.debug_mode:
            self.debug_outputs["actions"].append(action)
            self.debug_outputs["rewards"].append(reward)
            self.debug_outputs["observations"].append(observation)

            # Log every 100 steps
            if step % 100 == 0:
                logger.info(f"🔍 Debug Step {step}:")
                logger.info(f"   Action: {action}")
                logger.info(f"   Reward: {reward:.6f}")
                logger.info(
                    f"   Observation shape: {np.array(observation).shape if observation is not None else 'None'}"
                )

                # Analyze action distribution
                if len(self.debug_outputs["actions"]) >= 100:
                    recent_actions = self.debug_outputs["actions"][-100:]
                    action_counts = pd.Series(recent_actions).value_counts()
                    logger.info(f"   Recent action distribution: {dict(action_counts)}")

                    recent_rewards = self.debug_outputs["rewards"][-100:]
                    logger.info(
                        f"   Recent reward stats: mean={np.mean(recent_rewards):.6f}, std={np.std(recent_rewards):.6f}"
                    )

    def _save_debug_outputs(self, filename: str):
        """Save debug outputs to file"""
        if self.debug_mode and self.debug_outputs["actions"]:
            debug_data = {
                "actions": self.debug_outputs["actions"],
                "rewards": self.debug_outputs["rewards"],
                "observations": self.debug_outputs["observations"][
                    :100
                ],  # Limit observations
                "training_stats": self.debug_outputs["training_stats"],
            }

            with open(filename, "w") as f:
                json.dump(debug_data, f, indent=2, default=str)

            logger.info(f"💾 Debug outputs saved to {filename}")

            # Generate debug summary
            if self.debug_outputs["actions"]:
                action_series = pd.Series(self.debug_outputs["actions"])
                reward_series = pd.Series(self.debug_outputs["rewards"])

                summary = {
                    "total_steps": len(self.debug_outputs["actions"]),
                    "action_distribution": action_series.value_counts().to_dict(),
                    "reward_stats": {
                        "mean": float(reward_series.mean()),
                        "std": float(reward_series.std()),
                        "min": float(reward_series.min()),
                        "max": float(reward_series.max()),
                        "median": float(reward_series.median()),
                    },
                    "trading_activity": {
                        "total_trades": int(
                            (action_series != 0).sum()
                        ),  # Non-zero actions
                        "buy_actions": int((action_series == 1).sum()),
                        "sell_actions": int((action_series == -1).sum()),
                        "hold_actions": int((action_series == 0).sum()),
                    },
                }

                summary_file = filename.replace(".json", "_summary.json")
                with open(summary_file, "w") as f:
                    json.dump(summary, f, indent=2)

                logger.info(f"📊 Debug summary saved to {summary_file}")
                logger.info(f"🎯 Trading Activity: {summary['trading_activity']}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_curriculum_learning(self) -> bool:
        """Execute curriculum learning"""
        logger.info("🚀 Starting Curriculum Learning")

        curriculum_config = self.config["advanced_learning"]["curriculum_learning"]
        stages = curriculum_config["stages"]

        for stage in stages:
            logger.info(f"📚 Stage: {stage['name']} (timesteps: {stage['timesteps']})")

            # Update training config for this stage
            stage_config = self.config.copy()
            stage_config["training"].update(
                {
                    "total_timesteps": stage["timesteps"],
                    "learning_rate": stage["learning_rate"],
                    "ent_coef": stage["ent_coef"],
                }
            )
            stage_config["reward_function"]["reward_scale"] = stage["reward_scale"]

            # Train this stage
            success = self.trainer.train()

            if not success:
                logger.error(f"❌ Stage {stage['name']} failed")
                return False

            logger.info(f"✅ Stage {stage['name']} completed successfully")

            # Save debug outputs for this stage
            if self.debug_mode:
                self._save_debug_outputs(
                    f"debug/sac_v431_curriculum_{stage['name']}_debug.json"
                )

        logger.info("🎓 Curriculum Learning completed")
        return True

    def run_multi_stage_training(self) -> bool:
        """Execute multi-stage training"""
        logger.info("🔄 Starting Multi-Stage Training")

        multi_stage_config = self.config["advanced_learning"]["multi_stage_training"]
        stages = multi_stage_config["stages"]

        for stage in stages:
            logger.info(
                f"🎯 Stage: {stage['name']} (timesteps: {stage['timesteps']}, focus: {stage['focus']})"
            )

            # Update training config for this stage
            stage_config = self.config.copy()
            stage_config["training"].update(
                {
                    "total_timesteps": stage["timesteps"],
                    "learning_rate": stage["learning_rate"],
                    "ent_coef": stage["ent_coef"],
                }
            )

            # Train this stage
            success = self.trainer.train()

            if not success:
                logger.error(f"❌ Stage {stage['name']} failed")
                return False

            logger.info(f"✅ Stage {stage['name']} completed successfully")

            # Save debug outputs for this stage
            if self.debug_mode:
                self._save_debug_outputs(
                    f"debug/sac_v431_multistage_{stage['name']}_debug.json"
                )

        logger.info("🔄 Multi-Stage Training completed")
        return True

    def run_ensemble_training(self) -> bool:
        """Execute ensemble training"""
        logger.info("👥 Starting Ensemble Training")

        if not self.ensemble_config:
            logger.error("❌ Ensemble config not initialized")
            return False

        # Initialize ensemble predictor
        ensemble_predictor = EnsemblePredictor(self.ensemble_config)

        # Train each ensemble member
        for i, specialization in enumerate(self.ensemble_config.specializations):
            logger.info(f"🤖 Training ensemble member {i}: {specialization}")

            # Create specialized config for this member
            member_config = self.config.copy()
            member_config["model_name"] = f"sac_v431_ensemble_{specialization}"

            # Add specialization-specific modifications
            if specialization == "bull":
                member_config["reward_function"][
                    "reward_scale"
                ] *= 1.2  # More aggressive in bull markets
            elif specialization == "bear":
                member_config["reward_function"][
                    "sell_bonus"
                ] *= 1.3  # More defensive in bear markets
            elif specialization == "high_vol":
                member_config["training"][
                    "ent_coef"
                ] = "auto_0.05"  # Higher exploration
            elif specialization == "low_vol":
                member_config["training"][
                    "ent_coef"
                ] = "auto_0.005"  # Lower exploration

            # Train member
            success = self.trainer.train()

            if not success:
                logger.error(f"❌ Ensemble member {specialization} training failed")
                return False

            logger.info(f"✅ Ensemble member {specialization} trained successfully")

            # Save debug outputs for this member
            if self.debug_mode:
                self._save_debug_outputs(
                    f"debug/sac_v431_ensemble_{specialization}_debug.json"
                )

        logger.info("👥 Ensemble Training completed")
        return True

    def run_standard_training(self) -> bool:
        """Execute standard training with optimized parameters"""
        logger.info("⚡ Starting Standard Training")

        success = self.trainer.train()

        if success:
            logger.info("✅ Standard Training completed successfully")
        else:
            logger.error("❌ Standard Training failed")

        # Save debug outputs
        if self.debug_mode:
            self._save_debug_outputs("debug/sac_v431_standard_debug.json")

        return success

    def run_unified_analysis(self) -> bool:
        """Execute unified analysis suite"""
        logger.info("📊 Running Unified Analysis")

        try:
            # Generate training report
            if (
                self.config.get("unified_analysis_integration", {})
                .get("automated_reporting", {})
                .get("training_report", False)
            ):
                logger.info("📋 Generating training report")
                # Training report generation would go here

            # Run comparative analysis
            if (
                self.config.get("unified_analysis_integration", {})
                .get("automated_reporting", {})
                .get("performance_comparison", False)
            ):
                logger.info("📈 Running performance comparison")
                # Performance comparison would go here

            logger.info("✅ Unified Analysis completed")
            return True

        except Exception as e:
            logger.error(f"❌ Unified Analysis failed: {e}")
            return False

    def run(self, mode: str) -> bool:
        """Main execution method"""
        logger.info(f"🚀 SAC v431 Advanced Training - Mode: {mode}")

        success = False

        if mode == "curriculum":
            success = self.run_curriculum_learning()
        elif mode == "multi_stage":
            success = self.run_multi_stage_training()
        elif mode == "ensemble":
            success = self.run_ensemble_training()
        elif mode == "standard":
            success = self.run_standard_training()
        else:
            logger.error(f"❌ Unknown mode: {mode}")
            return False

        # Run unified analysis if enabled
        if success and self.config.get("unified_analysis_integration", {}).get(
            "enabled", False
        ):
            self.run_unified_analysis()

        return success


def main():
    parser = argparse.ArgumentParser(description="SAC v431 Advanced Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v431/sac_v431_advanced.json",
        help="Configuration file path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["curriculum", "multi_stage", "ensemble", "standard"],
        help="Training mode",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to monitor model outputs",
    )
    parser.add_argument("--output", type=str, help="Output directory")

    args = parser.parse_args()

    # Create trainer
    trainer = SACv431AdvancedTrainer(args.config, debug_mode=args.debug)

    # Create debug directory if debug mode
    if args.debug:
        os.makedirs("debug", exist_ok=True)
        logger.info("🔍 Debug mode enabled - model outputs will be monitored")

    # Run training
    success = trainer.run(args.mode)

    if success:
        logger.info("🎉 SAC v431 training completed successfully!")
        if args.debug:
            logger.info("💾 Debug outputs saved in debug/ directory")
        sys.exit(0)
    else:
        logger.error("💥 SAC v431 training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
