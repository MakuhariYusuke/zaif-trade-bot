#!/usr/bin/env python3
"""
Ensemble functionality mixin for training classes.

This mixin provides ensemble prediction capabilities that can be added to any trainer.
"""

from typing import Any, Dict, Optional

import numpy as np

from ztb.training.unified_trainer.ensemble_system import (
    EnsembleConfig,
    EnsemblePredictor,
)
from ztb.training.unified_trainer.reporting import TrainingReporter
from ztb.training.unified_trainer.ui import TrainingUI
from ztb.utils.logging_utils import get_logger


class EnsembleMixin:
    """
    Mixin class providing ensemble functionality for trainers.

    This mixin can be added to any trainer class to enable ensemble prediction capabilities.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensemble attributes
        self.ensemble_enabled = False
        self.ensemble_system: Optional[EnsemblePredictor] = None
        self.ensemble_config: Optional[EnsembleConfig] = None

        # Initialize logger
        self.ensemble_logger = get_logger(f"{self.__class__.__name__}.Ensemble")

    def initialize_ensemble(self, config: Dict[str, Any]) -> None:
        """
        Initialize ensemble system if enabled in config.

        Args:
            config: Training configuration
        """
        ensemble_config = config.get("ensemble", {})

        if not ensemble_config.get("enabled", False):
            self.ensemble_enabled = False
            self.ensemble_logger.info("Ensemble system disabled")
            return

        try:
            # Create ensemble configuration
            self.ensemble_config = EnsembleConfig(
                enabled=True,
                members=ensemble_config.get("num_members", 3),
                voting_mechanism=ensemble_config.get("voting_method", "majority"),
                diversity_weight=ensemble_config.get("stability_weight", 0.3),
            )

            # Initialize ensemble system
            self.ensemble_system = EnsemblePredictor(self.ensemble_config)
            self.ensemble_enabled = True

            self.ensemble_logger.info(
                f"Ensemble system initialized with {self.ensemble_config.members} members"
            )
            self.ensemble_logger.info(
                f"Voting method: {self.ensemble_config.voting_mechanism}"
            )

        except Exception as e:
            self.ensemble_logger.error(f"Failed to initialize ensemble system: {e}")
            self.ensemble_enabled = False

    def predict_with_ensemble(
        self, observation: np.ndarray, market_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make prediction using ensemble system.

        Args:
            observation: Current market observation
            market_state: Current market state information

        Returns:
            Dictionary containing ensemble prediction results
        """
        if not self.ensemble_enabled or not self.ensemble_system:
            return {"error": "ensemble_not_enabled"}

        try:
            # Get ensemble prediction
            prediction = self.ensemble_system.predict(observation, market_state)

            # Log ensemble decision for analysis
            self.ensemble_logger.debug(
                f"Ensemble prediction: action={prediction.get('final_action')}, "
                f"confidence={prediction.get('avg_confidence', 0):.3f}"
            )

            return prediction

        except Exception as e:
            self.ensemble_logger.error(f"Ensemble prediction failed: {e}")
            return {"error": str(e), "fallback_action": 1}  # Default to HOLD

    def update_ensemble_members(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """
        Update ensemble members with experience data.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is done
        """
        if not self.ensemble_enabled or not self.ensemble_system:
            return

        try:
            self.ensemble_system.update_members(
                observation, action, reward, next_observation, done
            )
        except Exception as e:
            self.ensemble_logger.error(f"Ensemble member update failed: {e}")

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """
        Get current ensemble statistics.

        Returns:
            Dictionary containing ensemble statistics
        """
        if not self.ensemble_enabled or not self.ensemble_system:
            return {"error": "ensemble_not_enabled"}

        try:
            return self.ensemble_system.get_ensemble_stats()
        except Exception as e:
            self.ensemble_logger.error(f"Failed to get ensemble stats: {e}")
            return {"error": str(e)}

    def adapt_ensemble_to_market(self, market_state: Dict[str, Any]) -> None:
        """
        Adapt ensemble to current market conditions.

        Args:
            market_state: Current market state
        """
        if not self.ensemble_enabled or not self.ensemble_system:
            return

        try:
            self.ensemble_system.adapt_ensemble(market_state)
        except Exception as e:
            self.ensemble_logger.error(f"Ensemble adaptation failed: {e}")

    def generate_ensemble_report(
        self, reporter: TrainingReporter, ui: TrainingUI
    ) -> Optional[str]:
        """
        Generate comprehensive ensemble analysis report.

        Args:
            reporter: Training reporter instance
            ui: Training UI instance

        Returns:
            Path to saved report, or None if failed
        """
        if not self.ensemble_enabled or not self.ensemble_system:
            return None

        try:
            ensemble_stats = self.get_ensemble_stats()
            decision_log = self.ensemble_system.decision_log

            ensemble_report = reporter.generate_ensemble_report(
                ensemble_stats, decision_log
            )
            ensemble_report_path = reporter.save_ensemble_report(ensemble_report)

            if ensemble_report_path:
                ui.print_success(
                    f"Ensemble analysis report saved to: {ensemble_report_path}"
                )

                # Display key ensemble metrics
                summary = ensemble_report["ensemble_analysis"]["summary"]
                ui.print_info(
                    f"Ensemble Performance: {summary.get('performance_score', 0):.3f}"
                )
                ui.print_info(
                    f"Ensemble Stability: {summary.get('stability_score', 0):.3f}"
                )

            return ensemble_report_path

        except Exception as e:
            self.ensemble_logger.error(f"Ensemble report generation failed: {e}")
            ui.print_error(f"Ensemble report generation failed: {e}")
            return None

    def print_ensemble_status(self, ui: TrainingUI) -> None:
        """
        Print current ensemble status.

        Args:
            ui: Training UI instance
        """
        if not self.ensemble_enabled or not self.ensemble_system:
            ui.print_info("Ensemble system: Disabled")
            return

        try:
            stats = self.get_ensemble_stats()

            ui.print_header("Ensemble System Status")
            ui.print_info(f"Members: {stats.get('num_members', 0)}")
            ui.print_info(f"Active Members: {stats.get('active_members', 0)}")
            ui.print_info(f"Average Confidence: {stats.get('avg_confidence', 0):.3f}")
            ui.print_info(f"Performance Score: {stats.get('avg_performance', 0):.3f}")
            ui.print_info(f"Stability Score: {stats.get('avg_stability', 0):.3f}")

            # Show member details
            member_stats = stats.get("member_stats", {})
            if member_stats:
                ui.print_subheader("Member Performance")
                for member_id, member_data in member_stats.items():
                    perf = member_data.get("performance_score", 0)
                    stab = member_data.get("stability_score", 0)
                    conf = member_data.get("confidence", 0)
                    ui.print_info(
                        f"  {member_id}: Perf={perf:.3f}, Stab={stab:.3f}, Conf={conf:.3f}"
                    )

        except Exception as e:
            self.ensemble_logger.error(f"Failed to print ensemble status: {e}")
            ui.print_error(f"Ensemble status display failed: {e}")
