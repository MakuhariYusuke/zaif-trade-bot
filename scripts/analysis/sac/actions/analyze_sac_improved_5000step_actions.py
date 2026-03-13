#!/usr/bin/env python3
"""Improved 5000-step SAC action-distribution analysis."""

from analyze_sac_5000step_common import (
    ActionAnalysisProfile,
    run_profile,
)


if __name__ == "__main__":
    run_profile(
        ActionAnalysisProfile(
            title="IMPROVED SAC 5000-STEP ACTION DISTRIBUTION ANALYSIS",
            model_path="models/sac_improved_5000step_final.zip",
            output_path="analysis/sac_improved_5000step_action_analysis.json",
            threshold=0.05,
            no_action_penalty=-0.0001,
            sell_ratio_warn=0.7,
            buy_ratio_warn=0.1,
            std_warn=0.05,
            reward_warn=-0.001,
            hold_ratio_warn=0.95,
        )
    )
