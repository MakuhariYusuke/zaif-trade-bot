#!/usr/bin/env python3
"""Baseline 5000-step SAC action-distribution analysis."""

from analyze_sac_5000step_common import (
    ActionAnalysisProfile,
    run_profile,
)


if __name__ == "__main__":
    run_profile(
        ActionAnalysisProfile(
            title="SAC 5000-STEP ACTION DISTRIBUTION ANALYSIS",
            model_path="models/sac_minimal_5000step_final.zip",
            output_path="analysis/sac_5000step_action_analysis.json",
            threshold=0.1,
            sell_ratio_warn=0.6,
            buy_ratio_warn=0.1,
            std_warn=0.1,
            reward_warn=-100.0,
        )
    )
