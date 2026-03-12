#!/usr/bin/env python3
"""Aggressive 5000-step SAC action-distribution analysis."""

from analyze_sac_5000step_common import (
    ActionAnalysisProfile,
    run_profile,
)


if __name__ == "__main__":
    run_profile(
        ActionAnalysisProfile(
            title="AGGRESSIVE EXPLORATION SAC 5000-STEP ACTION DISTRIBUTION ANALYSIS",
            model_path="models/sac_aggressive_5000step_final.zip",
            output_path="analysis/sac_aggressive_5000step_action_analysis.json",
            threshold=0.02,
            no_action_penalty=-0.001,
            action_bonus=0.0005,
            sell_ratio_warn=0.8,
            buy_ratio_warn=0.05,
            std_warn=0.02,
            reward_warn=-0.002,
            hold_ratio_warn=0.98,
            success_criteria=(
                "BUY actions are generated",
                "SELL actions are generated",
                "action std exceeds 0.05",
                "hold ratio below 90%",
            ),
        )
    )
