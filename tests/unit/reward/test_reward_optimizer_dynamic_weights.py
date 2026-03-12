#!/usr/bin/env python3
"""Regression tests for RewardFunctionOptimizer dynamic-weight updates."""

from ztb.training.reward_function_optimizer.reward_function_optimizer import (
    RewardFunctionOptimizer,
)


def test_update_dynamic_weights_from_history_normalizes_string_scores() -> None:
    optimizer = RewardFunctionOptimizer()
    optimizer.optimization_history.clear()

    history_scores = [
        {"profit": "0.10", "max_drawdown": "0.18", "win_rate": "0.70"},
        {"profit": "0.11", "max_drawdown": "0.16", "win_rate": "0.69"},
        {"profit": "0.12", "max_drawdown": "0.17", "win_rate": "0.68"},
        {"profit": "0.20", "max_drawdown": "0.19", "win_rate": "0.71"},
        {"profit": "0.21", "max_drawdown": "0.18", "win_rate": "0.72"},
        {"profit": "0.22", "max_drawdown": "0.17", "win_rate": "0.73"},
    ]
    for idx, score in enumerate(history_scores):
        optimizer.optimization_history.append({"trial_number": idx, "scores": score})

    optimizer._update_dynamic_weights_from_history()

    assert optimizer.dynamic_weights["performance_trend"] == "improving"
    assert optimizer.dynamic_weights["risk_level"] == "high"
    assert optimizer.dynamic_weights["market_regime"] == "bull"


def test_classify_risk_level_threshold_boundaries() -> None:
    optimizer = RewardFunctionOptimizer()

    assert optimizer._classify_risk_level(0.15) == "high"
    assert optimizer._classify_risk_level(0.05) == "moderate"
    assert optimizer._classify_risk_level(0.049) == "low"


def test_print_scores_handles_non_numeric_values(capsys) -> None:
    optimizer = RewardFunctionOptimizer()
    optimizer.set_console_output(verbose=True, show_progress=False, show_detailed_scores=True)

    optimizer._print_scores(
        {
            "profit": "N/A",
            "total_trades": "many",
            "avg_trade_return": None,
            "custom_metric": "custom",
        },
        "Robust Score Print",
    )

    captured = capsys.readouterr().out
    assert "Robust Score Print" in captured
    assert "Profit" in captured
    assert "Total Trades" in captured
