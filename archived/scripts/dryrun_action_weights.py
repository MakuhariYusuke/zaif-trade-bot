#!/usr/bin/env python3
"""
Dry-run Action Weights Calculator.

Computes inverse frequency weights from legal-step diagnostics
WITHOUT running training. Outputs normalized weights for inspection.

Usage:
    python scripts/dryrun_action_weights.py \
        --diagnostics artifacts/diagnostics/last_eval.csv \
        --beta 3.0 --ema 0.1 --out artifacts/weights/dryrun.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import EPSILON

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compute_action_weights(
    action_counts: Dict[str, int],
    beta: float = 3.0,
    epsilon: float = EPSILON,
) -> Dict[str, float]:
    """
    Compute inverse frequency weights with beta clipping.

    Args:
        action_counts: Dictionary with action counts (HOLD, BUY, SELL)
        beta: Maximum weight (clips to prevent extreme ratios)
        epsilon: Small value to avoid division by zero

    Returns:
        Dictionary with normalized weights (sum=3, average=1.0)
    """
    # Extract counts
    total = sum(action_counts.values())

    if total == 0:
        return {"HOLD": 1.0, "BUY": 1.0, "SELL": 1.0}

    # Compute frequencies
    frequencies = {
        action: max(count / total, epsilon) for action, count in action_counts.items()
    }

    # Compute raw weights (inverse frequency)
    weights_raw = {action: 1.0 / freq for action, freq in frequencies.items()}

    # Clip to beta
    weights_clipped = {
        action: min(weight, beta) for action, weight in weights_raw.items()
    }

    # Normalize to sum=3 (average=1.0 for 3 actions)
    weight_sum = sum(weights_clipped.values())
    weights_normalized = {
        action: weight * 3.0 / weight_sum for action, weight in weights_clipped.items()
    }

    return weights_normalized


def extract_legal_action_counts(diagnostics_path: Path) -> Dict[str, int]:
    """
    Extract action counts from diagnostics CSV (legal steps only).

    Args:
        diagnostics_path: Path to diagnostics CSV

    Returns:
        Dictionary with action counts
    """
    # Try to load as CSV
    try:
        df = pd.read_csv(diagnostics_path)
    except Exception as e:
        raise ValueError(f"Failed to load diagnostics CSV: {e}")

    # Expected columns: action, legal_mask (or similar)
    # For now, assume simple format with 'action' column
    if "action" not in df.columns:
        raise ValueError("Diagnostics CSV must have 'action' column")

    # Count actions (0=HOLD, 1=BUY, 2=SELL)
    action_map = {ACTION_HOLD: "HOLD", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}

    counts = {"HOLD": 0, "BUY": 0, "SELL": 0}

    for action in df["action"]:
        action_name = action_map.get(int(action), "UNKNOWN")
        if action_name in counts:
            counts[action_name] += 1

    return counts


def apply_ema_smoothing(
    current_counts: Dict[str, int],
    prev_counts: Dict[str, int],
    alpha: float = 0.1,
) -> Dict[str, int]:
    """
    Apply EMA smoothing to action counts.

    Args:
        current_counts: Current observation counts
        prev_counts: Previous smoothed counts
        alpha: EMA coefficient (0.1 = slow smoothing)

    Returns:
        Smoothed counts
    """
    smoothed = {}

    for action in current_counts:
        prev = prev_counts.get(action, 0)
        curr = current_counts[action]
        smoothed[action] = int(alpha * curr + (1 - alpha) * prev)

    return smoothed


def generate_dryrun_report(
    weights: Dict[str, float],
    counts: Dict[str, int],
    beta: float,
    ema_alpha: float,
) -> Dict[str, Any]:
    """
    Generate dry-run report with weights and diagnostics.

    Args:
        weights: Computed weights
        counts: Action counts
        beta: Beta parameter used
        ema_alpha: EMA alpha used

    Returns:
        Report dictionary
    """
    total = sum(counts.values())
    frequencies = {
        action: count / total if total > 0 else 0.0 for action, count in counts.items()
    }

    report = {
        "parameters": {
            "beta": beta,
            "ema_alpha": ema_alpha,
        },
        "action_counts": counts,
        "action_frequencies": frequencies,
        "computed_weights": weights,
        "weight_statistics": {
            "sum": sum(weights.values()),
            "mean": sum(weights.values()) / len(weights),
            "max": max(weights.values()),
            "min": min(weights.values()),
            "max_ratio": max(weights.values()) / min(weights.values())
            if min(weights.values()) > 0
            else float("inf"),
        },
        "gradient_scaling_estimate": {
            action: f"{weight:.2f}x baseline" for action, weight in weights.items()
        },
        "monitoring_thresholds": {
            "entropy_min": 0.05,
            "target_kl_violations_max": 0.03,
            "kl_consecutive_max": 3,
        },
        "recommendations": [
            "Start with weights=1.0 for first 5k steps (warmup)",
            "Apply cos-warmup from 5k to 15k steps",
            "Monitor moving entropy and target_kl violations",
            "If entropy < 0.05 OR kl_violations > 3%, revert weights to 1.0",
        ],
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Dry-run action weights calculator")
    parser.add_argument(
        "--diagnostics",
        type=Path,
        required=True,
        help="Path to diagnostics CSV with action logs",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=3.0,
        help="Maximum weight (default: 3.0)",
    )
    parser.add_argument(
        "--ema",
        type=float,
        default=0.1,
        help="EMA alpha for smoothing (default: 0.1)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for dry-run report JSON",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Action Weights Dry-Run Calculator")
    print("=" * 60)
    print(f"Diagnostics: {args.diagnostics}")
    print(f"Beta: {args.beta}")
    print(f"EMA alpha: {args.ema}")
    print(f"Output: {args.out}")
    print()

    # Extract action counts from diagnostics
    print("Extracting action counts from diagnostics...")
    counts = extract_legal_action_counts(args.diagnostics)

    print(f"  HOLD: {counts['HOLD']}")
    print(f"  BUY: {counts['BUY']}")
    print(f"  SELL: {counts['SELL']}")
    print(f"  Total: {sum(counts.values())}")
    print()

    # Compute weights
    print("Computing weights...")
    weights = compute_action_weights(counts, beta=args.beta)

    print(f"  HOLD: {weights['HOLD']:.4f}")
    print(f"  BUY: {weights['BUY']:.4f}")
    print(f"  SELL: {weights['SELL']:.4f}")
    print(f"  Sum: {sum(weights.values()):.4f} (target: 3.0)")
    print()

    # Generate report
    print("Generating dry-run report...")
    report = generate_dryrun_report(weights, counts, args.beta, args.ema)

    # Save report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {args.out}")
    print()

    # Print summary
    print("Summary:")
    print(f"  Weight ratio (max/min): {report['weight_statistics']['max_ratio']:.2f}")
    print("  Gradient scaling:")
    for action, scaling in report["gradient_scaling_estimate"].items():
        print(f"    {action}: {scaling}")
    print()

    print("Monitoring thresholds:")
    for key, value in report["monitoring_thresholds"].items():
        print(f"  {key}: {value}")
    print()

    print("✅ Dry-run complete. Review report before training.")


if __name__ == "__main__":
    main()
